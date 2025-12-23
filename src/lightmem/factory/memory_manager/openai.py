import concurrent
import concurrent.futures  # 为类型检查器显式导入 futures 模块（不改变运行时行为）
from collections import defaultdict
from openai import OpenAI
from typing import List, Dict, Optional, Literal, Any
import json, os, warnings
import httpx
from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig
from lightmem.memory.utils import clean_response
from lightmem.utils.llm_cache import llm_cache

model_name_context_windows = {
    "gpt-4o-mini": 128000,
    "qwen3-30b-a3b-instruct-2507": 128000,
    "qwen2.5:3b": 32768,
    "QwQ-32B": 131072,
    "DEFAULT": 128000,  # Recommended default context window
}


class OpenaiManager:
    """
    基于 OpenAI/OpenRouter 接口的记忆管理器。

    作用：
    - 统一封装聊天补全调用（chat.completions），支持直接 OpenAI 或通过 OpenRouter 网关。
    - 提供通用的响应解析逻辑（包含工具调用场景）。
    - 提供元数据抽取（并行批处理）与更新决策调用的辅助方法。

    配置说明（BaseMemoryManagerConfig）：
    - model：模型名称或配置；若为空，默认 "gpt-4o-mini"。
    - openai_base_url / openrouter_base_url：可选的自定义 API 基址。
    - site_url / app_name / models / route：当使用 OpenRouter 时，可选的 headers 与路由参数。
    - 其他采样参数（temperature、max_tokens、top_p 等）直接传递给 API。
    """
    def __init__(self, config: BaseMemoryManagerConfig):
        self.config = config

        if not self.config.model:
            self.config.model = "gpt-4o-mini"
        
        if self.config.model in model_name_context_windows:
            self.context_windows = model_name_context_windows[self.config.model]  # type: ignore[index]
        else:
            self.context_windows = model_name_context_windows["DEFAULT"]

        http_client = httpx.Client(verify=False)

        if os.environ.get("OPENROUTER_API_KEY"):  # 使用 OpenRouter
            self.client = OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=self.config.openrouter_base_url  # type: ignore[attr-defined]
                or os.getenv("OPENROUTER_API_BASE")
                or "https://openrouter.ai/api/v1",
            )
        else:
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            base_url = (
                self.config.openai_base_url
                or os.getenv("OPENAI_API_BASE")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            )

            self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    @llm_cache(key_prefix="openai_chat")
    def _call_api(self, **params):
        """
        带缓存的底层 API 调用方法。
        
        参数相同时会命中缓存，无需重复请求。
        报错的请求不会被缓存。
        """
        return self.client.chat.completions.create(**params)

    def _parse_response(self, response, tools):
        """
        根据是否使用了工具（tools）来处理模型返回：

        参数：
            response：API 原始返回对象。
            tools：请求中提供的工具列表。

        返回：
            str 或 dict：处理后的结果（若使用工具则返回包含工具调用信息的字典，否则返回纯文本）。
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> Optional[str]:
        """
    使用 OpenAI 接口基于给定消息生成回复。

        参数：
            messages (list)：消息列表（包含 'role' 与 'content'）。
            response_format (str 或对象，可选)：响应格式，默认 "text"。
            tools (list，可选)：可供模型调用的工具列表，默认 None。
            tool_choice (str，可选)：工具选择方式，默认 "auto"。

        返回：
            str：生成的回复内容（或在上层进一步解析）。
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        if os.getenv("OPENROUTER_API_KEY"):
            openrouter_params = {}
            if self.config.models:  # type: ignore[attr-defined]
                openrouter_params["models"] = self.config.models  # type: ignore[attr-defined]
                openrouter_params["route"] = self.config.route  # type: ignore[attr-defined]
                params.pop("model")

            if self.config.site_url and self.config.app_name:  # type: ignore[attr-defined]
                extra_headers = {
                    "HTTP-Referer": self.config.site_url,  # type: ignore[attr-defined]
                    "X-Title": self.config.app_name,  # type: ignore[attr-defined]
                }
                openrouter_params["extra_headers"] = extra_headers

            params.update(**openrouter_params)

        if response_format:
            params["response_format"] = response_format
        if tools:  # TODO：如果新增的记忆添加逻辑稳定，可移除 tools 相关参数
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self._call_api(**params)
        usage_info = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        parsed_response = self._parse_response(response, tools)

        return parsed_response, usage_info

    def meta_text_extract(
        self,
        system_prompt: str,
        extract_list: List[List[List[Dict]]],
        allowed_roles: list[str] = ["user"],
        topic_id_mapping: Optional[List[List[int]]] = None
    ) -> List[Optional[Dict]]:
        """
    使用并行处理从文本片段中抽取元数据（事实）。

        参数：
            system_prompt：用于元数据生成的系统提示词。
            extract_list：待处理的消息片段列表（分组后的多段）。
            messages_use：参与拼接的消息角色策略（user_only/assistant_only/hybrid）。
            topic_id_mapping: For each API call, the global topic IDs

        返回：
            List[Optional[Dict]]：每个 API 调用的抽取结果字典（失败为 None）。
        """
        if not extract_list:
            return []
        
        def concatenate_messages(segment: List[Dict], allowed_roles: List[str]) -> str:
            """Concatenate messages based on usage strategy"""
            message_lines = []

            for mes in segment:
                if mes.get("role") in allowed_roles:
                    sequence_id = mes["sequence_number"]
                    role = mes["role"]
                    content = mes.get("content", "")
                    speaker_name = mes.get("speaker_name", "")
                    time_stamp = mes.get("time_stamp", "")
                    weekday = mes.get("weekday", "")
                    
                    time_prefix = ""
                    if time_stamp and weekday:
                        time_prefix = f"[{time_stamp}, {weekday}] "

                    if speaker_name:
                        message_lines.append(f"{time_prefix}{sequence_id//2}.{speaker_name}: {content}")
                    else:
                        message_lines.append(f"{time_prefix}{sequence_id//2}.{role}: {content}")
            
            return "\n".join(message_lines)

        max_workers = min(len(extract_list), 5)

        def process_segment_wrapper(args):
            api_call_idx, api_call_segments = args
            try:
                user_prompt_parts: List[str] = []
                
                global_topic_ids: List[int] = []
                if topic_id_mapping and api_call_idx < len(topic_id_mapping):
                    global_topic_ids = topic_id_mapping[api_call_idx]

                for topic_idx, topic_segment in enumerate(api_call_segments):
                    if topic_idx < len(global_topic_ids):
                        global_topic_id = global_topic_ids[topic_idx]
                    else:
                        global_topic_id = topic_idx + 1
                    
                    topic_text = concatenate_messages(topic_segment, allowed_roles)
                    user_prompt_parts.append(f"--- Topic {global_topic_id} ---\n{topic_text}")

                print(f"User prompt for API call {api_call_idx}:\n" + "\n".join(user_prompt_parts))
                user_prompt = "\n".join(user_prompt_parts)
                
                metadata_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                
                raw_response, usage_info = self.generate_response(
                    messages=metadata_messages,
                    response_format={"type": "json_object"},
                )
                cleaned_result = clean_response(raw_response)  # type: ignore[arg-type]
                return {
                    "input_prompt": metadata_messages,
                    "output_prompt": raw_response,
                    "cleaned_result": cleaned_result,
                    "usage": usage_info,
                }
                
            except Exception as e:
                print(f"Error processing API call {api_call_idx}: {e}")
                return {
                    "input_prompt": [],
                    "output_prompt": "",
                    "cleaned_result": [],
                    "usage": None,
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                results = list(executor.map(process_segment_wrapper, enumerate(extract_list)))
            except Exception as e:
                print(f"Error in parallel processing: {e}")
                results = [None] * len(extract_list)

        return results  # type: ignore[return-value]
    
    def _call_update_llm(self, system_prompt, target_entry, candidate_sources):
        target_memory = target_entry["payload"]["memory"]
        candidate_memories = [c["payload"]["memory"] for c in candidate_sources]

        user_prompt = (
            f"Target memory:{target_memory}\n"
            f"Candidate memories:\n" + "\n".join([f"- {m}" for m in candidate_memories])
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response_text, usage_info = self.generate_response(
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response_text)  # type: ignore[arg-type]
            if "action" not in result:
                result = {"action": "ignore"}
            result["usage"] = usage_info  
            return result
        except Exception:
            return {"action": "ignore", "usage": usage_info if 'usage_info' in locals() else None}
