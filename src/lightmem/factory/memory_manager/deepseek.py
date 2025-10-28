from openai import OpenAI
from typing import List, Dict, Optional
import json
from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig

class DeepseekManager:
    def __init__(self, config: BaseMemoryManagerConfig):
        """
        初始化 DeepSeek 风格的记忆管理器：
        - 使用 OpenAI 兼容接口；
        - 从配置中读取模型/密钥/接口地址；
        - 仅负责请求与响应解析，不修改业务字符串。
        注意：此文件当前实现可能依赖外部配置结构，请确保调用方传入的配置项完整。
        """
        if not self.config.model:
            self.config.model = "deepseek-chat"
        self.api_key = self.config.api_key
        self.base_url = config.get("deepseek_base_url", "https://api.deepseek.com/v1")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @classmethod
    def from_config(cls, model_name, config):
        cls.validate_config(config)
        return cls(config)

    @staticmethod
    def validate_config(config):
        required_keys = ['api_key', 'endpoint']
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValueError(f"Missing required config keys for DeepSeek LLM: {missing}")

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
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """
        使用 DeepSeek 接口基于给定消息生成回复。

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
            "top_k": self.config.top_k
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        return self._parse_response(response, tools)