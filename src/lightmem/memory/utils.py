import os
import re
import json
import uuid
import tiktoken
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union


@dataclass
class MemoryEntry:
    # 记忆条目的标准结构：
    # - id：唯一标识，默认 UUID
    # - time_stamp：字符串格式时间（建议 ISO 格式），用于人类可读展示
    # - float_time_stamp：浮点时间戳（秒），便于数值排序与过滤
    # - weekday：星期信息（如 Mon/Tue），与人类时间理解相关
    # - category/subcategory/memory_class：可选分类标签，便于细粒度检索
    # - memory：事实文本（抽取得到的最终记忆）
    # - original_memory/compressed_memory：可选原文/压缩版本，便于回溯与存证
    # - hit_time：命中次数（如检索曝光计数），可用于“温度”或“重要性”更新
    # - update_queue：离线更新时的候选队列（包含其他条目的 id 与得分）
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    time_stamp: str = field(default_factory=lambda: datetime.now().isoformat())
    float_time_stamp: float = 0
    weekday: str = ""
    category: str = ""
    subcategory: str = ""
    memory_class: str = ""
    memory: str = ""
    original_memory: str = ""
    compressed_memory: str = ""
    hit_time: int = 0
    update_queue: List = field(default_factory=list)

def clean_response(response: str) -> List[Dict[str, Any]]:
    """
    清洗大模型响应：
    1. 去除外层代码块标记（```[language] ... ```）。
    2. 安全解析 JSON 内容。
    3. 若存在 "data" 键并为 list，则返回该列表；否则尝试返回解析结果（list/dict）。
    """
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, response.strip())
    cleaned = match.group(1).strip() if match else response.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict) and "data" in parsed and isinstance(parsed["data"], list):
        return parsed["data"]

    if isinstance(parsed, list):
        return parsed

    return []

def assign_sequence_numbers_with_timestamps(extract_list):
    """
    为抽取阶段整理的分段消息打上全局的 sequence_number，并收集其时间戳与星期。
    输入格式约定：extract_list 是一个多层列表，形如 [segments] -> [segment] -> [message dict]
    每个 message dict 预期包含 "time_stamp" 与 "weekday" 字段（由 MessageNormalizer 保证）。
    返回：更新后的 extract_list、时间戳列表、星期列表（按 sequence_number 对齐）。
    """
    current_index = 0
    timestamps_list = []
    weekday_list = []
    
    for segments in extract_list:
        for seg in segments:
            for message in seg:
                message["sequence_number"] = current_index
                timestamps_list.append(message["time_stamp"])
                weekday_list.append(message["weekday"])
                current_index += 1
    
    return extract_list, timestamps_list, weekday_list

# TODO：merge into context retriever
def save_memory_entries(memory_entries, file_path="memory_entries.json"):
    """
    将内存条目追加保存到 JSON 文件中，便于基于上下文（非向量）检索。
    若文件存在则合并写入，否则创建新文件；该函数不去重，调用方可在更高层处理。
    """
    def entry_to_dict(entry):
        return {
            "id": entry.id,
            "time_stamp": entry.time_stamp,
            "category": entry.category,
            "subcategory": entry.subcategory,
            "memory_class": entry.memory_class,
            "memory": entry.memory,
            "original_memory": entry.original_memory,
            "compressed_memory": entry.compressed_memory,
            "hit_time": entry.hit_time,
            "update_queue": entry.update_queue,
        }

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    new_data = [entry_to_dict(e) for e in memory_entries]
    existing_data.extend(new_data)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)


def resolve_tokenizer(tokenizer_or_name: Union[str, Any]):
    """
    解析 tokenizer：
    - 允许传入模型名字符串，根据内置映射解析到 tiktoken 的编码名；
    - 暂不支持直接传自定义 tokenizer 对象（保持与当前调用方一致）。
    - 未知模型名会抛出异常，提示更新映射表。
    """
    if tokenizer_or_name is None:
        raise ValueError("Tokenizer or model_name must be provided.")

    if isinstance(tokenizer_or_name, str):
        model_tokenizer_map = {
            "gpt-4o-mini": "o200k_base",
            "gpt-4o": "o200k_base",
            "gpt-4.1-mini": "o200k_base",
            "gpt-4.1": "o200k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "qwen3-30b-a3b-instruct-2507": "o200k_base",
            "qwen2.5:3b": "o200k_base",
            "QwQ-32B": "o200k_base"
        }

        if tokenizer_or_name not in model_tokenizer_map:
            raise ValueError(f"Unknown model_name '{tokenizer_or_name}', please update mapping.")

        encoding_name = model_tokenizer_map[tokenizer_or_name]
        # 调试信息：标注解析到的编码器名称（可按需保留/关闭上层日志）
        print("DEBUG: resolved to encoding", encoding_name)
        return tiktoken.get_encoding(encoding_name)

    raise TypeError(f"Unsupported tokenizer type: {type(tokenizer_or_name)}")
