from typing import List, Dict, Optional, Any, Union
from lightmem.memory.utils import resolve_tokenizer

class ShortMemBufferManager:
    """
    短期记忆缓冲（聚合窗口）：
    - 接收由感觉记忆缓冲切分后的片段（segments）；
    - 基于 messages_use（user_only/assistant_only/hybrid）对每个片段估算 token；
    - 当累计超过阈值或 force_extract=True 时，将当前缓冲的片段作为一次“抽取批次”输出；
    - 上层据此调用大模型进行事实抽取（metadata/text summary）。
    """
    def __init__(self, max_tokens: int = 2000, tokenizer: Optional[Any] = None):
        self.max_tokens = max_tokens
        self.tokenizer = resolve_tokenizer(tokenizer)
        self.buffer: List[List[Dict[str, Any]]] = [] 
        self.token_count: int = 0 
        print(f"ShortMemBufferManager initialized with max_tokens={self.max_tokens}")
    def _count_tokens(self, messages: List[Dict[str, Any]], allowed_roles: list[str]) -> int:
        text_list = [msg["content"] for msg in messages if msg["role"] in allowed_roles]

        text = " ".join(text_list)

        if self.tokenizer is None:
            return len(text)
        elif hasattr(self.tokenizer, "encode"):
            return len(self.tokenizer.encode(text))
        elif isinstance(self.tokenizer, str):
            raise ValueError(
                f"Tokenizer as model_name '{self.tokenizer}' not supported directly. "
                f"Please resolve to actual tokenizer before using."
            )
        else:
            raise TypeError("Invalid tokenizer type")


    def add_segments(self, all_segments: List[List[Dict[str, Any]]], allowed_roles: list[str], force_extract: bool = False):
        """
        聚合片段并基于阈值触发：
        - 若加入某段会使 token 超过上限，则先输出当前缓冲为一个“抽取批次”，再从头计数；
        - 每一次输出均复制当前缓冲，随后清空；
        - 当 force_extract=True 时，强制将当前缓冲输出为一个“抽取批次”。
        返回：触发次数与触发批次列表。
        """
        triggered: List[List[List[Dict[str, Any]]]] = []
        trigger_num = 0

        for seg in all_segments:
            tokens_needed = self._count_tokens(seg, allowed_roles)
            if self.token_count + tokens_needed > self.max_tokens:
                if self.buffer:  
                    triggered.append(self.buffer.copy())
                    trigger_num += 1
                    self.buffer.clear()
                    self.token_count = 0
            self.buffer.append(seg)
            self.token_count += tokens_needed

        if force_extract and self.buffer:
            triggered.append(self.buffer.copy())
            trigger_num += 1
            self.buffer.clear()
            self.token_count = 0
            
        return trigger_num, triggered

