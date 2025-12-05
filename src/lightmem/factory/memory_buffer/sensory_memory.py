import numpy as np
from typing import List, Dict, Optional, Any

class SenMemBufferManager:
    """
    感觉记忆缓冲（短窗口）：
    - 负责以 token 数为上限接收消息，并在溢出或强制触发时进行主题切分；
    - 先使用 segmenter 给出粗粒度边界，再基于 turn 语义相似度进行微调，形成相对一致的话题片段；
    - 片段返回给上层的短期记忆缓冲进行进一步的抽取触发。
    """
    def __init__(self, max_tokens: int = 512, tokenizer = None):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.buffer: List[Dict] = []  # 当前缓冲的消息列表
        self.big_buffer: List[Dict] = [] # 用于累计所有未处理的消息
        self.token_count: int = 0

    def _recount_tokens(self, roles: List[str] = ["user"]) -> None:
        # 仅统计 user 侧文本的 token 数，作为触发与截断的标准
        # 注：此处假定 tokenizer 在实例化时已正确传入且具备 encode 方法；
        # 若 tokenizer 为空，请在更高层保证不为 None。
        self.token_count = sum(len(self.tokenizer.encode(m["content"])) for m in self.buffer if m["role"] in roles)  # type: ignore[union-attr]

    def add_messages(self, messages: List[Dict], segmenter, text_embedder, allowed_roles: list[str]=["user"]) -> None:
        """
        将消息按顺序写入缓冲：
        - 若为 user 侧消息且不超限，则累计；
        - 若预计超限，则调用 cut_with_segmenter 执行切分并输出片段；
        - assistant 侧消息不占用 token 上限，直接并行推进（保持轮次结构）。
        返回收集到的全部片段。
        """
        # TODO : 应该 overlap：即将最后一个主题所属的消息放回缓冲区，避免主题切分过多，并且相同主题一起提取元事实更好。
        all_segments = []
        self.big_buffer.extend(messages)

        while self.big_buffer:
            for msg in self.big_buffer:
                if msg["role"] in allowed_roles:
                    cur_token_count = len(self.tokenizer.encode(msg["content"]))  # type: ignore[union-attr]
                    if self.token_count + cur_token_count <= self.max_tokens:
                        self.buffer.append(msg)
                        self.token_count += cur_token_count
                        self.big_buffer.remove(msg)
                    else:
                        segments = self.cut_with_segmenter(segmenter, text_embedder, allowed_roles)  # type: ignore[attr-defined]
                        all_segments.extend(segments)
                        break
                else:
                    self.buffer.append(msg)
                    self.big_buffer.remove(msg)

        return all_segments  # type: ignore[return-value]

    def should_trigger(self) -> bool:
        # 简单触发条件：当前累计 token 是否达到上限
        return self.token_count >= self.max_tokens

    def cut_with_segmenter(self, segmenter, text_embedder, allowed_roles: list[str]=["user"], force_segment: bool=False) -> List:
        """
        Cut buffer into segments using a two-stage strategy:
        1. Coarse boundaries from segmenter.
        2. Fine-grained adjustment based on semantic similarity.
        """
        # 根据策略决定计入 token 统计的角色
        segments = []
        buffer_texts = [m["content"] for m in self.buffer if m["role"] in allowed_roles]  # 只对用户输入做相似度
        boundaries = segmenter.propose_cut(buffer_texts)

        if not boundaries:
            # 若未给出粗粒度边界，直接整体作为一个片段输出并清空缓冲
            segments.append(self.buffer.copy())
            self.buffer.clear()
            self.token_count = 0
            return segments

        turns = []
        # 将 (user, assistant) 组成 turn，作为语义相似度计算的基本单位
        for i in range(0, len(self.buffer), 2):
            user_msg = self.buffer[i]["content"]
            if i + 1 < len(self.buffer):
                assistant_msg = self.buffer[i + 1]["content"]
                turns.append(user_msg + " " + assistant_msg)
            else:
                turns.append(user_msg)

        embeddings = []
        for turn in turns:
            emb = text_embedder.embed(turn)
            embeddings.append(np.array(emb, dtype=np.float32))
        embeddings = np.vstack(embeddings)

        fine_boundaries = []
        threshold = 0.2
        # 从低到高提升阈值，寻找相邻 turn 相似度的分界位置
        while threshold <= 0.5 and not fine_boundaries:
            for i in range(len(turns) - 1):
                sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                if sim < threshold:
                    fine_boundaries.append(i + 1)
            if not fine_boundaries:
                threshold += 0.05
        
        if not fine_boundaries:
            # 若仍无细粒度边界，整体作为一个片段输出并清空缓冲
            segments.append(self.buffer.copy())
            self.buffer.clear()
            self.token_count = 0
            return segments

        adjusted_boundaries = []
        # 将细分界与粗分界对齐（允许一定偏移），优先靠近粗分界的细分点
        print("话题细分界：", fine_boundaries)
        print("话题粗分界：", boundaries)
        for fb in fine_boundaries:
            for cb in boundaries:
                if abs(int(fb) - int(cb)) <= 3:
                    adjusted_boundaries.append(fb)
                    break
        if not adjusted_boundaries:
            adjusted_boundaries = fine_boundaries

        boundaries = sorted(set(adjusted_boundaries))
        print("调整后的话题边界：", boundaries)
        start_idx = 0
        # 根据边界切分出若干片段，每个边界对应 2*boundary 的消息索引（user+assistant）
        for i, boundary in enumerate(boundaries):
            end_idx = 2 * boundary
            seg = self.buffer[start_idx:end_idx]
            segments.append(seg)
            start_idx = 2 * boundary

        if force_segment:
            # 强制截断：剩余部分也作为一个片段输出
            segments.append(self.buffer[start_idx:])
            start_idx = len(boundaries)

        if start_idx > 0: 
            # 删除已输出的片段，并重算 token 数
            del self.buffer[:start_idx]
            self._recount_tokens(allowed_roles)

        return segments

    def _cosine_similarity(self, vec1, vec2):
        # 计算余弦相似度，返回标量
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
