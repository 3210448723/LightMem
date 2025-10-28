import uuid
import re
import copy
import concurrent
import logging
import json
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Tuple
from pydantic import ValidationError
from lightmem.configs.base import BaseMemoryConfigs
from lightmem.factory.pre_compressor.factory import PreCompressorFactory
from lightmem.factory.topic_segmenter.factory import TopicSegmenterFactory
from lightmem.factory.memory_manager.factory import MemoryManagerFactory
from lightmem.factory.text_embedder.factory import TextEmbedderFactory
from lightmem.factory.retriever.contextretriever.factory import ContextRetrieverFactory
from lightmem.factory.retriever.embeddingretriever.factory import EmbeddingRetrieverFactory
from lightmem.factory.memory_buffer.sensory_memory import SenMemBufferManager
from lightmem.factory.memory_buffer.short_term_memory import ShortMemBufferManager
from lightmem.memory.utils import MemoryEntry, assign_sequence_numbers_with_timestamps, save_memory_entries
from lightmem.memory.prompts import METADATA_GENERATE_PROMPT, UPDATE_PROMPT
from lightmem.configs.logging.utils import get_logger
from concurrent.futures import ThreadPoolExecutor  # 使用显式导入，避免静态检查器无法解析 concurrent.futures

# 说明：
# 本模块实现 LightMemory 对话记忆系统的核心流程，包括：
# - 消息规范化（时间戳标准化、会话粒度时间序列的生成）
# - 可选的预压缩（降低冗余 token，减少后续处理成本）
# - 主题分段（将连续对话切分为语义一致的片段）与感觉记忆缓存管理
# - 短期记忆触发抽取（基于策略/阈值将片段汇总为候选事实）
# - 元数据/事实抽取（调用大模型总结事实）
# - 记忆条目生成与入库（向量检索库 / 文件存储）
# - 离线更新（构建更新队列并进行增量更新或冲突删除）
# - 检索（基于查询返回格式化的记忆文本）
#
# 注意：严格不修改任何已有字符串字面量内容；仅添加中文注释，保证行为不变。


class MessageNormalizer:
    """
    消息标准化工具：
    - 输入可以是 dict 或 list[dict]，每条消息要求包含会话级的 "time_stamp"（原始字符串时间，如 "2023/05/20 (Sat) 00:44"）
    - 内部会将同一会话下的消息按固定偏移（offset_ms）依次递增，生成严格递增的 ISO 格式时间戳
    - 同时保留原始会话时间到 "session_time" 字段，记录星期到 "weekday"
    - 该处理有利于后续基于时间顺序的抽取与检索
    """

    # 支持的会话级时间戳格式正则：例如 "2023/05/20 (Sat) 00:44" 或以 "-" 分隔；秒数可选
    _SESSION_RE = re.compile(
        r'(?P<date>\d{4}[/-]\d{1,2}[/-]\d{1,2})\s*\((?P<weekday>[^)]+)\)\s*(?P<time>\d{1,2}:\d{2}(?::\d{2})?)'
    )

    def __init__(self, offset_ms: int = 1000):
        # 记录同一个原始 session 时间字符串下，最后一次赋值的具体时间戳，便于为下一条消息递增
        self.last_timestamp_map: Dict[str, datetime] = {}
        # 每条消息之间的时间偏移，默认 1000ms；可避免同一会话消息出现相同时间
        self.offset = timedelta(milliseconds=offset_ms)

    def _parse_session_timestamp(self, raw_ts: str) -> Tuple[datetime, str]:
        """
        Parse a session-level timestamp and return (base_datetime, weekday).
        Supports formats like "2023/05/20 (Sat) 00:44" (also accepts '-' as separator, and optional seconds).
        Raises ValueError if parsing fails.
        """
        # 优先尝试匹配预期格式（带星期的会话标记），否则回退尝试 ISO 格式
        m = self._SESSION_RE.search(raw_ts)
        if m:
            date_str = m.group('date').replace('-', '/')
            time_str = m.group('time')
            weekday = m.group('weekday')
            fmt = "%Y/%m/%d %H:%M:%S" if time_str.count(':') == 2 else "%Y/%m/%d %H:%M"
            base_dt = datetime.strptime(f"{date_str} {time_str}", fmt)
            return base_dt, weekday

        try:
            dt = datetime.fromisoformat(raw_ts)
            return dt, dt.strftime("%a")
        except Exception:
            raise ValueError(f"Failed to parse session time format: '{raw_ts}'. Expected something like '2023/05/20 (Sat) 00:44'")

    def normalize_messages(self, messages: Any) -> List[Dict[str, Any]]:
        """
        Accepts str / dict / list[dict]:
          - If str -> treated as a single user message (if 'time_stamp' is required, use dict form)
          - If dict -> single message
          - If list -> multiple messages (each must be a dict and contain 'time_stamp')
        Returns: List[Dict] (each item is a copied and enriched message)
        """
        # 将输入统一为列表形式，并严格校验每项必须包含会话级时间戳字段
        # Normalize input into a list
        if isinstance(messages, dict):
            messages_list = [messages]
        elif isinstance(messages, list):
            messages_list = messages
        elif isinstance(messages, str):
            raise ValueError("Please provide messages as dict or list[dict], and ensure each dict contains a 'time_stamp' field (session-level).")
        else:
            raise ValueError("messages must be dict or list[dict] (or str, but not recommended).")

        enriched_list: List[Dict[str, Any]] = []

        for msg in messages_list:
            if not isinstance(msg, dict):
                raise ValueError("Each item in messages list must be a dict.")
            raw_ts = msg.get("time_stamp")
            if not raw_ts:
                raise ValueError("Each message should contain a 'time_stamp' field (e.g., '2023/05/20 (Sat) 00:44').")

            # 解析会话基准时间与星期标记
            base_dt, weekday = self._parse_session_timestamp(raw_ts)

            # Maintain incrementing time based on raw_ts as session key
            # 使用原始会话时间字符串作为 key，在同一会话下让后续消息以固定 offset 递增
            last_dt = self.last_timestamp_map.get(raw_ts)
            if last_dt is None:
                new_dt = base_dt
            else:
                new_dt = last_dt + self.offset

            self.last_timestamp_map[raw_ts] = new_dt

            enriched = copy.deepcopy(msg)
            # 保留原始会话时间与星期信息，并将 time_stamp 规范为 ISO 字符串（毫秒精度）
            enriched["session_time"] = raw_ts  # '2023/05/20 (Sat) 02:21'
            enriched["time_stamp"] = new_dt.isoformat(timespec="milliseconds")  # 2023-05-20T02:21:00.000
            enriched["weekday"] = weekday  # Sat

            enriched_list.append(enriched)

        return enriched_list


class LightMemory:
    def __init__(self, config: BaseMemoryConfigs = BaseMemoryConfigs()):
        # LightMemory 主类：负责从原始对话消息构建结构化记忆、入库与检索。
        # 初始化阶段会按配置构建各个子模块（预压缩、主题分段、记忆管理、嵌入、检索器等）。
        """
        Initialize a LightMemory instance.

        This constructor initializes various memory-related components based on the provided configuration (`config`), 
        including the memory manager, optional pre-compressor, optional topic segmenter, text embedder, 
        and retrievers based on the configured strategies.

        This design supports flexible extension of the memory system, making it easy to integrate 
        different processing and retrieval capabilities.

        Args:
            config (BaseMemoryConfigs): The configuration object for the memory system, 
                containing initialization parameters for all submodules.

        Components initialized:
            - compressor (optional): Pre-compression model if pre_compress=True
            - segmenter (optional): Topic segmentation model if topic_segment=True
            - manager: Memory management model for metadata generation and text summarization
            - text_embedder (optional): Text embedding model if index_strategy is 'embedding' or 'hybrid'
            - context_retriever (optional): Context-based retriever if retrieve_strategy is 'context' or 'hybrid'
            - embedding_retriever (optional): Embedding-based retriever if retrieve_strategy is 'embedding' or 'hybrid'
            - graph (optional): Graph memory store if graph_mem is enabled

        Note:
            - Multimodal embedder initialization is currently commented out
            - Graph memory initialization is conditional on graph_mem configuration
        """
        if config.logging is not None:
            config.logging.apply()
        
        self.logger = get_logger("LightMemory")
        self.logger.info("Initializing LightMemory with provided configuration")
        
        self.config = config
        if self.config.pre_compress:
            self.logger.info("Initializing pre-compressor")
            # 预压缩器：减少输入 token，提升后续分段与抽取效率
            assert self.config.pre_compressor is not None, "pre_compressor config should not be None when pre_compress=True"
            self.compressor = PreCompressorFactory.from_config(self.config.pre_compressor)  # type: ignore[arg-type]
        if self.config.topic_segment:
            self.logger.info("Initializing topic segmenter")
            # 主题分段器：对连续消息进行语义切分；感觉记忆缓冲用于积攒与触发切分
            assert self.config.topic_segmenter is not None, "topic_segmenter config should not be None when topic_segment=True"
            assert self.config.precomp_topic_shared is not None, "precomp_topic_shared should not be None when topic_segment=True"
            self.segmenter = TopicSegmenterFactory.from_config(self.config.topic_segmenter, self.config.precomp_topic_shared, self.compressor)  # type: ignore[arg-type]
            self.senmem_buffer_manager = SenMemBufferManager(max_tokens=self.segmenter.buffer_len, tokenizer=self.segmenter.tokenizer)
        self.logger.info("Initializing memory manager")
        # 记忆管理器：调用大模型进行元数据生成与更新决策
        self.manager = MemoryManagerFactory.from_config(self.config.memory_manager)
        # 短期记忆缓冲：聚合分段结果并根据策略触发抽取
        self.shortmem_buffer_manager = ShortMemBufferManager(max_tokens = 1024, tokenizer=getattr(self.manager, "tokenizer", self.manager.config.model))
        if self.config.index_strategy == 'embedding' or self.config.index_strategy == 'hybrid':
            self.logger.info("Initializing text embedder")
            # 文本嵌入器：为记忆或查询生成向量表示
            assert self.config.text_embedder is not None, "text_embedder config should not be None when index_strategy includes 'embedding'"
            self.text_embedder = TextEmbedderFactory.from_config(self.config.text_embedder)  # type: ignore[arg-type]
        # if self.config.multimodal_embedder:
        if self.config.retrieve_strategy in ["context", "hybrid"]:
            self.logger.info("Initializing context retriever")
            # 基于上下文的检索器（例如从文件中召回）
            assert self.config.context_retriever is not None, "context_retriever config should not be None when retrieve_strategy includes 'context'"
            self.context_retriever = ContextRetrieverFactory.from_config(self.config.context_retriever)  # type: ignore[arg-type]
        if self.config.retrieve_strategy in ["embedding", "hybrid"]:
            self.logger.info("Initializing embedding retriever")
            # 向量检索器（如 Qdrant）：负责插入、搜索、更新、删除
            assert self.config.embedding_retriever is not None, "embedding_retriever config should not be None when retrieve_strategy includes 'embedding'"
            self.embedding_retriever = EmbeddingRetrieverFactory.from_config(self.config.embedding_retriever)  # type: ignore[arg-type]
        if self.config.graph_mem:
            from .graph import GraphMem
            self.logger.info("Initializing graph memory")
            # 图记忆（可选）：用于结构化存储实体与关系
            # GraphMem 支持可选配置入参
            self.graph = GraphMem(self.config.graph_mem)
        self.logger.info("LightMemory initialization completed successfully")

    @classmethod
    def from_config(cls, config: Dict[str,Any]):
        try:
            configs = BaseMemoryConfigs(**config)
        except ValidationError as e:
            print(f"Configuration validation error: {e}")
            raise
        return cls(configs)
    
    
    def add_memory(
        self,
        messages,
        *,
        force_segment: bool = False, 
        force_extract: bool = False
    ):
        """
        Add new memory entries from message history.

        This method serves as the main pipeline for constructing new memory units from 
        incoming messages. It performs message normalization, optional pre-compression,
        segmentation, and knowledge extraction to produce structured memory entries.

        The process is as follows:
          1. Normalize input messages with standardized timestamps and session tracking.
          2. Optionally compress messages using the pre-defined compression model (if enabled).
          3. If topic segmentation is enabled, split messages into coherent segments and add them to the sentence-level buffer.
          4. Trigger memory extraction based on configured thresholds or forced flags.
          5. Optionally perform metadata summarization using an external model if enabled.
          6. Convert extracted results into `MemoryEntry` objects and update memory storage
             (either in online or offline mode depending on configuration).

        Args:
            messages (dict or List[dict]): Input message(s) to process.
            force_segment (bool, optional): If True, forces segmentation regardless of buffer conditions.
            force_extract (bool, optional): If True, forces memory extraction even if thresholds are not met.

        Returns:
            dict: A dictionary containing the intermediate results of the memory addition pipeline.
                  Typically includes:
                    - `"add_input_prompt"`: List of input prompts used for metadata generation (if enabled)
                    - `"add_output_prompt"`: Corresponding output results from metadata generation
                    - `"api_call_nums"`: Number of API calls made for extraction/summarization
                    - (In early termination cases) A segmentation result dict with keys such as
                      `"triggered"`, `"cut_index"`, `"boundaries"`, and `"emitted_messages"`

        Notes:
            - If `self.config.pre_compress` is True, messages will first be token-compressed before segmentation.
            - If `self.config.topic_segment` is disabled, the function returns early with segmentation info only.
            - Memory extraction results are wrapped into `MemoryEntry` objects containing timestamps,
              weekdays, and extracted factual content.
            - Depending on `self.config.update`, the function triggers either online or offline memory updates.
        """
        # 注意：该函数是构建记忆流水线的入口。除返回结果字典外，真正的记忆条目会在离线/在线更新阶段入库。
        call_id = f"add_memory_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.logger.info(f"========== START {call_id} ==========")
        self.logger.info(f"force_segment={force_segment}, force_extract={force_extract}")
        result = {
            "add_input_prompt": [],
            "add_output_prompt": [],
            "api_call_nums": 0
        }
        self.logger.debug(f"[{call_id}] Raw input type: {type(messages)}")
        if isinstance(messages, list):
            self.logger.debug(f"[{call_id}] Raw input sample: {json.dumps(messages)}")
        # 1) 规范化消息，确保时间戳有序、星期与 session 信息齐全
        normalizer = MessageNormalizer(offset_ms=500)
        msgs = normalizer.normalize_messages(messages)
        self.logger.debug(f"[{call_id}] Normalized messages sample: {json.dumps(msgs)}")
        # 2) 可选预压缩：优先使用可用的 tokenizer
        if self.config.pre_compress:
            if hasattr(self.compressor, "tokenizer") and self.compressor.tokenizer is not None:
                args = (msgs, self.compressor.tokenizer)
            elif self.config.topic_segment and hasattr(self.segmenter, "tokenizer") and self.segmenter.tokenizer is not None:
                args = (msgs, self.segmenter.tokenizer)
            else:
                args = (msgs,)
            compressed_messages = self.compressor.compress(*args)
            cfg = getattr(self.compressor, "config", None)
            target_rate = None
            if cfg is not None:
                if hasattr(cfg, 'entropy_config') and isinstance(cfg.entropy_config, dict):
                    target_rate = cfg.entropy_config.get('compress_rate')
                elif hasattr(cfg, 'compress_config') and isinstance(cfg.compress_config, dict):
                    target_rate = cfg.compress_config.get('rate')
            self.logger.info(f"[{call_id}] Target compression rate: {target_rate}")
            self.logger.debug(f"[{call_id}] Compressed messages sample: {json.dumps(compressed_messages)}")
        else:
            compressed_messages = msgs
            self.logger.info(f"[{call_id}] Pre-compression disabled, using normalized messages")
        
        # 3) 若关闭分段，则直接返回规范化后的片段信息（早退）
        if not self.config.topic_segment:
            # TODO:
            self.logger.info(f"[{call_id}] Topic segmentation disabled, returning emitted messages")
            return {
                "triggered": True,
                "cut_index": len(msgs),
                "boundaries": [0, len(msgs)],
                "emitted_messages": msgs,
                "carryover_size": 0,
            }

        # 4) 感觉记忆缓冲：接收消息+分段器/嵌入器，返回片段列表
        all_segments = self.senmem_buffer_manager.add_messages(compressed_messages, self.segmenter, self.text_embedder)

        if force_segment:
            # 强制切分：通常用于会话末尾强制落盘
            all_segments = self.senmem_buffer_manager.cut_with_segmenter(self.segmenter, self.text_embedder, force_segment)
        
        if not all_segments:
            self.logger.debug(f"[{call_id}] No segments generated, returning empty result")
            return result # TODO

        self.logger.info(f"[{call_id}] Generated {len(all_segments)} segments")
        self.logger.debug(f"[{call_id}] Segments sample: {json.dumps(all_segments)}")

        # 5) 短期记忆缓冲：根据策略/阈值触发抽取，将片段汇总成序列化的消息集合
        extract_trigger_num, extract_list = self.shortmem_buffer_manager.add_segments(
            all_segments,
            self.config.messages_use or 'user_only',  # 确保传入非 None，默认优先抽取用户侧
            force_extract,
        )

        if extract_trigger_num == 0:
            self.logger.debug(f"[{call_id}] Extraction not triggered, returning result")
            return result # TODO 
        
        self.logger.info(f"[{call_id}] Extraction triggered {extract_trigger_num} times, extract_list length: {len(extract_list)}")
        # 为抽取消息标注全局序号（sequence_number），并收集其时间戳与星期
        extract_list, timestamps_list, weekday_list = assign_sequence_numbers_with_timestamps(extract_list)
        self.logger.info(f"[{call_id}] Assigned timestamps to {len(extract_list)} items")
        self.logger.debug(f"[{call_id}] Timestamps sample: {timestamps_list}")
        self.logger.debug(f"[{call_id}] Weekdays sample: {weekday_list}")
        self.logger.debug(f"[{call_id}] Extract list sample: {json.dumps(extract_list)}")

        # 6) 元数据/事实抽取：调用大模型对抽取片段进行事实级汇总
        if self.config.metadata_generate and self.config.text_summary:
            self.logger.info(f"[{call_id}] Starting metadata generation")
            extracted_results = self.manager.meta_text_extract(METADATA_GENERATE_PROMPT, extract_list, self.config.messages_use)
            for item in extracted_results:
                if item is not None:
                    result["add_input_prompt"].append(item["input_prompt"])
                    result["add_output_prompt"].append(item["output_prompt"])
                    result["api_call_nums"] += 1
            self.logger.info(f"[{call_id}] Metadata generation completed with {result['api_call_nums']} API calls")
            extracted_memory_entry = [item["cleaned_result"] for item in extracted_results if item]
            self.logger.info(f"[{call_id}] Extracted {len(extracted_memory_entry)} memory entries")
            self.logger.debug(f"[{call_id}] Extracted memory entry sample: {json.dumps(extracted_memory_entry)}")
        # 7) 组装 MemoryEntry：将事实绑定时间戳/星期，以便后续检索/更新
        memory_entries = []
        for topic_memory in extracted_memory_entry:
            if not topic_memory:
                continue
            for entry in topic_memory:
                # 设置安全默认值，防止后续未赋值引用
                now_dt = datetime.now()
                time_stamp: str = now_dt.isoformat(timespec="seconds")
                float_time_stamp: float = now_dt.timestamp()
                weekday: str = now_dt.strftime("%a")

                sequence_n = entry.get("source_id")
                try:
                    # 校验 source_id 的合法性
                    # 如果 sequence_n 是字符串且可以转换为整数，则继续处理
                    if isinstance(sequence_n, str) and sequence_n.isdigit():
                        sequence_n = int(sequence_n)

                    if sequence_n is None or not isinstance(sequence_n, int):
                        raise TypeError(f"Invalid source_id: {sequence_n}")
                    
                    # 如果 source_id 超出范围，抛出异常

                    # 从已对齐的列表中读取时间与星期
                    ts_value = timestamps_list[sequence_n]
                    wd_value = weekday_list[sequence_n]

                    if isinstance(ts_value, (int, float)):
                        # 若为浮点时间戳，转换回人类可读并记录双格式
                        float_time_stamp = float(ts_value)
                        time_stamp = datetime.fromtimestamp(float_time_stamp).isoformat(timespec="seconds")
                    elif isinstance(ts_value, str):
                        # 预期为 ISO 字符串
                        time_stamp = ts_value
                        try:
                            float_time_stamp = datetime.fromisoformat(time_stamp).timestamp()
                        except Exception:
                            # 兼容非严格 ISO 字符串
                            float_time_stamp = now_dt.timestamp()
                    else:
                        # 未知类型，回退到当前时间
                        time_stamp = now_dt.isoformat(timespec="seconds")
                        float_time_stamp = now_dt.timestamp()

                    weekday = str(wd_value) if wd_value is not None else weekday
                except (IndexError, TypeError) as e:
                    # 回退到默认时间，并记录告警但不中断流程
                    self.logger.warning(f"[{call_id}] Error getting timestamp for sequence {sequence_n}: {e}; fallback to now()")

                mem_obj = MemoryEntry(
                    time_stamp=time_stamp,
                    float_time_stamp=float_time_stamp,
                    weekday=weekday,
                    memory=entry.get("fact", ""),
                    # original_memory=entry.get("original_fact", ""),  # TODO
                    # compressed_memory=""  # TODO
                )
                memory_entries.append(mem_obj)

        self.logger.info(f"[{call_id}] Created {len(memory_entries)} MemoryEntry objects")
        for i, mem in enumerate(memory_entries):
            self.logger.debug(f"[{call_id}] MemoryEntry[{i}]: time={mem.time_stamp}, weekday={mem.weekday}, memory={mem.memory}")

        # 8) 入库：支持 online/offline 两种写入策略（默认 offline）
        if self.config.update == "online":
            self.online_update(memory_entries)
        elif self.config.update == "offline":
            self.offline_update(memory_entries)
        
        return result

    def online_update(self, memory_list: List):
        return None

    def offline_update(self, memory_list: List, construct_update_queue_trigger: bool = False, offline_update_trigger: bool = False):
        # 离线更新：
        # - embedding/hybrid 策略下，将记忆转向量并插入向量数据库；
        # - context/hybrid 策略下，将记忆落到文件以便上下文检索；
        # - 可选：构建更新队列、执行基于相似度与时间的批量更新/删除。
        call_id = f"offline_update_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.logger.info(f"========== START {call_id} ==========")
        self.logger.info(f"[{call_id}] Received {len(memory_list)} memory entries")
        self.logger.info(f"[{call_id}] construct_update_queue_trigger={construct_update_queue_trigger}, offline_update_trigger={offline_update_trigger}")

        if self.config.index_strategy in ["context", "hybrid"]:
            self.logger.info(f"[{call_id}] Saving memory entries to file (strategy: {self.config.index_strategy})")
            save_memory_entries(memory_list, "memory_entries.json")

        if self.config.index_strategy in ["embedding", "hybrid"]:
            inserted_count = 0
            self.logger.info(f"[{call_id}] Starting embedding and insertion to vector database")
            for mem_obj in memory_list:
                # 生成向量，分配唯一 id（若冲突则重试），组装 payload 并插入
                embedding_vector = self.text_embedder.embed(mem_obj.memory)
                ids = mem_obj.id
                while self.embedding_retriever.exists(ids):
                    ids = str(uuid.uuid4())
                    mem_obj.id = ids
                payload = {
                    "time_stamp": mem_obj.time_stamp,
                    "float_time_stamp": mem_obj.float_time_stamp,
                    "weekday": mem_obj.weekday,
                    "category": mem_obj.category,
                    "subcategory": mem_obj.subcategory,
                    "memory_class": mem_obj.memory_class,
                    "memory": mem_obj.memory,
                    "original_memory": mem_obj.original_memory,
                    "compressed_memory": mem_obj.compressed_memory,
                }
                self.embedding_retriever.insert(
                    vectors = [embedding_vector],
                    payloads = [payload],
                    ids = [ids],
                )
                inserted_count += 1

            self.logger.info(f"[{call_id}] Successfully inserted {inserted_count} entries to vector database")
            if construct_update_queue_trigger:
                self.logger.info(f"[{call_id}] Triggering update queue construction")
                self.construct_update_queue_all_entries(
                    top_k=20,
                    keep_top_n=10
                )
            
            if offline_update_trigger:
                self.logger.info(f"[{call_id}] Triggering offline update for all entries")
                self.offline_update_all_entries(
                    score_threshold = 0.8
                )

    def construct_update_queue_all_entries(self, top_k: int = 20, keep_top_n: int = 10, max_workers: int = 8):

        """
        Offline update all entries in parallel using multithreading.
        Each entry updates its own update_queue based on entries with earlier timestamps.

        Args:
            top_k (int): Number of nearest neighbors to consider for each entry.
            keep_top_n (int): Number of top entries to keep in update_queue.
            max_workers (int): Maximum number of threads to use.
        """
        # 依据当前库中所有向量，为每条记录搜集其“历史上最相似”的候选，形成 update_queue，
        # 以便后续执行跨时间的合并更新或冲突删除。
        call_id = f"construct_queue_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.logger.info(f"========== START {call_id} ==========")
        self.logger.info(f"[{call_id}] Parameters: top_k={top_k}, keep_top_n={keep_top_n}, max_workers={max_workers}")
        all_entries = self.embedding_retriever.get_all()
        self.logger.info(f"[{call_id}] Retrieved {len(all_entries)} entries from vector database")
        if not all_entries:
            self.logger.warning(f"[{call_id}] No entries found in database, skipping queue construction")
            self.logger.info(f"========== END {call_id} ==========")
            return
        updated_count = 0
        skipped_count = 0
        nonempty_queue_count = 0
        empty_queue_count = 0
        lock = threading.Lock()
        write_lock = threading.Lock()
        def _update_queue_construction(entry):
            nonlocal updated_count, skipped_count, nonempty_queue_count, empty_queue_count
            eid = entry["id"]
            payload = entry["payload"]
            vec = entry.get("vector")
            ts = payload.get("float_time_stamp", None)
            
            if vec is None or ts is None:
                self.logger.debug(f"[{call_id}] Skipping entry {eid}: missing vector={vec is None}, float_time_stamp={ts is None} ({ts})")
                with lock:
                    skipped_count += 1
                return

            # 只检索时间不晚于本条的候选（防止“未来”信息回流）
            hits = self.embedding_retriever.search(
                query_vector=vec,
                limit=top_k,
                filters={"float_time_stamp": {"lte": ts}}
            )

            candidates = []
            for h in hits:
                hid = h["id"]
                if hid == eid:
                    continue
                candidates.append({"id": hid, "score": h.get("score")})

            candidates.sort(key=lambda x: x["score"], reverse=True)
            update_queue = candidates[:keep_top_n]

            new_payload = dict(payload)
            new_payload["update_queue"] = update_queue

            if update_queue:
                with lock:
                    nonempty_queue_count += 1
                self.logger.debug(f"[{call_id}] Entry {eid} update_queue length={len(update_queue)} top_candidates=" + str(update_queue[:3]))
            else:
                with lock:
                    empty_queue_count += 1
                self.logger.debug(f"[{call_id}] Entry {eid} has no candidates after filtering (hits may be only itself)")

            with write_lock:
                self.embedding_retriever.update(vector_id=eid, vector=vec, payload=new_payload)

            with lock:
                updated_count += 1
        self.logger.info(f"[{call_id}] Starting parallel queue construction with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(_update_queue_construction, all_entries)
        self.logger.info(
            f"[{call_id}] Queue construction completed: {updated_count} updated, {skipped_count} skipped, "
            f"nonempty_queues={nonempty_queue_count}, empty_queues={empty_queue_count}"
        )
        self.logger.info(f"========== END {call_id} ==========")

    def offline_update_all_entries(self, score_threshold: float = 0.5, max_workers: int = 5):
        """
        Perform offline updates for all entries based on their update_queue, in parallel.

        Args:
            score_threshold (float): Minimum similarity score for considering update candidates.
            max_workers (int): Maximum number of worker threads.
        """
        # 遍历所有条目，找到把当前条目列入其 update_queue 的“来源条目们”；
        # 将这些来源作为候选事实，与当前条目对比：
        #  - 若冲突且候选更新更“新”，删除当前条目；
        #  - 若补充信息可合并，更新当前条目；
        #  - 若无关，忽略。
        call_id = f"offline_update_all_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.logger.info(f"========== START {call_id} ==========")
        self.logger.info(f"[{call_id}] Parameters: score_threshold={score_threshold}, max_workers={max_workers}")
        all_entries = self.embedding_retriever.get_all()
        self.logger.info(f"[{call_id}] Retrieved {len(all_entries)} entries from vector database")
        if not all_entries:
            self.logger.warning(f"[{call_id}] No entries found in database, skipping offline update")
            self.logger.info(f"========== END {call_id} ==========")
            return
        processed_count = 0
        updated_count = 0
        deleted_count = 0
        skipped_count = 0
        lock = threading.Lock()
        write_lock = threading.Lock()

        def update_entry(entry):
            nonlocal processed_count, updated_count, deleted_count, skipped_count
            
            eid = entry["id"]
            payload = entry["payload"]

            candidate_sources = []
            for other in all_entries:
                update_queue = other["payload"].get("update_queue", [])
                for candidate in update_queue:
                    if candidate["id"] == eid and candidate["score"] >= score_threshold:
                        candidate_sources.append(other)
                        break

            if not candidate_sources:
                with lock:
                    skipped_count += 1
                return

            with lock:
                processed_count += 1

            updated_entry = self.manager._call_update_llm(UPDATE_PROMPT, entry, candidate_sources)

            if updated_entry is None:
                return

            action = updated_entry.get("action")
            if action == "delete":
                with write_lock:
                    self.embedding_retriever.delete(eid)
                with lock:
                    deleted_count += 1
                self.logger.debug(f"[{call_id}] Deleted entry: {eid}")
            elif action == "update":
                new_payload = dict(payload)
                new_payload["memory"] = updated_entry.get("new_memory")
                vector = entry.get("vector")
                with write_lock:
                    self.embedding_retriever.update(vector_id=eid, vector=vector, payload=new_payload)
                with lock:
                    updated_count += 1
                self.logger.debug(f"[{call_id}] Updated entry: {eid}")
        self.logger.info(f"[{call_id}] Starting parallel offline update with {max_workers} workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(update_entry, all_entries)
        self.logger.info(f"[{call_id}] Offline update completed:")
        self.logger.info(f"[{call_id}]   - Processed: {processed_count} entries")
        self.logger.info(f"[{call_id}]   - Updated: {updated_count} entries")
        self.logger.info(f"[{call_id}]   - Deleted: {deleted_count} entries")
        self.logger.info(f"[{call_id}]   - Skipped (no candidates): {skipped_count} entries")
        self.logger.info(f"========== END {call_id} ==========")
    
    def retrieve(self, query: str, limit: int = 10, filters: Optional[dict] = None) -> str:
        """
        Retrieve relevant entries and return them as formatted strings.

        Args:
            query (str): The natural language query string.
            limit (int, optional): Number of results to return. Defaults to 10.
            filters (dict, optional): Optional filters to narrow down the search. Defaults to None.

        Returns:
            list[str]: A list of formatted strings containing time_stamp, weekday, and memory.
        """
        # 基于文本嵌入的相似度检索，返回格式化后的“时间 星期 记忆”字符串列表
        call_id = f"retrieve_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.logger.info(f"========== START {call_id} ==========")
        self.logger.info(f"[{call_id}] Query: {query}")
        self.logger.info(f"[{call_id}] Parameters: limit={limit}, filters={filters}")
        self.logger.debug(f"[{call_id}] Generating embedding for query")
        query_vector = self.text_embedder.embed(query)
        self.logger.debug(f"[{call_id}] Query embedding dimension: {len(query_vector)}")
        self.logger.info(f"[{call_id}] Searching vector database")
        results = self.embedding_retriever.search(
            query_vector=query_vector,
            limit=limit,
            filters=filters,
            return_full=True,
        )
        self.logger.info(f"[{call_id}] Found {len(results)} results")
        formatted_results = []
        for r in results:
            payload = r.get("payload", {})
            time_stamp = payload.get("time_stamp", "")
            weekday = payload.get("weekday", "")
            memory = payload.get("memory", "")
            formatted_results.append(f"{time_stamp} {weekday} {memory}")
            
        result_string = "\n".join(formatted_results)
        self.logger.info(f"[{call_id}] Formatted {len(formatted_results)} results into output string")
        self.logger.debug(f"[{call_id}] Output string length: {len(result_string)} characters")
        self.logger.info(f"========== END {call_id} ==========")
        return result_string

