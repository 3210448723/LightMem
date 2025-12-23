<<<<<<< HEAD
"""
LightMem LOCOMO 数据集记忆构建模块

本模块是 LightMem 大模型对话记忆系统的入口脚本，主要用于：
1. 加载 LOCOMO 数据集中的多轮对话数据
2. 利用 LightMem 系统为每个对话样本构建长期记忆
3. 支持多进程/多线程并行处理，提升处理效率
4. 执行离线记忆更新，优化记忆存储结构
5. 记录详细的处理日志和统计信息

LOCOMO 数据集特点：
- 包含多会话、多轮次的长对话
- 每个会话带有时间戳信息
- 支持多模态内容（文本+图像描述）

使用方式：
    python add_locomo.py

依赖组件：
    - LightMem: 核心记忆管理系统
    - Qdrant: 向量数据库，用于存储和检索记忆嵌入
    - LLMLingua-2: 文本压缩模型，用于压缩对话内容
    - OpenAI API: 用于调用大语言模型进行记忆提取和更新

作者: LightMem Team
"""

# ==================== 导入依赖库 ====================
from openai import OpenAI  # OpenAI API 客户端（当前未直接使用，但 LightMem 内部依赖）
import json  # JSON 数据解析
from tqdm import tqdm  # 进度条显示
import datetime  # 日期时间处理
import time  # 时间计量
import os  # 操作系统接口
import logging  # 日志记录
from lightmem.memory.lightmem import LightMemory  # LightMem 核心记忆类
from lightmem.configs.retriever.embeddingretriever.qdrant import QdrantConfig  # Qdrant 配置类
from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant  # Qdrant 向量检索器
from prompts import METADATA_GENERATE_PROMPT_locomo  # LOCOMO 专用元数据生成提示语
import sqlite3  # SQLite 数据库接口（用于读取 Qdrant 存储信息）
import shutil  # 文件/目录操作
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed  # 并行处理
import multiprocessing as mp  # 多进程支持


# ==================== 配置参数区域 ====================
# 本区域定义了所有可配置的参数，修改这些参数可以调整系统行为
# -------------------- 日志配置 --------------------
# 日志根目录，所有运行日志将存储在此目录下
=======
from openai import OpenAI
import json
from tqdm import tqdm
import datetime
import time
import os
import logging
from lightmem.memory.lightmem import LightMemory
from lightmem.configs.retriever.embeddingretriever.qdrant import QdrantConfig
from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant
from prompts import METADATA_GENERATE_PROMPT_locomo
import sqlite3
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# ============ Configuration ============
>>>>>>> 70437d4545d41981a7ce1b0a4b6998b4ad0bc3f3
LOGS_ROOT = "./logs"
# 每次运行创建独立的时间戳目录，便于追溯和管理
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOGS_ROOT, RUN_TIMESTAMP)
os.makedirs(RUN_LOG_DIR, exist_ok=True)  # 确保日志目录存在

# -------------------- API 配置 --------------------
# OpenAI API 密钥列表，支持多个密钥轮换使用以提高并发能力
# 多个密钥可以分配给不同的并行任务，避免单个密钥的速率限制
API_KEYS = [
    'sk-n5HTCSKC9XsLtn2zwAdBxaPxF7ubNWgAxwapGJ5Buxd24G80',
]
# API 服务的基础 URL（可以是 OpenAI 官方或兼容的代理服务）
API_BASE_URL = 'http://100.78.197.38:16868/'
# 使用的大语言模型名称
# LLM_MODEL = 'gpt-4o-mini'
LLM_MODEL = 'QwQ-32B'

# -------------------- 模型路径配置 --------------------
# LLMLingua-2 压缩模型路径，用于对话内容压缩
# 该模型可以在保持语义的前提下大幅减少 token 数量
LLMLINGUA_MODEL_PATH = 'microsoft/llmlingua-2-xlm-roberta-large-meetingbank'
# 文本嵌入模型路径，用于将文本转换为向量表示
# all-MiniLM-L6-v2 是一个轻量级但效果良好的句子嵌入模型
EMBEDDING_MODEL_PATH = 'all-MiniLM-L6-v2'

# -------------------- 数据配置 --------------------
# LOCOMO 数据集路径，包含需要处理的对话样本
DATA_PATH = '/home/user/yuanjinmin/LightMem/dataset/origin_data/locomo10.json'
# 数据集类型标识
DATASET_TYPE = 'locomo'

# -------------------- Qdrant 存储目录配置 --------------------
# 预更新目录：存储 add_memory 阶段后的记忆状态（作为备份）
# 这个目录保存了执行离线更新前的记忆快照
QDRANT_PRE_UPDATE_DIR = './qdrant_pre_update'
# 后更新目录：存储最终的记忆状态（经过离线更新优化后）
QDRANT_POST_UPDATE_DIR = './qdrant_post_update'

# 确保存储目录存在
os.makedirs(QDRANT_PRE_UPDATE_DIR, exist_ok=True)
os.makedirs(QDRANT_POST_UPDATE_DIR, exist_ok=True)

# -------------------- 并行处理配置 --------------------
# 最大并行工作进程/线程数
# 建议根据 CPU 核心数和 API 密钥数量合理设置
# MAX_WORKERS = 1
MAX_WORKERS = 4
# 是否使用进程池（True）还是线程池（False）
# 进程池适合 CPU 密集型任务，线程池适合 I/O 密集型任务
# 由于涉及模型推理，推荐使用进程池
USE_PROCESS_POOL = True

# -------------------- GPU 配置 --------------------
# 可用的 GPU ID 列表，每个进程将分配一个独立的 GPU
# 总共4张GPU，分配给4个进程
# AVAILABLE_GPUS = [0,]
AVAILABLE_GPUS = [0, 1, 2, 3]


# ==================== 工具函数区域 ====================

def get_process_logger(sample_id):
    """
    为指定样本创建独立的日志记录器
    
    每个并行处理的样本都有独立的日志文件，便于：
    1. 追踪单个样本的处理过程
    2. 并行处理时避免日志混乱
    3. 问题排查时快速定位
    
    参数:
        sample_id (str): 样本唯一标识符，用于命名日志文件
        
    返回:
        logging.Logger: 配置好的日志记录器实例
        
    日志输出:
        - 文件输出: {RUN_LOG_DIR}/{sample_id}.log
        - 控制台输出: 同步显示处理进度
    """
    # 创建以样本 ID 命名的日志记录器
    logger = logging.getLogger(f"lightmem.parallel.{sample_id}")
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器（多次调用时的保护机制）
    if not logger.handlers:
        # 文件处理器：将日志写入独立的日志文件
        fh = logging.FileHandler(
            os.path.join(RUN_LOG_DIR, f"{sample_id}.log"),
            mode='w'  # 写模式，每次运行覆盖旧日志
        )
        fh.setLevel(logging.INFO)
        # 日志格式：时间 - 记录器名称 - 级别 - 消息
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # 控制台处理器：实时显示处理进度
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


def parse_locomo_timestamp(timestamp_str):
    """
    解析 LOCOMO 数据集特有的时间戳格式
    
    LOCOMO 数据集的时间戳格式为: "(HH:MM AM/PM on DD Month, YYYY)"
    例如: "(10:30 AM on 15 January, 2023)"
    
    本函数将其转换为标准的数据库时间格式: "YYYY-MM-DD HH:MM:SS"
    
    参数:
        timestamp_str (str): LOCOMO 格式的时间戳字符串
        
    返回:
        str: 标准格式的时间戳字符串 "YYYY-MM-DD HH:MM:SS"
             如果解析失败，返回原始字符串
    """
    # 去除首尾的括号
    timestamp_str = timestamp_str.strip("()")
    try:
        # 使用 strptime 解析特定格式
        # %I: 12小时制小时, %M: 分钟, %p: AM/PM
        # %d: 日, %B: 月份全名, %Y: 四位年份
        dt = datetime.datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
        # 转换为标准数据库格式
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        # 解析失败时返回原始字符串，保证程序健壮性
        return timestamp_str


def extract_locomo_sessions(conversation_dict):
    """
    从 LOCOMO 对话数据中提取会话信息
    
    LOCOMO 数据集的对话结构：
    - 一个对话包含多个会话（session）
    - 每个会话有独立的时间戳
    - 每个会话包含多个对话轮次（turn）
    - 可能包含多模态内容（图像描述）
    
    参数:
        conversation_dict (dict): LOCOMO 格式的对话字典，包含：
            - speaker_a/speaker_b: 两个对话者的名称
            - session_1, session_2, ...: 各会话的对话内容
            - session_1_date_time, ...: 各会话的时间戳
            
    返回:
        tuple: 包含四个元素的元组
            - sessions (list): 消息列表的列表，每个内部列表是一个会话
            - timestamps (list): 每个会话对应的时间戳
            - speaker_a (str): 第一个对话者的名称
            - speaker_b (str): 第二个对话者的名称
    """
    # 获取对话者名称，提供默认值
    speaker_a = conversation_dict.get('speaker_a', 'Speaker_A')
    speaker_b = conversation_dict.get('speaker_b', 'Speaker_B')
    
    # 动态发现所有会话编号
    # LOCOMO 数据的会话键格式为 "session_N"，N 为编号
    session_nums = set()
    for key in conversation_dict.keys():
        # 筛选出会话键（排除时间戳键）
        if key.startswith('session_') and not key.endswith('_date_time'):
            try:
                num = int(key.split('_')[1])
                session_nums.add(num)
            except:
                continue
    
    sessions = []  # 存储所有会话的消息
    timestamps = []  # 存储每个会话的时间戳
    
    # 按会话编号顺序处理
    for num in sorted(session_nums):
        session_key = f'session_{num}'
        timestamp_key = f'{session_key}_date_time'
        
        # 跳过不存在的会话
        if session_key not in conversation_dict:
            continue
            
        session_data = conversation_dict[session_key]
        timestamp = conversation_dict.get(timestamp_key, '')
        
        messages = []
        # 处理会话中的每个对话轮次
        for turn in session_data:
            speaker_name = turn['speaker']
            # 将说话者名称映射为标准化的 ID
            speaker_id = 'speaker_a' if speaker_name == speaker_a else 'speaker_b'
            content = turn['text']
            
            # 处理多模态内容：如果有图像描述，附加到文本内容中
            if 'blip_caption' in turn and turn['blip_caption']:
                content = f"{content} (image description: {turn['blip_caption']})"
            
            # 构造用户消息（对话内容）
            messages.append({
                "role": "user",
                "content": content,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
            })
            # 构造空的助手回复（LightMem 格式要求）
            # 这是为了符合 LightMem 的消息格式，表示一个完整的对话轮次
            messages.append({
                "role": "assistant",
                "content": "",
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
            })
        
        sessions.append(messages)
        timestamps.append(parse_locomo_timestamp(timestamp))
    
    return sessions, timestamps, speaker_a, speaker_b


def load_lightmem(collection_name, api_key):
    """
    创建并配置 LightMem 记忆系统实例
    
    LightMem 是一个多阶段的大模型对话记忆系统，本函数配置其完整的处理流水线：
    1. 预压缩（Pre-compression）: 使用 LLMLingua-2 压缩对话文本
    2. 主题分割（Topic Segmentation）: 将长对话切分为语义连贯的片段
    3. 元数据生成（Metadata Generation）: 为记忆条目生成描述性元数据
    4. 文本摘要（Text Summary）: 生成对话内容的简洁摘要
    5. 向量嵌入（Embedding）: 将文本转换为向量用于相似度检索
    6. 存储检索（Storage & Retrieval）: 使用 Qdrant 进行向量存储和检索
    
    参数:
        collection_name (str): Qdrant 集合名称，通常使用样本 ID
        api_key (str): OpenAI API 密钥
        
    返回:
        LightMemory: 配置完成的 LightMem 实例
        
    配置说明:
        - pre_compress: 启用文本预压缩以减少 token 消耗
        - topic_segment: 启用主题分割以获得更细粒度的记忆单元
        - metadata_generate: 启用元数据生成以提升检索效果
        - text_summary: 启用文本摘要以便快速预览记忆内容
        - update="offline": 采用离线更新策略，先积累再批量处理
    """
    config = {
        # ==================== 预压缩配置 ====================
        # 启用预压缩功能，可显著减少 LLM 调用的 token 消耗
        "pre_compress": True,
        "pre_compressor": {
            # 使用 LLMLingua-2 作为压缩模型
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": LLMLINGUA_MODEL_PATH,  # 模型路径
                    "device_map": "cuda",  # 使用 GPU 加速
                    "use_llmlingua2": True,  # 使用 LLMLingua-2 版本
                },
                "compress_config": {
                    "instruction": "",  # 压缩指令（空表示使用默认）
                    "rate": 0.6,  # 压缩率，保留 60% 的内容
                    "target_token": -1  # 目标 token 数（-1 表示使用 rate）
                },
            }
        },
        
        # ==================== 主题分割配置 ====================
        # 启用主题分割，将长对话切分为语义连贯的片段
        "topic_segment": True,
        # 共享预压缩和主题分割的模型实例，节省内存
        "precomp_topic_shared": True,
        "topic_segmenter": {
            # 复用 LLMLingua-2 模型进行主题分割
            "model_name": "llmlingua-2",
        },
        
        # ==================== 消息处理配置 ====================
        # 只使用用户消息构建记忆，过滤掉空的助手回复
        "messages_use": "user_only",
        # 启用元数据生成，为记忆条目添加描述性信息
        "metadata_generate": True,
        # 启用文本摘要功能
        "text_summary": True,
        
        # ==================== LLM 配置 ====================
        # 记忆管理器使用的大语言模型配置
        "memory_manager": {
            "model_name": "openai",  # 使用 OpenAI 兼容的 API
            "configs": {
                "model": LLM_MODEL,  # 模型名称
                "api_key": api_key,  # API 密钥
                "max_tokens": 16000,  # 最大输出 token 数
                "openai_base_url": API_BASE_URL  # API 服务地址
            },
        },
        
        # ==================== 记忆提取配置 ====================
        # 记忆提取的阈值，低于此值的内容不会被提取为独立记忆
        "extract_threshold": 0.1,
        
        # ==================== 索引策略配置 ====================
        # 使用嵌入向量进行索引
        "index_strategy": "embedding",
        # 文本嵌入器配置
        "text_embedder": {
            "model_name": "huggingface",  # 使用 HuggingFace 模型
            "configs": {
                "model": EMBEDDING_MODEL_PATH,  # 嵌入模型路径
                "embedding_dims": 384,  # 嵌入向量维度
                "model_kwargs": {"device": "cuda"},  # 使用 GPU
            },
        },
        
        # ==================== 检索策略配置 ====================
        # 使用嵌入向量进行检索
        "retrieve_strategy": "embedding",
        # Qdrant 向量数据库配置
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection_name,  # 集合名称
                "embedding_model_dims": 384,  # 向量维度
                # 存储路径，每个样本独立的存储目录
                "path": f'{QDRANT_POST_UPDATE_DIR}/{collection_name}',
            }
        },
        
        # ==================== 更新策略配置 ====================
        # 使用离线更新策略，适合批量处理场景
        # 离线更新会在所有记忆添加完成后统一进行优化
        "update": "offline",
        
        # ==================== 日志配置 ====================
        "logging": {
            "level": "DEBUG",  # 日志级别
            "file_enabled": True,  # 启用文件日志
            "log_dir": RUN_LOG_DIR,  # 日志目录
        },
        
        # ==================== LOCOMO 特定配置 ====================
        "locomo_style": True,  # 启用 LOCOMO 数据集特定的处理逻辑
        "judge_only": False,   # 当试验结果存在且打算只做评测时，启用此选项
        "use_llm_judge": True, # 是否启用 LLM 评测判定
    }
    
    # 使用配置字典创建 LightMem 实例
    lightmem = LightMemory.from_config(config)
    return lightmem


def collection_entry_count(collection_name, base_dir):
    """
    获取 Qdrant 集合中的记忆条目数量
    
    该函数尝试多种方式获取集合中的条目数量：
    1. 首选：通过 Qdrant API 获取
    2. 备选：直接读取 SQLite 存储文件
    
    参数:
        collection_name (str): Qdrant 集合名称
        base_dir (str): Qdrant 存储的基础目录
        
    返回:
        int: 集合中的条目数量
             - 正数：实际条目数
             - 0：集合为空或不存在
             - -1：获取失败
    """
    try:
        # 方法1：通过 Qdrant API 获取
        cfg = QdrantConfig(
            collection_name=collection_name,
            path=base_dir,
            embedding_model_dims=384,
            on_disk=True,  # 使用磁盘存储模式
        )
        q = Qdrant(cfg)
        try:
            # 获取所有点，不包含向量和载荷数据以提高效率
            points = q.get_all(with_vectors=False, with_payload=False)
            if points:
                return len(points)
        except Exception:
            pass

        # 方法2：直接读取 SQLite 数据库
        # Qdrant 的本地存储使用 SQLite 作为底层存储引擎
        storage_sqlite = os.path.join(
            base_dir, collection_name, 'collection', collection_name, 'storage.sqlite'
        )
        if not os.path.exists(storage_sqlite):
            return 0

        try:
            conn = sqlite3.connect(storage_sqlite)
            cur = conn.execute("SELECT count(*) FROM points")
            row = cur.fetchone()
            conn.close()
            if row:
                return int(row[0])
            return 0
        except Exception:
            return -1
    except Exception:
        # 最后的备选方案：直接尝试读取 SQLite
        storage_sqlite = os.path.join(
            base_dir, collection_name, 'collection', collection_name, 'storage.sqlite'
        )
        if os.path.exists(storage_sqlite):
            try:
                conn = sqlite3.connect(storage_sqlite)
                cur = conn.execute("SELECT count(*) FROM points")
                row = cur.fetchone()
                conn.close()
                if row:
                    return int(row[0])
                return 0
            except Exception:
                return -1
        return -1


# ==================== 核心处理函数区域 ====================

def process_single_sample(sample, api_key, gpu_id):
    """
    处理单个 LOCOMO 数据样本的完整流程
    
    这是整个记忆构建流程的核心函数，包含三个主要阶段：
    
    阶段1 - 记忆添加 (Add Memory):
        - 按会话和轮次逐步添加对话内容到 LightMem
        - 触发记忆提取、压缩和索引过程
        - 在最后一个轮次强制执行分割和提取
    
    阶段2 - 状态备份 (Backup):
        - 将 add_memory 阶段后的 Qdrant 存储复制到备份目录
        - 保存更新前的状态，便于对比分析
    
    阶段3 - 离线更新 (Offline Update):
        - 构建更新队列，分析所有记忆条目
        - 执行记忆合并、去重和优化
        - 根据相似度阈值决定是否合并条目
    
    参数:
        sample (dict): LOCOMO 数据样本，包含：
            - sample_id: 样本唯一标识符
            - conversation: 对话数据字典
        api_key (str): 用于此任务的 OpenAI API 密钥
        gpu_id (int): 分配给此进程的 GPU ID
        
    返回:
        dict: 处理结果字典，包含：
            - sample_id: 样本标识
            - status: 'success' 或 'failed'
            - 成功时包含各种统计信息（时间、token 使用量等）
            - 失败时包含 error 信息
    """
    # 设置当前进程使用的 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    sample_id = sample['sample_id']
    # 为当前样本创建独立的日志记录器
    logger = get_process_logger(sample_id)
    
    try:
        # ==================== 处理开始日志 ====================
        logger.info(f"{'='*70}")
        logger.info(f"[Worker {mp.current_process().name}] Processing: {sample_id}")
        logger.info(f"[Worker {mp.current_process().name}] Using API Key: {api_key[:20]}...")
        logger.info(f"[Worker {mp.current_process().name}] Using GPU: {gpu_id}")
        logger.info(f"{'='*70}")
        
        # ==================== 数据预处理 ====================
        # 从样本中提取对话数据并解析为会话格式
        conversation = sample['conversation']
        sessions, timestamps, speaker_a, speaker_b = extract_locomo_sessions(conversation)
        
        logger.info(f"  Sessions: {len(sessions)}")
        logger.info(f"  Speakers: {speaker_a}, {speaker_b}")
        logger.info(f"\n{'─'*70}")
        logger.info("Phase 1: Building memory (add_memory)")
        logger.info(f"{'─'*70}")
        
        # ==================== 阶段1: 记忆添加 ====================
        # 创建 LightMem 实例
        lightmem = load_lightmem(collection_name=sample_id, api_key=api_key)
        logger.info(f"lightmem config: {lightmem.config}")
        
        # 记录初始统计数据，用于计算增量
        initial_stats = lightmem.get_token_statistics()
        case_start_time = time.time()
        add_memory_start_time = time.time()
        
        # 保存 add_memory 操作前的 token 统计
        initial_add_tokens = initial_stats['llm']['add_memory']['total_tokens']
        initial_add_calls = initial_stats['llm']['add_memory']['calls']
        
        # 遍历每个会话，逐轮次添加记忆
        # 这种方式模拟了真实对话场景中记忆的逐步积累过程
        for session, timestamp in zip(sessions, timestamps):
            # 确保会话以用户消息开始
            while session and session[0]["role"] != "user":
                session.pop(0)
            
            # 计算会话中的对话轮次数（每轮包含 user + assistant 两条消息）
            num_turns = len(session) // 2
            
            for turn_idx in range(num_turns):
                # 提取当前轮次的消息对
                turn_messages = session[turn_idx*2 : turn_idx*2 + 2]
                
                # 验证消息格式正确性
                if len(turn_messages) < 2 or turn_messages[0]["role"] != "user" or turn_messages[1]["role"] != "assistant":
                    continue
                
                # 为每条消息添加时间戳
                for msg in turn_messages:
                    msg["time_stamp"] = timestamp
                
                # 判断是否为最后一个轮次
                # 在最后一个轮次需要强制执行分割和提取，确保所有内容都被处理
                is_last_turn = (session is sessions[-1] and turn_idx == num_turns - 1)
                
                # 调用 LightMem 添加记忆
                # force_segment: 强制执行主题分割
                # force_extract: 强制执行记忆提取
                lightmem.add_memory(
                    messages=turn_messages,
                    METADATA_GENERATE_PROMPT=METADATA_GENERATE_PROMPT_locomo,
                    force_segment=is_last_turn,
                    force_extract=is_last_turn,
                )
        
        # 记录 add_memory 阶段的时间消耗
        add_memory_end_time = time.time()
        add_memory_duration = add_memory_end_time - add_memory_start_time
        
        # 计算 add_memory 阶段的 token 使用量
        add_memory_stats = lightmem.get_token_statistics()
        case_add_tokens = add_memory_stats['llm']['add_memory']['total_tokens'] - initial_add_tokens
        case_add_calls = add_memory_stats['llm']['add_memory']['calls'] - initial_add_calls
        case_add_prompt = add_memory_stats['llm']['add_memory']['prompt_tokens'] - initial_stats['llm']['add_memory']['prompt_tokens']
        case_add_completion = add_memory_stats['llm']['add_memory']['completion_tokens'] - initial_stats['llm']['add_memory']['completion_tokens']
        
        # 获取 add_memory 后的条目数量
        after_add_count = collection_entry_count(sample_id, QDRANT_POST_UPDATE_DIR)
        logger.info(f"✓ Add_memory completed: {after_add_count} entries in {add_memory_duration:.2f}s")
        
        # ==================== 阶段2: 状态备份 ====================
        logger.info(f"\n{'─'*70}")
        logger.info("Phase 2: Backing up pre-update state")
        logger.info(f"{'─'*70}")
        
        # 定义源目录和备份目录
        source_dir = f'{QDRANT_POST_UPDATE_DIR}/{sample_id}'
        backup_dir = f'{QDRANT_PRE_UPDATE_DIR}/{sample_id}'
        
        backup_start_time = time.time()
        
        # 如果备份目录已存在，先删除（确保是最新备份）
        if os.path.exists(backup_dir):
            logger.info(f"  Removing existing backup...")
            shutil.rmtree(backup_dir)
        
        # 复制整个 Qdrant 存储目录
        logger.info(f"  Copying: {source_dir} -> {backup_dir}")
        shutil.copytree(source_dir, backup_dir)
        
        backup_end_time = time.time()
        backup_duration = backup_end_time - backup_start_time
        
        # 验证备份完整性
        pre_update_count = collection_entry_count(sample_id, QDRANT_PRE_UPDATE_DIR)
        logger.info(f"✓ Backup completed: {pre_update_count} entries in {backup_duration:.2f}s")
        
        # ==================== 阶段3: 离线更新 ====================
        logger.info(f"\n{'─'*70}")
        logger.info("Phase 3: Performing offline update")
        logger.info(f"{'─'*70}")
        
        # 记录更新前的 token 统计
        update_start_stats = lightmem.get_token_statistics()
        initial_update_tokens = update_start_stats['llm']['update']['total_tokens']
        initial_update_calls = update_start_stats['llm']['update']['calls']
        
        update_start_time = time.time()
        
        # 构建更新队列：分析所有记忆条目，找出需要更新的候选
        lightmem.construct_update_queue_all_entries()
        
        # 执行离线更新：根据相似度阈值合并相似的记忆条目
        # score_threshold=0.9 表示相似度超过 90% 的条目会被合并
        lightmem.offline_update_all_entries(score_threshold=0.9)
        
        update_end_time = time.time()
        update_duration = update_end_time - update_start_time
        
        # 计算更新阶段的 token 使用量
        update_end_stats = lightmem.get_token_statistics()
        case_update_tokens = update_end_stats['llm']['update']['total_tokens'] - initial_update_tokens
        case_update_calls = update_end_stats['llm']['update']['calls'] - initial_update_calls
        case_update_prompt = update_end_stats['llm']['update']['prompt_tokens'] - update_start_stats['llm']['update']['prompt_tokens']
        case_update_completion = update_end_stats['llm']['update']['completion_tokens'] - update_start_stats['llm']['update']['completion_tokens']
        
        # 获取更新后的条目数量
        post_update_count = collection_entry_count(sample_id, QDRANT_POST_UPDATE_DIR)
        logger.info(f"✓ Update completed: {post_update_count} entries in {update_duration:.2f}s")
        
        # 记录总处理时间
        case_end_time = time.time()
        case_total_duration = case_end_time - case_start_time
        
        # ==================== 输出处理摘要 ====================
        logger.info(f"\n{'='*70}")
        logger.info(f"SUMMARY: {sample_id}")
        logger.info(f"{'='*70}")
        
        # 存储信息摘要
        logger.info(f"\n[Storage Information]")
        logger.info(f"  Pre-update:  {QDRANT_PRE_UPDATE_DIR}/{sample_id} ({pre_update_count} entries)")
        logger.info(f"  Post-update: {QDRANT_POST_UPDATE_DIR}/{sample_id} ({post_update_count} entries)")
        logger.info(f"  Change:      {post_update_count - pre_update_count:+d} entries")
        
        # 时间统计摘要
        logger.info(f"\n[Time Statistics]")
        logger.info(f"  Total:       {case_total_duration:.2f}s")
        logger.info(f"  ├─ Add:      {add_memory_duration:.2f}s ({add_memory_duration/case_total_duration*100:.1f}%)")
        logger.info(f"  ├─ Backup:   {backup_duration:.2f}s ({backup_duration/case_total_duration*100:.1f}%)")
        logger.info(f"  └─ Update:   {update_duration:.2f}s ({update_duration/case_total_duration*100:.1f}%)")
        
        # Add Memory 阶段的 Token 统计
        logger.info(f"\n[Token Statistics - Add Memory]")
        logger.info(f"  Calls:       {case_add_calls}")
        logger.info(f"  Prompt:      {case_add_prompt:,}")
        logger.info(f"  Completion:  {case_add_completion:,}")
        logger.info(f"  Total:       {case_add_tokens:,}")
        
        # Update 阶段的 Token 统计
        logger.info(f"\n[Token Statistics - Update]")
        logger.info(f"  Calls:       {case_update_calls}")
        logger.info(f"  Prompt:      {case_update_prompt:,}")
        logger.info(f"  Completion:  {case_update_completion:,}")
        logger.info(f"  Total:       {case_update_tokens:,}")
        
        # 总体使用量摘要
        logger.info(f"\n[Total Usage]")
        logger.info(f"  API Calls:   {case_add_calls + case_update_calls}")
        logger.info(f"  Tokens:      {case_add_tokens + case_update_tokens:,}")
        logger.info(f"{'='*70}\n")
        
        # 返回处理成功的结果
        return {
            'sample_id': sample_id,
            'status': 'success',
            'pre_update_count': pre_update_count,      # 更新前的条目数
            'post_update_count': post_update_count,    # 更新后的条目数
            'total_duration': case_total_duration,      # 总处理时间
            'add_memory_duration': add_memory_duration, # 添加记忆耗时
            'backup_duration': backup_duration,         # 备份耗时
            'update_duration': update_duration,         # 更新耗时
            'add_tokens': case_add_tokens,              # 添加记忆消耗的 token
            'add_calls': case_add_calls,                # 添加记忆的 API 调用次数
            'update_tokens': case_update_tokens,        # 更新消耗的 token
            'update_calls': case_update_calls,          # 更新的 API 调用次数
        }
        
    except Exception as e:
        # 异常处理：记录错误并返回失败状态
        logger.error(f"✗ {sample_id} failed: {str(e)}", exc_info=True)
        return {
            'sample_id': sample_id,
            'status': 'failed',
            'error': str(e)
        }


# ==================== 主函数区域 ====================

def main():
    """
    程序主入口函数
    
    执行流程：
    1. 初始化主日志记录器
    2. 加载 LOCOMO 数据集
    3. 扫描现有集合，筛选需要处理的样本
    4. 使用并行执行器处理所有待处理样本
    5. 汇总并输出最终统计信息
    
    并行处理策略：
    - 支持进程池和线程池两种模式
    - API 密钥轮换分配给不同任务
    - 使用 as_completed 实现实时进度更新
    
    断点续传机制：
    - 扫描已存在的集合
    - 跳过已完成处理的样本
    - 只处理缺失或不完整的样本
    """
    # ==================== 初始化主日志记录器 ====================
    main_logger = logging.getLogger("lightmem.parallel.main")
    main_logger.setLevel(logging.INFO)
    
    # 文件处理器：主日志文件
    fh = logging.FileHandler(os.path.join(RUN_LOG_DIR, "main.log"), mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    main_logger.addHandler(fh)
    
    # 控制台处理器：实时显示
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    main_logger.addHandler(ch)
    
    # ==================== 输出配置信息 ====================
    main_logger.info("=" * 70)
    main_logger.info("PARALLEL MEMORY BUILDING")
    main_logger.info("=" * 70)
    main_logger.info(f"Workers:         {MAX_WORKERS}")
    main_logger.info(f"API Keys:        {len(API_KEYS)}")
    main_logger.info(f"Executor:        {'ProcessPool' if USE_PROCESS_POOL else 'ThreadPool'}")
    main_logger.info(f"Post-update dir: {QDRANT_POST_UPDATE_DIR}")
    main_logger.info(f"Pre-update dir:  {QDRANT_PRE_UPDATE_DIR}")
    main_logger.info("=" * 70)
    
    # ==================== 加载数据集 ====================
    data = json.load(open(DATA_PATH, "r"))
    main_logger.info(f"\nLoaded {len(data)} samples from dataset")
    
    # ==================== 扫描现有集合状态 ====================
    # 实现断点续传：检查哪些样本已经处理完成，哪些需要重新处理
    main_logger.info("\n" + "=" * 70)
    main_logger.info("Scanning existing collections...")
    main_logger.info("=" * 70)
    
    missing = []  # 存储需要处理的样本
    for sample in data:
        sample_id = sample['sample_id']
        
        # 检查两个存储目录是否存在
        pre_update_dir = f'{QDRANT_PRE_UPDATE_DIR}/{sample_id}'
        post_update_dir = f'{QDRANT_POST_UPDATE_DIR}/{sample_id}'
        
        pre_exists = os.path.exists(pre_update_dir)
        post_exists = os.path.exists(post_update_dir)
        
        # 如果两个目录都存在，检查是否有有效数据
        if pre_exists and post_exists:
            pre_count = collection_entry_count(sample_id, QDRANT_PRE_UPDATE_DIR)
            post_count = collection_entry_count(sample_id, QDRANT_POST_UPDATE_DIR)
            
            # 只有当两个集合都有有效数据时，才认为处理完成
            if pre_count > 0 and post_count > 0:
                main_logger.info(
                    f"✓ {sample_id}: Complete "
                    f"(pre={pre_count}, post={post_count})"
                )
                continue
        
        # 记录缺失或不完整的状态
        status = []
        if not pre_exists:
            status.append("pre_missing")  # 预更新目录不存在
        elif collection_entry_count(sample_id, QDRANT_PRE_UPDATE_DIR) <= 0:
            status.append("pre_empty")    # 预更新目录为空
            
        if not post_exists:
            status.append("post_missing") # 后更新目录不存在
        elif collection_entry_count(sample_id, QDRANT_POST_UPDATE_DIR) <= 0:
            status.append("post_empty")   # 后更新目录为空
        
        main_logger.info(f"✗ {sample_id}: Needs processing ({', '.join(status)})")
        missing.append(sample)
    
    main_logger.info(f"\nScan complete: {len(missing)}/{len(data)} samples need processing\n")
    
    # 如果没有需要处理的样本，直接退出
    if not missing:
        main_logger.info("All samples complete. Exiting.")
        return
    
    # ==================== 开始并行处理 ====================
    main_logger.info("=" * 70)
    main_logger.info(f"Processing {len(missing)} samples in parallel")
    main_logger.info("=" * 70)
    
    # 输出 API 密钥分配方案
    # 使用轮换策略将样本分配给不同的 API 密钥和 GPU
    main_logger.info("\nAPI Key and GPU assignment:")
    for idx, sample in enumerate(missing):
        api_key_idx = idx % len(API_KEYS)  # 轮换分配
        api_key = API_KEYS[api_key_idx]
        gpu_id = AVAILABLE_GPUS[idx % len(AVAILABLE_GPUS)]  # 轮换分配 GPU
        main_logger.info(
            f"  Sample [{idx}] {sample['sample_id'][:30]}... "
            f"→ API Key [{api_key_idx}] ({api_key[:20]}...) → GPU {gpu_id}"
        )
    main_logger.info("")
    
    # 记录处理开始时间
    start_time = time.time()
    results = []        # 存储所有处理结果
    failed_samples = [] # 存储失败的样本 ID
    
    # 根据配置选择执行器类型
    # ProcessPoolExecutor: 多进程，适合 CPU 密集型任务
    # ThreadPoolExecutor: 多线程，适合 I/O 密集型任务
    ExecutorClass = ProcessPoolExecutor if USE_PROCESS_POOL else ThreadPoolExecutor
    
    with ExecutorClass(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务到执行器
        future_to_sample = {}
        for idx, sample in enumerate(missing):
            # 轮换分配 API 密钥
            api_key_idx = idx % len(API_KEYS)
            api_key = API_KEYS[api_key_idx]
            # 轮换分配 GPU
            gpu_id = AVAILABLE_GPUS[idx % len(AVAILABLE_GPUS)]
            
            # 提交异步任务
            future = executor.submit(process_single_sample, sample, api_key, gpu_id)
            future_to_sample[future] = sample
        
        # 使用 tqdm 显示进度条，并实时处理完成的任务
        with tqdm(total=len(missing), desc="Building memories") as pbar:
            # as_completed 返回已完成的 future，不保证顺序
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    # 获取任务结果
                    result = future.result()
                    results.append(result)
                    
                    # 更新进度条显示
                    if result['status'] == 'success':
                        pbar.set_postfix_str(f"✓ {result['sample_id']}")
                    else:
                        failed_samples.append(result['sample_id'])
                        pbar.set_postfix_str(f"✗ {result['sample_id']}")
                        
                except Exception as e:
                    # 处理意外异常
                    main_logger.error(f"Unexpected error for {sample['sample_id']}: {e}", exc_info=True)
                    failed_samples.append(sample['sample_id'])
                
                pbar.update(1)
    
    # 记录处理结束时间
    end_time = time.time()
    total_duration = end_time - start_time
    
    # ==================== 输出最终统计报告 ====================
    main_logger.info("\n" + "=" * 70)
    main_logger.info("PROCESSING COMPLETE")
    main_logger.info("=" * 70)
    
    # 筛选成功处理的结果
    successful = [r for r in results if r['status'] == 'success']
    
    # 总体统计信息
    main_logger.info(f"\n[Overall Statistics]")
    main_logger.info(f"  Total samples:   {len(missing)}")
    main_logger.info(f"  Successful:      {len(successful)}")
    main_logger.info(f"  Failed:          {len(failed_samples)}")
    main_logger.info(f"  Wall time:       {total_duration:.2f}s ({total_duration/60:.2f} min)")
    
    # 性能指标（仅当有成功样本时）
    if successful:
        # 计算平均处理时间
        avg_duration = sum(r['total_duration'] for r in successful) / len(successful)
        # 计算总 token 消耗
        total_tokens = sum(r['add_tokens'] + r['update_tokens'] for r in successful)
        # 计算总 API 调用次数
        total_calls = sum(r['add_calls'] + r['update_calls'] for r in successful)
        
        main_logger.info(f"\n[Performance Metrics]")
        main_logger.info(f"  Avg per sample:  {avg_duration:.2f}s")
        # 加速比：串行总时间 / 实际并行时间
        main_logger.info(f"  Speedup:         {avg_duration * len(successful) / total_duration:.2f}x")
        main_logger.info(f"  Total API calls: {total_calls}")
        main_logger.info(f"  Total tokens:    {total_tokens:,}")
    
    # 输出失败样本列表（如果有）
    if failed_samples:
        main_logger.info(f"\n[Failed Samples]")
        for sample_id in failed_samples:
            main_logger.info(f"  - {sample_id}")
    
    # 输出存储路径信息
    main_logger.info(f"\n{'='*70}")
    main_logger.info(f"Pre-update:  {QDRANT_PRE_UPDATE_DIR}")
    main_logger.info(f"Post-update: {QDRANT_POST_UPDATE_DIR}")
    main_logger.info(f"Logs:        {RUN_LOG_DIR}")
    main_logger.info("=" * 70)


# ==================== 程序入口点 ====================
if __name__ == "__main__":
    # 设置多进程启动方法为 'spawn'
    # 'spawn': 启动全新的 Python 解释器进程
    # 相比 'fork' 方法更安全，避免了共享状态导致的问题
    # 特别是在使用 CUDA 时，必须使用 'spawn' 避免 CUDA 上下文问题
    mp.set_start_method('spawn', force=True)
    
    # 执行主函数
    main()