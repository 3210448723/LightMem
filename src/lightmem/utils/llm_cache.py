"""
LLM 请求缓存模块

提供装饰器和缓存管理类，用于缓存大模型请求的输入和输出。
- 支持基于参数哈希的缓存命中检测
- 报错的请求不存储
- 支持文件持久化和内存缓存两种模式
"""

import hashlib
import json
import os
import pickle
import threading
import functools
from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _make_hashable(obj: Any) -> Any:
    """
    将对象转换为可哈希的形式，用于生成缓存键。
    
    Args:
        obj: 任意对象
        
    Returns:
        可哈希的对象表示
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return tuple(_make_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, set):
        return frozenset(_make_hashable(item) for item in obj)
    else:
        # 对于其他对象，尝试转为字符串
        return str(obj)


def _compute_cache_key(*args, **kwargs) -> str:
    """
    根据输入参数计算缓存键（MD5 哈希）。
    
    Args:
        *args: 位置参数
        **kwargs: 关键字参数
        
    Returns:
        str: 缓存键（MD5 哈希值）
    """
    hashable_args = _make_hashable(args)
    hashable_kwargs = _make_hashable(kwargs)
    
    combined = (hashable_args, hashable_kwargs)
    
    # 使用 pickle 序列化以确保一致性
    try:
        serialized = pickle.dumps(combined)
    except Exception:
        # 如果 pickle 失败，使用 JSON 序列化
        serialized = json.dumps(combined, sort_keys=True, default=str).encode('utf-8')
    
    return hashlib.md5(serialized).hexdigest()


class LLMCache:
    """
    LLM 请求缓存管理类。
    
    支持两种存储模式：
    - memory: 仅内存缓存，进程结束后丢失
    - file: 文件持久化缓存，可跨进程复用
    
    Attributes:
        cache_dir: 缓存文件目录（file 模式）
        storage_mode: 存储模式 ('memory' 或 'file')
        enabled: 是否启用缓存
    """
    
    _instance: Optional['LLMCache'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        storage_mode: str = "file",
        enabled: bool = True
    ):
        """
        初始化缓存管理器。
        
        Args:
            cache_dir: 缓存目录路径，默认为 ~/.lightmem/llm_cache
            storage_mode: 存储模式 ('memory' 或 'file')
            enabled: 是否启用缓存
        """
        if getattr(self, '_initialized', False):
            return
            
        self._memory_cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        self._enabled = enabled
        self._storage_mode = storage_mode
        
        if cache_dir is None:
            self._cache_dir = Path.home() / ".lightmem" / "llm_cache"
        else:
            self._cache_dir = Path(cache_dir)
        
        if self._storage_mode == "file":
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._stats = {
            "hits": 0,
            "misses": 0,
            "errors_skipped": 0
        }
        
        self._initialized = True
        logger.info(f"LLMCache initialized: mode={storage_mode}, dir={self._cache_dir}, enabled={enabled}")
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
        logger.info(f"LLMCache enabled set to: {value}")
    
    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        return self._cache_dir / f"{key}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """
        从缓存中获取数据。
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的数据，如果不存在则返回 None
        """
        if not self._enabled:
            return None
            
        with self._cache_lock:
            # 优先检查内存缓存
            if key in self._memory_cache:
                self._stats["hits"] += 1
                logger.debug(f"Cache hit (memory): {key[:16]}...")
                return self._memory_cache[key]
            
            # 文件模式下检查文件缓存
            if self._storage_mode == "file":
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    try:
                        with open(cache_path, 'rb') as f:
                            data = pickle.load(f)
                        # 加载到内存缓存
                        self._memory_cache[key] = data
                        self._stats["hits"] += 1
                        logger.debug(f"Cache hit (file): {key[:16]}...")
                        return data
                    except Exception as e:
                        logger.warning(f"Failed to load cache file {cache_path}: {e}")
            
            self._stats["misses"] += 1
            logger.debug(f"Cache miss: {key[:16]}...")
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        将数据存入缓存。
        
        Args:
            key: 缓存键
            value: 要缓存的数据
        """
        if not self._enabled:
            return
            
        with self._cache_lock:
            # 存入内存缓存
            self._memory_cache[key] = value
            
            # 文件模式下同时写入文件
            if self._storage_mode == "file":
                cache_path = self._get_cache_path(key)
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(value, f)
                    logger.debug(f"Cache saved to file: {key[:16]}...")
                except Exception as e:
                    logger.warning(f"Failed to save cache file {cache_path}: {e}")
    
    def delete(self, key: str) -> bool:
        """
        删除缓存条目。
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否成功删除
        """
        with self._cache_lock:
            deleted = False
            
            if key in self._memory_cache:
                del self._memory_cache[key]
                deleted = True
            
            if self._storage_mode == "file":
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                        deleted = True
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {cache_path}: {e}")
            
            return deleted
    
    def clear(self) -> None:
        """清空所有缓存"""
        with self._cache_lock:
            self._memory_cache.clear()
            
            if self._storage_mode == "file" and self._cache_dir.exists():
                for cache_file in self._cache_dir.glob("*.pkl"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {cache_file}: {e}")
            
            self._stats = {"hits": 0, "misses": 0, "errors_skipped": 0}
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        with self._cache_lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0
            return {
                **self._stats,
                "total_requests": total,
                "hit_rate": hit_rate,
                "memory_cache_size": len(self._memory_cache)
            }
    
    def record_error_skipped(self) -> None:
        """记录因错误而跳过缓存的次数"""
        with self._cache_lock:
            self._stats["errors_skipped"] += 1


# 全局缓存实例
_global_cache: Optional[LLMCache] = None


def get_cache(
    cache_dir: Optional[str] = None,
    storage_mode: str = "file",
    enabled: bool = True
) -> LLMCache:
    """
    获取全局缓存实例。
    
    Args:
        cache_dir: 缓存目录
        storage_mode: 存储模式
        enabled: 是否启用
        
    Returns:
        LLMCache: 缓存实例
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = LLMCache(
            cache_dir=cache_dir,
            storage_mode=storage_mode,
            enabled=enabled
        )
    return _global_cache


def llm_cache(
    cache_instance: Optional[LLMCache] = None,
    key_prefix: str = "",
    exclude_params: Optional[list] = None
) -> Callable:
    """
    LLM 请求缓存装饰器。
    
    用于装饰大模型调用方法，自动缓存输入输出。
    - 参数相同时命中缓存，无需重复请求
    - 报错的请求不存储缓存
    
    Args:
        cache_instance: 自定义缓存实例，为 None 时使用全局缓存
        key_prefix: 缓存键前缀，用于区分不同方法的缓存
        exclude_params: 不参与缓存键计算的参数名列表
        
    Returns:
        装饰器函数
        
    Example:
        @llm_cache(key_prefix="openai_chat")
        def generate_response(self, messages, response_format=None):
            ...
    """
    exclude_params = exclude_params or []
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache = cache_instance or get_cache()
            
            if not cache.enabled:
                return func(*args, **kwargs)
            
            # 过滤掉不参与缓存键计算的参数
            filtered_kwargs = {
                k: v for k, v in kwargs.items() 
                if k not in exclude_params
            }
            
            # 过滤 self 参数（第一个参数如果是实例方法）
            cache_args = args
            if len(args) > 0 and hasattr(args[0], '__class__'):
                # 检查是否为实例方法调用
                first_arg = args[0]
                if hasattr(first_arg, func.__name__):
                    # 是实例方法，排除 self，但包含 self 的关键配置信息
                    cache_args = args[1:]
                    # 将实例的关键配置信息加入缓存键
                    if hasattr(first_arg, 'config'):
                        config = first_arg.config
                        config_info = {
                            'model': getattr(config, 'model', None),
                            'temperature': getattr(config, 'temperature', None),
                            'max_tokens': getattr(config, 'max_tokens', None),
                            'top_p': getattr(config, 'top_p', None),
                        }
                        filtered_kwargs['_config_info'] = config_info
            
            # 生成缓存键
            cache_key = f"{key_prefix}_{_compute_cache_key(*cache_args, **filtered_kwargs)}"
            
            # 尝试从缓存获取
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"[LLMCache] Cache hit for {func.__name__}")
                return cached_result
            
            # 执行原函数
            try:
                result = func(*args, **kwargs)
                # 成功执行后存入缓存
                # 判断，如果大模型输出结果为空就不保存
                if result and len(result.choices[0].message.content)>0:
                    cache.set(cache_key, result)
                    logger.debug(f"[LLMCache] Cached result for {func.__name__}")
                else:
                    logger.debug(f"[LLMCache] Result empty, not caching for {func.__name__}")    
                return result
            except Exception as e:
                # 报错不缓存
                cache.record_error_skipped()
                logger.debug(f"[LLMCache] Error occurred, not caching: {e}")
                raise
        
        return wrapper
    return decorator


def configure_llm_cache(
    enabled: bool = True,
    cache_dir: Optional[str] = None,
    storage_mode: str = "file"
) -> LLMCache:
    """
    配置全局 LLM 缓存。
    
    Args:
        enabled: 是否启用缓存
        cache_dir: 缓存目录
        storage_mode: 存储模式 ('memory' 或 'file')
        
    Returns:
        LLMCache: 配置后的缓存实例
    """
    global _global_cache
    _global_cache = LLMCache(
        cache_dir=cache_dir,
        storage_mode=storage_mode,
        enabled=enabled
    )
    return _global_cache
