from openai import OpenAI
from typing import Optional, Literal
from sentence_transformers import SentenceTransformer
import numpy as np
from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig

class TextEmbedderHuggingface:
    """
    文本嵌入器（HuggingFace/SentenceTransformers 或远端 TEI）：
    - 若配置提供 huggingface_base_url，则通过兼容的 OpenAI Embeddings 接口请求远端向量；
    - 否则本地加载 SentenceTransformer 模型，按需推理并返回 numpy/list 向量；
    - 初始化阶段尽量补全缺省配置（model / embedding_dims / model_kwargs）。
    """
    def __init__(self, config: Optional[BaseTextEmbedderConfig] = None):
        self.config = config
        self.total_calls = 0
        self.total_tokens = 0
        if config.huggingface_base_url:  # type: ignore[union-attr]
            self.client = OpenAI(base_url=config.huggingface_base_url)  # type: ignore[union-attr]
            self.use_api = True
        else:
            self.config.model = config.model or "all-MiniLM-L6-v2"  # type: ignore[union-attr]
            self.model = SentenceTransformer(config.model, **(config.model_kwargs or {}))  # type: ignore[arg-type,union-attr]
            self.config.embedding_dims = (config.embedding_dims or self.model.get_sentence_embedding_dimension())  # type: ignore[union-attr]
            self.use_api = False

    @classmethod
    def from_config(cls, config):
        cls.validate_config(config)
        return cls(config)

    @staticmethod
    def validate_config(config):
        """
        Validate whether the provided config dictionary contains the required configuration items.
        Assume that the HuggingFace embedder requires at least 'model_name'.
        """
        required_keys = ['model_name']
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValueError(f"Missing required config keys for HuggingFace embedder: {missing}")

    def embed(self, text):
        """
        Get the embedding for the given text using Hugging Face.

        Args:
            text (str): The text to embed.
        Returns:
            list: The embedding vector.
        """
        self.total_calls += 1
        if self.config.huggingface_base_url:  # type: ignore[union-attr]
            response = self.client.embeddings.create(input=text, model="tei")
            self.total_tokens += getattr(response.usage, 'total_tokens', 0)
            return response.data[0].embedding
        else:
            result = self.model.encode(text, convert_to_numpy=True)
            if isinstance(result, np.ndarray):
                return result.tolist()
            else:
                return result
            
    def get_stats(self):
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
        }