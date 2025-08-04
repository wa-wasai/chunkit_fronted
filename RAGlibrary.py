from LLMmodel import LLM_model
from faiss_store import FAISSVectorStore  # 引入FAISS向量存储
from retrieve_model import retrieve_relevant_chunks
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import torch
from functools import lru_cache
import hashlib

collection_name = "document_embeddings"
local_model_path = "./cross-encoder-model"
class RAG():
    def __init__(self):
        self.tim = 0
        self.vector_store = FAISSVectorStore(index_path="./faiss_index", collection_name="document_embeddings")
        self.llm = LLM_model()
        self.llm.start_LLM()
        
        # 优化：检查CUDA可用性
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 优化：延迟加载模型以减少初始化时间
        self._cross_encoder = None
        self._embedding_model = None
        
        # 缓存机制
        self.query_cache = {}
        self.max_cache_size = 100
    
    @property
    def cross_encoder(self):
        """延迟加载交叉编码器"""
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(local_model_path)
        return self._cross_encoder
    
    @property
    def model(self):
        """延迟加载嵌入模型"""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(
                "./Qwen3-Embedding-0.6B",
                tokenizer_kwargs={"padding_side": "left"},
                trust_remote_code=True,
                device=self.device
            )
        return self._embedding_model
    
    def _get_query_hash(self, query):
        """生成查询的哈希值用于缓存"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()
    
    def _get_cached_chunks(self, query):
        """从缓存获取检索结果"""
        query_hash = self._get_query_hash(query)
        return self.query_cache.get(query_hash)
    
    def _cache_chunks(self, query, chunks):
        """缓存检索结果"""
        if len(self.query_cache) >= self.max_cache_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        query_hash = self._get_query_hash(query)
        self.query_cache[query_hash] = chunks

    def call_RAG_stream(self, query):
        """流式RAG调用，支持缓存优化"""
        # 尝试从缓存获取
        chunks = self._get_cached_chunks(query)
        if chunks is None:
            chunks = retrieve_relevant_chunks(
                user_query=query,
                vector_store=self.vector_store,
                embedding_model=self.model,
                cross_encoder1=self.cross_encoder
            )
            # 缓存结果
            self._cache_chunks(query, chunks)
        
        for delta in self.llm.call_llm_stream(query, chunks):
            yield delta

    def call_RAG(self, query):
        """标准RAG调用，支持缓存优化"""
        # 尝试从缓存获取
        chunks = self._get_cached_chunks(query)
        if chunks is None:
            chunks = retrieve_relevant_chunks(
                user_query=query,
                vector_store=self.vector_store,
                embedding_model=self.model,
                cross_encoder1=self.cross_encoder
            )
            # 缓存结果
            self._cache_chunks(query, chunks)
        
        return self.llm.call_llm(query, chunks)
    
    def clear_cache(self):
        """清空缓存"""
        self.query_cache.clear()
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        return {
            "cache_size": len(self.query_cache),
            "max_cache_size": self.max_cache_size,
            "device": self.device
        }



