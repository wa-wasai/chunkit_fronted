from LLMmodel import LLM_model
from faiss_store import FAISSVectorStore  # 引入FAISS向量存储
from retrieve_model import retrieve_relevant_chunks
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import torch

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
    


    def call_RAG_stream(self, query):
        """流式RAG调用
        
        Args:
            query: 用户查询
        """
        chunks = retrieve_relevant_chunks(
            user_query=query,
            vector_store=self.vector_store,
            embedding_model=self.model,
            cross_encoder1=self.cross_encoder
        )
        
        for delta in self.llm.call_llm_stream(query, chunks):
            yield delta

    def call_RAG(self, query):
        """标准RAG调用
        
        Args:
            query: 用户查询
        """
        chunks = retrieve_relevant_chunks(
            user_query=query,
            vector_store=self.vector_store,
            embedding_model=self.model,
            cross_encoder1=self.cross_encoder
        )
        
        return self.llm.call_llm(query, chunks)



