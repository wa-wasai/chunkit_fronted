from LLMmodel import LLM_model
from LLMmodel import LLM_psychology
from LLMmodel import LLM_fitness
from LLMmodel import LLM_compus
from LLMmodel import LLM_paper
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
        # 子类可以重写这些属性（在调用super()._init_()之前设置）
        if not hasattr(self, 'index_path'):
            self.index_path = "./faiss_index"
        if not hasattr(self, 'collection'):
            self.collection = "document_embeddings"

        # ✅ 修复：使用实际的 self.index_path 和 self.collection
        print(f"🔧 初始化向量存储: index_path={self.index_path}, collection={self.collection}")
        self.vector_store = FAISSVectorStore(
            index_path=self.index_path,  # ✅ 使用子类设置的路径
            collection_name=self.collection  # ✅ 使用子类设置的集合名
        )

        # 子类可以重写LLM属性
        if not hasattr(self, 'llm'):
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

#====================子类助手====================
#心理助手
class RAG_psychology(RAG):
    def __init__(self):
        self.index_path = "./faiss_index/psychology"  # ✅ 正确路径
        self.collection = "psychology_docs"           # ✅ 正确集合名
        self.llm = LLM_psychology()
        super().__init__()

#健身饮食助手
class RAG_fitness(RAG):
    def __init__(self):
        self.index_path = "./faiss_index/fitness"     # ✅ 正确路径
        self.collection = "fitness_docs"              # ✅ 正确集合名
        self.llm = LLM_fitness()
        super().__init__()

#校园知识问答
class RAG_compus(RAG):
    def __init__(self):
        self.index_path = "./faiss_index/campus"      # ✅ 正确路径
        self.collection = "campus_docs"               # ✅ 正确集合名
        self.llm = LLM_compus()
        super().__init__()

#论文助手
class RAG_paper(RAG):
    def __init__(self):
        self.index_path = "./faiss_index/paper"       # ✅ 正确路径
        self.collection = "paper_docs"                # ✅ 正确集合名
        self.llm = LLM_paper()
        super().__init__()