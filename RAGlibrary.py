from LLMmodel import  LLM_model
from faiss_store import FAISSVectorStore  # 引入FAISS向量存储
from retrieve_model import retrieve_relevant_chunks
collection_name = "document_embeddings"

class RAG():
    def __init__(self):
           self.tim = 0
           self.vector_store = FAISSVectorStore(index_path="./faiss_index", collection_name="document_embeddings")
           self.llm = LLM_model()
           self.llm.start_LLM()

    def call_RAG_stream(self,query):
        chunks = retrieve_relevant_chunks(user_query=query,vector_store=self.vector_store)
        for delta in self.llm.call_llm_stream(query,chunks):
            yield  delta

    def call_RAG(self,query):
        chunks = retrieve_relevant_chunks(user_query=query,vector_store=self.vector_store)
        return self.llm.call_llm(query,chunks)



