from LLMmodel import  LLM_model
from chromadb import PersistentClient  # 引入持久化客户端
from retrieve_model import retrieve_relevant_chunks
collection_name = "document_embeddings"

class RAG():
    def __init__(self):
           self.tim = 0
           self.client = PersistentClient(path="./chroma_db")
           self.collection = self.client.get_collection(name="document_embeddings")
           self.llm = LLM_model()
           self.llm.start_LLM()

    def call_RAG_stream(self,query):
        chunks = retrieve_relevant_chunks(user_query=query,collection=self.collection)
        for delta in self.llm.call_llm_stream(query,chunks):
            yield  delta

    def call_RAG(self,query):
        chunks = retrieve_relevant_chunks(user_query=query,collection=self.collection)
        return self.llm.call_llm(query,chunks)



