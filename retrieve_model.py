from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import torch
import numpy as np
from typing import List, Tuple, Optional
from faiss_store import FAISSVectorStore  # 引入FAISS向量存储

local_model_path = "./cross-encoder-model"
collection_name = "document_embeddings"

def retrieve_relevant_chunks(user_query: str, vector_store: FAISSVectorStore, 
                           top_k: int = 15, final_k: int = 5, 
                           embedding_model: Optional[SentenceTransformer] = None, 
                           cross_encoder1: Optional[CrossEncoder] = None) -> List[str]:
    """使用FAISS检索相关文档片段的函数，优化了性能和错误处理"""
    if embedding_model is None or cross_encoder1 is None:
        raise ValueError("embedding_model和cross_encoder1参数不能为None")
    
    model = embedding_model
    cross_encoder = cross_encoder1
    
    try:
        # 优化：生成查询向量，使用更高效的编码方式
        with torch.no_grad():  # 减少内存使用
            query_embedding = model.encode(
                user_query, 
                prompt_name="query",
                convert_to_tensor=False,  # 直接返回numpy数组
                normalize_embeddings=True  # 标准化嵌入向量
            ).tolist()

        # 从FAISS向量存储检索
        results = vector_store.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, 50)  # 限制最大检索数量
        )
        
        if not results["documents"][0]:
            return []  # 无匹配结果

        documents = results["documents"][0]
        
        # 优化：批量处理交叉编码器预测
        if len(documents) <= final_k:
            # 如果文档数量已经小于等于final_k，直接返回
            return documents
        
        # 使用交叉编码器重新排序 - 批量处理
        pairs = [(user_query, chunk) for chunk in documents]
        
        with torch.no_grad():  # 减少内存使用
            scores = cross_encoder.predict(pairs, batch_size=32)  # 批量预测
        
        # 优化：使用numpy进行更快的排序
        scores_array = np.array(scores)
        top_indices = np.argsort(scores_array)[::-1][:final_k]  # 获取top_k索引
        
        return [documents[i] for i in top_indices]
        
    except Exception as e:
        print(f"检索过程中发生错误: {e}")
        return []

def batch_retrieve_relevant_chunks(queries: List[str], vector_store: FAISSVectorStore,
                                  top_k: int = 15, final_k: int = 5,
                                  embedding_model: Optional[SentenceTransformer] = None,
                                  cross_encoder1: Optional[CrossEncoder] = None) -> List[List[str]]:
    """批量检索多个查询的相关文档片段"""
    if embedding_model is None or cross_encoder1 is None:
        raise ValueError("embedding_model和cross_encoder1参数不能为None")
    
    results = []
    for query in queries:
        chunks = retrieve_relevant_chunks(
            query, vector_store, top_k, final_k, 
            embedding_model, cross_encoder1
        )
        results.append(chunks)
    
    return results

# 使用示例
if __name__ == "__main__":
    # 初始化模型
    model = SentenceTransformer(
        "./Qwen3-Embedding-0.6B",
        tokenizer_kwargs={"padding_side": "left"},
        trust_remote_code=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    cross_encoder = CrossEncoder(local_model_path)
    
    user_query = input("请输入你的问题: ")
    vector_store = FAISSVectorStore(index_path="./faiss_index", collection_name=collection_name)
    relevant_chunks = retrieve_relevant_chunks(
        user_query=user_query,
        vector_store=vector_store,
        embedding_model=model,
        cross_encoder1=cross_encoder
    )
    
    print(f"检索到 {len(relevant_chunks)} 个相关文档片段:")
    for i, chunk in enumerate(relevant_chunks, 1):
        print(f"\n片段 {i}:\n{chunk[:200]}...")
