from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import torch
from faiss_store import FAISSVectorStore  # 引入FAISS向量存储

local_model_path = "./cross-encoder-model"
collection_name = "document_embeddings"
cross_encoder = CrossEncoder(local_model_path)

def retrieve_relevant_chunks(user_query, vector_store, top_k=15, final_k=5):
    """使用FAISS检索相关文档片段的函数，修复了meta tensor问题"""
    try:
        # 修复meta tensor问题的加载方式
        model = SentenceTransformer(
            "./Qwen3-Embedding-0.6B",
            # 注释掉device_map避免meta tensor问题
            # model_kwargs={"device_map": "auto"},
            tokenizer_kwargs={"padding_side": "left"},
            trust_remote_code=True,  # Qwen模型需要信任远程代码
            device="cpu"  # 先在CPU上加载
        )
        
        # 如果有GPU，手动移动到GPU
        if torch.cuda.is_available():
            model = model.to("cuda")
            
    except Exception as e:
        raise RuntimeError(f"加载Qwen3-Embedding模型失败: {str(e)}")

    # 生成查询向量
    query_embedding = model.encode(user_query, prompt_name="query").tolist()

    # 从FAISS向量存储检索
    results = vector_store.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    if not results["documents"][0]:
        return []  # 无匹配结果

    # 使用交叉编码器重新排序
    pairs = [(user_query, chunk) for chunk in results["documents"][0]]
    scores = cross_encoder.predict(pairs)
    scored_chunks = list(zip(results["documents"][0], scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [
        item[0]
        for item in scored_chunks[:final_k]
    ]

# 使用示例
if __name__ == "__main__":
    user_query = input("请输入你的问题: ")
    vector_store = FAISSVectorStore(index_path="./faiss_index", collection_name=collection_name)
    relevant_chunks = retrieve_relevant_chunks(
        user_query=user_query,
        vector_store=vector_store
    )
