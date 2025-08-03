from faiss_store import FAISSVectorStore
from retrieve_model import retrieve_relevant_chunks
import time

def interactive_search():
    print("FAISS交互式检索系统")
    print("====================")
    
    # 初始化FAISS向量存储
    print("正在加载FAISS索引...")
    start_time = time.time()
    vector_store = FAISSVectorStore(
        index_path="./faiss_index",
        collection_name="document_embeddings"
    )
    load_time = time.time() - start_time
    
    print(f"FAISS索引加载成功，耗时: {load_time:.2f}秒，当前文档数量: {vector_store.count()}")
    print("输入'exit'或'quit'退出程序\n")
    
    while True:
        query = input("请输入您的问题: ")
        
        if query.lower() in ['exit', 'quit']:
            print("感谢使用，再见！")
            break
        
        if not query.strip():
            continue
        
        # 计时查询性能
        start_time = time.time()
        
        # 使用retrieve_model.py中的函数进行检索
        chunks = retrieve_relevant_chunks(user_query=query, vector_store=vector_store, top_k=3)
        
        query_time = time.time() - start_time
        
        print(f"\n查询耗时: {query_time:.4f}秒，找到 {len(chunks)} 个相关文档片段:")
        for i, chunk in enumerate(chunks):
            print(f"\n--- 片段 {i+1} ---")
            print(chunk)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    try:
        interactive_search()
    except Exception as e:
        print(f"程序运行出错: {str(e)}")