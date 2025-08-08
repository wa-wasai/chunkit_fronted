import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
import torch
from sentence_transformers import SentenceTransformer

class FAISSVectorStore:
    """FAISS向量存储类，替代ChromaDB"""
    
    def __init__(self, 
                 index_path: str = "./faiss_index", 
                 collection_name: str = "document_embeddings",
                 dimension: int = 1024,
                 reset: bool = False):
        """初始化FAISS向量存储
        
        Args:
            index_path: FAISS索引文件保存路径
            collection_name: 集合名称
            dimension: 向量维度，Qwen3-0.6B的固定维度为1024
            reset: 是否重置索引
        """
        self.index_path = index_path
        self.collection_name = collection_name
        self.dimension = dimension
        
        # 创建索引目录
        os.makedirs(index_path, exist_ok=True)
        
        # 索引文件路径
        self.index_file = os.path.join(index_path, f"{collection_name}.index")
        self.documents_file = os.path.join(index_path, f"{collection_name}.documents")
        self.ids_file = os.path.join(index_path, f"{collection_name}.ids")
        
        # 初始化或加载索引
        if reset or not os.path.exists(self.index_file):
            self._create_new_index()
        else:
            self._load_index()
            
        print(f"FAISS索引初始化完成，当前文档数量: {len(self.documents)}")
    
    def _create_new_index(self):
        """创建新的FAISS索引"""
        # 创建L2距离的索引
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.ids = []
        print(f"创建新的FAISS索引，维度: {self.dimension}")
    
    def _load_index(self):
        """加载现有的FAISS索引"""
        try:
            self.index = faiss.read_index(self.index_file)
            
            with open(self.documents_file, 'rb') as f:
                self.documents = pickle.load(f)
                
            with open(self.ids_file, 'rb') as f:
                self.ids = pickle.load(f)
                
            print(f"成功加载FAISS索引，包含{len(self.documents)}个文档")
        except Exception as e:
            print(f"加载索引失败: {str(e)}，将创建新索引")
            self._create_new_index()
    
    def save(self):
        """保存索引和相关数据"""
        try:
            faiss.write_index(self.index, self.index_file)
            
            with open(self.documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
                
            with open(self.ids_file, 'wb') as f:
                pickle.dump(self.ids, f)
                
            print(f"FAISS索引保存成功，文档数量: {len(self.documents)}")
        except Exception as e:
            raise Exception(f"保存FAISS索引失败: {str(e)}")
    
    def add(self, documents: List[str], embeddings: List[List[float]], ids: List[str]):
        """添加文档和向量到索引
        
        Args:
            documents: 文档内容列表
            embeddings: 向量列表
            ids: 文档ID列表
        """
        if not documents or not embeddings or not ids:
            raise ValueError("文档、向量和ID不能为空")
            
        if len(documents) != len(embeddings) or len(documents) != len(ids):
            raise ValueError("文档、向量和ID数量必须一致")
        
        # 转换为numpy数组
        embeddings_np = np.array(embeddings).astype('float32')
        
        # 添加到索引
        self.index.add(embeddings_np)
        
        # 保存文档和ID
        self.documents.extend(documents)
        self.ids.extend(ids)
        
        # 自动保存
        self.save()
    
    def query(self, query_embeddings: List[List[float]], n_results: int = 10) -> Dict[str, List]:
        """查询最相似的文档
        
        Args:
            query_embeddings: 查询向量列表
            n_results: 返回结果数量
            
        Returns:
            包含documents、distances和ids的字典
        """
        if not self.documents:
            return {"documents": [[]], "distances": [[]], "ids": [[]]}
        
        # 转换为numpy数组
        query_np = np.array(query_embeddings).astype('float32')
        
        # 执行查询
        distances, indices = self.index.search(query_np, min(n_results, len(self.documents)))
        
        # 构建结果
        results = {
            "documents": [],
            "distances": distances.tolist(),
            "ids": []
        }
        
        # 对每个查询添加结果
        for query_indices in indices:
            query_docs = []
            query_ids = []
            
            for idx in query_indices:
                if idx >= 0 and idx < len(self.documents):  # 确保索引有效
                    query_docs.append(self.documents[idx])
                    query_ids.append(self.ids[idx])
            
            results["documents"].append(query_docs)
            results["ids"].append(query_ids)
        
        return results
    
    def count(self) -> int:
        """返回索引中的文档数量"""
        return len(self.documents)
    
    def reset(self):
        """重置索引"""
        self._create_new_index()
        self.save()
        print("FAISS索引已重置")

        # faiss_store.py（如果没有 search 方法，就加上）
        class FAISSVectorStore:
            ...

            def search(self, query_vector, top_k=5):
                try:
                    distances, indices = self.index.search(np.array([query_vector]), top_k)
                    results = []
                    for idx in indices[0]:
                        if 0 <= idx < len(self.texts):
                            results.append(self.texts[idx])
                    return results
                except Exception as e:
                    print(f"❌ FAISS 搜索出错：{e}")
                    return []
