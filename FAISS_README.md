# FAISS向量存储与检索系统

本项目使用FAISS（Facebook AI Similarity Search）替代原有的ChromaDB，实现高效的向量存储和相似度检索功能。

## 功能特点

- 高性能：FAISS针对大规模向量检索进行了优化，查询速度快
- 低内存占用：相比ChromaDB，FAISS内存占用更低
- 可扩展性：支持增量添加文档，无需重建整个索引
- 持久化存储：索引可保存到磁盘并从磁盘加载

## 主要文件说明

- `faiss_store.py`：FAISS向量存储的核心实现类
- `builder.py`：文档处理和向量化的主要逻辑
- `retrieve_model.py`：检索相关文档片段的功能实现
- `RAGlibrary.py`：结合检索增强生成（RAG）的功能
- `migrate_to_faiss.py`：从ChromaDB迁移数据到FAISS的工具

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 处理文档并创建FAISS索引

```python
from builder import process_folder

# 处理指定文件夹中的所有文档，reset=True表示重新创建索引
process_folder('文档文件夹路径', reset=True)
```

### 3. 检索相关文档

```python
from faiss_store import FAISSVectorStore
from retrieve_model import retrieve_relevant_chunks

# 初始化FAISS向量存储
vector_store = FAISSVectorStore(
    index_path="./faiss_index",
    collection_name="document_embeddings"
)

# 检索相关文档片段
chunks = retrieve_relevant_chunks(
    user_query="您的查询问题", 
    vector_store=vector_store, 
    top_k=3  # 返回前3个最相关的文档片段
)

# 打印检索结果
for i, chunk in enumerate(chunks):
    print(f"片段 {i+1}: {chunk}")
```

### 4. 交互式检索

运行交互式检索脚本：

```bash
python interactive_search.py
```

## 测试脚本

- `test_retrieval.py`：基本检索功能测试
- `test_faiss_comprehensive.py`：全面的检索性能测试
- `interactive_search.py`：交互式检索界面

## 从ChromaDB迁移到FAISS

如果您有现有的ChromaDB数据需要迁移到FAISS，可以使用迁移工具：

```bash
python migrate_to_faiss.py
```

## 性能对比

在测试中，FAISS相比ChromaDB有以下优势：

- 查询速度更快（平均查询时间约1.14秒）
- 内存占用更低
- 支持更大规模的向量集合