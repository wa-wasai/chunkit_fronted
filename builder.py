from textsplitters import RecursiveCharacterTextSplitter
import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer  # 需确保transformers>=4.51.0和sentence-transformers>=2.7.0
import os  # 新增：用于文件路径处理
import PyPDF2  # 处理PDF
from docx import Document  # 处理Word
from faiss_store import FAISSVectorStore  # 引入FAISS向量存储

def read_file(filename: str) -> str:
    """读取文件内容，支持多种编码格式"""
    # 保留原有txt处理逻辑，新增编码检测
    if filename.endswith('.txt'):
        try:
            # 首先尝试UTF-8编码
            with open(filename, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                # 如果UTF-8失败，尝试GBK编码
                with open(filename, 'r', encoding='gbk') as file:
                    return file.read()
            except UnicodeDecodeError:
                try:
                    # 最后尝试ANSI编码
                    with open(filename, 'r', encoding='ansi') as file:
                        return file.read()
                except Exception as e:
                    raise Exception(f"读取txt文件出错（编码问题）: {str(e)}")
        except Exception as e:
            raise Exception(f"读取txt文件出错: {str(e)}")

    # 新增：支持markdown文件
    elif filename.endswith('.md') or filename.endswith('.markdown'):
        try:
            # 首先尝试UTF-8编码
            with open(filename, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                # 如果UTF-8失败，尝试GBK编码
                with open(filename, 'r', encoding='gbk') as file:
                    return file.read()
            except Exception as e:
                raise Exception(f"读取md文件出错: {str(e)}")
        except Exception as e:
            raise Exception(f"读取md文件出错: {str(e)}")

    elif filename.endswith('.pdf'):
        try:
            text = ""
            with open(filename, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            raise Exception(f"读取pdf文件出错: {str(e)}")

    elif filename.endswith('.docx'):
        try:
            doc = Document(filename)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise Exception(f"读取docx文件出错: {str(e)}")

    else:
        raise ValueError(f"不支持的文件格式: {filename}")


def token_length_function(text: str) -> int:
    encoding = tiktoken.get_encoding('cl100k_base')
    return len(encoding.encode(text))


def split_document(filename: str) -> list[str]:
    content = read_file(filename)
    if not content.strip():
        raise ValueError("文件内容为空，无法进行分割")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # 每个块最大 300 token
        chunk_overlap=50,  # 块之间重叠 50 token
        length_function=token_length_function,
        separators=None,  # 使用默认分隔符
    )
    chunks = text_splitter.split_text(content)

    if not chunks:
        raise ValueError("文档分割后未得到任何片段，请检查文档内容")

    print(f"文档 {filename} 分割完成，共得到 {len(chunks)} 个片段")
    return chunks


def init_faiss_store(collection_name: str = "document_embeddings", reset: bool = False):
    """初始化 FAISS 向量存储，支持重置索引"""
    try:
        # 创建FAISS向量存储
        vector_store = FAISSVectorStore(
            index_path="./faiss_index",
            collection_name=collection_name,
            dimension=1024,  # Qwen3-0.6B的固定维度
            reset=reset
        )
        
        print(f"初始化FAISS向量存储成功，当前文档数量: {vector_store.count()}")
        return vector_store
    except Exception as e:
        raise Exception(f"初始化FAISS向量存储失败: {str(e)}")


def embed_and_store_chunks(chunks: list[str], vector_store):
    """将 chunks 向量化并存储到 FAISS，增加严格验证"""
    # 加载 Qwen3-Embedding 模型 - 修复设备映射问题
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SentenceTransformer(
        "./Qwen3-Embedding-0.6B",
        device=device,  # 直接指定设备而不是使用device_map
        tokenizer_kwargs={"padding_side": "left"}
    )
    print("模型加载成功")

    # 生成嵌入向量（文档不需要加指令）
    embeddings = model.encode(chunks)
    print(f"生成嵌入向量完成，共 {len(embeddings)} 个向量")

    # 准备批量添加数据
    documents = []
    embeddings_list = []
    ids = []
    
    # 收集所有数据
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        documents.append(chunk)
        embeddings_list.append(embedding.tolist())
        ids.append(f"chunk_{i}")
    
    # 批量添加到FAISS索引
    vector_store.add(
        documents=documents,
        embeddings=embeddings_list,
        ids=ids
    )
    
    print(f"成功存储 {len(chunks)} 个文档片段到 FAISS 向量存储")


# 新增：处理文件夹中所有文件的函数
def process_folder(folder_name: str, collection_name: str = "document_embeddings", reset: bool = False):
    """处理指定文件夹中的所有文件，将其分割并存储到数据库"""
    if not os.path.isdir(folder_name):
        raise NotADirectoryError(f"文件夹 {folder_name} 不存在或不是一个有效的文件夹")

    # 初始化FAISS向量存储
    vector_store = init_faiss_store(collection_name, reset)

    # 获取文件夹中所有文件
    all_files = [f for f in os.listdir(folder_name)
                 if os.path.isfile(os.path.join(folder_name, f))]

    if not all_files:
        print(f"文件夹 {folder_name} 中没有找到任何文件")
        return

    print(f"开始处理文件夹 {folder_name}，共发现 {len(all_files)} 个文件")

    # 加载嵌入模型（避免重复加载）- 修复设备映射问题
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SentenceTransformer(
        "./Qwen3-Embedding-0.6B",
        device=device,  # 直接指定设备而不是使用device_map
        tokenizer_kwargs={"padding_side": "left"}
    )
    print("嵌入模型加载成功，开始批量处理文件")

    total_chunks = 0

    # 处理每个文件
    for file_idx, filename in enumerate(all_files, 1):
        file_path = os.path.join(folder_name, filename)
        try:
            print(f"\n处理第 {file_idx}/{len(all_files)} 个文件: {filename}")
            # 分割文档
            chunks = split_document(file_path)
            total_chunks += len(chunks)

            # 生成嵌入向量
            embeddings = model.encode(chunks)

            # 准备批量添加数据
            documents = []
            embeddings_list = []
            ids = []
            
            # 收集所有数据
            for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                unique_id = f"file_{file_idx}_chunk_{chunk_idx}"
                documents.append(chunk)
                embeddings_list.append(embedding.tolist())
                ids.append(unique_id)
            
            # 批量添加到FAISS索引
            vector_store.add(
                documents=documents,
                embeddings=embeddings_list,
                ids=ids
            )

            print(f"文件 {filename} 处理完成，新增 {len(chunks)} 个片段")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}，将跳过该文件")
            continue

    print(f"\n所有文件处理完成，共处理 {len(all_files)} 个文件，生成 {total_chunks} 个片段")


def debug():
    try:
        # 分割文档
        chunks = split_document("debug.txt")
        # 初始化FAISS向量存储（测试时使用reset=True清除旧数据）
        vector_store = init_faiss_store(reset=True)
        # 嵌入并存储
        embed_and_store_chunks(chunks, vector_store)
    except Exception as e:
        print(f"调试过程出错: {str(e)}")


if __name__ == '__main__':
    # 可以通过修改此处来选择运行模式
    # debug()  # 原有调试模式
    str1 = input("请输入文件夹名:")
    # 新增：处理文件夹模式（示例）
    try:
        process_folder(
            folder_name=str1,  
            collection_name="document_embeddings",
            reset=True  # 首次运行建议设为True，清空旧数据
        )
    except Exception as e:
        print(f"处理文件夹时出错: {str(e)}")
