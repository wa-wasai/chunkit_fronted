from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # 添加CORS导入
from pydantic import BaseModel
from typing import Optional, Generator
import json
from RAGlibrary import RAG
import uvicorn

# 创建FastAPI应用实例
app = FastAPI(
    title="RAG问答系统API - 流式版本",
    description="基于检索增强生成的智能问答系统（仅支持流式输出）",
    version="1.0.0"
)

# 添加CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

# 全局RAG实例
rag_instance = None

# 请求模型定义
class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str

class StreamChunk(BaseModel):
    """流式响应块模型"""
    delta: str
    finished: bool = False

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化RAG系统"""
    global rag_instance
    try:
        rag_instance = RAG()
        print("RAG系统初始化成功")
    except Exception as e:
        print(f"RAG系统初始化失败: {str(e)}")
        raise e

@app.get("/")
async def root():
    """根路径健康检查"""
    return {
        "message": "RAG问答系统API服务正在运行（仅支持流式输出）", 
        "status": "healthy",
        "supported_endpoints": ["/query", "/query/simple", "/health"]
    }

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "rag_initialized": rag_instance is not None,
        "output_mode": "stream_only"
    }

@app.post("/query")
async def query_rag_stream(request: QueryRequest):
    """流式问答接口（主要接口）"""
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG系统未初始化")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")
    
    def generate_stream() -> Generator[str, None, None]:
        """生成流式响应
        
        Yields:
            str: SSE格式的流式数据
        """
        try:
            for delta in rag_instance.call_RAG_stream(request.query):
                # 构造SSE格式的数据
                chunk_data = StreamChunk(delta=delta, finished=False)
                yield f"data: {chunk_data.json()}\n\n"
            
            # 发送结束标志
            end_chunk = StreamChunk(delta="", finished=True)
            yield f"data: {end_chunk.json()}\n\n"
            
        except Exception as e:
            error_chunk = {
                "error": str(e),
                "finished": True
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.post("/query/simple")
async def simple_query_stream(query: str):
    """简化的流式查询接口（直接传递字符串参数）"""
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG系统未初始化")
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")
    
    def generate_simple_stream() -> Generator[str, None, None]:
        """生成简化的流式响应
        
        Yields:
            str: 纯文本流式数据
        """
        try:
            for delta in rag_instance.call_RAG_stream(query):
                yield delta
        except Exception as e:
            yield f"\n\n[错误]: {str(e)}"
    
    return StreamingResponse(
        generate_simple_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(
        "fastapi_server_stream_only:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )