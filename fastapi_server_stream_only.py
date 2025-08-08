from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Generator
import json
from RAGlibrary import RAG
from Intent_answer import InteractiveAgent  
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
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

# 全局RAG实例
rag_instance = None
agent_instance = None  

class IntentQueryRequest(BaseModel):
    """意图识别查询请求模型"""
    query: str
    stream: bool = True 

class QueryRequest(BaseModel):
    """基础查询请求模型"""
    query: str

# 添加流式响应数据模型
class StreamChunk(BaseModel):
    """流式响应数据块模型"""
    delta: str
    finished: bool = False

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化系统"""
    global rag_instance, agent_instance
    try:
        rag_instance = RAG()
        agent_instance = InteractiveAgent()  # 初始化智能体
        print("RAG系统和智能体初始化成功")
    except Exception as e:
        print(f"系统初始化失败: {str(e)}")
        raise e

@app.post("/intent")
async def predict_intent(request: QueryRequest):
    """仅进行意图识别的API接口
    
    Args:
        request: 包含用户查询的请求对象
        
    Returns:
        dict: 包含意图、置信度和头像信息
    """
    if not agent_instance:
        raise HTTPException(status_code=500, detail="智能体系统未初始化")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")
    
    try:
        result = agent_instance.predict_intent_only(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"意图识别失败: {str(e)}")

@app.post("/query_with_intent")
async def query_with_intent_stream(request: IntentQueryRequest):
    """带意图识别的问答接口（流式）
    
    Args:
        request: 包含查询和流式选项的请求对象
        
    Returns:
        StreamingResponse: 流式响应，包含意图信息和回答内容
    """
    if not agent_instance:
        raise HTTPException(status_code=500, detail="智能体系统未初始化")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")
    
    def generate_intent_stream() -> Generator[str, None, None]:
        """生成带意图信息的流式响应
        
        Yields:
            str: SSE格式的流式数据，包含意图和内容信息
        """
        try:
            response_generator = agent_instance.process_question_with_intent(
                request.query, 
                stream_mode=request.stream
            )
            
            if request.stream:
                # 流式模式
                for chunk in response_generator:
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            else:
                # 非流式模式，直接返回完整结果
                result = response_generator
                yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
                
        except Exception as e:
            error_chunk = {
                "type": "error",
                "error": str(e),
                "finished": True
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_intent_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

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
                chunk_data = StreamChunk(delta=delta, finished=False)
                yield f"data: {chunk_data.json()}\n\n"
            
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