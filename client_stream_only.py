import requests
import json
import sseclient  # pip install sseclient-py

class RAGStreamClient:
    """RAG流式API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """初始化客户端
        
        Args:
            base_url: API服务器地址
        """
        self.base_url = base_url
    
    def health_check(self) -> dict:
        """健康检查
        
        Returns:
            dict: 服务状态信息
        """
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def query_stream(self, question: str):
        """流式查询（SSE格式）
        
        Args:
            question: 用户问题
            
        Yields:
            str: 回答片段
        """
        payload = {"query": question}
        
        response = requests.post(
            f"{self.base_url}/query",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )
        
        if response.status_code == 200:
            client = sseclient.SSEClient(response)
            for event in client.events():
                if event.data:
                    try:
                        data = json.loads(event.data)
                        if "error" in data:
                            raise Exception(data["error"])
                        if data["finished"]:
                            break
                        yield data["delta"]
                    except json.JSONDecodeError:
                        continue
        else:
            raise Exception(f"请求失败: {response.text}")
    
    def simple_query_stream(self, question: str):
        """简化流式查询（纯文本格式）
        
        Args:
            question: 用户问题
            
        Yields:
            str: 回答片段
        """
        response = requests.post(
            f"{self.base_url}/query/simple",
            params={"query": question},
            stream=True
        )
        
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                if chunk:
                    yield chunk
        else:
            raise Exception(f"请求失败: {response.text}")

# 使用示例
if __name__ == "__main__":
    client = RAGStreamClient()
    
    # 健康检查
    print("服务状态:", client.health_check())
    
    question = "请介绍一下这个系统的功能"
    print(f"\n问题: {question}")
    
    # SSE格式流式查询
    print("\nSSE流式回答:")
    for chunk in client.query_stream(question):
        print(chunk, end="")
    
    print("\n\n" + "="*50)
    
    # 简化格式流式查询
    print("简化流式回答:")
    for chunk in client.simple_query_stream(question):
        print(chunk, end="")
    print("\n")