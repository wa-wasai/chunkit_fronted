from dashscope import Application
from http import HTTPStatus
import os

APP_ID = os.getenv("LLM_appid")

class LLM_model:
    def __init__(self):
        """
        初始化LLM模型实例
        """
        self.session_id = "default_session"

    def start_LLM(self):
        """
        启动LLM服务
        
        Returns:
            str: 启动状态信息
        """
        return "LLM model started successfully"

    def call_llm(self, query, list) -> str:
        """
        调用大语言模型生成回答
        
        Args:
            query (str): 用户问题
            list (list): 相关文档片段列表
            
        Returns:
            str: 生成的回答文本
        """
        # 将换行符提取到变量中，避免f-string中的反斜杠问题
        separator = "\n\n"
        prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。请不要在回答中使用任何表情符号。

用户问题: {query}

相关片段:
{separator.join(list)}

请基于上述内容作答，不要编造信息，不要使用表情符号。"""
        resp = Application.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            app_id=APP_ID,
            prompt=prompt,
            session_id=self.session_id
        )
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(resp)
        return resp.output.text

    def call_llm_stream(self, query, list):
        """
        流式生成回答，只返回增量字符
        
        Args:
            query (str): 用户问题
            list (list): 相关文档片段列表
            
        Yields:
            str: 生成的文本增量
        """
        # 将换行符提取到变量中，避免f-string中的反斜杠问题
        separator = "\n\n"
        prompt = f"""你是一位知识助手，请根据用户的问题和可能的相关背景知识。请不要在回答中使用任何表情符号。

用户问题: {query}

背景知识:
{separator.join(list)}

若用户问题与背景知识无关，则用通用知识解决问题。请不要使用表情符号。
"""

        prev = ""  # 记录上一次完整内容，用于计算增量
        responses = Application.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            app_id=APP_ID,
            prompt=prompt,
            session_id=self.session_id,
            stream=True
        )
        
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                current = response.output.text
                # 计算增量：当前完整内容减去之前的内容
                delta = current[len(prev):]
                prev = current
                if delta:  # 只有当有新内容时才yield
                    yield delta
            else:
                print(f'Request id: {response.request_id}')
                print(f'Status code: {response.status_code}')
                print(f'Error code: {response.code}')
                print(f'Error message: {response.message}')
                break