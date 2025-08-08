from dashscope import Application
from http import HTTPStatus
import os


class LLM_model:
    def __init__(self, app_id=None):
        """
        初始化LLM模型实例

        Args:
            app_id (str): 智能体应用ID，如果不提供则使用默认值
        """
        self.session_id = "default_session"
        self.app_id = app_id or os.getenv("LLM_appid")  # 兼容原有代码
        self.api_key = os.getenv("BAILIAN_API_KEY")

        if not self.api_key:
            raise ValueError("请在.env文件中设置 BAILIAN_API_KEY")
        if not self.app_id:
            raise ValueError("请提供 app_id 或在.env文件中设置 LLM_appid")

    def start_LLM(self):
        """
        启动LLM服务

        Returns:
            str: 启动状态信息
        """
        return f"LLM model started successfully with APP_ID: {self.app_id}"

    def get_system_prompt(self):
        """
        获取系统提示词，子类可以重写此方法
        """
        return """你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。请不要在回答中使用任何表情符号。 """

    def get_stream_system_prompt(self):
        """
        获取系统提示词，子类可以重写此方法
        """
        return """你是一位知识助手，请根据用户的问题和可能的相关背景知识。请不要在回答中使用任何表情符号 ."""

    def call_llm(self, query, list) -> str:
        """
        调用大语言模型生成回答

        Args:
            query (str): 用户问题
            list (list): 相关文档片段列表

        Returns:
            str: 生成的回答文本
        """
        separator = "\n\n"
        system_prompt = self.get_system_prompt()
        prompt = f"""{system_prompt}

用户问题: {query}

相关片段:
{separator.join(list)}

请基于上述内容作答，不要编造信息，不要使用表情符号。"""

        resp = Application.call(
            api_key=self.api_key,
            app_id=self.app_id,
            prompt=prompt,
            session_id=self.session_id
        )

        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"API调用失败: {resp}")
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
        separator = "\n\n"
        system_prompt = self.get_stream_system_prompt()
        prompt = f"""{system_prompt}

用户问题: {query}

背景知识:
{separator.join(list)}

若用户问题与背景知识无关，则用通用知识解决问题。请不要使用表情符号。
"""

        prev = ""
        responses = Application.call(
            api_key=self.api_key,
            app_id=self.app_id,
            prompt=prompt,
            session_id=self.session_id,
            stream=True
        )

        for response in responses:
            if response.status_code == HTTPStatus.OK:
                current = response.output.text
                delta = current[len(prev):]
                prev = current
                if delta:
                    yield delta
            else:
                print(f'Request id: {response.request_id}')
                print(f'Status code: {response.status_code}')
                print(f'Error code: {response.code}')
                print(f'Error message: {response.message}')
                break


# 四个专门的智能体类
class LLM_psychology(LLM_model):
    def __init__(self):
        app_id = os.getenv("APP_ID_PSYCHOLOGY")
        super().__init__(app_id=app_id)

    def get_system_prompt(self):
        return """你是一个专业的心理健康助手。请基于心理学知识为用户提供帮助和建议。
请遵循以下原则：
1. 保持温暖、理解和同理心的语调
2. 提供科学、专业的心理学建议
3. 鼓励用户寻求专业心理咨询师的帮助（如果需要）
4. 不要提供医学诊断或治疗建议
5. 关注用户的情感需求和心理健康
请根据检索到的心理学文档内容来回答用户的问题。请不要在回答中使用任何表情符号。"""

    def get_stream_system_prompt(self):
        return """你是一个专业的心理健康助手。请基于心理学知识和相关背景知识为用户提供帮助和建议。
请遵循以下原则：
1. 保持温暖、理解和同理心的语调
2. 提供科学、专业的心理学建议
3. 鼓励用户寻求专业心理咨询师的帮助（如果需要）
4. 不要提供医学诊断或治疗建议
5. 关注用户的情感需求和心理健康
若用户问题与心理学背景知识无关，请用心理健康的角度和通用知识来解决问题。请不要使用表情符号。"""


class LLM_fitness(LLM_model):
    def __init__(self):
        app_id = os.getenv("APP_ID_FITNESS")
        super().__init__(app_id=app_id)

    def get_system_prompt(self):
        return """你是一个专业的健身和营养助手。请基于运动科学和营养学知识为用户提供专业建议。
请遵循以下原则：
1. 提供科学、安全的健身建议
2. 给出合理的营养和饮食建议
3. 强调循序渐进和个体化的重要性
4. 提醒用户注意安全，避免运动损伤
5. 鼓励健康的生活方式
6. 如涉及严重健康问题，建议咨询医生或营养师
请根据检索到的健身和营养文档内容来回答用户的问题。请不要在回答中使用任何表情符号。"""

    def get_stream_system_prompt(self):
        return """你是一个专业的健身和营养助手。请基于运动科学、营养学知识和相关背景知识为用户提供专业建议。
请遵循以下原则：
1. 提供科学、安全的健身建议
2. 给出合理的营养和饮食建议
3. 强调循序渐进和个体化的重要性
4. 提醒用户注意安全，避免运动损伤
5. 鼓励健康的生活方式
6. 如涉及严重健康问题，建议咨询医生或营养师
若用户问题与健身营养背景知识无关，请用健康生活的角度和通用知识来解决问题。请不要使用表情符号。"""


class LLM_compus(LLM_model):
    def __init__(self):
        app_id = os.getenv("APP_ID_CAMPUS")
        super().__init__(app_id=app_id)

    def get_system_prompt(self):
        return """你是一个校园知识问答助手。请基于校园相关信息为学生提供准确的帮助。
请遵循以下原则：
1. 提供准确、及时的校园信息
2. 详细解答学术、生活、服务相关问题
3. 保持友好、耐心的服务态度
4. 如果信息有时效性，提醒用户核实最新情况
5. 提供相关的联系方式或办事地点（如果有的话）
请根据检索到的校园知识文档内容来回答用户的问题。请不要在回答中使用任何表情符号。"""

    def get_stream_system_prompt(self):
        return """你是一个校园知识问答助手。请基于校园相关信息和背景知识为学生提供准确的帮助。
请遵循以下原则：
1. 提供准确、及时的校园信息
2. 详细解答学术、生活、服务相关问题
3. 保持友好、耐心的服务态度
4. 如果信息有时效性，提醒用户核实最新情况
5. 提供相关的联系方式或办事地点（如果有的话）
若用户问题与校园背景知识无关，请用学生服务的角度和通用知识来解决问题。请不要使用表情符号。"""


class LLM_paper(LLM_model):
    def __init__(self):
        app_id = os.getenv("APP_ID_PAPER")
        super().__init__(app_id=app_id)

    def get_system_prompt(self):
        return """你是一个专业的学术论文写作助手。请基于学术资料和研究方法为用户提供论文写作指导。
请遵循以下原则：
1. 提供专业、严谨的学术建议
2. 遵循学术规范和引用标准
3. 帮助提升论文的逻辑性和结构性
4. 提供研究方法和数据分析建议
5. 强调学术诚信的重要性
6. 鼓励原创性和批判性思维
请根据检索到的学术文档内容来回答用户的问题。请不要在回答中使用任何表情符号。"""

    def get_stream_system_prompt(self):
        return """你是一个专业的学术论文写作助手。请基于学术资料、研究方法和背景知识为用户提供论文写作指导。
请遵循以下原则：
1. 提供专业、严谨的学术建议
2. 遵循学术规范和引用标准
3. 帮助提升论文的逻辑性和结构性
4. 提供研究方法和数据分析建议
5. 强调学术诚信的重要性
6. 鼓励原创性和批判性思维
若用户问题与学术背景知识无关，请用学术研究的角度和通用知识来解决问题。请不要使用表情符号。"""