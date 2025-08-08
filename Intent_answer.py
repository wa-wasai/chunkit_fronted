#=====================这是接受用户信息，获取回答的主函数===================
import os
from dotenv import load_dotenv
from Intent_Recognition.code.intent_classifier import IntentClassifier
from RAGlibrary import RAG_psychology, RAG_fitness, RAG_compus, RAG_paper
# 加载 .env 文件
load_dotenv("Agent.env")

# 验证环境变量是否设置
required_env_vars = [
    "BAILIAN_API_KEY",
    "APP_ID_PSYCHOLOGY",
    "APP_ID_CAMPUS",
    "APP_ID_FITNESS",
    "APP_ID_PAPER"
]

missing_vars = []
for var in required_env_vars:
    if not os.getenv(var):
        missing_vars.append(var)

if missing_vars:
    print(f"请在.env文件中设置以下环境变量: {', '.join(missing_vars)}")
    exit(1)

print("所有环境变量配置验证成功")
print(f"使用的智能体应用:")
print(f"   - 心理助手: {os.getenv('APP_ID_PSYCHOLOGY')}")
print(f"   - 健身助手: {os.getenv('APP_ID_FITNESS')}")
print(f"   - 校园助手: {os.getenv('APP_ID_CAMPUS')}")
print(f"   - 论文助手: {os.getenv('APP_ID_PAPER')}")
print()




class InteractiveAgent:
    def __init__(self):
        try:
            # 初始化意图分类器
            self.classifier = IntentClassifier()
            print("意图分类器初始化成功")

            # 初始化 RAG 智能体（延迟初始化以提高启动速度）
            self.rag_agents = {}
            self.agent_classes = {
                "心理助手": RAG_psychology,
                "健身饮食助手": RAG_fitness,
                "校园知识问答": RAG_compus,
                "论文助手": RAG_paper
            }
            
            # 意图到头像的映射关系
            self.intent_avatar_mapping = {
                "心理助手": "007-gin tonic.svg",
                "健身饮食助手": "014-mojito.svg", 
                "校园知识问答": "042-milkshake.svg",
                "论文助手": "044-whiskey sour.svg",
                "其他": "050-lemon juice.svg"
            }
            
            print("RAG 智能体类加载成功")

        except Exception as e:
            print(f"初始化失败: {e}")
            raise

    def get_rag_agent(self, intent):
        """延迟初始化RAG智能体"""
        if intent not in self.rag_agents:
            if intent in self.agent_classes:
                print(f"🔧 正在初始化 {intent} RAG智能体...")
                try:
                    self.rag_agents[intent] = self.agent_classes[intent]()
                    print(f"{intent} RAG智能体初始化成功")
                except Exception as e:
                    print(f"{intent} RAG智能体初始化失败: {e}")
                    return None
            else:
                return None

        return self.rag_agents.get(intent)

    def check_rag_status(self, intent, rag_agent):
        """检查RAG知识库状态"""
        try:
            doc_count = rag_agent.vector_store.count()
            if doc_count == 0:
                print(f"{intent} 知识库中暂无文档")
                return False
            else:
                print(f"{intent} 知识库包含 {doc_count} 个文档片段")
                return True
        except Exception as e:
            print(f"检查 {intent} 知识库状态失败: {e}")
            return False

    def chat(self):
        print("=== 欢迎使用智能助手系统 ===")
        print("本系统使用本地RAG检索增强 + 远程智能体架构")
        print("支持交叉编码器精确检索和流式回答")
        print("输入你的问题（输入 'exit' 退出，'stream' 切换流式模式）：\n")

        stream_mode = False

        while True:
            user_input = input("你：")

            if user_input.lower() in ["exit", "quit"]:
                print("再见！")
                break

            if user_input.lower() == "stream":
                stream_mode = not stream_mode
                print(f"流式模式: {'开启' if stream_mode else '关闭'}")
                continue

            # 1. 预测意图
            try:
                result = self.classifier.predict_intent(user_input)
                intent = result["best_intent"]
                confidence = result["confidence"]
                print(f" 识别意图：{intent} (置信度 {confidence:.2f})")
            except Exception as e:
                print(f" 意图识别失败: {e}\n")
                continue

            # 2. 获取对应的RAG智能体
            rag_agent = self.get_rag_agent(intent)
            if rag_agent is None:
                print(f" 暂不支持该意图类型: {intent}\n")
                continue

            # 显示使用的APP ID
            print(f"使用智能体APP ID: {rag_agent.llm.app_id}")

            # 3. 检查知识库状态
            has_docs = self.check_rag_status(intent, rag_agent)
            if not has_docs:
                print(" 知识库为空，智能体将基于通用知识回答")

            # 4. 调用 RAG 智能体
            try:
                if stream_mode:
                    print(f" {intent} 正在思考")
                    print(" 流式回答：", end="", flush=True)

                    full_response = ""
                    for delta in rag_agent.call_RAG_stream(user_input):
                        print(delta, end="", flush=True)
                        full_response += delta
                    print("\n")  # 换行

                else:
                    print(f" {intent} 正在检索相关信息...")
                    answer = rag_agent.call_RAG(user_input)
                    print(f" {intent} 回答：{answer}\n")

            except Exception as e:
                print(f"调用 {intent} RAG智能体失败：{e}\n")

    def predict_intent_only(self, user_input):
        """仅进行意图识别，返回意图和对应头像
        
        Args:
            user_input (str): 用户输入的问题
            
        Returns:
            dict: 包含意图、置信度和头像信息的字典
        """
        try:
            # 进行意图识别
            result = self.classifier.predict_intent(user_input)
            intent = result["best_intent"]
            confidence = result["confidence"]
            
            # 获取对应的头像
            avatar = self.intent_avatar_mapping.get(intent, self.intent_avatar_mapping["其他"])
            
            return {
                "success": True,
                "intent": intent,
                "confidence": confidence,
                "avatar": avatar,
                "message": f"识别意图：{intent} (置信度 {confidence:.2f})"
            }
            
        except Exception as e:
            return {
                "success": False,
                "intent": "其他",
                "confidence": 0.0,
                "avatar": self.intent_avatar_mapping["其他"],
                "error": str(e),
                "message": "意图识别失败"
            }
    
    def process_question_with_intent(self, user_input, stream_mode=False):
        """处理用户问题并返回完整响应（包含意图信息）
        
        Args:
            user_input (str): 用户输入的问题
            stream_mode (bool): 是否使用流式模式
            
        Returns:
            dict: 包含意图信息和回答的完整响应
        """
        # 1. 预测意图
        intent_result = self.predict_intent_only(user_input)
        
        if not intent_result["success"]:
            return intent_result
        
        intent = intent_result["intent"]
        
        # 2. 获取对应的RAG智能体
        rag_agent = self.get_rag_agent(intent)
        if rag_agent is None:
            return {
                "success": False,
                "intent": intent,
                "avatar": intent_result["avatar"],
                "confidence": intent_result["confidence"],
                "error": f"暂不支持该意图类型: {intent}",
                "message": f"暂不支持该意图类型: {intent}"
            }
        
        # 3. 检查知识库状态
        has_docs = self.check_rag_status(intent, rag_agent)
        
        # 4. 调用RAG智能体获取回答
        try:
            if stream_mode:
                # 流式模式返回生成器
                def generate_response():
                    yield {
                        "type": "intent",
                        "intent": intent,
                        "avatar": intent_result["avatar"],
                        "confidence": intent_result["confidence"]
                    }
                    
                    for delta in rag_agent.call_RAG_stream(user_input):
                        yield {
                            "type": "content",
                            "delta": delta
                        }
                    
                    yield {
                        "type": "finished",
                        "finished": True
                    }
                
                return generate_response()
            else:
                # 非流式模式
                answer = rag_agent.call_RAG(user_input)
                return {
                    "success": True,
                    "intent": intent,
                    "avatar": intent_result["avatar"],
                    "confidence": intent_result["confidence"],
                    "answer": answer,
                    "has_docs": has_docs,
                    "app_id": rag_agent.llm.app_id
                }
                
        except Exception as e:
            return {
                "success": False,
                "intent": intent,
                "avatar": intent_result["avatar"],
                "confidence": intent_result["confidence"],
                "error": str(e),
                "message": f"处理问题时发生错误: {str(e)}"
            }


if __name__ == "__main__":
    try:
        agent = InteractiveAgent()
        agent.chat() 
        
    except KeyboardInterrupt:
        print("\n 程序被用户中断，再见！")
    except Exception as e:
        print(f"程序运行失败: {e}")
