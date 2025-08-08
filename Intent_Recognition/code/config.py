# /config.py

# --- 模型相关配置 ---
# SentenceTransformer 模型路径 (这个路径 'code' 文件夹之外，所以我们保持它不变)
MODEL_PATH = r"D:\Recognition\Model\Qwen3-Embedding-0.6B"

# 训练好的随机森林分类器模型保存路径 (修改为你的项目路径下的绝对路径)
CLASSIFIER_MODEL_PATH = "D:/Recognition/code/saved_models/random_forest_classifier.joblib"

# --- 百炼 API 相关配置 ---
# API 服务地址
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

# API 密钥 (强烈建议从环境变量中读取，避免硬编码)
API_KEY = "sk-93817db303964020bbc79b017be4768b"

# 不同意图对应百炼应用的 AppID
INTENT_TO_APPID = {
    "校园知识问答": "d158402e162f4672b90e3dc3e2b56ef5",
    "健身饮食助手": "dcbc79836e4047fe90634ff94f49cdf0",
    "论文助手": "0a337f720e0641d5a0b6b52c0f820240",
    "心理助手": "b363c9fcb13f462a94b44747ce071191"
}

# 不同意图对应的 System Prompt
INTENT_TO_PROMPT = {
    "校园知识问答": "你是一个校园知识问答助手，请提供简明扼要的答复。",
    "健身饮食助手": "你是一个健身饮食助手，请根据用户的健康需求提供专业建议。",
    "论文助手": "你是一个论文写作助手，请帮助用户查找资料、润色语言或排版格式。",
    "心理助手": "你是一个心理陪伴助手，请温暖、真诚地倾听用户，并提供合适的建议或陪伴。"
}


# --- 业务逻辑相关配置 ---
# 意图识别的置信度阈值，高于此阈值则直接采纳，否则需要用户澄清
CONFIDENCE_THRESHOLD = 0.7

# 当置信度不足时，向用户展示的最可能的前 N 个选项
TOP_N_OPTIONS = 1
