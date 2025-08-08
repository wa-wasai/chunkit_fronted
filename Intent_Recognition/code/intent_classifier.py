# intent_project/intent_classifier.py
import numpy as np

# 从项目内部模块导入
from Intent_Recognition.code import config
from Intent_Recognition.code.preprocessing import TextEncoder
from Intent_Recognition.code.model import IntentRecognitionModel

class IntentClassifier:
    """
    意图分类器主类。
    封装了数据加载、文本编码、模型训练/加载、意图预测和API调用等全部流程。
    """

    def __init__(self, train_if_not_exist=True):
        """
        初始化意图分类器。

        Args:
            train_if_not_exist (bool): 如果本地没有找到已训练的模型文件，是否自动进行训练。
        """
        self.text_encoder = TextEncoder(config.MODEL_PATH)
        self.model = IntentRecognitionModel()

        # 尝试加载已训练的模型
        model_loaded = self.model.load_model(config.CLASSIFIER_MODEL_PATH)

        # 如果模型加载失败且允许自动训练，则执行训练流程
        if not model_loaded and train_if_not_exist:
            print("Pre-trained model not found. Starting training process...")
            self.train()



    def predict_intent(self, text: str):
        """
        对输入的文本进行意图预测。

        Args:
            text (str): 用户输入的文本。

        Returns:
            dict: 包含预测结果的字典，例如：
                  {
                      "best_intent": "校园知识问答",
                      "confidence": 0.95,
                      "all_probs": {"校园知识问答": 0.95, ...},
                      "top_options": [("校园知识问答", 0.95), ("论文助手", 0.03)]
                  }
        """
        if not self.model.is_trained:
            raise RuntimeError("Classifier model is not ready. Please train or load it first.")

        # 1. 编码输入文本
        vector = self.text_encoder.encode([text])

        # 2. 获取模型预测结果
        predicted_intent, max_prob, probabilities, classes = self.model.predict(vector)

        # 3. 整理返回结果
        all_probs = dict(zip(classes, probabilities))

        # 获取最可能的N个选项
        top_indices = np.argsort(probabilities)[-config.TOP_N_OPTIONS:][::-1]
        top_options = [(classes[i], probabilities[i]) for i in top_indices]

        result = {
            "best_intent": predicted_intent,
            "confidence": max_prob,
            "all_probs": all_probs,
            "top_options": top_options
        }

        return result



