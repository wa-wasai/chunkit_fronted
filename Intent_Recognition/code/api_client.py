# intent_project/api_client.py

import requests
import json
from typing import Dict


class BailianApiClient:
    """
    用于与百炼大模型服务进行交互的 API 客户端。
    """

    def __init__(self, api_key: str, api_url: str, app_id_map: Dict[str, str], prompt_map: Dict[str, str]):
        """
        初始化 API 客户端。

        Args:
            api_key (str): 百炼平台的 API Key。
            api_url (str): API 请求的 URL。
            app_id_map (Dict[str, str]): 意图到 AppID 的映射字典。
            prompt_map (Dict[str, str]): 意图到系统提示的映射字典。
        """
        self.api_key = api_key
        self.api_url = api_url
        self.app_id_map = app_id_map
        self.prompt_map = prompt_map

    def get_response(self, intent: str, user_query: str) -> str:
        """
        根据识别出的意图和用户问题，调用百炼 API 获取回复。

        Args:
            intent (str): 已确定的用户意图。
            user_query (str): 用户的原始问题。

        Returns:
            str: 从 API 获取的文本回复或错误信息。
        """
        system_prompt = self.prompt_map.get(intent, "你是一个智能助手，请回答用户问题。")
        app_id = self.app_id_map.get(intent)

        if not app_id:
            return f"❌ 错误：未找到意图 '{intent}' 对应的 AppID。"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-DashScope-AppId": app_id
        }

        payload = {
            "model": "qwen-turbo",
            "input": {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
            },
            "parameters": {"temperature": 0.7}
        }

        try:
            print(f"Calling Bailian API for intent: {intent}...")
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()  # 如果请求失败（如4xx或5xx），则抛出异常

            result = response.json()
            # 稳定的获取返回文本的方式
            text_response = result.get("output", {}).get("text", "抱歉，未能从API获取有效回复。")

            # 检查是否返回了错误码
            if result.get("code"):
                return f"❌ API返回错误: {result.get('message', '未知错误')}"

            return text_response

        except requests.exceptions.RequestException as e:
            return f"❌ 网络请求失败: {e}"
        except json.JSONDecodeError:
            return f"❌ 返回格式解析失败: 无法解析JSON。原始返回: {response.text}"
        except Exception as e:
            return f"❌ 未知错误: {e}\n原始返回: {response.text if 'response' in locals() else 'N/A'}"
