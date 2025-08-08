from transformers import AutoTokenizer, AutoModel #来自 HuggingFace Transformers，用于加载预训练模型和分词器。
"""
AutoTokenizer: 自动加载与模型匹配的分词器（Tokenizer），负责将文本分解为模型可识别的 token 序列。
AutoModel: 自动加载模型本体（如 Qwen3 的 embedding 模型），输出 token 的表示向量。
"""
from typing import List, Optional, Union #Python 的类型提示模块，定义函数参数类型，如 List[str]、Optional[str]。
import torch #PyTorch 框架，处理张量、计算图、GPU加速等，是整个深度学习运算的核心。
import numpy as np #NumPy：数值运算库，用于处理向量化结果（矩阵拼接、归一化等）。
import logging #设置日志信息。
from tqdm import tqdm #显示编码进度条
import gc #调用 Python 的垃圾回收机制（释放内存）。
import time #计时工具。
import warnings #忽略非关键警告信息。



warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEncoder:
    """
    文本编码器，负责将文本转换为向量表示。
    使用 Qwen3-Embedding-0.6B 模型进行文本向量化。
    支持批处理、内存优化和多种池化策略。
    """

    def __init__(self,
                 model_path: str,
                 max_length: int = 512,
                 batch_size: int = 32,
                 use_mean_pooling: bool = True,
                 device: Optional[str] = None):
        """
        初始化编码器并加载模型

        Args:
            model_path (str): Qwen3-Embedding-0.6B 模型的本地路径
            max_length (int): 最大序列长度
            batch_size (int): 默认批处理大小，顾名思义表示一次同时处理多少条样本
            use_mean_pooling (bool): 是否使用平均池化（而非CLS token）
            device (Optional[str]): 指定设备 ('cuda', 'cpu', None为自动选择)
        """
        self.model_path = model_path
        self.max_length = max_length
        self.default_batch_size = batch_size
        self.use_mean_pooling = use_mean_pooling
        self.device = self._setup_device(device)

        # 模型组件
        self.tokenizer = None
        self.model = None
        self.embedding_dim = None

        # 性能统计
        self.total_texts_encoded = 0 #已编码的文本数量
        self.total_encoding_time = 0.0 #编码总时间

        logger.info("初始化 TextEncoder...")
        self._load_model()

    def _setup_device(self, device: Optional[str]) -> torch.device:
        """设置计算设备"""
        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            device = torch.device('cuda')
            # 显示GPU信息
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"使用GPU: {gpu_name}")
            logger.info(f"GPU显存: {gpu_memory:.1f}GB")

            # 检查可用显存
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            cached = torch.cuda.memory_reserved(0) / (1024 ** 3)
            logger.info(f"显存使用: {allocated:.1f}GB 已分配, {cached:.1f}GB 已缓存")
        else:
            device = torch.device('cpu')
            logger.info("使用CPU进行推理")

        return device

    def _load_model(self):
        """加载模型和分词器"""
        try:
            logger.info("加载Qwen3-Embedding分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            logger.info("加载Qwen3-Embedding模型...")
            # 根据设备选择数据类型
            torch_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32

            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype
            )

            # 移动到指定设备
            self.model.to(self.device)
            self.model.eval()

            # 获取嵌入维度
            self.embedding_dim = self._detect_embedding_dimension()

            logger.info(f"✅ 模型加载成功！")
            logger.info(f"嵌入维度: {self.embedding_dim}")
            logger.info(f"最大序列长度: {self.max_length}")
            logger.info(f"默认批大小: {self.default_batch_size}")

        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise

    def _detect_embedding_dimension(self) -> int:
        """自动检测嵌入向量维度"""
        try:
            # 使用简单测试文本
            test_embedding = self.encode(["测试"], batch_size=1, show_progress=False)
            return test_embedding.shape[1]
        except Exception as e:
            logger.warning(f"无法自动检测嵌入维度: {e}")
            return 768  # 默认维度

    def mean_pooling(self, model_output, attention_mask):
        """
        平均池化，考虑attention mask，通常比CLS token效果更好

        Args:
            model_output: 模型输出，包含last_hidden_state
            attention_mask: 注意力掩码

        Returns:
            torch.Tensor: 池化后的向量表示
        """
        token_embeddings = model_output.last_hidden_state

        # 扩展attention_mask维度以匹配token_embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # 计算加权平均，忽略padding token
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def max_pooling(self, model_output, attention_mask):
        """
        最大池化策略

        Args:
            model_output: 模型输出
            attention_mask: 注意力掩码

        Returns:
            torch.Tensor: 池化后的向量
        """
        token_embeddings = model_output.last_hidden_state

        # 将padding位置设为很小的值
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9

        # 最大池化
        return torch.max(token_embeddings, 1)[0]

    def encode(self,
               texts: List[str],
               batch_size: Optional[int] = None,
               show_progress: bool = True,
               normalize: bool = True,
               pooling_strategy: str = 'mean') -> np.ndarray:
        """
        将文本列表编码为向量矩阵

        Args:
            texts (List[str]): 需要编码的文本列表
            batch_size (Optional[int]): 批处理大小，None使用默认值
            show_progress (bool): 是否显示进度条
            normalize (bool): 是否对向量进行L2归一化
            pooling_strategy (str): 池化策略 ('mean', 'max', 'cls')

        Returns:
            np.ndarray: 文本向量矩阵，shape为(n_texts, embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)

        # 验证输入
        texts = self._validate_and_preprocess_texts(texts)

        if batch_size is None:
            batch_size = self.default_batch_size

        # 动态调整批大小以适应GPU内存
        if self.device.type == 'cuda':
            batch_size = self._adjust_batch_size_for_memory(batch_size, len(texts[0]) if texts else 100)

        logger.info(f"编码 {len(texts)} 个文本")
        logger.info(f"批大小: {batch_size}, 池化策略: {pooling_strategy}")

        start_time = time.time()
        all_embeddings = []

        # 使用tqdm显示进度
        iterator = range(0, len(texts), batch_size)
        if show_progress and len(texts) > batch_size:
            iterator = tqdm(iterator, desc="编码进度", unit="batch")

        try:
            for i in iterator:
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self._encode_batch(batch_texts, pooling_strategy)
                all_embeddings.append(batch_embeddings)

                # GPU内存管理
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            # 拼接所有批次结果
            result = np.vstack(all_embeddings)

            # L2归一化
            if normalize:
                result = self._normalize_vectors(result)

            # 更新统计信息
            encoding_time = time.time() - start_time
            self.total_texts_encoded += len(texts)
            self.total_encoding_time += encoding_time

            logger.info(f"✅ 编码完成")
            logger.info(f"输出shape: {result.shape}")
            logger.info(f"耗时: {encoding_time:.2f}秒 ({len(texts) / encoding_time:.1f} 文本/秒)")

            return result

        except Exception as e:
            logger.error(f"❌ 编码过程中出错: {e}")
            raise

    def _validate_and_preprocess_texts(self, texts: List[str]) -> List[str]:
        """验证和预处理输入文本"""
        processed_texts = []

        for i, text in enumerate(texts):
            # 类型检查
            if not isinstance(text, str):
                text = str(text)

            # 去除首尾空白
            text = text.strip()

            # 处理空文本
            if not text:
                text = "[空文本]"
                logger.debug(f"文本 {i} 为空，已替换为占位符")

            # 长度检查
            if len(text) > self.max_length * 3:  # 粗略估计字符与token比例
                logger.warning(f"文本 {i} 可能过长 (长度: {len(text)})，将被截断")

            processed_texts.append(text)

        return processed_texts

    def _adjust_batch_size_for_memory(self, batch_size: int, avg_text_length: int) -> int:
        """根据GPU内存动态调整批大小"""
        if self.device.type != 'cuda':
            return batch_size

        try:
            # 获取可用显存
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            available_memory = total_memory - allocated_memory

            # 简单的内存估算（很粗糙，但有用）
            estimated_memory_per_sample = avg_text_length * self.embedding_dim * 4  # 4 bytes per float32
            safe_batch_size = max(1, int(available_memory * 0.5 / estimated_memory_per_sample))

            adjusted_batch_size = min(batch_size, safe_batch_size)

            if adjusted_batch_size != batch_size:
                logger.info(f"根据GPU内存调整批大小: {batch_size} -> {adjusted_batch_size}")

            return adjusted_batch_size

        except Exception as e:
            logger.warning(f"无法调整批大小: {e}")
            return batch_size

    def _encode_batch(self, texts: List[str], pooling_strategy: str = 'mean') -> np.ndarray:
        """
        编码单个批次

        Args:
            texts (List[str]): 文本列表
            pooling_strategy (str): 池化策略

        Returns:
            np.ndarray: 编码后的向量矩阵
        """
        # 分词
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=self.max_length,
            add_special_tokens=True
        )

        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 模型推理
        with torch.no_grad():
            outputs = self.model(**inputs)

            # 根据策略选择池化方法
            if pooling_strategy == 'mean':
                embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
            elif pooling_strategy == 'max':
                embeddings = self.max_pooling(outputs, inputs['attention_mask'])
            elif pooling_strategy == 'cls':
                # 使用CLS token (第一个token)
                embeddings = outputs.last_hidden_state[:, 0, :]
            else:
                raise ValueError(f"不支持的池化策略: {pooling_strategy}")

            # 转换为numpy数组
            embeddings = embeddings.cpu().numpy()

        return embeddings

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        L2归一化向量，使其具有单位长度

        Args:
            vectors (np.ndarray): 输入向量矩阵

        Returns:
            np.ndarray: 归一化后的向量矩阵
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # 避免除零错误
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def encode_single(self, text: str,
                      normalize: bool = True,
                      pooling_strategy: str = 'mean') -> np.ndarray:
        """
        编码单个文本

        Args:
            text (str): 单个文本
            normalize (bool): 是否归一化
            pooling_strategy (str): 池化策略

        Returns:
            np.ndarray: 文本向量
        """
        result = self.encode(
            [text],
            batch_size=1,
            show_progress=False,
            normalize=normalize,
            pooling_strategy=pooling_strategy
        )
        return result[0]

    def compute_similarity(self, text1: str, text2: str,
                           similarity_metric: str = 'cosine') -> float:
        """
        计算两个文本的相似度

        Args:
            text1 (str): 第一个文本
            text2 (str): 第二个文本
            similarity_metric (str): 相似度度量 ('cosine', 'euclidean', 'dot')

        Returns:
            float: 相似度分数
        """
        #文本向量化
        vectors = self.encode([text1, text2], show_progress=False, normalize=True)

        if similarity_metric == 'cosine':
            # 余弦相似度 (向量已归一化，所以就是点积)
            similarity = np.dot(vectors[0], vectors[1])
        elif similarity_metric == 'euclidean':
            # 欧氏距离 (转换为相似度)
            distance = np.linalg.norm(vectors[0] - vectors[1])
            similarity = 1 / (1 + distance)
        elif similarity_metric == 'dot':
            # 点积
            similarity = np.dot(vectors[0], vectors[1])
        else:
            raise ValueError(f"不支持的相似度度量: {similarity_metric}")

        return float(similarity)

    def find_most_similar(self, query_text: str,
                          candidate_texts: List[str],
                          top_k: int = 5) -> List[tuple]:
        """
        找到与查询文本最相似的候选文本

        Args:
            query_text (str): 查询文本
            candidate_texts (List[str]): 候选文本列表
            top_k (int): 返回top-k结果

        Returns:
            List[tuple]: (相似度分数, 文本索引, 文本内容) 的列表
        """
        # 编码所有文本
        all_texts = [query_text] + candidate_texts
        all_vectors = self.encode(all_texts, normalize=True)

        query_vector = all_vectors[0]
        candidate_vectors = all_vectors[1:]

        # 计算相似度
        similarities = np.dot(candidate_vectors, query_vector)

        # 获取top-k结果
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (float(similarities[idx]), int(idx), candidate_texts[idx])
            for idx in top_indices
        ]

        return results

    def get_encoder_info(self) -> dict:
        """
        获取编码器的详细信息

        Returns:
            dict: 编码器配置和统计信息
        """
        info = {
            "model_path": self.model_path,
            "device": str(self.device),
            "max_length": self.max_length,
            "embedding_dim": self.embedding_dim,
            "default_batch_size": self.default_batch_size,
            "use_mean_pooling": self.use_mean_pooling,
            "total_texts_encoded": self.total_texts_encoded,
            "total_encoding_time": round(self.total_encoding_time, 2),
            "avg_speed": round(self.total_texts_encoded / max(self.total_encoding_time, 1), 2)
        }

        # 添加分词器信息
        if self.tokenizer:
            info["vocab_size"] = len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else "未知"
            info["model_max_length"] = getattr(self.tokenizer, 'model_max_length', "未知")

        # 添加GPU信息
        if self.device.type == 'cuda':
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.1f}GB"
            info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / (1024 ** 3):.1f}GB"

        return info

    def print_encoder_summary(self):
        """打印编码器摘要信息"""
        info = self.get_encoder_info()

        print("\n" + "=" * 50)
        print("文本编码器摘要")
        print("=" * 50)
        print(f"模型路径: {info['model_path']}")
        print(f"计算设备: {info['device']}")
        print(f"嵌入维度: {info['embedding_dim']}")
        print(f"最大长度: {info['max_length']}")
        print(f"默认批大小: {info['default_batch_size']}")

        if info['total_texts_encoded'] > 0:
            print(f"\n性能统计:")
            print(f"已编码文本数: {info['total_texts_encoded']}")
            print(f"总编码时间: {info['total_encoding_time']}秒")
            print(f"平均速度: {info['avg_speed']} 文本/秒")

        if 'gpu_name' in info:
            print(f"\nGPU信息:")
            print(f"GPU型号: {info['gpu_name']}")
            print(f"总显存: {info['gpu_memory_total']}")
            print(f"已用显存: {info['gpu_memory_allocated']}")

    def benchmark_encoding_speed(self, test_texts: Optional[List[str]] = None,
                                 batch_sizes: List[int] = None) -> dict:
        """
        测试编码速度性能

        Args:
            test_texts (Optional[List[str]]): 测试文本，None则生成默认测试文本
            batch_sizes (List[int]): 要测试的批大小列表

        Returns:
            dict: 性能测试结果
        """
        if test_texts is None:
            # 生成测试文本
            test_texts = [
                f"这是第{i}个测试文本，用于评估编码器的性能表现。" * (i % 3 + 1)
                for i in range(100)
            ]

        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32, 64]

        results = {}

        print("\n开始编码速度基准测试...")
        for batch_size in batch_sizes:
            print(f"\n测试批大小: {batch_size}")

            start_time = time.time()
            try:
                vectors = self.encode(test_texts, batch_size=batch_size, show_progress=False)
                end_time = time.time()

                elapsed_time = end_time - start_time
                speed = len(test_texts) / elapsed_time

                results[batch_size] = {
                    "elapsed_time": round(elapsed_time, 3),
                    "speed": round(speed, 2),
                    "success": True
                }

                print(f"  耗时: {elapsed_time:.3f}秒")
                print(f"  速度: {speed:.2f} 文本/秒")

            except Exception as e:
                results[batch_size] = {
                    "error": str(e),
                    "success": False
                }
                print(f"  错误: {e}")

        # 找到最佳批大小
        successful_results = {k: v for k, v in results.items() if v.get('success', False)}
        if successful_results:
            best_batch_size = max(successful_results.keys(),
                                  key=lambda k: successful_results[k]['speed'])
            print(f"\n推荐批大小: {best_batch_size} (速度: {successful_results[best_batch_size]['speed']:.2f} 文本/秒)")

        return results

    def __del__(self):
        """析构函数，清理GPU内存"""
        try:
            if hasattr(self, 'device') and self.device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
        except:
            pass