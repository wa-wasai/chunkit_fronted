import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Tuple, Dict, Any, Optional
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class IntentRecognitionModel:
    """
    意图识别模型，封装了随机森林分类器的训练、预测、保存和加载功能。
    支持超参数优化和全面的模型评估。
    """

    def __init__(self,
                 n_estimators: int = 50,#森林中的数量。
                 max_depth: Optional[int] = 30,#树的最大深度
                 min_samples_split: int = 15,  # 分割内部节点所需的最小样本数，默认为5。该参数用于控制过拟合，较大的值可以防止模型学习过于具体的规则，从而提高泛化能力。
                 min_samples_leaf: int = 4,   # 叶节点上的最小样本数，默认为2。此参数确保叶子节点不会太小，有助于减少模型对训练数据的过度拟合，提升模型的稳定性。
                 max_features: str = 'sqrt',#每次分裂时考虑的特征数量。'sqrt'表示取总特征数的平方根，能降低相关性；这里其实也可以指定比例比如'log2'
                 bootstrap: bool = True,#是否在构造树的时候使用有放回抽样，True表述典型的随机森林
                 oob_score: bool = True,#是否使用袋外样本估计泛化误差（Out-Of-Bag评分）
                 n_jobs: int = -1,# -1表述使用全部可用核心
                 random_state: int = 42,#随机种子
                 class_weight: str = 'balanced'):#类别权重策略。设置为'balanced'时，会根据样本分布字典跳转各类的权重，缓解类别不平衡的问题
        """
        小于5的不会被分割成一个内部节点，小于2的不会被分割成一个叶节点

        """
        """
        初始化模型。

        Args:
            n_estimators (int): 随机森林中树的数量，建议100-300
            max_depth (Optional[int]): 树的最大深度，None表示不限制
            min_samples_split (int): 内部节点分割所需的最小样本数
            min_samples_leaf (int): 叶子节点的最小样本数
            max_features (str): 每次分割考虑的特征数量 ('sqrt', 'log2', None)
            bootstrap (bool): 是否使用bootstrap采样
            oob_score (bool): 是否计算袋外误差
            n_jobs (int): 并行作业数，-1表示使用所有处理器
            random_state (int): 随机种子，确保结果可复现
            class_weight (str): 类别权重策略，'balanced'自动处理类别不平衡
        """
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight=class_weight
        )
        
        # 初始化标准化器，用于特征缩放
        self.scaler = StandardScaler()
        self.is_trained = False#模型是否已训练
        self.accuracy = None#模型准确度
        self.best_params = None#最佳参数字典
        self.class_distribution = None#类别分布字典
        self.feature_importance = None#特征重要性字典
        self.cross_val_scores = None#交叉验证得分

    def analyze_data_distribution(self, y) -> Dict[str, int]:
        """
        分析数据分布，检查类别不平衡

        Args:
            y: 标签数据

        Returns:
            Dict[str, int]: 类别分布字典
        """
        class_counts = Counter(y)#用counter来统计各意图类别的样本数
        self.class_distribution = dict(class_counts)#输出保存至self.class_distribution

        print("=" * 50)
        print("数据分布分析")
        print("=" * 50)
        print(f"总样本数: {len(y)}")
        print(f"类别数量: {len(class_counts)}")
        print("\n各类别分布:")

        for intent, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(y)) * 100
            print(f"  {intent}: {count} ({percentage:.1f}%)")

        # 检查类别不平衡
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count

        if imbalance_ratio > 3:
            print(f"\n⚠️  警告: 数据存在类别不平衡")
            print(f"最大类别是最小类别的 {imbalance_ratio:.1f} 倍")
            print("建议: 已启用 class_weight='balanced' 来处理这个问题")

        return self.class_distribution

    def optimize_hyperparameters(self, X, y, cv: int = 5, n_iter: int = 50) -> Dict[str, Any]:
        """
        使用随机搜索优化超参数

        Args:
            X: 特征数据
            y: 标签数据
            cv (int): 交叉验证折数
            n_iter (int): 随机搜索迭代次数

        Returns:
            Dict[str, Any]: 最佳参数字典
        """
        print("\n" + "=" * 50)
        print("开始超参数优化")
        print("=" * 50)

        # 定义参数搜索空间
        param_distributions = {
            'n_estimators': [50, 100, 150, 200, 300],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
            'bootstrap': [True, False]
        }

        # 创建基础分类器
        base_clf = RandomForestClassifier(
            oob_score=True,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )

        # 随机搜索
        print(f"正在尝试 {n_iter} 种参数组合...")
        random_search = RandomizedSearchCV(
            estimator=base_clf,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=0  # 减少输出
        )

        # 执行搜索
        random_search.fit(X, y)

        # 保存最佳参数
        self.best_params = random_search.best_params_
        self.clf = random_search.best_estimator_

        print(f"✅ 超参数优化完成!")
        print(f"最佳交叉验证得分: {random_search.best_score_:.4f}")
        print("最佳参数:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")

        return self.best_params

    def train(self, X, y, test_size: float = 0.20, optimize: bool = False, validate: bool = True):
        """
        训练模型，支持超参数优化和全面评估

        Args:
            X: 特征数据（文本向量）
            y: 标签数据（意图）
            test_size (float): 测试集比例
            optimize (bool): 是否进行超参数优化
            validate (bool): 是否进行详细验证评估
        """
        print("开始模型训练...")

        # 分析数据分布
        self.analyze_data_distribution(y)

        # 数据分割（使用分层采样保持类别比例）
        print(f"\n划分训练集和测试集 (测试集比例: {test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        print(f"训练集: {len(X_train)} 样本")
        print(f"测试集: {len(X_test)} 样本")

        # 特征标准化（虽然随机森林对此不敏感，但可能有帮助）
        print("\n进行特征标准化...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 超参数优化（可选）
        if optimize:
            print("\n🔧 启用超参数优化...")
            self.optimize_hyperparameters(X_train_scaled, y_train)
        else:
            print("\n使用默认参数训练...")

        # 训练模型
        print("\n训练随机森林模型...")
        self.clf.fit(X_train_scaled, y_train)
        self.is_trained = True

        # 模型验证
        if validate:
            self._comprehensive_evaluation(X_train_scaled, X_test_scaled, y_train, y_test)

        print("\n✅ 模型训练完成！")

    def _comprehensive_evaluation(self, X_train, X_test, y_train, y_test):
        """
        全面的模型性能评估
        """
        print("\n" + "=" * 50)
        print("模型性能评估")
        print("=" * 50)

        # 基本性能指标
        train_accuracy = self.clf.score(X_train, y_train)
        test_accuracy = self.clf.score(X_test, y_test)
        self.accuracy = test_accuracy

        print(f"训练集准确率: {train_accuracy:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")

        # 过拟合检查
        overfitting = train_accuracy - test_accuracy
        if overfitting > 0.1:
            print(f"⚠️  可能存在过拟合 (差异: {overfitting:.4f})")
            print("建议: 增加 min_samples_leaf 或减少 max_depth")
        elif overfitting < 0.02:
            print("✅ 模型拟合良好")

        # 袋外误差（如果启用）
        if hasattr(self.clf, 'oob_score_') and self.clf.oob_score_:
            print(f"袋外准确率: {self.clf.oob_score_:.4f}")

        # 交叉验证
        print("\n进行5折交叉验证...")
        cv_scores = cross_val_score(self.clf, X_train, y_train, cv=5, scoring='accuracy')
        self.cross_val_scores = cv_scores
        print(f"交叉验证准确率: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")

        # 详细分类报告
        y_pred = self.clf.predict(X_test)
        print("\n详细分类报告:")
        print(classification_report(y_test, y_pred, digits=4))

        # 特征重要性分析
        self._analyze_feature_importance()

        # 混淆矩阵
        if len(set(y_test)) <= 10:  # 只为较少类别显示混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            print(f"\n混淆矩阵:")
            print(cm)

    def _analyze_feature_importance(self, top_n: int = 20):
        """
        分析和显示特征重要性
        """
        if hasattr(self.clf, 'feature_importances_'):
            self.feature_importance = self.clf.feature_importances_

            # 统计信息
            total_features = len(self.feature_importance)
            non_zero_features = np.sum(self.feature_importance > 0)
            avg_importance = np.mean(self.feature_importance)

            print(f"\n特征重要性分析:")
            print(f"总特征数: {total_features}")
            print(f"有贡献特征数: {non_zero_features}")
            print(f"平均重要性: {avg_importance:.6f}")

            # 显示最重要的特征
            indices = np.argsort(self.feature_importance)[::-1]
            print(f"\n前{min(top_n, total_features)}个最重要特征:")
            for i in range(min(top_n, total_features)):
                importance = self.feature_importance[indices[i]]
                if importance > 0:
                    print(f"  特征 {indices[i]:4d}: {importance:.6f}")

    def predict(self, vector: np.ndarray) -> Tuple[str, float, np.ndarray, np.ndarray]:
        """
        对单个向量进行意图预测

        Args:
            vector (np.ndarray): 单个文本的向量表示

        Returns:
            Tuple[str, float, np.ndarray, np.ndarray]:
                预测意图、置信度、所有类别概率、类别标签
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练。请先调用 train() 或 load_model() 方法。")

        # 确保输入是二维数组
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        # 特征标准化
        vector_scaled = self.scaler.transform(vector)

        # 预测概率
        probabilities = self.clf.predict_proba(vector_scaled)[0]
        max_prob_index = np.argmax(probabilities)
        max_prob = probabilities[max_prob_index]
        predicted_intent = self.clf.classes_[max_prob_index]

        return predicted_intent, max_prob, probabilities, self.clf.classes_

    def predict_batch(self, vectors: np.ndarray, confidence_threshold: float = 0.5) -> list:
        """
        批量预测，支持置信度阈值

        Args:
            vectors (np.ndarray): 文本向量矩阵
            confidence_threshold (float): 置信度阈值，低于此值的预测会被标记

        Returns:
            list: 预测结果列表，每个元素包含预测信息
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练。请先调用 train() 或 load_model() 方法。")

        # 特征标准化
        vectors_scaled = self.scaler.transform(vectors)

        # 批量预测
        probabilities = self.clf.predict_proba(vectors_scaled)
        predicted_classes = self.clf.predict(vectors_scaled)

        results = []
        for i, (pred_class, probs) in enumerate(zip(predicted_classes, probabilities)):
            max_prob = np.max(probs)

            # 构建结果字典
            result = {
                'predicted_intent': pred_class,
                'confidence': float(max_prob),
                'low_confidence': max_prob < confidence_threshold,
                'all_probabilities': {
                    str(cls): float(prob)
                    for cls, prob in zip(self.clf.classes_, probs)
                }
            }
            results.append(result)

        return results

    def save_model(self, path: str):
        """
        保存完整的模型信息到文件

        Args:
            path (str): 模型保存路径
        """
        if not self.is_trained:
            raise RuntimeError("无法保存未训练的模型。")

        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)

        print(f"保存模型到 {path}...")

        # 保存完整的模型数据
        model_data = {
            "model": self.clf,
            "scaler": self.scaler,
            "accuracy": self.accuracy,
            "best_params": self.best_params,
            "class_distribution": self.class_distribution,
            "feature_importance": self.feature_importance,
            "cross_val_scores": self.cross_val_scores,
            "model_params": self.clf.get_params(),
            "version": "2.0",
            "n_features": self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else None
        }

        joblib.dump(model_data, path)
        print("✅ 模型保存成功！")

    def load_model(self, path: str) -> bool:
        """
        从文件加载预训练模型

        Args:
            path (str): 模型文件路径

        Returns:
            bool: 加载成功返回True，否则返回False
        """
        if not os.path.exists(path):
            print(f"❌ 模型文件不存在: {path}")
            return False

        print(f"从 {path} 加载模型...")

        try:
            data = joblib.load(path)

            if isinstance(data, dict):
                # 新版本格式 - 完整数据
                self.clf = data["model"]
                self.scaler = data.get("scaler", StandardScaler())
                self.accuracy = data.get("accuracy", None)
                self.best_params = data.get("best_params", None)
                self.class_distribution = data.get("class_distribution", None)
                self.feature_importance = data.get("feature_importance", None)
                self.cross_val_scores = data.get("cross_val_scores", None)

                # 显示版本信息
                version = data.get("version", "1.0")
                print(f"模型版本: {version}")

            else:
                # 旧版本格式 - 仅模型
                print("⚠️  检测到旧版本模型格式")
                self.clf = data
                self.scaler = StandardScaler()
                print("警告: 缺少预处理器，可能影响预测准确性")

            # 验证模型有效性
            if hasattr(self.clf, 'predict_proba') and hasattr(self.clf, 'classes_'):
                self.is_trained = True

                # 显示模型信息
                accuracy_info = f"准确率: {self.accuracy:.4f}" if self.accuracy else "准确率未知"
                print(f"✅ 模型加载成功！ {accuracy_info}")

                if self.class_distribution:
                    print(f"支持 {len(self.class_distribution)} 个意图类别")

                if hasattr(self.clf, 'n_estimators'):
                    print(f"随机森林树数量: {self.clf.n_estimators}")

                return True
            else:
                print(f"❌ 加载的对象不是有效的分类器")
                return False

        except Exception as e:
            print(f"❌ 加载模型时出错: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型的详细信息

        Returns:
            Dict[str, Any]: 包含模型各种信息的字典
        """
        if not self.is_trained:
            return {"status": "未训练"}

        info = {
            "status": "已训练",
            "accuracy": self.accuracy,
            "model_type": "RandomForestClassifier",
            "model_params": self.clf.get_params(),
            "best_params": self.best_params,
            "class_distribution": self.class_distribution,
            "n_classes": len(self.clf.classes_),
            "classes": list(self.clf.classes_),
            "n_features": getattr(self.scaler, 'n_features_in_', None)
        }

        # 添加性能信息
        if hasattr(self.clf, 'oob_score_'):
            info["oob_score"] = self.clf.oob_score_

        if self.cross_val_scores is not None:
            info["cross_val_mean"] = float(np.mean(self.cross_val_scores))
            info["cross_val_std"] = float(np.std(self.cross_val_scores))

        # 添加特征重要性统计
        if self.feature_importance is not None:
            info["feature_importance_stats"] = {
                "mean": float(np.mean(self.feature_importance)),
                "std": float(np.std(self.feature_importance)),
                "max": float(np.max(self.feature_importance)),
                "non_zero_features": int(np.sum(self.feature_importance > 0))
            }

        return info

    def print_model_summary(self):
        """打印模型摘要信息"""
        if not self.is_trained:
            print("模型尚未训练")
            return

        info = self.get_model_info()

        print("\n" + "=" * 50)
        print("模型摘要")
        print("=" * 50)
        print(f"模型类型: {info['model_type']}")
        print(f"准确率: {info['accuracy']:.4f}")
        print(f"支持类别数: {info['n_classes']}")
        print(f"特征维度: {info['n_features']}")

        if 'oob_score' in info:
            print(f"袋外得分: {info['oob_score']:.4f}")

        if 'cross_val_mean' in info:
            print(f"交叉验证: {info['cross_val_mean']:.4f} (±{info['cross_val_std']:.4f})")

        print(f"随机森林参数:")
        key_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
        for param in key_params:
            if param in info['model_params']:
                print(f"  {param}: {info['model_params'][param]}")