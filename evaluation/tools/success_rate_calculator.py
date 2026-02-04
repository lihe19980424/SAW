# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==========================================================
# success_rate_calculator.py
# Description: Calculate success rate of watermark detection
# ==========================================================

from typing import List, Dict, Union
from exceptions.exceptions import TypeMismatchException, ConfigurationError
# add
from sklearn.metrics import roc_auc_score, roc_curve

class DetectionResult:
    """Detection result."""

    def __init__(self, gold_label: bool, detect_result: Union[bool, float]) -> None:
        """
            Initialize the detection result.

            Parameters:
                gold_label (bool): The expected watermark presence.
                detect_result (Union[bool, float]): The detection result.
        """
        self.gold_label = gold_label
        self.detect_result = detect_result


class BaseSuccessRateCalculator:
    """Base class for success rate calculator."""
    def __init__(self, labels: List[str] = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']) -> None:
        """
            Initialize the success rate calculator.

            Parameters:
                labels (List[str]): The list of metric labels to include in the output.
        """
        self.labels = labels
    
    def _check_instance(self, data: List[Union[bool, float]], expected_type: type):
        """Check if the data is an instance of the expected type."""
        for d in data:
            if not isinstance(d, expected_type):
                raise TypeMismatchException(expected_type, type(d))
    
    def _filter_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Filter metrics based on the provided labels."""
        return {label: metrics[label] for label in self.labels if label in metrics}
    
    def calculate(self, watermarked_result: List[Union[bool, float]], non_watermarked_result: List[Union[bool, float]]) -> Dict[str, float]:
        """Calculate success rates based on provided results."""
        pass


class FundamentalSuccessRateCalculator(BaseSuccessRateCalculator):
    """
        Calculator for fundamental success rates of watermark detection.

        This class specifically handles the calculation of success rates for scenarios involving
        watermark detection after fixed thresholding. It provides metrics based on comparisons
        between expected watermarked results and actual detection outputs.

        Use this class when you need to evaluate the effectiveness of watermark detection algorithms
        under fixed thresholding conditions.
    """


    def __init__(self, labels: List[str] = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']) -> None:
        """
            Initialize the fundamental success rate calculator.

            Parameters:
                labels (List[str]): The list of metric labels to include in the output.
        """

        super().__init__(labels)  # 调用基类的初始化方法，并传入度量标签
    
    def _compute_metrics(self, inputs: List[DetectionResult]) -> Dict[str, float]:
        """Compute metrics based on the provided inputs."""
        TP = sum(1 for d in inputs if d.detect_result and d.gold_label)
        TN = sum(1 for d in inputs if not d.detect_result and not d.gold_label)
        FP = sum(1 for d in inputs if d.detect_result and not d.gold_label)
        FN = sum(1 for d in inputs if not d.detect_result and d.gold_label)

        TPR = TP / (TP + FN) if TP + FN else 0.0
        FPR = FP / (FP + TN) if FP + TN else 0.0
        TNR = TN / (TN + FP) if TN + FP else 0.0
        FNR = FN / (FN + TP) if FN + TP else 0.0
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / (TP + FN) if TP + FN else 0.0
        F1 = 2 * (P * R) / (P + R) if P + R else 0.0
        ACC = (TP + TN) / (len(inputs)) if inputs else 0.0

        return {
            'TPR': TPR, 'TNR': TNR, 'FPR': FPR, 'FNR': FNR,
            'P': P, 'R': R, 'F1': F1, 'ACC': ACC
        }

    def calculate(self, watermarked_result: List[bool], non_watermarked_result: List[bool]) -> Dict[str, float]:
        """calculate success rates of watermark detection based on provided results."""
        self._check_instance(watermarked_result, bool)
        self._check_instance(non_watermarked_result, bool)

        inputs = [DetectionResult(True, x) for x in watermarked_result] + [DetectionResult(False, x) for x in non_watermarked_result]
        metrics = self._compute_metrics(inputs)
        return self._filter_metrics(metrics)


class DynamicThresholdSuccessRateCalculator(BaseSuccessRateCalculator):
    """
        Calculator for success rates of watermark detection with dynamic thresholding.

        This class calculates success rates for watermark detection scenarios where the detection
        thresholds can dynamically change based on varying conditions. It supports evaluating the
        effectiveness of watermark detection algorithms that adapt to different signal or noise conditions.

        Use this class to evaluate detection systems where the threshold for detecting a watermark
        is not fixed and can vary.
    """
    """
        计算带有动态阈值的水印检测成功率。

        该类用于计算水印检测场景中的成功率，在这些场景中，检测阈值可以根据不同的条件动态变化。
        它支持评估能够适应不同信号或噪声条件的水印检测算法的有效性。
        
        使用该类来评估水印检测系统，其中检测水印的阈值不是固定的，而是可变的。
    """
    def __init__(self, 
                 labels: List[str] = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'], 
                 rule='best', 
                 target_fpr=None,
                 reverse=False) -> None:
        """
            Initialize the dynamic threshold success rate calculator.

            Parameters:
                labels (List[str]): The list of metric labels to include in the output.
                rule (str): The rule for determining the threshold. Choose from 'best' or 'target_fpr'.
                target_fpr (float): The target false positive rate to achieve.
                reverse (bool): Whether to reverse the sorting order of the detection results.
                                True: higher values are considered positive.
                                False: lower values are considered positive.
        """
        """
        初始化动态阈值成功率计算器。

        参数:
            labels (List[str]): 输出中包含的度量标签列表。
            rule (str): 决定阈值的规则，可选择 'best' 或 'target_fpr'。
            target_fpr (float): 需要达到的目标假阳性率（FPR）。
            reverse (bool): 是否反转检测结果的排序顺序。
                            True 表示较大的值被视为阳性（正例），False 表示较小的值被视为阳性。
        """
        super().__init__(labels)
        self.rule = rule  # 保存阈值决策规则
        self.target_fpr = target_fpr  #  'target_fpr' 时有效
        self.reverse = reverse  # 是否反转检测结果的目标假阳性率，当规则为排序顺序
        
        # Validate rule configuration  # 验证规则是否合法，必须为 'best' 或 'target_fpr'
        if self.rule not in ['best', 'target_fpr']:
            raise ConfigurationError(f"Invalid rule specified: {self.rule}. Choose from 'best' or 'target_fpr'.")

        # Validate target_fpr configuration based on the rule
        if self.rule == 'target_fpr':
            if self.target_fpr is None:
                raise ConfigurationError("target_fpr must be set when rule is 'target_fpr'.")
            if not isinstance(self.target_fpr, (float, int)) or not (0 <= self.target_fpr <= 1):
                raise ConfigurationError("target_fpr must be a float or int within the range [0, 1].") # target_fpr 必须是 0 到 1 之间的 float 或 int。

    def _find_best_threshold(self, inputs: List[DetectionResult]) -> float:
        """Find the best threshold that maximizes F1."""  """查找最佳阈值，使 F1 分数最大化。"""
        best_threshold = 0  # 初始化最佳阈值
        best_metrics = None  # 初始化最佳度量
        # 遍历输入，找到使 F1 分数最大的阈值
        for i in range(len(inputs) - 1):
            # 计算当前检测结果两点之间的中间阈值
            threshold = (inputs[i].detect_result + inputs[i + 1].detect_result) / 2
            # 根据当前阈值计算度量
            metrics = self._compute_metrics(inputs, threshold)
            # 如果当前度量是最佳的（或首次计算），更新最佳阈值和最佳度量
            if best_metrics is None or metrics['F1'] > best_metrics['F1']:
                best_threshold = threshold
                best_metrics = metrics
        return best_threshold  # 返回最佳的阈值

    def _find_threshold_by_fpr(self, inputs: List[DetectionResult]) -> float:
        """Find the threshold that achieves the target FPR."""  """根据目标 FPR 找到阈值。"""
        threshold = 0  # 初始化阈值
         # 遍历输入，找到满足目标假阳性率（FPR）的阈值
        for i in range(len(inputs) - 1):
            # 计算当前检测结果两点之间的中间阈值
            threshold = (inputs[i].detect_result + inputs[i + 1].detect_result) / 2
            # 计算当前阈值的度量
            metrics = self._compute_metrics(inputs, threshold)
            # 如果当前假阳性率小于或等于目标 FPR，则停止查找
            if metrics['FPR'] <= self.target_fpr:
                break
        return threshold  # 返回符合目标 FPR 的阈值

    def _find_threshold(self, inputs: List[DetectionResult]) -> float:
        """Find the threshold based on the specified rule."""  """根据指定的规则查找阈值。"""
        # 根据检测结果的 detect_result 值进行排序，是否反转取决于 reverse 参数
        sorted_inputs = sorted(inputs, key=lambda x: x.detect_result, reverse=self.reverse)
        
        # If the rule is to find the best threshold by maximizing accuracy  # 如果规则为 'best'，则查找通过最大化 F1 分数找到的最佳阈值
        if self.rule == 'best':
            return self._find_best_threshold(sorted_inputs)
        else:
            # 否则，根据目标 FPR 查找阈值
            # If the rule is to find the threshold that achieves the target FPR
            return self._find_threshold_by_fpr(sorted_inputs)

    def _compute_metrics(self, inputs: List[DetectionResult], threshold: float) -> Dict[str, float]:
        """Compute metrics based on the provided inputs and threshold."""  """根据输入和阈值计算度量。"""
        # 计算真正例（TP）、假正例（FP）、真反例（TN）和假反例（FN）的数量
        if not self.reverse:
            # 如果 reverse=False，阈值以上的检测结果被视为阳性（正例）
            TP = sum(1 for x in inputs if x.detect_result >= threshold and x.gold_label)
            FP = sum(1 for x in inputs if x.detect_result >= threshold and not x.gold_label)
            TN = sum(1 for x in inputs if x.detect_result < threshold and not x.gold_label)
            FN = sum(1 for x in inputs if x.detect_result < threshold and x.gold_label)
        else:
            # 如果 reverse=True，阈值以下的检测结果被视为阳性（正例）
            TP = sum(1 for x in inputs if x.detect_result <= threshold and x.gold_label)
            FP = sum(1 for x in inputs if x.detect_result <= threshold and not x.gold_label)
            TN = sum(1 for x in inputs if x.detect_result > threshold and not x.gold_label)
            FN = sum(1 for x in inputs if x.detect_result > threshold and x.gold_label)
        # 计算并返回度量值
        metrics = {
            'TPR': TP / (TP + FN) if TP + FN else 0,  # 真阳性率
            'FPR': FP / (FP + TN) if FP + TN else 0,  # 假阳性率
            'TNR': TN / (TN + FP) if TN + FP else 0,  # 真阴性率
            'FNR': FN / (FN + TP) if FN + TP else 0,  # 假阴性率
            'P': TP / (TP + FP) if TP + FP else 0,  # 精确率
            'R': TP / (TP + FN) if TP + FN else 0,  # 召回率
            'F1': 2 * TP / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0,  # F1 分数
            'ACC': (TP + TN) / (len(inputs)) if inputs else 0  # 准确率
        }
        return metrics


    def calculate(self, watermarked_result: List[float], non_watermarked_result: List[float]) -> Dict[str, float]:
        """Calculate success rates based on provided results."""
        self._check_instance(watermarked_result + non_watermarked_result, float)

        # 1. 计算常规指标 (F1, TPR 等)
        inputs = [DetectionResult(True, x) for x in watermarked_result] + [DetectionResult(False, x) for x in non_watermarked_result]
        threshold = self._find_threshold(inputs)
        metrics = self._compute_metrics(inputs, threshold)
        filtered_metrics = self._filter_metrics(metrics)

        # 2. 计算 AUC
        y_true = [1] * len(watermarked_result) + [0] * len(non_watermarked_result)
        y_scores = watermarked_result + non_watermarked_result
        
        # 处理分数反转 (如 PPL 越低越是水印)
        if self.reverse:
            y_scores = [-x for x in y_scores]

        try:
            # [修改 1] 先检查是否同时包含正负样本 (0 和 1)
            # 如果某一个列表为空，y_true 就只有一种标签，无法计算 AUC
            if len(set(y_true)) < 2:
                # 这种情况下 AUC 未定义，设为 0.0 或 0.5，并不打印警告
                auc_value = 0.0 
            else:
                auc_value = roc_auc_score(y_true, y_scores)
            
            # [修改 2] 强制转换为 Python float，解决 np.float64() 显示问题
            filtered_metrics['AUC'] = float(auc_value)
            
        except Exception as e:
            # 如果发生其他未知错误，捕获并设为 0.0
            # print(f"Warning: AUC calculation failed. {e}") # 可以注释掉以保持清爽
            filtered_metrics['AUC'] = 0.0

        return filtered_metrics, threshold