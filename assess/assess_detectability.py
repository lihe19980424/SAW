# =================================================================
# assess_detectability.py
# Description: Assess the detectability of a watermarking algorithm
# =================================================================

import torch
from evaluation.dataset import C4Dataset, WMT16DE_ENDataset, HumanEvalDataset  # 从evaluation模块导入C4Dataset，用于加载数据集
from watermark.auto_watermark import AutoWatermark  # 导入AutoWatermark类，用于加载水印算法
from utils.transformers_config import TransformersConfig  # 导入TransformersConfig类，配置transformer模型
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, LlamaForCausalLM, T5ForConditionalGeneration, AutoTokenizer, LlamaTokenizer, T5Tokenizer  # 导入transformers库的模型和tokenizer类
from evaluation.tools.text_editor import TruncatePromptTextEditor  # 导入文本编辑器工具，用于编辑和截断提示
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator, FundamentalSuccessRateCalculator  # 动态阈值成功率计算器
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType  # 导入检测流水线相关类，用于水印和非水印文本检测

# 判断是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(device) # 输出设备信息
# 定义评估水印算法检测性的函数  


# 静态阈值的代码
# def assess_detectability(algorithm_name, labels):

# 原来动态阈值的代码 
def assess_detectability(algorithm_name, labels, rules, target_fpr):
    # 加载处理过的C4数据集 /home/lihe/MarkLLM/dataset/wmt16_de_en/validation.jsonl
    my_dataset = C4Dataset('dataset/c4/processed_c4.json')
    # my_dataset = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl')
    # my_dataset = HumanEvalDataset('dataset/human_eval/test.jsonl')
    # 配置transformer模型和tokenizer
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("./models/opt-1.3b").to(device),
                                             # model=LlamaForCausalLM.from_pretrained("./models/llama-7b-hf").to(device),
                                             # model=T5ForConditionalGeneration.from_pretrained("./models/t5-v1_1-xxl").to(device),
                                             tokenizer=AutoTokenizer.from_pretrained("./models/opt-1.3b"),
                                             # tokenizer=LlamaTokenizer.from_pretrained("./models/llama-7b-hf"),
                                             # tokenizer=T5Tokenizer.from_pretrained("./models/t5-v1_1-xxl"),
                                             vocab_size=50272,
                                             # vocab_size=32000,
                                             # vocab_size=32128,
                                             device=device,
                                             max_new_tokens=200,
                                             min_length=200,
                                             do_sample=True,
                                             no_repeat_ngram_size=4)
    
    # 原来的代码
    # 加载指定的水印算法，使用配置文件
    my_watermark = AutoWatermark.load(f'{algorithm_name}',  # 算法名称
                                    algorithm_config=f'config/{algorithm_name}.json',  # 对应的算法配置文件
                                    transformers_config=transformers_config)  # 上述配置的transformer模型
    
    # 创建一个用于检测水印文本的流水线 # 使用C4数据集  # 使用文本编辑器截断提示词 # 显示进度 # 返回类型为分数
    pipline1 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[TruncatePromptTextEditor()],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES) 
    # 创建一个用于检测非水印文本的流水线 # 使用相同的数据集 # 不使用文本编辑器 # 显示进度 # 返回类型为分数
    pipline2 = UnWatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES)   # IS_WATERMARKED
    # 动态阈值成功率计算器，用于评估流水线的成功率 # 评估指标 # 评估规则 # 目标的错误接受率（False Positive Rate）
    calculator = DynamicThresholdSuccessRateCalculator(labels=labels, rule=rules, target_fpr=target_fpr,reverse=False)
    # 静态阈值
    # calculator = FundamentalSuccessRateCalculator(labels=labels)
    # 计算并输出水印和非水印文本的成功率
    print(calculator.calculate(pipline1.evaluate(my_watermark), pipline2.evaluate(my_watermark)))

# 主函数入口
if __name__ == '__main__':
    import argparse  # 导入argparse模块，用于处理命令行参数
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数：算法名称
    parser.add_argument('--algorithm', type=str, default='Rethinking')  # 水印算法名称
    # 添加命令行参数：评估指标
    parser.add_argument('--labels', nargs='+', default=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'])
    # 添加命令行参数：评估规则
    parser.add_argument('--rules', type=str, default='best') #target_fpr
    # 添加命令行参数：目标错误接受率
    parser.add_argument('--target_fpr', type=float, default=0.01)
    # 解析命令行参数
    args = parser.parse_args()

    # 调用评估函数，传入解析后的参数  
    # KGW(rerverse=false)               : {'TPR': 1.0, 'TNR': 1.0, 'FPR': 0.0, 'FNR': 0.0, 'P': 1.0, 'R': 1.0, 'F1': 1.0, 'ACC': 1.0}
    # Black_Box(rerverse=true,FPR=None) : {'TPR': 0.98, 'TNR': 0.0, 'FPR': 1.0, 'FNR': 0.02, 'P': 0.494949494949495, 'R': 0.98, 'F1': 0.6577181208053692, 'ACC': 0.49}
    # Black_Box(rerverse=true,FPR=0.01) : {'TPR': 0.98, 'TNR': 0.0, 'FPR': 1.0, 'FNR': 0.02, 'P': 0.494949494949495, 'R': 0.98, 'F1': 0.6577181208053692, 'ACC': 0.49}
    # Black_Box(rerverse=false)         : {'TPR': 0.94, 'TNR': 0.16, 'FPR': 0.84, 'FNR': 0.06, 'P': 0.5280898876404494, 'R': 0.94, 'F1': 0.6762589928057554, 'ACC': 0.55}
    
    # 原来（动态阈值）的代码
    assess_detectability(args.algorithm, args.labels, args.rules, args.target_fpr)
    
    # 静态阈值的代码
    # assess_detectability(args.algorithm, args.labels)