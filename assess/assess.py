# =================================================================
# assess_detectability.py
# Description: Assess the detectability of a watermarking algorithm
# =================================================================

import torch
from translate import Translator
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, LlamaForCausalLM, T5ForConditionalGeneration, AutoTokenizer, LlamaTokenizer, T5Tokenizer  # 导入transformers库的模型和tokenizer类
from watermark.auto_watermark import AutoWatermark  # 导入AutoWatermark类，用于加载水印算法
from evaluation.dataset import C4Dataset, WMT16DE_ENDataset, HumanEvalDataset  # 从evaluation模块导入C4Dataset，用于加载数据集
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator, FundamentalSuccessRateCalculator  # 动态阈值成功率计算器
from evaluation.tools.text_editor import TruncatePromptTextEditor, WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution, GPTParaphraser, DipperParaphraser, BackTranslationTextEditor
from evaluation.tools.text_quality_analyzer import PPLCalculator, LogDiversityAnalyzer, BLEUCalculator, PassOrNotJudger, GPTTextDiscriminator
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType  # 导入检测流水线相关类，用于水印和非水印文本检测
from evaluation.pipelines.quality_analysis import DirectTextQualityAnalysisPipeline, QualityPipelineReturnType, ReferencedTextQualityAnalysisPipeline, ExternalDiscriminatorTextQualityAnalysisPipeline
from utils.transformers_config import TransformersConfig  # 导入TransformersConfig类，配置transformer模型
from utils.utils import load_config_file

# 判断是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(device) # 输出设备信息

def assess(algorithm_name, labels, rules, target_fpr, attack_name, metric):
    
    # 加载处理过的C4数据集 /home/lihe/MarkLLM/dataset/wmt16_de_en/validation.jsonl
    global_my_dataset = C4Dataset('dataset/c4/processed_c4.json')
    # global_my_dataset = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl')
    # global_my_dataset = HumanEvalDataset('dataset/human_eval/test.jsonl')
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
    # 获取配置文件
    algorithm_config=f'config/{algorithm_name}.json'
    # 配置文件
    config_dict = load_config_file(algorithm_config)
    # 加载指定的水印算法，使用配置文件
    my_watermark = AutoWatermark.load(f'{algorithm_name}',  # 算法名称
                                    algorithm_config=algorithm_config,  # 对应的算法配置文件
                                    transformers_config=transformers_config)  # 上述配置的transformer模型
    
    if attack_name == 'Word-D':
        attack = WordDeletion(ratio=0.5) # 0.3
    elif attack_name == 'Word-S':
        attack = SynonymSubstitution(ratio=0.5)
    elif attack_name == 'Word-S(Context)':
        attack = ContextAwareSynonymSubstitution(ratio=0.5,
                                                 tokenizer=BertTokenizer.from_pretrained('./models/compositional-bert-large-uncased'),
                                                 model=BertForMaskedLM.from_pretrained('./models/compositional-bert-large-uncased').to(device))
    elif attack_name == 'Doc-P(GPT-3.5)':
        attack = GPTParaphraser(openai_model='gpt-3.5-turbo',
                                prompt='Please rewrite the following text: ')
    elif attack_name == 'Doc-P(Dipper)':
        attack = DipperParaphraser(tokenizer=T5Tokenizer.from_pretrained('./models/t5-v1_1-xxl/'),
                                   model=T5ForConditionalGeneration.from_pretrained('./models/dipper-paraphraser-xxl/', device_map='auto'),
                                   lex_diversity=60, order_diversity=0, sent_interval=1, 
                                   max_new_tokens=100, do_sample=True, top_p=0.75, top_k=None)
    elif attack_name == 'Translation':
        attack = BackTranslationTextEditor(translate_to_intermediary = Translator(from_lang="en", to_lang="zh").translate,
                                           translate_to_source = Translator(from_lang="zh", to_lang="en").translate)
        
        
    if metric == 'PPL':
        my_dataset = C4Dataset('dataset/c4/processed_c4.json')
        pipeline = DirectTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                     watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                     unwatermarked_text_editor_list=[],
                                                     analyzer=PPLCalculator(model=LlamaForCausalLM.from_pretrained("./models/llama-7b-hf", device_map='auto'),
                                                                            tokenizer=LlamaTokenizer.from_pretrained("./models/llama-7b-hf", legacy=False),
                                                                            device=device),
                                                     unwatermarked_text_source='generated', show_progress=True, 
                                                     return_type=QualityPipelineReturnType.MEAN_SCORES)
        transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("./models/opt-1.3b").to(device),
                                                 tokenizer=AutoTokenizer.from_pretrained("./models/opt-1.3b"),                                                
                                                 vocab_size=50272,
                                                 device=device,
                                                 max_new_tokens=200,
                                                 min_length=200,
                                                 do_sample=True,
                                                 no_repeat_ngram_size=4)

    elif metric == 'Log Diversity':
        my_dataset = C4Dataset('dataset/c4/processed_c4.json')
        pipeline = DirectTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                     watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                     unwatermarked_text_editor_list=[],
                                                     analyzer=LogDiversityAnalyzer(),
                                                     # unwatermarked_text_source='natural', show_progress=True, 
                                                     unwatermarked_text_source='generated', show_progress=True, 
                                                     return_type=QualityPipelineReturnType.MEAN_SCORES)
        transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("./models/opt-1.3b").to(device),
                                                 tokenizer=AutoTokenizer.from_pretrained("./models/opt-1.3b"),
                                                 vocab_size=50272,
                                                 device=device,
                                                 max_new_tokens=200,
                                                 min_length=230,
                                                 do_sample=True,
                                                 no_repeat_ngram_size=4)
    elif metric == 'BLEU':
        my_dataset = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl')
        tokenizer = AutoTokenizer.from_pretrained("./nllb-200-distilled-600M/", src_lang="deu_Latn")
        transformers_config = TransformersConfig(model=AutoModelForSeq2SeqLM.from_pretrained("./nllb-200-distilled-600M/").to(device),
                                                 tokenizer=tokenizer,
                                                 device=device,
                                                 vocab_size=256206,
                                                 forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
        pipeline = ReferencedTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                         watermarked_text_editor_list=[],
                                                         unwatermarked_text_editor_list=[],
                                                         analyzer=BLEUCalculator(),
                                                         unwatermarked_text_source='generated', show_progress=True, 
                                                         return_type=QualityPipelineReturnType.MEAN_SCORES)
    elif metric == 'pass@1':
        my_dataset = HumanEvalDataset('dataset/human_eval/test.jsonl')
        transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("/data2/shared_model/starcoder/", device_map='auto'),
                                                 tokenizer=AutoTokenizer.from_pretrained("/data2/shared_model/starcoder/"),
                                                 device=device,
                                                 min_length=200,
                                                 max_length=400)
        pipeline = ReferencedTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                         watermarked_text_editor_list=[TruncateTaskTextEditor(),CodeGenerationTextEditor()],
                                                         unwatermarked_text_editor_list=[TruncateTaskTextEditor(), CodeGenerationTextEditor()],
                                                         analyzer=PassOrNotJudger(),
                                                         unwatermarked_text_source='generated', show_progress=True, 
                                                         return_type=QualityPipelineReturnType.MEAN_SCORES)
    elif metric == 'GPT-4 Judge':
        my_dataset = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl')
        tokenizer = AutoTokenizer.from_pretrained("./models/nllb-200-distilled-600M/", src_lang="deu_Latn")
        transformers_config = TransformersConfig(model=AutoModelForSeq2SeqLM.from_pretrained("./models/nllb-200-distilled-600M/").to(device),
                                                 tokenizer=tokenizer,
                                                 device=device,
                                                 vocab_size=256206,
                                                 forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
        pipeline = ExternalDiscriminatorTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                                    watermarked_text_editor_list=[],
                                                                    unwatermarked_text_editor_list=[],
                                                                    analyzer=GPTTextDiscriminator(openai_model='gpt-4',
                                                                                                  task_description='Translate the following German text to English.'),
                                                                    unwatermarked_text_source='generated', show_progress=True, 
                                                                    return_type=QualityPipelineReturnType.MEAN_SCORES)
    else:
        raise ValueError('Invalid metric')
    
    # 创建一个用于检测水印文本的流水线 # 使用C4数据集  # 使用文本编辑器截断提示词 # 显示进度 # 返回类型为分数
    pipline1 = WatermarkedTextDetectionPipeline(dataset=global_my_dataset, text_editor_list=[TruncatePromptTextEditor()],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES) 
    # 创建一个用于检测非水印文本的流水线 # 使用相同的数据集 # 不使用文本编辑器 # 显示进度 # 返回类型为分数
    pipline2 = UnWatermarkedTextDetectionPipeline(dataset=global_my_dataset, text_editor_list=[],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES, text_source_mode = "generated")   # IS_WATERMARKED
    # 创建一个用于检测水印文本的有攻击的流水线 # 使用C4数据集  # 使用文本编辑器截断提示词 # 显示进度 # 返回类型为分数
    pipline3 = WatermarkedTextDetectionPipeline(dataset=global_my_dataset, text_editor_list=[TruncatePromptTextEditor(), attack],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES) 

    
    print("\n无攻击的水印pipline:\n水印算法名称:", algorithm_name, ", α:", config_dict['α'], ", β:", config_dict['β'], ", theta:", config_dict['theta'])
    watermark_evaluate = pipline1.evaluate(my_watermark)
    print("\n无水印的pipline:\n水印算法名称:", algorithm_name, ", α:", config_dict['α'], ", β:", config_dict['β'], ", theta:", config_dict['theta'])
    unwatermark_evaluate = pipline2.evaluate(my_watermark)
    print("\n有攻击的水印pipline:", "攻击名称", attack_name, "\n水印算法名称:", algorithm_name, ", α:", config_dict['α'], ", β:", config_dict['β'], ", theta:", config_dict['theta'])
    attack_watermark_evaluate = pipline3.evaluate(my_watermark)

    # 动态阈值成功率计算器，用于评估流水线的成功率 # 评估指标 # 评估规则 # 目标的错误接受率（False Positive Rate）
    calculator = DynamicThresholdSuccessRateCalculator(labels=labels, rule=rules, target_fpr=target_fpr,reverse=False)
    # 静态阈值
    # calculator = FundamentalSuccessRateCalculator(labels=labels)
    
    # 打印算法名称和参数
    print("\n检测成功率的信息如下：", "\n水印算法名称:", algorithm_name, ", α:", config_dict['α'], ", β:", config_dict['β'], ", theta:", config_dict['theta'])
    # 计算并输出水印和非水印文本的成功率
    print(calculator.calculate(watermark_evaluate, unwatermark_evaluate))
    # 打印攻击算法名称和参数
    print("\n抗攻击鲁棒性的信息如下：", "攻击名称", attack_name, "\n水印算法名称:", algorithm_name, ", α:", config_dict['α'], ", β:", config_dict['β'], ", theta:", config_dict['theta'])
    # 计算并输出攻击后的水印和非水印文本的成功率
    print(calculator.calculate(attack_watermark_evaluate, unwatermark_evaluate))
    
    print("\n文本质量的信息如下:", ", 评估指标", metric, "\n水印算法名称:", algorithm_name, ", α:", config_dict['α'], ", β:", config_dict['β'], ", theta:", config_dict['theta'])
    # 计算并输出水印和非水印文本的文本质量
    print(pipeline.evaluate(my_watermark))
    

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
    # 添加命令行参数：攻击名称
    parser.add_argument('--attack_name', type=str, default='Word-D')
    # 添加命令行参数：评估文本质量的方法
    parser.add_argument('--metric', type=str, default='PPL')
    # 解析命令行参数
    args = parser.parse_args()

    # 代码入口
    assess(args.algorithm, args.labels, args.rules, args.target_fpr, args.attack_name, args.metric)