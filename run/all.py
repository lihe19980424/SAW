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
from evaluation.tools.text_editor import TruncateTaskTextEditor
from evaluation.tools.text_editor import CodeGenerationTextEditor
from evaluation.tools.text_quality_analyzer import PPLCalculator, LogDiversityAnalyzer, BLEUCalculator, PassOrNotJudger, GPTTextDiscriminator
from evaluation.pipelines.detection_all import WatermarkedTextDetectionPipeline, DetectionPipelineReturnType, WatermarkDetectionPipeline  # 导入检测流水线相关类，用于水印和非水印文本检测
from evaluation.pipelines.quality_analysis import DirectTextQualityAnalysisPipeline, QualityPipelineReturnType, ReferencedTextQualityAnalysisPipeline, ExternalDiscriminatorTextQualityAnalysisPipeline
from utils.transformers_config import TransformersConfig  # 导入TransformersConfig类，配置transformer模型
from utils.utils import load_config_file
import time
from tqdm import tqdm

# 判断是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device) # 输出设备信息

def assess(args):
    # 获取配置文件
    algorithm_config=f'config/{args.algorithm}.json'
    # 配置文件
    config_dict = load_config_file(algorithm_config)
    if args.algorithm == "Rethinking_uni" or args.algorithm == "Rethinking_gaosi":
        config_dict["beta"] = args.beta
        config_dict["std"] = args.std
    
    config_dict["temperature_inner"] = args.temperature_inner
    
    content_to_print = "algorithm: " + str(args.algorithm) + " dataset: " + str(args.dataset) + " temperature_inner: " + str(args.temperature_inner)+ " max_new_tokens: " + str(args.max_new_tokens) + " min_length: " + str(args.min_length) +" data_lines: " + str(args.data_lines) 
    print(content_to_print)
    if args.dataset =='c4':
        dataset_path = C4Dataset('dataset/c4/processed_c4.json', args.data_lines)
        model_path = AutoModelForCausalLM.from_pretrained("./models/opt-1.3b").to(device)
        tokenizer_path = AutoTokenizer.from_pretrained("./models/opt-1.3b")
        my_vocab_size = 50272
        # 配置transformer模型和tokenizer
        transformers_config = TransformersConfig(model=model_path,
                                                # model=LlamaForCausalLM.from_pretrained("./models/llama-7b-hf").to(device),
                                                # model=T5ForConditionalGeneration.from_pretrained("./models/t5-v1_1-xxl").to(device),
                                                tokenizer=tokenizer_path,
                                                # tokenizer=LlamaTokenizer.from_pretrained("./models/llama-7b-hf"),
                                                # tokenizer=T5Tokenizer.from_pretrained("./models/t5-v1_1-xxl"),
                                                vocab_size=my_vocab_size,
                                                # vocab_size=32000,
                                                # vocab_size=32128,
                                                device=device,
                                                max_new_tokens=args.max_new_tokens,
                                                min_length=args.min_length,
                                                do_sample=True,
                                                no_repeat_ngram_size=4)
    elif args.dataset =='wmt16_de_en':
        dataset_path = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl', args.data_lines)
        tokenizer=AutoTokenizer.from_pretrained("/home/lihe/MarkLLM/models/nllb-200-distilled-600M/", src_lang="deu_Latn")
        transformers_config = TransformersConfig(model=AutoModelForSeq2SeqLM.from_pretrained("/home/lihe/MarkLLM/models/nllb-200-distilled-600M/").to(device),
                                                tokenizer=tokenizer,
                                                device=device,
                                                vocab_size=256206,
                                                # forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"]
                                                # forced_bos_token_id = tokenizer.encode("eng_Latn")[0]
                                                forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
                                                )
        # transformers_config = TransformersConfig(model=LlamaForCausalLM.from_pretrained("./models/llama-7b-hf").to(device),
        #                                 # model=T5ForConditionalGeneration.from_pretrained("./models/t5-v1_1-xxl").to(device),
        #                                 tokenizer=LlamaTokenizer.from_pretrained("./models/llama-7b-hf", legacy=False),
        #                                 # tokenizer=T5Tokenizer.from_pretrained("./models/t5-v1_1-xxl"),
        #                                 vocab_size=32000,
        #                                 # vocab_size=32128,
        #                                 device=device)
    else:
        dataset_path = HumanEvalDataset('dataset/human_eval/test.jsonl', args.data_lines)
        tokenizer= AutoTokenizer.from_pretrained("/home/lihe/MarkLLM/models/starcoder")
        transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("/home/lihe/MarkLLM/models/starcoder", device_map='auto'),
                                             tokenizer=tokenizer,
                                             device=device,
                                             min_length=200,
                                             max_length=400,
                                             pad_token_id=0, 
                                             eos_token_id=0)


    if args.algorithm == "DIP" or args.algorithm == 'EXP' or args.algorithm == 'TS' or args.algorithm == 'SynthID' or args.algorithm == 'SIR':
        # 加载指定的水印算法，使用配置文件
        my_watermark = AutoWatermark.load(f'{args.algorithm}',  # 算法名称
                                        algorithm_config=f'config/{args.algorithm}.json',  # 对应的算法配置文件algorithm_config
                                        transformers_config=transformers_config)  # 上述配置的transformer模型
    else:
        # 加载指定的水印算法，使用配置文件
        my_watermark = AutoWatermark.load(f'{args.algorithm}',  # 算法名称
                                        algorithm_config= config_dict,  # 对应的算法配置文件algorithm_config
                                        transformers_config=transformers_config)  # 上述配置的transformer模型
    
    if args.algorithm == 'EXP':
        # 用于无攻击best动态阈值成功率计算器，用于评估流水线的成功率 # 评估指标 # 评估规则 # 目标的错误接受率（False Positive Rate）
        calculator_unattack = DynamicThresholdSuccessRateCalculator(labels=args.labels, rule=args.rules, target_fpr=args.target_fpr,reverse=True)
        # 用于各种攻击动态阈值成功率计算器，用于评估流水线的成功率 # 评估指标 # 评估规则 # 目标的错误接受率（False Positive Rate）
        calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'], rule=args.rules, target_fpr=args.target_fpr,reverse=True)
        # fpr=10%
        calculator_fpr_10 = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'], rule="target_fpr", target_fpr=0.1, reverse=True)
        # fpr=1%
        calculator_fpr_1 = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'], rule="target_fpr", target_fpr=0.01, reverse=True)
    else:    
        # 用于无攻击best动态阈值成功率计算器，用于评估流水线的成功率 # 评估指标 # 评估规则 # 目标的错误接受率（False Positive Rate）
        calculator_unattack = DynamicThresholdSuccessRateCalculator(labels=args.labels, rule=args.rules, target_fpr=args.target_fpr,reverse=False)
        # 用于各种攻击动态阈值成功率计算器，用于评估流水线的成功率 # 评估指标 # 评估规则 # 目标的错误接受率（False Positive Rate）
        calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'], rule=args.rules, target_fpr=args.target_fpr,reverse=False)
        # fpr=10%
        calculator_fpr_10 = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'], rule="target_fpr", target_fpr=0.1, reverse=False)
        # fpr=1%
        calculator_fpr_1 = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'], rule="target_fpr", target_fpr=0.01, reverse=False)

    # 创建一个用于检测文本的流水线  # 使用文本编辑器截断提示词 # 显示进度 # 返回类型为分数 # WatermarkedTextDetectionPipeline
    pipline_watermark = WatermarkDetectionPipeline(dataset=dataset_path, text_editor_list=[TruncatePromptTextEditor()],
                                                        unwatermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                        show_progress=True, return_type=DetectionPipelineReturnType.FULL, device=device) 
    
    # print("\n水印算法名称:", algorithm_name,"最小生成长度：230","z_threshold: ", config_dict['z_threshold'])
    # print("topk:", config_dict['topk'], "α:", config_dict['α'], ", beta:", config_dict['beta'], ", mean:", config_dict['mean'], ", std:", config_dict['std'])
    unwatermark_evaluate, watermark_evaluate, attack_watermark_evaluate_Word_D_1, attack_watermark_evaluate_Word_D_3, attack_watermark_evaluate_Word_D_5, attack_watermark_evaluate_Word_D_7, attack_watermark_evaluate_Word_S_1, attack_watermark_evaluate_Word_S_3, attack_watermark_evaluate_Word_S_5, attack_watermark_evaluate_Word_S_7, attack_watermark_evaluate_doc_P_dipper, ppl_evaluation_result, logdiversity_evaluation_result, BLEU_evaluation_result, GPT_evaluation_result, Pass_evaluation_result, execution_time_unwatermarked_200_sum, execution_time_watermarked_200_sum, execution_time_unwatermarked_200_avg, execution_time_watermarked_200_avg = pipline_watermark.evaluate(my_watermark, args.dataset, args.data_lines)
    print("平均生成每一条文本需要的时间/秒：", execution_time_watermarked_200_avg)
    
    # 打印算法名称和参数
    # print("\n无攻击下检测成功率(动态阈值)：")
    # best计算并输出水印和非水印文本的成功率  
    result_unattack, threshold_unattack = calculator_unattack.calculate([float(result.detect_result['score']) for result in watermark_evaluate], [float(result.detect_result['score']) for result in unwatermark_evaluate]) 
    # print(result_unattack)
    # 打印算法名称和参数
    # print("\n无攻击下检测成功率(静态阈值FPR=0.01),")
    # frp=10%计算并输出水印和非水印文本的成功率
    result_unattack_fpr_10, threshold_unattack_fpr_10 = calculator_fpr_10.calculate([float(result.detect_result['score']) for result in watermark_evaluate], [float(result.detect_result['score']) for result in unwatermark_evaluate]) 
    # print(result_unattack_fpr_10)
    # frp=1%计算并输出水印和非水印文本的成功率
    result_unattack_fpr_1, threshold_unattack_fpr_1 = calculator_fpr_1.calculate([float(result.detect_result['score']) for result in watermark_evaluate], [float(result.detect_result['score']) for result in unwatermark_evaluate]) 
    # print(result_unattack_fpr_1)
    
    if args.dataset =='c4':
        if args.data_lines ==200:
            # 打印攻击算法名称和参数
            # print("\nattack_Word-D_1攻击下检测成功率(动态阈值)：")
            # 计算并输出攻击后的水印和非水印文本的成功率
            result_attack_Word_D_1, threshold_attack_Word_D_1 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_D_1], [float(result.detect_result['score']) for result in unwatermark_evaluate])
            # print(result_attack_Word_D_1)
            
            # 打印攻击算法名称和参数
            # print("\nattack_Word-D攻击下检测成功率(动态阈值)：")
            # 计算并输出攻击后的水印和非水印文本的成功率
            result_attack_Word_D_3, threshold_attack_Word_D_3 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_D_3], [float(result.detect_result['score']) for result in unwatermark_evaluate])
            # print(result_attack_Word_D)
            
            # 打印攻击算法名称和参数
            # print("\nattack_Word-D_1攻击下检测成功率(动态阈值)：")
            # 计算并输出攻击后的水印和非水印文本的成功率
            result_attack_Word_D_7, threshold_attack_Word_D_7 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_D_7], [float(result.detect_result['score']) for result in unwatermark_evaluate])
            # print(result_attack_Word_D_1)
        
        # 打印攻击算法名称和参数
        # print("\nattack_Word-D_1攻击下检测成功率(动态阈值)：")
        # 计算并输出攻击后的水印和非水印文本的成功率
        result_attack_Word_D_5, threshold_attack_Word_D_5 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_D_5], [float(result.detect_result['score']) for result in unwatermark_evaluate])
        # print(result_attack_Word_D_1)
        
        if args.data_lines ==200:
            # 打印攻击算法名称和参数
            # print("\nattack_Word-S攻击下检测成功率(动态阈值)：")
            # 计算并输出攻击后的水印和非水印文本的成功率
            result_attack_Word_S_1, threshold_attack_Word_S_1 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_S_1], [float(result.detect_result['score']) for result in unwatermark_evaluate])
            # print(result_attack_Word_S)
            
            # 打印攻击算法名称和参数
            # print("\nattack_Word-S攻击下检测成功率(动态阈值)：")
            # 计算并输出攻击后的水印和非水印文本的成功率
            result_attack_Word_S_3, threshold_attack_Word_S_3 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_S_3], [float(result.detect_result['score']) for result in unwatermark_evaluate])
            # print(result_attack_Word_S)
            
            # 打印攻击算法名称和参数
            # print("\nattack_Word-S攻击下检测成功率(动态阈值)：")
            # 计算并输出攻击后的水印和非水印文本的成功率
            result_attack_Word_S_7, threshold_attack_Word_S_7 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_S_7], [float(result.detect_result['score']) for result in unwatermark_evaluate])
            # print(result_attack_Word_S)
    
        # 打印攻击算法名称和参数
        # print("\nattack_Word-S攻击下检测成功率(动态阈值)：")
        # 计算并输出攻击后的水印和非水印文本的成功率
        result_attack_Word_S_5, threshold_attack_Word_S_5 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_S_5], [float(result.detect_result['score']) for result in unwatermark_evaluate])
        # print(result_attack_Word_S)
        
        # 打印攻击算法名称和参数
        # print("\nattack_doc_P_dipper攻击下检测成功率(动态阈值)：")
        # 计算并输出攻击后的水印和非水印文本的成功率
        result_attack_doc_P_dipper, threshold_attack_doc_P_dipper = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_doc_P_dipper], [float(result.detect_result['score']) for result in unwatermark_evaluate])
        # print(result_attack_Word_S) 
        
    elif args.dataset =='wmt16_de_en':
        # 打印攻击算法名称和参数
        # print("\nattack_Word-D_7攻击下检测成功率(动态阈值)：")
        # 计算并输出攻击后的水印和非水印文本的成功率
        result_attack_Word_D_3, threshold_attack_Word_D_3 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_D_3], [float(result.detect_result['score']) for result in unwatermark_evaluate])
        # print(result_attack_Word_D_7)
        
        # 打印攻击算法名称和参数
        # print("\nattack_Word-S_7攻击下检测成功率(动态阈值)：")
        # 计算并输出攻击后的水印和非水印文本的成功率
        result_attack_Word_S_3, threshold_attack_Word_S_3 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_S_3], [float(result.detect_result['score']) for result in unwatermark_evaluate])
        # print(result_attack_Word_S_7)
        
        # 打印攻击算法名称和参数
        # print("\nattack_doc_P_dipper攻击下检测成功率(动态阈值)：")
        # 计算并输出攻击后的水印和非水印文本的成功率
        result_attack_doc_P_dipper, threshold_attack_doc_P_dipper = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_doc_P_dipper], [float(result.detect_result['score']) for result in unwatermark_evaluate])
        # print(result_attack_Word_S) 
    else:
        # 打印攻击算法名称和参数
        # print("\nattack_Word-D_7攻击下检测成功率(动态阈值)：")
        # 计算并输出攻击后的水印和非水印文本的成功率
        result_attack_Word_D_3, threshold_attack_Word_D_3 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_D_3], [float(result.detect_result['score']) for result in unwatermark_evaluate])
        # print(result_attack_Word_D_7)
        
        # 打印攻击算法名称和参数
        # print("\nattack_Word-S_7攻击下检测成功率(动态阈值)：")
        # 计算并输出攻击后的水印和非水印文本的成功率
        result_attack_Word_S_3, threshold_attack_Word_S_3 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_S_3], [float(result.detect_result['score']) for result in unwatermark_evaluate])
        # print(result_attack_Word_S_7)      
        
        # 打印攻击算法名称和参数
        # print("\nattack_doc_P_dipper攻击下检测成功率(动态阈值)：")
        # 计算并输出攻击后的水印和非水印文本的成功率
        result_attack_doc_P_dipper, threshold_attack_doc_P_dipper = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_doc_P_dipper], [float(result.detect_result['score']) for result in unwatermark_evaluate])
        # print(result_attack_Word_S) 
    
    
    # 打印攻击算法名称和参数
    # print("\nattack_Word-S(context)攻击下检测成功率(动态阈值)：")
    # 计算并输出攻击后的水印和非水印文本的成功率
    # result_attack_Word_S_context, threshold_attack_Word_S_context = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_S_context], [float(result.detect_result['score']) for result in unwatermark_evaluate])
    # print(result_attack_Word_S)
    
    
    if args.dataset =='c4':
        # print("\n文本质量的信息如下:", ", 评估指标:", "ppl")
        # 计算并输出水印和非水印文本的文本质量
        result_PPL = {'watermarked': sum([result.watermarked_quality_score for result in ppl_evaluation_result]) / len(ppl_evaluation_result), 
                    'unwatermarked': sum([result.unwatermarked_quality_score for result in ppl_evaluation_result]) / len(ppl_evaluation_result)}
        # print(result_PPL)
        
        # print("\n文本质量的信息如下:", ", 评估指标:", "log")
        # 计算并输出水印和非水印文本的文本质量
        result_Log_Diversity = {'watermarked': sum([result.watermarked_quality_score for result in logdiversity_evaluation_result]) / len(logdiversity_evaluation_result), 
                            'unwatermarked': sum([result.unwatermarked_quality_score for result in logdiversity_evaluation_result]) / len(logdiversity_evaluation_result)}
        # print(result_Log_Diversity)
    elif args.dataset =='wmt16_de_en':
        # print("\n文本质量的信息如下:", ", 评估指标:", "log")
        # 计算并输出水印和非水印文本的文本质量
        result_BLEU = {'watermarked': sum([result.watermarked_quality_score for result in BLEU_evaluation_result]) / len(BLEU_evaluation_result), 
                        'unwatermarked': sum([result.unwatermarked_quality_score for result in BLEU_evaluation_result]) / len(BLEU_evaluation_result)}
        # print(result_BLEU)
    else:
        # print("\n文本质量的信息如下:", ", 评估指标:", "PassOrNotJudger")
        # 计算并输出水印和非水印文本的文本质量
        result_Pass = {'watermarked': sum([result.watermarked_quality_score for result in Pass_evaluation_result]) / len(Pass_evaluation_result), 
                        'unwatermarked': sum([result.unwatermarked_quality_score for result in Pass_evaluation_result]) / len(Pass_evaluation_result)}
        # print(result_Pass)
        
        # print("\n文本质量的信息如下:", ", 评估指标:", "log")
        # 计算并输出水印和非水印文本的文本质量
        # result_GPT = {'result_GPT': sum([result.watermarked_quality_score for result in GPT_evaluation_result]) / len(GPT_evaluation_result)}
        # print(result_GPT)
    

    with open("output_human_eval.txt", "a") as file:
        file.write("\nparameter as follows:\n") 
        file.write(content_to_print) 
        file.write("\n算法的参数信息如下：\n") 
        file.write(str(config_dict)) 
        file.write("\n生成200条无水印文本需要的总时间/秒：") 
        file.write(str(execution_time_unwatermarked_200_sum)) 
        file.write("\n平均生成每一条无水印文本需要的时间/秒：") 
        file.write(str(execution_time_unwatermarked_200_avg)) 
        file.write("\n生成200条有水印文本需要的总时间/秒：") 
        file.write(str(execution_time_watermarked_200_sum)) 
        file.write("\n平均生成每一条有水印文本需要的时间/秒：") 
        file.write(str(execution_time_watermarked_200_avg)) 
        if args.dataset =='c4':
            file.write("\n文本质量PPL指标如下:\n")  
            file.write(str(result_PPL))
            file.write("\n文本质量Log_Diversity指标如下:\n")  
            file.write(str(result_Log_Diversity))  
        elif args.dataset =='wmt16_de_en':
            file.write("\n文本质量BLEU指标如下:\n")  
            file.write(str(result_BLEU)) 
            #file.write("\n文本质量GPT指标如下:\n")  
            #file.write(str(result_GPT)) 
        else:
            file.write("\n文本质量Pass指标如下:\n")  
            file.write(str(result_Pass))
        file.write("\n无攻击(fpr=best)的检测成功率：\n") 
        file.write(str(result_unattack)) 
        file.write("\n无攻击(fpr=best)的最佳z-score阈值是:") 
        file.write(str(threshold_unattack)) 
        file.write("\n无攻击(fpr=10%)的检测成功率：\n") 
        file.write(str(result_unattack_fpr_10)) 
        file.write("\n无攻击(fpr=10%)的最佳z-score阈值是:") 
        file.write(str(threshold_unattack_fpr_10)) 
        file.write("\n无攻击(fpr=1%)的检测成功率：\n") 
        file.write(str(result_unattack_fpr_1)) 
        file.write("\n无攻击(fpr=1%)的最佳z-score阈值是:") 
        file.write(str(threshold_unattack_fpr_1)) 
        if args.dataset =='c4':
            if args.data_lines ==200:
                file.write("\nattack_Word-D_1攻击下检测成功率(动态阈值)：\n") 
                file.write(str(result_attack_Word_D_1)) 
                file.write("\nattack_Word-D_1攻击的最佳z-score阈值是:") 
                file.write(str(threshold_attack_Word_D_1)) 
                file.write("\nattack_Word-D_3攻击下检测成功率(动态阈值)：\n") 
                file.write(str(result_attack_Word_D_3)) 
                file.write("\nattack_Word-D_3攻击的最佳z-score阈值是:") 
                file.write(str(threshold_attack_Word_D_3)) 
                file.write("\nattack_Word-D_7攻击下检测成功率(动态阈值)：\n") 
                file.write(str(result_attack_Word_D_7)) 
                file.write("\nattack_Word-D_7攻击的最佳z-score阈值是:") 
                file.write(str(threshold_attack_Word_D_7)) 
            file.write("\nattack_Word-D_5攻击下检测成功率(动态阈值)：\n") 
            file.write(str(result_attack_Word_D_5)) 
            file.write("\nattack_Word-D_5攻击的最佳z-score阈值是:") 
            file.write(str(threshold_attack_Word_D_5)) 
            if args.data_lines ==200:
                file.write("\nattack_Word-S_1攻击下检测成功率(动态阈值)：\n")
                file.write(str(result_attack_Word_S_1))
                file.write("\nattack_Word-S_1攻击的最佳z-score阈值是:") 
                file.write(str(threshold_attack_Word_S_1)) 
                file.write("\nattack_Word-S_3攻击下检测成功率(动态阈值)：\n")
                file.write(str(result_attack_Word_S_3))
                file.write("\nattack_Word-S_3攻击的最佳z-score阈值是:") 
                file.write(str(threshold_attack_Word_S_3)) 
                file.write("\nattack_Word-S_7攻击下检测成功率(动态阈值)：\n")
                file.write(str(result_attack_Word_S_7))
                file.write("\nattack_Word-S_7攻击的最佳z-score阈值是:") 
                file.write(str(threshold_attack_Word_S_7)) 
            file.write("\nattack_Word-S_5攻击下检测成功率(动态阈值)：\n")
            file.write(str(result_attack_Word_S_5))
            file.write("\nattack_Word-S_5攻击的最佳z-score阈值是:") 
            file.write(str(threshold_attack_Word_S_5))
            
            file.write("\nattack_doc_P_dipper攻击下检测成功率(动态阈值)：\n")
            file.write(str(result_attack_doc_P_dipper))
            file.write("\nattack_doc_P_dipper攻击的最佳z-score阈值是:") 
            file.write(str(threshold_attack_doc_P_dipper))  
        elif args.dataset =='wmt16_de_en':
            file.write("\nattack_Word-D_3攻击下检测成功率(动态阈值)：\n") 
            file.write(str(result_attack_Word_D_3)) 
            file.write("\nattack_Word-D_3攻击的最佳z-score阈值是:") 
            file.write(str(threshold_attack_Word_D_3)) 
            
            file.write("\nattack_Word-S_3攻击下检测成功率(动态阈值)：\n")
            file.write(str(result_attack_Word_S_3))
            file.write("\nattack_Word-S_3攻击的最佳z-score阈值是:") 
            file.write(str(threshold_attack_Word_S_3)) 
            
            file.write("\nattack_doc_P_dipper攻击下检测成功率(动态阈值)：\n")
            file.write(str(result_attack_doc_P_dipper))
            file.write("\nattack_doc_P_dipper攻击的最佳z-score阈值是:") 
            file.write(str(threshold_attack_doc_P_dipper)) 
        else:
            file.write("\nattack_Word-D_3攻击下检测成功率(动态阈值)：\n") 
            file.write(str(result_attack_Word_D_3)) 
            file.write("\nattack_Word-D_3攻击的最佳z-score阈值是:") 
            file.write(str(threshold_attack_Word_D_3)) 
            
            file.write("\nattack_Word-S_3攻击下检测成功率(动态阈值)：\n") 
            file.write(str(result_attack_Word_S_3)) 
            file.write("\nattack_Word-S_3攻击的最佳z-score阈值是:") 
            file.write(str(threshold_attack_Word_S_3)) 
            
            file.write("\nattack_doc_P_dipper攻击下检测成功率(动态阈值)：\n")
            file.write(str(result_attack_doc_P_dipper))
            file.write("\nattack_doc_P_dipper攻击的最佳z-score阈值是:") 
            file.write(str(threshold_attack_doc_P_dipper)) 
                
        # file.write("\nattack_Word-S(context)攻击下检测成功率(动态阈值)：\n")
        # file.write(str(result_attack_Word_S_context))
        # file.write("\nattack_Word-S(context)攻击的最佳z-score阈值是:") 
        # file.write(str(threshold_attack_Word_S_context)) 
        
        file.write("\n==========分割线==========\n")

# 主函数入口
if __name__ == '__main__':
    import argparse  # 导入argparse模块，用于处理命令行参数
    # 创建参数解析器 # 获取配置文件
    # algorithm_config=f'config/{algorithm_name}.json'
    parser = argparse.ArgumentParser()
    # 添加命令行参数：算法名称
    parser.add_argument('--algorithm', type=str, default='KGW')  # 水印算法名称 Rethinking
    parser.add_argument('--dataset', type=str, default='human_eval')  # wmt16_de_en    human_eval  c4
    parser.add_argument('--max_new_tokens', type=int, default=400)
    parser.add_argument('--min_length', type=int, default=200)
    parser.add_argument('--data_lines', type=int, default=100)
    # 添加命令行参数：评估指标
    parser.add_argument('--labels', nargs='+', default=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'])
    # 添加命令行参数：评估规则
    parser.add_argument('--rules', type=str, default='best') #target_fpr
    # 添加命令行参数：目标错误接受率
    parser.add_argument('--target_fpr', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default='1.0')
    parser.add_argument('--std', type=float, default='0.09')
    parser.add_argument('--temperature_inner', type=float, default='1.0')
    # 解析命令行参数
    args = parser.parse_args()
    # 代码入口
    assess(args)