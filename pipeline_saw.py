import torch
from translate import Translator
from transformers import AutoModel, BertForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM, LlamaForCausalLM, T5ForConditionalGeneration, GPT2Model, GPT2LMHeadModel, Qwen2ForCausalLM, BertTokenizer, AutoTokenizer, LlamaTokenizer, T5Tokenizer,  GPT2Tokenizer, LlamaTokenizerFast, Qwen2Tokenizer # 导入transformers库的模型和tokenizer类 Qwen2_5OmniModel, Qwen2_5OmniProcessor
# from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
# from qwen_omni_utils import process_mm_info
from watermark.auto_watermark import AutoWatermark  # 导入AutoWatermark类，用于加载水印算法
from evaluation.dataset import C4Dataset, WMT16DE_ENDataset, HumanEvalDataset, ELI5Dataset, MULTINEWSDataset, ROCSTORIESDataset, Flickr30kDataset, CNNDAILYMAILDataset  # 从evaluation模块导入C4Dataset，用于加载数据集
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

from transformers import logging
logging.set_verbosity_error()

from watermark.base import BaseWatermark, BaseConfig

from transformers import BlipProcessor, BlipForConditionalGeneration

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(device)

def assess(args):
    algorithm_config=f'config/{args.algorithm}.json'
    config_dict = load_config_file(algorithm_config)
    if args.algorithm == "SMOOTH":
        config_dict["alpha"] = args.alpha
        config_dict["gamma"] = args.gamma
        config_dict["epsilon"] = args.epsilon
        config_dict["delta"] = args.delta
        config_dict["eta"] = args.eta
        config_dict["z_threshold"] = args.z_threshold
        config_dict["resilience"] = args.resilience
        config_dict["min_length"] = args.min_length
        config_dict["fixed_pos"] = args.fixed_pos
        config_dict["entropy_threshold"] = args.entropy_threshold
    elif args.algorithm == "Rethinking_uni" or args.algorithm == "Rethinking_gaosi" or args.algorithm == "SAW":
        config_dict["beta"] = args.beta
        config_dict["std"] = args.std
        config_dict["noise"] = args.noise

    # config_dict["temperature_inner"] = args.temperature_inner
    
    config_algorithm = "algorithm: " + str(args.algorithm) + " dataset: " + str(args.dataset) + " model: " + str(args.model) + " max_new_tokens: " + str(args.max_new_tokens) + " min_length: " + str(args.min_length) +" data_lines: " + str(args.data_lines) 
    
    model_dipper = T5ForConditionalGeneration.from_pretrained('./models/paraphrase-dipper-no-ctx').to(device) # , device_map='auto'  .to(device)
    tokenizer_dipper = T5Tokenizer.from_pretrained('./models/t5-large', legacy=False)

    model_Bert = BertForMaskedLM.from_pretrained('./models/roberta-base').to(device) # , device_map='auto'  .to(device)
    tokenizer_Bert = AutoTokenizer.from_pretrained('./models/roberta-base', legacy=False) 

    print(config_algorithm)
    print(config_dict)
    if args.dataset == 'c4':
        dataset_path = C4Dataset('./dataset/c4/processed_c4.json', args.data_lines)
    elif args.dataset == 'eli5':
        dataset_path = ELI5Dataset('./dataset/eli5/data/train/eli5_split_csv_0.jsonl', args.data_lines)
    elif args.dataset == 'multinews':   
        dataset_path = MULTINEWSDataset('./dataset/muti_news_dataset_edited/multinews-testdata-edited.jsonl', args.data_lines)
    elif args.dataset == 'rocstories':  
        dataset_path = ROCSTORIESDataset('./dataset/ROCStories/test.jsonl', args.data_lines)
    elif args.dataset == 'wmt16_de_en':
        dataset_path = WMT16DE_ENDataset('./dataset/wmt16_de_en/validation.jsonl', args.data_lines)
    elif args.dataset == 'human_eval':
        dataset_path = HumanEvalDataset('./dataset/human_eval/test.jsonl', args.data_lines)
    elif args.dataset == 'flickr30k':
        dataset_path = Flickr30kDataset('./dataset/flickr30k', args.data_lines)
    elif args.dataset == 'cnn_daily_mail':
        dataset_path = CNNDAILYMAILDataset('./dataset/cnn_daily_mail/test.json', args.data_lines)

    if args.model =='opt-1.3b':
        model_path = AutoModelForCausalLM.from_pretrained("./models/opt-1.3b").to(device) # , device_map='auto'  .to(device)
        tokenizer_path = AutoTokenizer.from_pretrained("./models/opt-1.3b", legacy=False)
        # my_vocab_size = 50272
        my_vocab_size = model_path.get_output_embeddings().weight.shape[0]
    elif args.model =='llama-7b-hf':
        model_path = LlamaForCausalLM.from_pretrained("./models/llama-7b-hf").to(device) # device_map='auto'  .to(device)
        tokenizer_path = LlamaTokenizer.from_pretrained("./models/llama-7b-hf", legacy=False) 
        # my_vocab_size = 32000
        my_vocab_size = model_path.get_output_embeddings().weight.shape[0]
    elif args.model =='Llama-3-8B-Instruct':      
        model_path = AutoModelForCausalLM.from_pretrained("./models/Meta-Llama-3-8B-Instruct").to(device)  # device_map='auto'  .to(device)
        tokenizer_path = AutoTokenizer.from_pretrained("./models/Meta-Llama-3-8B-Instruct", legacy=False) 
        # my_vocab_size = 128256
        my_vocab_size = model_path.get_output_embeddings().weight.shape[0]
    elif args.model =='compositional-bert-large-uncased':
        model_path = AutoModelForCausalLM.from_pretrained("./models/compositional-bert-large-uncased", device_map='auto')
        tokenizer_path = AutoTokenizer.from_pretrained("./models/compositional-bert-large-uncased", legacy=False)
        # my_vocab_size = 30522
        my_vocab_size = model_path.get_output_embeddings().weight.shape[0]
    elif args.model =='roberta-base':
        model_path = AutoModelForCausalLM.from_pretrained("./models/roberta-base", device_map='auto')
        tokenizer_path = AutoTokenizer.from_pretrained("./models/roberta-base", legacy=False)
        # my_vocab_size = 50265
        my_vocab_size = model_path.get_output_embeddings().weight.shape[0]
    elif args.model =='t5-v1_1-xxl':
        model_path = T5ForConditionalGeneration.from_pretrained('./models/dipper-paraphraser-xxl',device_map='auto') # ,device_map='auto'  .to(device)
        tokenizer_path = T5Tokenizer.from_pretrained('./models/t5-v1_1-xxl', legacy=False)
        # my_vocab_size = 32128
        my_vocab_size = model_path.get_output_embeddings().weight.shape[0]
    elif args.model =='chatglm2-6b':
        model_path = AutoModelForCausalLM.from_pretrained("./models/chatglm2-6b", trust_remote_code=True).to(device)
        tokenizer_path = AutoTokenizer.from_pretrained("./models/chatglm2-6b", trust_remote_code=True)
        # my_vocab_size = 65024
        my_vocab_size = model_path.get_output_embeddings().weight.shape[0]
    elif args.model =='gpt2-xl':
        model_path = GPT2LMHeadModel.from_pretrained('./models/gpt2-xl').to(device)
        tokenizer_path = GPT2Tokenizer.from_pretrained('./models/gpt2-xl')
        # my_vocab_size = 50257
        my_vocab_size = model_path.get_output_embeddings().weight.shape[0]
    elif args.model =='DeepSeek-R1-Distill-Qwen-7B':
        model_path = Qwen2ForCausalLM.from_pretrained('./models/DeepSeek-R1-Distill-Qwen-7B').to(device)
        tokenizer_path = LlamaTokenizerFast.from_pretrained('./models/DeepSeek-R1-Distill-Qwen-7B')
        # my_vocab_size = 152064
        my_vocab_size = model_path.get_output_embeddings().weight.shape[0]
    elif args.model =='Qwen2.5-Omni-7B':     
        # model_path = Qwen2_5OmniForConditionalGeneration.from_pretrained('./models/Qwen2.5-Omni-7B').to(device)
        model_path = AutoModelForCausalLM.from_pretrained('./models/Qwen2.5-Omni-7B',torch_dtype=torch.float16,device_map="auto")
        # tokenizer_path = Qwen2_5OmniProcessor.from_pretrained("./models/Qwen2.5-Omni-7B")
        tokenizer_path = AutoModelForCausalLM.AutoTokenizer.from_pretrained("./models/Qwen2.5-Omni-7B")
        # my_vocab_size = 152064
        # my_vocab_size = tokenizer_path.vocab_size
        my_vocab_size = model_path.get_output_embeddings().weight.shape[0]
    elif args.model =='Qwen2.5-0.5B':     
        model_path = Qwen2ForCausalLM.from_pretrained('./models/Qwen2.5-0.5B').to(device)  # , device_map='auto'
        tokenizer_path = Qwen2Tokenizer.from_pretrained("./models/Qwen2.5-0.5B")
        # my_vocab_size = 151936
        my_vocab_size = model_path.get_output_embeddings().weight.shape[0]
    elif args.model =='Qwen2.5-7B':     
        model_path = Qwen2ForCausalLM.from_pretrained('./models/Qwen2.5-7B').to(device)  # , device_map='auto'
        tokenizer_path = Qwen2Tokenizer.from_pretrained("./models/Qwen2.5-7B")
        # my_vocab_size = tokenizer_path.vocab_size
        my_vocab_size = model_path.get_output_embeddings().weight.shape[0]
    elif args.model =='nllb-200-distilled-600M':      
        model_path = AutoModelForSeq2SeqLM.from_pretrained("./models/nllb-200-distilled-600M/").to(device)  # , device_map='auto'   .to(device)
        tokenizer_path=AutoTokenizer.from_pretrained("./models/nllb-200-distilled-600M/", src_lang="deu_Latn")
        # my_vocab_size = 256206
        my_vocab_size = model_path.get_output_embeddings().weight.shape[0]
    elif args.model =='starcoder':
        model_path = AutoModelForCausalLM.from_pretrained("./models/starcoder/").to(device)
        tokenizer_path= AutoTokenizer.from_pretrained("./models/starcoder/")
    elif args.model =='BLIP':
        model_path = BlipForConditionalGeneration.from_pretrained("models/BLIP").to(device)
        tokenizer_path = BlipProcessor.from_pretrained("models/BLIP", use_fast=True)
        # my_vocab_size = 30524  
        my_vocab_size = model_path.get_output_embeddings().weight.shape[0]

        
    if args.model =='nllb-200-distilled-600M':
        transformers_config = TransformersConfig(model=model_path,
                                                tokenizer=tokenizer_path,
                                                device=device,
                                                vocab_size=my_vocab_size,
                                                # forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"]
                                                # forced_bos_token_id = tokenizer.encode("eng_Latn")[0]
                                                forced_bos_token_id = tokenizer_path.convert_tokens_to_ids("eng_Latn"),
                                                do_sample=False,
                                                )
    elif args.model =='starcoder':
        transformers_config = TransformersConfig(
            model=model_path,
            tokenizer=tokenizer_path,
            device=device,
            # min_length=200,
            # max_length=400,
            do_sample=True
        )
    elif args.model =='BLIP':
        transformers_config = TransformersConfig(
            model=model_path,
            tokenizer=tokenizer_path,
            device=device,
            vocab_size=my_vocab_size,
            max_new_tokens=args.max_new_tokens,
            min_length=args.min_length,
            do_sample=True,
            no_repeat_ngram_size=4,
            num_beams=5
        ) 
    elif args.model =='Qwen2.5-7B':
        transformers_config = TransformersConfig(
            model=model_path,
            tokenizer=tokenizer_path,
            vocab_size=152064,
            device=device,
            max_new_tokens=args.max_new_tokens,
            min_length=args.min_length,
            do_sample=True,
            no_repeat_ngram_size=4
        )     
    else:        
        transformers_config = TransformersConfig(model=model_path,
                                                tokenizer=tokenizer_path,
                                                vocab_size=my_vocab_size,
                                                device=device,
                                                max_new_tokens=args.max_new_tokens,
                                                min_length=args.min_length,
                                                do_sample=True,
                                                no_repeat_ngram_size=4)      

    if args.algorithm == "KGW" or args.algorithm == "DIP" or args.algorithm == 'EXP' or args.algorithm == 'TS' or args.algorithm == 'SynthID' or args.algorithm == 'SIR' or args.algorithm == 'MorphMark':
        my_watermark = AutoWatermark.load(f'{args.algorithm}',  
                                        algorithm_config=f'config/{args.algorithm}.json',  
                                        transformers_config=transformers_config)  
    else:
        my_watermark = AutoWatermark.load(f'{args.algorithm}',  
                                        algorithm_config= config_dict, 
                                        transformers_config=transformers_config) 
    

    if args.algorithm == 'EXP':
        # 用于无攻击best动态阈值成功率计算器，用于评估流水线的成功率 # 评估指标 # 评估规则 # 目标的错误接受率（False Positive Rate）
        calculator_unattack = DynamicThresholdSuccessRateCalculator(labels=args.labels, rule=args.rules, target_fpr=args.target_fpr,reverse=True)
        # 用于各种攻击动态阈值成功率计算器，用于评估流水线的成功率 # 评估指标 # 评估规则 # 目标的错误接受率（False Positive Rate）
        calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'], rule=args.rules, target_fpr=args.target_fpr,reverse=True)
        # fpr=1%
        calculator_fpr_1 = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'], rule="target_fpr", target_fpr=0.01, reverse=True)
        # fpr=10%
        calculator_fpr_10 = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'], rule="target_fpr", target_fpr=0.1, reverse=True)
    else:    
        # 用于无攻击best动态阈值成功率计算器，用于评估流水线的成功率 # 评估指标 # 评估规则 # 目标的错误接受率（False Positive Rate）
        calculator_unattack = DynamicThresholdSuccessRateCalculator(labels=args.labels, rule=args.rules, target_fpr=args.target_fpr,reverse=False)
        # 用于各种攻击动态阈值成功率计算器，用于评估流水线的成功率 # 评估指标 # 评估规则 # 目标的错误接受率（False Positive Rate）
        calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'], rule=args.rules, target_fpr=args.target_fpr,reverse=False)
        # fpr=1%
        calculator_fpr_1 = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'], rule="target_fpr", target_fpr=0.01, reverse=False)
        # fpr=10%
        calculator_fpr_10 = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'], rule="target_fpr", target_fpr=0.1, reverse=False)

    # 创建一个用于检测文本的流水线  # 使用文本编辑器截断提示词 # 显示进度 # 返回类型为分数 # WatermarkedTextDetectionPipeline
    pipline_watermark = WatermarkDetectionPipeline(
        dataset=dataset_path, text_editor_list=[TruncatePromptTextEditor()],
        unwatermarked_text_editor_list=[TruncatePromptTextEditor()],
        show_progress=True, return_type=DetectionPipelineReturnType.FULL, 
        device=device,
        model_ppl=model_path, tokenizer_ppl=tokenizer_path,
        model_Bert=model_Bert, tokenizer_Bert=tokenizer_Bert,
        model_dipper=model_dipper, tokenizer_dipper=tokenizer_dipper
    ) 
    
    # print("\n水印算法名称:", algorithm_name,"最小生成长度:230","z_threshold: ", config_dict['z_threshold'])
    # print("topk:", config_dict['topk'], "α:", config_dict['α'], ", alpha:", config_dict['alpha'], ", mean:", config_dict['mean'], ", epsilon:", config_dict['epsilon'])
    unwatermark_evaluate, watermark_evaluate, attack_watermark_evaluate_Word_D_3, attack_watermark_evaluate_Word_D_5, attack_watermark_evaluate_Word_D_7, attack_watermark_evaluate_Word_S_7, attack_watermark_evaluate_Word_S_context_3, attack_watermark_evaluate_typo, attack_watermark_evaluate_doc_P_dipper, ppl_evaluation_result, logdiversity_evaluation_result, BLEU_evaluation_result, BERTScore_evaluation_result, GPT_evaluation_result, Pass_evaluation_result, execution_time_unwatermarked_200_sum, execution_time_watermarked_200_sum, execution_time_unwatermarked_200_avg, execution_time_watermarked_200_avg, execution_time_detect_unwatermarked_sum, execution_time_detect_watermarked_sum, execution_time_detect_unwatermarked_avg, execution_time_detect_watermarked_avg = pipline_watermark.evaluate(my_watermark, args.dataset, args.data_lines)
    # print("The average time required to generate each text: ", execution_time_watermarked_200_avg, "(s)")
    
    # 打印算法名称和参数
    # print("\n无攻击下检测成功率(动态阈值):")
    # best计算并输出水印和非水印文本的成功率  
    result_unattack, threshold_unattack = calculator_unattack.calculate([float(result.detect_result['score']) for result in watermark_evaluate], [float(result.detect_result['score']) for result in unwatermark_evaluate]) 
    # print(result_unattack)
    # 打印算法名称和参数
    # print("\n无攻击下检测成功率(静态阈值FPR=0.01),")

    # frp=1%计算并输出水印和非水印文本的成功率
    result_unattack_fpr_1, threshold_unattack_fpr_1 = calculator_fpr_1.calculate([float(result.detect_result['score']) for result in watermark_evaluate], [float(result.detect_result['score']) for result in unwatermark_evaluate]) 
    # print(result_unattack_fpr_1)
    
    # frp=10%计算并输出水印和非水印文本的成功率
    result_unattack_fpr_10, threshold_unattack_fpr_10 = calculator_fpr_10.calculate([float(result.detect_result['score']) for result in watermark_evaluate], [float(result.detect_result['score']) for result in unwatermark_evaluate]) 
    # print(result_unattack_fpr_10)
    

    # 打印攻击算法名称和参数
    # print("\nattack_Word-D_1攻击下检测成功率(动态阈值):")
    # 计算并输出攻击后的水印和非水印文本的成功率
    # result_attack_Word_D_1, threshold_attack_Word_D_1 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_D_1], [float(result.detect_result['score']) for result in unwatermark_evaluate])
    # print(result_attack_Word_D_1)
    
    # 打印攻击算法名称和参数
    # print("\nattack_Word-D攻击下检测成功率(动态阈值):")
    # 计算并输出攻击后的水印和非水印文本的成功率
    result_attack_Word_D_3, threshold_attack_Word_D_3 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_D_3], [float(result.detect_result['score']) for result in unwatermark_evaluate])
    # print(result_attack_Word_D)
    
    # 打印攻击算法名称和参数
    # print("\nattack_Word-D_1攻击下检测成功率(动态阈值):")
    # 计算并输出攻击后的水印和非水印文本的成功率
    result_attack_Word_D_5, threshold_attack_Word_D_5 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_D_5], [float(result.detect_result['score']) for result in unwatermark_evaluate])
    # print(result_attack_Word_D_1)
    
    # 打印攻击算法名称和参数
    # print("\nattack_Word-D_1攻击下检测成功率(动态阈值):")
    # 计算并输出攻击后的水印和非水印文本的成功率
    result_attack_Word_D_7, threshold_attack_Word_D_7 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_D_7], [float(result.detect_result['score']) for result in unwatermark_evaluate])
    # print(result_attack_Word_D_1)

    # 打印攻击算法名称和参数
    # print("\nattack_Word-S攻击下检测成功率(动态阈值):")
    # 计算并输出攻击后的水印和非水印文本的成功率
    # result_attack_Word_S_1, threshold_attack_Word_S_1 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_S_1], [float(result.detect_result['score']) for result in unwatermark_evaluate])
    # print(result_attack_Word_S)
    
    # 打印攻击算法名称和参数
    # print("\nattack_Word-S攻击下检测成功率(动态阈值):")
    # 计算并输出攻击后的水印和非水印文本的成功率
    # result_attack_Word_S_3, threshold_attack_Word_S_3 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_S_3], [float(result.detect_result['score']) for result in unwatermark_evaluate])
    # print(result_attack_Word_S)
    
    # 打印攻击算法名称和参数
    # print("\nattack_Word-S攻击下检测成功率(动态阈值):")
    # 计算并输出攻击后的水印和非水印文本的成功率
    # result_attack_Word_S_5, threshold_attack_Word_S_5 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_S_5], [float(result.detect_result['score']) for result in unwatermark_evaluate])
    # print(result_attack_Word_S)
    
    # 打印攻击算法名称和参数
    # print("\nattack_Word-S攻击下检测成功率(动态阈值):")
    # 计算并输出攻击后的水印和非水印文本的成功率
    result_attack_Word_S_7, threshold_attack_Word_S_7 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_S_7], [float(result.detect_result['score']) for result in unwatermark_evaluate])
    # print(result_attack_Word_S)
    
    # 打印攻击算法名称和参数
    # print("\nattack_Word-S(context)_3攻击下检测成功率(动态阈值):")
    # 计算并输出攻击后的水印和非水印文本的成功率
    result_attack_Word_S_context_3, threshold_attack_Word_S_context_3 = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_Word_S_context_3], [float(result.detect_result['score']) for result in unwatermark_evaluate])
    # print(result_attack_Word_S_3)

    # 打印攻击算法名称和参数
    # print("\nattack_doc_P_dipper攻击下检测成功率(动态阈值):")
    # 计算并输出攻击后的水印和非水印文本的成功率
    result_attack_doc_P_dipper, threshold_attack_doc_P_dipper = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_doc_P_dipper], [float(result.detect_result['score']) for result in unwatermark_evaluate])
    # print(result_attack_Word_S) 
    
    # 打印攻击算法名称和参数
    # print("\nattack_doc_P_dipper攻击下检测成功率(动态阈值):")
    # 计算并输出攻击后的水印和非水印文本的成功率
    # result_attack_doc_P_GPT, threshold_attack_doc_P_GPT = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_doc_P_GPT], [float(result.detect_result['score']) for result in unwatermark_evaluate])
    # print(result_attack_Word_S) 
    
    # 打印攻击算法名称和参数
    # print("\nattack_misspelling攻击下检测成功率(动态阈值):")
    # 计算并输出攻击后的水印和非水印文本的成功率
    # result_attack_misspelling, threshold_attack_misspelling = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_misspelling], [float(result.detect_result['score']) for result in unwatermark_evaluate])
    # print(result_attack_misspelling) 
    
    # 打印攻击算法名称和参数
    # print("\nattack_typo攻击下检测成功率(动态阈值):")
    # 计算并输出攻击后的水印和非水印文本的成功率
    result_attack_typo, threshold_attack_typo = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_typo], [float(result.detect_result['score']) for result in unwatermark_evaluate])
    # print(result_attack_typo) 
    
    # 打印攻击算法名称和参数
    # print("\nattack_contraction攻击下检测成功率(动态阈值):")
    # 计算并输出攻击后的水印和非水印文本的成功率
    # result_attack_contraction, threshold_attack_contraction = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_contraction], [float(result.detect_result['score']) for result in unwatermark_evaluate])
    # print(result_attack_contraction) 
    
    # 打印攻击算法名称和参数
    # print("\nattack_swap攻击下检测成功率(动态阈值):")
    # 计算并输出攻击后的水印和非水印文本的成功率
    # result_attack_swap, threshold_attack_swap = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_swap], [float(result.detect_result['score']) for result in unwatermark_evaluate])
    # print(result_attack_swap) 
    
    # 打印攻击算法名称和参数
    # print("\nattack_lowercase攻击下检测成功率(动态阈值):")
    # 计算并输出攻击后的水印和非水印文本的成功率
    # result_attack_lowercase, threshold_attack_lowercase = calculator.calculate([float(result.detect_result['score']) for result in attack_watermark_evaluate_lowercase], [float(result.detect_result['score']) for result in unwatermark_evaluate])
    # print(result_attack_lowercase) 
    
    if args.dataset =='c4' or args.dataset =='eli5' or args.dataset =='multinews' or args.dataset =='rocstories' or args.dataset =='flickr30k' or args.dataset =='cnn_daily_mail':
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
        
        # print("\n文本质量的信息如下:", ", 评估指标:", "log")
        # 计算并输出水印和非水印文本的文本质量
        # result_BLEU = {'watermarked': sum([result.watermarked_quality_score for result in BLEU_evaluation_result]) / len(BLEU_evaluation_result), 
        #                 'unwatermarked': sum([result.unwatermarked_quality_score for result in BLEU_evaluation_result]) / len(BLEU_evaluation_result)}
        # print(result_BLEU)
        
        # <--- 新增：计算 BERTScore 平均值
        # print("\n文本质量的信息如下:", ", 评估指标:", "BERTScore")
        result_BERTScore = {'watermarked': sum([result.watermarked_quality_score for result in BERTScore_evaluation_result]) / len(BERTScore_evaluation_result), 
                            'unwatermarked': sum([result.unwatermarked_quality_score for result in BERTScore_evaluation_result]) / len(BERTScore_evaluation_result)}
        # print(result_BERTScore)
    elif args.dataset =='wmt16_de_en':
        # print("\n文本质量的信息如下:", ", 评估指标:", "log")
        # 计算并输出水印和非水印文本的文本质量
        result_BLEU = {'watermarked': sum([result.watermarked_quality_score for result in BLEU_evaluation_result]) / len(BLEU_evaluation_result), 
                        'unwatermarked': sum([result.unwatermarked_quality_score for result in BLEU_evaluation_result]) / len(BLEU_evaluation_result)}
        # print(result_BLEU)
        
        # <--- 新增：计算 BERTScore 平均值
        # print("\n文本质量的信息如下:", ", 评估指标:", "BERTScore")
        result_BERTScore = {'watermarked': sum([result.watermarked_quality_score for result in BERTScore_evaluation_result]) / len(BERTScore_evaluation_result), 
                            'unwatermarked': sum([result.unwatermarked_quality_score for result in BERTScore_evaluation_result]) / len(BERTScore_evaluation_result)}
        # print(result_BERTScore)
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
    
    with open("output_saw_KDD_c4_Llama-3-8B-Instruct_7attacks_temp_1_tokens_200_datalines_100.txt", "a") as file:
        file.write("\nparameter of config as follows:\n") 
        file.write(config_algorithm) 
        file.write("\nparameter of algorithm as follows:\n") 
        file.write(str(config_dict)) 

        if args.dataset =='c4' or args.dataset =='eli5' or args.dataset =='multinews' or args.dataset =='rocstories' or args.dataset =='flickr30k' or args.dataset =='cnn_daily_mail':
            file.write("\nPPL:\n")  
            file.write(str(result_PPL))
            file.write("\nLog_Diversity:\n")  
            file.write(str(result_Log_Diversity))  
            # file.write("\nBLEU:\n")  
            # file.write(str(result_BLEU)) 
            # <--- 新增：写入 BERTScore
            file.write("\nBERTScore:\n")
            file.write(str(result_BERTScore))
        elif args.dataset =='wmt16_de_en':
            file.write("\nBLEU:\n")  
            file.write(str(result_BLEU)) 
            #file.write("\nGPT:\n")  
            #file.write(str(result_GPT)) 
            # <--- 新增：写入 BERTScore
            file.write("\nBERTScore:\n")
            file.write(str(result_BERTScore))
        else:
            file.write("\nPass:\n")  
            file.write(str(result_Pass))

        # file.write("\ndetection accuracy of attack_Word-D_1:\n")
        # file.write(str(result_attack_Word_D_1))
        # file.write("\nz-score of attack_Word-D_1:")
        # file.write(str(threshold_attack_Word_D_1))
        file.write("\ndetection accuracy of attack_Word-D_3:\n") 
        file.write(str(result_attack_Word_D_3)) 
        file.write("\nz-score of attack_Word-D_3:") 
        file.write(str(threshold_attack_Word_D_3)) 
        file.write("\ndetection accuracy of attack_Word-D_5:\n") 
        file.write(str(result_attack_Word_D_5)) 
        file.write("\nz-score of attack_Word-D_5:") 
        file.write(str(threshold_attack_Word_D_5))
        file.write("\ndetection accuracy of attack_Word-D_7:\n") 
        file.write(str(result_attack_Word_D_7)) 
        file.write("\nz-score of attack_Word-D_7:") 
        file.write(str(threshold_attack_Word_D_7)) 
        
        # file.write("\ndetection accuracy of attack_Word-S_1:\n")
        # file.write(str(result_attack_Word_S_1))
        # file.write("\nz-score of attack_Word-S_1:") 
        # file.write(str(threshold_attack_Word_S_1)) 
        # file.write("\ndetection accuracy of attack_Word-S_3:\n")
        # file.write(str(result_attack_Word_S_3))
        # file.write("\nz-score of attack_Word-S_3:") 
        # file.write(str(threshold_attack_Word_S_3)) 
        # file.write("\ndetection accuracy of attack_Word-S_5:\n")
        # file.write(str(result_attack_Word_S_5))
        # file.write("\nz-score of attack_Word-S_5:") 
        # file.write(str(threshold_attack_Word_S_5))
        file.write("\ndetection accuracy of attack_Word-S_7:\n")
        file.write(str(result_attack_Word_S_7))
        file.write("\nz-score of attack_Word-S_7:") 
        file.write(str(threshold_attack_Word_S_7)) 
        
        file.write("\ndetection accuracy of attack_Word-S(context)_3:\n")
        file.write(str(result_attack_Word_S_context_3))
        file.write("\nz-score of attack_Word-S(context)_3:") 
        file.write(str(threshold_attack_Word_S_context_3)) 
        
        file.write("\ndetection accuracy of attack_doc_P_dipper\n")
        file.write(str(result_attack_doc_P_dipper))
        file.write("\nz-score of attack_doc_P_dipper") 
        file.write(str(threshold_attack_doc_P_dipper))  
        
        # file.write("\ndetection accuracy of attack_doc_P_GPT\n")
        # file.write(str(result_attack_doc_P_GPT))
        # file.write("\nz-score of attack_doc_P_GPT") 
        # file.write(str(threshold_attack_doc_P_GPT))  
        
        # file.write("\ndetection accuracy of attack_misspelling:\n")
        # file.write(str(result_attack_misspelling))
        # file.write("\nz-score of attack_misspelling:") 
        # file.write(str(threshold_attack_misspelling))  
        
        file.write("\ndetection accuracy of attack_typo:\n")
        file.write(str(result_attack_typo))
        file.write("\nz-score of attack_typo:") 
        file.write(str(threshold_attack_typo))  
        
        # file.write("\ndetection accuracy of attack_contraction:\n")
        # file.write(str(result_attack_contraction))
        # file.write("\nz-score of attack_contraction:") 
        # file.write(str(threshold_attack_contraction))  
        
        # file.write("\ndetection accuracy of attack_swap:\n")
        # file.write(str(result_attack_swap))
        # file.write("\nz-score of attack_swap:") 
        # file.write(str(threshold_attack_swap))  
        
        # file.write("\ndetection accuracy of attack_lowercase:\n")
        # file.write(str(result_attack_lowercase))
        # file.write("\nz-score of attack_lowercase:") 
        # file.write(str(threshold_attack_lowercase))  
        
        file.write("\ndetection accuracy of fpr=best :\n") 
        file.write(str(result_unattack)) 
        file.write("\nz-score of fpr=best :") 
        file.write(str(threshold_unattack)) 

        file.write("\ndetection accuracy of fpr=1% :\n") 
        file.write(str(result_unattack_fpr_1)) 
        file.write("\nz-score of fpr=1% :") 
        file.write(str(threshold_unattack_fpr_1)) 
        
        file.write("\ndetection accuracy of fpr=10% :\n") 
        file.write(str(result_unattack_fpr_10)) 
        file.write("\nz-score of fpr=10% :") 
        file.write(str(threshold_unattack_fpr_10)) 
        
        file.write("\n--- Generation Time Statistics ---")
        file.write("\nexecution_time_generate_unwatermarked_sum:") 
        file.write(str(execution_time_unwatermarked_200_sum)) 
        file.write("\nexecution_time_generate_unwatermarked_avg:") 
        file.write(str(execution_time_unwatermarked_200_avg)) 
        file.write("\nexecution_time_generate_watermarked_sum:") 
        file.write(str(execution_time_watermarked_200_sum)) 
        file.write("\nexecution_time_generate_watermarked_avg:") 
        file.write(str(execution_time_watermarked_200_avg)) 
        
        # <--- 新增：写入检测时间统计
        file.write("\n--- Detection Time Statistics ---")
        file.write("\nexecution_time_detect_unwatermarked_sum: ")
        file.write(str(execution_time_detect_unwatermarked_sum))
        file.write("\nexecution_time_detect_unwatermarked_avg: ")
        file.write(str(execution_time_detect_unwatermarked_avg))

        file.write("\nexecution_time_detect_watermarked_sum: ")
        file.write(str(execution_time_detect_watermarked_sum))
        file.write("\nexecution_time_detect_watermarked_avg: ")
        file.write(str(execution_time_detect_watermarked_avg))
        # <--- 新增结束    
    
        file.write("\n=========================\n")


if __name__ == '__main__':
    import argparse  
    # algorithm_config=f'config/{algorithm_name}.json'
    parser = argparse.ArgumentParser()
    # KGW SWEET SIR SAW EWD SynthID DIP                  MorphMark Unbiased SMOOTH  Unigram  EXP 
    parser.add_argument('--algorithm', type=str, default='KGW')  # 水印算法名称 
    parser.add_argument('--dataset', type=str, default='c4')      # c4       cnn_daily_mail      rocstories    eli5                                  wmt16_de_en              human_eval 
    parser.add_argument('--model', type=str, default='Llama-3-8B-Instruct')  # opt-1.3b Llama-3-8B-Instruct Qwen2.5-0.5B Qwen2.5-7B DeepSeek-R1-Distill-Qwen-7B nllb-200-distilled-600M  starcoder    
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--min_length', type=int, default=200)
    parser.add_argument('--data_lines', type=int, default=10)
    parser.add_argument('--temperature_inner', type=float, default='1.0')  

    # parser.add_argument('--resilience', type=str, default='soft') # hard soft
    # parser.add_argument('--alpha', type=float, default='0.34')
    # parser.add_argument('--fixed_pos', type=list, default=["n", "a"])
    # parser.add_argument('--gamma', type=float, default='0.5')
    # parser.add_argument('--epsilon', type=float, default='0.1')
    # parser.add_argument('--delta', type=float, default='2.0')
    # parser.add_argument('--eta', type=float, default='3.0')
    # parser.add_argument('--entropy_threshold', type=float, default='0.9')
    # parser.add_argument('--z_threshold', type=float, default='4.0') 
    
    parser.add_argument('--noise', type=str, default='uniform')  # gaussian  uniform
    parser.add_argument('--beta', type=float, default='0.7') 
    parser.add_argument('--std', type=float, default='0.05') 

    parser.add_argument('--labels', nargs='+', default=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'])
    parser.add_argument('--rules', type=str, default='best') # target_fpr
    parser.add_argument('--target_fpr', type=float, default=0.01)
    
    args = parser.parse_args()
    assess(args)