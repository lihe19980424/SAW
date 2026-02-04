# ================================================================
# assess_robustness.py
# Description: Assess the robustness of a watermarking algorithm
# ================================================================

import torch
from translate import Translator
from evaluation.dataset import C4Dataset, WMT16DE_ENDataset, HumanEvalDataset #evaluation
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, LlamaForCausalLM, BertForMaskedLM, T5ForConditionalGeneration, AutoTokenizer, LlamaTokenizer, T5Tokenizer, BertTokenizer
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType
from evaluation.tools.text_editor import TruncatePromptTextEditor, WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution, GPTParaphraser, DipperParaphraser, BackTranslationTextEditor


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(device)
def assess_robustness(algorithm_name, attack_name):
    my_dataset = C4Dataset('dataset/c4/processed_c4.json')
    # my_dataset = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl')
    # my_dataset = HumanEvalDataset('dataset/human_eval/test.jsonl')
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
                                             min_length=200,  # 230
                                             do_sample=True,
                                             no_repeat_ngram_size=4)
    my_watermark = AutoWatermark.load(f'{algorithm_name}', 
                                    algorithm_config=f'config/{algorithm_name}.json',
                                    transformers_config=transformers_config)
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

    pipline1 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[TruncatePromptTextEditor(), attack],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES) 

    pipline2 = UnWatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES, text_source_mode = "generated")
    
    # 静态阈值
    # calculator = FundamentalSuccessRateCalculator(labels=labels)
    # 动态阈值  # ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']   ['TPR', 'F1']
    calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'], rule='best',reverse=False)
    
    watermark_evaluate = pipline1.evaluate(my_watermark)
    unwatermark_evaluate = pipline2.evaluate(my_watermark)
    
    print(calculator.calculate(watermark_evaluate, unwatermark_evaluate))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='Rethinking')
    parser.add_argument('--attack', type=str, default='Word-D')
    args = parser.parse_args()
    # KGW:Word-D:{'TPR': 1.0, 'F1': 0.992555831265508}
    # Black_Box:Word-D:{'TPR': 0.7, 'F1': 0.7}
    # KGW:Word-S:{'TPR': 0.990, 'F1': 0.9635036496350365}
    # Black_Box:Word-S:{'TPR': 0.86, 'F1': 0.7166666666666667}
    assess_robustness(args.algorithm, args.attack)