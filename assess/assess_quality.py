# ==========================================================================
# assess_quality.py
# Description: Assess the impact on text quality of a watermarking algorithm
# ==========================================================================

import torch
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.dataset import C4Dataset, HumanEvalDataset, WMT16DE_ENDataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, LlamaForCausalLM, T5ForConditionalGeneration, AutoTokenizer, LlamaTokenizer, T5Tokenizer
from evaluation.tools.text_editor import TruncatePromptTextEditor, TruncateTaskTextEditor, CodeGenerationTextEditor
from evaluation.tools.text_quality_analyzer import PPLCalculator, LogDiversityAnalyzer, BLEUCalculator, PassOrNotJudger, GPTTextDiscriminator
from evaluation.pipelines.quality_analysis import DirectTextQualityAnalysisPipeline, QualityPipelineReturnType, ReferencedTextQualityAnalysisPipeline, ExternalDiscriminatorTextQualityAnalysisPipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device) # 输出设备信息

def assess_quality(algorithm_name, metric):
    if metric == 'PPL':
        my_dataset = C4Dataset('dataset/c4/processed_c4.json')
        # my_dataset = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl')
        # my_dataset = HumanEvalDataset('dataset/human_eval/test.jsonl')
        pipeline = DirectTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                     watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                     unwatermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                     analyzer=PPLCalculator(model=LlamaForCausalLM.from_pretrained("./models/llama-7b-hf", device_map='auto'),
                                                                            tokenizer=LlamaTokenizer.from_pretrained("./models/llama-7b-hf"),
                                                                            device=device),
                                                     unwatermarked_text_source='natural', show_progress=True, 
                                                     return_type=QualityPipelineReturnType.MEAN_SCORES)
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
                                                 min_length=230,
                                                 do_sample=True,
                                                 no_repeat_ngram_size=4)

    elif metric == 'Log Diversity':
        my_dataset = C4Dataset('dataset/c4/processed_c4.json')
        pipeline = DirectTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                     watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                     unwatermarked_text_editor_list=[],
                                                     analyzer=LogDiversityAnalyzer(),
                                                     unwatermarked_text_source='natural', show_progress=True, 
                                                     # unwatermarked_text_source='generated', show_progress=True, 
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
    
    
    my_watermark = AutoWatermark.load(f'{algorithm_name}', 
                                    algorithm_config=f'config/{algorithm_name}.json',
                                    transformers_config=transformers_config)
    print(pipeline.evaluate(my_watermark))

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='Rethinking')
    parser.add_argument('--metric', type=str, default='PPL')
    args = parser.parse_args()
    # KGW:{'watermarked': 12.809204187393188, 'unwatermarked': 7.912650513648987}
    # Block_Box:{'watermarked': 13.62254020690918, 'unwatermarked': 7.912650513648987}
    assess_quality(args.algorithm, args.metric)