import torch
import json
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load data
with open('dataset/c4/processed_c4.json', 'r') as f:
    lines = f.readlines()
item = json.loads(lines[0])
prompt = item['prompt']
natural_text = item['natural_text']

opt_path = "./models/opt-1.3b" # device_map='auto'  .to(device)


def test_algorithm(algorithm_name):
    # Check algorithm name
    assert algorithm_name in ['ColorMark', 'KGW', 'Unigram', 'SWEET', 'EWD', 'SIR', 'XSIR', 'UPV', 'EXP', 'EXPEdit', 'SynthID']

    # Device
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # Transformers config
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained(opt_path).to(device),
                                            tokenizer=AutoTokenizer.from_pretrained(opt_path),
                                            vocab_size=50272,
                                            device=device,
                                            max_new_tokens=200,
                                            min_length=230,
                                            do_sample=True,
                                            no_repeat_ngram_size=4)

    # Load watermark algorithm
    myWatermark = AutoWatermark.load(f'{algorithm_name}', algorithm_config=f'config/{algorithm_name}.json', transformers_config=transformers_config)

    # Generate text
    watermarked_text = myWatermark.generate_watermarked_text(prompt)
    unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)

    # Detect
    detect_result1 = myWatermark.detect_watermark(watermarked_text)
    detect_result2 = myWatermark.detect_watermark(unwatermarked_text)
    detect_result3 = myWatermark.detect_watermark(natural_text)

    print("LLM-generated watermarked text:")
    print(watermarked_text)
    print('\n')
    print(detect_result1)
    print('\n')

    print("LLM-generated unwatermarked text:")
    print(unwatermarked_text)
    print('\n')
    print(detect_result2)
    print('\n')

    print("Natural text:")
    print(natural_text)
    print('\n')
    print(detect_result3)
    
    
test_algorithm('ColorMark')