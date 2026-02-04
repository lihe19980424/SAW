import os

# # output_saw_KDD_c4_5attacks_temp_1_tokens_200_datalines_100.txt

# # baselines  output_saw_KDD_c4_5attacks_temp_1_tokens_200_datalines_100.txt
# os.system("python3 pipeline_saw.py --algorithm KGW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SWEET --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm EWD --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm DIP --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SynthID --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SIR --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")

# # beta  output_saw_KDD_c4_5attacks_temp_1_tokens_200_datalines_100.txt
# os.system("python3 pipeline_saw.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.0 --std 0.05 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.3 --std 0.05 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.5 --std 0.05 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.7 --std 0.05 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.05 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 1.0 --std 0.05 --temperature_inner 1.0")

# # std  output_saw_KDD_c4_5attacks_temp_1_tokens_200_datalines_100.txt
# os.system("python3 pipeline_saw.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.01 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.02 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.03 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.04 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.05 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.06 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.07 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.08 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.09 --temperature_inner 1.0")









# output_saw_KDD_wmt16_de_en_5attacks_temp_1_tokens_200_datalines_100.txt

# baselines  output_saw_KDD_wmt16_de_en_5attacks_temp_1_tokens_200_datalines_100.txt

# os.system("python3 pipeline_saw.py --algorithm KGW --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SWEET --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm EWD --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm DIP --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SynthID --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SIR --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
# os.system("python3 pipeline_saw.py --algorithm SAW --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.7 --std 0.05 --temperature_inner 1.0")









# output_saw_KDD_c4_Llama-3-8B-Instruct_7attacks_temp_1_tokens_200_datalines_100.txt

# baselines  output_saw_KDD_c4_Llama-3-8B-Instruct_7attacks_temp_1_tokens_200_datalines_100.txt

os.system("python3 pipeline_saw.py --algorithm KGW --dataset c4 --model Llama-3-8B-Instruct --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
os.system("python3 pipeline_saw.py --algorithm SWEET --dataset c4 --model Llama-3-8B-Instruct --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
os.system("python3 pipeline_saw.py --algorithm EWD --dataset c4 --model Llama-3-8B-Instruct --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
os.system("python3 pipeline_saw.py --algorithm DIP --dataset c4 --model Llama-3-8B-Instruct --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
os.system("python3 pipeline_saw.py --algorithm SynthID --dataset c4 --model Llama-3-8B-Instruct --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
os.system("python3 pipeline_saw.py --algorithm SIR --dataset c4 --model Llama-3-8B-Instruct --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
os.system("python3 pipeline_saw.py --algorithm SAW --dataset c4 --model Llama-3-8B-Instruct --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.7 --std 0.05 --temperature_inner 1.0")


os.system("python3 pipeline_saw.py --algorithm KGW --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
os.system("python3 pipeline_saw.py --algorithm SWEET --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
os.system("python3 pipeline_saw.py --algorithm EWD --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
os.system("python3 pipeline_saw.py --algorithm DIP --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
os.system("python3 pipeline_saw.py --algorithm SynthID --dataset c4 --model Llama-3-8B-Instruct --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
os.system("python3 pipeline_saw.py --algorithm SIR --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 100 --temperature_inner 1.0")
os.system("python3 pipeline_saw.py --algorithm SAW --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.7 --std 0.05 --temperature_inner 1.0")