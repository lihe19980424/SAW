import os

# output_saw_KDD_c4_5attacks_temp_1_tokens_200_datalines_100_beta.txt
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.0 --std 0.05 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.3 --std 0.05 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.5 --std 0.05 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.7 --std 0.05 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.05 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 1.0 --std 0.05 --temperature_inner 1.0")

# output_saw_KDD_c4_5attacks_temp_1_tokens_200_datalines_100_std.txt
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.01 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.02 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.03 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.04 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.05 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.06 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.07 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.08 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --noise uniform --beta 0.9 --std 0.09 --temperature_inner 1.0")



# output_saw_c4_8attacks_temp_1_tokens_300.txt
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --noise uniform --beta 0.9 --std 0.04 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --noise gaussian --beta 0.9 --std 0.04 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm KGW --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SWEET --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm EWD --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm Unigram --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm DIP --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm EXP --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SynthID --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SIR --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm Unbiased --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")

# output_saw_cnn_8attacks_temp_1_tokens_300.txt
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset cnn_daily_mail --model Llama-3-8B-Instruct --max_new_tokens 300 --min_length 300 --data_lines 200 --noise uniform --beta 0.9 --std 0.04 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset cnn_daily_mail --model Llama-3-8B-Instruct --max_new_tokens 300 --min_length 300 --data_lines 200 --noise gaussian --beta 0.9 --std 0.04 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm KGW --dataset cnn_daily_mail --model Llama-3-8B-Instruct --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SWEET --dataset cnn_daily_mail --model Llama-3-8B-Instruct --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm EWD --dataset cnn_daily_mail --model Llama-3-8B-Instruct --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm Unigram --dataset cnn_daily_mail --model Llama-3-8B-Instruct --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm DIP --dataset cnn_daily_mail --model Llama-3-8B-Instruct --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm EXP --dataset cnn_daily_mail --model Llama-3-8B-Instruct --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SynthID --dataset cnn_daily_mail --model Llama-3-8B-Instruct --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SIR --dataset cnn_daily_mail --model Llama-3-8B-Instruct --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm Unbiased --dataset cnn_daily_mail --model Llama-3-8B-Instruct --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")

# # output_saw_rocstories_8attacks_temp_1_tokens_300.txt
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset rocstories --model Qwen2.5-0.5B --max_new_tokens 300 --min_length 300 --data_lines 200 --noise uniform --beta 0.9 --std 0.04 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset rocstories --model Qwen2.5-0.5B --max_new_tokens 300 --min_length 300 --data_lines 200 --noise gaussian --beta 0.9 --std 0.04 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm KGW --dataset rocstories --model Qwen2.5-0.5B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SWEET --dataset rocstories --model Qwen2.5-0.5B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm EWD --dataset rocstories --model Qwen2.5-0.5B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm Unigram --dataset rocstories --model Qwen2.5-0.5B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm DIP --dataset rocstories --model Qwen2.5-0.5B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm EXP --dataset rocstories --model Qwen2.5-0.5B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SynthID --dataset rocstories --model Qwen2.5-0.5B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SIR --dataset rocstories --model Qwen2.5-0.5B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm Unbiased --dataset rocstories --model Qwen2.5-0.5B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")

# output_saw_eli5_8attacks_temp_1_tokens_300.txt
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --noise uniform --beta 0.9 --std 0.04 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SAW --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --noise gaussian --beta 0.9 --std 0.04 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm KGW --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SWEET --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm EWD --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm Unigram --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm DIP --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm EXP --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SynthID --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm SIR --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
# os.system("python3 pipeline_saw_1.py --algorithm Unbiased --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0")
