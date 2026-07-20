import os



# output_smooth_hard_del05_temp07_tokens300.txt
# os.system("python3 pipeline_smooth_del05_tem07_hard_1.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.7 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_del05_tem07_hard_1.py --algorithm SMOOTH --dataset c4 --model gpt2-xl --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.7 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_del05_tem07_hard_1.py --algorithm SMOOTH --dataset cnn_daily_mail --model Llama-3-8B-Instruct --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.7 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0")
os.system("python3 pipeline_smooth_del05_tem07_hard_1.py --algorithm SMOOTH --dataset rocstories --model Qwen2.5-0.5B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.7 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_del05_tem07_hard_1.py --algorithm SMOOTH --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.7 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0")
