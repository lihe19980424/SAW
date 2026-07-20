import os


# smooth4 different alpha 5x2h=10h
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.0 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0")  
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.1 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.2 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.4 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0") 

# # smooth3 different alpha 5x2h=10h
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.0 --gamma 0.5 --epsilon 0.0 --delta 1.0 --z_threshold 4.0")  
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.1 --gamma 0.5 --epsilon 0.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.2 --gamma 0.5 --epsilon 0.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 0.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.4 --gamma 0.5 --epsilon 0.0 --delta 1.0 --z_threshold 4.0") 

# smooth1 different alpha 3x2h=6h
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.00 --gamma 0.5 --epsilon 0.0 --delta 2.0 --z_threshold 4.0")  
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.25 --gamma 0.5 --epsilon 0.0 --delta 2.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.5 --epsilon 0.0 --delta 2.0 --z_threshold 4.0") 

# smooth4 different epsilon 5x2h=10h
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 0.0 --delta 1.0 --z_threshold 4.0")  
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 0.5 --delta 1.0 --z_threshold 4.0")  
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 1.5 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 3.0 --delta 1.0 --z_threshold 4.0")   
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 4.0 --delta 1.0 --z_threshold 4.0")  

# smooth1 different epsilon 4x2h=8h
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.00 --gamma 0.5 --epsilon 1.0 --delta 2.0 --z_threshold 4.0")  
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.00 --gamma 0.5 --epsilon 2.0 --delta 2.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.00 --gamma 0.5 --epsilon 4.0 --delta 2.0 --z_threshold 4.0") 


# smooth1 different gamma 3x2h=6h
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.00 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")  
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.25 --epsilon 2.0 --delta 2.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 4.0") 

# smooth1 different delta 2x2h=2h
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 4.0 --z_threshold 4.0")

# smooth1 different z_threshold 1x2h=2h
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 5.0")










# # smooth4 machine translation different alpha 5x5min=25min
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.0 --gamma 0.5 --epsilon 0.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.1 --gamma 0.5 --epsilon 0.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.2 --gamma 0.5 --epsilon 0.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 0.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.4 --gamma 0.5 --epsilon 0.0 --delta 1.0 --z_threshold 4.0")

# # smooth3 machine translation different alpha 5x5min=25min
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.0 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.1 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.2 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.4 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0")

# # smooth2 machine translation different alpha 5x5min=25min 跑完啦
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.0 --gamma 0.4 --epsilon 1.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.1 --gamma 0.4 --epsilon 1.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.2 --gamma 0.4 --epsilon 1.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.4 --epsilon 1.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.4 --gamma 0.4 --epsilon 1.0 --delta 1.0 --z_threshold 4.0")

# # smooth1 machine translation  different alpha 8x5min=40min
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.00 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.10 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.20 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.30 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.40 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.60 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.70 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")


# # smooth4 machine translation different epsilon 5x5min=25min
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 0.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 3.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 4.0 --delta 1.0 --z_threshold 4.0")

# # smooth2 machine translation different epsilon 5x5min=25min  没跑
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.2 --gamma 0.4 --epsilon 0.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.2 --gamma 0.4 --epsilon 1.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.2 --gamma 0.4 --epsilon 2.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.2 --gamma 0.4 --epsilon 3.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.2 --gamma 0.4 --epsilon 4.0 --delta 1.0 --z_threshold 4.0")

# # smooth1 machine translation different epsilon 8x5min=40min
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 0.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 1.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 3.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 4.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 5.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 6.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 7.0 --delta 2.0 --z_threshold 4.0")


# # smooth1 machine translation  different gamma 8x5min=40min
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.00 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.10 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.20 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.30 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.40 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.60 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.70 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")

# # smooth1 machine translation  different delta 8x5min=40min
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 0.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 4.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 5.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 6.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 7.0 --z_threshold 4.0")

# # smooth1 machine translation  different z_threshold 8x5min=40min
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 1.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 2.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 3.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 5.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 6.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.50 --gamma 0.50 --epsilon 2.0 --delta 2.0 --z_threshold 7.0")




















# # smooth2 different alpha 3x1.5h=4.5h
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset human_eval --model starcoder --max_new_tokens 400 --min_length 200 --data_lines 100 --temperature_inner 1.0 --alpha 0.0 --gamma 0.4 --epsilon 2.0 --delta 1.0 --z_threshold 4.0")  
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset human_eval --model starcoder --max_new_tokens 400 --min_length 200 --data_lines 100 --temperature_inner 1.0 --alpha 0.2 --gamma 0.4 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset human_eval --model starcoder --max_new_tokens 400 --min_length 200 --data_lines 100 --temperature_inner 1.0 --alpha 0.4 --gamma 0.4 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 

# # smooth2 different gamma 3x1.5h=4.5h 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset human_eval --model starcoder --max_new_tokens 400 --min_length 200 --data_lines 100 --temperature_inner 1.0 --alpha 0.20 --gamma 0.2 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset human_eval --model starcoder --max_new_tokens 400 --min_length 200 --data_lines 100 --temperature_inner 1.0 --alpha 0.20 --gamma 0.4 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 

# # smooth2 different epsilon 2x1.5h=3h
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset human_eval --model starcoder --max_new_tokens 400 --min_length 200 --data_lines 100 --temperature_inner 1.0 --alpha 0.2 --gamma 0.4 --epsilon 0.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset human_eval --model starcoder --max_new_tokens 400 --min_length 200 --data_lines 100 --temperature_inner 1.0 --alpha 0.2 --gamma 0.4 --epsilon 1.0 --delta 1.0 --z_threshold 4.0")  

# # smooth2 different delta 1x1.5h=1.5h
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset human_eval --model starcoder --max_new_tokens 400 --min_length 200 --data_lines 100 --temperature_inner 1.0 --alpha 0.2 --gamma 0.4 --epsilon 2.0 --delta 2.0 --z_threshold 4.0")

# # smooth2 different z_threshold 1x1.5h=1.5h
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset human_eval --model starcoder --max_new_tokens 400 --min_length 200 --data_lines 100 --temperature_inner 1.0 --alpha 0.2 --gamma 0.4 --epsilon 2.0 --delta 1.0 --z_threshold 5.0")





















# # resilience hard
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model gpt2-xl --max_new_tokens 100 --min_length 100 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 

# translation not run due to bug
# os.system("python3 pipeline_smooth_1.py --algorithm EXP --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 2.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth_1.py --algorithm SynthID --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 2.0 --z_threshold 4.0") 

# gpt2-xl soft
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model gpt2-xl --max_new_tokens 100 --min_length 100 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 

# # different temperature soft
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.1 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.5 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.7 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.9 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 

# # different temperature hard
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.1 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.5 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.9 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 

# # temperature=0.7 hard
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model gpt2-xl --max_new_tokens 100 --min_length 100 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 








# # alpha=0
# os.system("python3 pipeline_smooth_1.py --algorithm SMOOTH --dataset c4 --model gpt2-xl --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.0 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 









# output_smooth7_7attacks_c4.txt 1x9=9h
# os.system("python3 pipeline_smooth_2.py --algorithm KGW --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth_2.py --algorithm SWEET --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth_2.py --algorithm EWD --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_2.py --algorithm Unigram --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_2.py --algorithm DIP --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_2.py --algorithm EXP --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_2.py --algorithm SynthID --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_2.py --algorithm SIR --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_2.py --algorithm Unbiased --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")









# output_smooth7_7attacks_wmt16_de_en.txt  0.5hx9≈5h
# os.system("python3 pipeline_smooth_2.py --algorithm KGW --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth_2.py --algorithm SWEET --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth_2.py --algorithm EWD --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_2.py --algorithm Unigram --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_2.py --algorithm DIP --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_2.py --algorithm EXP --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_2.py --algorithm SynthID --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_2.py --algorithm SIR --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_2.py --algorithm Unbiased --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")








# output_smooth7_7attacks_ablation.txt
os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience hard --temperature_inner 0.7 --fixed_pos [] --alpha 0.36 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --entropy_threshold 0.9 --z_threshold 4.0") 
os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience hard --temperature_inner 0.7 --fixed_pos ['n','a'] --alpha 0.36 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --entropy_threshold 0.0 --z_threshold 4.0") 


