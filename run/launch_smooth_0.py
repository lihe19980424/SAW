import os


# # smooth4 different alpha 5x2h=10h
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.0 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0")  
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.1 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.2 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --temperature_inner 1.0 --alpha 0.4 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0") 

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

# # smooth3 machine translation different epsilon 5x5min=25min 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 0.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 0.5 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 1.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 1.5 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 3.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset wmt16_de_en --model nllb-200-distilled-600M --max_new_tokens 200 --min_length 200 --data_lines 200 --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 4.0 --delta 1.0 --z_threshold 4.0")

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


















# supplement 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 

# different token length soft
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 50 --min_length 50 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 100 --min_length 100 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 400 --min_length 400 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 

# w/o sharpness-aware
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 400 --min_length 400 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 20000.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 400 --min_length 400 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 20000.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 20000.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 20000.0 --delta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 20000.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 20000.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model gpt2-xl --max_new_tokens 100 --min_length 100 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 20000.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model gpt2-xl --max_new_tokens 100 --min_length 100 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 20000.0 --delta 1.0 --z_threshold 4.0") 

# # different token length hard
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 50 --min_length 50 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 100 --min_length 100 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 400 --min_length 400 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 

# # temperature=0.7
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.7 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model gpt2-xl --max_new_tokens 100 --min_length 100 --data_lines 200 --resilience soft --temperature_inner 0.7 --alpha 0.3 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 









# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.0 --gamma 0.5 --epsilon 2.0 --delta 1.0 --z_threshold 4.0") 



















# SMOOTH5 hard 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 100 --resilience hard --temperature_inner 0.6 --alpha 0.50 --gamma 0.00 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")

# SMOOTH5 soft 5h 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 100 --resilience soft --temperature_inner 0.6 --alpha 0.10 --gamma 0.40 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 100 --resilience soft --temperature_inner 0.6 --alpha 0.50 --gamma 0.00 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 100 --resilience soft --temperature_inner 0.6 --alpha 0.20 --gamma 0.30 --epsilon 2.0 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 100 --resilience soft --temperature_inner 0.6 --alpha 0.20 --gamma 0.30 --epsilon 0.1 --delta 2.0 --eta 12.0 --z_threshold 4.0")

# eli5 DeepSeek-R1-Distill-Qwen-7B   5h
# os.system("python3 pipeline_smooth.py --algorithm KGW --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.6 --alpha 0.20 --gamma 0.30 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm DIP --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.6 --alpha 0.20 --gamma 0.30 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm Unbiased --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.6 --alpha 0.20 --gamma 0.30 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")  

# SMOOTH5 hard 5h 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 100 --resilience hard --temperature_inner 0.6 --alpha 0.10 --gamma 0.40 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 100 --resilience hard --temperature_inner 0.6 --alpha 0.20 --gamma 0.30 --epsilon 2.0 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 100 --resilience hard --temperature_inner 0.6 --alpha 0.20 --gamma 0.30 --epsilon 0.1 --delta 2.0 --eta 12.0 --z_threshold 4.0")

# other method
# os.system("python3 pipeline_smooth.py --algorithm EXP --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.6 --alpha 0.20 --gamma 0.30 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SIR --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.6 --alpha 0.20 --gamma 0.30 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm Unigram --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.6 --alpha 0.20 --gamma 0.30 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SynthID --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.6 --alpha 0.20 --gamma 0.30 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm EWD --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.6 --alpha 0.20 --gamma 0.30 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SWEET --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.6 --alpha 0.20 --gamma 0.30 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")



















# DeepSeek C4 1.0
# os.system("python3 pipeline_smooth.py --algorithm KGW --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SWEET --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm EWD --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm Unigram --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm DIP --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm EXP --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SynthID --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SIR --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm Unbiased --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")


# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")


# DeepSeek C4 0.7
# os.system("python3 pipeline_smooth.py --algorithm KGW --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.7 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SWEET --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.7 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm EWD --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.7 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm Unigram --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.7 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm DIP --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.7 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm EXP --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.7 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SynthID --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.7 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SIR --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.7 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm Unbiased --dataset c4 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 0.7 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")









# output_smooth5.5_DeepSeekR1_eli5_different_token_length.txt     9x1h=9h
# os.system("python3 pipeline_smooth.py --algorithm KGW --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 100 --min_length 100 --data_lines 200 --resilience hard --temperature_inner 0.6 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SWEET --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 100 --min_length 100 --data_lines 200 --resilience hard --temperature_inner 0.6 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm EWD --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 100 --min_length 100 --data_lines 200 --resilience hard --temperature_inner 0.6 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm Unigram --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 100 --min_length 100 --data_lines 200 --resilience hard --temperature_inner 0.6 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm DIP --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 100 --min_length 100 --data_lines 200 --resilience hard --temperature_inner 0.6 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm EXP --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 100 --min_length 100 --data_lines 200 --resilience hard --temperature_inner 0.6 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SynthID --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 100 --min_length 100 --data_lines 200 --resilience hard --temperature_inner 0.6 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SIR --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 100 --min_length 100 --data_lines 200 --resilience hard --temperature_inner 0.6 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm Unbiased --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 100 --min_length 100 --data_lines 200 --resilience hard --temperature_inner 0.6 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")









# smooth5.5_test.txt
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --alpha 0.20 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 1.0 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")

# smooth5.5_test.txt
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience waved --temperature_inner 1.0 --alpha 0.30 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience soft --temperature_inner 1.0 --alpha 0.30 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience hard --temperature_inner 1.0 --alpha 0.30 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")

# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience hard --temperature_inner 1.0 --alpha 0.10 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")

# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 100 --min_length 100 --data_lines 50 --resilience hard --temperature_inner 1.0 --alpha 0.01 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 100 --min_length 100 --data_lines 50 --resilience hard --temperature_inner 1.0 --alpha 0.25 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 100 --min_length 100 --data_lines 50 --resilience hard --temperature_inner 1.0 --alpha 0.49 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")

# os.system("python3 pipeline_smooth.py --algorithm KGW --dataset c4 --model opt-1.3b --max_new_tokens 100 --min_length 100 --data_lines 50 --resilience hard --temperature_inner 1.0 --alpha 0.49 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")








# output_smooth5.5_DeepSeekR1_eli5_different_token_length.txt     9x1h=9h
# os.system("python3 pipeline_smooth.py --algorithm KGW --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SWEET --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm EWD --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm Unigram --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm DIP --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm EXP --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SynthID --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SIR --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm Unbiased --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")


# # output_smooth5.5_DeepSeekR1_eli5_different_token_length.txt     9x1.5h=13.5h
# os.system("python3 pipeline_smooth.py --algorithm KGW --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SWEET --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm EWD --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm Unigram --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm DIP --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm EXP --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SynthID --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SIR --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm Unbiased --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")


# # output_smooth5.5_DeepSeekR1_eli5_different_token_length.txt     9x2h=18h
# os.system("python3 pipeline_smooth.py --algorithm KGW --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 400 --min_length 400 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SWEET --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 400 --min_length 400 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm EWD --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 400 --min_length 400 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm Unigram --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 400 --min_length 400 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm DIP --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 400 --min_length 400 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm EXP --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 400 --min_length 400 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SynthID --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 400 --min_length 400 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SIR --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 400 --min_length 400 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm Unbiased --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 400 --min_length 400 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")

# synthID
# os.system("python3 pipeline_smooth.py --algorithm SynthID --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SynthID --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SynthID --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 400 --min_length 400 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")








# output_smooth7_7attacks_eli5.txt  0.5hx9≈5h
# os.system("python3 pipeline_smooth.py --algorithm KGW --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SWEET --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm EWD --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm Unigram --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm DIP --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm EXP --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SynthID --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SIR --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm Unbiased --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")








# output_smooth7_7attacks_eli5.txt  0.5hx9≈5h  ["n", "a", "p", "v", "r"] ['n', 'a', 'p', 'v', 'r']
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7  --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")










# output_smooth7_7attacks_pos.txt.txt  3hx3≈9h
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --fixed_pos ['n'] --alpha 0.34 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience hard --temperature_inner 0.7 --fixed_pos ['n','a'] --alpha 0.36 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience hard --temperature_inner 0.7 --fixed_pos ['n','a','v'] --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience hard --temperature_inner 0.7 --fixed_pos ['n','a','v','r'] --alpha 0.41 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")









# output_smooth7_7attacks_flickr30k.txt  0.2hx9≈2h
# os.system("python3 pipeline_smooth.py --algorithm KGW --dataset flickr30k --model BLIP --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SWEET --dataset flickr30k --model BLIP --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm EWD --dataset flickr30k --model BLIP --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm Unigram --dataset flickr30k --model BLIP --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm DIP --dataset flickr30k --model BLIP --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm EXP --dataset flickr30k --model BLIP --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SynthID --dataset flickr30k --model BLIP --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SIR --dataset flickr30k --model BLIP --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm Unbiased --dataset flickr30k --model BLIP --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience hard --temperature_inner 0.7 --alpha 0.40 --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0")








# output_smooth7_7attacks_eta.txt
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience soft --temperature_inner 0.7 --fixed_pos navr --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience soft --temperature_inner 0.7 --fixed_pos navr --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 2.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience soft --temperature_inner 0.7 --fixed_pos navr --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 4.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience soft --temperature_inner 0.7 --fixed_pos navr --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 8.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience soft --temperature_inner 0.7 --fixed_pos navr --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 16.0 --z_threshold 4.0") 










# output_smooth7_7attacks_eta.txt
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience soft --temperature_inner 0.7 --fixed_pos ['n','a'] --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience soft --temperature_inner 0.7 --fixed_pos ['n','a'] --gamma 0.50 --epsilon 0.5 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience soft --temperature_inner 0.7 --fixed_pos ['n','a'] --gamma 0.50 --epsilon 1.0 --delta 2.0 --eta 3.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience soft --temperature_inner 0.7 --fixed_pos ['n','a'] --gamma 0.50 --epsilon 1.5 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 200 --resilience soft --temperature_inner 0.7 --fixed_pos ['n','a'] --gamma 0.50 --epsilon 2.0 --delta 2.0 --eta 3.0 --z_threshold 4.0") 








# output_smooth7_7attacks_sharpness_aware.txt
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience hard --temperature_inner 0.7 --fixed_pos navr --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience waved --temperature_inner 0.7 --fixed_pos navr --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm KGW --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience waved --temperature_inner 0.7 --fixed_pos navr --gamma 0.50 --epsilon 0.1 --delta 2.0 --eta 3.0 --z_threshold 4.0") 









# output_smooth8_6attacks_c4_llama-7b-hf.txt
os.system("python3 pipeline_smooth.py --algorithm KGW --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience soft --temperature_inner 1.0 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SWEET --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience soft --temperature_inner 1.0 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm EWD --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience soft --temperature_inner 1.0 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm Unigram --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience soft --temperature_inner 1.0 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm DIP --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience soft --temperature_inner 1.0 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm EXP --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience soft --temperature_inner 1.0 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SynthID --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience soft --temperature_inner 1.0 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm SIR --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience soft --temperature_inner 1.0 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0") 
# os.system("python3 pipeline_smooth.py --algorithm Unbiased --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience soft --temperature_inner 1.0 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth.py --algorithm SMOOTH --dataset c4 --model llama-7b-hf --max_new_tokens 200 --min_length 200 --data_lines 100 --resilience soft --temperature_inner 1.0 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0")









# output_smooth8_8attacks_opt_navr41_soft_tem1_.txt
# os.system("python3 pipeline_smooth_0.py --algorithm SMOOTH --dataset c4 --model opt-1.3b --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_0.py --algorithm SMOOTH --dataset cnn_daily_mail --model Llama-3-8B-Instruct --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_0.py --algorithm SMOOTH --dataset rocstories --model Qwen2.5-0.5B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0")
# os.system("python3 pipeline_smooth_0.py --algorithm SMOOTH --dataset eli5 --model DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 300 --min_length 300 --data_lines 200 --resilience soft --temperature_inner 1.0 --fixed_pos navr --alpha 0.41 --gamma 0.50 --epsilon 0.0 --delta 2.0 --eta 1.0 --z_threshold 4.0")