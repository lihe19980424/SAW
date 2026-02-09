# Ablation


# import matplotlib.pyplot as plt
# import numpy as np
# import re
# from scipy.interpolate import make_interp_spline

# # ==========================================
# # 1. 数据解析函数 (保持不变)
# # ==========================================
# def parse_log(file_path):
#     with open(file_path, 'r') as f:
#         content = f.read()
    
#     blocks = content.split('=========================')
#     results = []
#     for block in blocks:
#         if not block.strip(): continue
        
#         # 提取 Beta, PPL, 以及 TPR
#         beta_match = re.search(r"'beta': ([\d.]+)", block)
#         ppl_match = re.search(r"PPL:\s+\{'watermarked': ([\d.]+)", block)
#         tpr_clean_match = re.search(r"detection accuracy of fpr=1% :\s+\{'TPR': ([\d.]+)", block)
        
#         # 提取 Word-D_7
#         attack_match = re.search(r"detection accuracy of attack_Word-D_7:\s+\{'TPR': ([\d.]+)", block)
        
#         if beta_match and ppl_match and tpr_clean_match:
#             beta = float(beta_match.group(1))
#             ppl = float(ppl_match.group(1))
#             tpr_clean = float(tpr_clean_match.group(1))
#             tpr_attack = float(attack_match.group(1)) if attack_match else 0.0
#             results.append({'beta': beta, 'ppl': ppl, 'tpr_clean': tpr_clean, 'tpr_attack': tpr_attack})
    
#     results.sort(key=lambda x: x['beta'])
#     return results

# # 解析文件
# file_name = '/home/lihe/MarkLLM/output/output_saw_KDD/output_saw_KDD_c4_5attacks_temp_1_tokens_200_datalines_100_beta.txt'
# data_points = parse_log(file_name)

# betas = np.array([d['beta'] for d in data_points])
# tpr_clean = np.array([d['tpr_clean'] for d in data_points])
# tpr_attack = np.array([d['tpr_attack'] for d in data_points])
# inv_ppl = np.array([1 / d['ppl'] for d in data_points])

# # ==========================================
# # 2. 平滑处理函数
# # ==========================================
# def smooth_curve(x, y, num_points=300, k=2):
#     """
#     使用 B-Spline 进行平滑插值
#     k=2 (quadratic) 或 k=3 (cubic) 通常效果最好
#     """
#     x_new = np.linspace(x.min(), x.max(), num_points)
#     spl = make_interp_spline(x, y, k=k)
#     y_smooth = spl(x_new)
#     # 限制 y 值范围，防止插值产生的过冲 (overshoot)
#     y_smooth = np.clip(y_smooth, y.min()*0.95, y.max()*1.05) 
#     return x_new, y_smooth

# # 生成平滑数据
# x_smooth, tpr_clean_smooth = smooth_curve(betas, tpr_clean)
# _, tpr_attack_smooth = smooth_curve(betas, tpr_attack)
# _, inv_ppl_smooth = smooth_curve(betas, inv_ppl)

# # ==========================================
# # 3. 绘图设置 (适配双栏论文中的 1/4 宽度位置)
# # ==========================================
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['axes.labelsize'] = 11
# plt.rcParams['xtick.labelsize'] = 10
# plt.rcParams['ytick.labelsize'] = 10
# plt.rcParams['legend.fontsize'] = 9

# # 紧凑型画布
# fig, ax1 = plt.subplots(figsize=(3.5, 3.0)) 

# # --- 绘制左侧轴 (TPR) ---
# color_clean = '#1F618D' # 深蓝
# color_attack = '#239B56' # 深绿

# ax1.set_xlabel(r'Hyperparameter $\beta$', fontsize=12)
# ax1.set_ylabel('Detection TPR', color='black')

# # 1. 画散点 (原始数据)
# ax1.scatter(betas, tpr_clean, color=color_clean, marker='o', s=30, alpha=0.6, label='_nolegend_')
# ax1.scatter(betas, tpr_attack, color=color_attack, marker='s', s=30, alpha=0.6, label='_nolegend_')

# # 2. 画平滑曲线
# l1, = ax1.plot(x_smooth, tpr_clean_smooth, color=color_clean, linewidth=2.0, label='Clean')
# l2, = ax1.plot(x_smooth, tpr_attack_smooth, color=color_attack, linewidth=2.0, linestyle='--', label='Word-D (0.7)')

# ax1.set_ylim(0.75, 1.05)
# ax1.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# ax1.grid(True, linestyle=':', alpha=0.5)

# # --- 绘制右侧轴 (1/PPL) ---
# ax2 = ax1.twinx()
# color_ppl = '#C0392B' # 深红
# ax2.set_ylabel(r'Quality ($1/\mathrm{PPL}$)', color=color_ppl)

# # 3. 画散点 (原始数据)
# ax2.scatter(betas, inv_ppl, color=color_ppl, marker='^', s=35, alpha=0.6, label='_nolegend_')

# # 4. 画平滑曲线
# l3, = ax2.plot(x_smooth, inv_ppl_smooth, color=color_ppl, linewidth=2.0, linestyle='-.', label=r'$1/\mathrm{PPL}$')

# ax2.tick_params(axis='y', labelcolor=color_ppl)
# # 动态调整右轴范围
# y2_min, y2_max = min(inv_ppl), max(inv_ppl)
# ax2.set_ylim(y2_min * 0.95, y2_max * 1.05)

# # ==========================================
# # 4. 布局与导出
# # ==========================================
# # 合并图例并置于图表上方
# lines = [l1, l2, l3]
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), 
#            ncol=3, frameon=False, columnspacing=0.8)

# plt.tight_layout()
# plt.savefig("SAW_Sensitivity_Beta_Smoothed.pdf", format='pdf', bbox_inches='tight')
# plt.show()

# print("平滑拟合版灵敏度分析图已生成。")









# import matplotlib.pyplot as plt
# import numpy as np
# import re
# from scipy.interpolate import make_interp_spline

# # ==========================================
# # 1. 数据解析 (解析 std 相关的日志文件)
# # ==========================================
# def parse_std_log(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         content = f.read()
    
#     blocks = content.split('=========================')
#     results = []
#     for block in blocks:
#         if not block.strip(): continue
        
#         # 提取关键数据：Std, PPL, 以及无攻击下的 TPR
#         std_match = re.search(r"'std': ([\d.]+)", block)
#         ppl_match = re.search(r"PPL:\s+\{'watermarked': ([\d.]+)", block)
#         fpr1_match = re.search(r"detection accuracy of fpr=1% :\s+\{'TPR': ([\d.]+)", block)
        
#         if std_match and ppl_match and fpr1_match:
#             results.append({
#                 'std': float(std_match.group(1)),
#                 'ppl': float(ppl_match.group(1)),
#                 'tpr_clean': float(fpr1_match.group(1))
#             })
    
#     # 按 Std 排序以确保插值曲线轨迹正确
#     results.sort(key=lambda x: x['std'])
#     return results

# # 解析数据文件
# # 请确保该文件在您的当前运行目录下
# log_file = '/home/lihe/MarkLLM/output/output_saw_KDD/output_saw_KDD_c4_5attacks_temp_1_tokens_200_datalines_100_std.txt'
# data = parse_std_log(log_file)

# x_std = np.array([d['std'] for d in data])
# y_clean = np.array([d['tpr_clean'] for d in data])
# y_inv_ppl = np.array([1 / d['ppl'] for d in data])

# # ==========================================
# # 2. 曲线平滑处理 (Spline Interpolation)
# # ==========================================
# def get_smooth(x, y):
#     x_new = np.linspace(x.min(), x.max(), 300)
#     # 使用二次样条插值 (k=2) 实现平滑
#     spl = make_interp_spline(x, y, k=2) 
#     y_smooth = spl(x_new)
#     # 限制 TPR 范围在合理区间
#     if "tpr" in str(y).lower() or (np.max(y) <= 1.0):
#         y_smooth = np.clip(y_smooth, 0.0, 1.02)
#     return x_new, y_smooth

# x_smooth, y_clean_s = get_smooth(x_std, y_clean)
# _, y_inv_ppl_s = get_smooth(x_std, y_inv_ppl)

# # ==========================================
# # 3. 绘图 (适配双栏论文单侧展示)
# # ==========================================
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['axes.labelsize'] = 11
# plt.rcParams['xtick.labelsize'] = 10
# plt.rcParams['ytick.labelsize'] = 10
# plt.rcParams['legend.fontsize'] = 9

# fig, ax1 = plt.subplots(figsize=(3.5, 3.0))

# # --- 左轴: TPR (检测效能) ---
# c_clean = '#1F618D' # 深蓝色
# ax1.set_xlabel(r'Standard Deviation ($\sigma$)', fontsize=12)
# ax1.set_ylabel('Detection TPR', color='black')

# # 绘制原始数据散点
# ax1.scatter(x_std, y_clean, color=c_clean, marker='o', s=30, alpha=0.5, label='_nolegend_')

# # 绘制平滑拟合曲线 (Clean)
# l1, = ax1.plot(x_smooth, y_clean_s, color=c_clean, linewidth=2.0, label='Clean TPR')

# ax1.set_ylim(-0.05, 1.1)
# ax1.set_xticks([0.01, 0.03, 0.05, 0.07, 0.09])
# ax1.grid(True, linestyle=':', alpha=0.5)

# # --- 右轴: 1/PPL (文本质量) ---
# ax2 = ax1.twinx()
# c_ppl = '#C0392B' # 深红色
# ax2.set_ylabel(r'Quality ($1/\mathrm{PPL}$)', color=c_ppl)

# # 绘制原始数据散点
# ax2.scatter(x_std, y_inv_ppl, color=c_ppl, marker='^', s=35, alpha=0.5, label='_nolegend_')

# # 绘制平滑拟合曲线 (Quality)
# l3, = ax2.plot(x_smooth, y_inv_ppl_s, color=c_ppl, linewidth=2.0, linestyle='-.', label=r'Quality ($1/\mathrm{PPL}$)')

# ax2.tick_params(axis='y', labelcolor=c_ppl)
# # 动态调整刻度范围，增加上下余量
# y2_min, y2_max = min(y_inv_ppl), max(y_inv_ppl)
# ax2.set_ylim(y2_min * 0.95, y2_max * 1.05)

# # --- 图例与布局优化 ---
# # 仅保留两条线的图例，水平排列在上方
# lines = [l1, l3]
# ax1.legend(lines, [l.get_label() for l in lines], loc='lower center', 
#            bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, columnspacing=1.0)

# plt.tight_layout()
# plt.savefig("SAW_Sensitivity_Std_Smoothed.pdf", format='pdf', bbox_inches='tight')
# plt.show()

# print("精简版标准差灵敏度分析图已生成。")









# import matplotlib.pyplot as plt
# import numpy as np
# import re
# from scipy.interpolate import make_interp_spline

# # ==========================================
# # 1. 数据解析 (解析新的 mean 实验日志)
# # ==========================================
# def parse_mean_log(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         content = f.read()
    
#     blocks = content.split('=========================')
#     results = []
#     for block in blocks:
#         if not block.strip(): continue
        
#         # 提取 Mean, PPL, 及 TPR 数据
#         mean_match = re.search(r"'mean': ([\d.]+)", block)
#         ppl_match = re.search(r"PPL:\s+\{'watermarked': ([\d.]+)", block)
#         fpr1_match = re.search(r"detection accuracy of fpr=1% :\s+\{'TPR': ([\d.]+)", block)
        
#         # 提取 Word-S (Substitution) 攻击数据 (指定为 attack_Word-S_7)
#         attack_match = re.search(r"detection accuracy of attack_Word-S_7:\s+\{'TPR': ([\d.]+)", block)
        
#         if mean_match and ppl_match and fpr1_match:
#             results.append({
#                 'mean': float(mean_match.group(1)),
#                 'ppl': float(ppl_match.group(1)),
#                 'tpr_clean': float(fpr1_match.group(1)),
#                 'tpr_attack': float(attack_match.group(1)) if attack_match else 0.0
#             })
    
#     # 按 Mean 排序以进行正确插值
#     results.sort(key=lambda x: x['mean'])
#     return results

# # 解析新数据文件
# log_file = '/home/lihe/MarkLLM/output/output_saw_KDD/output_saw_KDD_c4_opt_7attacks_temp_1_tokens_200_datalines_100_mean_new.txt'
# data = parse_mean_log(log_file)

# x_mean = np.array([d['mean'] for d in data])
# y_clean = np.array([d['tpr_clean'] for d in data])
# y_attack = np.array([d['tpr_attack'] for d in data])
# y_inv_ppl = np.array([1 / d['ppl'] for d in data])

# # ==========================================
# # 2. 曲线平滑处理 (Spline Interpolation)
# # ==========================================
# def get_smooth(x, y):
#     x_new = np.linspace(x.min(), x.max(), 300)
#     # 针对 4 个数据点，使用 k=2 (二次样条) 能在保持平滑的同时避免震荡
#     spl = make_interp_spline(x, y, k=2) 
#     y_smooth = spl(x_new)
#     # 限制 TPR 上限，防止插值导致的过冲
#     if "tpr" in str(y).lower():
#         y_smooth = np.clip(y_smooth, 0.0, 1.0)
#     return x_new, y_smooth

# x_smooth, y_clean_s = get_smooth(x_mean, y_clean)
# _, y_attack_s = get_smooth(x_mean, y_attack)
# _, y_inv_ppl_s = get_smooth(x_mean, y_inv_ppl)

# # ==========================================
# # 3. 绘图 (适配双栏论文 1/4 宽度)
# # ==========================================
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['axes.labelsize'] = 11
# plt.rcParams['xtick.labelsize'] = 10
# plt.rcParams['ytick.labelsize'] = 10
# plt.rcParams['legend.fontsize'] = 9

# fig, ax1 = plt.subplots(figsize=(3.5, 3.0))

# # --- 左轴: TPR (检测率) ---
# c_clean, c_attack = '#1F618D', '#239B56'
# ax1.set_xlabel(r'Mean of Noise ($\mu$)', fontsize=12)
# ax1.set_ylabel('Detection TPR', color='black')

# # 绘制原始数据散点 (透明度设为 0.5)
# ax1.scatter(x_mean, y_clean, color=c_clean, marker='o', s=30, alpha=0.5, label='_nolegend_')
# ax1.scatter(x_mean, y_attack, color=c_attack, marker='s', s=30, alpha=0.5, label='_nolegend_')

# # 绘制平滑拟合曲线
# l1, = ax1.plot(x_smooth, y_clean_s, color=c_clean, linewidth=1.8, label='Clean')
# l2, = ax1.plot(x_smooth, y_attack_s, color=c_attack, linewidth=1.8, linestyle='--', label='Word-S (0.7)')

# ax1.set_ylim(0.8, 1.05) # 根据 F1/TPR 数据调整范围
# ax1.set_xticks(x_mean) # 强制显示实验的 4 个均值点
# ax1.grid(True, linestyle=':', alpha=0.5)

# # --- 右轴: 1/PPL (文本质量) ---
# ax2 = ax1.twinx()
# c_ppl = '#C0392B'
# ax2.set_ylabel(r'Quality ($1/\mathrm{PPL}$)', color=c_ppl)

# ax2.scatter(x_mean, y_inv_ppl, color=c_ppl, marker='^', s=35, alpha=0.5, label='_nolegend_')
# l3, = ax2.plot(x_smooth, y_inv_ppl_s, color=c_ppl, linewidth=1.8, linestyle='-.', label=r'$1/\mathrm{PPL}$')

# ax2.tick_params(axis='y', labelcolor=c_ppl)
# # 自动调整右轴刻度
# y2_min, y2_max = min(y_inv_ppl), max(y_inv_ppl)
# ax2.set_ylim(y2_min * 0.95, y2_max * 1.05)

# # --- 图例与导出 ---
# lines = [l1, l2, l3]
# ax1.legend(lines, [l.get_label() for l in lines], loc='lower center', 
#            bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, columnspacing=0.6)

# plt.tight_layout()
# plt.savefig("SAW_Sensitivity_Mean_Smoothed.pdf", format='pdf', bbox_inches='tight')
# plt.show()

# print("噪声均值灵敏度分析图（替换攻击版）已生成。")










# import matplotlib.pyplot as plt
# import numpy as np
# import re
# from scipy.interpolate import make_interp_spline

# # ==========================================
# # 1. 数据解析 (筛选指定的 topk 值)
# # ==========================================
# def parse_topk_log(file_path, selected_topks):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         content = f.read()
    
#     blocks = content.split('=========================')
#     results = []
#     for block in blocks:
#         if not block.strip(): continue
        
#         # 提取 topk, PPL, 及 无攻击 TPR (fpr=1%)
#         topk_match = re.search(r"'topk': ([\d.]+)", block)
#         ppl_match = re.search(r"PPL:\s+\{'watermarked': ([\d.]+)", block)
#         fpr1_match = re.search(r"detection accuracy of fpr=1% :\s+\{'TPR': ([\d.]+)", block)
        
#         if topk_match and ppl_match and fpr1_match:
#             topk_val = int(float(topk_match.group(1)))
#             # 仅保留指定的 topk 值
#             if topk_val in selected_topks:
#                 results.append({
#                     'topk': topk_val,
#                     'ppl': float(ppl_match.group(1)),
#                     'tpr_clean': float(fpr1_match.group(1))
#                 })
    
#     # 排序
#     results.sort(key=lambda x: x['topk'])
#     return results

# # 指定要展示的 topk 值
# selected_values = [25, 35, 50, 65, 100]
# log_file = '/home/lihe/MarkLLM/output/output_saw_KDD/output_saw_KDD_c4_opt_7attacks_temp_1_tokens_200_datalines_100_topk.txt'
# data = parse_topk_log(log_file, selected_values)

# x_topk = np.array([d['topk'] for d in data])
# y_clean = np.array([d['tpr_clean'] for d in data])
# y_inv_ppl = np.array([1 / d['ppl'] for d in data])

# # ==========================================
# # 2. 曲线平滑处理 (线性空间 Spline)
# # ==========================================
# def get_smooth(x, y):
#     x_new = np.linspace(x.min(), x.max(), 300)
#     # 针对少量数据点，使用 k=2 (二次样条) 效果最自然
#     spl = make_interp_spline(x, y, k=2) 
#     y_smooth = spl(x_new)
#     # 限制 TPR 范围
#     if np.max(y) <= 1.0:
#         y_smooth = np.clip(y_smooth, 0.0, 1.02)
#     return x_new, y_smooth

# x_smooth, y_clean_s = get_smooth(x_topk, y_clean)
# _, y_inv_ppl_s = get_smooth(x_topk, y_inv_ppl)

# # ==========================================
# # 3. 绘图 (适配双栏论文 1/4 宽度)
# # ==========================================
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['axes.labelsize'] = 11
# plt.rcParams['xtick.labelsize'] = 10
# plt.rcParams['ytick.labelsize'] = 10
# plt.rcParams['legend.fontsize'] = 9

# fig, ax1 = plt.subplots(figsize=(3.5, 3.0))

# # --- 左轴: TPR (检测率) ---
# c_clean = '#1F618D' # 深蓝
# ax1.set_xlabel(r'Filtering Top-$k$', fontsize=12)
# ax1.set_ylabel('Detection TPR', color='black')

# # 原始数据散点
# ax1.scatter(x_topk, y_clean, color=c_clean, marker='o', s=30, alpha=0.5, label='_nolegend_')

# # 平滑拟合曲线 (Clean)
# l1, = ax1.plot(x_smooth, y_clean_s, color=c_clean, linewidth=2.0, label='Clean TPR')

# ax1.set_ylim(0.9, 1.05) # 聚焦在高性能区间
# ax1.set_xticks(selected_values) # 强制显示指定的刻度
# ax1.grid(True, linestyle=':', alpha=0.5)

# # --- 右轴: 1/PPL (文本质量) ---
# ax2 = ax1.twinx()
# c_ppl = '#C0392B' # 深红
# ax2.set_ylabel(r'Quality ($1/\mathrm{PPL}$)', color=c_ppl)

# # 原始数据散点
# ax2.scatter(x_topk, y_inv_ppl, color=c_ppl, marker='^', s=35, alpha=0.5, label='_nolegend_')

# # 平滑拟合曲线 (1/PPL)
# l3, = ax2.plot(x_smooth, y_inv_ppl_s, color=c_ppl, linewidth=2.0, linestyle='-.', label=r'Quality ($1/\mathrm{PPL}$)')

# ax2.tick_params(axis='y', labelcolor=c_ppl)
# # 动态调整刻度范围
# y2_min, y2_max = min(y_inv_ppl), max(y_inv_ppl)
# ax2.set_ylim(y2_min * 0.95, y2_max * 1.05)

# # --- 图例与布局优化 ---
# lines = [l1, l3]
# ax1.legend(lines, [l.get_label() for l in lines], loc='lower center', 
#            bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, columnspacing=1.0)

# plt.tight_layout()
# plt.savefig("SAW_Sensitivity_Topk_LogSmoothed.pdf", format='pdf', bbox_inches='tight')
# plt.show()

# print("Top-k 灵敏度分析图（精简平滑版）已生成。")