# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import font_manager, rcParams

# # =========================
# # Load local Arial font
# # =========================
# font_path = './font/arial.ttf'
# font_manager.fontManager.addfont(font_path)
# prop = font_manager.FontProperties(fname=font_path)

# rcParams['font.family'] = prop.get_name()
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42
# rcParams['axes.labelsize'] = 18
# rcParams['xtick.labelsize'] = 16
# rcParams['ytick.labelsize'] = 16
# rcParams['legend.fontsize'] = 14

# # =========================
# # Data
# # =========================
# x = np.array([0.0, 0.3, 0.5, 0.7, 0.9, 1.0])

# no_attack = np.array([1.0000, 0.9950, 0.9851, 0.9950, 1.0000, 0.9950])

# deletion = np.array([0.7843, 0.8855, 0.9216, 0.9412, 0.9538, 0.9744])

# # =========================
# # Plot
# # =========================
# fig, ax = plt.subplots(figsize=(6.8, 4.8))

# ax.plot(x, no_attack,
#         color='#1f77b4', marker='o', linewidth=2.2,
#         label='No Attack')

# ax.plot(x, deletion,
#         color='#ff7f0e', marker='^', linestyle='--', linewidth=2.0,
#         label='Word Deletion Attack (Ratio = 0.7)')

# # =========================
# # Axis & style
# # =========================
# ax.set_xlabel('Proportion of global noise $\\beta$')
# ax.set_ylabel('Detection F1 Score')
# ax.set_xlim(0.0, 1.0)
# ax.set_ylim(0.65, 1.02)

# ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.5)
# ax.legend(loc='lower right', frameon=False)

# plt.tight_layout()
# plt.savefig(
#     './SAW_beta_F1.pdf',
#     dpi=1000,
#     bbox_inches='tight'
# )
# plt.show()









# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import font_manager, rcParams

# # =========================
# # Load local Arial font
# # =========================
# font_path = './font/arial.ttf'
# font_manager.fontManager.addfont(font_path)
# prop = font_manager.FontProperties(fname=font_path)

# rcParams['font.family'] = prop.get_name()
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42
# rcParams['axes.labelsize'] = 18
# rcParams['xtick.labelsize'] = 16
# rcParams['ytick.labelsize'] = 16
# rcParams['legend.fontsize'] = 14

# # =========================
# # Data
# # =========================
# x = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])

# no_attack = np.array([0.7417, 0.8018, 0.9268, 0.9901, 0.9901,  0.9950, 1.0000, 1.0000, 1.0000])

# deletion = np.array([0.6721, 0.7957, 0.8093, 0.9238, 0.9703, 0.9694, 0.9950, 0.9950, 1.0000])

# substitution = np.array([0.7099, 0.7300, 0.8163, 0.8436, 0.9055, 0.9245, 0.9252, 0.9655, 0.9504])

# # =========================
# # Plot
# # =========================
# fig, ax = plt.subplots(figsize=(6.8, 4.8))

# ax.plot(x, no_attack,
#         color='#1f77b4', marker='o', linewidth=2.2,
#         label='No Attack')

# ax.plot(x, deletion,
#         color='#ff7f0e', marker='^', linestyle='--', linewidth=2.0,
#         label='Word Deletion Attack (Ratio = 0.7)')

# ax.plot(x, substitution,
#         color='#2ca02c', marker='s', linestyle='-.', linewidth=2.0,
#         label='Word Substitution Attack (Ratio = 0.7)')

# # =========================
# # Axis & style
# # =========================
# ax.set_xlabel('Different standard deviation $\\sigma$')
# ax.set_ylabel('Detection F1 Score')
# ax.set_xlim(0.01, 0.09)
# ax.set_ylim(0.65, 1.02)

# ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.5)
# ax.legend(loc='lower right', frameon=False)

# plt.tight_layout()
# plt.savefig(
#     './SAW_std_F1.pdf',
#     dpi=1000,
#     bbox_inches='tight'
# )
# plt.show()











# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import font_manager, rcParams

# # =========================
# # Load local Arial font
# # =========================
# font_path = './font/arial.ttf'
# font_manager.fontManager.addfont(font_path)
# prop = font_manager.FontProperties(fname=font_path)

# rcParams['font.family'] = prop.get_name()
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42
# rcParams['axes.labelsize'] = 18
# rcParams['xtick.labelsize'] = 16
# rcParams['ytick.labelsize'] = 16
# rcParams['legend.fontsize'] = 14

# # =========================
# # Data
# # =========================
# x = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])

# perplexity = np.array([11.5340, 11.5803, 12.0803, 12.2656, 12.5281, 13.3440, 13.4739, 13.3404, 14.3286])

# # =========================
# # Plot
# # =========================
# fig, ax = plt.subplots(figsize=(6.8, 4.8))

# ax.plot(x, perplexity,
#         color='#e41a1c', marker='s', linestyle='--', linewidth=2.0,
#         label='Text quality')

# # =========================
# # Axis & style
# # =========================
# ax.set_xlabel('Different standard deviations $\\sigma$')
# ax.set_ylabel('Perplexity (PPL)')
# ax.set_xlim(0.01, 0.09)
# ax.set_ylim(10.0000, 15.0000)

# ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.5)
# ax.legend(loc='lower right', frameon=False)

# plt.tight_layout()
# plt.savefig(
#     './SAW_std_PPL.pdf',
#     dpi=1000,
#     bbox_inches='tight'
# )
# plt.show()









# import os
# from matplotlib import font_manager
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# # ==========================================
# # 1. 全局样式设置（使用本地 Arial 字体）
# # ==========================================

# # 字体文件路径（与你的 python 文件同级）
# font_path = os.path.join(os.path.dirname(__file__), "font", "arial.ttf")

# # 注册字体
# font_manager.fontManager.addfont(font_path)

# # 获取字体的真实 family name
# arial_font = font_manager.FontProperties(fname=font_path)
# arial_name = arial_font.get_name()

# print("Loaded font family:", arial_name)

# # 全局设置
# # ==========================================
# plt.rcParams['font.family'] = arial_name
# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.rm'] = arial_name
# plt.rcParams['mathtext.it'] = arial_name
# plt.rcParams['mathtext.bf'] = arial_name
# plt.rcParams['mathtext.cal'] = arial_name
# plt.rcParams['axes.unicode_minus'] = False

# # ==========================================
# # 2. 数据准备
# # ==========================================
# # 移除了 "Unigram" 和 "EXP"
# # 顺序: 0:KGW, 1:SWEET, 2:EWD, 3:DiPmark, 4:SynthID, 5:SIR, 6:SAW
# method_names = ["KGW", "SWEET", "EWD", "DiPmark", "SynthID", "SIR", "SAW (Ours)"]

# # 更新后的数据
# avg_f1 = np.array([0.8358, 0.8359, 0.9007, 0.6808, 0.6993, 0.7229, 0.9283])
# rc_ppl = np.array([28.54, 20.33, 28.70, 15.79, 18.99, 26.58, 14.32])

# # ==========================================
# # 3. 计算 Pareto Frontier (仅针对基线)
# # ==========================================
# # 逻辑调整：因为 SAW 甚至在质量上也超过了 DiPmark，在鲁棒性上超过了 EWD，
# # 它完全支配了所有基线。为了画出“对比线”，我们只计算基线方法的 Pareto 前沿，
# # 从而展示 SAW 是如何突破现有技术瓶颈的。

# # 提取基线数据 (前6个)
# baseline_f1 = avg_f1[:-1]
# baseline_ppl = rc_ppl[:-1]

# # 目标：F1 越大越好，PPL 越小越好
# points = np.column_stack((baseline_f1, -baseline_ppl))
# is_pareto = np.ones(points.shape[0], dtype=bool)
# for i, c in enumerate(points):
#     if is_pareto[i]:
#         is_pareto[i] = not np.any(np.all(points >= c, axis=1) & np.any(points > c, axis=1))

# pareto_indices = np.where(is_pareto)[0]
# pareto_points = points[pareto_indices]

# # 按 F1 排序以便连线
# sorted_indices = np.argsort(pareto_points[:, 0])
# pareto_x = pareto_points[sorted_indices, 0]
# pareto_y = -pareto_points[sorted_indices, 1]  # 还原 PPL 正值

# # ==========================================
# # 4. 绘图与视觉设计
# # ==========================================
# fig, ax = plt.subplots(figsize=(10, 8))
# ax.grid(True, linestyle=':', alpha=0.5, color='gray', zorder=0)

# # -------------------------------------------------------
# # A. 绘制背景“优化方向”箭头
# # -------------------------------------------------------
# # 调整箭头位置以适应新的坐标轴范围
# ax.annotate('', xy=(0.95, 12), xytext=(0.80, 32),
#             arrowprops=dict(arrowstyle="->", color='#E0E0E0', lw=5, mutation_scale=40),
#             zorder=0)
# ax.text(0.92, 14, "Better Performance", rotation=-38, fontsize=18,
#         color='#BDBDBD', ha='center', weight='bold', zorder=0, style='italic')

# # -------------------------------------------------------
# # B. 绘制 Pareto Frontier (基线前沿)
# # -------------------------------------------------------
# # 绘制虚线
# ax.plot(pareto_x, pareto_y, color='#7F8C8D', linestyle='--', linewidth=2.5, alpha=0.6, zorder=1)

# # 阴影区域 (填充基线前沿下方的区域)
# # 扩展边界以填充底部
# fill_x = np.concatenate(([pareto_x[0]], pareto_x, [pareto_x[-1]]))
# fill_y = np.concatenate(([40], pareto_y, [40]))  # 40 是底部界限 (PPL值大代表差)
# ax.fill_between(pareto_x, pareto_y, 40, color='#D7DBDD', alpha=0.3, zorder=0)

# # -------------------------------------------------------
# # C. 绘制散点
# # -------------------------------------------------------
# # 颜色映射 (共7个)
# # KGW(灰), SWEET(灰), EWD(灰), DiPmark(紫), SynthID(深蓝), SIR(橙), SAW(红)
# colors = ['#7F8C8D', '#7F8C8D', '#7F8C8D', '#8E44AD', '#2C3E50', '#E67E22', '#C0392B']
# markers = ['o', 's', '^', 'v', 'p', 'h', '*']

# # 动态计算 sizes
# sizes = [180] * (len(method_names) - 1) + [700]  # SAW 更大

# for i in range(len(method_names)):
#     is_saw = (i == len(method_names) - 1)

#     edge_c = 'black' if is_saw else 'white'
#     lw = 2.0 if is_saw else 1.0
#     z = 20 if is_saw else 10
#     alpha_v = 1.0 if is_saw else 0.9

#     ax.scatter(avg_f1[i], rc_ppl[i], c=colors[i], marker=markers[i], s=sizes[i],
#                edgecolor=edge_c, linewidth=lw, zorder=z, alpha=alpha_v)

# # -------------------------------------------------------
# # D. 智能标注 (Offsets 微调)
# # -------------------------------------------------------
# # 0:KGW, 1:SWEET, 2:EWD, 3:DiPmark, 4:SynthID, 5:SIR, 6:SAW
# offsets = [
#     (0, 15),    # KGW (上)
#     (0, -18),   # SWEET (下，避免与 KGW 重叠)
#     (0, 15),    # EWD (上)
#     (-20, -15), # DiPmark (左下)
#     (20, 10),   # SynthID (右上，避免与 DiPmark 重叠)
#     (0, 15),    # SIR
#     (-50, 10)   # SAW (左侧，醒目)
# ]

# for i, txt in enumerate(method_names):
#     is_saw = (i == len(method_names) - 1)
    
#     # 字体样式
#     fw = 'bold' if is_saw else 'normal'
#     fs = 24 if is_saw else 18
#     fc = '#C0392B' if is_saw else '#2C3E50'
    
#     # 对基线 Frontier 上的点 (DiPmark, SWEET, EWD 等) 使用深色强调
#     # 注意：pareto_indices 是基于基线数据的索引
#     if i in pareto_indices and not is_saw:
#         fc = '#555555' 

#     ax.annotate(txt, (avg_f1[i], rc_ppl[i]), xytext=offsets[i], textcoords='offset points',
#                 ha='center', va='center', fontsize=fs, weight=fw, color=fc,
#                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))

# # -------------------------------------------------------
# # E. 标注 "Pareto Frontier"
# # -------------------------------------------------------
# # 我们把文字放在基线前沿的中间位置，暗示这是 Baselines 的极限
# # 找到基线前沿中最左(DiPmark)和最右(EWD)的点
# idx_dip = 3 # DiPmark
# idx_ewd = 2 # EWD

# # 计算位置
# mid_x = (avg_f1[idx_dip] + avg_f1[idx_ewd]) / 2
# mid_y = (rc_ppl[idx_dip] + rc_ppl[idx_ewd]) / 2

# ax.text(mid_x + 0.02, mid_y + 2, "Baselines Frontier",
#         fontsize=18, color='#7F8C8D', weight='bold',
#         ha='center', va='bottom',
#         rotation=-15,
#         zorder=5)

# # -------------------------------------------------------
# # F. 坐标轴与最终美化 (适配新数据范围)
# # -------------------------------------------------------
# ax.set_xlabel(r'Robustness: Average F1 Score ($\rightarrow$ Better)', fontsize=24, labelpad=12)
# ax.set_ylabel(r'Text Quality: RC-PPL ($\rightarrow$ Better)', fontsize=24, labelpad=12)

# # 反转 Y 轴 (PPL 越小越好)
# ax.invert_yaxis()

# # 设置范围 (基于新数据微调)
# # F1: 0.68 -> 0.93. 设置为 0.65 -> 0.96
# ax.set_xlim(0.65, 0.96)
# # PPL: 14.3 -> 28.7. 设置为 35 -> 10
# ax.set_ylim(35, 10)

# ax.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)

# for spine in ax.spines.values():
#     spine.set_linewidth(2.0)
#     spine.set_color('black')

# # ==========================================
# # 7. 保存与显示
# # ==========================================
# plt.tight_layout()
# plt.savefig("./SAW_Pareto_Frontier.pdf", dpi=1000, format="pdf", bbox_inches='tight')
# plt.show()