import matplotlib.pyplot as plt
import numpy as np

# 设置学术绘图风格
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# 扰动比例 p
p = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

# 准备数据 (原始 H@1)
# FB-DB15K
fbdb_base = np.array([0.4268, 0.4226, 0.4208, 0.4165, 0.4152, 0.4107, 0.4101, 0.4088, 0.4058])
fbdb_ours = np.array([0.5060, 0.5026, 0.5013, 0.4983, 0.4969, 0.4939, 0.4917, 0.4906, 0.4877])

# FB-YG15K
fbyg_base = np.array([0.3065, 0.3022, 0.2994, 0.2941, 0.2906, 0.2885, 0.2831, 0.2785, 0.2748])
fbyg_ours = np.array([0.4073, 0.4047, 0.4016, 0.3983, 0.3937, 0.3917, 0.3882, 0.3831, 0.3781])

# DBP15K (zh-en)
dbp_base = np.array([0.8175, 0.8063, 0.7975, 0.7837, 0.7717, 0.7578, 0.7462, 0.7328, 0.7187])
dbp_ours = np.array([0.8452, 0.8385, 0.8349, 0.8235, 0.8163, 0.8079, 0.7995, 0.7920, 0.7831])

# 统一转换为相对于 p=0 的衰减 (Performance Drop)
datasets = [
    ("FB-DB15K", fbdb_base - fbdb_base[0], fbdb_ours - fbdb_ours[0]),
    ("FB-YG15K", fbyg_base - fbyg_base[0], fbyg_ours - fbyg_ours[0]),
    ("DBP15K (zh-en)", dbp_base - dbp_base[0], dbp_ours - dbp_ours[0])
]

# 创建 1x3 的画布
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

color_ours = '#8B0000' # 深红色
color_base = '#4A5568' # 灰蓝色

for idx, (title, drop_base, drop_ours) in enumerate(datasets):
    ax = axes[idx]
    
    # 绘制折线
    ax.plot(p, drop_ours, label='Ours Drop', color=color_ours, marker='^', linewidth=2.5, markersize=8)
    ax.plot(p, drop_base, label='Baseline Drop', color=color_base, marker='o', linewidth=2, linestyle='--', markersize=7)
    
    # 核心设计：面积填充高亮鲁棒性增益
    ax.fill_between(p, drop_base, drop_ours, color=color_ours, alpha=0.15, label='Robustness Gain')
    
    ax.set_title(title, pad=15, fontweight='bold')
    ax.set_xlabel('Visual Perturbation Ratio ($p$)')
    
    if idx == 0:
        ax.set_ylabel('Performance Drop ($\Delta$ Hits@1)')
        
    ax.set_xticks(p)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 设置 Y 轴略微超出最大掉点值，让图表呼吸感更好
    y_min = min(np.min(drop_base), np.min(drop_ours)) * 1.1
    ax.set_ylim(y_min, 0.005)

# 提取全局图例并放置在底部居中
lines, labels = axes[0].get_legend_handles_labels()
fig.legend(lines, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=13)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2) # 给底部图例留出空间

# 保存高清格式供论文使用
plt.savefig('fig_perturb_drop_all.pdf', bbox_inches='tight')
plt.savefig('fig_perturb_drop_all.png', dpi=300, bbox_inches='tight')
print("图表已保存为 fig_perturb_drop_all.pdf 和 fig_perturb_drop_all.png")
