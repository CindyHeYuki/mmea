"""
Experiment 3 visualization - 4 alternative designs.
Each saved as a separate PNG for side-by-side comparison.

Data:
  thresholds = [5, 10, 20, 30]
  baseline / ours hub-error rate per dataset
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif']
mpl.rcParams['axes.linewidth'] = 1.0

# --- palette ---
C_UNIFORM = '#B0B0B0'    # 灰  -- 均匀参考
C_BASE    = '#4A5A6A'    # 蓝灰 -- baseline
C_OURS    = '#8B1A1A'    # 深红 -- ours
C_FILL    = '#8B1A1A'

THR = np.array([5, 10, 20, 30])

DATA = {
    'FB-DB15K':       {'base': [15.49, 26.59, 43.60, 54.07],
                       'ours': [12.37, 21.28, 37.28, 46.49]},
    'FB-YG15K':       {'base': [14.09, 25.10, 43.85, 54.05],
                       'ours': [11.19, 21.47, 39.60, 49.23]},
    'DBP15K (zh-en)': {'base': [11.75, 19.79, 34.88, 50.97],
                       'ours': [ 8.45, 15.97, 28.85, 45.56]},
}

OUT_DIR = '.'


# ============================================================
# 方案 1: 三柱并立 (uniform / baseline / ours), 4 个阈值并排
#         每个数据集一个 subplot, 横向三联
# ============================================================
def plot_option1():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

    x = np.arange(len(THR))
    width = 0.27

    for ax, (name, d) in zip(axes, DATA.items()):
        uni = THR.astype(float)
        base = np.array(d['base'])
        ours = np.array(d['ours'])

        b1 = ax.bar(x - width, uni, width, color=C_UNIFORM,
                    edgecolor='white', linewidth=0.8, label='Uniform (random)')
        b2 = ax.bar(x,         base, width, color=C_BASE,
                    edgecolor='white', linewidth=0.8, label='Baseline')
        b3 = ax.bar(x + width, ours, width, color=C_OURS,
                    edgecolor='white', linewidth=0.8, label='Ours')

        # 标数值
        for bars in [b1, b2, b3]:
            for rect in bars:
                h = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2, h + 0.8,
                        f'{h:.1f}', ha='center', va='bottom',
                        fontsize=8.5, color='#333333')

        ax.set_xticks(x)
        ax.set_xticklabels([f'{t}%' for t in THR], fontsize=11)
        ax.set_xlabel('Hub Threshold Percentile', fontsize=12)
        ax.set_title(f'{name}', fontsize=13, fontweight='bold', pad=10)
        ax.set_ylim(0, 65)
        ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color('#333333')

    axes[0].set_ylabel('Hub Error Rate (%)', fontsize=12)
    axes[0].legend(loc='upper left', fontsize=10, frameon=True,
                   framealpha=0.95, edgecolor='#888888')
    fig.suptitle('Hub Error Rate: Uniform vs Baseline vs Ours',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = f'{OUT_DIR}/fig_hub_option1_triple_bars.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# 方案 2: 倍率图 (r / τ), 均匀参考 = 1.0
# ============================================================
def plot_option2():
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5), dpi=150)

    n_thr = len(THR)
    n_ds = len(DATA)
    width = 0.18
    group_w = n_ds * 2 * width + 0.08

    # 横轴: 4 个阈值, 每个阈值下 3 数据集 × 2 方法 = 6 柱
    x_base = np.arange(n_thr) * group_w * 1.4

    colors_base = ['#5A6A7A', '#3A4A5A', '#1A2A3A']  # 渐变蓝
    colors_ours = ['#A53A3A', '#8B1A1A', '#6B0A0A']  # 渐变红

    for i, (name, d) in enumerate(DATA.items()):
        base_ratio = np.array(d['base']) / THR
        ours_ratio = np.array(d['ours']) / THR
        offset_b = (i - n_ds/2 + 0.5) * 2 * width - width/2
        offset_o = (i - n_ds/2 + 0.5) * 2 * width + width/2

        ax.bar(x_base + offset_b, base_ratio, width,
               color=colors_base[i], edgecolor='white', linewidth=0.6,
               label=f'{name} (Baseline)')
        ax.bar(x_base + offset_o, ours_ratio, width,
               color=colors_ours[i], edgecolor='white', linewidth=0.6,
               label=f'{name} (Ours)')

    # 均匀参考线
    ax.axhline(1.0, color='#555555', linestyle='--', linewidth=1.5,
               label='Uniform (= 1.0)', zorder=10)
    ax.text(x_base[-1] + 0.6, 1.05, 'Uniform reference',
            fontsize=10, color='#555555', style='italic')

    ax.set_xticks(x_base)
    ax.set_xticklabels([f'τ = {t}%' for t in THR], fontsize=11)
    ax.set_xlabel('Hub Threshold', fontsize=12)
    ax.set_ylabel(r'Hubness Ratio  $r_\tau / \tau$', fontsize=12)
    ax.set_title('Hubness Bias Intensity (Higher = stronger bias)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_ylim(0, 3.2)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#333333')

    ax.legend(loc='upper right', fontsize=9, frameon=True,
              framealpha=0.95, edgecolor='#888888', ncol=2)
    plt.tight_layout()
    out = f'{OUT_DIR}/fig_hub_option2_ratio.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# 方案 3: 双面板
#   上: hubness 倍率 (偏倚强度) - 每个数据集一组柱
#   下: 减幅 Δabs (方法效果)   - 每个数据集一组柱
# ============================================================
def plot_option3():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), dpi=150)

    n_thr = len(THR)
    n_ds = len(DATA)
    width = 0.22

    x_base = np.arange(n_thr)
    colors_per_ds = ['#C77878', '#8B1A1A', '#4A0A0A']
    colors_per_ds_blue = ['#7A8A9A', '#4A5A6A', '#1A2A3A']

    # ---- 上排: hubness 倍率 ----
    for i, (name, d) in enumerate(DATA.items()):
        base_ratio = np.array(d['base']) / THR
        offset = (i - n_ds/2 + 0.5) * width
        ax1.bar(x_base + offset, base_ratio, width,
                color=colors_per_ds_blue[i], edgecolor='white', linewidth=0.8,
                label=name)
        for j, v in enumerate(base_ratio):
            ax1.text(x_base[j] + offset, v + 0.05, f'{v:.2f}×',
                     ha='center', va='bottom', fontsize=8.5, color='#333333')

    ax1.axhline(1.0, color='#555555', linestyle='--', linewidth=1.5)
    ax1.text(n_thr - 0.4, 1.08, 'Uniform = 1.0×',
             fontsize=10, color='#555555', style='italic', ha='right')
    ax1.set_xticks(x_base)
    ax1.set_xticklabels([f'τ = {t}%' for t in THR], fontsize=11)
    ax1.set_ylabel(r'Baseline Hubness Ratio  $r^{\rm base}_\tau / \tau$',
                   fontsize=12)
    ax1.set_title('(a) Hubness Bias Strength of Baseline (× over uniform)',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylim(0, 3.5)
    ax1.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.95)
    for spine in ax1.spines.values():
        spine.set_color('#333333')

    # ---- 下排: 减幅 ----
    for i, (name, d) in enumerate(DATA.items()):
        delta = np.array(d['base']) - np.array(d['ours'])
        offset = (i - n_ds/2 + 0.5) * width
        ax2.bar(x_base + offset, delta, width,
                color=colors_per_ds[i], edgecolor='white', linewidth=0.8,
                label=name)
        for j, v in enumerate(delta):
            ax2.text(x_base[j] + offset, v + 0.15, f'+{v:.2f}',
                     ha='center', va='bottom', fontsize=8.5, color='#333333')

    ax2.set_xticks(x_base)
    ax2.set_xticklabels([f'τ = {t}%' for t in THR], fontsize=11)
    ax2.set_xlabel('Hub Threshold', fontsize=12)
    ax2.set_ylabel(r'Reduction  $r^{\rm base}_\tau - r^{\rm ours}_\tau$ (pp)',
                   fontsize=12)
    ax2.set_title('(b) Hub Error Rate Reduction by Ours (percentage points)',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylim(0, 9.5)
    ax2.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.95)
    for spine in ax2.spines.values():
        spine.set_color('#333333')

    plt.tight_layout()
    out = f'{OUT_DIR}/fig_hub_option3_dual_panel.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# 方案 4: 单阈值 (10%), 三数据集横向, 三柱并立 (uniform/base/ours)
#         超简洁正文主图
# ============================================================
def plot_option4():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5), dpi=150)

    n_ds = len(DATA)
    width = 0.25
    x = np.arange(n_ds)

    uni_vals  = [10, 10, 10]
    base_vals = [DATA[k]['base'][1] for k in DATA]   # idx 1 = τ=10%
    ours_vals = [DATA[k]['ours'][1] for k in DATA]

    b1 = ax.bar(x - width, uni_vals,  width, color=C_UNIFORM,
                edgecolor='white', linewidth=1.0, label='Uniform (random)')
    b2 = ax.bar(x,         base_vals, width, color=C_BASE,
                edgecolor='white', linewidth=1.0, label='Baseline (MEAformer)')
    b3 = ax.bar(x + width, ours_vals, width, color=C_OURS,
                edgecolor='white', linewidth=1.0, label='Ours')

    # 数值标注
    for bars in [b1, b2, b3]:
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, h + 0.4,
                    f'{h:.1f}%', ha='center', va='bottom',
                    fontsize=11, color='#222222', fontweight='bold')

    # 在 baseline 柱上方标注倍率 (位置抬高, 远离 baseline 数值和灰柱区域)
    for i, (b, u) in enumerate(zip(base_vals, uni_vals)):
        ratio = b / u
        ax.text(x[i], b + 4.0, f'{ratio:.1f}× over uniform',
                ha='center', fontsize=9.5, color=C_BASE, style='italic',
                fontweight='bold')

    # 减幅箭头: 三个箭头**等长**, 且**距各自 ours 柱顶距离相同**, 跟随红柱浮动
    ARROW_GAP    = 2     # 箭头底端距 ours 柱顶的间隙 (三个数据集统一)
    ARROW_LEN    = 3.0     # 箭头垂直长度 (三个数据集统一)
    for i, (bv, ov) in enumerate(zip(base_vals, ours_vals)):
        delta = bv - ov
        arrow_x = x[i] + width
        arrow_bottom = ov + ARROW_GAP
        arrow_top    = arrow_bottom + ARROW_LEN
        ax.annotate('', xy=(arrow_x, arrow_bottom),
                    xytext=(arrow_x, arrow_top),
                    arrowprops=dict(arrowstyle='->', color=C_OURS,
                                    lw=2.0, shrinkA=0, shrinkB=0))
        # 减幅标签紧贴箭头右侧, 垂直居中于箭头中点
        label_y = (arrow_top + arrow_bottom) / 2
        ax.text(arrow_x + 0.07, label_y,
                f'−{delta:.1f} pp', fontsize=10.5, color=C_OURS,
                fontweight='bold', va='center', ha='left',
                bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                          edgecolor='none', alpha=0.92))

    ax.set_xticks(x)
    ax.set_xticklabels(list(DATA.keys()), fontsize=12)
    ax.set_xlim(-0.55, x[-1] + 0.85)   # 右侧多留空间给减幅标签
    ax.set_ylabel('Hub Error Rate (%)', fontsize=12)
    ax.set_title('Top-10% Hub Error Rate: Baseline shows 2–2.7× over-attraction; Ours reduces it',
                 fontsize=12.5, fontweight='bold', pad=12)
    ax.set_ylim(0, 38)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#333333')
    ax.legend(loc='upper right', fontsize=10, frameon=True,
              framealpha=0.95, edgecolor='#888888')

    plt.tight_layout()
    out = f'{OUT_DIR}/fig_hub_option4_single_threshold.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")


# ============================================================
if __name__ == '__main__':
    plot_option1()
    plot_option2()
    plot_option3()
    plot_option4()
    print("\nAll 4 options generated.")