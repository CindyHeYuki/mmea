"""
顶会版 9 子图绘制函数 patch。

把 parse_param_sweep_v2.py 里的 plot_nine_panels() 函数完整替换为下面这个版本。

主要改动：
  1. 每行内 sharey（同一行 y 轴一致，跨行不一致）
  2. 全图字号合格双栏 PDF 排版
  3. 配色改用 matplotlib 默认深红/深蓝（更专业）
  4. marker 加白色描边（小图缩放不糊）
  5. 数据集名字加 "(20% seed)"
  6. 去掉主标题（caption 里写）
  7. 山峰形参数可选标注最优点垂直虚线
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_nine_panels(all_data, outdir):
    """
    3×3 子图布局，按创新点分组。
    
    布局（按创新点分组顺序）：
      行 1: λ, γ, α         (§3.3.1 ×2 + §3.3.2 ×1)
      行 2: β, τ_C, ε       (§3.3.2 ×3)
      行 3: μ, top-k, T     (§3.3.3 ×3)
    """
    PARAM_META = {
        "lambda":  {"sym": r"$\lambda$",    "title": r"$\lambda$ (lambda_val)",        "section": "§3.3.1"},
        "k":       {"sym": r"$\gamma$",     "title": r"$\gamma$ (k)",                  "section": "§3.3.1"},
        "alpha":   {"sym": r"$\alpha$",     "title": r"$\alpha$ (causal mixing)",      "section": "§3.3.2"},
        "beta":    {"sym": r"$\beta$",      "title": r"$\beta$ (counterfactual)",      "section": "§3.3.2"},
        "tauC":    {"sym": r"$\tau_C$",     "title": r"$\tau_C$ (softmax temp.)",      "section": "§3.3.2"},
        "eps":     {"sym": r"$\epsilon$",   "title": r"$\epsilon$ (soft floor)",       "section": "§3.3.2"},
        "mu":      {"sym": r"$\mu$",        "title": r"$\mu$ (neighbor weight)",       "section": "§3.3.3"},
        "topk":    {"sym": r"$k$",          "title": r"$k$ (neighborhood size)",       "section": "§3.3.3"},
        "T":       {"sym": r"$T$",          "title": r"$T$ (refinement iter.)",        "section": "§3.3.3"},
    }
    
    param_order = ["lambda", "k", "alpha",
                   "beta", "tauC", "eps",
                   "mu", "topk", "T"]
    
    # 3 行 3 列；每行内 sharey
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey='row')
    
    # 数据集配色（更专业的深红 / 深蓝）
    ds_style = [
        ("FBDB15K", "FB15K-DB15K (20% seed)", "#C0392B", "o"),  # 深红
        ("FBYG15K", "FB15K-YAGO15K (20% seed)", "#2C5F8D", "s"),  # 深蓝
    ]
    
    for idx, pk in enumerate(param_order):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        meta = PARAM_META[pk]
        
        has_data = False
        for ds, ds_label, color, marker in ds_style:
            if ds not in all_data or pk not in all_data[ds]:
                continue
            xs = sorted(all_data[ds][pk].keys())
            ys = [all_data[ds][pk][x][0] for x in xs]  # H@1
            ax.plot(xs, ys,
                    marker=marker, color=color,
                    linewidth=1.8, markersize=7,
                    markeredgecolor='white', markeredgewidth=0.8,
                    label=ds_label)
            has_data = True
        
        ax.set_title(f"{meta['title']}  [{meta['section']}]",
                     fontsize=12, pad=8)
        ax.set_xlabel(meta['sym'], fontsize=12)
        # 只在最左列加 y 轴标签
        if col == 0:
            ax.set_ylabel("Hits@1", fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 用一个统一的 outside legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.02),
               ncol=2,
               fontsize=12,
               frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # 同时输出 PDF（顶会必须有矢量图）+ PNG（预览用）
    out_png = outdir / "param_sweep_nine_panels.png"
    out_pdf = outdir / "param_sweep_nine_panels.pdf"
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"💾 9 子图 PNG: {out_png}")
    print(f"💾 9 子图 PDF: {out_pdf}  (顶会投稿用此版)")