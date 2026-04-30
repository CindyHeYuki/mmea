#!/usr/bin/env python3
"""
从 sensitivity 扫描的 log 目录里:
  1) 提取每个 (dataset, axis, value) 对应的 H@1 / H@10 / MRR
  2) 输出 4 张数据表 (markdown)
  3) 画 1×2 子图(左 α 曲线,右 β 曲线),保存为 figure4_alpha_beta_sensitivity.png

用法:
    python plot_sensitivity.py <log_dir>
"""
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 文件名格式: <LABEL>__<axis>_<value>.log
# axis ∈ {alpha, beta},value 是浮点数字符串 like "0.0", "0.125"
LOG_PATTERN = re.compile(r"^(?P<label>[^_]+_\d+)__(?P<axis>alpha|beta)_(?P<value>[\d.]+)\.log$")
# main.py 末尾输出格式: l2r: acc of top [1, 10, 50] = [0.5772 0.7969 0.8731], mr=...., mrr=0.6522
METRIC_PATTERN = re.compile(
    r"l2r:\s*acc of top.*?\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\].*?mrr\s*=\s*([\d.]+)"
)


def extract_metrics(logfile: Path):
    """返回 (h1, h10, mrr) 三元组,失败返回 None"""
    try:
        text = logfile.read_text(errors="ignore")
    except Exception:
        return None
    matches = METRIC_PATTERN.findall(text)
    if not matches:
        return None
    h1, h10, _, mrr = matches[-1]  # 取最后一次,即 sensitivity 这一轮的结果
    return float(h1), float(h10), float(mrr)


def main(log_dir: str):
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        print(f"❌ {log_dir} 不存在")
        sys.exit(1)

    # 收集所有点: data[label][axis] = [(value, h1, h10, mrr), ...]
    data: dict = {}
    for f in log_dir.iterdir():
        m = LOG_PATTERN.match(f.name)
        if not m:
            continue
        metrics = extract_metrics(f)
        if metrics is None:
            print(f"  ⚠ 跳过 {f.name}(没找到 metric)")
            continue
        label = m.group("label")
        axis = m.group("axis")
        value = float(m.group("value"))
        data.setdefault(label, {}).setdefault(axis, []).append((value,) + metrics)

    if not data:
        print(f"❌ {log_dir} 下没找到任何 sensitivity 日志")
        sys.exit(1)

    # 每条曲线按 value 排序
    for label in data:
        for axis in data[label]:
            data[label][axis].sort(key=lambda x: x[0])

    # ============ 1) 输出 markdown 数据表 ============
    print()
    print("=" * 70)
    print("📊 α/β Sensitivity Analysis Results")
    print("=" * 70)

    for label in sorted(data):
        for axis in ("alpha", "beta"):
            if axis not in data[label]:
                continue
            print(f"\n### {label}  —  扫 {axis}")
            print(f"\n| {axis:<8} | H@1     | H@10    | MRR     |")
            print(  "|----------|---------|---------|---------|")
            for v, h1, h10, mrr in data[label][axis]:
                print(f"| {v:<8.3f} | {h1:.4f}  | {h10:.4f}  | {mrr:.4f}  |")

    # ============ 2) 绘图 1×2 ============
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    # 颜色 & marker 区分 dataset
    colors = {"fbdb15k_20": "#E64B35", "fbyg15k_20": "#3C5488"}
    markers = {"fbdb15k_20": "o", "fbyg15k_20": "s"}
    label_pretty = {"fbdb15k_20": "FB15K-DB15K (20%)", "fbyg15k_20": "FB15K-YAGO15K (20%)"}

    for label in sorted(data):
        if "alpha" in data[label]:
            xs = [pt[0] for pt in data[label]["alpha"]]
            ys = [pt[1] for pt in data[label]["alpha"]]   # H@1
            ax_a.plot(xs, ys,
                      color=colors.get(label, "gray"),
                      marker=markers.get(label, "x"),
                      linewidth=1.8, markersize=7,
                      label=label_pretty.get(label, label))
        if "beta" in data[label]:
            xs = [pt[0] for pt in data[label]["beta"]]
            ys = [pt[1] for pt in data[label]["beta"]]
            ax_b.plot(xs, ys,
                      color=colors.get(label, "gray"),
                      marker=markers.get(label, "x"),
                      linewidth=1.8, markersize=7,
                      label=label_pretty.get(label, label))

    ax_a.set_xlabel(r"$\alpha$ (causal mixing coefficient)", fontsize=12)
    ax_a.set_ylabel("Hits@1", fontsize=12)
    ax_a.set_title(r"Sensitivity to $\alpha$", fontsize=13)
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(loc="best", fontsize=10)

    ax_b.set_xlabel(r"$\beta$ (counterfactual convex weight)", fontsize=12)
    ax_b.set_ylabel("Hits@1", fontsize=12)
    ax_b.set_title(r"Sensitivity to $\beta$", fontsize=13)
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(loc="best", fontsize=10)

    plt.tight_layout()
    out_path = log_dir / "figure4_alpha_beta_sensitivity.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\n📈 图已保存:")
    print(f"     {out_path}")
    print(f"     {out_path.with_suffix('.pdf')}")

    # ============ 3) 标注最优值 ============
    print()
    print("-" * 70)
    print("💡 各曲线的峰值点(供论文叙事参考):")
    print("-" * 70)
    for label in sorted(data):
        for axis in ("alpha", "beta"):
            if axis not in data[label]:
                continue
            best = max(data[label][axis], key=lambda x: x[1])
            print(f"  {label} 扫 {axis}: 峰值 {axis}={best[0]:.3f}, H@1={best[1]:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])