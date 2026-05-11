#!/usr/bin/env python3
"""
低资源实验结果提取脚本。

输入：low_resource_*_test.log（由 run_low_resource_test.sh 生成）
输出：
  - markdown 表：每个 dataset 一张
  - PNG 图：两个 subplot（FBDB / FBYG），每个图 H@1 + MRR 双 y 轴
  - JSON 汇总

用法：
    python parse_low_resource.py
    # 或指定日志目录
    python parse_low_resource.py --log-dir /path/to/logs/
    # 或与 baseline 对比（如果你有 baseline 的低资源数据）
"""
import os
import re
import sys
import json
import argparse
import glob
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 文件名格式：low_resource_FBDB15K_0.05_test.log
LOG_NAME_RE = re.compile(r"low_resource_(?P<dataset>FBDB15K|FBYG15K)_(?P<rate>[\d.]+)_test\.log")

# 结果行（与 parse_param_sweep_v2 同款）
METRIC_RE = re.compile(
    r"l2r:\s*acc of top.*?\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\].*?mrr\s*=\s*([\d.]+)",
    re.IGNORECASE
)


def parse_one_log(logfile: Path):
    """从单个推理 log 中读取最后一次的 H@1/H@10/H@50/MRR"""
    try:
        text = logfile.read_text(errors="ignore")
    except Exception:
        return None
    
    matches = METRIC_RE.findall(text)
    if not matches:
        return None
    
    # 取最后一次（应该只有一次，因为是 only_test 模式）
    h1, h10, h50, mrr = matches[-1]
    return float(h1), float(h10), float(h50), float(mrr)


def collect_all(log_dir: Path):
    """扫一遍目录，按 (dataset, rate) 归集结果"""
    data = defaultdict(dict)  # data[dataset][rate] = (h1, h10, h50, mrr)
    
    log_files = list(log_dir.glob("low_resource_*_test.log"))
    if not log_files:
        print(f"❌ {log_dir} 下没找到 low_resource_*_test.log")
        sys.exit(1)
    
    for lf in log_files:
        m = LOG_NAME_RE.match(lf.name)
        if not m:
            continue
        ds = m.group("dataset")
        rate = float(m.group("rate"))
        
        result = parse_one_log(lf)
        if result is None:
            print(f"⚠️  解析失败: {lf}")
            continue
        
        data[ds][rate] = result
        print(f"✅ {ds} rate={rate:.2f}: H@1={result[0]:.4f}, H@10={result[1]:.4f}, MRR={result[3]:.4f}")
    
    return data


def print_markdown(data, outdir: Path):
    md = []
    md.append("# 低资源场景下的鲁棒性实验\n")
    md.append("在 FB15K-DB15K 与 FB15K-YAGO15K 上测试方法在极低 seed 比例（5%-30%）下的表现。\n")
    
    for ds in sorted(data.keys()):
        md.append(f"\n## {ds}\n")
        md.append("| Training Rate | H@1 | H@10 | H@50 | MRR |")
        md.append("|---|---|---|---|---|")
        for rate in sorted(data[ds].keys()):
            h1, h10, h50, mrr = data[ds][rate]
            md.append(f"| {int(rate*100)}% | {h1:.4f} | {h10:.4f} | {h50:.4f} | {mrr:.4f} |")
    
    md_text = "\n".join(md)
    print(md_text)
    
    out_md = outdir / "low_resource_results.md"
    out_md.write_text(md_text, encoding="utf-8")
    print(f"\n💾 Markdown 已保存: {out_md}")


def plot_low_resource(data, outdir: Path):
    """2 子图：FBDB / FBYG，每个图同时画 H@1 和 MRR"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    titles = {
        "FBDB15K": "FB15K-DB15K",
        "FBYG15K": "FB15K-YAGO15K",
    }
    
    for idx, ds in enumerate(["FBDB15K", "FBYG15K"]):
        ax = axes[idx]
        if ds not in data:
            ax.axis('off')
            continue
        
        rates = sorted(data[ds].keys())
        h1s = [data[ds][r][0] for r in rates]
        h10s = [data[ds][r][1] for r in rates]
        mrrs = [data[ds][r][3] for r in rates]
        
        rates_pct = [int(r * 100) for r in rates]
        
        ax.plot(rates_pct, h1s, marker='o', color="#C0392B", linewidth=2,
                markersize=8, markeredgecolor='white', markeredgewidth=1,
                label='H@1')
        ax.plot(rates_pct, h10s, marker='s', color="#E67E22", linewidth=2,
                markersize=7, markeredgecolor='white', markeredgewidth=1,
                label='H@10', linestyle='--')
        ax.plot(rates_pct, mrrs, marker='^', color="#2C5F8D", linewidth=2,
                markersize=8, markeredgecolor='white', markeredgewidth=1,
                label='MRR')
        
        # 在每个点上标数值
        for r, h1 in zip(rates_pct, h1s):
            ax.annotate(f"{h1:.3f}", xy=(r, h1), xytext=(0, 8),
                       textcoords='offset points', fontsize=8,
                       ha='center', color="#C0392B")
        
        ax.set_title(titles[ds], fontsize=13, pad=10)
        ax.set_xlabel("Training Rate (%)", fontsize=12)
        ax.set_ylabel("Performance", fontsize=12)
        ax.set_xticks(rates_pct)
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    out_png = outdir / "low_resource_curves.png"
    out_pdf = outdir / "low_resource_curves.pdf"
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"💾 PNG 已保存: {out_png}")
    print(f"💾 PDF 已保存: {out_pdf}  (顶会投稿用此版)")


def save_json(data, outdir: Path):
    serializable = {}
    for ds, by_rate in data.items():
        serializable[ds] = {}
        for rate, (h1, h10, h50, mrr) in sorted(by_rate.items()):
            key = f"{rate:g}"
            serializable[ds][key] = {
                "H@1": h1, "H@10": h10, "H@50": h50, "MRR": mrr
            }
    out_json = outdir / "low_resource_results.json"
    out_json.write_text(json.dumps(serializable, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"💾 JSON 已保存: {out_json}")


def print_robustness_stats(data):
    """统计每个 dataset 的鲁棒性指标"""
    print("\n" + "=" * 70)
    print("📊 低资源鲁棒性分析（用于论文叙事）")
    print("=" * 70)
    
    for ds in sorted(data.keys()):
        rates = sorted(data[ds].keys())
        if not rates:
            continue
        h1s = [data[ds][r][0] for r in rates]
        
        print(f"\n### {ds}")
        print(f"  Rate range: {int(min(rates)*100)}% - {int(max(rates)*100)}%")
        print(f"  H@1 range:  {min(h1s):.4f} - {max(h1s):.4f}")
        print(f"  H@1 drop from 30% to 5%: {(h1s[rates.index(0.3)] - h1s[rates.index(0.05)]):.4f}"
              if 0.3 in rates and 0.05 in rates else "  (rate 0.3 or 0.05 missing)")
        
        # 计算"低资源相对保留率"：5% 时 H@1 占 30% 时的百分比
        if 0.3 in rates and 0.05 in rates:
            h1_30 = data[ds][0.3][0]
            h1_5 = data[ds][0.05][0]
            retention = h1_5 / h1_30 * 100
            print(f"  低资源保留率（5% vs 30%）: {retention:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=str, default=".",
                    help="包含 low_resource_*_test.log 的目录（默认当前目录）")
    ap.add_argument("--outdir", type=str, default="results_low_resource")
    args = ap.parse_args()
    
    log_dir = Path(args.log_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"📖 扫描日志目录: {log_dir.resolve()}")
    data = collect_all(log_dir)
    
    if not data:
        print("❌ 没找到有效数据")
        sys.exit(1)
    
    print_markdown(data, outdir)
    plot_low_resource(data, outdir)
    save_json(data, outdir)
    print_robustness_stats(data)


if __name__ == "__main__":
    main()