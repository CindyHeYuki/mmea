#!/usr/bin/env python3
"""
从 run_seed_fb.sh / run_seed_dbp.sh 的 log 目录里:
  1) 抽取每个 (config, seed) 的最优 H@1/H@10/MRR (从 sweep log 里读最优组合)
  2) 计算 mean±std (5 seed 聚合)
  3) 对每个配置做 one-sample t-test,与论文助手提供的最强 baseline 比较

用法:
    python collect_seed_stats.py <log_dir> [<log_dir2> ...]

可以一次传入 FB 和 DBP 两个 log 目录,合并出表。

baseline 数字 (对应论文助手 handover_to_code_claude.md):
    FBDB15K 20%: MIMEA = 50.6
    FBYG15K 20%: MIMEA = 41.7
    DBP15K ZH-EN: PMF = 83.5
其他配置的 baseline 我没有,在表里留空,你后面手动补。
"""
import os
import re
import sys
import math
from pathlib import Path
from collections import defaultdict

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠ scipy 未安装,t-test 跳过。pip install scipy")


# ---- log 文件名格式: <DS_LABEL>_seed<N>__sweep.log ----
LOG_PATTERN = re.compile(r"^(?P<label>[^_]+(?:_[^_]+)*)_seed(?P<seed>\d+)__sweep\.log$")

# ---- sweep log 里"最优组合"那一行 ----
# 例: 🎯 最优组合: causal_α=0.225, csc_α=0.225, neighbor_α=0.5
#     Hits@1=0.5772, Hits@10=0.7969, MRR=0.6522
BEST_HITS_PATTERN = re.compile(
    r"Hits@1\s*=\s*([\d.]+).*?Hits@10\s*=\s*([\d.]+).*?MRR\s*=\s*([\d.]+)",
    re.DOTALL
)
# csls_iter sweep 里"全局最优"那一行
GLOBAL_BEST_PATTERN = re.compile(
    r"全局最优.*?Hits@1\s*=\s*([\d.]+)",
    re.DOTALL
)
# fallback: 普通的 l2r: acc of top
FALLBACK_PATTERN = re.compile(
    r"l2r:\s*acc of top.*?\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\].*?mrr\s*=\s*([\d.]+)"
)


# ---- 已知 baseline 数字(从论文助手文档) ----
BASELINES = {
    "fbdb15k_20": ("MIMEA", 0.506),
    "fbyg15k_20": ("MIMEA", 0.417),
    "dbp_zh_w_surf":  ("PMF", 0.835),
    # 其他没有 baseline,保留 None
}


def extract_best_metrics(logfile: Path):
    """
    从 sweep log 中抽出最优 H@1/H@10/MRR.
    优先级:
      1. csls_iter_sweep 的"全局最优"行(只有 H@1)
      2. alpha_sweep 的"最优组合"+"Hits@1=...Hits@10=...MRR=..."行
      3. 训练末次 / sweep 末次的 l2r: acc of top 行
    """
    try:
        text = logfile.read_text(errors="ignore")
    except Exception:
        return None

    # 优先级 1: csls_iter_sweep 全局最优
    # 这种 log 里同时也会有详细的 alpha_sweep 排行榜,前 1 名就是全局最优
    matches = re.findall(
        r"⭐\s+1\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        text
    )
    if matches:
        h1, h10, mrr = matches[-1]
        return float(h1), float(h10), float(mrr)

    # 优先级 2: alpha_sweep 最优组合
    # log 形如:
    #   🎯 最优组合: causal_α=0.225, csc_α=0.225, neighbor_α=0.5
    #     Hits@1=0.5772, Hits@10=0.7969, MRR=0.6522
    m = re.search(
        r"最优组合.*?Hits@1\s*=\s*([\d.]+).*?Hits@10\s*=\s*([\d.]+).*?MRR\s*=\s*([\d.]+)",
        text, re.DOTALL
    )
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))

    # 优先级 3: 末次 l2r
    matches = FALLBACK_PATTERN.findall(text)
    if matches:
        h1, h10, _, mrr = matches[-1]
        return float(h1), float(h10), float(mrr)

    return None


def collect(log_dirs):
    """data[label][seed] = (h1, h10, mrr)"""
    data = defaultdict(dict)
    for log_dir in log_dirs:
        log_dir = Path(log_dir)
        if not log_dir.is_dir():
            print(f"⚠ {log_dir} 不存在,跳过")
            continue
        for f in log_dir.iterdir():
            m = LOG_PATTERN.match(f.name)
            if not m:
                continue
            label = m.group("label")
            seed = int(m.group("seed"))
            metrics = extract_best_metrics(f)
            if metrics is None:
                print(f"  ⚠ 跳过 {f.name}(没找到 metric)")
                continue
            data[label][seed] = metrics
    return data


def fmt_pct(v):
    return f"{v*100:.2f}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    data = collect(sys.argv[1:])

    if not data:
        print("❌ 没找到任何 sweep log")
        sys.exit(1)

    print()
    print("=" * 100)
    print("📊 多种子统计 (Multi-Seed Statistics, 单位: %)")
    print("=" * 100)

    # 表头
    print()
    print(f"{'Config':<22} | {'N':>3} | {'H@1 mean±std':>16} | {'H@10 mean±std':>16} | {'MRR mean±std':>16} | {'Baseline':<14} | {'p-value':>10}")
    print("-" * 120)

    for label in sorted(data):
        seeds_data = data[label]
        if not seeds_data:
            continue

        # 排序 seed 输出
        sorted_seeds = sorted(seeds_data.keys())
        h1_list = [seeds_data[s][0] for s in sorted_seeds]
        h10_list = [seeds_data[s][1] for s in sorted_seeds]
        mrr_list = [seeds_data[s][2] for s in sorted_seeds]

        n = len(h1_list)
        h1_mean = sum(h1_list) / n
        h10_mean = sum(h10_list) / n
        mrr_mean = sum(mrr_list) / n

        if n > 1:
            h1_std = math.sqrt(sum((x - h1_mean) ** 2 for x in h1_list) / (n - 1))
            h10_std = math.sqrt(sum((x - h10_mean) ** 2 for x in h10_list) / (n - 1))
            mrr_std = math.sqrt(sum((x - mrr_mean) ** 2 for x in mrr_list) / (n - 1))
        else:
            h1_std = h10_std = mrr_std = 0.0

        # baseline 比较
        baseline_str = ""
        pvalue_str = "—"
        if label in BASELINES:
            bl_name, bl_val = BASELINES[label]
            baseline_str = f"{bl_name}={fmt_pct(bl_val)}"
            if HAS_SCIPY and n >= 2:
                # one-sample t-test, alternative='greater' (检验是否显著大于 baseline)
                tstat, pval = stats.ttest_1samp(h1_list, bl_val, alternative='greater')
                pvalue_str = f"{pval:.4f}"

        print(f"{label:<22} | {n:>3} | "
              f"{fmt_pct(h1_mean)}±{fmt_pct(h1_std):<6} | "
              f"{fmt_pct(h10_mean)}±{fmt_pct(h10_std):<6} | "
              f"{fmt_pct(mrr_mean)}±{fmt_pct(mrr_std):<6} | "
              f"{baseline_str:<14} | {pvalue_str:>10}")

    # 详细 per-seed 表
    print()
    print("=" * 100)
    print("📋 详细每 seed 数字")
    print("=" * 100)
    for label in sorted(data):
        print(f"\n### {label}")
        sorted_seeds = sorted(data[label].keys())
        print(f"\n| seed | H@1   | H@10  | MRR   |")
        print(  "|------|-------|-------|-------|")
        for s in sorted_seeds:
            h1, h10, mrr = data[label][s]
            print(f"| {s:<4} | {fmt_pct(h1)} | {fmt_pct(h10)} | {fmt_pct(mrr)} |")

    print()
    print("-" * 100)
    print("💡 说明:")
    print("  - mean±std 中的数字单位是百分比(乘了100,小数点后保留2位,与论文表格一致)")
    print("  - p-value 来自 one-sample t-test (alternative='greater'),")
    print("    检验我们的 5-seed H@1 是否显著大于 baseline。p<0.05 即为显著。")
    print("  - 部分 baseline 数字尚未提供,需要手动补充进 BASELINES 字典。")


if __name__ == "__main__":
    main()