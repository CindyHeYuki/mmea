#!/usr/bin/env python3
"""
从参数扫描日志中提取 6 个参数的 H@1/H@10/MRR 结果。

支持两类日志格式：
  - 训练日志（[TRAIN] 标记）：用于 λ、γ 这种需要重训的参数
  - 推理日志（[TEST] 标记）：用于 τ_C、ε、μ、top-k 这种只推理的参数

支持多个日志文件输入（同一数据集可以拆多个 log）。

用法：
    # 单 log
    python parse_param_sweep.py \
        --fbdb run_param_sweep_test_only_fbdb_0511.log \
        --fbyg run_param_sweep_test_only_fbyg_0511.log
    
    # 多 log（每个 dataset 可以传多个日志，用空格分隔）
    python parse_param_sweep.py \
        --fbdb run_param_sweep_fbdb_0509.log run_param_sweep_test_only_fbdb_0511.log \
        --fbyg run_param_sweep_fbyg_0509.log run_param_sweep_test_only_fbyg_0511.log \
        [--outdir results_param_sweep]
"""
import os
import re
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 正则匹配：训练 / 推理两种 exp_id 来源
# ============================================================
# [TRAIN] 行示例：
#   >>> [TRAIN] FBDB15K rate=0.2 exp_id=FBDB15K_0.2_lambda_0.2
TRAIN_HEADER_RE = re.compile(
    r"\[TRAIN\]\s+(?P<dataset>\S+)\s+rate=(?P<rate>\S+)\s+exp_id=(?P<exp_id>\S+)"
)
# [TEST] 行示例：
#   >>> [TEST]  FBDB15K rate=0.2 ckpt=FBDB15K_0.2_default_ exp_id=FBDB15K_0.2_tauC_1.0
TEST_HEADER_RE = re.compile(
    r"\[TEST\]\s+(?P<dataset>\S+)\s+rate=(?P<rate>\S+)\s+ckpt=\S+\s+exp_id=(?P<exp_id>\S+)"
)
# 结果行：l2r: acc of top [1, 10, 50] = [0.4795 0.7538 0.8200], mr=..., mrr=0.5750
METRIC_RE = re.compile(
    r"l2r:\s*acc of top.*?\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\].*?mrr\s*=\s*([\d.]+)",
    re.IGNORECASE
)

PARAM_META = {
    "lambda":  {"sym": "λ",    "title": "λ (lambda_val)",       "section": "§3.3.1"},
    "k":       {"sym": "γ",    "title": "γ (k)",                "section": "§3.3.1"},
    "tauC":    {"sym": "τ_C",  "title": "τ_C (tau_C)",          "section": "§3.3.2"},
    "eps":     {"sym": "ε",    "title": "ε (epsilon_floor)",    "section": "§3.3.2"},
    "mu":      {"sym": "μ",    "title": "μ (neighbor_alpha)",   "section": "§3.3.3"},
    "topk":    {"sym": "top-k","title": "top-k (csls_k)",       "section": "§3.3.3"},
}

# exp_id 里的参数信息
EXP_ID_RE = re.compile(
    r"(?P<dataset>FBDB15K|FBYG15K)_(?P<rate>[\d.]+)_(?P<param>lambda|k|tauC|eps|mu|topk)_(?P<value>[-\d.]+)"
)


def parse_log(logfile: Path):
    """
    扫一次 log，返回:
        data[dataset][param_key][value] = (H@1, H@10, H@50, MRR)
    
    对训练日志：每次训练可能产生多次 metric，**保留最后一次**（最终 epoch）
    对推理日志：每个 [TEST] 只产生一次 metric
    """
    text = logfile.read_text(errors="ignore")
    lines = text.splitlines()
    
    data = defaultdict(lambda: defaultdict(dict))
    current_exp = None    # 当前正在跑的 exp_id
    current_mode = None   # "TRAIN" 或 "TEST"
    last_metric_per_exp = {}  # exp_id -> (h1, h10, h50, mrr)

    for ln in lines:
        # 1) 训练头
        m = TRAIN_HEADER_RE.search(ln)
        if m:
            current_exp = m.group("exp_id")
            current_mode = "TRAIN"
            continue
        
        # 2) 推理头
        m = TEST_HEADER_RE.search(ln)
        if m:
            current_exp = m.group("exp_id")
            current_mode = "TEST"
            continue

        # 3) metric 行：绑定到当前 exp_id
        m = METRIC_RE.search(ln)
        if m and current_exp:
            h1, h10, h50, mrr = map(float, m.groups())
            # 训练日志会有多次 metric（每 eval_epoch 一次），覆盖直到最后一次
            # 推理日志只会有一次，效果一样
            last_metric_per_exp[current_exp] = (h1, h10, h50, mrr)
            continue
        
        # 4) 看到新的训练/推理头或者文件末尾时，把上一段绑定到 data
        # （上面的 1）2）已经会覆盖 current_exp，所以这里不用单独处理）
    
    # 最后统一登记
    for exp_id, metrics in last_metric_per_exp.items():
        mm = EXP_ID_RE.match(exp_id)
        if not mm:
            continue
        dataset = mm.group("dataset")
        param_key = mm.group("param")
        try:
            value = float(mm.group("value"))
        except ValueError:
            value = mm.group("value")
        data[dataset][param_key][value] = metrics

    return data


def merge(*sources):
    """合并多个 parse_log 结果到一个 dict"""
    out = defaultdict(lambda: defaultdict(dict))
    for src in sources:
        for ds, by_param in src.items():
            for pk, by_val in by_param.items():
                for v, m in by_val.items():
                    out[ds][pk][v] = m
    return out


def print_markdown_tables(all_data, outdir: Path):
    md_lines = []
    md_lines.append("# 参数敏感性扫描结果汇总\n")
    md_lines.append("覆盖三个创新点共 6 个参数：\n")
    md_lines.append("- §3.3.1：λ (lambda_val), γ (k)")
    md_lines.append("- §3.3.2：τ_C (tau_C), ε (epsilon_floor)")
    md_lines.append("- §3.3.3：μ (neighbor_alpha), top-k (csls_k)\n")
    
    for ds in sorted(all_data.keys()):
        md_lines.append(f"\n## {ds}\n")
        for pk in ["lambda", "k", "tauC", "eps", "mu", "topk"]:
            if pk not in all_data[ds]:
                continue
            meta = PARAM_META[pk]
            md_lines.append(f"\n### {ds} — Sweep {meta['title']}  ({meta['section']})\n")
            md_lines.append(f"| {meta['sym']} | H@1 | H@10 | H@50 | MRR |")
            md_lines.append("|------|------|------|------|------|")
            for v in sorted(all_data[ds][pk].keys()):
                h1, h10, h50, mrr = all_data[ds][pk][v]
                v_str = f"{v:g}" if isinstance(v, float) else str(v)
                md_lines.append(
                    f"| {v_str} | {h1:.4f} | {h10:.4f} | {h50:.4f} | {mrr:.4f} |"
                )

    md_text = "\n".join(md_lines)
    print(md_text)

    out_md = outdir / "param_sweep_results.md"
    out_md.write_text(md_text, encoding="utf-8")
    print(f"\n💾 Markdown 表格已保存: {out_md}")


def plot_six_panels(all_data, outdir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    param_order = ["lambda", "k", "tauC", "eps", "mu", "topk"]
    
    for idx, pk in enumerate(param_order):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        meta = PARAM_META[pk]
        
        has_data = False
        for ds, color, marker in [
            ("FBDB15K", "#E64A4A", "o"),
            ("FBYG15K", "#3A78B0", "s"),
        ]:
            if ds not in all_data or pk not in all_data[ds]:
                continue
            xs = sorted(all_data[ds][pk].keys())
            ys = [all_data[ds][pk][x][0] for x in xs]  # H@1
            ax.plot(xs, ys, marker=marker, color=color, linewidth=2,
                    markersize=7, label=ds)
            has_data = True
        
        ax.set_title(f"{meta['title']}  [{meta['section']}]", fontsize=11)
        ax.set_xlabel(meta['sym'])
        ax.set_ylabel("H@1")
        ax.grid(alpha=0.3)
        if has_data:
            ax.legend(fontsize=9)
    
    plt.suptitle("Hyper-parameter Sensitivity Analysis (H@1)", fontsize=13, y=1.00)
    plt.tight_layout()
    
    out_png = outdir / "param_sweep_six_panels.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 6 子图 PNG 已保存: {out_png}")


def save_json(all_data, outdir: Path):
    serializable = {}
    for ds, by_param in all_data.items():
        serializable[ds] = {}
        for pk, by_val in by_param.items():
            serializable[ds][pk] = {}
            for v, (h1, h10, h50, mrr) in sorted(by_val.items()):
                key = f"{v:g}" if isinstance(v, float) else str(v)
                serializable[ds][pk][key] = {
                    "H@1": h1, "H@10": h10, "H@50": h50, "MRR": mrr
                }
    out_json = outdir / "param_sweep_results.json"
    out_json.write_text(json.dumps(serializable, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"💾 JSON 汇总已保存: {out_json}")


def print_summary_stats(all_data):
    print("\n" + "=" * 70)
    print("📈 每个参数的 H@1 平台分析（用于论文叙事）")
    print("=" * 70)
    
    for ds in sorted(all_data.keys()):
        print(f"\n### {ds}")
        print(f"{'参数':<12}{'H@1 max':<12}{'H@1 min':<12}{'极差':<12}{'最优值':<12}")
        print("-" * 60)
        for pk in ["lambda", "k", "tauC", "eps", "mu", "topk"]:
            if pk not in all_data[ds]:
                continue
            sym = PARAM_META[pk]["sym"]
            items = sorted(all_data[ds][pk].items())
            h1_list = [m[0] for v, m in items]
            v_list = [v for v, m in items]
            h_max = max(h1_list)
            h_min = min(h1_list)
            best_v = v_list[h1_list.index(h_max)]
            best_v_str = f"{best_v:g}" if isinstance(best_v, float) else str(best_v)
            print(f"{sym:<12}{h_max:<12.4f}{h_min:<12.4f}{h_max-h_min:<12.4f}{best_v_str:<12}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbdb", type=str, nargs='+', required=True,
                    help="FBDB 的 log 路径（可多个，空格分隔）")
    ap.add_argument("--fbyg", type=str, nargs='+', required=True,
                    help="FBYG 的 log 路径（可多个，空格分隔）")
    ap.add_argument("--outdir", type=str, default="results_param_sweep")
    args = ap.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    fbdb_data_list = []
    for f in args.fbdb:
        p = Path(f)
        if not p.is_file():
            print(f"❌ {p} 不存在", file=sys.stderr); sys.exit(1)
        print(f"📖 解析 FBDB log: {p}")
        fbdb_data_list.append(parse_log(p))
    
    fbyg_data_list = []
    for f in args.fbyg:
        p = Path(f)
        if not p.is_file():
            print(f"❌ {p} 不存在", file=sys.stderr); sys.exit(1)
        print(f"📖 解析 FBYG log: {p}")
        fbyg_data_list.append(parse_log(p))
    
    all_data = merge(*fbdb_data_list, *fbyg_data_list)
    
    print("\n📊 解析结果概览：")
    for ds in sorted(all_data.keys()):
        print(f"  {ds}:")
        for pk in ["lambda", "k", "tauC", "eps", "mu", "topk"]:
            n = len(all_data[ds].get(pk, {}))
            sym = PARAM_META[pk]["sym"]
            mark = "✅" if n > 0 else "❌"
            print(f"    {mark} {sym:<8} : {n} 个点")
    
    print_markdown_tables(all_data, outdir)
    plot_six_panels(all_data, outdir)
    save_json(all_data, outdir)
    print_summary_stats(all_data)


if __name__ == "__main__":
    main()