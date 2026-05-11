#!/usr/bin/env python3
"""
9 参数完整版解析脚本。

数据来源：
  1) run_param_sweep_*_0511.log：6 个新参数 (λ, γ, τ_C, ε, μ, top-k)
  2) JSON 文件：T (csls_iter) 的扫描结果
  3) 旧 α/β 数据：直接在脚本里硬编码（来自 0428 sensitivity 实验）

用法：
    python parse_param_sweep_v2.py \
        --fbdb-log run_param_sweep_fbdb_0509.log run_param_sweep_test_only_fbdb_0511.log \
        --fbyg-log run_param_sweep_fbyg_0509.log run_param_sweep_test_only_fbyg_0511.log \
        --fbdb-T-json /data0/hwx/mmea_copy/data/mmkg/sweep_results/csls_iter_sanity_FBDB15K_norm_0.2_surface0.json \
        --fbyg-T-json /data0/hwx/mmea_copy/data/mmkg/sweep_results/csls_iter_sanity_FBYG15K_norm_0.2_surface0.json \
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
# 正则
# ============================================================
TRAIN_HEADER_RE = re.compile(
    r"\[TRAIN\]\s+(?P<dataset>\S+)\s+rate=(?P<rate>\S+)\s+exp_id=(?P<exp_id>\S+)"
)
TEST_HEADER_RE = re.compile(
    r"\[TEST\]\s+(?P<dataset>\S+)\s+rate=(?P<rate>\S+)\s+ckpt=\S+\s+exp_id=(?P<exp_id>\S+)"
)
METRIC_RE = re.compile(
    r"l2r:\s*acc of top.*?\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\].*?mrr\s*=\s*([\d.]+)",
    re.IGNORECASE
)
EXP_ID_RE = re.compile(
    r"(?P<dataset>FBDB15K|FBYG15K)_(?P<rate>[\d.]+)_(?P<param>lambda|k|tauC|eps|mu|topk)_(?P<value>[-\d.]+)"
)


PARAM_META = {
    "lambda":  {"sym": "λ",     "title": "λ (lambda_val)",        "section": "§3.3.1"},
    "k":       {"sym": "γ",     "title": "γ (k)",                 "section": "§3.3.1"},
    "alpha":   {"sym": "α",     "title": "α (causal_lambda)",     "section": "§3.3.2"},
    "beta":    {"sym": "β",     "title": "β (csc_lambda_0)",      "section": "§3.3.2"},
    "tauC":    {"sym": "τ_C",   "title": "τ_C (tau_C)",           "section": "§3.3.2"},
    "eps":     {"sym": "ε",     "title": "ε (epsilon_floor)",     "section": "§3.3.2"},
    "mu":      {"sym": "μ",     "title": "μ (neighbor_alpha)",    "section": "§3.3.3"},
    "topk":    {"sym": "top-k", "title": "top-k (csls_k)",        "section": "§3.3.3"},
    "T":       {"sym": "T",     "title": "T (csls_iter)",         "section": "§3.3.3"},
}


# ============================================================
# α / β 历史数据（从 0428 sensitivity 实验）
# ============================================================
HISTORICAL_ALPHA_BETA = {
    "FBDB15K": {
        "alpha": {
            0.0: (0.4795, 0.7538, 0, 0.5750),
            0.05: (0.4884, 0.7594, 0, 0.5830),
            0.10: (0.4926, 0.7620, 0, 0.5870),
            0.125: (0.4951, 0.7631, 0, 0.5890),
            0.15: (0.4960, 0.7643, 0, 0.5900),
            0.20: (0.4974, 0.7651, 0, 0.5910),
            0.25: (0.4964, 0.7643, 0, 0.5900),
            0.30: (0.4944, 0.7649, 0, 0.5880),
            0.40: (0.4902, 0.7598, 0, 0.5830),
            0.50: (0.4826, 0.7550, 0, 0.5750),
        },
        "beta": {
            0.0: (0.4867, 0.7517, 0, 0.5780),
            0.05: (0.4921, 0.7561, 0, 0.5830),
            0.10: (0.4929, 0.7600, 0, 0.5850),
            0.125: (0.4937, 0.7613, 0, 0.5870),
            0.20: (0.4962, 0.7643, 0, 0.5900),
            0.25: (0.4963, 0.7652, 0, 0.5900),
            0.30: (0.4952, 0.7657, 0, 0.5900),
            0.40: (0.4896, 0.7642, 0, 0.5850),
            0.50: (0.4745, 0.7559, 0, 0.5720),
        },
    },
    "FBYG15K": {
        "alpha": {
            0.0: (0.3845, 0.6288, 0, 0.4680),
            0.05: (0.3942, 0.6372, 0, 0.4780),
            0.10: (0.3992, 0.6436, 0, 0.4840),
            0.125: (0.4017, 0.6468, 0, 0.4870),
            0.15: (0.4039, 0.6482, 0, 0.4890),
            0.20: (0.4047, 0.6500, 0, 0.4900),
            0.25: (0.4036, 0.6493, 0, 0.4890),
            0.30: (0.4021, 0.6475, 0, 0.4860),
            0.40: (0.3942, 0.6432, 0, 0.4790),
            0.50: (0.3818, 0.6320, 0, 0.4670),
        },
        "beta": {
            0.0: (0.4022, 0.6429, 0, 0.4860),
            0.05: (0.4038, 0.6460, 0, 0.4880),
            0.10: (0.4046, 0.6474, 0, 0.4890),
            0.125: (0.4045, 0.6471, 0, 0.4880),
            0.20: (0.3975, 0.6448, 0, 0.4840),
            0.25: (0.3929, 0.6414, 0, 0.4800),
            0.30: (0.3886, 0.6378, 0, 0.4760),
            0.40: (0.3741, 0.6253, 0, 0.4620),
            0.50: (0.3584, 0.6097, 0, 0.4460),
        },
    },
}


def parse_log(logfile: Path):
    text = logfile.read_text(errors="ignore")
    lines = text.splitlines()
    data = defaultdict(lambda: defaultdict(dict))
    current_exp = None
    last_metric_per_exp = {}

    for ln in lines:
        m = TRAIN_HEADER_RE.search(ln)
        if m:
            current_exp = m.group("exp_id")
            continue
        m = TEST_HEADER_RE.search(ln)
        if m:
            current_exp = m.group("exp_id")
            continue
        m = METRIC_RE.search(ln)
        if m and current_exp:
            h1, h10, h50, mrr = map(float, m.groups())
            last_metric_per_exp[current_exp] = (h1, h10, h50, mrr)
    
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


def parse_T_json(json_file: Path, dataset: str):
    """从 csls_iter_sanity JSON 中读取 T 扫描结果"""
    if not json_file.is_file():
        print(f"⚠ T JSON 不存在: {json_file}")
        return {}
    obj = json.loads(json_file.read_text())
    result = {}
    for entry in obj.get("results", []):
        T = int(entry["csls_iter"])
        # JSON 里只有 hits1/hits10/mrr，没有 hits50，用 0 占位
        result[T] = (entry["hits1"], entry["hits10"], 0.0, entry["mrr"])
    print(f"📖 解析 T JSON: {json_file} → {len(result)} 个点 ({dataset})")
    return result


def merge(*sources):
    out = defaultdict(lambda: defaultdict(dict))
    for src in sources:
        for ds, by_param in src.items():
            for pk, by_val in by_param.items():
                for v, m in by_val.items():
                    out[ds][pk][v] = m
    return out


def print_markdown_tables(all_data, outdir: Path):
    md_lines = []
    md_lines.append("# 9 参数敏感性扫描结果汇总\n")
    md_lines.append("覆盖三个创新点共 9 个参数：\n")
    md_lines.append("- §3.3.1：λ (lambda_val), γ (k)")
    md_lines.append("- §3.3.2：α (causal_lambda), β (csc_lambda_0), τ_C (tau_C), ε (epsilon_floor)")
    md_lines.append("- §3.3.3：μ (neighbor_alpha), top-k (csls_k), T (csls_iter)\n")
    
    param_order = ["lambda", "k", "alpha", "beta", "tauC", "eps", "mu", "topk", "T"]
    for ds in sorted(all_data.keys()):
        md_lines.append(f"\n## {ds}\n")
        for pk in param_order:
            if pk not in all_data[ds]:
                continue
            meta = PARAM_META[pk]
            md_lines.append(f"\n### {ds} — Sweep {meta['title']}  ({meta['section']})\n")
            md_lines.append(f"| {meta['sym']} | H@1 | H@10 | H@50 | MRR |")
            md_lines.append("|------|------|------|------|------|")
            for v in sorted(all_data[ds][pk].keys()):
                h1, h10, h50, mrr = all_data[ds][pk][v]
                v_str = f"{v:g}" if isinstance(v, float) else str(v)
                h50_str = f"{h50:.4f}" if h50 > 0 else "—"
                md_lines.append(
                    f"| {v_str} | {h1:.4f} | {h10:.4f} | {h50_str} | {mrr:.4f} |"
                )

    md_text = "\n".join(md_lines)
    print(md_text)

    out_md = outdir / "param_sweep_9params.md"
    out_md.write_text(md_text, encoding="utf-8")
    print(f"\n💾 Markdown 表格已保存: {out_md}")


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
    out_json = outdir / "param_sweep_9params.json"
    out_json.write_text(json.dumps(serializable, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"💾 JSON 汇总已保存: {out_json}")


def print_summary_stats(all_data):
    print("\n" + "=" * 70)
    print("📈 每个参数的 H@1 平台分析")
    print("=" * 70)
    
    param_order = ["lambda", "k", "alpha", "beta", "tauC", "eps", "mu", "topk", "T"]
    for ds in sorted(all_data.keys()):
        print(f"\n### {ds}")
        print(f"{'参数':<12}{'H@1 max':<12}{'H@1 min':<12}{'极差':<12}{'最优值':<12}")
        print("-" * 60)
        for pk in param_order:
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
    ap.add_argument("--fbdb-log", type=str, nargs='+', required=True,
                    help="FBDB 的 6 参数 log 路径（可多个）")
    ap.add_argument("--fbyg-log", type=str, nargs='+', required=True,
                    help="FBYG 的 6 参数 log 路径（可多个）")
    ap.add_argument("--fbdb-T-json", type=str, default=None,
                    help="FBDB T 扫描 JSON 路径")
    ap.add_argument("--fbyg-T-json", type=str, default=None,
                    help="FBYG T 扫描 JSON 路径")
    ap.add_argument("--outdir", type=str, default="results_param_sweep")
    ap.add_argument("--no-alpha-beta", action="store_true",
                    help="不加载历史 α/β 数据（默认会加载）")
    args = ap.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 1) 解析 6 参数 log
    fbdb_log_data = []
    for f in args.fbdb_log:
        p = Path(f)
        if not p.is_file():
            print(f"❌ {p} 不存在", file=sys.stderr); sys.exit(1)
        print(f"📖 解析 FBDB log: {p}")
        fbdb_log_data.append(parse_log(p))
    
    fbyg_log_data = []
    for f in args.fbyg_log:
        p = Path(f)
        if not p.is_file():
            print(f"❌ {p} 不存在", file=sys.stderr); sys.exit(1)
        print(f"📖 解析 FBYG log: {p}")
        fbyg_log_data.append(parse_log(p))
    
    # 2) 解析 T JSON
    T_data = defaultdict(lambda: defaultdict(dict))
    if args.fbdb_T_json:
        T_data["FBDB15K"]["T"] = parse_T_json(Path(args.fbdb_T_json), "FBDB15K")
    if args.fbyg_T_json:
        T_data["FBYG15K"]["T"] = parse_T_json(Path(args.fbyg_T_json), "FBYG15K")
    
    # 3) 加载历史 α/β
    alpha_beta_data = defaultdict(lambda: defaultdict(dict))
    if not args.no_alpha_beta:
        for ds, by_param in HISTORICAL_ALPHA_BETA.items():
            for pk, by_val in by_param.items():
                for v, m in by_val.items():
                    alpha_beta_data[ds][pk][v] = m
        print(f"📖 加载历史 α/β 数据（0428 sensitivity 实验）")
    
    # 合并所有数据
    all_data = merge(*fbdb_log_data, *fbyg_log_data, T_data, alpha_beta_data)
    
    # 概览
    print("\n📊 解析结果概览：")
    param_order = ["lambda", "k", "alpha", "beta", "tauC", "eps", "mu", "topk", "T"]
    for ds in sorted(all_data.keys()):
        print(f"  {ds}:")
        for pk in param_order:
            n = len(all_data[ds].get(pk, {}))
            sym = PARAM_META[pk]["sym"]
            mark = "✅" if n > 0 else "❌"
            print(f"    {mark} {sym:<8} : {n} 个点")
    
    print_markdown_tables(all_data, outdir)
    plot_nine_panels(all_data, outdir)
    save_json(all_data, outdir)
    print_summary_stats(all_data)


if __name__ == "__main__":
    main()