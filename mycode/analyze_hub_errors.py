"""
Experiment 3: Hub Node Misalignment Analysis

For each dataset, compute:
  1. final_emb from baseline checkpoint
  2. ρ(e) = mean top-k similarity for each e in test_right (candidate side)
  3. Define hub = ρ top-X% entities in test_right
  4. For both baseline and ours inference:
       - Get Top-1 prediction for each test sample
       - Count errors (rank > 0)
       - Count "errors where Top-1 lands in hub"
       - Report hub-error rate

Outputs:
  - hub_stats_<dataset>.csv  : per-sample details
  - hub_summary_<dataset>.csv: hub-rate at multiple thresholds
  - stdout summary table

Reuses eval_by_stratum.py helpers:
  - build_args()
  - locate_checkpoint()
  - load_data unpacking order (test_ill at idx 3, eval_ill at idx 4)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ---- reuse helpers from eval_by_stratum.py ----
# 假设 eval_by_stratum.py 在同一目录
from eval_perturb import build_args, locate_checkpoint  # noqa

from src.data import load_data
from src.utils import pairwise_distances, csls_sim
from model.MEAformer import MEAformer  # adapt import path if needed


# ============================================================
# Inference paths
# ============================================================
def run_baseline_inference(model, KGs, test_left, test_right, args):
    """Baseline (vanilla MEAformer) inference -> distance matrix + final_emb."""
    model.eval()
    with torch.no_grad():
        final_emb, _ = model.joint_emb_generat()
        final_emb = F.normalize(final_emb)

    el = torch.LongTensor(test_left).cuda()
    er = torch.LongTensor(test_right).cuda()
    with torch.no_grad():
        distance = pairwise_distances(final_emb[el], final_emb[er])
        if args.csls:
            distance = 1 - csls_sim(1 - distance, args.csls_k)
    return distance.cpu().numpy(), final_emb.detach().cpu().numpy()


def run_ours_inference(model, KGs, test_left, test_right, args,
                        causal_alpha, csc_alpha, neighbor_alpha):
    """
    Ours inference. Reuse the exact same code path used in Experiment 2.

    !!! 重要 !!!
    把实验 2 用来跑 ours 的函数 import 进来,在这里调用。
    占位实现:假设有一个 'ours_inference' 函数返回 (distance, final_emb_ours)。

    e.g.
        from ours_inference import ours_inference
        distance, final_emb = ours_inference(
            model, KGs, test_left, test_right, args,
            causal_alpha=causal_alpha,
            csc_alpha=csc_alpha,
            neighbor_alpha=neighbor_alpha,
        )
    """
    from ours_inference import ours_inference   # <-- replace with实验2的实际入口
    distance, final_emb = ours_inference(
        model, KGs, test_left, test_right, args,
        causal_alpha=causal_alpha,
        csc_alpha=csc_alpha,
        neighbor_alpha=neighbor_alpha,
    )
    return distance, final_emb


# ============================================================
# Hub computation
# ============================================================
def compute_hub_score(distance, k=10):
    """
    Compute N_k(e) = k-occurrence count for each candidate (test_right) entity.

    N_k(e) = number of test_left queries that rank e within their top-k
             nearest candidates (under the given distance matrix).

    This follows Radovanović et al. (2010): a hub is an entity that is
    "over-attracted" by many queries — exactly the geometric distortion
    that §3.3.3 targets.

    Parameters
    ----------
    distance : (Q, M) numpy
        Distance from each test_left query (Q rows) to each test_right
        candidate (M columns). Use baseline distance to avoid circular
        evidence (the same hub set is then applied to both methods).
    k : int
        Neighborhood size.

    Returns
    -------
    Nk : (M,) int numpy array
        Hub score for each candidate. Larger = more hub-like.
    """
    Q, M = distance.shape
    # 对每个 query 取 top-k 最近的候选索引
    topk_idx = np.argpartition(distance, k, axis=1)[:, :k]   # (Q, k)
    Nk = np.zeros(M, dtype=np.int64)
    flat = topk_idx.reshape(-1)
    np.add.at(Nk, flat, 1)
    return Nk



def get_hub_mask(rho, top_pct):
    """Return boolean mask of size len(rho); True if e is in top_pct% by ρ."""
    n_hub = max(1, int(round(len(rho) * top_pct / 100.0)))
    threshold = np.partition(rho, -n_hub)[-n_hub]
    return rho >= threshold


# ============================================================
# Error stats
# ============================================================
def analyze_errors(distance, hub_mask_local):
    """
    distance       : (Q, M) numpy, distance from test_left to test_right
                     l2r convention: ground truth at column i for row i
    hub_mask_local : (M,) bool, hub flag in the candidate (test_right) index space

    Returns dict with:
      n_total, n_error, n_error_hub, hub_error_rate, hub_population_rate
    """
    Q, M = distance.shape

    # 1. 算出结果后，立刻转回 CPU 的 numpy 数组
    pred_top1 = distance.argmin(axis=1)
    if hasattr(pred_top1, 'cpu'): # 如果是 Tensor
        pred_top1 = pred_top1.cpu().numpy()

    gt = np.arange(Q)
    is_error = pred_top1 != gt
    n_total = Q
    n_error = int(is_error.sum())

    # 2. 确保 hub_mask_local 也是 numpy 数组
    if hasattr(hub_mask_local, 'cpu'):
        hub_mask_local = hub_mask_local.cpu().numpy()

    error_top1_hub = hub_mask_local[pred_top1] & is_error
    n_error_hub = int(error_top1_hub.sum())

    # Q, M = distance.shape
    # pred_top1 = distance.argmin(axis=1)                   # (Q,)
    # gt = np.arange(Q)                                     # diagonal
    # is_error = pred_top1 != gt
    # n_total = Q
    # n_error = int(is_error.sum())
    # # 错误样本里, Top-1 落在 hub 的占比
    # error_top1_hub = hub_mask_local[pred_top1] & is_error
    # n_error_hub = int(error_top1_hub.sum())
    return {
        'n_total': n_total,
        'n_error': n_error,
        'n_error_hub': n_error_hub,
        'hub_error_rate': n_error_hub / max(n_error, 1),
        'hub_population_rate': hub_mask_local.mean(),
    }


# ============================================================
# Main
# ============================================================
def main(dataset, split, rate, k_neighbor=10,
         thresholds=(5, 10, 20, 30),
         causal_alpha=0.0, csc_alpha=0.0, neighbor_alpha=0.0,
         out_dir='eval_output'):
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. args + data ----
    args = build_args(dataset, split, rate,0)
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    # KGs, non_train, train_ill, test_ill, eval_ill, test_ill_ = load_data(logger, args)
    KGs, _, train_ill, test_ill, eval_ill, test_ill_ = load_data(logger, args)
    assert test_ill is not None, "load_data 返回顺序又出错了, 检查解包"
    test_left  = test_ill[:, 0]
    test_right = test_ill[:, 1]

    # ---- 2. load checkpoint ----
    # ckpt_path = locate_checkpoint(dataset, split, rate)

    ckpt_path = "/data0/hwx/mmea_copy/data/mmkg/MEAformer/save/FBDB15K_0.2_.pkl"
    # ckpt_path = "/data0/hwx/mmea_copy/data/mmkg/MEAformer/save/FBYG15K_rate_0.2_.pkl"
    # ckpt_path = "/data0/hwx/mmea_copy/data/mmkg/MEAformer/save/v2_dbp_zh_wo_surf_seed1_.pkl"


    print(f"[exp3] loading checkpoint: {ckpt_path}")
    model = MEAformer(KGs, args).cuda()
    state = torch.load(ckpt_path, map_location='cuda')
    # model.load_state_dict(state if not isinstance(state, dict) or 'state_dict' not in state
    #                       else state['state_dict'])
    model.load_state_dict(
        state if not isinstance(state, dict) or 'state_dict' not in state else state['state_dict'], 
        strict=False
    )

    # ---- 3. baseline inference ----
    print("[exp3] running baseline inference...")
    dist_base, final_emb = run_baseline_inference(model, KGs, test_left, test_right, args)

    # ---- 4. ours inference ----
    print("[exp3] running ours inference...")
    dist_ours, _ = run_ours_inference(
        model, KGs, test_left, test_right, args,
        causal_alpha=causal_alpha,
        csc_alpha=csc_alpha,
        neighbor_alpha=neighbor_alpha,
    )

    # ---- 5. ρ on test_right ----
    print(f"[exp3] computing ρ with k={k_neighbor} on test_right (|cand|={len(test_right)})...")
    # final_emb 是 baseline 的相似度空间; ours 不重新算 ρ, 用同一套 hub 定义,
    # 保证两种方法在同一 hub 集合上做对比.
    rho = compute_hub_score(dist_base, k=k_neighbor)

    # ---- 6. sweep thresholds ----
    rows = []
    for pct in thresholds:
        hub_mask = get_hub_mask(rho, pct)
        st_b = analyze_errors(dist_base, hub_mask)
        st_o = analyze_errors(dist_ours, hub_mask)
        rows.append({
            'threshold_pct': pct,
            'n_hub': int(hub_mask.sum()),
            'baseline_n_error': st_b['n_error'],
            'baseline_n_error_hub': st_b['n_error_hub'],
            'baseline_hub_error_rate': st_b['hub_error_rate'],
            'ours_n_error': st_o['n_error'],
            'ours_n_error_hub': st_o['n_error_hub'],
            'ours_hub_error_rate': st_o['hub_error_rate'],
            'reduction_abs': st_b['hub_error_rate'] - st_o['hub_error_rate'],
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, f'hub_summary_{dataset}_{split}.csv')
    df.to_csv(out_csv, index=False)
    print(f"[exp3] saved: {out_csv}")

    # ---- 7. per-sample dump (for top 10% threshold) ----
    hub_mask_10 = get_hub_mask(rho, 10)
    pred_b = dist_base.argmin(axis=1)
    pred_o = dist_ours.argmin(axis=1)

    if hasattr(pred_o, 'cpu'):
        pred_o = pred_o.cpu().numpy()

    gt = np.arange(len(test_right))

    if hasattr(gt, 'cpu'):
        gt = gt.cpu().numpy()

    per_sample = pd.DataFrame({
        'test_left_id'   : test_left,
        'gt_right_id'    : test_right,
        'baseline_pred_local'  : pred_b,
        'ours_pred_local'      : pred_o,
        'baseline_is_error'    : (pred_b != gt).astype(int),
        'ours_is_error'        : (pred_o != gt).astype(int),
        'baseline_top1_is_hub' : hub_mask_10[pred_b].astype(int),
        'ours_top1_is_hub'     : hub_mask_10[pred_o].astype(int),
    })
    out_csv2 = os.path.join(out_dir, f'hub_stats_{dataset}_{split}.csv')
    per_sample.to_csv(out_csv2, index=False)
    print(f"[exp3] saved: {out_csv2}")

    # ---- 8. print summary table ----
    print("\n" + "="*88)
    print(f"Summary: {dataset} (split={split}, rate={rate}, k_neighbor={k_neighbor})")
    print("="*88)
    print(f"{'thr%':<6}{'n_hub':<8}{'base_err':<10}{'base_hub%':<12}"
          f"{'ours_err':<10}{'ours_hub%':<12}{'Δabs':<10}")
    print("-"*88)
    for r in rows:
        print(f"{r['threshold_pct']:<6}"
              f"{r['n_hub']:<8}"
              f"{r['baseline_n_error']:<10}"
              f"{r['baseline_hub_error_rate']*100:<12.2f}"
              f"{r['ours_n_error']:<10}"
              f"{r['ours_hub_error_rate']*100:<12.2f}"
              f"{r['reduction_abs']*100:<+10.2f}")


# ============================================================
# CLI
# ============================================================
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                   choices=['FBDB15K', 'FBYG15K', 'DBP15K'])
    p.add_argument('--split', required=True)         # norm / zh_en
    p.add_argument('--rate', type=float, required=True)
    p.add_argument('--k_neighbor', type=int, default=10)
    p.add_argument('--causal_alpha', type=float, default=0.0)
    p.add_argument('--csc_alpha', type=float, default=0.0)
    p.add_argument('--neighbor_alpha', type=float, default=0.0)
    p.add_argument('--thresholds', type=int, nargs='+', default=[5, 10, 20, 30])
    p.add_argument('--out_dir', default='eval_output')
    p.add_argument('--num_layers', default= 1 )
    p.add_argument('--fusion_layers', default= 1 )
    a = p.parse_args()

    main(dataset=a.dataset, split=a.split, rate=a.rate,
         k_neighbor=a.k_neighbor,
         thresholds=tuple(a.thresholds),
         causal_alpha=a.causal_alpha,
         csc_alpha=a.csc_alpha,
         neighbor_alpha=a.neighbor_alpha,
         out_dir=a.out_dir)