"""
轻量分析: Z_hat 分层 vs rho (局部密度 / hubness)

回答: 高 Z_hat 层的对齐对, 其 KG-2 侧候选实体是否系统性地具有更高的 rho?
如果是, 则建立 §3.3.1 (难度混杂 Z_hat) 与 §3.3.3 (几何 hubness rho) 的内在联系。

用法 (在 code/MEAformer 目录下):
    python analyze_zhat_vs_rho.py --data_choice FBDB15K --data_split norm --data_rate 0.2 --gpu 0
    python analyze_zhat_vs_rho.py --data_choice FBYG15K --data_split norm --data_rate 0.2 --gpu 0
    python analyze_zhat_vs_rho.py --data_choice DBP15K --data_split zh_en --data_rate 0.3 --gpu 0

前置: 已训好 checkpoint + 已跑 compute_z_hat.py 生成 z_hat csv

设计:
- rho 在 baseline 训好的 final_emb 空间算 (反映模型表征几何, 不是原始特征)
- rho 只算 KG-2 侧实体 (实体对齐 = 给定 KG-1 query 在 KG-2 候选里检索, hub 是 KG-2 侧)
- rho(e) = 该 KG-2 实体与其在 KG-2 内 top-k 近邻的平均相似度 (越高 = 越处于稠密区 = 越可能成为 hub)
- 对每个测试对齐对 (e1, e2), 取 e2 (KG-2 侧) 的 rho, 按 Z_hat 分层看 rho 是否单调上升
- 报 Spearman corr(Z_hat, rho_e2) + 分层均值 + Z_hat 与 rho 的相关系数 (检查是否冗余)

输出:
    ./analysis_output/{data_choice}_{data_split}_{data_rate}_zhat_vs_rho.csv
    控制台: 分层 rho 均值 + 相关性
"""

import os
import os.path as osp
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from config import cfg as cfg_class
from src.data import load_data
from src.utils import pairwise_distances, csls_sim
from torchlight import set_seed
from model import MEAformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def build_args(data_choice, data_split, data_rate, gpu=0):
    """复用 config.cfg, 和训练时 args 完全一致 (同 eval_by_stratum.py)"""
    fake_argv = [
        'analyze.py',
        '--gpu', str(gpu),
        '--eval_epoch', '1',
        '--only_test', '1',
        '--model_name', 'MEAformer',
        '--data_choice', data_choice,
        '--data_split', data_split,
        '--data_rate', str(data_rate),
        '--epoch', '500',
        '--lr', '5e-4',
        '--hidden_units', '300,300,300',
        '--save_model', '0',
        '--batch_size', '3500',
        '--csls',
        '--csls_k', '3',
        '--random_seed', '42',
        '--exp_name', 'Zhat_rho_analysis',
        '--exp_id', f'v1_{data_split}_{data_rate}',
        '--workers', '4',
        '--dist', '0',
        '--accumulation_steps', '1',
        '--scheduler', 'cos',
        '--attr_dim', '300',
        '--img_dim', '300',
        '--name_dim', '300',
        '--char_dim', '300',
        '--hidden_size', '300',
        '--tau', '0.1',
        '--structure_encoder', 'gat',
        '--num_attention_heads', '1',
        '--num_hidden_layers', '1',
        '--use_surface', '0',
        '--use_intermediate', '1',
        '--enable_sota',
        '--replay', '0',
    ]
    old_argv = sys.argv
    sys.argv = fake_argv
    try:
        c = cfg_class()
        c.get_args()
        args = c.update_train_configs()
    finally:
        sys.argv = old_argv
    return args


def locate_checkpoint(args):
    """模糊匹配 checkpoint (同 eval_by_stratum.py)"""
    save_dir = osp.join(args.data_path, args.model_name, 'save')
    expected_path = osp.join(save_dir, f"{args.exp_id}.pkl")
    if osp.exists(expected_path):
        return expected_path
    if not osp.exists(save_dir):
        raise FileNotFoundError(f"save dir not exists: {save_dir}")
    candidates = [f for f in os.listdir(save_dir) if f.endswith('.pkl')]
    matched = [f for f in candidates
               if args.data_choice in f and args.data_split in f and f"_{args.data_rate}" in f]
    if len(matched) == 0:
        matched = [f for f in candidates if args.data_choice in f and args.data_split in f]
    if len(matched) == 0:
        logger.error(f"Available checkpoints: {candidates}")
        raise FileNotFoundError(f"No checkpoint matching {args.data_choice}/{args.data_split}")
    if len(matched) > 1:
        matched.sort(key=lambda f: os.path.getmtime(osp.join(save_dir, f)), reverse=True)
    return osp.join(save_dir, matched[0])


def compute_rho_kg2(final_emb, test_right_ids, k, logger):
    """
    算 KG-2 侧候选实体的 rho (局部密度).

    rho(e) = 该实体与其在 *KG-2 候选集合内* top-k 近邻的平均 cosine 相似度.
    rho 越高 = 该实体处于表征空间稠密区 = 越可能吸引大量 query = 越可能成为 hub.

    final_emb: (ENT_NUM, dim), 已 F.normalize
    test_right_ids: KG-2 侧测试候选实体 id (unique)
    返回: dict {ent_id: rho}
    """
    # 取 KG-2 测试候选的 unique 实体
    unique_right = np.unique(test_right_ids)
    logger.info(f"  #unique KG-2 candidate entities: {len(unique_right)}")

    right_emb = final_emb[torch.LongTensor(unique_right).cuda()]  # (M, dim), 已 normalize

    # cosine 相似度矩阵 (M x M). 已 normalize, 所以 dot = cosine
    sim = right_emb @ right_emb.t()  # (M, M)

    # 排除自身: 对角线置 -inf
    M = sim.shape[0]
    sim.fill_diagonal_(-float('inf'))

    # top-k 近邻的平均相似度
    topk_vals, _ = torch.topk(sim, k=min(k, M - 1), dim=1)  # (M, k)
    rho_vals = topk_vals.mean(dim=1).cpu().numpy()  # (M,)

    rho_dict = {int(eid): float(r) for eid, r in zip(unique_right, rho_vals)}
    logger.info(f"  rho stats: min={rho_vals.min():.4f}, max={rho_vals.max():.4f}, "
                f"mean={rho_vals.mean():.4f}, std={rho_vals.std():.4f}")
    return rho_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_choice', type=str, required=True,
                        choices=['FBDB15K', 'FBYG15K', 'DBP15K'])
    parser.add_argument('--data_split', type=str, required=True)
    parser.add_argument('--data_rate', type=float, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lambda_z', type=float, default=0.5)
    parser.add_argument('--k', type=int, default=10, help='top-k neighbors for rho')
    parser.add_argument('--z_hat_dir', type=str, default='./z_hat_output')
    parser.add_argument('--output_dir', type=str, default='./analysis_output')
    parser.add_argument('--K', type=int, default=5, help='number of strata')
    cli_args = parser.parse_args()

    torch.cuda.set_device(cli_args.gpu)
    logger.info(f"using GPU {cli_args.gpu}, k={cli_args.k}")

    args = build_args(cli_args.data_choice, cli_args.data_split, cli_args.data_rate, gpu=cli_args.gpu)
    args.device = torch.device('cuda')
    set_seed(args.random_seed)

    # ----- 加载数据 -----
    logger.info("=" * 60)
    logger.info("Step 1: load data")
    logger.info("=" * 60)
    # 注意解包顺序: 第4个是 test_ill, 第5个是 eval_ill (总是 None)
    KGs, non_train, train_set, test_set, eval_set, test_ill_ = load_data(logger, args)
    test_ill = test_set.data
    test_left = torch.LongTensor(test_ill[:, 0]).cuda()
    test_right = torch.LongTensor(test_ill[:, 1]).cuda()
    N_test = test_ill.shape[0]
    logger.info(f"N_test = {N_test}")

    # ----- 加载 Z_hat -----
    logger.info("=" * 60)
    logger.info("Step 2: load Z_hat")
    logger.info("=" * 60)
    z_hat_path = osp.join(
        cli_args.z_hat_dir,
        f"{cli_args.data_choice}_{cli_args.data_split}_{cli_args.data_rate}_lambda{cli_args.lambda_z}.csv"
    )
    assert osp.exists(z_hat_path), f"Z_hat csv not exists: {z_hat_path}"
    z_data = np.genfromtxt(z_hat_path, delimiter=',', skip_header=1)
    z_e1 = z_data[:, 1].astype(int)
    z_e2 = z_data[:, 2].astype(int)
    z_hat = z_data[:, 12]  # z_hat 在第 13 列
    assert len(z_hat) == N_test, f"#z_hat ({len(z_hat)}) != #test ({N_test}), random_seed mismatch?"
    # 一致性检查
    assert (z_e1 == test_ill[:, 0]).all() and (z_e2 == test_ill[:, 1]).all(), \
        "test_ill order mismatch with z_hat csv. Check random_seed=42."
    logger.info("test_ill order matches Z_hat csv")

    # ----- 构建模型, 加载 checkpoint -----
    logger.info("=" * 60)
    logger.info("Step 3: build model & load checkpoint")
    logger.info("=" * 60)
    model = MEAformer(KGs, args).cuda()
    ckpt_path = locate_checkpoint(args)
    logger.info(f"loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cuda')
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # ----- inference 得到 final_emb -----
    logger.info("=" * 60)
    logger.info("Step 4: inference -> final_emb")
    logger.info("=" * 60)
    with torch.no_grad():
        final_emb, weight_norm = model.joint_emb_generat()
        final_emb = F.normalize(final_emb)  # 归一化, 后面 dot = cosine
    logger.info(f"final_emb shape: {final_emb.shape}")

    # ----- 算 KG-2 侧 rho -----
    logger.info("=" * 60)
    logger.info(f"Step 5: compute rho on KG-2 candidates (k={cli_args.k})")
    logger.info("=" * 60)
    rho_dict = compute_rho_kg2(final_emb, test_ill[:, 1], cli_args.k, logger)

    # 对每个测试对齐对, 取 e2 (KG-2 侧) 的 rho
    rho_e2 = np.array([rho_dict[int(e2)] for e2 in test_ill[:, 1]])
    logger.info(f"rho_e2 (per pair) stats: min={rho_e2.min():.4f}, max={rho_e2.max():.4f}, mean={rho_e2.mean():.4f}")

    # ----- 分层分析 -----
    logger.info("=" * 60)
    logger.info(f"Step 6: stratify by Z_hat, look at rho per stratum (K={cli_args.K})")
    logger.info("=" * 60)
    K = cli_args.K
    quantile_pts = [(k + 1) / K for k in range(K - 1)]
    quantiles = np.quantile(z_hat, quantile_pts)
    strata = np.digitize(z_hat, quantiles)

    print()
    print(f"{'Stratum':<10} {'N':<6} {'z_hat range':<22} {'mean rho':<12} {'rho std':<10}")
    print("-" * 65)
    per_stratum = []
    for k in range(K):
        mask = strata == k
        n = int(mask.sum())
        if n == 0:
            continue
        rho_mean = float(rho_e2[mask].mean())
        rho_std = float(rho_e2[mask].std())
        z_lo, z_hi = float(z_hat[mask].min()), float(z_hat[mask].max())
        per_stratum.append({'stratum': k + 1, 'n': n, 'rho_mean': rho_mean})
        print(f"Q{k+1:<9} {n:<6} [{z_lo:.4f}, {z_hi:.4f}]   {rho_mean:<12.4f} {rho_std:<10.4f}")
    print()

    # ----- 相关性分析 -----
    logger.info("=" * 60)
    logger.info("Step 7: correlation analysis")
    logger.info("=" * 60)
    # Spearman corr(Z_hat, rho_e2): 期望正相关 (高 Z_hat -> 高 rho -> 更易成 hub)
    rho_spearman, p_sp = scipy.stats.spearmanr(z_hat, rho_e2)
    # Pearson corr: 看线性相关强度, 判断是否冗余
    pearson, p_pe = scipy.stats.pearsonr(z_hat, rho_e2)
    logger.info(f"Spearman corr(Z_hat, rho) = {rho_spearman:.4f}, p = {p_sp:.4e}")
    logger.info(f"Pearson  corr(Z_hat, rho) = {pearson:.4f}, p = {p_pe:.4e}")

    if rho_spearman > 0.15 and p_sp < 0.01:
        logger.info("  -> POSITIVE & significant: high-Z_hat entities ARE systematically denser/more hub-prone.")
        logger.info("     This bridges §3.3.1 (Z_hat) and §3.3.3 (rho/hubness).")
    elif rho_spearman > 0:
        logger.info("  -> weakly positive. Some link between difficulty and density, but not strong.")
    else:
        logger.info("  -> NOT positive. Z_hat and rho capture different aspects (no redundancy, but no bridge either).")

    # 冗余性判断
    if abs(pearson) > 0.8:
        logger.warning(f"  [!] |Pearson|={abs(pearson):.3f} > 0.8: Z_hat and rho may be REDUNDANT. "
                        f"Reviewers might ask if §3.3.1 and §3.3.3 overlap. Consider discussing distinction.")
    elif 0.3 <= abs(pearson) <= 0.6:
        logger.info(f"  [+] |Pearson|={abs(pearson):.3f} in [0.3, 0.6]: IDEAL. "
                     f"Linked but not redundant -- supports unified framework w/o overlap concern.")
    else:
        logger.info(f"  |Pearson|={abs(pearson):.3f}: linkage exists but weak; both components have independent value.")

    # ----- 保存 -----
    os.makedirs(cli_args.output_dir, exist_ok=True)
    out_path = osp.join(
        cli_args.output_dir,
        f"{cli_args.data_choice}_{cli_args.data_split}_{cli_args.data_rate}_zhat_vs_rho.csv"
    )
    header = "test_idx,e1_id,e2_id,z_hat,stratum,rho_e2"
    out_data = np.column_stack([
        np.arange(N_test), test_ill[:, 0], test_ill[:, 1],
        z_hat, strata + 1, rho_e2,
    ])
    fmt = ["%d", "%d", "%d", "%.6f", "%d", "%.6f"]
    np.savetxt(out_path, out_data, header=header, comments='', delimiter=',', fmt=fmt)
    logger.info(f"saved to {out_path}")

    # ----- 汇总 -----
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dataset: {cli_args.data_choice} {cli_args.data_split} (rate={cli_args.data_rate}), k={cli_args.k}")
    logger.info(f"Stratified mean rho: " + " | ".join([f"Q{r['stratum']}={r['rho_mean']:.4f}" for r in per_stratum]))
    mono = all(per_stratum[i]['rho_mean'] <= per_stratum[i+1]['rho_mean'] for i in range(len(per_stratum)-1))
    logger.info(f"rho monotonically increasing with Z_hat? {'YES' if mono else 'NO'}")
    logger.info(f"Spearman corr(Z_hat, rho) = {rho_spearman:.4f} (p={p_sp:.2e})")
    logger.info(f"Pearson  corr(Z_hat, rho) = {pearson:.4f}")


if __name__ == '__main__':
    main()