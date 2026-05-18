"""
density_probe.py

统计三个数据集 rel_features / att_features / img / gph(adj) 的非零密度,
用硬数字佐证 "DBP15K 的 rel/att 比 FB 系列丰富" 这一 §4.5.3 归因。

用法 (像 eval_gph_perturb.py 一样, 改 data_choice 跑三次, 不需要 ckpt):
  python density_probe.py --data_choice FBDB15K --data_split norm --data_rate 0.2 --gpu 0
  python density_probe.py --data_choice FBYG15K --data_split norm --data_rate 0.2 --gpu 0
  python density_probe.py --data_choice DBP15K  --data_split zh_en --data_rate 0.3 --gpu 0
"""

import os.path as osp
import sys
import argparse
import logging

import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from config import cfg as base_cfg
from torchlight import set_seed
from src.data import load_data

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def build_args(data_choice, data_split, data_rate, gpu):
    cfg_obj = base_cfg()
    fake_argv = [
        '--data_choice', data_choice,
        '--data_split', data_split,
        '--data_rate', str(data_rate),
        '--gpu', str(gpu),
        '--only_test', '1',
        '--save_model', '0',
        '--dist', '0',
        '--model_name', 'MEAformer',
        '--csls',
        '--csls_k', '3',
        '--enable_sota',
    ]
    old_argv = sys.argv
    sys.argv = [old_argv[0]] + fake_argv
    try:
        cfg_obj.get_args()
        args = cfg_obj.update_train_configs()
    finally:
        sys.argv = old_argv
    args.device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    args.rank = 0
    args.hidden_units = "300,300,300"
    args.heads = "2,2"
    return args


def density_stats(name, feat):
    """打印一个特征矩阵的非零密度统计。"""
    if feat is None:
        print(f"  {name:18s}: None (模态未加载)")
        return
    if hasattr(feat, 'detach'):
        feat = feat.detach().cpu().numpy()
    feat = np.asarray(feat)
    if feat.ndim != 2:
        print(f"  {name:18s}: shape={feat.shape} (非 2D, 跳过)")
        return

    N, D = feat.shape
    nz_mask = np.abs(feat) > 1e-8
    # 每个实体平均非零维度数
    nz_per_ent = nz_mask.sum(axis=1)
    # 全是 0 的实体比例 (该模态对这些实体完全无信息)
    empty_ent_ratio = (nz_per_ent == 0).mean()
    overall_density = nz_mask.mean()

    print(f"  {name:18s}: shape=({N},{D})  "
          f"非零密度={overall_density:.4f}  "
          f"每实体非零维={nz_per_ent.mean():.1f}±{nz_per_ent.std():.1f}  "
          f"全零实体占比={empty_ent_ratio:.4f}")


def adj_density(adj, N):
    """gph 模态的图结构密度: 平均度数。"""
    if adj is None:
        print(f"  {'gph(adj)':18s}: None")
        return
    if adj.is_sparse:
        nnz = adj._nnz()
    else:
        nnz = (torch.abs(adj) > 1e-8).sum().item()
    avg_degree = nnz / max(N, 1)
    print(f"  {'gph(adj)':18s}: N={N}  非零边={nnz}  平均度数={avg_degree:.2f}")


def main(a):
    set_seed(42)
    args = build_args(a.data_choice, a.data_split, a.data_rate, a.gpu)

    print("=" * 80)
    print(f"Density probe: {a.data_choice} (split={a.data_split}, rate={a.data_rate})")
    print("=" * 80)

    KGs, *_ = load_data(logger, args)

    N = KGs.get("ent_num", None)
    print(f"\n[{a.data_choice}] ent_num = {N}")
    print("-" * 80)

    density_stats("rel_features", KGs.get("rel_features"))
    density_stats("att_features", KGs.get("att_features"))
    density_stats("images_list", KGs.get("images_list"))
    density_stats("name_features", KGs.get("name_features"))
    density_stats("char_features", KGs.get("char_features"))

    adj = KGs.get("adj")
    adj_density(adj, N if N else (adj.shape[0] if adj is not None else 0))

    print("-" * 80)
    print(f"[{a.data_choice}] 关注 rel_features / att_features 的 '每实体非零维' 和 "
          f"'全零实体占比' —— FB 系列应显著稀疏于 DBP15K")
    print("=" * 80)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_choice', required=True)
    ap.add_argument('--data_split', required=True)
    ap.add_argument('--data_rate', type=float, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    main(ap.parse_args()) 