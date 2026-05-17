"""
eval_perturb.py

实验 2: 模态先验偏倚的存在性 —— 视觉扰动下的鲁棒性对比。

在干净数据和扰动数据上分别跑两个版本的 inference, 对比 H@1 衰减幅度:
  - baseline: 同 checkpoint, 关掉 use_causal_bias / use_csc / use_neighbor
  - ours:     同 checkpoint, 打开三个推理时模块, 用干净数据上的最优 α

用法:
  python eval_perturb.py \\
      --data_choice FBDB15K --data_split norm --data_rate 0.2 \\
      --causal_alpha 0.4 --csc_alpha 0.2 --neighbor_alpha 0.2 \\
      --perturb_ratios 0.0,0.1,0.2,0.3 --seed 42 \\
      --gpu 0 --output_csv eval_output/perturb_FBDB15K.csv

关键设计:
  1. baseline 和 ours 使用同一个 checkpoint, 仅推理时差异
  2. 每个扰动比例 p 跑一次 inference 即可, 同一次的 distance 矩阵分两个分支
     (这样可以减半 inference 时间, 因为联合 emb 和模态 emb 都不变)
  3. random_seed=42 固定, 扰动可复现
"""

import os
import os.path as osp
import sys
import argparse
import json
import csv
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import scipy

# 复用 codebase 的工具
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from config import cfg as base_cfg
from torchlight import set_seed
from src.data import load_data
from src.utils import pairwise_distances, csls_sim
from model import MEAformer

# 本目录里的扰动注入工具
from inject_visual_noise import inject_visual_noise, diagnose


# ============================================================
# 构造 args (复用实验 1 eval_by_stratum.py 的 build_args 逻辑)
# ============================================================

def build_args(data_choice, data_split, data_rate, gpu, args_overrides=None):
    """
    构造与训练时一致的 args。重要: 必须和训练用的 run_meaformer_zhat.sh 保持一致,
    否则模型结构会对不上, checkpoint 加载会出错。
    """
    cfg_obj = base_cfg()
    # 模仿命令行解析
    fake_argv = [
        '--data_choice', data_choice,
        '--data_split', data_split,
        '--data_rate', str(data_rate),
        '--gpu', str(gpu),
        '--only_test', '1',
        '--save_model', '0',
        '--dist', '0',
        '--model_name', 'MEAformer',
        # ↓ 与 run_meaformer_zhat.sh 一致
        '--csls',
        '--csls_k', '3',
        '--enable_sota', 
        # iterative learning 不开
        # '--il',
    ]
    # 临时替换 sys.argv 让 get_args() 读到
    old_argv = sys.argv
    sys.argv = [old_argv[0]] + fake_argv
    try:
        cfg_obj.get_args()
        args = cfg_obj.update_train_configs()
    finally:
        sys.argv = old_argv

    args.device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    args.rank = 0

    # 强制设置一些 attribute 防止 _test() 里 getattr 找不到
    defaults = {
        'use_causal_bias': 0,
        'use_csc': 0,
        'use_neighbor': 0,
        'use_sample_schedule': 1,  # checkpoint 已经训过, 这个无所谓但写上
        'causal_lambda': 0.0,
        'csc_lambda_0': 0.0,
        'neighbor_alpha': 0.0,
        'tau_C': 1.0,
        'epsilon_floor': 0.01,
        'csls_iter': 1,
        'use_bidirectional_consistency': 0,
        'ablate_modal': "",
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    
    args.hidden_size = 300
    args.num_hidden_layers = 2
    args.attr_dim = 300
    args.img_dim = 300
    args.name_dim = 300
    args.char_dim = 300

    if args_overrides:
        for k, v in args_overrides.items():
            setattr(args, k, v)

    args.entity_embedding_dim = 300
    args.hidden_dim = 300
    args.gph_dim = 300
    args.hidden_units = "300,300,300"
    args.heads = "2,2"

    return args


# ============================================================
# Checkpoint 加载 (复用 locate_checkpoint 思路, 模糊匹配)
# ============================================================

def locate_checkpoint(args):
    """
    在 data/mmkg/MEAformer/save/ 下模糊匹配匹配 checkpoint 文件名。
    命名规则:
      FB-X:    MEAformer_FBDB15K_0.2_v1_norm_0.2.pkl
      DBP15K:  MEAformer_DBP15K_zh_en_v1_zh_en_0.3.pkl
    """
    save_dir = osp.join(args.data_path, args.model_name, 'save')
    if not osp.exists(save_dir):
        raise FileNotFoundError(f"checkpoint dir not found: {save_dir}")

    # 必须同时匹配 data_choice 和 data_split
    candidates = []
    for fn in os.listdir(save_dir):
        if not fn.endswith('.pkl'):
            continue
        if args.data_choice not in fn:
            continue
        if args.data_choice in ['DBP15K']:
            if args.data_split not in fn:
                continue
        candidates.append(fn)

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"no checkpoint matches data_choice={args.data_choice}, "
            f"data_split={args.data_split} in {save_dir}"
        )
    if len(candidates) > 1:
        print(f"[locate_checkpoint] WARNING: multiple matches: {candidates}, using first")

    ckpt_path = osp.join(save_dir, candidates[0])

    # ckpt_path = "/data0/hwx/mmea_copy/data/mmkg/MEAformer/save/FBDB15K_0.2_.pkl"
    # ckpt_path = "/data0/hwx/mmea_copy/data/mmkg/MEAformer/save/FBYG15K_rate_0.2_.pkl"
    ckpt_path = "/data0/hwx/mmea_copy/data/mmkg/MEAformer/save/v2_dbp_zh_wo_surf_seed1_.pkl"

    # 顺便找一下 _cj.json
    cj_path = ckpt_path.replace('.pkl', '_cj.json')

    return ckpt_path, cj_path


def load_model(KGs, args, ckpt_path, cj_path):
    args.num_hidden_layers = 1  # 强制 2 层 fusion
    args.hidden_units = "300,300,300"
    args.heads = "2,2"
    print(f"DEBUG hidden_size={args.hidden_size}, num_hidden_layers={args.num_hidden_layers}")
    model = MEAformer(KGs, args)
    print(f"DEBUG args.num_hidden_layers={args.num_hidden_layers}")
    print(f"DEBUG fusion_layer len={len(model.multimodal_encoder.fusion.fusion_layer)}")
    print(f"DEBUG entity_emb shape: {model.multimodal_encoder.entity_emb.weight.shape}")

    state_dict = torch.load(ckpt_path, map_location=args.device)

    result = model.load_state_dict(state_dict, strict=False)
    print(f"DEBUG missing keys: {result.missing_keys[:5]}")
    print(f"DEBUG unexpected keys: {result.unexpected_keys[:5]}")
    print(f"DEBUG state_dict keys with 'fusion_layer.1': {[k for k in state_dict.keys() if 'fusion_layer.1' in k][:3]}")
    print(f"DEBUG model keys with 'fusion_layer.1': {[k for k in model.state_dict().keys() if 'fusion_layer.1' in k][:3]}")
    

    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()

    # 加载 C_j (因果置信度), 推理时用
    if osp.exists(cj_path):
        with open(cj_path, 'r') as f:
            cj_data = json.load(f)
        if hasattr(model, 'causal_Cj'):
            model.causal_Cj.update(cj_data)
            print(f"[load_model] loaded C_j: {cj_data}")
    else:
        print(f"[load_model] WARNING: C_j file not found at {cj_path}, "
              f"will use defaults (uniform 0.5)")

    return model


# ============================================================
# Inference (双分支: baseline 和 ours 在同一次中分别算)
# ============================================================

@torch.no_grad()
def run_inference_both(model, args, test_left, test_right,
                       causal_alpha, csc_alpha, neighbor_alpha,
                       csls_k=3):
    """
    一次性算出 baseline 和 ours 的 H@1 / H@10 / MRR。
    两边共用 final_emb 和模态 emb (只算一次), 在 distance 层分两条路径。

    Returns:
        dict with keys: baseline_h1/h10/mrr, ours_h1/h10/mrr
    """
    # ====== 一次性提取 final_emb 和模态 emb ======
    final_emb, weight_norm = model.joint_emb_generat()
    final_emb = F.normalize(final_emb)

    gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, _, _, _ = \
        model.joint_emb_generat(only_joint=False)

    # ====== 路径 A: baseline ======
    distance_baseline = pairwise_distances(final_emb[test_left], final_emb[test_right])
    if args.csls:
        distance_baseline = 1 - csls_sim(1 - distance_baseline, csls_k)
    metrics_baseline = compute_metrics(distance_baseline)

    # ====== 路径 B: ours ======
    distance_ours = pairwise_distances(final_emb[test_left], final_emb[test_right])

    # B1: 邻居增强
    if neighbor_alpha > 0:
        adj = model.adj
        neighbor_emb = torch.sparse.mm(adj, final_emb)
        neighbor_emb = F.normalize(neighbor_emb)
        neighbor_distance = pairwise_distances(neighbor_emb[test_left],
                                                neighbor_emb[test_right])
        distance_ours = (1 - neighbor_alpha) * distance_ours + neighbor_alpha * neighbor_distance

    # B2: 因果证据聚合 (模态 C_m 加权距离)
    if causal_alpha > 0:
        import math
        causal_Cj = model.causal_Cj
        modal_emb_dict = {'img': img_emb, 'att': att_emb, 'rel': rel_emb,
                          'gph': gph_emb, 'name': name_emb, 'char': char_emb}
        modal_distances, modal_weights = [], []
        for m_name, m_emb in modal_emb_dict.items():
            if m_emb is None:
                continue
            Cj = max(causal_Cj.get(m_name, 0.0), 1e-4)
            m_emb_norm = F.normalize(m_emb)
            m_dist = pairwise_distances(m_emb_norm[test_left], m_emb_norm[test_right])
            modal_distances.append(m_dist)
            modal_weights.append(Cj)

        if len(modal_distances) > 0:
            tau_C = getattr(args, 'tau_C', 1.0)
            eps = getattr(args, 'epsilon_floor', 0.01)
            exp_w = [math.exp(w / tau_C) for w in modal_weights]
            s = sum(exp_w)
            modal_weights = [e / s for e in exp_w]
            M = len(modal_weights)
            modal_weights = [(1 - eps) * w + eps / M for w in modal_weights]
            causal_distance = sum(w * d for w, d in zip(modal_weights, modal_distances))
            distance_ours = (1 - causal_alpha) * distance_ours + causal_alpha * causal_distance

    # B3: 反事实一致性 (均匀加权 cf_joint)
    if csc_alpha > 0:
        modal_embs_list = [e for e in [img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb]
                           if e is not None]
        if len(modal_embs_list) >= 2:
            stacked = torch.stack([F.normalize(e, dim=-1) for e in modal_embs_list], dim=1)
            M = stacked.shape[1]
            uniform_w = torch.ones(stacked.shape[0], M, device=stacked.device) / M
            cf_joint = torch.sum(uniform_w.unsqueeze(2) * stacked, dim=1)
            cf_joint = F.normalize(cf_joint)
            cf_distance = pairwise_distances(cf_joint[test_left], cf_joint[test_right])
            distance_ours = (1 - csc_alpha) * distance_ours + csc_alpha * cf_distance

    # CSLS
    if args.csls:
        distance_ours = 1 - csls_sim(1 - distance_ours, csls_k)

    metrics_ours = compute_metrics(distance_ours)

    return {
        'baseline_h1': metrics_baseline['h1'],
        'baseline_h10': metrics_baseline['h10'],
        'baseline_mrr': metrics_baseline['mrr'],
        'ours_h1': metrics_ours['h1'],
        'ours_h10': metrics_ours['h10'],
        'ours_mrr': metrics_ours['mrr'],
    }


def compute_metrics(distance):
    """计算 l2r 方向的 H@1 / H@10 / MRR。distance: [N, N], 对角线为正确对齐。"""
    top_k = [1, 10]
    acc = np.zeros(len(top_k))
    mrr = 0.0
    N = distance.shape[0]

    for idx in range(N):
        _, indices = torch.sort(distance[idx, :], descending=False)
        rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
        mrr += 1.0 / (rank + 1)
        for i, k in enumerate(top_k):
            if rank < k:
                acc[i] += 1
    acc /= N
    mrr /= N
    return {'h1': float(acc[0]), 'h10': float(acc[1]), 'mrr': float(mrr)}


# ============================================================
# 主流程
# ============================================================

def main(args_cli):
    set_seed(args_cli.random_seed)

    # 1. 构造 args (与训练时一致)
    args = build_args(args_cli.data_choice, args_cli.data_split, args_cli.data_rate, args_cli.gpu)

    # 2. 加载数据 (一次, 后面只改 KGs["images_list"])
    print("=" * 70)
    print(f"Loading data for {args_cli.data_choice} (split={args_cli.data_split}, rate={args_cli.data_rate})")
    print("=" * 70)
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    KGs, non_train, train_ill, test_ill, eval_ill, test_ill_ = load_data(logger, args)
    # 注意: load_data 返回顺序 (KGs, non_train, train_ill, test_ill, eval_ill, test_ill_)
    # test_ill 是 EADataset, 用 .data 拿底层 numpy
    # 但 eval_by_stratum 那边好像是直接拿 train_ill = self.train_set.data, 这里要看清楚
    # 实际上 main.py 的 data_init() 是这样写的:
    #   self.train_set, self.eval_set, self.test_set, self.test_ill_
    # 然后 self.test_ill = self.test_set.data
    # 所以这里 test_ill 也是 dataset 对象
    if hasattr(test_ill, 'data'):
        test_ill_np = test_ill.data  # np.ndarray, shape [N, 2]
    else:
        test_ill_np = test_ill
    test_ill_np = np.asarray(test_ill_np)

    test_left = torch.LongTensor(test_ill_np[:, 0]).to(args.device)
    test_right = torch.LongTensor(test_ill_np[:, 1]).to(args.device)
    print(f"[main] test_ill shape: {test_ill_np.shape}, "
          f"left range: [{test_ill_np[:, 0].min()}, {test_ill_np[:, 0].max()}], "
          f"right range: [{test_ill_np[:, 1].min()}, {test_ill_np[:, 1].max()}]")

    # 3. 备份原始图像 (因为 KGs["images_list"] 会被改)
    images_orig = np.asarray(KGs["images_list"]).copy()

    # 4. 定位 checkpoint
    ckpt_path, cj_path = locate_checkpoint(args)
    print(f"[main] checkpoint: {ckpt_path}")
    print(f"[main] C_j file:   {cj_path}")

    # 5. 跑每个扰动比例
    perturb_ratios = [float(x) for x in args_cli.perturb_ratios.split(',')]
    results = []

    for p in perturb_ratios:
        print("\n" + "=" * 70)
        print(f"Perturbation ratio p = {p}")
        print("=" * 70)

        # 注入扰动
        if p == 0.0:
            KGs["images_list"] = images_orig
            print("[main] clean run (no perturbation)")
        else:
            images_p, perturb_ents = inject_visual_noise(
                images_orig, test_ill_np, p, seed=args_cli.seed
            )
            KGs["images_list"] = images_p
            diagnose(images_orig, images_p, perturb_ents)

        # 重新构造模型 (注意: KGs 改了, 模型的 self.img_features 会重新读)
        model = load_model(KGs, args, ckpt_path, cj_path)

        # 双分支 inference
        m = run_inference_both(
            model, args, test_left, test_right,
            causal_alpha=args_cli.causal_alpha,
            csc_alpha=args_cli.csc_alpha,
            neighbor_alpha=args_cli.neighbor_alpha,
            csls_k=args.csls_k
        )

        print(f"\n[result] p={p}:")
        print(f"  baseline:  H@1={m['baseline_h1']:.4f}  H@10={m['baseline_h10']:.4f}  MRR={m['baseline_mrr']:.4f}")
        print(f"  ours:      H@1={m['ours_h1']:.4f}  H@10={m['ours_h10']:.4f}  MRR={m['ours_mrr']:.4f}")
        print(f"  ours - baseline: ΔH@1 = {m['ours_h1'] - m['baseline_h1']:+.4f}")

        results.append({
            'dataset': args_cli.data_choice,
            'split': args_cli.data_split,
            'p': p,
            **m
        })

        # 清理模型, 防止 OOM
        del model
        torch.cuda.empty_cache()

    # 6. 写 CSV
    if args_cli.output_csv:
        os.makedirs(osp.dirname(args_cli.output_csv) or '.', exist_ok=True)
        with open(args_cli.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[main] results saved to {args_cli.output_csv}")

    # 7. 打印总结表
    print("\n" + "=" * 80)
    print(f"Summary: {args_cli.data_choice} (split={args_cli.data_split}, rate={args_cli.data_rate})")
    print(f"  causal_α={args_cli.causal_alpha}, csc_α={args_cli.csc_alpha}, neighbor_α={args_cli.neighbor_alpha}")
    print("=" * 80)
    print(f"{'p':<8}{'baseline H@1':<16}{'ours H@1':<14}{'ΔH@1':<12}{'baseline drop':<16}{'ours drop':<14}")
    print("-" * 80)
    h1_baseline_clean = results[0]['baseline_h1']
    h1_ours_clean = results[0]['ours_h1']
    for r in results:
        b_drop = r['baseline_h1'] - h1_baseline_clean if r['p'] > 0 else 0.0
        o_drop = r['ours_h1'] - h1_ours_clean if r['p'] > 0 else 0.0
        delta = r['ours_h1'] - r['baseline_h1']
        print(f"{r['p']:<8}{r['baseline_h1']:<16.4f}{r['ours_h1']:<14.4f}"
              f"{delta:<+12.4f}{b_drop:<+16.4f}{o_drop:<+14.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_choice', type=str, required=True,
                        choices=['FBDB15K', 'FBYG15K', 'DBP15K'])
    parser.add_argument('--data_split', type=str, required=True)
    parser.add_argument('--data_rate', type=float, required=True)
    parser.add_argument('--gpu', type=int, default=0)

    # α 参数 (干净数据上的最优)
    parser.add_argument('--causal_alpha', type=float, required=True)
    parser.add_argument('--csc_alpha', type=float, required=True)
    parser.add_argument('--neighbor_alpha', type=float, required=True)

    # 扰动设置
    parser.add_argument('--perturb_ratios', type=str, default='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7')
    parser.add_argument('--seed', type=int, default=42, help='seed for perturbation')
    parser.add_argument('--random_seed', type=int, default=42, help='seed for data split (must = training seed)')

    # 输出
    parser.add_argument('--output_csv', type=str, default='')

    args_cli = parser.parse_args()
    main(args_cli)