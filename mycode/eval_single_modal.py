"""
eval_single_modal.py

补跑 DBP15K wo_surf 的单模态独立对齐 H@1 (only img / only att / only rel /
only gph 各自)。用于佐证 §4.5.3: gph 坏了 DBP 是否有有效替代证据可切换。

逻辑直接复用 main.py 里 evaluate_Cj() 的做法:
  拿 joint_emb_generat(only_joint=False) 的各模态独立 emb,
  每个模态 F.normalize 后单独算 pairwise 距离 + H@1。

用法 (改 main 里 ckpt_path, 跟 gph 实验一致):
  python eval_single_modal.py --data_choice DBP15K --data_split zh_en --data_rate 0.3 --gpu 0
  # 顺便也可跑 FB 对照:
  python eval_single_modal.py --data_choice FBDB15K --data_split norm --data_rate 0.2 --gpu 0
  python eval_single_modal.py --data_choice FBYG15K --data_split norm --data_rate 0.2 --gpu 0
"""

import os.path as osp
import sys
import argparse
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from config import cfg as base_cfg
from torchlight import set_seed
from src.data import load_data
from src.utils import pairwise_distances, csls_sim
from model import MEAformer

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
    # 与实验 2 / gph 实验跑通版一致的维度硬编码
    args.hidden_size = 300
    args.num_hidden_layers = 2
    args.attr_dim = 300
    args.img_dim = 300
    args.name_dim = 300
    args.char_dim = 300
    args.entity_embedding_dim = 300
    args.hidden_dim = 300
    args.gph_dim = 300
    args.hidden_units = "300,300,300"
    args.heads = "2,2"
    return args


def load_model(KGs, args, ckpt_path, cj_path):
    args.num_hidden_layers = 1
    args.hidden_units = "300,300,300"
    args.heads = "2,2"
    model = MEAformer(KGs, args)
    state_dict = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()
    if cj_path and osp.exists(cj_path):
        with open(cj_path) as f:
            cj = json.load(f)
        if hasattr(model, 'causal_Cj'):
            model.causal_Cj.update(cj)
    return model


def compute_h1_h10_mrr(distance):
    top_k = [1, 10]
    acc = np.zeros(len(top_k))
    mrr = 0.0
    N = distance.shape[0]
    for idx in range(N):
        _, ind = torch.sort(distance[idx, :], descending=False)
        rank = (ind == idx).nonzero(as_tuple=False).squeeze().item()
        mrr += 1.0 / (rank + 1)
        for i, k in enumerate(top_k):
            if rank < k:
                acc[i] += 1
    acc /= N
    mrr /= N
    return float(acc[0]), float(acc[1]), float(mrr)


@torch.no_grad()
def main(a):
    set_seed(42)
    args = build_args(a.data_choice, a.data_split, a.data_rate, a.gpu)

    print("=" * 80)
    print(f"Single-modal alignment: {a.data_choice} (split={a.data_split}, rate={a.data_rate})")
    if a.img_perturb > 0:
        print(f"  >>> WITH visual perturbation p={a.img_perturb} (诊断: img 被扰后各单模态还剩多少)")
    print("=" * 80)
    KGs, non_train, train_ill, test_ill, eval_ill, test_ill_ = load_data(logger, args)
    test_ill_np = np.asarray(test_ill.data if hasattr(test_ill, 'data') else test_ill)
    test_left = torch.LongTensor(test_ill_np[:, 0]).to(args.device)
    test_right = torch.LongTensor(test_ill_np[:, 1]).to(args.device)

    # ===== 视觉扰动注入 (在 MEAformer 读 images_list 之前) =====
    if a.img_perturb > 0:
        from inject_visual_noise import inject_visual_noise, diagnose as img_diag
        images_orig = np.asarray(KGs["images_list"]).copy()
        images_p, perturb_ents = inject_visual_noise(
            images_orig, test_ill_np, a.img_perturb, seed=42
        )
        KGs["images_list"] = images_p
        img_diag(images_orig, images_p, perturb_ents)
    # ========================================================

    # ============================================
    # !!! 像 gph 实验一样, 手动改 ckpt_path !!!
    # FBDB15K:  /data0/hwx/mmea_copy/data/mmkg/MEAformer/save/FBDB15K_0.2_.pkl
    # FBYG15K:  /data0/hwx/mmea_copy/data/mmkg/MEAformer/save/FBYG15K_rate_0.2_.pkl
    # DBP15K:   /data0/hwx/mmea_copy/data/mmkg/MEAformer/save/v2_dbp_zh_wo_surf_seed1_.pkl
    # ============================================
    ckpt_path = "/data0/hwx/mmea_copy/data/mmkg/MEAformer/save/FBYG15K_rate_0.2_.pkl"
    cj_path = ckpt_path.replace('.pkl', '_cj.json')
    print(f"[main] ckpt_path: {ckpt_path}")

    model = load_model(KGs, args, ckpt_path, cj_path)

    # 复用 evaluate_Cj 的做法: 拿各模态独立 emb
    gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, _, _, _ = \
        model.joint_emb_generat(only_joint=False)

    embs = {'img': img_emb, 'att': att_emb, 'rel': rel_emb,
            'gph': gph_emb, 'name': name_emb, 'char': char_emb}

    print(f"\n[{a.data_choice}] 单模态独立对齐 (test pairs: {len(test_ill_np)})")
    print("-" * 80)
    print(f"{'modal':<8}{'H@1':<10}{'H@10':<10}{'MRR':<10}{'note':<30}")
    print("-" * 80)

    results = {}
    for m, emb in embs.items():
        if emb is None:
            print(f"{m:<8}{'--':<10}{'--':<10}{'--':<10}{'模态未加载 (wo_surf?)':<30}")
            results[m] = None
            continue
        emb_n = F.normalize(emb)
        # 不加 CSLS, 纯单模态 L2 距离 (与 evaluate_Cj 一致)
        dist = pairwise_distances(emb_n[test_left], emb_n[test_right])
        h1, h10, mrr = compute_h1_h10_mrr(dist)
        results[m] = {'h1': h1, 'h10': h10, 'mrr': mrr}
        print(f"{m:<8}{h1:<10.4f}{h10:<10.4f}{mrr:<10.4f}{'':<30}")

    print("-" * 80)
    print(f"[{a.data_choice}] 关键: gph 坏掉后, img/att/rel 中是否有 H@1 够高的可作替代证据")
    print("=" * 80)

    # 顺手打印 C_j (训练时学到的模态因果贡献度, 可与单模态 H@1 对照)
    if hasattr(model, 'causal_Cj'):
        print(f"[{a.data_choice}] C_j (训练学到的模态因果贡献度): "
              f"{ {k: round(v,4) for k,v in model.causal_Cj.items()} }")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_choice', required=True)
    ap.add_argument('--data_split', required=True)
    ap.add_argument('--data_rate', type=float, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--img_perturb', type=float, default=0.0,
                    help='视觉扰动比例; >0 时先扰 img 再测各单模态 (诊断用)')
    main(ap.parse_args())