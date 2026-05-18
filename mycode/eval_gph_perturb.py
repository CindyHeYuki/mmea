"""
eval_gph_perturb.py

§4.5.3 补充实验: 扰动 gph (图结构) 模态的鲁棒性对比。

与 eval_perturb.py (视觉扰动) 的唯一区别: 扰动的是 gph_emb 而非 images_list。
gph_emb 是 GAT 算出来的, 不是静态矩阵, 所以用 monkey-patch 注入 (注入点 A)。

用法 (像实验 2 一样, 手动改下方 ckpt_path, 跑三次):
  python eval_gph_perturb.py --data_choice FBDB15K --data_split norm --data_rate 0.2 \\
      --causal_alpha 0.275 --csc_alpha 0.25 --neighbor_alpha 0.65 \\
      --perturb_ratios 0.0,0.4,0.8 --seed 42 --gpu 0 \\
      --output_csv eval_output/gph_perturb_FBDB15K.csv
"""

import os
import os.path as osp
import sys
import argparse
import json
import csv
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


# ============================================================
# gph 扰动注入 (注入点 A: monkey-patch joint_emb_generat)
# ============================================================

def sample_perturb_entities(test_ill, p, seed=42):
    """单端扰动采样, 与视觉扰动完全一致。p = 受影响测试对齐对比例。"""
    if p == 0.0:
        return np.array([], dtype=np.int64)
    rng = np.random.RandomState(seed)
    test_ill = np.asarray(test_ill)
    N_test = test_ill.shape[0]
    side_choice = rng.randint(0, 2, size=N_test)
    chosen = np.where(side_choice == 0, test_ill[:, 0], test_ill[:, 1])
    n_perturb = int(round(p * N_test))
    if n_perturb == 0:
        return np.array([], dtype=np.int64)
    uniq = np.unique(chosen)
    n_perturb = min(n_perturb, len(uniq))
    idx = rng.choice(len(uniq), size=n_perturb, replace=False)
    return uniq[idx].astype(np.int64)


def patch_gph(model, perturb_ents, seed=42):
    """
    monkey-patch model.joint_emb_generat: only_joint=False 返回的 gph_emb
    中 perturb_ents 行替换为高斯噪声 (mean/std 取自其余实体 gph_emb 逐维统计)。
    返回 restore 函数。
    """
    if len(perturb_ents) == 0:
        return lambda: None

    original_fn = model.joint_emb_generat
    pe = torch.as_tensor(perturb_ents, dtype=torch.long)

    def patched(only_joint=True, epoch=0, total_epochs=1):
        out = original_fn(only_joint=only_joint, epoch=epoch, total_epochs=total_epochs)
        if only_joint:
            # joint 已混合 gph, 这条路径不干净 — 调用方应改用 only_joint=False
            return out
        gph_emb = out[0]
        device = gph_emb.device
        idx = pe.to(device)
        mask = torch.ones(gph_emb.shape[0], dtype=torch.bool, device=device)
        mask[idx] = False
        if mask.sum() == 0:
            mean, std = gph_emb.mean(0), gph_emb.std(0)
        else:
            mean, std = gph_emb[mask].mean(0), gph_emb[mask].std(0)
        std = torch.where(std < 1e-6, torch.full_like(std, 1e-6), std)
        g = torch.Generator(device='cpu').manual_seed(seed)
        noise = torch.normal(mean.cpu().expand(len(idx), -1),
                             std.cpu().expand(len(idx), -1),
                             generator=g).to(device).to(gph_emb.dtype)
        gph_new = gph_emb.clone()
        gph_new[idx] = noise
        return (gph_new,) + tuple(out[1:])

    patched.__name__ = 'patched'
    model.joint_emb_generat = patched
    return lambda: setattr(model, 'joint_emb_generat', original_fn)


@torch.no_grad()
def build_perturbed_joint(model):
    """走 only_joint=False 拿被扰动的 gph_emb, 重新调 fusion 层融合出 joint。"""
    gph, img, rel, att, name, char, _, _, _ = model.joint_emb_generat(only_joint=False)
    # fusion 期望顺序: [img, att, rel, gph, name, char]
    emb_list = [img, att, rel, gph, name, char]
    fusion = model.multimodal_encoder.fusion
    joint, _, wn = fusion(emb_list, causal_bias=None)
    return F.normalize(joint), wn


# ============================================================
# build_args (与实验 2 你本地调通版一致的三处硬编码)
# ============================================================

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

    defaults = {
        'use_causal_bias': 0, 'use_csc': 0, 'use_neighbor': 0,
        'use_sample_schedule': 1, 'causal_lambda': 0.0, 'csc_lambda_0': 0.0,
        'neighbor_alpha': 0.0, 'tau_C': 1.0, 'epsilon_floor': 0.01,
        'csls_iter': 1, 'use_bidirectional_consistency': 0, 'ablate_modal': "",
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # ===== 与实验 2 跑通版完全一致的维度硬编码 =====
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
    # =============================================
    return args


# ============================================================
# load_model (ckpt_path 手动写死, 与实验 2 一致)
# ============================================================

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
            print(f"[load_model] loaded C_j: {cj}")
    else:
        print(f"[load_model] WARNING: C_j not found ({cj_path}), use defaults")
    return model


# ============================================================
# inference: baseline 和 ours (gph 已被 patch)
# ============================================================

@torch.no_grad()
def run_baseline(model, args, test_left, test_right, csls_k=3):
    """baseline: gph 扰动后重新融合 joint, 不开任何推理时模块。"""
    joint_emb, _ = build_perturbed_joint(model)
    distance = pairwise_distances(joint_emb[test_left], joint_emb[test_right])
    if args.csls:
        distance = 1 - csls_sim(1 - distance, csls_k)
    return compute_metrics(distance)


@torch.no_grad()
def run_ours(model, args, test_left, test_right,
             causal_alpha, csc_alpha, neighbor_alpha, csls_k=3):
    """ours: 用被扰动的 gph_emb 走完整推理时融合。"""
    # final_emb 用被扰动 gph 重新融合的 joint
    final_emb, weight_norm = build_perturbed_joint(model)
    gph, img, rel, att, name, char, _, _, _ = model.joint_emb_generat(only_joint=False)

    distance = pairwise_distances(final_emb[test_left], final_emb[test_right])

    # neighbor
    if neighbor_alpha > 0:
        adj = model.adj
        nbr = F.normalize(torch.sparse.mm(adj, final_emb))
        nbr_d = pairwise_distances(nbr[test_left], nbr[test_right])
        distance = (1 - neighbor_alpha) * distance + neighbor_alpha * nbr_d

    # causal
    if causal_alpha > 0:
        import math
        cj = model.causal_Cj
        md, mw = [], []
        for mn, me in {'img': img, 'att': att, 'rel': rel,
                       'gph': gph, 'name': name, 'char': char}.items():
            if me is None:
                continue
            c = max(cj.get(mn, 0.0), 1e-4)
            men = F.normalize(me)
            md.append(pairwise_distances(men[test_left], men[test_right]))
            mw.append(c)
        if md:
            tc = getattr(args, 'tau_C', 1.0)
            eps = getattr(args, 'epsilon_floor', 0.01)
            ew = [math.exp(w / tc) for w in mw]
            s = sum(ew)
            mw = [e / s for e in ew]
            M = len(mw)
            mw = [(1 - eps) * w + eps / M for w in mw]
            cd = sum(w * d for w, d in zip(mw, md))
            distance = (1 - causal_alpha) * distance + causal_alpha * cd

    # csc
    if csc_alpha > 0:
        ml = [e for e in [img, att, rel, gph, name, char] if e is not None]
        if len(ml) >= 2:
            st = torch.stack([F.normalize(e, dim=-1) for e in ml], dim=1)
            M = st.shape[1]
            uw = torch.ones(st.shape[0], M, device=st.device) / M
            cf = F.normalize(torch.sum(uw.unsqueeze(2) * st, dim=1))
            cfd = pairwise_distances(cf[test_left], cf[test_right])
            distance = (1 - csc_alpha) * distance + csc_alpha * cfd

    if args.csls:
        it = getattr(args, 'csls_iter', 1)
        for _ in range(it):
            distance = 1 - csls_sim(1 - distance, csls_k)

    return compute_metrics(distance)


def compute_metrics(distance):
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
    return {'h1': float(acc[0]), 'h10': float(acc[1]), 'mrr': float(mrr)}


# ============================================================
# main
# ============================================================

def main(a):
    set_seed(a.random_seed)
    args = build_args(a.data_choice, a.data_split, a.data_rate, a.gpu)

    print("=" * 70)
    print(f"Loading {a.data_choice} (split={a.data_split}, rate={a.data_rate})")
    print("=" * 70)
    KGs, non_train, train_ill, test_ill, eval_ill, test_ill_ = load_data(logger, args)
    test_ill_np = np.asarray(test_ill.data if hasattr(test_ill, 'data') else test_ill)
    test_left = torch.LongTensor(test_ill_np[:, 0]).to(args.device)
    test_right = torch.LongTensor(test_ill_np[:, 1]).to(args.device)
    print(f"[main] test_ill shape: {test_ill_np.shape}")

    # ============================================
    # !!! 像实验 2 一样, 手动改这里的 ckpt_path !!!
    # FBDB15K:  /data0/hwx/mmea_copy/data/mmkg/MEAformer/save/FBDB15K_0.2_.pkl
    # FBYG15K:  /data0/hwx/mmea_copy/data/mmkg/MEAformer/save/FBYG15K_rate_0.2_.pkl
    # DBP15K:   /data0/hwx/mmea_copy/data/mmkg/MEAformer/save/v2_dbp_zh_wo_surf_seed1_.pkl
    # ============================================
    ckpt_path = "/data0/hwx/mmea_copy/data/mmkg/MEAformer/save/v2_dbp_zh_wo_surf_seed1_.pkl"
    cj_path = ckpt_path.replace('.pkl', '_cj.json')
    print(f"[main] ckpt_path: {ckpt_path}")

    args.csls_iter = a.csls_iter


    model = load_model(KGs, args, ckpt_path, cj_path)

    perturb_ratios = [float(x) for x in a.perturb_ratios.split(',')]
    results = []

    for p in perturb_ratios:
        print("\n" + "=" * 70)
        print(f"gph perturbation ratio p = {p}")
        print("=" * 70)

        perturb_ents = sample_perturb_entities(test_ill_np, p, seed=a.seed)
        print(f"[main] # entities perturbed: {len(perturb_ents)} "
              f"(test pairs: {len(test_ill_np)}, ratio: {len(perturb_ents)/max(len(test_ill_np),1):.3f})")

        # patch gph
        restore = patch_gph(model, perturb_ents, seed=a.seed)
        try:
            # 诊断: 确认 gph 真被改 (只在第一个非零 p 打一次)
            if p > 0 and len([r for r in results if r['p'] > 0]) == 0:
                with torch.no_grad():
                    gp = model.joint_emb_generat(only_joint=False)[0]
                restore()
                with torch.no_grad():
                    go = model.joint_emb_generat(only_joint=False)[0]
                restore = patch_gph(model, perturb_ents, seed=a.seed)
                idx = torch.as_tensor(perturb_ents, dtype=torch.long, device=go.device)
                m = torch.ones(go.shape[0], dtype=torch.bool, device=go.device)
                m[idx] = False
                print(f"[Diagnose] max diff UNTOUCHED: {(go[m]-gp[m]).abs().max().item():.2e} (~0)")
                print(f"[Diagnose] mean diff PERTURBED: {(go[idx]-gp[idx]).abs().mean().item():.4f} (>0)")

            mb = run_baseline(model, args, test_left, test_right, csls_k=args.csls_k)
            mo = run_ours(model, args, test_left, test_right,
                          a.causal_alpha, a.csc_alpha, a.neighbor_alpha,
                          csls_k=args.csls_k)
        finally:
            restore()

        print(f"\n[result] p={p}:")
        print(f"  baseline:  H@1={mb['h1']:.4f}  H@10={mb['h10']:.4f}  MRR={mb['mrr']:.4f}")
        print(f"  ours:      H@1={mo['h1']:.4f}  H@10={mo['h10']:.4f}  MRR={mo['mrr']:.4f}")
        print(f"  ours - baseline ΔH@1 = {mo['h1']-mb['h1']:+.4f}")

        results.append({
            'dataset': a.data_choice, 'split': a.data_split, 'p': p,
            'baseline_h1': mb['h1'], 'baseline_h10': mb['h10'], 'baseline_mrr': mb['mrr'],
            'ours_h1': mo['h1'], 'ours_h10': mo['h10'], 'ours_mrr': mo['mrr'],
        })

    if a.output_csv:
        os.makedirs(osp.dirname(a.output_csv) or '.', exist_ok=True)
        with open(a.output_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"\n[main] saved to {a.output_csv}")

    # 总结表
    print("\n" + "=" * 80)
    print(f"Summary: {a.data_choice} gph perturbation")
    print(f"  causal_α={a.causal_alpha}, csc_α={a.csc_alpha}, neighbor_α={a.neighbor_alpha}")
    print("=" * 80)
    print(f"{'p':<8}{'base H@1':<12}{'ours H@1':<12}{'ΔH@1':<10}{'base drop':<12}{'ours drop':<12}")
    print("-" * 80)
    b0, o0 = results[0]['baseline_h1'], results[0]['ours_h1']
    for r in results:
        bd = r['baseline_h1'] - b0 if r['p'] > 0 else 0.0
        od = r['ours_h1'] - o0 if r['p'] > 0 else 0.0
        print(f"{r['p']:<8}{r['baseline_h1']:<12.4f}{r['ours_h1']:<12.4f}"
              f"{r['ours_h1']-r['baseline_h1']:<+10.4f}{bd:<+12.4f}{od:<+12.4f}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_choice', required=True)
    ap.add_argument('--data_split', required=True)
    ap.add_argument('--data_rate', type=float, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--causal_alpha', type=float, required=True)
    ap.add_argument('--csc_alpha', type=float, required=True)
    ap.add_argument('--neighbor_alpha', type=float, required=True)
    ap.add_argument('--perturb_ratios', default='0.0,0.4,0.8')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--random_seed', type=int, default=42)
    ap.add_argument('--output_csv', default='')
    ap.add_argument('--csls_iter', type=int, default=10)
    main(ap.parse_args())