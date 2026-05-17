"""
ours_inference.py

Ours 推理路径,供实验 3 调用。
逻辑完全等同于实验 2 eval_perturb.py 的 run_inference_both 里的 ours 分支。
"""
import math
import torch
import torch.nn.functional as F
from src.utils import pairwise_distances, csls_sim


@torch.no_grad()
def ours_inference(model, KGs, test_left, test_right, args,
                   causal_alpha, csc_alpha, neighbor_alpha,
                   csls_k=3):
    """
    Returns:
        distance: torch.Tensor [N_left, N_right] — ours 的距离矩阵
        final_emb: torch.Tensor [ENT_NUM, dim] — ours 路径上用到的联合表征
                   (注意: ours 在 distance 层做融合,所以这里 final_emb 仅是
                    joint_emb_generat() 的 F.normalize 输出,不含距离层融合)
    """
    # ====== 拿表征 ======
    final_emb, weight_norm = model.joint_emb_generat()
    final_emb = F.normalize(final_emb)

    gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, _, _, _ = \
        model.joint_emb_generat(only_joint=False)

    # ====== distance 起点: joint 距离 ======
    distance = pairwise_distances(final_emb[test_left], final_emb[test_right])

    # ====== neighbor 增强 (拓扑邻接) ======
    if neighbor_alpha > 0:
        adj = model.adj
        neighbor_emb = torch.sparse.mm(adj, final_emb)
        neighbor_emb = F.normalize(neighbor_emb)
        neighbor_distance = pairwise_distances(neighbor_emb[test_left],
                                                neighbor_emb[test_right])
        distance = (1 - neighbor_alpha) * distance + neighbor_alpha * neighbor_distance

    # ====== 因果证据聚合 (按 C_m 加权的逐模态距离) ======
    if causal_alpha > 0:
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
            distance = (1 - causal_alpha) * distance + causal_alpha * causal_distance

    # ====== 反事实一致性 (均匀加权 cf_joint) ======
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
            distance = (1 - csc_alpha) * distance + csc_alpha * cf_distance

    # ====== CSLS ======
    if getattr(args, 'csls', True):
        csls_iter = getattr(args, 'csls_iter', 1)
        for _ in range(csls_iter):
            distance = 1 - csls_sim(1 - distance, csls_k)

    return distance, final_emb