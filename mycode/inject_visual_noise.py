"""
inject_visual_noise.py

为实验 2 (模态先验偏倚的存在性) 准备视觉扰动数据。

接收 KGs["images_list"] (numpy array, shape [ENT_NUM, 4096]) 和测试集对齐对
test_ill (shape [N_test, 2])，按比例 p 单端扰动 (每对随机选一端)，用其它实体
图像的均值/方差作为高斯噪声参数。

返回:
    images_perturbed: 扰动后的 images_list (同 shape, 同 dtype)
    perturb_ents: 被扰动的实体 id 集合 (用于诊断)
"""

import numpy as np


def inject_visual_noise(images_list, test_ill, p, seed=42):
    """
    单端视觉扰动: 在测试对齐对中按比例 p 随机选择 N_test * p 对, 每对随机扰
    动其一端实体的图像特征 —— 用其它实体图像的均值/方差作为参数生成的高斯
    噪声替换。

    扰动比例 p 的语义为 "ratio of affected test alignment pairs": p=0.2 意味
    着 20% 的测试对齐对中有一端实体的图像信号变成无信息噪声 (即 baseline 看
    到的图像通道在这 20% 对上无判别力)。

    Args:
        images_list: np.ndarray or torch.Tensor, shape [ENT_NUM, IMG_DIM]
            原始图像特征矩阵 (通常 4096 维 VGG-16 fc7)
        test_ill: np.ndarray, shape [N_test, 2]
            测试集对齐对 (左端 KG-1 实体 id, 右端 KG-2 实体 id)
        p: float in [0, 1]
            扰动比例 — 受影响的测试对齐对比例 (单端扰动)
            数值上, 被扰动的实体数 = floor(p * N_test), 对应 p% 的对齐对
            一端被扰动。
        seed: int
            随机种子, 保证可复现

    Returns:
        images_perturbed: same shape and dtype as images_list
        perturb_ents: np.ndarray, shape [n_perturb], 被扰动的实体 id
    """
    # 兼容 torch.Tensor 输入
    is_torch = hasattr(images_list, 'numpy')
    if is_torch:
        images_np = images_list.detach().cpu().numpy().copy()
    else:
        images_np = np.asarray(images_list).copy()

    if p == 0.0:
        # p=0 直接返回原始 (确保 clean run 完全无扰动)
        empty = np.array([], dtype=np.int64)
        if is_torch:
            import torch
            return torch.from_numpy(images_np).to(images_list.device), empty
        return images_np, empty

    rng = np.random.RandomState(seed)
    test_ill = np.asarray(test_ill)
    N_test = test_ill.shape[0]

    # 1. 单端选择: 每对对齐对随机选一端 (0=左/KG-1, 1=右/KG-2)
    side_choice = rng.randint(0, 2, size=N_test)
    chosen_ents = np.where(side_choice == 0, test_ill[:, 0], test_ill[:, 1])

    # 2. 从 chosen_ents 里按比例 p 采样要扰动的 (不重复)
    n_perturb = int(round(p * N_test))
    if n_perturb == 0:
        empty = np.array([], dtype=np.int64)
        if is_torch:
            import torch
            return torch.from_numpy(images_np).to(images_list.device), empty
        return images_np, empty

    # 注意: chosen_ents 可能有重复 (一个实体可能在多对里出现), 先去重再采样
    unique_chosen = np.unique(chosen_ents)
    n_perturb = min(n_perturb, len(unique_chosen))
    perturb_idx = rng.choice(len(unique_chosen), size=n_perturb, replace=False)
    perturb_ents = unique_chosen[perturb_idx].astype(np.int64)

    # 3. 用其它实体的图像统计量做高斯噪声 (mean/std 沿实体维度算, 4096 维各自一组)
    other_mask = np.ones(len(images_np), dtype=bool)
    other_mask[perturb_ents] = False

    # 防御: 万一所有实体都要扰动 (不可能, 但 robust 一点)
    if other_mask.sum() == 0:
        # 用全局统计量
        mean = images_np.mean(axis=0)
        std = images_np.std(axis=0)
    else:
        mean = images_np[other_mask].mean(axis=0)
        std = images_np[other_mask].std(axis=0)

    # std 为 0 的维度 (理论上不会, 但 robust): 用 1e-6 兜底
    std = np.where(std < 1e-6, 1e-6, std)

    noise = rng.normal(loc=mean, scale=std,
                       size=(len(perturb_ents), images_np.shape[1])).astype(images_np.dtype)
    images_np[perturb_ents] = noise

    if is_torch:
        import torch
        return torch.from_numpy(images_np).to(images_list.device), perturb_ents

    return images_np, perturb_ents


def diagnose(images_orig, images_perturbed, perturb_ents):
    """
    打印扰动后的诊断信息, 跑实验前先验证扰动是否如预期。
    """
    if hasattr(images_orig, 'numpy'):
        images_orig = images_orig.detach().cpu().numpy()
    if hasattr(images_perturbed, 'numpy'):
        images_perturbed = images_perturbed.detach().cpu().numpy()

    print(f"[Diagnose] images shape: {images_orig.shape}, dtype: {images_orig.dtype}")
    print(f"[Diagnose] # entities perturbed: {len(perturb_ents)}")
    print(f"[Diagnose] # entities total:     {len(images_orig)}")
    print(f"[Diagnose] perturb ratio (entity-level): {len(perturb_ents)/len(images_orig):.4f}")

    if len(perturb_ents) == 0:
        print("[Diagnose] no entities perturbed (p=0?), skip stat comparison")
        return

    # 验证未扰动实体的图像完全没变
    untouched_mask = np.ones(len(images_orig), dtype=bool)
    untouched_mask[perturb_ents] = False
    diff_untouched = np.abs(images_orig[untouched_mask] - images_perturbed[untouched_mask]).max()
    print(f"[Diagnose] max abs diff on UNTOUCHED entities: {diff_untouched:.2e} (should be 0)")

    # 验证扰动实体的图像确实变了
    diff_perturbed = np.abs(images_orig[perturb_ents] - images_perturbed[perturb_ents]).mean()
    print(f"[Diagnose] mean abs diff on PERTURBED entities: {diff_perturbed:.4f}")

    # 比较扰动前后的统计量
    print(f"[Diagnose] orig images mean (over all): {images_orig.mean():.4f}, "
          f"std: {images_orig.std():.4f}")
    print(f"[Diagnose] perturbed images mean: {images_perturbed.mean():.4f}, "
          f"std: {images_perturbed.std():.4f}")
    print(f"[Diagnose] perturbed-entities row mean: "
          f"{images_perturbed[perturb_ents].mean():.4f}, "
          f"std: {images_perturbed[perturb_ents].std():.4f}")


if __name__ == "__main__":
    # 单元测试: 用随机数据走一次
    print("=" * 60)
    print("Unit test: inject_visual_noise()")
    print("=" * 60)
    ENT_NUM, IMG_DIM = 1000, 4096
    images = np.random.randn(ENT_NUM, IMG_DIM).astype(np.float32) * 0.5 + 1.0
    # 构造伪 test_ill: 500 对
    test_ill = np.column_stack([
        np.random.choice(ENT_NUM, 500, replace=False),
        np.random.choice(ENT_NUM, 500, replace=False)
    ])

    for p in [0.0, 0.1, 0.2, 0.3]:
        print(f"\n--- p = {p} ---")
        images_p, perturb_ents = inject_visual_noise(images, test_ill, p, seed=42)
        diagnose(images, images_p, perturb_ents)