import types
import torch
import transformers
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
import pdb
import math
from .Tool_model import AutomaticWeightedLoss
from .MEAformer_tools import MultiModalEncoder
from .MEAformer_loss import CustomMultiLossLayer, icl_loss

from src.utils import pairwise_distances
import os.path as osp
import json


class MEAformer(nn.Module):
    def __init__(self, kgs, args):
        super().__init__()
        self.kgs = kgs
        self.args = args
        self.img_features = F.normalize(torch.FloatTensor(kgs["images_list"])).cuda()
        self.input_idx = kgs["input_idx"].cuda()
        self.adj = kgs["adj"].cuda()
        self.rel_features = torch.Tensor(kgs["rel_features"]).cuda()
        self.att_features = torch.Tensor(kgs["att_features"]).cuda()
        self.name_features = None
        self.char_features = None
        if kgs["name_features"] is not None:
            self.name_features = kgs["name_features"].cuda()
            self.char_features = kgs["char_features"].cuda()

        img_dim = self._get_img_dim(kgs)

        char_dim = kgs["char_features"].shape[1] if self.char_features is not None else 100

        self.multimodal_encoder = MultiModalEncoder(args=self.args,
                                                    ent_num=kgs["ent_num"],
                                                    img_feature_dim=img_dim,
                                                    char_feature_dim=char_dim,
                                                    use_project_head=self.args.use_project_head,
                                                    attr_input_dim=kgs["att_features"].shape[1])

        self.multi_loss_layer = CustomMultiLossLayer(loss_num=6)  # 6
        self.criterion_cl = icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2)
        self.criterion_cl_joint = icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2, replay=self.args.replay, neg_cross_kg=self.args.neg_cross_kg)

        tmp = -1 * torch.ones(self.input_idx.shape[0], dtype=torch.int64).cuda()
        self.replay_matrix = torch.stack([self.input_idx, tmp], dim=1).cuda()
        self.replay_ready = 0
        self.idx_one = torch.ones(self.args.batch_size, dtype=torch.int64).cuda()
        self.idx_double = torch.cat([self.idx_one, self.idx_one]).cuda()
        self.last_num = 1000000000000
        # self.idx_one = np.ones(self.args.batch_size, dtype=np.int64)
        
        # ====== 新增：因果信号状态维护 ======
        # 严格对应 Encoder 里的顺序: [img, att, rel, gph, name, char]
        self.modal_names = ['img', 'att', 'rel', 'gph', 'name', 'char']
        
        # D_j: 瞬时因果效应 (梯度范数滑动平均)，初始化为 0
        self.causal_Dj = {m: 0.0 for m in self.modal_names}
        
        # C_j: 固有因果置信度，初始化为一个中等偏上的先验值 (如 0.5)
        self.causal_Cj = {m: 0.5 for m in self.modal_names}
        # ====================================

        # 假设实体的总数是 ent_num，模态数是 modal_num (例如 6)
        # modal_num = len(self.modal_names)
        # # 初始化均匀权重，并使用 register_buffer 确保它不参与梯度计算，但能随模型保存
        # self.register_buffer('alpha_prev', torch.ones(args.ent_num, modal_num) / modal_num)


    

    # def compute_csc_loss(self, input_idx, embs_list, joint_emb_factual, alpha_t, epoch, total_epochs):
    #     # embs_list: 融合前的模态表示列表 X 
    #     # joint_emb_factual: 当前步计算出的事实状态表征 z_v_i [cite: 80, 82]
    #     # alpha_t: 当前步的融合权重 
        
    #     batch_size = joint_emb_factual.size(0)
    #     modal_num = len(embs_list)
        
    #     # 堆叠各个模态特征: shape [batch_size, modal_num, hidden_size]
    #     X = torch.stack(embs_list, dim=1) 
        
    #     # --- (2) 正向反事实干预：时序平稳 z_prev_v_i ---
    #     # 获取历史权重，并使用 detach() 实施因果干预，阻断当前梯度对历史权重的更新 [cite: 87, 88]
    #     alpha_prev_batch = self.alpha_prev[input_idx].detach() 
    #     # 计算历史权重下的加权融合表征
    #     z_prev = torch.sum(alpha_prev_batch.unsqueeze(-1) * X, dim=1) 

    #     # --- (3) 负向反事实干预：因果盲目 z_uni_v_i ---
    #     # 强制进行均匀融合 [cite: 90, 93]
    #     z_uni = torch.mean(X, dim=1) 

    #     # --- (4) 计算基于间隔的反事实对比损失 ---
    #     # 采用余弦距离: distance = 1 - cosine_similarity [cite: 95, 96, 97]
    #     d_pos = 1.0 - F.cosine_similarity(joint_emb_factual, z_prev, dim=-1)
    #     d_neg = 1.0 - F.cosine_similarity(joint_emb_factual, z_uni, dim=-1)
        
    #     # max(0, d_pos - d_neg + gamma) [cite: 98, 99]
    #     margin = self.args.csc_gamma
    #     loss_csc = torch.clamp(d_pos - d_neg + margin, min=0.0).mean()

    #     # --- (5) 计算时间衰减系数 ---
    #     # lambda_t = lambda_0 * exp(-eta * t / T) [cite: 101, 102]
    #     lambda_t = self.args.csc_lambda_0 * math.exp(-self.args.csc_eta * (epoch / total_epochs))

    #     # --- 更新历史权重 ---
    #     # 使用当前步的权重更新 buffer（动量更新或直接替换均可，此处为直接替换以吻合论文"上一训练步骤"的设定）
    #     self.alpha_prev[input_idx] = alpha_t.detach()

    #     return lambda_t * loss_csc


    def forward(self, input_batch, epoch=0, total_epochs=1):
        # 生成所有实体和隐藏层嵌入 (必须在最前面，为了后面统一获取 Device)
        gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states, weight_norm = self.joint_emb_generat(only_joint=False, epoch=epoch, total_epochs=total_epochs)
        # gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states = self.joint_emb_generat(only_joint=False, epoch=epoch, total_epochs=total_epochs)
        gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, joint_emb_hid = self.generate_hidden_emb(hidden_states)

        # 动态获取所在设备，避免硬编码 .cuda() 或未定义的 self.device
        device = joint_emb.device 

        # ====== 新增：初始化软权重 ======
        sample_weights = None 

        # 统一处理输入格式
        if isinstance(input_batch, dict):  # 训练数据（包含难度信息）
            batch_tensor = torch.stack([
                torch.tensor(input_batch['ent1'], dtype=torch.int64, device=device),
                torch.tensor(input_batch['ent2'], dtype=torch.int64, device=device)
            ], dim=1)
            difficulties = torch.tensor(input_batch['difficulties'], device=device)
            
            # ====== 核心创新：软加权 (Soft Weighting) 调度 ======
            if self.args.use_sample_schedule == 1 and self.training:
                # 1.计算当前 Epoch 的平滑阈值
                threshold = 1 - math.exp(-self.args.k * epoch / total_epochs)
                # 2. 计算标准的软权重
                sample_weights = torch.exp(- (difficulties - threshold).clamp(min=0))
                
                # 3. 【新增】设定绝对噪声天花板 (Noise Ceiling)
                # DBP15K 数据集中，难度极高(>0.85)的通常是纯噪声或严重缺失特征的实体
                noise_ceiling = 0.85 
                
                # 凡是难度大于天花板的样本，直接给它套上一个强惩罚（或者直接设为0）
                # 这里我们使用硬截断（Hard Mask），超过 0.85 的样本权重直接归零，防止后期毒害模型
                noise_mask = (difficulties <= noise_ceiling).float()
                sample_weights = sample_weights * noise_mask
                
                # 【绝妙公式】：(difficulties - threshold).clamp(min=0)
                # 难度比阈值低，结果为0，exp(0)=1，权重为1，全盘吸收！
                # 难度比阈值高，差值越大，exp(-差值) 越接近 0，实现软惩罚！
                # sample_weights = torch.exp(- (difficulties - threshold).clamp(min=0))
            # ===================================================
        else:  # 测试数据（无难度信息）
            batch_tensor = torch.tensor(input_batch, dtype=torch.int64, device=device)
            difficulties = None

        # 计算对比损失
        if self.args.replay:
            all_ent_batch = torch.cat([batch_tensor[:, 0], batch_tensor[:, 1]])
            
            if not self.replay_ready:
                # 【修改】下发 sample_weights
                loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch_tensor, sample_weights=sample_weights)
            else:
                neg_l = self.replay_matrix[batch_tensor[:, 0], self.idx_one[:batch_tensor.shape[0]]]
                neg_r = self.replay_matrix[batch_tensor[:, 1], self.idx_one[:batch_tensor.shape[0]]]
                
                neg_l_set = set(neg_l.tolist())
                neg_r_set = set(neg_r.tolist())
                all_ent_set = set(all_ent_batch.tolist())
                
                neg_l_list = list(neg_l_set - all_ent_set)
                neg_r_list = list(neg_r_set - all_ent_set)
                
                neg_l_ipt = torch.tensor(neg_l_list, dtype=torch.int64, device=device)
                neg_r_ipt = torch.tensor(neg_r_list, dtype=torch.int64, device=device)
                
                # 【修改】下发 sample_weights
                loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch_tensor, neg_l_ipt, neg_r_ipt, sample_weights=sample_weights)
            
            index = (
                all_ent_batch,
                self.idx_double[:batch_tensor.shape[0] * 2],
            )
            new_value = torch.cat([l_neg, r_neg]).to(device)
            self.replay_matrix = self.replay_matrix.index_put(index, new_value)
            
            if self.replay_ready == 0:
                num = torch.sum(self.replay_matrix < 0)
                if num == self.last_num:
                    self.replay_ready = 1
                    print("-----------------------------------------")
                    print("begin replay!")
                    print("-----------------------------------------")
                else:
                    self.last_num = num
        else:
            # 【修改】下发 sample_weights
            loss_joi = self.criterion_cl_joint(joint_emb, batch_tensor, sample_weights=sample_weights)
        
        # 计算内部视图损失与输出视图损失
        # 【修改】完全丢弃原来的 difficulties 判断，统一传入计算好的 sample_weights
        in_loss = self.inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, batch_tensor, sample_weights=sample_weights)
        out_loss = self.inner_view_loss(gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, batch_tensor, sample_weights=sample_weights)
        

        # 总损失
        loss_all = loss_joi + in_loss + out_loss

        # # 统一处理输入格式
        # if isinstance(input_batch, dict):  # 训练数据（包含难度信息）
        #     batch_tensor = torch.stack([
        #         torch.tensor(input_batch['ent1'], dtype=torch.int64, device=device),
        #         torch.tensor(input_batch['ent2'], dtype=torch.int64, device=device)
        #     ], dim=1)
        #     difficulties = torch.tensor(input_batch['difficulties'], device=device)
        # else:  # 测试数据（无难度信息）
        #     batch_tensor = torch.tensor(input_batch, dtype=torch.int64, device=device)
        #     difficulties = None

        # 计算对比损失
        # if self.args.replay:
        #     all_ent_batch = torch.cat([batch_tensor[:, 0], batch_tensor[:, 1]])
            
        #     if not self.replay_ready:
        #         # 【修复】传完整的 joint_emb，而不是切片后的 ent_emb
        #         loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch_tensor)
        #     else:
        #         neg_l = self.replay_matrix[batch_tensor[:, 0], self.idx_one[:batch_tensor.shape[0]]]
        #         neg_r = self.replay_matrix[batch_tensor[:, 1], self.idx_one[:batch_tensor.shape[0]]]
                
        #         neg_l_set = set(neg_l.tolist())
        #         neg_r_set = set(neg_r.tolist())
        #         all_ent_set = set(all_ent_batch.tolist())
                
        #         neg_l_list = list(neg_l_set - all_ent_set)
        #         neg_r_list = list(neg_r_set - all_ent_set)
                
        #         neg_l_ipt = torch.tensor(neg_l_list, dtype=torch.int64, device=device)
        #         neg_r_ipt = torch.tensor(neg_r_list, dtype=torch.int64, device=device)
                
        #         # 【修复】同理，恢复原有的参数签名
        #         loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch_tensor, neg_l_ipt, neg_r_ipt)
            
        #     index = (
        #         all_ent_batch,
        #         self.idx_double[:batch_tensor.shape[0] * 2],
        #     )
        #     new_value = torch.cat([l_neg, r_neg]).to(device)
        #     self.replay_matrix = self.replay_matrix.index_put(index, new_value)
            
        #     if self.replay_ready == 0:
        #         num = torch.sum(self.replay_matrix < 0)
        #         if num == self.last_num:
        #             self.replay_ready = 1
        #             print("-----------------------------------------")
        #             print("begin replay!")
        #             print("-----------------------------------------")
        #         else:
        #             self.last_num = num
        # else:
        #     # 【修复】恢复原有传参
        #     loss_joi = self.criterion_cl_joint(joint_emb, batch_tensor)
        
        # # 计算内部视图损失与输出视图损失
        # # 【修复】安全传入 difficulties：只有在它是字典输入时，才传入 difficulties 参数，防止测试集报错
        # if difficulties is not None:
        #     in_loss = self.inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, batch_tensor, difficulties=difficulties)
        #     out_loss = self.inner_view_loss(gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, batch_tensor, difficulties=difficulties)
        # else:
        #     in_loss = self.inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, batch_tensor)
        #     out_loss = self.inner_view_loss(gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, batch_tensor)
        

        # # 总损失
        # loss_all = loss_joi + in_loss + out_loss
        loss_dic = {"joint_Intra_modal": loss_joi.item(), "Intra_modal": in_loss.item()}
        output = {"loss_dic": loss_dic, "emb": joint_emb}

        # if self.args.use_csc and self.training:
        #     # 你需要透传 input_batch_idx (实体的绝对索引)，用于去 buffer 里拿历史权重
        #     loss_csc = self.compute_csc_loss(input_idx, embs_list, joint_emb, weight_norm, epoch, total_epochs)
        #     loss_all = loss_all + loss_csc
        #     # 记录到字典输出给 Tensorboard
        #     output["loss_dic"]["CSC_Loss"] = loss_csc.item() 
        # ====== 新增：构造 CSC 模块需要的输入 ======
        if self.args.use_csc and self.training:
            # 1. 打包 embs_list (注意顺序必须和 Encoder 里一致)
            embs_list = [img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb]
            
            # 2. 获取所有实体的索引
            input_idx = self.input_idx
            
            # 3. 计算反事实约束 Loss (传入打包好的变量)
            loss_csc = self.compute_csc_loss(input_idx, embs_list, weight_norm, epoch, total_epochs)
            
            # 4. 叠加 Loss 并记录
            loss_all = loss_all + loss_csc
            output["loss_dic"]["CSC_Loss"] = loss_csc.item()
        # ==========================================

        
        # # 准备输出
        # loss_dic = {"joint_Intra_modal": loss_joi.item(), "Intra_modal": in_loss.item()}
        # output = {"loss_dic": loss_dic, "emb": joint_emb}

        # ====== 新增：把因果参数传给主循环监控 ======
        if self.args.use_causal_bias and hasattr(self, 'current_causal_bias'):
            output["causal_bias"] = self.current_causal_bias
            output["causal_Cj"] = self.causal_Cj
        # ==========================================

        # ====== 修复：将平均模态融合权重传给画图系统 ======
        if weight_norm is not None:
            # weight_norm 的形状可能是 [ent_num, modal_num] 
            if weight_norm.dim() == 3:
                avg_weight = weight_norm.mean(dim=[0, 1]) 
            else:
                # 对所有实体取平均，得到当前 batch / 全局的模态平均权重
                avg_weight = weight_norm.mean(dim=0) 
            
            # 转成 python 的 list 传出去
            output["weight"] = avg_weight.detach().cpu().tolist()
        # =================================================
        
        return loss_all, output



        # def forward(self, input_batch):
        #     # 统一处理输入格式
        #     if isinstance(input_batch, dict):  # 训练数据（包含难度信息）
        #         ent_pairs = torch.stack([
        #             torch.tensor(input_batch['ent1'], device=self.device),
        #             torch.tensor(input_batch['ent2'], device=self.device)
        #         ], dim=1)
        #         difficulties = torch.tensor(input_batch['difficulties'], device=self.device)
        #     else:  # 测试数据（无难度信息）
        #         ent_pairs = torch.tensor(input_batch, device=self.device)
        #         difficulties = None
            
        #     # 生成所有实体嵌入
        #     gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states = self.joint_emb_generat(only_joint=False)
            
        #     # 提取当前批次的实体嵌入
        #     ent1_emb = joint_emb[ent_pairs[:, 0]]
        #     ent2_emb = joint_emb[ent_pairs[:, 1]]
            
        #     # 生成隐藏层嵌入
        #     gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, joint_emb_hid = self.generate_hidden_emb(hidden_states)
            
        #     # 计算对比损失（考虑难度权重）
        #     if self.args.replay:
        #         batch_tensor = ent_pairs.clone().detach()
        #         all_ent_batch = torch.cat([batch_tensor[:, 0], batch_tensor[:, 1]])
                
        #         if not self.replay_ready:
        #             loss_joi, l_neg, r_neg = self.criterion_cl_joint(ent1_emb, ent2_emb, batch_tensor)
        #         else:
        #             neg_l = self.replay_matrix[batch_tensor[:, 0], self.idx_one[:batch_tensor.shape[0]]]
        #             neg_r = self.replay_matrix[batch_tensor[:, 1], self.idx_one[:batch_tensor.shape[0]]]
                    
        #             neg_l_set = set(neg_l.tolist())
        #             neg_r_set = set(neg_r.tolist())
        #             all_ent_set = set(all_ent_batch.tolist())
                    
        #             neg_l_list = list(neg_l_set - all_ent_set)
        #             neg_r_list = list(neg_r_set - all_ent_set)
                    
        #             neg_l_ipt = torch.tensor(neg_l_list, dtype=torch.int64).cuda()
        #             neg_r_ipt = torch.tensor(neg_r_list, dtype=torch.int64).cuda()
                    
        #             loss_joi, l_neg, r_neg = self.criterion_cl_joint(
        #                 ent1_emb, ent2_emb, batch_tensor, 
        #                 neg_l_ipt, neg_r_ipt
        #             )
                
        #         index = (
        #             all_ent_batch,
        #             self.idx_double[:batch_tensor.shape[0] * 2],
        #         )
        #         new_value = torch.cat([l_neg, r_neg]).cuda()
        #         self.replay_matrix = self.replay_matrix.index_put(index, new_value)
                
        #         if self.replay_ready == 0:
        #             num = torch.sum(self.replay_matrix < 0)
        #             if num == self.last_num:
        #                 self.replay_ready = 1
        #                 print("-----------------------------------------")
        #                 print("begin replay!")
        #                 print("-----------------------------------------")
        #             else:
        #                 self.last_num = num
        #     else:
        #         loss_joi = self.criterion_cl_joint(ent1_emb, ent2_emb, ent_pairs)
            
        #     # 计算内部视图损失（考虑难度权重）
        #     in_loss = self.inner_view_loss(
        #         gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, 
        #         ent_pairs, difficulties
        #     )
            
        #     # 计算输出视图损失（考虑难度权重）
        #     out_loss = self.inner_view_loss(
        #         gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, 
        #         name_emb_hid, char_emb_hid, ent_pairs, difficulties
        #     )
            
        #     # 总损失
        #     loss_all = loss_joi + in_loss + out_loss
            
        #     # 准备输出
        #     loss_dic = {
        #         "joint_Intra_modal": loss_joi.item(), 
        #         "Intra_modal": in_loss.item()
        #     }
        #     output = {
        #         "loss_dic": loss_dic, 
        #         "emb": joint_emb
        #     }
            
        #     return loss_all, output

    # def forward(self, batch):
    #     gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states = self.joint_emb_generat(only_joint=False)
    #     gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, joint_emb_hid = self.generate_hidden_emb(hidden_states)
    #     if self.args.replay:
    #         batch = torch.tensor(batch, dtype=torch.int64).cuda()
    #         all_ent_batch = torch.cat([batch[:, 0], batch[:, 1]])
    #         if not self.replay_ready:
    #             loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch)
    #         else:
    #             neg_l = self.replay_matrix[batch[:, 0], self.idx_one[:batch.shape[0]]]
    #             neg_r = self.replay_matrix[batch[:, 1], self.idx_one[:batch.shape[0]]]
    #             neg_l_set = set(neg_l.tolist())
    #             neg_r_set = set(neg_r.tolist())
    #             all_ent_set = set(all_ent_batch.tolist())
    #             neg_l_list = list(neg_l_set - all_ent_set)
    #             neg_r_list = list(neg_r_set - all_ent_set)
    #             neg_l_ipt = torch.tensor(neg_l_list, dtype=torch.int64).cuda()
    #             neg_r_ipt = torch.tensor(neg_r_list, dtype=torch.int64).cuda()
    #             loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch, neg_l_ipt, neg_r_ipt)

    #         index = (
    #             all_ent_batch,
    #             self.idx_double[:batch.shape[0] * 2],
    #         )
    #         new_value = torch.cat([l_neg, r_neg]).cuda()

    #         self.replay_matrix = self.replay_matrix.index_put(index, new_value)
    #         if self.replay_ready == 0:
    #             num = torch.sum(self.replay_matrix < 0)
    #             if num == self.last_num:
    #                 self.replay_ready = 1
    #                 print("-----------------------------------------")
    #                 print("begin replay!")
    #                 print("-----------------------------------------")
    #             else:
    #                 self.last_num = num
    #     else:
    #         loss_joi = self.criterion_cl_joint(joint_emb, batch)

    #     in_loss = self.inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, batch)
    #     out_loss = self.inner_view_loss(gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, batch)

    #     loss_all = loss_joi + in_loss + out_loss

    #     loss_dic = {"joint_Intra_modal": loss_joi.item(), "Intra_modal": in_loss.item()}
    #     output = {"loss_dic": loss_dic, "emb": joint_emb}
    #     return loss_all, output

    def generate_hidden_emb(self, hidden):
        gph_emb = F.normalize(hidden[:, 0, :].squeeze(1))
        rel_emb = F.normalize(hidden[:, 1, :].squeeze(1))
        att_emb = F.normalize(hidden[:, 2, :].squeeze(1))
        img_emb = F.normalize(hidden[:, 3, :].squeeze(1))
        if hidden.shape[1] >= 6:
            name_emb = F.normalize(hidden[:, 4, :].squeeze(1))
            char_emb = F.normalize(hidden[:, 5, :].squeeze(1))
            joint_emb = torch.cat([gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb], dim=1)
        else:
            name_emb, char_emb = None, None
            loss_name, loss_char = None, None
            joint_emb = torch.cat([gph_emb, rel_emb, att_emb, img_emb], dim=1)

        return gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, joint_emb
    
    def inner_view_loss(self, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill, sample_weights=None):
        # 【修改】接收 sample_weights，打包成字典透传给底层的 loss 函数
        kwargs = {'sample_weights': sample_weights} if sample_weights is not None else {}
        
        loss_GCN = self.criterion_cl(gph_emb, train_ill, **kwargs) if gph_emb is not None else 0
        loss_rel = self.criterion_cl(rel_emb, train_ill, **kwargs) if rel_emb is not None else 0
        loss_att = self.criterion_cl(att_emb, train_ill, **kwargs) if att_emb is not None else 0
        loss_img = self.criterion_cl(img_emb, train_ill, **kwargs) if img_emb is not None else 0
        loss_name = self.criterion_cl(name_emb, train_ill, **kwargs) if name_emb is not None else 0
        loss_char = self.criterion_cl(char_emb, train_ill, **kwargs) if char_emb is not None else 0

        total_loss = self.multi_loss_layer([loss_GCN, loss_rel, loss_att, loss_img, loss_name, loss_char])
        return total_loss
        # # 如果传入了 difficulties，就打包成参数字典传给底层的 loss 函数
        # # ====== 新增：拦截 difficulties，防止传给底层报错 ======
        # difficulties = kwargs.pop('difficulties', None)
        # # =======================================================
        # # kwargs = {'difficulties': difficulties} if difficulties is not None else {}
        
        # loss_GCN = self.criterion_cl(gph_emb, train_ill, **kwargs) if gph_emb is not None else 0
        # loss_rel = self.criterion_cl(rel_emb, train_ill, **kwargs) if rel_emb is not None else 0
        # loss_att = self.criterion_cl(att_emb, train_ill, **kwargs) if att_emb is not None else 0
        # loss_img = self.criterion_cl(img_emb, train_ill, **kwargs) if img_emb is not None else 0
        # loss_name = self.criterion_cl(name_emb, train_ill, **kwargs) if name_emb is not None else 0
        # loss_char = self.criterion_cl(char_emb, train_ill, **kwargs) if char_emb is not None else 0

        # total_loss = self.multi_loss_layer([loss_GCN, loss_rel, loss_att, loss_img, loss_name, loss_char])
        # return total_loss

    # def inner_view_loss(self, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill):
    #     # pdb.set_trace()
    #     loss_GCN = self.criterion_cl(gph_emb, train_ill) if gph_emb is not None else 0
    #     loss_rel = self.criterion_cl(rel_emb, train_ill) if rel_emb is not None else 0
    #     loss_att = self.criterion_cl(att_emb, train_ill) if att_emb is not None else 0
    #     loss_img = self.criterion_cl(img_emb, train_ill) if img_emb is not None else 0
    #     loss_name = self.criterion_cl(name_emb, train_ill) if name_emb is not None else 0
    #     loss_char = self.criterion_cl(char_emb, train_ill) if char_emb is not None else 0

    #     total_loss = self.multi_loss_layer([loss_GCN, loss_rel, loss_att, loss_img, loss_name, loss_char])
    #     return total_loss

    # --------- necessary ---------------

    def joint_emb_generat(self, only_joint=True, epoch=0, total_epochs=1):
        # 计算因果偏置
        causal_bias = self._compute_causal_bias(epoch, total_epochs)
        gph_emb, img_emb, rel_emb, att_emb, \
            name_emb, char_emb, joint_emb, hidden_states, weight_norm = self.multimodal_encoder(
                self.input_idx, self.adj, self.img_features, self.rel_features, 
                self.att_features, self.name_features, self.char_features,
                causal_bias=causal_bias # 传入偏置
            )
        # 绑定梯度捕获 (只在重新生成嵌入时挂载 Hook)
        embs_dict = {'img': img_emb, 'att': att_emb, 'rel': rel_emb, 
                     'gph': gph_emb, 'name': name_emb, 'char': char_emb}
        self._register_grad_hooks(embs_dict)
        
        # gph_emb, img_emb, rel_emb, att_emb, \
        #     name_emb, char_emb, joint_emb, hidden_states, weight_norm = self.multimodal_encoder(self.input_idx,
        #                                                                                         self.adj,
        #                                                                                         self.img_features,
        #                                                                                         self.rel_features,
        #                                                                                         self.att_features,
        #                                                                                         self.name_features,
        #                                                                                         self.char_features)
        if only_joint:
            return joint_emb, weight_norm
        else:
            return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states, weight_norm

    # --------- share ---------------

    def _get_img_dim(self, kgs):
        if isinstance(kgs["images_list"], list):
            img_dim = kgs["images_list"][0].shape[1]
        elif isinstance(kgs["images_list"], np.ndarray) or torch.is_tensor(kgs["images_list"]):
            img_dim = kgs["images_list"].shape[1]
        return img_dim

    def Iter_new_links(self, epoch, left_non_train, final_emb, right_non_train, new_links=[]):
        if len(left_non_train) == 0 or len(right_non_train) == 0:
            return new_links
        distance_list = []
        for i in np.arange(0, len(left_non_train), 1000):
            d = pairwise_distances(final_emb[left_non_train[i:i + 1000]], final_emb[right_non_train])
            distance_list.append(d)
        distance = torch.cat(distance_list, dim=0)
        preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
        preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
        del distance_list, distance, final_emb
        if (epoch + 1) % (self.args.semi_learn_step * 5) == self.args.semi_learn_step:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if preds_r[p] == i]
        else:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if (preds_r[p] == i) and ((left_non_train[i], right_non_train[p]) in new_links)]

        return new_links

    def data_refresh(self, logger, train_ill, test_ill_, left_non_train, right_non_train, new_links=[]):
        if len(new_links) != 0 and (len(left_non_train) != 0 and len(right_non_train) != 0):
            new_links_select = new_links
            train_ill = np.vstack((train_ill, np.array(new_links_select)))
            num_true = len([nl for nl in new_links_select if nl in test_ill_])
            # remove from left/right_non_train
            for nl in new_links_select:
                left_non_train.remove(nl[0])
                right_non_train.remove(nl[1])

            if self.args.rank == 0:
                logger.info(f"#new_links_select:{len(new_links_select)}")
                logger.info(f"train_ill.shape:{train_ill.shape}")
                logger.info(f"#true_links: {num_true}")
                logger.info(f"true link ratio: {(100 * num_true / len(new_links_select)):.1f}%")
                logger.info(f"#entity not in train set: {len(left_non_train)} (left) {len(right_non_train)} (right)")

            new_links = []
        else:
            logger.info("len(new_links) is 0")

        return left_non_train, right_non_train, train_ill, new_links
    
    
    def _compute_causal_bias(self, epoch, total_epochs):
        if not self.args.use_causal_bias or not self.training:
            return None
            
        # (1) 进度信号 α (公式 9)
        alpha = epoch / total_epochs
        
        # 难度惩罚与低置信度惩罚权重函数 (公式 17 的动态权重)
        w_D = 1.0 - alpha  # 随训练递减，早期抑制瞬时剧烈变化
        w_C = alpha        # 随训练递增，晚期抑制低置信度模态

        # ====== 核心修复：对 D_j 进行相对归一化 ======
        raw_Dj = torch.tensor([self.causal_Dj[m] for m in self.modal_names], dtype=torch.float32, device=self.args.device)
        Dj_sum = raw_Dj.sum() + 1e-8 
        norm_Dj = raw_Dj / Dj_sum  # 归一化到 [0, 1]
        # ============================================
        
        biases = []
        for idx, m in enumerate(self.modal_names):
            # (2) 计算偏置 Bias_j (公式 16 对应的惩罚逻辑)
            # Bias 为负值，起到惩罚作用
            bias_j = - (w_D * norm_Dj[idx] + w_C * (1.0 - self.causal_Cj[m]))
            # bias_j = - (w_D * self.causal_Dj[m] + w_C * (1.0 - self.causal_Cj[m]))
            biases.append(bias_j)
            
        # 转换为 Tensor，并乘以超参数 λ
        causal_bias = torch.tensor(biases, dtype=torch.float32, device=self.args.device) * self.args.causal_lambda

        # 为了方便监控，我们把这一轮的 bias 存进实例变量
        self.current_causal_bias = {m: biases[idx].item() * self.args.causal_lambda for idx, m in enumerate(self.modal_names)}

        return causal_bias
    
    def compute_csc_loss(self, input_idx, embs_list,  weight_norm, epoch, total_epochs):
        import torch.nn.functional as F
        import math
        
        # 1. 过滤掉被关闭的模态 (值为 None 的特征)
        valid_embs = [emb for emb in embs_list if emb is not None]
        valid_modal_num = len(valid_embs)
        
        # 沿着维度1堆叠: 形状变为 [ent_num, valid_modal_num, hidden_size]
        X = torch.stack(valid_embs, dim=1) 
        
        # 2. 处理 weight_norm (即 alpha_t)
        if weight_norm.dim() == 3:
            alpha_t = weight_norm.mean(dim=1)  # [ent_num, modal_num]
        else:
            alpha_t = weight_norm

        # 3. 动态初始化或校准 alpha_prev
        if not hasattr(self, 'alpha_prev') or self.alpha_prev.size(1) != valid_modal_num:
            self.register_buffer('alpha_prev', torch.ones(X.size(0), valid_modal_num, device=X.device) / valid_modal_num)

        # ====== 核心计算：事实融合表征 z_factual ======
        # 用当前权重加权求和得到当前状态的表征
        z_factual = torch.sum(alpha_t.unsqueeze(-1) * X, dim=1)

        # ====== (正向) 时序平稳的反事实表征 z_prev ======
        alpha_prev_batch = self.alpha_prev[input_idx].detach() 
        z_prev = torch.sum(alpha_prev_batch.unsqueeze(-1) * X, dim=1) 

        # 采用余弦距离计算 (Distance = 1 - Cosine Similarity)
        # 只要保证模型当前状态 (z_factual) 和上一状态 (z_prev) 不要偏离太远即可
        d_pos = 1.0 - F.cosine_similarity(z_factual, z_prev, dim=-1)
        
        # 【重要修改】：删除了 d_neg (远离均匀分布的约束)，避免模型为了逃避惩罚而走向极端单模态
        loss_csc = d_pos.mean()

        # ======== CSC 模块的延迟预热 (Warm-up) 机制 ========
        if epoch < 10:
            # 前 10 个 Epoch 绝对不干预！
            # 让模型靠基础 Loss 自由探索，建立起健康的正向历史锚点
            lambda_t = 0.0  
        else:
            # 10 轮之后，健康的锚点已经成型，逐渐加大约束力度，防止模型在后续训练中震荡或灾难性遗忘
            lambda_t = self.args.csc_lambda_0 * ((epoch - 10) / (total_epochs - 10))

        # 更新历史权重 buffer，供下一个 Epoch 使用 (记得 detach 截断梯度)
        self.alpha_prev[input_idx] = alpha_t.detach()

        return lambda_t * loss_csc
    
    
    # def compute_csc_loss(self, input_idx, embs_list, weight_norm, epoch, total_epochs):
    #     import torch.nn.functional as F
        
    #     valid_modal_num = weight_norm.size(-1)
        
    #     if weight_norm.dim() == 3:
    #         alpha_t = weight_norm.mean(dim=1) 
    #     else:
    #         alpha_t = weight_norm

    #     # 1. 初始化慢速移动的“历史锚点”
    #     if not hasattr(self, 'alpha_prev') or self.alpha_prev.size(1) != valid_modal_num:
    #         self.register_buffer('alpha_prev', torch.ones(alpha_t.size(0), valid_modal_num, device=alpha_t.device) / valid_modal_num)

    #     # 负向锚点：永远盲目的均匀分布
    #     uniform_dist = torch.ones_like(alpha_t) / valid_modal_num

    #     # =====================================================================
    #     # 🌟 理论升级 1：纯粹的权重空间约束 (Weight Space KL-Divergence)
    #     # 抛弃高维特征空间的 Cosine 距离，直接用 KL 散度约束注意力的概率分布！
    #     # 梯度极其平滑、干净，彻底杜绝梯度消失和特征撕裂！
    #     # =====================================================================
    #     # d_pos: 当前权重离“历史锚点”有多远？ (要求靠近)
    #     d_pos = F.kl_div(torch.log(alpha_t + 1e-8), self.alpha_prev.detach(), reduction='none').sum(dim=-1)
        
    #     # d_neg: 当前权重离“盲目均匀分布”有多远？ (要求远离)
    #     d_neg = F.kl_div(torch.log(alpha_t + 1e-8), uniform_dist, reduction='none').sum(dim=-1)

    #     # 🌟 理论升级 2：课程感知掩码 (判断历史锚点是否成熟)
    #     # 如果历史锚点本身就是均匀分布(KL散度接近0)，说明样本刚进入训练，不予约束
    #     sim_anchor = F.kl_div(torch.log(self.alpha_prev + 1e-8), uniform_dist, reduction='none').sum(dim=-1)
    #     mature_mask = (sim_anchor > 0.01).float() 

    #     # 计算 Margin Loss
    #     margin = self.args.csc_gamma
    #     loss_per_sample = torch.clamp(d_pos - d_neg + margin, min=0.0)
        
    #     # 应用掩码并求平均
    #     loss_csc = (loss_per_sample * mature_mask).mean()

    #     # 延迟预热 (前 10 轮让模块一自由发挥)
    #     if epoch < 10:
    #         lambda_t = 0.0  
    #     else:
    #         lambda_t = self.args.csc_lambda_0 * ((epoch - 10) / (total_epochs - 10))

    #     # =====================================================================
    #     # 💣 致命 Bug 修复：解决“金鱼记忆”问题 (真正的 EMA 慢速更新)
    #     # 保证 alpha_prev 记录的是真实的跨 Epoch 历史，而不是上一个 Batch 的状态！
    #     # =====================================================================
    #     if self.training:
    #         momentum = 0.999  # 0.95 意味着它记忆了过去约 20 个 Step 的平滑状态
    #         self.alpha_prev.data = momentum * self.alpha_prev.data + (1.0 - momentum) * alpha_t.detach()

    #     return lambda_t * loss_csc
    
    #提供一个健壮的 compute_csc_loss 实现
    # def compute_csc_loss(self, input_idx, embs_list, weight_norm, epoch, total_epochs):
    #     import torch.nn.functional as F
    #     import math
        
    #     # 1. 过滤掉被关闭的模态 (值为 None 的特征)
    #     valid_embs = [emb for emb in embs_list if emb is not None]
    #     valid_modal_num = len(valid_embs)
        
    #     # 沿着维度1堆叠: 形状变为 [ent_num, valid_modal_num, hidden_size]
    #     X = torch.stack(valid_embs, dim=1)

    #     # =====================================================================
    #     # 💣 终极拆弹：切断梯度，保护底层特征空间不被撕裂！
    #     # CSC 模块的作用是“约束注意力权重”，绝不能让梯度流回底层特征去瞎改特征！
    #     # =====================================================================
    #     X_detach = X.detach() 
        
    #     # 2. 处理 weight_norm (即 alpha_t)
    #     if weight_norm.dim() == 3:
    #         alpha_t = weight_norm.mean(dim=1)  # [ent_num, modal_num]
    #     else:
    #         alpha_t = weight_norm

    #     # 3. 动态初始化或校准 alpha_prev
    #     if not hasattr(self, 'alpha_prev') or self.alpha_prev.size(1) != valid_modal_num:
    #         self.register_buffer('alpha_prev', torch.ones(X.size(0), valid_modal_num, device=X.device) / valid_modal_num)

    #     # ====== 核心修复：手动计算事实融合表征 z_factual (对应论文公式 8) ======
    #     # 不使用拼接的 1200 维 joint_emb_factual，而是用当前权重加权求和得到 300 维表征
    #     # ====== 使用 X_detach 计算事实融合表征 z_factual ======
    #     # 此时，z_factual 的梯度只能顺着 alpha_t 流向注意力生成网络，完美！
    #     # z_factual = torch.sum(alpha_t.unsqueeze(-1) * X, dim=1)
    #     z_factual = torch.sum(alpha_t.unsqueeze(-1) * X_detach, dim=1)

    #     # (正向) 时序平稳的反事实表征 z_prev
    #     alpha_prev_batch = self.alpha_prev[input_idx].detach() 
    #     z_prev = torch.sum(alpha_prev_batch.unsqueeze(-1) * X_detach, dim=1)

    #     # (负向) 因果盲目的反事实表征 z_uni (均匀融合)
    #     z_uni = torch.mean(X_detach, dim=1)

    #     # =====================================================================
    #     # 🌟 核心理论创新：课程感知掩码 (Curriculum-Aware Gating)
    #     # 诊断样本是否"成熟"：比较历史锚点(z_prev)和盲目锚点(z_uni)
    #     # 如果它们高度重合(余弦相似度>0.99)，说明该样本刚被模块一放进来，还没学到有效权重
    #     # =====================================================================
    #     sim_anchor = F.cosine_similarity(z_prev, z_uni, dim=-1)
    #     mature_mask = (sim_anchor < 0.99).float() # 1表示成熟可约束，0表示刚引入需自由探索

    #     # 采用余弦距离计算 (Distance = 1 - Cosine Similarity)
    #     # 注意这里把 joint_emb_factual 换成了 z_factual
    #     d_pos = 1.0 - F.cosine_similarity(z_factual, z_prev, dim=-1)
    #     d_neg = 1.0 - F.cosine_similarity(z_factual, z_uni, dim=-1)
        
    #     # 计算基于间隔(Margin)的对比损失: max(0, d_pos - d_neg + gamma)
    #     margin = self.args.csc_gamma
    #     loss_per_sample = torch.clamp(d_pos - d_neg + margin, min=0.0)
    #     # 应用成熟度掩码，只对成熟样本计算 CSC Loss
    #     loss_csc = (loss_per_sample * mature_mask).mean()
    #     # loss_csc = torch.clamp(d_pos - d_neg + margin, min=0.0).mean()

    #     # 计算时间衰减平滑系数 lambda(t)
    #     # ======== 修复：CSC 模块的延迟预热 (Warm-up) 机制 ========
    #     if epoch < 10:
    #         # 前 10 个 Epoch 绝对不干预！
    #         # 让模型靠基础 Loss 自由探索，建立起健康的、摆脱了均匀分布的正向历史锚点 (W_avg)
    #         lambda_t = 0.0  
    #     else:
    #         # 10 轮之后，健康的锚点已经成型
    #         # 此时逐渐加大约束力度，防止模型在后续训练中震荡或灾难性遗忘
    #         lambda_t = self.args.csc_lambda_0 * ((epoch - 10) / (total_epochs - 10))
    #     # lambda_t = self.args.csc_lambda_0 * math.exp(-self.args.csc_eta * (epoch / total_epochs))

    #     # 更新历史权重 buffer，供下一个 Epoch 使用 (记得 detach)
    #     self.alpha_prev[input_idx] = alpha_t.detach()

    #     return lambda_t * loss_csc

    def _register_grad_hooks(self, embs_dict):
        """利用 Hook 在反向传播时自动捕获并更新 D_j"""
        if not self.training or not self.args.use_causal_bias:
            return
            
        for name, emb in embs_dict.items():
            if emb is not None and emb.requires_grad:
                # 注册 Hook，当当前模态的 embedding 计算出梯度时回调
                emb.register_hook(lambda grad, n=name: self._update_Dj(n, grad))

    def _update_Dj(self, modal_name, grad):
        """(2) 模态瞬时因果效应信号更新 (公式 11)"""
        grad_norm = grad.norm(2).item()
        gamma = self.args.causal_gamma
        # 滑动平均更新 D_j
        self.causal_Dj[modal_name] = gamma * self.causal_Dj[modal_name] + (1 - gamma) * grad_norm
        
    def update_Cj(self, modal_hits_dict):
        """
        (3) 模态固有因果置信度更新 (外部验证集触发调用) (公式 14)
        modal_hits_dict 格式如: {'img': 0.65, 'gph': 0.82, ...}
        """
        beta = self.args.causal_beta
        for m, hits_score in modal_hits_dict.items():
            if m in self.causal_Cj:
                self.causal_Cj[m] = beta * hits_score + (1 - beta) * self.causal_Cj[m]
