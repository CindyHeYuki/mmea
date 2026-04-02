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



    def forward(self, input_batch, epoch=0, total_epochs=1):
        # 生成所有实体和隐藏层嵌入 (必须在最前面，为了后面统一获取 Device)
        gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states = self.joint_emb_generat(only_joint=False, epoch=epoch, total_epochs=total_epochs)
        gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, joint_emb_hid = self.generate_hidden_emb(hidden_states)

        # 动态获取所在设备，避免硬编码 .cuda() 或未定义的 self.device
        device = joint_emb.device 

        # 统一处理输入格式
        if isinstance(input_batch, dict):  # 训练数据（包含难度信息）
            batch_tensor = torch.stack([
                torch.tensor(input_batch['ent1'], dtype=torch.int64, device=device),
                torch.tensor(input_batch['ent2'], dtype=torch.int64, device=device)
            ], dim=1)
            difficulties = torch.tensor(input_batch['difficulties'], device=device)
        else:  # 测试数据（无难度信息）
            batch_tensor = torch.tensor(input_batch, dtype=torch.int64, device=device)
            difficulties = None

        # 计算对比损失
        if self.args.replay:
            all_ent_batch = torch.cat([batch_tensor[:, 0], batch_tensor[:, 1]])
            
            if not self.replay_ready:
                # 【修复】传完整的 joint_emb，而不是切片后的 ent_emb
                loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch_tensor)
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
                
                # 【修复】同理，恢复原有的参数签名
                loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch_tensor, neg_l_ipt, neg_r_ipt)
            
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
            # 【修复】恢复原有传参
            loss_joi = self.criterion_cl_joint(joint_emb, batch_tensor)
        
        # 计算内部视图损失与输出视图损失
        # 【修复】安全传入 difficulties：只有在它是字典输入时，才传入 difficulties 参数，防止测试集报错
        if difficulties is not None:
            in_loss = self.inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, batch_tensor, difficulties=difficulties)
            out_loss = self.inner_view_loss(gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, batch_tensor, difficulties=difficulties)
        else:
            in_loss = self.inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, batch_tensor)
            out_loss = self.inner_view_loss(gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, batch_tensor)
        
        # 总损失
        loss_all = loss_joi + in_loss + out_loss
        
        # 准备输出
        loss_dic = {"joint_Intra_modal": loss_joi.item(), "Intra_modal": in_loss.item()}
        output = {"loss_dic": loss_dic, "emb": joint_emb}
        
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
    
    def inner_view_loss(self, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill, difficulties=None):
        # 如果传入了 difficulties，就打包成参数字典传给底层的 loss 函数
        kwargs = {'difficulties': difficulties} if difficulties is not None else {}
        
        loss_GCN = self.criterion_cl(gph_emb, train_ill, **kwargs) if gph_emb is not None else 0
        loss_rel = self.criterion_cl(rel_emb, train_ill, **kwargs) if rel_emb is not None else 0
        loss_att = self.criterion_cl(att_emb, train_ill, **kwargs) if att_emb is not None else 0
        loss_img = self.criterion_cl(img_emb, train_ill, **kwargs) if img_emb is not None else 0
        loss_name = self.criterion_cl(name_emb, train_ill, **kwargs) if name_emb is not None else 0
        loss_char = self.criterion_cl(char_emb, train_ill, **kwargs) if char_emb is not None else 0

        total_loss = self.multi_loss_layer([loss_GCN, loss_rel, loss_att, loss_img, loss_name, loss_char])
        return total_loss

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
            return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states

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
        
        biases = []
        for m in self.modal_names:
            # (2) 计算偏置 Bias_j (公式 16 对应的惩罚逻辑)
            # Bias 为负值，起到惩罚作用
            bias_j = - (w_D * self.causal_Dj[m] + w_C * (1.0 - self.causal_Cj[m]))
            biases.append(bias_j)
            
        # 转换为 Tensor，并乘以超参数 λ
        causal_bias = torch.tensor(biases, dtype=torch.float32, device=self.args.device) * self.args.causal_lambda
        return causal_bias

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
