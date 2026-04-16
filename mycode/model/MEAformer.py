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

from transformers import AutoModel, AutoTokenizer


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
        self.use_plm = (getattr(self.args, "use_plm", 0) == 1) and (kgs.get("plm_input_ids") is not None)
        if kgs["name_features"] is not None:
            self.name_features = kgs["name_features"].cuda()
            self.char_features = kgs["char_features"].cuda()

        # ======= PLM 模型加载逻辑 =======
        if self.use_plm:
            self.plm_input_ids = kgs["plm_input_ids"].cuda()
            self.plm_attention_mask = kgs["plm_attention_mask"].cuda()
            
            print(f"Loading PLM model [{self.args.plm_name}] into memory...")
            self.plm = AutoModel.from_pretrained(self.args.plm_name).cuda()
            self.plm_tokenizer = AutoTokenizer.from_pretrained(self.args.plm_name) # ⚠️ 新增 Tokenizer
            
            # 冻结参数控制
            if self.args.freeze_plm == 1:
                for param in self.plm.parameters():
                    param.requires_grad = False
                self.plm.eval()
            
            # 投影层：将 BERT 的隐藏层维度映射为 MEAformer 需要的文本维度
            name_dim = self.args.name_dim if hasattr(self.args, 'name_dim') else 300
            self.plm_proj = nn.Sequential(
                nn.Linear(self.args.plm_hidden_dim, name_dim),
                nn.LayerNorm(name_dim)
            ).cuda()
            
            if self.args.freeze_plm == 1:
                print("Extracting offline PLM features for names...")
                self.static_plm_features = self._extract_plm_features().detach().cuda()
                
            # # ====== 🚀 新增：关系和属性的 PLM 语义提取 ======
            # if getattr(self.args, "plm_embed_rel", 0) == 1 and "rel_texts" in kgs:
            #     print("Extracting PLM features for Relations...")
            #     # 提取关系文本向量 (1000 x 768)
            #     self.rel_text_embs = self._extract_plm_features(kgs["rel_texts"]).detach().cuda()
            #     self.rel_plm_proj = nn.Sequential(
            #         nn.Linear(self.args.plm_hidden_dim, self.args.hidden_size),
            #         nn.LayerNorm(self.args.hidden_size)
            #     ).cuda()
            #     self.raw_rel_features = self.rel_features # 备份实体的稀疏矩阵 [ent_num, 1000]

            #     # ====== 🔥 核心修复：关系专属残差偏移参数 ======
            #     self.rel_id_shift = nn.Parameter(torch.zeros(self.raw_rel_features.shape[1], self.args.hidden_size)).cuda()
            #     nn.init.xavier_normal_(self.rel_id_shift)

            # if getattr(self.args, "plm_embed_attr", 0) == 1 and "attr_texts" in kgs:
            #     print("Extracting PLM features for Attributes...")
            #     # 提取属性文本向量 (1000 x 768)
            #     self.att_text_embs = self._extract_plm_features(kgs["attr_texts"]).detach().cuda()
            #     self.att_plm_proj = nn.Sequential(
            #         nn.Linear(self.args.plm_hidden_dim, self.args.hidden_size),
            #         nn.LayerNorm(self.args.hidden_size)
            #     ).cuda()
            #     self.raw_att_features = self.att_features # 备份实体的稀疏矩阵 [ent_num, 1000]

            #     # ====== 🔥 核心修复：属性专属残差偏移参数 ======
            #     self.att_id_shift = nn.Parameter(torch.zeros(self.raw_att_features.shape[1], self.args.hidden_size)).cuda()
            #     nn.init.xavier_normal_(self.att_id_shift)
            # # ===============================================

        img_dim = self._get_img_dim(kgs)
        char_dim = kgs["char_features"].shape[1] if self.char_features is not None else 100

        # # ⚠️ 修改：动态计算属性特征输入维度 (如果开启 PLM 属性，则为 300 维)
        # attr_input_dim = self.args.hidden_size if (getattr(self.args, "use_plm", 0) == 1 and getattr(self.args, "plm_embed_attr", 0) == 1) else kgs["att_features"].shape[1]
        attr_input_dim = kgs["att_features"].shape[1]

        self.multimodal_encoder = MultiModalEncoder(args=self.args,
                                                    ent_num=kgs["ent_num"],
                                                    img_feature_dim=img_dim,
                                                    char_feature_dim=char_dim,
                                                    use_project_head=self.args.use_project_head,
                                                    attr_input_dim=attr_input_dim) 

        # # ====== 🚀 终极维度对齐补丁 ======
        # # 因为我们已经用 PLM 把关系和属性特征降维且语义化到了 300 维，
        # # 所以必须把 Encoder 底层原本写死的 1000 维接收器替换掉！
        # if getattr(self.args, "use_plm", 0) == 1:
        #     if getattr(self.args, "plm_embed_rel", 0) == 1:
        #         # 强行覆盖底层关系投影层：300 -> 300
        #         self.multimodal_encoder.rel_fc = nn.Linear(self.args.hidden_size, self.args.hidden_size).cuda()
                
        #     if getattr(self.args, "plm_embed_attr", 0) == 1:
        #         # 强行覆盖底层属性投影层：300 -> 300 (双保险)
        #         if hasattr(self.multimodal_encoder, 'att_fc'):
        #             self.multimodal_encoder.att_fc = nn.Linear(self.args.hidden_size, self.args.hidden_size).cuda()
        # # ==================================

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

        # ================= 🚀 终极版：早融合 PLM 模块 =================
        self.plm_ent_attr = getattr(self.args, "plm_ent_attr", 0)
        if self.plm_ent_attr == 1:
            plm_dim = getattr(self.args, "plm_hidden_dim", 768)
            # 核心改变：直接映射到你最原始的属性维度 (通常是 1000)
            target_dim = self.att_features.shape[1] 
            
            self.plm_early_adapter = nn.Sequential(
                nn.Linear(plm_dim, target_dim),
                nn.GELU(),
                nn.LayerNorm(target_dim)
            )
            # 👇 极其重要的安全锁，必须保留！
            self.plm_gate = nn.Parameter(torch.tensor(0.01))
            print(f"✅ Early Fusion PLM Module Activated! ({plm_dim} -> {target_dim})")
            
        self.plm_features = None 
        # =============================================================================


        
    def _extract_plm_features(self):
        """
        专用名称特征提取：
        利用冻结的 PLM 提取实体表面名称 (Surface Name) 的初始特征。
        """
        is_training = self.plm.training
        self.plm.eval()
        all_embs = []
        batch_size = 512
        
        with torch.set_grad_enabled(not self.args.freeze_plm):
            # 原有的实体名称特征提取逻辑 (用于 self.use_plm 开启时提取名字)
            for i in range(0, self.plm_input_ids.shape[0], batch_size):
                b_ids = self.plm_input_ids[i : i+batch_size]
                b_mask = self.plm_attention_mask[i : i+batch_size]
                
                # 送入 PLM 提取特征
                outputs = self.plm(input_ids=b_ids, attention_mask=b_mask)
                cls_emb = outputs.last_hidden_state[:, 0, :] 
                all_embs.append(cls_emb)
                    
        plm_features = torch.cat(all_embs, dim=0)
        
        if is_training:
            self.plm.train()
            
        return plm_features

    def forward(self, input_batch, epoch=0, total_epochs=1):
        device = self.input_idx.device # 安全获取设备
        
        # ================= 🚀 核心创新：前置语义拦截注入 =================
        # 我们在 GCN 和注意力机制运转【之前】，把 PLM 的丰富语义悄悄注入到属性特征中！
        original_att = self.att_features
        if getattr(self, "plm_ent_attr", 0) == 1 and self.plm_features is not None:
            plm_feat = self.plm_features.to(device)
            adapted_plm = self.plm_early_adapter(plm_feat)
            
            # 动态覆盖原始属性特征（这样网络内部的 Attention 就能看到高级语义了！）
            self.att_features = original_att + self.plm_gate * adapted_plm
        # =====================================================================

        # 生成所有实体和隐藏层嵌入 (此时里面的 Attention 会自然给 att 分配高权重！)
        gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states, weight_norm = self.joint_emb_generat(only_joint=False, epoch=epoch, total_epochs=total_epochs)
        gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, joint_emb_hid = self.generate_hidden_emb(hidden_states)

        # ====== 极其重要：恢复原始特征，防止内存泄漏和计算图崩溃 ======
        self.att_features = original_att
        # =====================================================================


        # ====== 新增：初始化软权重 ======
        sample_weights = None 

        # 统一处理输入格式
        if isinstance(input_batch, dict):  # 训练数据（包含难度信息）
            batch_tensor = torch.stack([
                torch.tensor(input_batch['ent1'], dtype=torch.int64, device=device),
                torch.tensor(input_batch['ent2'], dtype=torch.int64, device=device)
            ], dim=1)
            difficulties = torch.tensor(input_batch['difficulties'], device=device)

            # 在 forward() 的样本调度部分
            if self.args.use_sample_schedule == 1 and self.training:
                progress = epoch / total_epochs
                threshold = 1.0 / (1.0 + math.exp(-self.args.k * (progress - 0.5)))
                
                # 软权重计算（保留）
                sample_weights = torch.exp(-(difficulties - threshold).clamp(min=0))
                
                # 【修改】去掉 noise_ceiling 硬截断！
                # 改用温和的下界限制，保证最难的样本也有一定的学习权重
                sample_weights = sample_weights.clamp(min=0.05)
                


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
        in_loss = self.inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, 
                                batch_tensor, sample_weights=sample_weights)
        out_loss = self.inner_view_loss(gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, 
                                 name_emb_hid, char_emb_hid, batch_tensor, 
                                 sample_weights=sample_weights)
        

        # 总损失
        loss_all = loss_joi + in_loss + out_loss
        

        # # 总损失
        # loss_all = loss_joi + in_loss + out_loss
        loss_dic = {"joint_Intra_modal": loss_joi.item(), "Intra_modal": in_loss.item()}
        output = {"loss_dic": loss_dic, "emb": joint_emb}

        # # ====== 新增：构造 CSC 模块需要的输入 ======
        # if self.args.use_csc and self.training:
        #     # 1. 打包 embs_list (注意顺序必须和 Encoder 里一致)
        #     embs_list = [img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb]
            
        #     # 2. 获取所有实体的索引
        #     input_idx = self.input_idx
            
        #     # 3. 计算反事实约束 Loss (传入打包好的变量)
        #     loss_csc = self.compute_csc_loss(input_idx, embs_list, weight_norm, epoch, total_epochs)
            
        #     # 4. 叠加 Loss 并记录
        #     loss_all = loss_all + loss_csc
        #     output["loss_dic"]["CSC_Loss"] = loss_csc.item()
        # # ==========================================

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


    def joint_emb_generat(self, only_joint=True, epoch=0, total_epochs=1):
        # 【修改】只更新 C_j 等因果信号状态（用于推理时融合），不把 bias 注入 attention
        if self.args.use_causal_bias and self.training:
            self._compute_causal_bias(epoch, total_epochs)
        causal_bias = None  # 始终传 None，不干扰训练时的注意力
    
        # 默认特征
        current_name_features = self.name_features
        current_rel_features = self.rel_features
        current_att_features = self.att_features
        
        if getattr(self, "use_plm", False):
            # 1. 实体名称嵌入 (Surface) - 安全检查：只有原特征不为 None 才注入
            if getattr(self.args, "plm_embed_name", 1) == 1 and self.name_features is not None:
                if self.args.freeze_plm == 1:
                    current_name_features = F.normalize(self.plm_proj(self.static_plm_features), dim=1)
                else:
                    plm_feats = self._extract_plm_features()
                    current_name_features = F.normalize(self.plm_proj(plm_feats), dim=1)
            
            # 2. 关系文本语义嵌入 (Relation) - 安全检查
            if getattr(self.args, "plm_embed_rel", 0) == 1 and hasattr(self, "rel_plm_proj") and self.rel_features is not None:
                rel_semantic = self.rel_plm_proj(self.rel_text_embs)
                rel_fused = rel_semantic + self.rel_id_shift
                current_rel_features = torch.matmul(self.raw_rel_features, rel_fused)
                current_rel_features = F.normalize(current_rel_features, dim=1)
                
            # 3. 属性文本语义嵌入 (Attribute) - 安全检查
            if getattr(self.args, "plm_embed_attr", 0) == 1 and hasattr(self, "att_plm_proj") and self.att_features is not None:
                att_semantic = self.att_plm_proj(self.att_text_embs)
                att_fused = att_semantic + self.att_id_shift
                current_att_features = torch.matmul(self.raw_att_features, att_fused)
                current_att_features = F.normalize(current_att_features, dim=1)

        # ... 后面 encoder 调用中 causal_bias=None ...
        gph_emb, img_emb, rel_emb, att_emb, \
            name_emb, char_emb, joint_emb, hidden_states, weight_norm = self.multimodal_encoder(
                self.input_idx, self.adj, self.img_features, 
                current_rel_features, current_att_features, 
                current_name_features, self.char_features,
                causal_bias=None  # 不再注入 attention！
        )
        

        # 绑定梯度捕获 (只在重新生成嵌入时挂载 Hook)
        embs_dict = {'img': img_emb, 'att': att_emb, 'rel': rel_emb, 
                     'gph': gph_emb, 'name': name_emb, 'char': char_emb}
        self._register_grad_hooks(embs_dict)

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
    

    def compute_csc_loss(self, input_idx, embs_list, weight_norm, epoch, total_epochs):
        """
        反事实一致性约束 (Counterfactual Sufficiency Constraint)
        """
        valid_embs = [e for e in embs_list if e is not None]
        if len(valid_embs) < 2 or weight_norm is None:
            return torch.tensor(0.0, device=self.input_idx.device)

        N = weight_norm.shape[0]
        sample_size = min(N, 2048)
        rand_idx = torch.randperm(N, device=weight_norm.device)[:sample_size]

        sampled_embs = [F.normalize(e[rand_idx], dim=-1) for e in valid_embs]
        curr_weights = weight_norm[rand_idx]

        # Teacher: 联合表征
        stacked = torch.stack(sampled_embs, dim=1)
        teacher_emb = torch.sum(curr_weights.unsqueeze(2) * stacked, dim=1)
        teacher_emb = F.normalize(teacher_emb.detach(), dim=-1)

        # Student: 每个模态独立与 teacher 做对比对齐
        tau_distill = self.args.tau
        loss_distill = 0.0

        for m_idx in range(len(valid_embs)):
            student = sampled_embs[m_idx]
            logits = torch.mm(student, teacher_emb.t()) / tau_distill
            labels = torch.arange(sample_size, device=logits.device)
            loss_m = F.cross_entropy(logits, labels)
            loss_distill += loss_m

        loss_distill = loss_distill / len(valid_embs)

        # ====== 改动1: 退火方向反转 —— 前期强约束建立基础，后期放松让模型精细调整 ======
        progress = epoch / max(1, total_epochs)
        # 余弦退火：从 1.0 平滑衰减到 0
        lambda_t = 0.5 * (1.0 + math.cos(math.pi * progress))

        # ====== 改动2: 量级缩放 —— 将 CSC loss 压缩到主 loss 的 5-10% ======
        # InfoNCE 原始量级约 4-6，主 loss (joint) 约 1-2
        # 目标：CSC 贡献约 0.1 ~ 0.3，所以需要除以一个归一化因子
        normalizer = max(loss_distill.item(), 1.0)  # 防止除零
        loss_distill_normalized = loss_distill / normalizer  # 归一化到 ~1.0 附近

        return self.args.csc_lambda_0 * lambda_t * loss_distill_normalized

    

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
