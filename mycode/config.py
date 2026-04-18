import os.path as osp
import numpy as np
import random
import torch
from easydict import EasyDict as edict
import argparse


class cfg():
    def __init__(self):
        self.this_dir = osp.dirname(__file__)
        # change
        self.data_root = osp.abspath(osp.join(self.this_dir, '..', '..', 'data', ''))

    def get_args(self):
        parser = argparse.ArgumentParser()
        # base
        parser.add_argument('--gpu', default=0, type=int)
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--epoch', default=100, type=int)
        parser.add_argument("--save_model", default=0, type=int, choices=[0, 1])
        parser.add_argument("--only_test", default=0, type=int, choices=[0, 1])

        # torthlight
        parser.add_argument("--no_tensorboard", default=False, action="store_true")
        parser.add_argument("--exp_name", default="EA_exp", type=str, help="Experiment name")
        parser.add_argument("--dump_path", default="dump/", type=str, help="Experiment dump path")
        parser.add_argument("--exp_id", default="001", type=str, help="Experiment ID")
        parser.add_argument("--random_seed", default=42, type=int)
        parser.add_argument("--data_path", default="mmkg", type=str, help="Experiment path")

        # --------- EA -----------
        parser.add_argument("--data_choice", default="DBP15K", type=str, choices=["DBP15K", "DWY", "FBYG15K", "FBDB15K"], help="Experiment path")
        parser.add_argument("--data_rate", type=float, default=0.3, help="training set rate")
        # parser.add_argument("--data_rate", type=float, default=0.3, choices=[0.2, 0.3, 0.5, 0.8], help="training set rate")

        # ====== 新增：模块一 样本调度机制 ======
        parser.add_argument("--k", default=6, type=float, 
                    help="调度速度系数，控制样本引入速度（指数衰减参数）")
        parser.add_argument("--lambda_val", default=0.2, type=float,
                    help="结构稀疏度权重（0-1），用于计算样本难度：λ·ρ_struct + (1-λ)·ρ_modal")
        parser.add_argument("--use_sample_schedule", default=1, type=int, choices=[0, 1],
                    help="消融实验开关：是否启用样本调度 (1: 开启, 0: 关闭)")
        # ====== B3: 三维难度度量参数 ======
        parser.add_argument("--use_3d_difficulty", default=0, type=int, choices=[0, 1],
                    help="是否启用三维难度度量（B3）。0=只用旧的二维 lambda_val 公式")
        parser.add_argument("--lambda_struct", default=0.4, type=float,
                            help="难度计算中结构稀疏度的权重 (建议 0.3-0.5)")
        parser.add_argument("--lambda_modal", default=0.3, type=float,
                            help="难度计算中模态不一致度的权重 (建议 0.2-0.4)")
        parser.add_argument("--lambda_ambig", default=0.3, type=float,
                            help="难度计算中名称非独特性的权重 (建议 0.2-0.4)")
        # ===================================
        # ===================================================

        # ====== 新增：模块二 动态因果效应加权机制参数 ======
        parser.add_argument("--use_causal_bias", default=1, type=int, choices=[0, 1],
                    help="消融实验开关：是否启用动态因果加权机制 (1: 开启, 0: 关闭)")
        parser.add_argument("--causal_lambda", default=0.1, type=float,
                    help="公式中的 λ：因果引导偏置项的整体强度")
        parser.add_argument("--causal_gamma", default=0.9, type=float,
                    help="计算瞬时因果效应 D_j 的滑动平均因子 (平滑因子)")
        parser.add_argument("--causal_beta", default=0.5, type=float,
                    help="计算固有因果置信度 C_j 的历史衰减因子 β")
        parser.add_argument("--causal_eval_k", default=10, type=int,
                    help="每 K 个 epoch 评估一次固有因果置信度 C_j")
        # ===================================================

        # ====== 新增：模块三 反事实平滑约束参数 ======
        parser.add_argument("--use_csc", default=1, type=int, choices=[0, 1], help="是否启用反事实平滑约束")
        parser.add_argument("--csc_lambda_0", default=0.1, type=float, help="初始平滑强度 lambda_0")
        parser.add_argument("--csc_eta", default=5.0, type=float, help="衰减速率控制参数 eta")
        parser.add_argument("--csc_gamma", default=0.1, type=float, help="反事实对比的间隔 Margin")
        # =============================================

        parser.add_argument("--do_alpha_sweep", default=0, type=int, choices=[0, 1],
                    help="是否在最后一次评估时进行 α 扫描")
        

        # ====== B1: 邻居增强参数 ======
        parser.add_argument("--use_neighbor", default=0, type=int, choices=[0, 1],
                            help="是否启用邻居增强距离融合 (B1)")
        parser.add_argument("--neighbor_alpha", default=0.2, type=float,
                            help="邻居距离融合权重 (建议 0.1-0.3)")
        # ===============================

        # ====== 新增：预训练语言模型 (PLM) 模块参数 ======
        # ====== PLM 全能控制模块 ======
        parser.add_argument("--use_plm", default=1, type=int, choices=[0, 1], 
                            help="是否启用PLM来替代传统的GloVe名称特征 (1: 开启, 0: 关闭)")
        parser.add_argument("--plm_name", default="bert-base-multilingual-cased", type=str, 
                            help="HuggingFace模型名称, 高度推荐用多语言模型, 支持 bert, roberta, xlm-roberta 等")
        parser.add_argument("--plm_max_len", default=16, type=int, 
                            help="文本最大截断长度 (实体名字一般较短，16或32即可)")
        parser.add_argument("--freeze_plm", default=1, type=int, choices=[0, 1], 
                            help="是否冻结PLM的参数。由于图节点庞大，推荐为1防OOM。投影层将保持可训练。")
        parser.add_argument("--plm_hidden_dim", default=768, type=int, 
                            help="PLM的输出维度 (bert-base通常为768)")
       
        # --- 新增任务细分开关 ---
        parser.add_argument("--plm_embed_name", default=1, type=int, choices=[0, 1], 
                            help="是否用PLM做名称(Surface)嵌入")
        parser.add_argument("--plm_embed_rel", default=0, type=int, choices=[0, 1], 
                            help="是否用PLM做关系(Relation)文本嵌入")
        parser.add_argument("--plm_embed_attr", default=0, type=int, choices=[0, 1], 
                            help="是否用PLM做属性名(Attribute)文本嵌入")
        # 降维策略
        parser.add_argument("--plm_reduce_method", default="cls", choices=["cls", "mean"], help="提取特征的方式")

        # =================================================


        # TODO: add some dynamic variable
        parser.add_argument("--model_name", default="MEAformer", type=str, choices=["EVA", "MCLEA", "MSNEA", "MEAformer"], help="model name")
        parser.add_argument("--model_name_save", default="", type=str, help="model name for model load")

        parser.add_argument('--workers', type=int, default=8)
        parser.add_argument('--accumulation_steps', type=int, default=1)
        parser.add_argument("--scheduler", default="linear", type=str, choices=["linear", "cos", "fixed"])
        parser.add_argument("--optim", default="adamw", type=str, choices=["adamw", "adam"])
        parser.add_argument('--lr', type=float, default=3e-5)
        parser.add_argument('--weight_decay', type=float, default=0.0001)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument('--eval_epoch', default=100, type=int, help='evaluate each n epoch')
        parser.add_argument("--enable_sota", action="store_true", default=False)

        parser.add_argument('--margin', default=1, type=float, help='The fixed margin in loss function. ')
        parser.add_argument('--emb_dim', default=1000, type=int, help='The embedding dimension in KGE model.')
        parser.add_argument('--adv_temp', default=1.0, type=float, help='The temperature of sampling in self-adversarial negative sampling.')
        parser.add_argument("--contrastive_loss", default=0, type=int, choices=[0, 1])
        parser.add_argument('--clip', type=float, default=1., help='gradient clipping')

        # --------- EVA -----------
        parser.add_argument("--data_split", default="fr_en", type=str, help="Experiment split", choices=["dbp_wd_15k_V2", "dbp_wd_15k_V1", "zh_en", "ja_en", "fr_en", "norm"])
        parser.add_argument("--hidden_units", type=str, default="128,128,128", help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
        parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for layers")
        parser.add_argument("--attn_dropout", type=float, default=0.0, help="dropout rate for gat layers")
        parser.add_argument("--distance", type=int, default=2, help="L1 distance or L2 distance. ('1', '2')", choices=[1, 2])
        parser.add_argument("--csls", action="store_true", default=False, help="use CSLS for inference")
        parser.add_argument("--csls_k", type=int, default=10, help="top k for csls")
        parser.add_argument("--il", action="store_true", default=False, help="Iterative learning?")
        parser.add_argument("--semi_learn_step", type=int, default=10, help="If IL, what's the update step?")
        parser.add_argument("--il_start", type=int, default=500, help="If Il, when to start?")
        parser.add_argument("--unsup", action="store_true", default=False)
        parser.add_argument("--unsup_k", type=int, default=1000, help="|visual seed|")

        # --------- MCLEA -----------
        parser.add_argument("--unsup_mode", type=str, default="img", help="unsup mode", choices=["img", "name", "char"])
        parser.add_argument("--tau", type=float, default=0.1, help="the temperature factor of contrastive loss")
        parser.add_argument("--alpha", type=float, default=0.2, help="the margin of InfoMaxNCE loss")
        parser.add_argument("--with_weight", type=int, default=1, help="Whether to weight the fusion of different ")
        parser.add_argument("--structure_encoder", type=str, default="gat", help="the encoder of structure view", choices=["gat", "gcn"])
        parser.add_argument("--ab_weight", type=float, default=0.5, help="the weight of NTXent Loss")

        parser.add_argument("--projection", action="store_true", default=False, help="add projection for model")
        parser.add_argument("--heads", type=str, default="2,2", help="heads in each gat layer, splitted with comma")
        parser.add_argument("--instance_normalization", action="store_true", default=False, help="enable instance normalization")
        parser.add_argument("--attr_dim", type=int, default=100, help="the hidden size of attr and rel features")
        parser.add_argument("--img_dim", type=int, default=100, help="the hidden size of img feature")
        parser.add_argument("--name_dim", type=int, default=100, help="the hidden size of name feature")
        parser.add_argument("--char_dim", type=int, default=100, help="the hidden size of char feature")

        parser.add_argument("--w_gcn", action="store_false", default=True, help="with gcn features")
        parser.add_argument("--w_rel", action="store_false", default=True, help="with rel features")
        parser.add_argument("--w_attr", action="store_false", default=True, help="with attr features")
        parser.add_argument("--w_name", action="store_false", default=True, help="with name features")
        parser.add_argument("--w_char", action="store_false", default=True, help="with char features")
        parser.add_argument("--w_img", action="store_false", default=True, help="with img features")
        parser.add_argument("--use_surface", type=int, default=0, help="whether to use the surface")

        parser.add_argument("--inner_view_num", type=int, default=6, help="the number of inner view")
        parser.add_argument("--word_embedding", type=str, default="glove", help="the type of word embedding, [glove|fasttext]", choices=["glove", "bert"])
        # projection head
        parser.add_argument("--use_project_head", action="store_true", default=False, help="use projection head")
        parser.add_argument("--zoom", type=float, default=0.1, help="narrow the range of losses")
        parser.add_argument("--reduction", type=str, default="mean", help="[sum|mean]", choices=["sum", "mean"])

        # --------- MEAformer -----------
        parser.add_argument("--hidden_size", type=int, default=100, help="the hidden size of MEAformer")
        parser.add_argument("--intermediate_size", type=int, default=400, help="the hidden size of MEAformer")
        parser.add_argument("--num_attention_heads", type=int, default=5, help="the number of attention_heads of MEAformer")
        parser.add_argument("--num_hidden_layers", type=int, default=2, help="the number of hidden_layers of MEAformer")
        parser.add_argument("--position_embedding_type", default="absolute", type=str)
        parser.add_argument("--use_intermediate", type=int, default=1, help="whether to use_intermediate")
        parser.add_argument("--replay", type=int, default=0, help="whether to use replay strategy")
        parser.add_argument("--neg_cross_kg", type=int, default=0, help="whether to force the negative samples in the opposite KG")

        # --------- MSNEA -----------
        parser.add_argument("--dim", type=int, default=100, help="the hidden size of MSNEA")
        parser.add_argument("--neg_triple_num", type=int, default=1, help="neg triple num")
        parser.add_argument("--use_bert", type=int, default=0)
        parser.add_argument("--use_attr_value", type=int, default=0)
        # parser.add_argument("--learning_rate", type=int, default=0.001)
        # parser.add_argument("--optimizer", type=str, default="Adam")
        # parser.add_argument("--max_epoch", type=int, default=200)

        # parser.add_argument("--save_path", type=str, default="save_pkl", help="save path")

        # ------------ Para ------------
        parser.add_argument('--rank', type=int, default=0, help='rank to dist')
        parser.add_argument('--dist', type=int, default=0, help='whether to dist')
        parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
        parser.add_argument('--world-size', default=3, type=int,
                            help='number of distributed processes')
        parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
        parser.add_argument("--local_rank", default=-1, type=int)

        self.cfg = parser.parse_args()

    def update_train_configs(self):
        # add some constraint for parameters
        # e.g. cannot save and test at the same time
        assert not (self.cfg.save_model and self.cfg.only_test)

        # update some dynamic variable
        self.cfg.data_root = self.data_root

        #new 1
        self.cfg.k = self.cfg.k
        self.cfg.lambda_val = self.cfg.lambda_val
        #end new1


        if self.cfg.use_surface:
            self.cfg.w_name = True
            self.cfg.w_char = True
        else:
            self.cfg.w_name = False
            self.cfg.w_char = False

        if self.cfg.data_choice in ["FBYG15K", "FBDB15K"]:
            self.cfg.use_intermediate = 0
            self.cfg.data_split = "norm"
            
            # ====== 修改：动态响应命令行的 use_surface 参数 ======
            if self.cfg.use_surface == 1:
                self.cfg.inner_view_num = 6  # 凑齐 6 个视图 (图,关系,属性,图,名字,字符)
                self.cfg.w_name = True
                self.cfg.w_char = True
            else:
                self.cfg.inner_view_num = 4
                self.cfg.w_name = False
                self.cfg.w_char = False
            # ====================================================
            
            data_split_name = f"{self.cfg.data_rate}_"

        # if self.cfg.data_choice in ["FBYG15K", "FBDB15K"]:
        #     self.cfg.use_intermediate = 0
        #     self.cfg.data_split = "norm"
        #     self.cfg.inner_view_num = 4
        #     # assert self.cfg.data_rate in [0.2, 0.5, 0.8]
        #     self.cfg.w_name = False
        #     self.cfg.w_char = False
        #     self.cfg.use_surface = 0
        #     data_split_name = f"{self.cfg.data_rate}_"
        else:
            data_split_name = f"{self.cfg.data_split}_"
            if self.cfg.w_name and self.cfg.w_char:
                data_split_name = f"{data_split_name}with_surface_"


        # ⬇️ 就是下面这一行在捣鬼，把它注释掉！
        #self.cfg.exp_id = f"{self.cfg.model_name}_{self.cfg.data_choice}_{data_split_name}{self.cfg.exp_id}"
        self.cfg.data_path = osp.join(self.data_root, self.cfg.data_path)
        self.cfg.dump_path = osp.join(self.cfg.data_path, self.cfg.dump_path)
        if self.cfg.only_test == 1:
            self.save_model = 0
            self.dist = 0

        # --------- MSNEA -----------
        self.cfg.dim = self.cfg.attr_dim

        # --------- MEAformer -----------
        self.cfg.max_position_embeddings = self.cfg.inner_view_num + 1
        assert self.cfg.hidden_size == self.cfg.attr_dim

        # use SOTA param
        if self.cfg.enable_sota:
            if self.cfg.il:
                self.cfg.eval_epoch = max(2, self.cfg.eval_epoch)
                self.cfg.weight_decay = max(0.0005, self.cfg.weight_decay)
                if self.cfg.data_rate > 0.5:
                    self.cfg.weight_decay = max(0.001, self.cfg.weight_decay)
                if self.cfg.data_choice == "DBP15K":
                    if not self.cfg.use_surface:
                        self.cfg.weight_decay = max(0.001, self.cfg.weight_decay)
            else:
                if self.cfg.data_choice == "DBP15K" or "FBYG" in self.cfg.data_choice:
                    self.cfg.epoch = 250
                else:
                    self.cfg.epoch = 500

        return self.cfg
