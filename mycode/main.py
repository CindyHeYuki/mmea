import os
import os.path as osp
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
from easydict import EasyDict as edict
from tqdm import tqdm
import pdb
import pprint
import json
import pickle
from collections import defaultdict

from config import cfg
from torchlight import initialize_exp, set_seed, get_dump_path
from src.data import load_data, Collator_base, EADataset
from src.utils import set_optim, Loss_log, pairwise_distances, csls_sim
from model import MEAformer

from src.distributed_utils import init_distributed_mode, dist_pdb, is_main_process, reduce_value, cleanup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import scipy
import gc
import copy
import math 


class Runner:
    def __init__(self, args, writer=None, logger=None, rank=0):
        self.datapath = edict()
        self.datapath.log_dir = get_dump_path(args)
        self.datapath.model_dir = os.path.join(self.datapath.log_dir, 'model')
        self.rank = rank
        self.args = args
        self.writer = writer
        self.logger = logger
        self.scaler = GradScaler()
        self.model_list = []
        set_seed(args.random_seed)
        self.data_init()
        self.model_choise()
        set_seed(args.random_seed)

        if self.args.only_test:
            self.dataloader_init(test_set=self.test_set)
        else:
            self.dataloader_init(train_set=self.train_set, eval_set=self.eval_set, test_set=self.test_set)
            if self.args.dist:
                self.model_sync()
            else:
                self.model_list = [self.model]
            if self.args.il:
                assert self.args.il_start < self.args.epoch
                train_epoch_1_stage = self.args.il_start
            else:
                train_epoch_1_stage = self.args.epoch
            self.optim_init(self.args, total_epoch=train_epoch_1_stage)

    def model_sync(self):
        folder = osp.join(self.args.data_path, "tmp")
        if not os.path.exists(folder):
            os.makedirs(folder)
        checkpoint_path = osp.join(folder, "initial_weights.pt")
        if self.rank == 0:
            torch.save(self.model.state_dict(), checkpoint_path)
        dist.barrier()
        self.model = self._model_sync(self.model, checkpoint_path)

    def _model_sync(self, model, checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.args.device))
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.args.device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.gpu], find_unused_parameters=True)
        self.model_list.append(model)
        model = model.module
        return model

    def model_choise(self):
        assert self.args.model_name in ["EVA", "MCLEA", "MSNEA", "MEAformer"]
        if self.args.model_name == "MEAformer":
            self.model = MEAformer(self.KGs, self.args)

        self.model = self._load_model(self.model, model_name=self.args.model_name_save)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"total params num: {total_params}")

    def optim_init(self, opt, total_step=None, total_epoch=None, accumulation_step=None):
        step_per_epoch = len(self.train_dataloader)
        if total_epoch is not None:
            opt.total_steps = int(step_per_epoch * total_epoch)
        else:
            opt.total_steps = int(step_per_epoch * opt.epoch) if total_step is None else int(total_step)
        opt.warmup_steps = int(opt.total_steps * 0.15)

        if self.rank == 0 and total_step is None:
            self.logger.info(f"warmup_steps: {opt.warmup_steps}")
            self.logger.info(f"total_steps: {opt.total_steps}")
            self.logger.info(f"weight_decay: {opt.weight_decay}")
        freeze_part = []

        self.optimizer, self.scheduler = set_optim(opt, self.model_list, freeze_part, accumulation_step)

    def data_init(self):
        self.KGs, self.non_train, self.train_set, self.eval_set, self.test_set, self.test_ill_ = load_data(self.logger, self.args)
        self.train_ill = self.train_set.data
        self.eval_left = torch.LongTensor(self.eval_set[:, 0].squeeze()).cuda()
        self.eval_right = torch.LongTensor(self.eval_set[:, 1].squeeze()).cuda()
        if self.test_set is not None:
            self.test_left = torch.LongTensor(self.test_ill[:, 0].squeeze()).cuda()
            self.test_right = torch.LongTensor(self.test_ill[:, 1].squeeze()).cuda()

        self.eval_sampler = None
        if self.args.dist and not self.args.only_test:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)
            self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.eval_set)
            if self.test_set is not None:
                self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_set)

    def dataloader_init(self, train_set=None, eval_set=None, test_set=None):
        bs = self.args.batch_size
        collator = Collator_base(self.args)
        if self.args.dist and not self.args.only_test:
            self.args.workers = min([os.cpu_count(), self.args.batch_size, self.args.workers])
            if train_set is not None:
                self.train_dataloader = self._dataloader_dist(train_set, self.train_sampler, bs, collator)
            if test_set is not None:
                self.test_dataloader = self._dataloader_dist(test_set, self.test_sampler, bs, collator)
            if eval_set is not None:
                self.eval_dataloader = self._dataloader_dist(eval_set, self.eval_sampler, bs, collator)
        else:
            self.args.workers = min([os.cpu_count(), self.args.batch_size, self.args.workers])
            if train_set is not None:
                self.train_dataloader = self._dataloader(train_set, bs, collator)
            if test_set is not None:
                self.test_dataloader = self._dataloader(test_set, bs, collator)
            if eval_set is not None:
                self.eval_dataloader = self._dataloader(eval_set, bs, collator)

    def _dataloader_dist(self, train_set, train_sampler, batch_size, collator):
        train_dataloader = DataLoader(
            train_set,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=self.args.workers,
            persistent_workers=True,  # True
            drop_last=True,
            batch_size=batch_size,
            collate_fn=collator
        )
        return train_dataloader

    def _dataloader(self, train_set, batch_size, collator):
        train_dataloader = DataLoader(
            train_set,
            num_workers=self.args.workers,
            persistent_workers=True,  # True
            shuffle=(self.args.only_test == 0),
            # drop_last=(self.args.only_test == 0),
            drop_last=False,
            batch_size=batch_size,
            collate_fn=collator
        )
        return train_dataloader

    def run(self):
        self.loss_log = Loss_log()
        self.curr_loss = 0.
        self.lr = self.args.lr
        self.curr_loss_dic = defaultdict(float)
        self.weight = [1, 1, 1, 1, 1, 1]
        self.loss_weight = [1, 1]
        self.loss_item = 99999.
        self.step = 1
        self.epoch = 0
        self.new_links = []
        self.best_model_wts = None

        self.best_mrr = 0

        self.early_stop_init = 1000
        self.early_stop_count = self.early_stop_init
        self.stage = 0

        with tqdm(total=self.args.epoch) as _tqdm:
            for i in range(self.args.epoch):
                # _tqdm.set_description(f'Train | epoch {i} Loss {self.loss_log.get_loss():.5f} Acc {self.loss_log.get_acc()*100:.3f}%')
                if self.args.dist and not self.args.only_test:
                    self.train_sampler.set_epoch(i)
                # -------------------------------
                self.epoch = i
                if self.args.il and (self.epoch == self.args.il_start and self.stage == 0) or (self.early_stop_count <= 0 and self.epoch <= self.args.il_start):
                    if self.early_stop_count <= 0:
                        logger.info(f"Early stop in epoch {self.epoch}... Begin iteration....")
                    self.stage = 1
                    self.early_stop_init = 2000
                    self.early_stop_count = self.early_stop_init

                    self.eval_epoch = 1

                    self.step = 1
                    self.args.lr = self.args.lr / 5
                    self.optim_init(self.args, total_epoch=(self.args.epoch - self.args.il_start) * 3)
                    if self.best_model_wts is not None:
                        self.logger.info("load from the best model before IL... ")
                        self.model.load_state_dict(self.best_model_wts)
                    name = self._save_name_define()
                    self.test(save_name=f"{name}_test_ep{self.args.epoch}_no_iter")
                    if self.rank == 0:
                        if not self.args.only_test and self.args.save_model:
                            self._save_model(self.model, input_name=f"{name}_non_iter")

                if self.stage == 1 and (self.epoch + 1) % self.args.semi_learn_step == 0 and self.args.il:
                    self.il_for_ea()

                if self.stage == 1 and (self.epoch + 1) % (self.args.semi_learn_step * 10) == 0 and len(self.new_links) != 0 and self.args.il:
                    self.il_for_data_ref()

                self.train(_tqdm)
                self.loss_log.update(self.curr_loss)
                self.loss_item = self.loss_log.get_loss()
                _tqdm.set_description(f'Train | Ep [{self.epoch}/{self.args.epoch}] Step [{self.step}/{self.args.total_steps}] LR [{self.lr:.5f}] Loss {self.loss_log.get_loss():.5f} ')
                self.update_loss_log()
                if (i + 1) % self.args.eval_epoch == 0:
                    self.eval()

                # ====== 新增：定期评估 C_j ======
                if self.args.use_causal_bias and (i + 1) % self.args.causal_eval_k == 0:
                    self.evaluate_Cj()  # 调用真实评估！
                # ===============================
                # # ====== 在这里紧接着插入新增的定期评估 C_j 代码 ======
                # if self.args.use_causal_bias and (i + 1) % self.args.causal_eval_k == 0:
                #     # 提示：由于分别评估单个模态需要阻断其他模态，为简化代码并提升训练效率，
                #     # 这里你可以选择传入上一轮的独立指标，或者调用一个单独的评估函数 self.eval_single_modal()
                #     # 假定返回的是各个模态的 hits@1 字典，例如:
                #     # mock_hits_dict = {'img': 0.6, 'att': 0.7, 'rel': 0.5, 'gph': 0.8, 'name': 0.9, 'char': 0.85}
                #     # self.model.module.update_Cj(mock_hits_dict) if self.args.dist else self.model.update_Cj(mock_hits_dict)
                #     pass
                # # =======================================================
                
                _tqdm.update(1)
                if self.stage == 1 and self.early_stop_count <= 0:
                    logger.info(f"Early stop in epoch {self.epoch}")
                    break

        name = self._save_name_define()
        if self.best_model_wts is not None:
            self.logger.info("load from the best model before final testing ... ")
            self.model.load_state_dict(self.best_model_wts)
        self.test(save_name=f"{name}_test_ep{self.args.epoch}")

        if self.rank == 0:
            self.logger.info(f"min loss {self.loss_log.get_min_loss()}")
            if not self.args.only_test and self.args.save_model:
                self._save_model(self.model, input_name=name)

    def il_for_ea(self):
        with torch.no_grad():
            if self.args.model_name in ["MEAformer"]:
                final_emb, weight_norm = self.model.joint_emb_generat()
            else:
                final_emb = self.model.joint_emb_generat()
            final_emb = F.normalize(final_emb)
            self.new_links = self.model.Iter_new_links(self.epoch, self.non_train["left"], final_emb, self.non_train["right"], new_links=self.new_links)
            if (self.epoch + 1) % (self.args.semi_learn_step * 5) == 0:
                self.logger.info(f"[epoch {self.epoch}] #links in candidate set: {len(self.new_links)}")

    def il_for_data_ref(self):
        self.non_train["left"], self.non_train["right"], self.train_ill, self.new_links = self.model.data_refresh(
            self.logger, self.train_ill, self.test_ill_, self.non_train["left"], self.non_train["right"], new_links=self.new_links)
        set_seed(self.args.random_seed)
        self.train_set = EADataset(self.train_ill)
        self.dataloader_init(train_set=self.train_set)
        # one time train

    def _save_name_define(self):
        prefix = ""
        if self.args.dist:
            prefix = f"dist_{prefix}"
        if self.args.il:
            prefix = f"il{self.args.epoch-self.args.il_start}_b{self.args.il_start}_{prefix}"
        name = f'{self.args.exp_id}_{prefix}'
        return name

    def train(self, _tqdm):
        self.model.train()
        curr_loss = 0.
        self.loss_log.acc_init()
        accumulation_steps = self.args.accumulation_steps
        # torch.cuda.empty_cache()

        # ====== 新增样本调度逻辑 ======
        # 计算当前轮次的阈值 φ(e; E, k)
        threshold = 1 - math.exp(-self.args.k * self.epoch / self.args.epoch)


        for batch_dict in self.train_dataloader:
            # 无论如何，保证传入完整 batch，稳定模型的心智！
            loss, output = self.model(batch_dict, epoch=self.epoch, total_epochs=self.args.epoch)


            #loss, output = self.model(batch)
            loss = loss / accumulation_steps
            self.scaler.scale(loss).backward()
            if self.args.dist:
                loss = reduce_value(loss, average=True)
            self.step += 1
            if not self.args.dist or is_main_process():
                curr_loss += loss.item()
                self.output_statistic(loss, output)

            if self.step % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                for model in self.model_list:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                skip_lr_sched = (scale > self.scaler.get_scale())
                if not skip_lr_sched:
                    self.scheduler.step()

                if not self.args.dist or is_main_process():
                    self.lr = self.scheduler.get_last_lr()[-1]
                    self.writer.add_scalars("lr", {"lr": self.lr}, self.step)
                for model in self.model_list:
                    model.zero_grad(set_to_none=True)

            if self.args.dist:
                torch.cuda.synchronize(self.args.device)

        return curr_loss

    def output_statistic(self, loss, output):
        self.curr_loss += loss.item()
        if output is None:
            return
        for key in output['loss_dic'].keys():
            self.curr_loss_dic[key] += output['loss_dic'][key]
        if 'weight' in output and output['weight'] is not None:
            self.weight = output['weight']
        if 'loss_weight' in output and output['loss_weight'] is not None:
            self.loss_weight = output['loss_weight']
        # 新增接住因果参数
        if 'causal_bias' in output:
            self.causal_bias_log = output['causal_bias']
            self.causal_Cj_log = output['causal_Cj']


    def update_loss_log(self):
        vis_dict = {"train_loss": self.curr_loss}
        vis_dict.update(self.curr_loss_dic)
        self.writer.add_scalars("loss", vis_dict, self.step)

        weight_str = "" # 记录权重的字符串
        if self.weight is not None:
            weight_dic = {}
            weight_dic["img"] = self.weight[0]
            weight_dic["attr"] = self.weight[1]
            weight_dic["rel"] = self.weight[2]
            weight_dic["graph"] = self.weight[3]
            if len(self.weight) >= 6:
                weight_dic["name"] = self.weight[4]
                weight_dic["char"] = self.weight[5]
                weight_str = f"W:[img:{self.weight[0]:.3f}, att:{self.weight[1]:.3f}, rel:{self.weight[2]:.3f}, gph:{self.weight[3]:.3f}, name:{self.weight[4]:.3f}, char:{self.weight[5]:.3f}]"
            else:
                weight_str = f"W:[img:{self.weight[0]:.3f}, att:{self.weight[1]:.3f}, rel:{self.weight[2]:.3f}, gph:{self.weight[3]:.3f}]"
            self.writer.add_scalars("modal_weight", weight_dic, self.step)

        if self.loss_weight is not None and self.loss_weight != [1, 1]:
            weight_dic = {}
            weight_dic["mask"] = 1 / (self.loss_weight[0]**2)
            weight_dic["kpi"] = 1 / (self.loss_weight[1]**2)
            self.writer.add_scalars("loss_weight", weight_dic, self.step)

        if hasattr(self, 'causal_bias_log') and self.causal_bias_log is not None:
            self.writer.add_scalars("Causal/Bias_Value", self.causal_bias_log, self.step)
        if hasattr(self, 'causal_Cj_log') and self.causal_Cj_log is not None:
            self.writer.add_scalars("Causal/Inherent_Confidence_Cj", self.causal_Cj_log, self.step)


        # ========================================================================
        # 🚀 核心新增：将 Loss 细分项和模态权重，在每个 Epoch 结束时打印
        # ========================================================================
        # 移除 step % 50 的条件，直接无条件打印（因为这个函数本身就是在 Epoch 结束时调用的）
        
        # 1. 拼接所有的 Loss
        loss_str = f"Loss:[Tot:{self.curr_loss:.3f}"
        for k, v in self.curr_loss_dic.items():
            loss_str += f", {k}:{v:.4f}"
        loss_str += "]"
        
        # 2. 拼接因果参数 (如果你开了 Causal 模块的话)
        causal_str = ""
        if hasattr(self, 'causal_bias_log') and self.causal_bias_log is not None:
            causal_str = f" | Bias: { {k: round(v, 4) for k,v in self.causal_bias_log.items()} }"

        # 3. 写入 logger
        self.logger.info(f"Ep [{self.epoch}/{self.args.epoch}] Step [{self.step}] | {loss_str} | {weight_str}{causal_str}")
        # ========================================================================


        self.curr_loss = 0.
        for key in self.curr_loss_dic:
            self.curr_loss_dic[key] = 0.

        progress = self.epoch / self.args.epoch
        threshold = 1.0 / (1.0 + math.exp(-self.args.k * (progress - 0.5)))
        self.writer.add_scalar('scheduler/threshold', threshold, self.step) 

        if hasattr(self, 'batch_difficulties'):
            avg_difficulty = torch.mean(self.batch_difficulties).item()
            self.writer.add_scalar("scheduler/avg_difficulty", avg_difficulty, self.step)
    # def update_loss_log(self):
    #     vis_dict = {"train_loss": self.curr_loss}
    #     vis_dict.update(self.curr_loss_dic)
    #     self.writer.add_scalars("loss", vis_dict, self.step)

    #     if self.weight is not None:
    #         weight_dic = {}
    #         weight_dic["img"] = self.weight[0]
    #         weight_dic["attr"] = self.weight[1]
    #         weight_dic["rel"] = self.weight[2]
    #         weight_dic["graph"] = self.weight[3]
    #         if self.args.w_name or self.args.w_char:
    #             weight_dic["name"] = self.weight[4]
    #             weight_dic["char"] = self.weight[5]
    #         self.writer.add_scalars("modal_weight", weight_dic, self.step)

    #     if self.loss_weight is not None and self.loss_weight != [1, 1]:
    #         weight_dic = {}
    #         weight_dic["mask"] = 1 / (self.loss_weight[0]**2)
    #         weight_dic["kpi"] = 1 / (self.loss_weight[1]**2)
    #         self.writer.add_scalars("loss_weight", weight_dic, self.step)

    #     # ====== 新增：把 Causal Bias 和 C_j 画进 TensorBoard ======
    #     if hasattr(self, 'causal_bias_log') and self.causal_bias_log is not None:
    #         self.writer.add_scalars("Causal/Bias_Value", self.causal_bias_log, self.step)
    #     if hasattr(self, 'causal_Cj_log') and self.causal_Cj_log is not None:
    #         self.writer.add_scalars("Causal/Inherent_Confidence_Cj", self.causal_Cj_log, self.step)
    #     # ==========================================================

    #     self.curr_loss = 0.
    #     for key in self.curr_loss_dic:
    #         self.curr_loss_dic[key] = 0.

    #     # ====== 新增调度阈值监控 ======
    #     progress = self.epoch / self.args.epoch
    #     threshold = 1.0 / (1.0 + math.exp(-self.args.k * (progress - 0.5)))
    #     self.writer.add_scalar('scheduler/threshold', threshold, self.step) # global_step 或你自己的步数变量

    #     # threshold = 1 - math.exp(-self.args.k * self.epoch / self.args.epoch)
    #     # self.writer.add_scalar("scheduler/threshold", threshold, self.step)

    #     # ====== 新增样本难度分布监控 ======
    #     if hasattr(self, 'batch_difficulties'):
    #         avg_difficulty = torch.mean(self.batch_difficulties).item()
    #         self.writer.add_scalar("scheduler/avg_difficulty", avg_difficulty, self.step)



    def eval(self, last_epoch=False, save_name=""):
        test_left = self.eval_left
        test_right = self.eval_right
        self.model.eval()
        self._test(test_left, test_right, last_epoch=last_epoch, save_name=save_name)


    def evaluate_Cj(self):
        """极其高效的单模态独立评估，用于计算 C_j"""
        self.model.eval()
        with torch.no_grad():
            # 1. 拿到还没融合的独立特征
            if self.args.model_name in ["MEAformer"]:
                gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, _, _, _ = self.model.joint_emb_generat(only_joint=False)
            else:
                return # 仅支持 MEAformer
                
            embs_dict = {'img': img_emb, 'att': att_emb, 'rel': rel_emb, 'gph': gph_emb, 'name': name_emb, 'char': char_emb}
            test_left = self.eval_left
            test_right = self.eval_right
            
            hits1_dict = {}
            for m, emb in embs_dict.items():
                if emb is None:
                    hits1_dict[m] = 0.0
                    continue
                
                emb = F.normalize(emb)
                # 计算 L2 距离 (为了速度，这里用最简单的矩阵运算)
                distance = pairwise_distances(emb[test_left], emb[test_right])
                
                # 统计 Hits@1
                preds = torch.argmin(distance, dim=1)
                labels = torch.arange(len(test_left)).to(preds.device)
                hits1 = (preds == labels).float().mean().item()
                hits1_dict[m] = hits1
                
        # 2. 更新模型中的 C_j
        if self.args.dist:
            self.model.module.update_Cj(hits1_dict)
        else:
            self.model.update_Cj(hits1_dict)
            
        self.logger.info(f"[Causal Eval] Ep {self.epoch} | 独立模态 Hits@1: {hits1_dict}")
        self.model.train() # 切回训练模式
    

    # one time test
    def test(self, save_name="", last_epoch=True):
        if self.test_set is None:
            test_left = self.eval_left
            test_right = self.eval_right
        else:
            test_left = self.test_left
            test_right = self.test_right
        self.model.eval()
        self.logger.info(" --------------------- Test result --------------------- ")
        self._test(test_left, test_right, last_epoch=last_epoch, save_name=save_name)

    def _test(self, test_left, test_right, last_epoch=False, save_name="", loss=None):
        with torch.no_grad():
            w_normalized = None
            if self.args.model_name in ["MEAformer"]:
                final_emb, weight_norm = self.model.joint_emb_generat()
            else:
                final_emb = self.model.joint_emb_generat()
                weight_norm = None
            final_emb = F.normalize(final_emb)

        # pdb.set_trace()
        top_k = [1, 10, 50]
        acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
        acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
        test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.
        if self.args.distance == 2:
            distance = pairwise_distances(final_emb[test_left], final_emb[test_right])
        elif self.args.distance == 1:
            distance = torch.FloatTensor(scipy.spatial.distance.cdist(
                final_emb[test_left].cpu().data.numpy(),
                final_emb[test_right].cpu().data.numpy(), metric="cityblock"))
        if self.args.csls is True:
            distance = 1 - csls_sim(1 - distance, self.args.csls_k)

        if last_epoch:
            to_write = []
            test_left_np = test_left.cpu().numpy()
            test_right_np = test_right.cpu().numpy()
            to_write.append(["idx", "rank", "query_id", "gt_id", "ret1", "ret2", "ret3", "v1", "v2", "v3"])
        for idx in range(test_left.shape[0]):
            values, indices = torch.sort(distance[idx, :], descending=False)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
            mean_l2r += (rank + 1)
            mrr_l2r += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_l2r[i] += 1
            if last_epoch:
                indices = indices.cpu().numpy()
                to_write.append([idx, rank, test_left_np[idx], test_right_np[idx], test_right_np[indices[0]], test_right_np[indices[1]],
                                 test_right_np[indices[2]], round(values[0].item(), 4), round(values[1].item(), 4), round(values[2].item(), 4)])
        if last_epoch:
            import csv
            if save_name == "":
                save_name = self.args.model_name
            save_pred_path = osp.join(self.args.data_path, self.args.model_name, f"{save_name}_pred")
            os.makedirs(save_pred_path, exist_ok=True)
            with open(osp.join(save_pred_path, f"{self.args.model_name}_{self.args.data_choice}_{self.args.data_split}_{self.args.data_rate}_ep{self.args.il_start}_pred.txt"), "w") as f:
                wr = csv.writer(f, dialect='excel')
                wr.writerows(to_write)
            if w_normalized is not None:
                with open(osp.join(save_pred_path, f"{self.args.model_name}_{self.args.data_choice}_{self.args.data_split}_{self.args.data_rate}_ep{self.args.il_start}_wight.json"), "w") as fp:
                    json.dump(w_normalized.cpu().tolist(), fp)
            if weight_norm is not None:
                wight_dic = {"all": weight_norm.cpu(), "left": weight_norm[test_left].cpu(), "right": weight_norm[test_right].cpu()}
                with open(osp.join(save_pred_path, f"{self.args.model_name}_{self.args.data_choice}_{self.args.data_split}_{self.args.data_rate}_ep{self.args.il_start}_wight_dic.pkl"), "wb") as fp:
                    pickle.dump(wight_dic, fp)

        for idx in range(test_right.shape[0]):
            _, indices = torch.sort(distance[:, idx], descending=False)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
            mean_r2l += (rank + 1)
            mrr_r2l += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_r2l[i] += 1
        mean_l2r /= test_left.size(0)
        mean_r2l /= test_right.size(0)
        mrr_l2r /= test_left.size(0)
        mrr_r2l /= test_right.size(0)
        for i in range(len(top_k)):
            acc_l2r[i] = round(acc_l2r[i] / test_left.size(0), 4)
            acc_r2l[i] = round(acc_r2l[i] / test_right.size(0), 4)
        gc.collect()
        if not self.args.only_test:
            Loss_out = f", Loss = {self.loss_item:.4f}"
        else:
            Loss_out = ""
            self.epoch = "Test"
            self.early_stop_count = 1

        if self.rank == 0:
            self.logger.info(f"Ep {self.epoch} | l2r: acc of top {top_k} = {acc_l2r}, mr = {mean_l2r:.3f}, mrr = {mrr_l2r:.3f}{Loss_out}")
            self.logger.info(f"Ep {self.epoch} | r2l: acc of top {top_k} = {acc_r2l}, mr = {mean_r2l:.3f}, mrr = {mrr_r2l:.3f}{Loss_out}")
            self.early_stop_count -= 1
        if not self.args.only_test and mrr_l2r > max(self.loss_log.acc) and not last_epoch:
            self.logger.info(f"Best model update in Ep {self.epoch}: MRR from [{max(self.loss_log.acc)}] --> [{mrr_l2r}] ... ")
            self.loss_log.update_acc(mrr_l2r)
            self.early_stop_count = self.early_stop_init
            self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def _load_model(self, model, model_name=None):
        if model_name is None:
            model_name = self.args.model_name_save
        save_path = osp.join(self.args.data_path, self.args.model_name, 'save')
        save_path = osp.join(save_path, f'{model_name}.pkl')
        if len(model_name) > 0 and not os.path.exists(save_path):
            print(f"not exists {model_name} !! ")
            pdb.set_trace()
        if (len(model_name) == 0 or not os.path.exists(save_path)) and self.rank == 0:
            if len(model_name) > 0:
                self.logger.info(f"{model_name}.pkl not exist!!")
            else:
                self.logger.info("Random init...")
            model.cuda()
            return model
        if 'Dist' in self.args.model_name:
            model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_path, map_location=self.args.device).items()})
        else:
            model.load_state_dict(torch.load(save_path, map_location=self.args.device))

        model.cuda()
        if self.rank == 0:
            self.logger.info(f"loading model [{model_name}.pkl] done!")

        return model

    def _save_model(self, model, input_name=""):

        model_name = self.args.model_name

        save_path = osp.join(self.args.data_path, model_name, 'save')
        os.makedirs(save_path, exist_ok=True)

        if input_name == "":
            input_name = self._save_name_define()
        save_path = osp.join(save_path, f'{input_name}.pkl')

        if model is None:
            return
        if self.args.save_model:
            torch.save(model.state_dict(), save_path)

            self.logger.info(f"saving [{save_path}] done!")

        return save_path


if __name__ == '__main__':
    cfg = cfg()
    cfg.get_args()
    cfgs = cfg.update_train_configs()



    # ##################################################
    # # 添加参数检查
    # ##################################################
    # print("\n" + "="*60)
    # print("参数检查:")
    # print(f"  GPU设备: {cfgs.gpu}")
    # print(f"  Epoch数: {cfgs.epoch} ")
    # print(f"  Batch大小: {cfgs.batch_size} ")
    # print(f"  数据集: {cfgs.data_choice}")
    # print(f"  数据分割: {cfgs.data_split}")
    # print(f"  数据比例: {cfgs.data_rate}")
    # print(f"  是否使用表面特征: {cfgs.use_surface}")
    # print(f"  实验名称: {cfgs.exp_name}")
    # print(f"  实验ID: {cfgs.exp_id}")

    # # ====== 建议加上的消融实验参数 ======
    # print("-" * 30)
    # print("  [消融实验模块状态]")
    # print(f"  模块一 (样本调度): {'开启' if cfgs.use_sample_schedule else '关闭'} (k={cfgs.k})")
    # print(f"  模块二 (因果加权): {'开启' if cfgs.use_causal_bias else '关闭'} (lambda={cfgs.causal_lambda})")
    # print(f"  模块三 (反事实Loss): {'开启' if cfgs.use_csc else '关闭'} (lambda_0={cfgs.csc_lambda_0})")
    # # ==================================

    # # ====== 新增：PLM 状态打印 ======
    # print("-" * 30)
    # print("  [预训练语言模型 (PLM) 状态]")
    
    # # 真实的开启条件：不仅要求 use_plm=1，而且文本特征(w_name)必须是存活的
    # is_plm_active = hasattr(cfgs, 'use_plm') and cfgs.use_plm == 1 and getattr(cfgs, 'w_name', False)

    # if is_plm_active:
    #     print("  状态: 🟢 开启")
    #     print(f"  模型名称: {cfgs.plm_name}")
    #     print(f"  冻结参数 (防OOM): {'是' if cfgs.freeze_plm == 1 else '否'}")
    #     print(f"  截断长度: {cfgs.plm_max_len}")
    #     print(f"  输出维度: {cfgs.plm_hidden_dim}")
    # else:
    #     if hasattr(cfgs, 'use_plm') and cfgs.use_plm == 1:
    #         print("  状态: 🔴 关闭 (被系统拦截：未开启 --use_surface 1 或 数据集不支持文本)")
    #     else:
    #         print("  状态: 🔴 关闭 (退回使用默认GloVe或无文本模式)")
    # # ===============================


    # # ====== 新增：在参数单里加上 TensorBoard 相对目录 ======
    # print("-" * 30)
    # print(f"  📊 TensorBoard 实验主目录: dump/{cfgs.exp_id}/tensorboard/")
    # # ======================================================

    # print("="*60 + "\n")
    



    set_seed(cfgs.random_seed)
    # -----  Init ----------
    if cfgs.dist and not cfgs.only_test:
        init_distributed_mode(args=cfgs)
    else:
        torch.multiprocessing.set_sharing_strategy('file_system')
    rank = cfgs.rank
    # pprint.pprint(cfgs)

    writer, logger = None, None
    if rank == 0:
        logger = initialize_exp(cfgs)
        logger_path = get_dump_path(cfgs)
        # ====================================================================
        # 🚀 豪华版实验状态仪表盘 (打印到控制台和 Log 文件)
        # ====================================================================
        logger.info("============================================================")
        logger.info("🚀 [MEAformer 核心实验配置与开关状态] 🚀")
        logger.info("------------------------------------------------------------")
        logger.info(f"  📊 数据集: {cfgs.data_choice} (Split: {cfgs.data_split}, Rate: {cfgs.data_rate})")
        logger.info(f"  💻 GPU设备: {cfgs.gpu}  |  🔥 Epoch: {cfgs.epoch}  |  📦 Batch: {cfgs.batch_size}")
        logger.info("------------------------------------------------------------")
        
        # 1. PLM 状态检查
        use_plm = getattr(cfgs, 'use_plm', 0)
        logger.info(f"  [多模态 PLM 语义嵌入状态]")
        logger.info(f"  总开关 (use_plm)      : {'🟢 开启' if use_plm else '🔴 关闭'}")
        
        if use_plm:
            embed_name = getattr(cfgs, 'plm_embed_name', 0)
            embed_rel = getattr(cfgs, 'plm_embed_rel', 0)
            embed_attr = getattr(cfgs, 'plm_embed_attr', 0)
            
            logger.info(f"    ├─ 实体名称 (name)  : {'✅ 开启' if embed_name else '❌ 关闭'}")
            logger.info(f"    ├─ 关系文本 (rel)   : {'✅ 开启' if embed_rel else '❌ 关闭'}")
            logger.info(f"    ├─ 属性文本 (attr)  : {'✅ 开启' if embed_attr else '❌ 关闭'}")
            logger.info(f"  模型名称: {getattr(cfgs, 'plm_name', 'None')}")
            logger.info(f"  参数冻结 (freeze)     : {'❄️ 是 (防OOM)' if getattr(cfgs, 'freeze_plm', 1) else '🔥 否 (全局微调)'}")
        
        logger.info("------------------------------------------------------------")
        
        # 2. 消融模块状态检查
        logger.info("  [进阶消融模块状态]")
        logger.info(f"  S型样本调度 (Sch)     : {'🟢 开启' if getattr(cfgs, 'use_sample_schedule', 0) else '🔴 关闭'}")
        logger.info(f"  因果偏置 (Causal)     : {'🟢 开启' if getattr(cfgs, 'use_causal_bias', 0) else '🔴 关闭'}")
        logger.info(f"  反事实防塌陷 (CSC)    : {'🟢 开启' if getattr(cfgs, 'use_csc', 0) else '🔴 关闭'}")
        logger.info("============================================================")
        # ====================================================================

        
        cfgs.time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        comment = f'bath_size={cfgs.batch_size} exp_id={cfgs.exp_id}'
        if not cfgs.no_tensorboard and not cfgs.only_test:
            writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard', cfgs.time_stamp), comment=comment)
            # ====== 新增：高亮打印 TensorBoard 路径 ======
            print("\n" + "🚀"*20)
            print(f"✅ 本次实验的 TensorBoard 绝对路径为:\n  {writer.log_dir}")
            print("  运行命令: tensorboard --logdir " + writer.log_dir + " --port 6006")
            print("🚀"*20 + "\n")
            # ============================================

    cfgs.device = torch.device(cfgs.device)

    # print("print c to continue...")
    # -----  Begin ----------
    torch.cuda.set_device(cfgs.gpu)
    runner = Runner(cfgs, writer, logger, rank)
    if cfgs.only_test:
        runner.test(last_epoch=False)
    else:
        runner.run()

    # -----  End ----------
    if not cfgs.no_tensorboard and not cfgs.only_test and rank == 0:
        writer.close()
        logger.info("done!")

    if cfgs.dist and not cfgs.only_test:
        dist.barrier()
        dist.destroy_process_group()
