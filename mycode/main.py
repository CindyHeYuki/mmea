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
        # ====== 新增：提取 PLM 特征并转为 GPU Tensor ======
        self.plm_features = None
        # data_init 通常会把数据字典存为 self.data
        if hasattr(self, 'data') and self.data.get('plm_features') is not None:
            import torch # 确保顶部导入了 torch
            self.plm_features = torch.tensor(self.data['plm_features'], dtype=torch.float32).cuda()
            if self.logger:
                self.logger.info("✅ PLM Features converted to Tensor and moved to CUDA.")
        # ===============================================

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
            # ====== 新增：将离线特征绑定到模型内部 ======
            self.model.plm_features = self.plm_features
            # ===========================================

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
        # ====== 新增：only_test 模式 → 跳过训练循环，直接加载测试 + α 扫描 ======
        if self.args.only_test:
            name = self._save_name_define()
            self.logger.info("\n" + "=" * 60)
            self.logger.info(f"🚀 [ONLY_TEST MODE] 跳过训练，使用已加载的模型")
            self.logger.info(f"🚀 [ONLY_TEST MODE] model_name_save = '{self.args.model_name_save}'")
            self.logger.info(f"🚀 [ONLY_TEST MODE] do_alpha_sweep  = {getattr(self.args, 'do_alpha_sweep', 0)}")
            self.logger.info("=" * 60)

            # 模型在 __init__ → model_choise() 里已经通过 _load_model 加载好 state_dict + C_j
            self.model.eval()

            # 1. 先跑一次标准推理，确认加载成功（数值应该和训练末尾的 best 接近）
            self.test(save_name=f"{name}_only_test_baseline")

            # 2. 如果开启了 α 扫描，直接扫描
            if getattr(self.args, 'do_alpha_sweep', 0) == 1:
                self.logger.info("\n🔍 [ONLY_TEST MODE] 开始 α 扫描...")
                self.alpha_sweep(self.eval_left, self.eval_right)

            return  # ← 关键：直接返回，不进训练循环
        # =========================================================================


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
                
                _tqdm.update(1)
                if self.stage == 1 and self.early_stop_count <= 0:
                    logger.info(f"Early stop in epoch {self.epoch}")
                    break

        name = self._save_name_define()
        if self.best_model_wts is not None:
            self.logger.info("load from the best model before final testing ... ")
            self.model.load_state_dict(self.best_model_wts)
        self.test(save_name=f"{name}_test_ep{self.args.epoch}")
        # ====== 新增：训练结束后触发 α 扫描 ======
        if getattr(self.args, 'do_alpha_sweep', 0) == 1:
            self.logger.info("\n\n🔍 训练结束，开始 α 扫描...")
            test_left = self.eval_left
            test_right = self.eval_right
            self.model.eval()
            self.alpha_sweep(test_left, test_right)
        # ========================================

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



    def eval(self, last_epoch=False, save_name=""):
        test_left = self.eval_left
        test_right = self.eval_right
        self.model.eval()
        self._test(test_left, test_right, last_epoch=last_epoch, save_name=save_name)
    
    
    def alpha_sweep(self, test_left, test_right):
        """
        两阶段 α 扫描：粗扫 → 细扫 → 历史对比
        
        流程:
          1. 读取历史最优 α（如果有），以其为中心做大范围粗扫
          2. 在粗扫 top-1 附近做局部细扫
          3. 打印对比表：历史最优 vs 本次粗扫最优 vs 本次细扫最优
          4. 如果本次最优更好，自动提示更新历史档案
        """
        if self.rank != 0:
            return

        import json
        import csv
        import os

        # ====== 历史档案路径 & 当前配置 key ======
        history_path = osp.join(osp.dirname(osp.abspath(__file__)), 'best_alpha_history.json')
        config_key = f"{self.args.data_choice}_{self.args.data_split}_{self.args.data_rate}_surface{self.args.use_surface}"

        self.logger.info("=" * 80)
        self.logger.info("🔍 开始两阶段 α 扫描 (Two-Stage Alpha Sweep)")
        self.logger.info(f"   当前配置: {config_key}")
        self.logger.info(f"   Checkpoint: {self.args.model_name_save}")
        self.logger.info("=" * 80)

        # ====== 读取历史最优 ======
        history = {}
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                self.logger.info(f"📚 历史档案已加载: {history_path}")
            except Exception as e:
                self.logger.info(f"⚠️  历史档案加载失败（将视为空档案）: {e}")
        else:
            self.logger.info(f"📚 历史档案不存在，将从头开始")

        hist_record = history.get(config_key, None)
        hist_alpha = hist_record['best_alpha'] if hist_record else None

        if hist_alpha:
            self.logger.info(f"🎯 历史最优 α: causal={hist_alpha['causal_alpha']}, "
                            f"csc={hist_alpha['csc_alpha']}, neighbor={hist_alpha['neighbor_alpha']}")
            self.logger.info(f"   历史最优 Hits@1: {hist_record['best_metrics']['hits1']:.4f}")
        else:
            self.logger.info(f"🎯 无历史最优记录，将使用默认粗扫范围")

        # ====== 阶段 0：先评估历史最优 α（如果有）======
        hist_alpha_new_result = None
        if hist_alpha:
            self.logger.info("\n" + "-" * 80)
            self.logger.info("📊 [阶段 0] 评估历史最优 α 在当前模型上的表现...")
            self.logger.info("-" * 80)
            m = self._test(test_left, test_right, last_epoch=False, save_name="",
                          override_causal_alpha=hist_alpha['causal_alpha'],
                          override_csc_alpha=hist_alpha['csc_alpha'],
                          override_neighbor_alpha=hist_alpha['neighbor_alpha'])
            if m is not None:
                hist_alpha_new_result = {
                    'causal_alpha': hist_alpha['causal_alpha'],
                    'csc_alpha': hist_alpha['csc_alpha'],
                    'neighbor_alpha': hist_alpha['neighbor_alpha'],
                    'hits1': float(m['hits1_l2r']),
                    'hits10': float(m['hits10_l2r']),
                    'mrr': float(m['mrr_l2r']),
                }
                self.logger.info(f"   历史 α 本次 Hits@1={hist_alpha_new_result['hits1']:.4f}")

        # ====== 阶段 1：粗扫 ======
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 [阶段 1] 粗扫 (Coarse Grid Search)")
        self.logger.info("=" * 80)

        if hist_alpha:
            # 以历史最优为中心的自适应范围（±0.2，步长 0.1）
            def grid_around(center, radius=0.2, step=0.1, lo=0.0, hi=0.7):
                grid = []
                v = center - radius
                while v <= center + radius + 1e-9:
                    if v >= lo and v <= hi:
                        grid.append(round(v, 3))
                    v += step
                return sorted(set(grid))
            
            coarse_causal = grid_around(hist_alpha['causal_alpha'])
            coarse_csc = grid_around(hist_alpha['csc_alpha'])
            coarse_neighbor = grid_around(hist_alpha['neighbor_alpha'])
        else:
            # 默认大范围粗扫
            coarse_causal = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            coarse_csc = [0.0, 0.05, 0.1, 0.15, 0.2]
            coarse_neighbor = [0.0, 0.2, 0.4, 0.5, 0.6]

        self.logger.info(f"   causal_α 网格 ({len(coarse_causal)} 点): {coarse_causal}")
        self.logger.info(f"   csc_α 网格 ({len(coarse_csc)} 点): {coarse_csc}")
        self.logger.info(f"   neighbor_α 网格 ({len(coarse_neighbor)} 点): {coarse_neighbor}")
        self.logger.info(f"   总组合数: {len(coarse_causal) * len(coarse_csc) * len(coarse_neighbor)}")

        coarse_results = self._run_sweep_grid(test_left, test_right,
                                              coarse_causal, coarse_csc, coarse_neighbor,
                                              tag="粗扫")

        coarse_best = sorted(coarse_results, key=lambda x: x['hits1'], reverse=True)[0]
        self.logger.info(f"\n🥇 粗扫最优: causal={coarse_best['causal_alpha']}, "
                        f"csc={coarse_best['csc_alpha']}, neighbor={coarse_best['neighbor_alpha']} "
                        f"→ Hits@1={coarse_best['hits1']:.4f}")

        # ====== 阶段 2：细扫（粗扫最优点附近）======
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 [阶段 2] 细扫 (Fine Grid Search around coarse best)")
        self.logger.info("=" * 80)

        def fine_grid_around(center, radius=0.075, step=0.025, lo=0.0, hi=0.7):
            grid = []
            v = center - radius
            while v <= center + radius + 1e-9:
                if v >= lo and v <= hi:
                    grid.append(round(v, 4))
                v += step
            return sorted(set(grid))

        fine_causal = fine_grid_around(coarse_best['causal_alpha'])
        fine_csc = fine_grid_around(coarse_best['csc_alpha'])
        fine_neighbor = fine_grid_around(coarse_best['neighbor_alpha'])

        self.logger.info(f"   causal_α 网格 ({len(fine_causal)} 点): {fine_causal}")
        self.logger.info(f"   csc_α 网格 ({len(fine_csc)} 点): {fine_csc}")
        self.logger.info(f"   neighbor_α 网格 ({len(fine_neighbor)} 点): {fine_neighbor}")
        self.logger.info(f"   总组合数: {len(fine_causal) * len(fine_csc) * len(fine_neighbor)}")

        fine_results = self._run_sweep_grid(test_left, test_right,
                                            fine_causal, fine_csc, fine_neighbor,
                                            tag="细扫")

        # ====== 合并所有结果 ======
        all_results = coarse_results + fine_results
        # 去重（同一 α 组合可能在粗扫和细扫都出现）
        seen = set()
        unique_results = []
        for r in all_results:
            key = (round(r['causal_alpha'], 4), round(r['csc_alpha'], 4), round(r['neighbor_alpha'], 4))
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        all_results = unique_results

        results_sorted = sorted(all_results, key=lambda x: x['hits1'], reverse=True)
        final_best = results_sorted[0]

        # ====== 打印最终排行榜（Top 20）======
        self.logger.info("\n" + "=" * 100)
        self.logger.info("🏆 α 扫描最终排行榜 (Top 20, 按 Hits@1 排序)")
        self.logger.info("=" * 100)
        self.logger.info(f"{'排名':<6}{'causal_α':<12}{'csc_α':<10}{'neighbor_α':<14}{'Hits@1':<10}{'Hits@10':<10}{'MRR':<10}")
        self.logger.info("-" * 100)
        for i, r in enumerate(results_sorted[:20]):
            marker = "⭐" if i == 0 else "  "
            self.logger.info(f"{marker} {i+1:<4}{r['causal_alpha']:<12}{r['csc_alpha']:<10}"
                            f"{r['neighbor_alpha']:<14}{r['hits1']:<10.4f}{r['hits10']:<10.4f}{r['mrr']:<10.4f}")
        self.logger.info("=" * 100)

        # ====== 关键对比表 ======
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📋 关键对比表")
        self.logger.info("=" * 80)
        self.logger.info(f"{'来源':<28}{'causal_α':<12}{'csc_α':<10}{'neighbor_α':<14}{'Hits@1':<10}")
        self.logger.info("-" * 80)

        if hist_record:
            hist_h1 = hist_record['best_metrics']['hits1']
            self.logger.info(f"{'历史最优(档案记录)':<25}{hist_alpha['causal_alpha']:<12}"
                            f"{hist_alpha['csc_alpha']:<10}{hist_alpha['neighbor_alpha']:<14}{hist_h1:<10.4f}")

        if hist_alpha_new_result:
            self.logger.info(f"{'历史α(本次模型表现)':<25}{hist_alpha_new_result['causal_alpha']:<12}"
                            f"{hist_alpha_new_result['csc_alpha']:<10}{hist_alpha_new_result['neighbor_alpha']:<14}"
                            f"{hist_alpha_new_result['hits1']:<10.4f}")

        self.logger.info(f"{'本次粗扫最优':<28}{coarse_best['causal_alpha']:<12}"
                        f"{coarse_best['csc_alpha']:<10}{coarse_best['neighbor_alpha']:<14}{coarse_best['hits1']:<10.4f}")
        self.logger.info(f"{'本次最终最优(含细扫)':<25}{final_best['causal_alpha']:<12}"
                        f"{final_best['csc_alpha']:<10}{final_best['neighbor_alpha']:<14}{final_best['hits1']:<10.4f}")
        self.logger.info("=" * 80)

        # ====== 自动更新历史档案（如果本次更优）======
        should_update = False
        if hist_record is None:
            should_update = True
            update_reason = "无历史记录"
        elif final_best['hits1'] > hist_record['best_metrics']['hits1']:
            should_update = True
            update_reason = f"本次({final_best['hits1']:.4f}) > 历史({hist_record['best_metrics']['hits1']:.4f})"
        else:
            update_reason = f"本次({final_best['hits1']:.4f}) ≤ 历史({hist_record['best_metrics']['hits1']:.4f})，保持历史记录"

        self.logger.info(f"\n🔄 历史档案更新决策: {'✅ 更新' if should_update else '❌ 不更新'}")
        self.logger.info(f"   原因: {update_reason}")

        if should_update:
            from datetime import datetime
            history[config_key] = {
                "best_alpha": {
                    "causal_alpha": final_best['causal_alpha'],
                    "csc_alpha": final_best['csc_alpha'],
                    "neighbor_alpha": final_best['neighbor_alpha'],
                },
                "best_metrics": {
                    "hits1": final_best['hits1'],
                    "hits10": final_best['hits10'],
                    "mrr": final_best['mrr'],
                },
                "checkpoint": self.args.model_name_save,
                "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "notes": f"two-stage sweep, {len(coarse_results)} coarse + {len(fine_results)} fine",
            }
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            self.logger.info(f"   ✅ 已写入 {history_path}")

        # ====== 保存本次完整结果 ======
        save_dir = osp.join(self.args.data_path, 'sweep_results')
        os.makedirs(save_dir, exist_ok=True)
        exp_tag = f"{self.args.data_choice}_{self.args.data_split}_{self.args.data_rate}_surface{self.args.use_surface}"

        json_path = osp.join(save_dir, f'sweep_{exp_tag}.json')
        with open(json_path, 'w') as f:
            json.dump(results_sorted, f, indent=2)

        csv_path = osp.join(save_dir, f'sweep_{exp_tag}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['causal_alpha', 'csc_alpha', 'neighbor_alpha', 'hits1', 'hits10', 'mrr'])
            writer.writeheader()
            writer.writerows(results_sorted)

        self.logger.info(f"\n💾 完整结果保存: {json_path}")
        self.logger.info(f"💾 CSV 保存:       {csv_path}")

        # ====== 生成热力图（沿用你之前的函数）======
        self._plot_sweep_heatmaps(all_results, save_dir, exp_tag, final_best)

    def _run_sweep_grid(self, test_left, test_right, causal_grid, csc_grid, neighbor_grid, tag=""):
        """在给定网格上跑扫描，返回结果列表"""
        results = []
        total = len(causal_grid) * len(csc_grid) * len(neighbor_grid)
        counter = 0

        for c_alpha in causal_grid:
            for s_alpha in csc_grid:
                for n_alpha in neighbor_grid:
                    counter += 1
                    if counter % 20 == 0 or counter == 1 or counter == total:
                        self.logger.info(f"  [{tag}] 进度: [{counter}/{total}]")

                    metrics = self._test(test_left, test_right, last_epoch=False, save_name="",
                                        override_causal_alpha=c_alpha,
                                        override_csc_alpha=s_alpha,
                                        override_neighbor_alpha=n_alpha)

                    if metrics is not None:
                        results.append({
                            'causal_alpha': c_alpha,
                            'csc_alpha': s_alpha,
                            'neighbor_alpha': n_alpha,
                            'hits1': float(metrics['hits1_l2r']),
                            'hits10': float(metrics['hits10_l2r']),
                            'mrr': float(metrics['mrr_l2r']),
                        })
        return results

    def _plot_sweep_heatmaps(self, results, save_dir, exp_tag, best):
        """生成三组热力图：固定一个参数，看另外两个参数的影响"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # 无显示器模式
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            self.logger.info("⚠️ matplotlib 未安装，跳过热力图生成")
            return

        # 获取所有唯一值
        causal_vals = sorted(set(r['causal_alpha'] for r in results))
        csc_vals = sorted(set(r['csc_alpha'] for r in results))
        neighbor_vals = sorted(set(r['neighbor_alpha'] for r in results))

        # 构建查找表
        lookup = {}
        for r in results:
            key = (r['causal_alpha'], r['csc_alpha'], r['neighbor_alpha'])
            lookup[key] = r['hits1']

        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        fig.suptitle(f'Alpha Sweep Heatmaps - {exp_tag}\n'
                    f'Best: causal={best["causal_alpha"]}, csc={best["csc_alpha"]}, '
                    f'neighbor={best["neighbor_alpha"]}, Hits@1={best["hits1"]:.4f}',
                    fontsize=14, fontweight='bold')

        # ====== 热力图 1: 固定 neighbor_α = best, 看 causal vs csc ======
        ax = axes[0]
        best_n = best['neighbor_alpha']
        matrix = np.zeros((len(csc_vals), len(causal_vals)))
        for i, s in enumerate(csc_vals):
            for j, c in enumerate(causal_vals):
                matrix[i, j] = lookup.get((c, s, best_n), 0)

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', origin='lower')
        ax.set_xticks(range(len(causal_vals)))
        ax.set_xticklabels([str(v) for v in causal_vals], rotation=45)
        ax.set_yticks(range(len(csc_vals)))
        ax.set_yticklabels([str(v) for v in csc_vals])
        ax.set_xlabel('causal_α')
        ax.set_ylabel('csc_α')
        ax.set_title(f'neighbor_α={best_n} (fixed)')
        # 在每个格子里标数字
        for i in range(len(csc_vals)):
            for j in range(len(causal_vals)):
                val = matrix[i, j]
                if val > 0:
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7,
                        color='white' if val > matrix.max() * 0.7 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # ====== 热力图 2: 固定 csc_α = best, 看 causal vs neighbor ======
        ax = axes[1]
        best_s = best['csc_alpha']
        matrix = np.zeros((len(neighbor_vals), len(causal_vals)))
        for i, n in enumerate(neighbor_vals):
            for j, c in enumerate(causal_vals):
                matrix[i, j] = lookup.get((c, best_s, n), 0)

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', origin='lower')
        ax.set_xticks(range(len(causal_vals)))
        ax.set_xticklabels([str(v) for v in causal_vals], rotation=45)
        ax.set_yticks(range(len(neighbor_vals)))
        ax.set_yticklabels([str(v) for v in neighbor_vals])
        ax.set_xlabel('causal_α')
        ax.set_ylabel('neighbor_α')
        ax.set_title(f'csc_α={best_s} (fixed)')
        for i in range(len(neighbor_vals)):
            for j in range(len(causal_vals)):
                val = matrix[i, j]
                if val > 0:
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7,
                        color='white' if val > matrix.max() * 0.7 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # ====== 热力图 3: 固定 causal_α = best, 看 csc vs neighbor ======
        ax = axes[2]
        best_c = best['causal_alpha']
        matrix = np.zeros((len(neighbor_vals), len(csc_vals)))
        for i, n in enumerate(neighbor_vals):
            for j, s in enumerate(csc_vals):
                matrix[i, j] = lookup.get((best_c, s, n), 0)

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', origin='lower')
        ax.set_xticks(range(len(csc_vals)))
        ax.set_xticklabels([str(v) for v in csc_vals], rotation=45)
        ax.set_yticks(range(len(neighbor_vals)))
        ax.set_yticklabels([str(v) for v in neighbor_vals])
        ax.set_xlabel('csc_α')
        ax.set_ylabel('neighbor_α')
        ax.set_title(f'causal_α={best_c} (fixed)')
        for i in range(len(neighbor_vals)):
            for j in range(len(csc_vals)):
                val = matrix[i, j]
                if val > 0:
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7,
                        color='white' if val > matrix.max() * 0.7 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        fig_path = osp.join(save_dir, f'heatmap_{exp_tag}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"📊 热力图保存到: {fig_path}")


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


    def _test(self, test_left, test_right, last_epoch=False, save_name="", loss=None,
          override_causal_alpha=None, override_csc_alpha=None, override_neighbor_alpha=None):
        with torch.no_grad():
            w_normalized = None
            if self.args.model_name in ["MEAformer"]:
                final_emb, weight_norm = self.model.joint_emb_generat()
                
                # ====== 新增：获取各模态独立 embedding，供推理时融合使用 ======
                need_modal_embs = (getattr(self.args, 'use_causal_bias', 0) == 1) or \
                                (getattr(self.args, 'use_csc', 0) == 1)
                if need_modal_embs:
                    gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, _, _, _ = \
                        self.model.joint_emb_generat(only_joint=False)
                else:
                    gph_emb = img_emb = rel_emb = att_emb = name_emb = char_emb = None
                # ============================================================
            else:
                final_emb = self.model.joint_emb_generat()
                weight_norm = None
                gph_emb = img_emb = rel_emb = att_emb = name_emb = char_emb = None
            final_emb = F.normalize(final_emb)


            # ====== 模块三：推理时反事实融合 ======
            cf_emb = None
            if self.args.use_csc and self.args.model_name in ["MEAformer"] and weight_norm is not None:
                # 获取各模态的独立 embedding
                gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, _, _, _ = \
                    self.model.joint_emb_generat(only_joint=False)
                
                modal_embs = [e for e in [img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb] if e is not None]
                
                if len(modal_embs) >= 2:
                    stacked = torch.stack([F.normalize(e, dim=-1) for e in modal_embs], dim=1)  # [N, M, dim]
                    
                    # 找到每个实体的主导模态，构造去掉主导模态的反事实表征
                    _, dominant_idx = torch.max(weight_norm, dim=1)  # [N]
                    mask = torch.ones_like(weight_norm)
                    mask.scatter_(1, dominant_idx.unsqueeze(1), 0.0)
                    cf_weights = weight_norm * mask
                    cf_weights = cf_weights / (cf_weights.sum(dim=1, keepdim=True) + 1e-8)
                    
                    cf_joint = torch.sum(cf_weights.unsqueeze(2) * stacked, dim=1)  # [N, dim]
                    cf_emb = F.normalize(torch.cat([cf_joint], dim=1))  # [N, dim] 注意这里维度是 dim 不是 M*dim
                    
                    # 拼接事实表征和反事实表征
                    # final_emb 是 [N, M*dim]，cf_emb 是 [N, dim]
                    # 用加权平均的方式融合距离
            # =============================================
        neighbor_distance = None  # 初始化，防止精排代码找不到变量
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
            

        # ==================== 模块四 (B1)：邻居增强距离融合 ====================
        neighbor_distance = None
        if getattr(self.args, 'use_neighbor', 0) == 1 and self.args.model_name in ["MEAformer"]:
            # 1. 获取 adj 矩阵（稀疏的邻接矩阵）
            adj_matrix = self.model.module.adj if self.args.dist else self.model.adj
            
            # 2. 把 adj 转换成稠密形式来算邻居平均（如果实体数太大可能 OOM，需要分批）
            # adj 通常是 normalized 过的稀疏矩阵，行 i 表示实体 i 与所有实体的连接权重
            
            # 用稀疏矩阵直接乘法：neighbor_emb = adj @ final_emb
            # 这等价于：每个实体的 neighbor_emb = 所有邻居 emb 的加权平均（adj 已归一化）
            neighbor_emb = torch.sparse.mm(adj_matrix, final_emb)
            neighbor_emb = F.normalize(neighbor_emb)
            
            # 3. 计算邻居距离矩阵
            neighbor_distance = pairwise_distances(neighbor_emb[test_left], neighbor_emb[test_right])
            
            # 4. 融合
            neighbor_alpha = override_neighbor_alpha if override_neighbor_alpha is not None \
                 else getattr(self.args, 'neighbor_alpha', 0.2)
            distance = (1 - neighbor_alpha) * distance + neighbor_alpha * neighbor_distance
            self.logger.info(f"[Module 4 - Neighbor] Neighbor fusion applied: alpha={neighbor_alpha}")
        # ==================================================================



            # ==================== 模块二：因果置信度感知的推理时距离融合 ====================
        if getattr(self.args, 'use_causal_bias', 0) == 1 and self.args.model_name in ["MEAformer"]:
            # 获取模型里维护的因果置信度 C_j
            causal_Cj = self.model.module.causal_Cj if self.args.dist else self.model.causal_Cj
            
            modal_emb_dict = {'img': img_emb, 'att': att_emb, 'rel': rel_emb, 
                            'gph': gph_emb, 'name': name_emb, 'char': char_emb}
            
            modal_distances = []
            modal_weights = []
            for m_name, m_emb in modal_emb_dict.items():
                if m_emb is None:
                    continue
                Cj = causal_Cj.get(m_name, 0.0)
                # 只使用置信度超过阈值的模态（过滤掉单独没用的模态）
                if Cj < 0.05:
                    continue
                m_emb_norm = F.normalize(m_emb)
                m_dist = pairwise_distances(m_emb_norm[test_left], m_emb_norm[test_right])
                modal_distances.append(m_dist)
                modal_weights.append(Cj)
            
            if len(modal_distances) > 0:
                # 归一化权重
                weights_sum = sum(modal_weights)
                modal_weights = [w / weights_sum for w in modal_weights]
                # 加权融合单模态距离，作为"因果置信距离"
                causal_distance = sum(w * d for w, d in zip(modal_weights, modal_distances))
                # 与 joint 距离融合
                causal_alpha = override_causal_alpha if override_causal_alpha is not None else getattr(self.args, 'causal_lambda', 0.1)
                distance = (1 - causal_alpha) * distance + causal_alpha * causal_distance
                self.logger.info(f"[Module 2] Causal fusion applied: alpha={causal_alpha}, "
                                f"used {len(modal_distances)} modalities")
        # ==============================================================================
        
        # ==================== 模块三：反事实决策一致性融合 ====================
        if getattr(self.args, 'use_csc', 0) == 1 and self.args.model_name in ["MEAformer"] and weight_norm is not None:
            modal_embs_list = [e for e in [img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb] if e is not None]
            M = len(modal_embs_list)
            
            if M >= 2:
                # 构造反事实场景：假设模型无先验注意力偏好（均匀权重融合）
                stacked = torch.stack([F.normalize(e, dim=-1) for e in modal_embs_list], dim=1)  # [N, M, dim]
                uniform_weights = torch.ones(stacked.shape[0], M, device=stacked.device) / M
                cf_joint = torch.sum(uniform_weights.unsqueeze(2) * stacked, dim=1)  # [N, dim]
                cf_joint = F.normalize(cf_joint)
                
                cf_distance = pairwise_distances(cf_joint[test_left], cf_joint[test_right])
                
                # 融合：事实距离为主，反事实距离做辅助验证
                csc_alpha = override_csc_alpha if override_csc_alpha is not None else getattr(self.args, 'csc_lambda_0', 0.1)
                distance = (1 - csc_alpha) * distance + csc_alpha * cf_distance
                self.logger.info(f"[Module 3] Counterfactual fusion applied: alpha={csc_alpha}")
        # ====================================================================
        
        if self.args.csls is True:
            distance = 1 - csls_sim(1 - distance, self.args.csls_k)
        
        if self.args.csls is True:
            distance = 1 - csls_sim(1 - distance, self.args.csls_k)
        
        # ... 后面不变 ...
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
            # ====== 新增：将评估指标写入 TensorBoard ======
            if hasattr(self, 'writer') and self.writer is not None and self.epoch != "Test":
                self.writer.add_scalar('eval/Hits_at_1', acc_l2r[0], self.epoch)
                self.writer.add_scalar('eval/Hits_at_10', acc_l2r[1], self.epoch)
                self.writer.add_scalar('eval/Hits_at_50', acc_l2r[2], self.epoch)
                self.writer.add_scalar('eval/MRR', mrr_l2r, self.epoch)
                self.writer.add_scalar('eval/MR', mean_l2r, self.epoch)
            # =============================================
        if not self.args.only_test and mrr_l2r > max(self.loss_log.acc) and not last_epoch:
            self.logger.info(f"Best model update in Ep {self.epoch}: MRR from [{max(self.loss_log.acc)}] --> [{mrr_l2r}] ... ")
            self.loss_log.update_acc(mrr_l2r)
            self.early_stop_count = self.early_stop_init
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
        # 返回评估指标，方便外部使用
        return {
            'hits1_l2r': acc_l2r[0],
            'hits10_l2r': acc_l2r[1],
            'hits50_l2r': acc_l2r[2],
            'mrr_l2r': mrr_l2r,
            'mr_l2r': mean_l2r,
            'hits1_r2l': acc_r2l[0],
            'hits10_r2l': acc_r2l[1],
            'mrr_r2l': mrr_r2l,
        }


    def _load_model(self, model, model_name=None):
        if model_name is None:
            model_name = self.args.model_name_save
        save_path = osp.join(self.args.data_path, self.args.model_name, 'save')
        save_path = osp.join(save_path, f'{model_name}.pkl')

        # ====== 新增：加载入口打印 ======
        if self.rank == 0:
            self.logger.info("=" * 60)
            self.logger.info(f"📂 [LOAD MODEL] model_name_save = '{model_name}'")
            self.logger.info(f"📂 [LOAD MODEL] target path     = {save_path}")
            self.logger.info(f"📂 [LOAD MODEL] file exists     = {os.path.exists(save_path)}")
        # ================================

        if len(model_name) > 0 and not os.path.exists(save_path):
            print(f"not exists {model_name} !! ")
            pdb.set_trace()
        if (len(model_name) == 0 or not os.path.exists(save_path)) and self.rank == 0:
            if len(model_name) > 0:
                self.logger.info(f"📂 [LOAD MODEL] ⚠️  {model_name}.pkl not exist, falling back to random init")
            else:
                self.logger.info("📂 [LOAD MODEL] 🎲 Random init (no checkpoint specified)")
            self.logger.info("=" * 60)
            model.cuda()
            return model
        if 'Dist' in self.args.model_name:
            model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_path, map_location=self.args.device).items()})
        else:
            model.load_state_dict(torch.load(save_path, map_location=self.args.device))

        model.cuda()
        if self.rank == 0:
            self.logger.info(f"📂 [LOAD MODEL] ✅ model state_dict loaded: {save_path}")
            self.logger.info(f"📂 [LOAD MODEL]    file size             = {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")

            # ====== 新增：加载 C_j ======
            cj_path = save_path.replace('.pkl', '_cj.json')
            if os.path.exists(cj_path):
                try:
                    with open(cj_path, 'r') as f:
                        cj_data = json.load(f)
                    if hasattr(model, 'causal_Cj') and model.causal_Cj is not None:
                        model.causal_Cj.update(cj_data)
                        self.logger.info(f"📂 [LOAD MODEL] ✅ C_j loaded:           {cj_path}")
                        self.logger.info(f"📂 [LOAD MODEL]    C_j content          = {cj_data}")
                    else:
                        self.logger.info(f"📂 [LOAD MODEL] ⚠️  model.causal_Cj not found, C_j file ignored")
                except Exception as e:
                    self.logger.info(f"📂 [LOAD MODEL] ⚠️  C_j load failed: {e}")
            else:
                self.logger.info(f"📂 [LOAD MODEL] ⚠️  C_j file not found: {cj_path}")
                self.logger.info(f"📂 [LOAD MODEL]    (causal module will start from zeros)")
            # =============================

            self.logger.info("=" * 60)

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

        # ====== 新增：保存入口打印 ======
        self.logger.info("=" * 60)
        self.logger.info(f"💾 [SAVE MODEL] save_model flag = {self.args.save_model}")
        self.logger.info(f"💾 [SAVE MODEL] target path     = {save_path}")
        # ================================

        if self.args.save_model:
            torch.save(model.state_dict(), save_path)
            self.logger.info(f"💾 [SAVE MODEL] ✅ model state_dict saved: {save_path}")
            self.logger.info(f"💾 [SAVE MODEL] file size        = {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")

            # ====== 新增：保存 C_j（供推理时因果模块使用）======
            cj_path = save_path.replace('.pkl', '_cj.json')
            if hasattr(model, 'causal_Cj') and model.causal_Cj is not None:
                try:
                    cj_data = {k: float(v) for k, v in model.causal_Cj.items()}
                    with open(cj_path, 'w') as f:
                        json.dump(cj_data, f, indent=2)
                    self.logger.info(f"💾 [SAVE MODEL] ✅ C_j saved:       {cj_path}")
                    self.logger.info(f"💾 [SAVE MODEL]    C_j content    = {cj_data}")
                except Exception as e:
                    self.logger.info(f"💾 [SAVE MODEL] ⚠️  C_j save failed: {e}")
            else:
                self.logger.info(f"💾 [SAVE MODEL] ⚠️  model.causal_Cj not found, skipping C_j save")
            # ==================================================
        else:
            self.logger.info(f"💾 [SAVE MODEL] ⚠️  save_model flag is OFF, model NOT saved!")

        self.logger.info("=" * 60)
        return save_path
    


if __name__ == '__main__':
    cfg = cfg()
    cfg.get_args()
    cfgs = cfg.update_train_configs()

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
        logger.info(f"  [进阶消融模块状态]")

        # 模块一：样本调度
        sch_on = getattr(cfgs, 'use_sample_schedule', 0) == 1
        logger.info(f"  S型样本调度 (Sch)     : {'🟢 开启' if sch_on else '🔴 关闭'}")
        if sch_on:
            logger.info(f"    ├─ k (调度速度)    : {getattr(cfgs, 'k', 'N/A')}")
            logger.info(f"    ├─ λ_struct       : {getattr(cfgs, 'lambda_struct', 'N/A')}  (难度-结构稀疏权重)")
            logger.info(f"    ├─ λ_modal        : {getattr(cfgs, 'lambda_modal', 'N/A')}  (难度-模态不一致权重)")
            logger.info(f"    └─ λ_ambig        : {getattr(cfgs, 'lambda_ambig', 'N/A')}  (难度-名称非独特权重)")

        # 模块二：因果偏置（推理时）
        causal_on = getattr(cfgs, 'use_causal_bias', 0) == 1
        logger.info(f"  因果偏置 (Causal)     : {'🟢 开启 (推理时融合)' if causal_on else '🔴 关闭'}")
        if causal_on:
            logger.info(f"    ├─ causal_lambda  : {getattr(cfgs, 'causal_lambda', 'N/A')}  (推理时距离融合强度)")
            logger.info(f"    ├─ causal_gamma   : {getattr(cfgs, 'causal_gamma', 'N/A')}  (D_j 滑动平均因子)")
            logger.info(f"    ├─ causal_beta    : {getattr(cfgs, 'causal_beta', 'N/A')}  (C_j 历史衰减因子)")
            logger.info(f"    └─ causal_eval_k  : {getattr(cfgs, 'causal_eval_k', 'N/A')}  (C_j 评估间隔 epoch)")

        # 模块三：反事实一致性（推理时）
        csc_on = getattr(cfgs, 'use_csc', 0) == 1
        logger.info(f"  反事实防塌陷 (CSC)    : {'🟢 开启 (推理时融合)' if csc_on else '🔴 关闭'}")
        if csc_on:
            logger.info(f"    └─ csc_lambda_0   : {getattr(cfgs, 'csc_lambda_0', 'N/A')}  (反事实距离融合强度)")

        

        # 模块四：邻居增强
        nbr_on = getattr(cfgs, 'use_neighbor', 0) == 1
        logger.info(f"  邻居增强 (Neighbor)   : {'🟢 开启 (推理时融合)' if nbr_on else '🔴 关闭'}")
        if nbr_on:
            logger.info(f"    └─ neighbor_alpha : {getattr(cfgs, 'neighbor_alpha', 'N/A')}")

        logger.info("------------------------------------------------------------")
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
    # ====== 新增：运行模式总览 ======
    if rank == 0:
        logger.info("=" * 60)
        if cfgs.only_test:
            logger.info(f"🎯 [RUN MODE] ONLY_TEST — 将加载已保存模型")
            logger.info(f"🎯 [RUN MODE] 待加载 checkpoint name = '{cfgs.model_name_save}'")
        else:
            logger.info(f"🎯 [RUN MODE] TRAINING — 将从头/checkpoint 训练")
            logger.info(f"🎯 [RUN MODE] save_model flag = {getattr(cfgs, 'save_model', 0)}")
            logger.info(f"🎯 [RUN MODE] 加载起点 checkpoint = '{cfgs.model_name_save}' (空则随机初始化)")
        logger.info(f"🎯 [RUN MODE] do_alpha_sweep = {getattr(cfgs, 'do_alpha_sweep', 0)}")
        logger.info("=" * 60)
    # =================================
    runner = Runner(cfgs, writer, logger, rank)
    if cfgs.only_test:
        runner.test(last_epoch=False)
        # 新增：only_test 模式下也触发 α 扫描
        # ====== 新增：only_test 模式下也支持 α 扫描 ======
        if getattr(cfgs, 'do_alpha_sweep', 0) == 1 and rank == 0:
            runner.logger.info("\n\n🔍 [only_test] 触发 α 扫描...")
            runner.model.eval()
            if runner.test_set is not None:
                test_left = runner.test_left
                test_right = runner.test_right
            else:
                test_left = runner.eval_left
                test_right = runner.eval_right
            runner.alpha_sweep(test_left, test_right)
    else:
        runner.run()

    # -----  End ----------
    if not cfgs.no_tensorboard and not cfgs.only_test and rank == 0:
        writer.close()
        logger.info("done!")

    if cfgs.dist and not cfgs.only_test:
        dist.barrier()
        dist.destroy_process_group()
