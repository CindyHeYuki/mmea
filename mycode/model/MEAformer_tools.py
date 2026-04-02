
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math

from transformers.activations import ACT2FN
from transformers.pytorch_utils import apply_chunking_to_forward

from .layers import ProjectionHead
from .Tool_model import GAT, GCN
import pdb





class MformerFusion(nn.Module):
    def __init__(self, args, modal_num, with_weight=1):
        super().__init__()
        self.args = args
        self.modal_num = modal_num
        self.fusion_layer = nn.ModuleList([BertLayer(args) for _ in range(args.num_hidden_layers)])
        # self.type_embedding = nn.Embedding(args.inner_view_num, args.hidden_size)
        self.type_id = torch.tensor([0, 1, 2, 3, 4, 5]).cuda()

    def forward(self, embs,causal_bias=None):
        # 记录有效模态的索引，保证 causal_bias 和实际传入的模态对齐
        valid_indices = [idx for idx in range(len(embs)) if embs[idx] is not None]
        embs = [embs[idx] for idx in valid_indices]
        # embs = [embs[idx] for idx in range(len(embs)) if embs[idx] is not None]
        modal_num = len(embs)
        
        # 过滤掉 None 模态对应的 bias
        if causal_bias is not None:
            causal_bias = causal_bias[valid_indices]
        

        hidden_states = torch.stack(embs, dim=1)
        bs = hidden_states.shape[0]
        for i, layer_module in enumerate(self.fusion_layer):
            # 将 causal_bias 传给每一层
            layer_outputs = layer_module(hidden_states, output_attentions=True, causal_bias=causal_bias)
            hidden_states = layer_outputs[0]
            
            # layer_outputs = layer_module(hidden_states, output_attentions=True)
            # hidden_states = layer_outputs[0]
        # torch.Size([30355, 5, 4, 4])
        # attention_pro = layer_outputs[1]
        # torch.Size([30355, 4, 4])
        attention_pro = torch.sum(layer_outputs[1], dim=-3)
        attention_pro_comb = torch.sum(attention_pro, dim=-2) / math.sqrt(modal_num * self.args.num_attention_heads)
        weight_norm = F.softmax(attention_pro_comb, dim=-1)
        embs = [weight_norm[:, idx].unsqueeze(1) * F.normalize(embs[idx]) for idx in range(modal_num)]
        joint_emb = torch.cat(embs, dim=1)

        return joint_emb, hidden_states, weight_norm


class MultiModalEncoder(nn.Module):
    """
    entity embedding: (ent_num, input_dim)
    gcn layer: n_units

    """

    def __init__(self, args,
                 ent_num,
                 img_feature_dim,
                 char_feature_dim=None,
                 use_project_head=False,
                 attr_input_dim=1000):
        super(MultiModalEncoder, self).__init__()

        self.args = args
        attr_dim = self.args.attr_dim
        img_dim = self.args.img_dim
        name_dim = self.args.name_dim
        char_dim = self.args.char_dim
        dropout = self.args.dropout
        self.ENT_NUM = ent_num
        self.use_project_head = use_project_head

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

        #########################
        # Entity Embedding
        #########################
        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        self.entity_emb.requires_grad = True

        #########################
        # Modal Encoder
        #########################

        self.rel_fc = nn.Linear(1000, attr_dim)
        self.att_fc = nn.Linear(attr_input_dim, attr_dim)
        self.img_fc = nn.Linear(img_feature_dim, img_dim)
        self.name_fc = nn.Linear(300, char_dim)
        self.char_fc = nn.Linear(char_feature_dim, char_dim)
        # self.graph_fc = nn.Linear(self.input_dim, char_dim)

        # structure encoder
        if self.args.structure_encoder == "gcn":
            self.cross_graph_model = GCN(self.n_units[0], self.n_units[1], self.n_units[2],
                                         dropout=self.args.dropout)
        elif self.args.structure_encoder == "gat":
            self.cross_graph_model = GAT(n_units=self.n_units, n_heads=self.n_heads, dropout=args.dropout,
                                         attn_dropout=args.attn_dropout,
                                         instance_normalization=self.args.instance_normalization, diag=True)

        #########################
        # Fusion Encoder
        #########################
        self.fusion = MformerFusion(args, modal_num=self.args.inner_view_num,
                                    with_weight=self.args.with_weight)

    # 在 MultiModalEncoder.forward 的参数列表最后加上 causal_bias=None
    def forward(self,
                input_idx,
                adj,
                img_features=None,
                rel_features=None,
                att_features=None,
                name_features=None,
                char_features=None,
                causal_bias=None):

        if self.args.w_gcn:
            gph_emb = self.cross_graph_model(self.entity_emb(input_idx), adj)
        else:
            gph_emb = None
        if self.args.w_img:
            img_emb = self.img_fc(img_features)
        else:
            img_emb = None
        if self.args.w_rel:
            rel_emb = self.rel_fc(rel_features)
        else:
            rel_emb = None
        if self.args.w_attr:
            att_emb = self.att_fc(att_features)
        else:
            att_emb = None
        if self.args.w_name and name_features is not None:
            name_emb = self.name_fc(name_features)
        else:
            name_emb = None
        if self.args.w_char and char_features is not None:
            char_emb = self.char_fc(char_features)
        else:
            char_emb = None
            
        # 注意这里的模态顺序必须固定，与后续在 MEAformer 中生成的 bias 顺序完全一致
        emb_list = [img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb]
        # 将 emb_list 和 causal_bias 传给 fusion
        joint_emb, hidden_states, weight_norm = self.fusion(emb_list, causal_bias=causal_bias)

        # joint_emb, hidden_states, weight_norm = self.fusion([img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb])

        return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states, weight_norm


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # [8, 8, 3, 256]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        # return x
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions=False,
        causal_bias=None,  # 新增因果偏置参数
    ):
        mixed_query_layer = self.query(hidden_states)
        # [8, 3, 8, 256]
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # # [8, 3, 8, 8]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # ====== 核心：注入动态因果偏置项 (公式 19) ======
        if causal_bias is not None:
            # causal_bias 形状为 [modal_num]，通过 view 广播到 [bs, num_heads, modal_num, modal_num]
            # 加在 key 的维度上，表示分配给对应模态 j 的偏置
            attention_scores = attention_scores + causal_bias.view(1, 1, 1, -1)
        # ===============================================

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # [8, 3, 8, 8]
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # [8, 8, 768]
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions=False,
        causal_bias=None
    ):
        self_outputs = self.self(
            hidden_states,
            output_attentions,
            causal_bias=causal_bias,
        )

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # attention: torch.Size([30355, 5, 4, 4])
        # 5: head
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN["gelu"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = 0
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        if self.config.use_intermediate:
            self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: torch.Tensor, output_attentions=False,causal_bias=None):
        self_attention_outputs = self.attention(
            hidden_states,
            output_attentions=output_attentions,
            causal_bias=causal_bias,
        )
        if not self.config.use_intermediate:
            return (self_attention_outputs[0], self_attention_outputs[1])

        attention_output = self_attention_outputs[0]
        # if decoder, the last output is tuple of self-attn cache
        outputs = self_attention_outputs[1]
        # present_key_value = self_attention_outputs[-1]
        # torch.Size([30355, 4, 300])
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output, outputs)
        # if decoder, return the attn key/values as the last output
        # outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
        # return attention_output


# ====== 新增样本调度策略函数 ======
def compute_scheduler_threshold(epoch, total_epochs, k):
    """
    计算当前轮次的调度阈值 φ(e; E, k) = 1 - exp(-k * e / E)
    
    参数:
        epoch (int): 当前训练轮次 (0 ≤ e < E)
        total_epochs (int): 总训练轮次 E
        k (float): 调度速度系数
        
    返回:
        float: 当前轮次的阈值 φ(e)
    """
    return 1 - math.exp(-k * epoch / total_epochs)


def select_samples_by_difficulty(sample_difficulties, threshold):
    """
    根据难度阈值筛选样本 T(e) = 𝕀[ρ(p_i) < φ(e)]
    
    参数:
        sample_difficulties (Tensor): 样本难度值 ρ(p_i) [batch_size]
        threshold (float): 当前轮次阈值 φ(e)
        
    返回:
        Tensor: 布尔掩码，True表示选中该样本
    """
    return sample_difficulties < threshold