#!/bin/bash
# ============================================================================
# 从已保存的模型加载 → 跳过训练 → 直接执行推理 + α 扫描
# ----------------------------------------------------------------------------
# 用法:
#   bash run_sweep_only.sh <GPU> <DATASET> <RATE> <USE_SURFACE> <CKPT_NAME>
# 
# 示例（FBYG15K 20% seed，当前目标）:
#   bash run_sweep_only.sh 0 FBYG15K 0.2 0 fbyg15k_20_base_
#
# 示例（FBDB15K 20% seed with surface）:
#   bash run_sweep_only.sh 0 FBDB15K 0.2 1 fbdb15k_20_base_
# ============================================================================
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

echo "===== SWEEP ONLY DEBUG INFO ====="
echo "当前目录: $(pwd)"
echo "脚本参数: $0 $@"
echo "===== SWEEP ONLY DEBUG INFO END ====="

GPU=${1:-0}
DATASET=${2:-FBYG15K}
DATA_SPLIT=${3:-norm}
RATE=${4:-0.2}
USE_SURFACE=${5:-0}
CKPT=${6:-fbyg15k_20_base_}

echo "=========================================================="
echo "🚀 [SWEEP ONLY] GPU=$GPU"
echo "🚀 [SWEEP ONLY] DATASET=$DATASET"
echo "🚀 [SWEEP ONLY] DATA_SPLIT=$DATA_SPLIT"
echo "🚀 [SWEEP ONLY] RATE=$RATE"
echo "🚀 [SWEEP ONLY] USE_SURFACE=$USE_SURFACE"
echo "🚀 [SWEEP ONLY] CHECKPOINT=$CKPT"
echo "=========================================================="

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
            --gpu           $GPU    \
            --eval_epoch    1  \
            --only_test     1   \
            --model_name_save  $CKPT \
            --model_name    MEAformer \
            --data_choice   $DATASET \
            --data_split    $DATA_SPLIT \
            --data_rate     $RATE \
            --epoch         500 \
            --lr            5e-4  \
            --hidden_units  "300,300,300" \
            --save_model    0 \
            --batch_size    500 \
            --csls          \
            --csls_k        3 \
            --random_seed   42 \
            --exp_name      sweep_only_${DATASET}_${RATE} \
            --exp_id        ${CKPT%_}_sweep \
            --workers       12 \
            --dist          0 \
            --accumulation_steps 1 \
            --scheduler     cos \
            --attr_dim      300     \
            --img_dim       300     \
            --name_dim      300     \
            --char_dim      300     \
            --hidden_size   300     \
            --tau           0.1     \
            --structure_encoder "gat" \
            --num_attention_heads 1 \
            --num_hidden_layers 1 \
            --use_surface   $USE_SURFACE \
            --use_intermediate 1   \
            --replay 0 \
            --use_sample_schedule 1 \
            --k 6 \
            --use_causal_bias 1 \
            --causal_lambda 0.1 \
            --causal_eval_k 10 \
            --use_csc 1 \
            --csc_lambda_0 0.1 \
            --csc_gamma 0.5 \
            --use_plm 0 \
            --plm_name '/data0/hwx/mmea_copy/models/bert-base-multilingual-cased' \
            --freeze_plm 1 \
            --plm_max_len 16 \
            --plm_embed_name 0 \
            --plm_embed_rel 0 \
            --plm_embed_attr 0 \
            --do_alpha_sweep 1 \
            --use_3d_difficulty 0 \
            --use_neighbor 1 \
            --neighbor_alpha 0.5 \
            --use_bidirectional_consistency 0 \
            --bidir_lambda 0.05 \
            --csls_iter 4