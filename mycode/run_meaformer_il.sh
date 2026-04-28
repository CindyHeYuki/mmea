#!/bin/bash
# ============================================================================
#  IL 训练脚本 (派生自 run_meaformer_sweep.sh + 原版 run_meaformer_il.sh)
#
#  与原版 IL 脚本的关键差异 (修正项):
#    1. --save_model 1            : 必须存 ckpt(原版误写为 0,会让本次实验白跑)
#    2. --semi_learn_step 10      : 与交接文档对齐(原版误写为 5)
#    3. --causal_lambda 0.1       : 与 baseline 对齐(原版误写为 0.4)
#    4. --csc_lambda_0 0.1        : 与 baseline 对齐(原版误写为 0.2)
#    5. --use_neighbor 1          : 显式声明(原版漏写)
#    6. --exp_id 加 _il 后缀      : 避免覆盖 baseline ckpt
#    7. --do_alpha_sweep 0        : 训练阶段不扫,扫描留给单独 sweep 阶段
#    8. 保留 --enable_sota        : 复现 MEAformer 原版 IL 行为
#                                   注:它会把 weight_decay 改为 0.0005~0.001,
#                                   下方有显式打印
#    9. 保留 --batch_size 3500    : MEAformer 原版 IL 就用 3500
#
#  用法:
#    bash run_meaformer_il.sh <GPU> <DATASET> <SPLIT> <RATE> <USE_SURFACE>
#    例: bash run_meaformer_il.sh 0 FBDB15K norm 0.2 0
#
#  产出 (在 <data_path>/MEAformer/save/ 下):
#    - <name>_non_iter.pkl       : IL 启动前(epoch 500)的 ckpt - 天然对照组
#    - <name>.pkl                : IL 完成后(epoch 1000)的 ckpt - 真正测试目标
#    - <name>_cj.json / <name>_non_iter_cj.json : 对应的因果 C_j 信号
# ============================================================================

set -u

if [ $# -lt 5 ]; then
    echo "❌ 用法: bash run_meaformer_il.sh <GPU> <DATASET> <SPLIT> <RATE> <USE_SURFACE>"
    echo "   例: bash run_meaformer_il.sh 0 FBDB15K norm 0.2 0"
    exit 1
fi

GPU=$1
DATASET=$2
SPLIT=$3
RATE=$4
USE_SURFACE=$5

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

echo "=============================================================="
echo "🚀 IL 训练启动"
echo "   GPU         : ${GPU}"
echo "   Dataset     : ${DATASET} / ${SPLIT} / rate=${RATE} / surface=${USE_SURFACE}"
echo "   Epochs      : 1000 (前 500 普通训练 + 后 500 IL)"
echo "   batch_size  : 3500 (与 MEAformer 原版 IL 对齐)"
echo "   il_start    : 500  | semi_learn_step : 10"
echo "   ⚠️  --enable_sota 已开启,weight_decay 会被偷偷改为 0.0005~0.001"
echo "       请检查 log 中 [SAVE MODEL] 行确认实际值"
echo "=============================================================="

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --gpu ${GPU}  --eval_epoch 1  --only_test 0 \
    --model_name MEAformer \
    --data_choice ${DATASET}  --data_split ${SPLIT}  --data_rate ${RATE} \
    --epoch 1000  --lr 5e-4  --hidden_units "300,300,300" \
    --save_model 1 \
    --batch_size 3500 \
    --csls --csls_k 3  --random_seed 42 \
    --exp_name il_${USE_SURFACE}_1000-IL  --exp_id v1_${SPLIT}_${RATE}_il \
    --workers 12  --dist 0  --accumulation_steps 1 \
    --scheduler cos \
    --attr_dim 300 --img_dim 300 --name_dim 300 --char_dim 300 --hidden_size 300 \
    --tau 0.1 --structure_encoder gat \
    --num_attention_heads 1 --num_hidden_layers 1 \
    --use_surface ${USE_SURFACE} --use_intermediate 1 --replay 0 \
    --il --il_start 500 --semi_learn_step 10 \
    --enable_sota \
    --use_sample_schedule 1 --k 6 \
    --use_causal_bias 1 --causal_lambda 0.1 --causal_eval_k 10 \
    --use_csc 1 --csc_lambda_0 0.1 --csc_gamma 0.5 \
    --use_neighbor 1 --neighbor_alpha 0.5 \
    --use_plm 0 --freeze_plm 1 --plm_max_len 16 \
    --plm_name '/data0/hwx/mmea_copy/models/bert-base-multilingual-cased' \
    --plm_embed_name 0 --plm_embed_rel 0 --plm_embed_attr 0 \
    --do_alpha_sweep 0 --use_3d_difficulty 0

RC=$?
echo "=============================================================="
if [ ${RC} -eq 0 ]; then
    echo "✅ IL 训练完成 (rc=0)"
    echo ""
    echo "下一步: 检查 save/ 目录下生成的 ckpt 文件名,然后跑:"
    echo "  bash run_fb_sweep_il.sh <GPU>"
    echo ""
    echo "建议先跑这条命令确认 ckpt 名字:"
    echo "  ls -lh \$(grep 'target path' <最近的 log> | tail -2)"
else
    echo "❌ IL 训练失败 (rc=${RC})"
fi
echo "=============================================================="
exit ${RC}