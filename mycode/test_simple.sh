#!/bin/bash
# test_simple.sh
# 这个脚本不依赖任何参数，直接硬编码

echo "=== 运行简化测试脚本 ==="
echo "当前时间: $(date)"
echo "当前目录: $(pwd)"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 直接运行，不使用参数传递
python main.py \
    --gpu 0 \
    --eval_epoch 1 \
    --only_test 0 \
    --model_name MEAformer \
    --data_choice FBDB15K \
    --data_split norm \
    --data_rate 0.2 \
    --epoch 3 \           # 改为3，更容易看到变化
    --lr 5e-4 \
    --hidden_units "300,300,300" \
    --save_model 0 \
    --batch_size 300 \    # 改为300
    --csls \
    --csls_k 3 \
    --random_seed 42 \
    --exp_name "TEST_epoch3_bs300" \
    --exp_id "debug_v1" \
    --workers 4 \
    --dist 0 \
    --accumulation_steps 1 \
    --scheduler cos \
    --attr_dim 300 \
    --img_dim 300 \
    --name_dim 300 \
    --char_dim 300 \
    --hidden_size 300 \
    --tau 0.1 \
    --structure_encoder "gat" \
    --num_attention_heads 1 \
    --num_hidden_layers 1 \
    --use_surface 0 \
    --use_intermediate 1 \
    --enable_sota \
    --replay 0

echo "=== 脚本执行结束 ==="