#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
# 在脚本最开头添加
echo "===== DEBUG INFO ====="
echo "当前目录: $(pwd)"
echo "脚本参数: $0 $@"
echo "===== DEBUG INFO END ====="
export TOKENIZERS_PARALLELISM=false

#bash run_meaformer_sweep.sh <GPU> <DATASET> <SPLIT> <RATE> <USE_SURFACE> <CKPT_NAME>
#nohup bash run_meaformer_sweep.sh 2 FBDB15K norm 0.2 0 fbdb15k_20_base_ > train_fbdb15k_20.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 python  main.py \
            --gpu           $1    \
            --eval_epoch    1  \
            --only_test     0   \
            --model_name    MEAformer \
            --data_choice   $2 \
            --data_split    $3 \
            --data_rate     $4 \
            --epoch         500 \
            --lr            5e-4  \
            --hidden_units  "300,300,300" \
            --save_model    1 \
            --batch_size    500 \
	        --csls          \
	        --csls_k        3 \
	        --random_seed   42 \
            --exp_name      ${2}_${4}_train \
            --exp_id        ${6}_train \
            --model_name_save ${6} \
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
            --use_surface   $5     \
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
            --do_alpha_sweep 0 \
            --use_3d_difficulty 0 \
            --use_neighbor 0 \
            --neighbor_alpha 0.5 \

            #--enable_sota \
            