#!/bin/bash
# 在脚本最开头添加
echo "===== DEBUG INFO ====="
echo "当前目录: $(pwd)"
echo "脚本参数: $0 $@"
echo "epoch 参数: 应该为500"
echo "batch_size 参数: 应该为500"
echo "===== DEBUG INFO END ====="

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
            --save_model    0 \
            --batch_size    500 \
	        --csls          \
	        --csls_k        3 \
	        --random_seed   42 \
            --exp_name      mycode_$5_500-Norm \
            --exp_id        v1_$3_$4 \
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
            --k 0.5 \
            --lambda_val 0.2 \
            --use_causal_bias 0 \
            --causal_lambda 0.1 \
            --causal_eval_k 10 \
            --use_csc 0 \
            --csc_lambda_0 1 \
            --csc_eta 5.0 \
            --csc_gamma 0.1 \
            --exp_id "module1_k0.5" \

            #--enable_sota \
            