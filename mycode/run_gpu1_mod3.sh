#!/bin/bash

# 1. 获取当前时间戳 (例如: 20260402_180530)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 2. 定义本次运行的日志子文件夹路径
LOG_DIR="logs/${TIMESTAMP}"

# 3. 创建这个子文件夹 (-p 会自动连带创建父文件夹 logs)
mkdir -p ${LOG_DIR}

echo "===== 本批次实验开始 ====="
echo "所有日志将保存在: ${LOG_DIR}"

# # w/o surface
# FBDB15K
echo "Running FBDB15K norm 0.8 wo_surf..."
bash run_meaformer_mod3.sh 1 FBDB15K norm 0.8 0 > ${LOG_DIR}/FBDB15K_norm_0.8_wo_surf.log 2>&1

echo "Running FBDB15K norm 0.5 wo_surf..."
bash run_meaformer_mod3.sh 1 FBDB15K norm 0.5 0 > ${LOG_DIR}/FBDB15K_norm_0.5_wo_surf.log 2>&1

echo "Running FBDB15K norm 0.2 wo_surf..."
bash run_meaformer_mod3.sh 1 FBDB15K norm 0.2 0 > ${LOG_DIR}/FBDB15K_norm_0.2_wo_surf.log 2>&1

# FBYG15K
echo "Running FBYG15K norm 0.8 wo_surf..."
bash run_meaformer_mod3.sh 1 FBYG15K norm 0.8 0 > ${LOG_DIR}/FBYG15K_norm_0.8_wo_surf.log 2>&1

echo "Running FBYG15K norm 0.5 wo_surf..."
bash run_meaformer_mod3.sh 1 FBYG15K norm 0.5 0 > ${LOG_DIR}/FBYG15K_norm_0.5_wo_surf.log 2>&1

echo "Running FBYG15K norm 0.2 wo_surf..."
bash run_meaformer_mod3.sh 1 FBYG15K norm 0.2 0 > ${LOG_DIR}/FBYG15K_norm_0.2_wo_surf.log 2>&1

# DBP15K (w/o surface)
echo "Running DBP15K zh_en 0.3 wo_surf..."
bash run_meaformer_mod3.sh 1 DBP15K zh_en 0.3 0 > ${LOG_DIR}/DBP15K_zh_en_0.3_wo_surf.log 2>&1

echo "Running DBP15K ja_en 0.3 wo_surf..."
bash run_meaformer_mod3.sh 1 DBP15K ja_en 0.3 0 > ${LOG_DIR}/DBP15K_ja_en_0.3_wo_surf.log 2>&1

echo "Running DBP15K fr_en 0.3 wo_surf..."
bash run_meaformer_mod3.sh 1 DBP15K fr_en 0.3 0 > ${LOG_DIR}/DBP15K_fr_en_0.3_wo_surf.log 2>&1

# # w/ surface
# DBP15K (w/ surface)
echo "Running DBP15K zh_en 0.3 w_surf..."
bash run_meaformer_mod3.sh 1 DBP15K zh_en 0.3 1 > ${LOG_DIR}/DBP15K_zh_en_0.3_w_surf.log 2>&1

echo "Running DBP15K ja_en 0.3 w_surf..."
bash run_meaformer_mod3.sh 1 DBP15K ja_en 0.3 1 > ${LOG_DIR}/DBP15K_ja_en_0.3_w_surf.log 2>&1

echo "Running DBP15K fr_en 0.3 w_surf..."
bash run_meaformer_mod3.sh 1 DBP15K fr_en 0.3 1 > ${LOG_DIR}/DBP15K_fr_en_0.3_w_surf.log 2>&1

echo "===== 本批次实验全部完成 ====="