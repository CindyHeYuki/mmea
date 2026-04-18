#!/bin/bash

# ==========================================
# 使用方式：bash run_all_il.sh <GPU_ID>
# 例如：   bash run_all_il.sh 1
# ==========================================

# 检查是否传入 GPU 参数
if [ -z "$1" ]; then
    echo "❌ 错误：请在命令行指定 GPU ID"
    echo "用法: bash run_all_il.sh <GPU_ID>"
    echo "例如: bash run_all_il.sh 1"
    exit 1
fi

GPU=$1

# 1. 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 2. 定义本次运行的日志子文件夹路径
LOG_DIR="logs/${TIMESTAMP}_il_gpu${GPU}"

# 3. 创建这个子文件夹
mkdir -p ${LOG_DIR}

echo "===================================================="
echo "===== 本批次实验开始 (迭代学习版本) ====="
echo "===== GPU: ${GPU} ====="
echo "===== 日志目录: ${LOG_DIR} ====="
echo "===================================================="


# ==========================================
# FBDB15K 实验 (w/o surface) - IL
# ==========================================
echo "[$(date '+%H:%M:%S')] Running FBDB15K norm 0.8 wo_surf IL..."
bash run_meaformer_il.sh ${GPU} FBDB15K norm 0.8 0 > ${LOG_DIR}/FBDB15K_norm_0.8_wo_surf_il.log 2>&1

echo "[$(date '+%H:%M:%S')] Running FBDB15K norm 0.5 wo_surf IL..."
bash run_meaformer_il.sh ${GPU} FBDB15K norm 0.5 0 > ${LOG_DIR}/FBDB15K_norm_0.5_wo_surf_il.log 2>&1

echo "[$(date '+%H:%M:%S')] Running FBDB15K norm 0.2 wo_surf IL..."
bash run_meaformer_il.sh ${GPU} FBDB15K norm 0.2 0 > ${LOG_DIR}/FBDB15K_norm_0.2_wo_surf_il.log 2>&1

# ==========================================
# FBYG15K 实验 (w/o surface) - IL
# ==========================================
echo "[$(date '+%H:%M:%S')] Running FBYG15K norm 0.8 wo_surf IL..."
bash run_meaformer_il.sh ${GPU} FBYG15K norm 0.8 0 > ${LOG_DIR}/FBYG15K_norm_0.8_wo_surf_il.log 2>&1

echo "[$(date '+%H:%M:%S')] Running FBYG15K norm 0.5 wo_surf IL..."
bash run_meaformer_il.sh ${GPU} FBYG15K norm 0.5 0 > ${LOG_DIR}/FBYG15K_norm_0.5_wo_surf_il.log 2>&1

echo "[$(date '+%H:%M:%S')] Running FBYG15K norm 0.2 wo_surf IL..."
bash run_meaformer_il.sh ${GPU} FBYG15K norm 0.2 0 > ${LOG_DIR}/FBYG15K_norm_0.2_wo_surf_il.log 2>&1

# ==========================================
# DBP15K 实验 (w/o surface) - IL
# ==========================================
echo "[$(date '+%H:%M:%S')] Running DBP15K zh_en 0.3 wo_surf IL..."
bash run_meaformer_il.sh ${GPU} DBP15K zh_en 0.3 0 > ${LOG_DIR}/DBP15K_zh_en_0.3_wo_surf_il.log 2>&1

echo "[$(date '+%H:%M:%S')] Running DBP15K ja_en 0.3 wo_surf IL..."
bash run_meaformer_il.sh ${GPU} DBP15K ja_en 0.3 0 > ${LOG_DIR}/DBP15K_ja_en_0.3_wo_surf_il.log 2>&1

echo "[$(date '+%H:%M:%S')] Running DBP15K fr_en 0.3 wo_surf IL..."
bash run_meaformer_il.sh ${GPU} DBP15K fr_en 0.3 0 > ${LOG_DIR}/DBP15K_fr_en_0.3_wo_surf_il.log 2>&1

# ==========================================
# DBP15K 实验 (w/ surface) - IL
# ==========================================
echo "[$(date '+%H:%M:%S')] Running DBP15K zh_en 0.3 w_surf IL..."
bash run_meaformer_il.sh ${GPU} DBP15K zh_en 0.3 1 > ${LOG_DIR}/DBP15K_zh_en_0.3_w_surf_il.log 2>&1

echo "[$(date '+%H:%M:%S')] Running DBP15K ja_en 0.3 w_surf IL..."
bash run_meaformer_il.sh ${GPU} DBP15K ja_en 0.3 1 > ${LOG_DIR}/DBP15K_ja_en_0.3_w_surf_il.log 2>&1

echo "[$(date '+%H:%M:%S')] Running DBP15K fr_en 0.3 w_surf IL..."
bash run_meaformer_il.sh ${GPU} DBP15K fr_en 0.3 1 > ${LOG_DIR}/DBP15K_fr_en_0.3_w_surf_il.log 2>&1

echo "===================================================="
echo "===== 本批次实验全部完成 (迭代) ====="
echo "===== 日志保存在: ${LOG_DIR} ====="
echo "===================================================="