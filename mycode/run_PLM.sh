#!/bin/bash

# 1. 获取当前时间戳 (例如: 20260402_180530)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 2. 定义本次运行的日志子文件夹路径
LOG_DIR="logs/${TIMESTAMP}"

# 3. 创建这个子文件夹 (-p 会自动连带创建父文件夹 logs)
mkdir -p ${LOG_DIR}

echo "===== 本批次实验开始 ====="
echo "所有日志将保存在: ${LOG_DIR}"

# 删除了末尾的 &，它们现在会乖乖排队，按顺序串行执行！
echo "全关闭"
bash run_meaformer.sh 3 FBDB15K norm 0.2 0 0 0 0 0 > test_baseline.out 2>&1 
# 这 5 个 0 分别代表：关闭 surface, 关闭 PLM总开关, 关闭 name, 关闭 rel, 关闭 attr

echo "正在运行 [1/4]: 开PLM name1 rel1 attr1 ..."
bash run_meaformer.sh 3 FBDB15K norm 0.2 1 1 1 1 1 > ${LOG_DIR}/test111.log 2>&1

echo "正在运行 [2/4]: 开PLM name0 rel1 attr0 ..."
bash run_meaformer.sh 2 FBDB15K norm 0.2 1 1 0 1 0 > ${LOG_DIR}/test010.log 2>&1

echo "正在运行 [3/4]: 开PLM name0 rel0 attr1 ..."
bash run_meaformer.sh 2 FBDB15K norm 0.2 1 1 0 0 1 > ${LOG_DIR}/test001.log 2>&1

echo "正在运行 [4/4]: 开PLM name0 rel1 attr1 ..."
bash run_meaformer.sh 2 FBDB15K norm 0.2 1 1 0 1 1 > ${LOG_DIR}/test011.log 2>&1

echo "===== 本批次所有实验全部完成！ ====="