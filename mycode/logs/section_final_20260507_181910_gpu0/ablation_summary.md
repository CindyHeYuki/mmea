# Ablation Results (Final)

Log dir: `logs/section_final_20260507_181910_gpu0`

数据集：fbdb20 fbdb50 fbdb80 fbyg20 fbyg50 fbyg80 dbp_zh dbp_ja dbp_fr

---

## FBDB15K 20%

### 叠加视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Backbone | 37.56 | 67.38 | 47.70 |
| + Sample Scheduling | 36.82 | 67.83 | 47.30 |
| + Differentiated Aggregation | 41.60 | 71.74 | 51.80 |
| + Counterfactual Consistency | 41.09 | 71.30 | 51.40 |
| + Topology-Consistent Repr | 42.33 | 70.22 | 51.80 |
| Full Model | 57.72 | 79.69 | 65.20 |

### 移除视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Full Model | 57.72 | 79.69 | 65.20 |
| w/o Sample Scheduling | 55.33 | 78.79 | 63.40 |
| w/o Differentiated Aggregation | 55.82 | 78.52 | 63.70 |
| w/o Counterfactual Consistency | 55.17 | 76.83 | 62.70 |
| w/o Topology-Consistent Repr | 52.35 | 77.59 | 61.20 |
| w/o Similarity Refinement | 42.33 | 70.22 | 51.80 |

---

## FBDB15K 50%

### 叠加视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Backbone | 55.89 | 80.55 | 64.80 |
| + Sample Scheduling | 55.83 | 80.87 | 64.70 |
| + Differentiated Aggregation | 59.43 | 83.43 | 67.90 |
| + Counterfactual Consistency | 58.91 | 82.61 | 67.40 |
| + Topology-Consistent Repr | 58.42 | 81.32 | 66.30 |
| Full Model | 71.87 | 87.56 | 77.60 |

### 移除视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Full Model | 71.87 | 87.56 | 77.60 |
| w/o Sample Scheduling | 70.73 | 87.19 | 76.50 |
| w/o Differentiated Aggregation | 69.80 | 86.56 | 75.80 |
| w/o Counterfactual Consistency | 70.15 | 85.88 | 75.80 |
| w/o Topology-Consistent Repr | 69.83 | 87.36 | 76.20 |
| w/o Similarity Refinement | 58.42 | 81.32 | 66.30 |

---

## FBDB15K 80%

### 叠加视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Backbone | 69.42 | 88.83 | 76.40 |
| + Sample Scheduling | 68.17 | 87.28 | 75.10 |
| + Differentiated Aggregation | 73.31 | 90.54 | 79.60 |
| + Counterfactual Consistency | 72.41 | 89.73 | 78.90 |
| + Topology-Consistent Repr | 69.81 | 87.78 | 76.40 |
| Full Model | 81.98 | 92.88 | 86.10 |

### 移除视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Full Model | 81.98 | 92.88 | 86.10 |
| w/o Sample Scheduling | 80.43 | 92.72 | 85.00 |
| w/o Differentiated Aggregation | 78.91 | 90.70 | 83.30 |
| w/o Counterfactual Consistency | 79.77 | 91.32 | 84.20 |
| w/o Topology-Consistent Repr | 82.26 | 94.09 | 86.70 |
| w/o Similarity Refinement | 69.81 | 87.78 | 76.40 |

---

## FBYG15K 20%

### 叠加视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Backbone | 29.82 | 54.24 | 38.20 |
| + Sample Scheduling | 26.42 | 52.78 | 35.40 |
| + Differentiated Aggregation | 30.88 | 57.72 | 40.10 |
| + Counterfactual Consistency | 30.59 | 57.06 | 39.60 |
| + Topology-Consistent Repr | 33.16 | 57.44 | 41.50 |
| Full Model | 45.62 | 66.24 | 52.70 |

### 移除视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Full Model | 45.62 | 66.24 | 52.70 |
| w/o Sample Scheduling | 44.60 | 65.70 | 52.00 |
| w/o Differentiated Aggregation | 43.74 | 64.53 | 51.00 |
| w/o Counterfactual Consistency | 44.82 | 64.82 | 51.60 |
| w/o Topology-Consistent Repr | 36.52 | 61.99 | 45.20 |
| w/o Similarity Refinement | 33.16 | 57.44 | 41.50 |

---

## FBYG15K 50%

### 叠加视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Backbone | 50.98 | 75.32 | 59.60 |
| + Sample Scheduling | 50.55 | 75.05 | 59.30 |
| + Differentiated Aggregation | 54.95 | 78.21 | 63.30 |
| + Counterfactual Consistency | 54.14 | 77.71 | 62.60 |
| + Topology-Consistent Repr | 52.70 | 74.84 | 60.40 |
| Full Model | 65.16 | 81.36 | 70.90 |

### 移除视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Full Model | 65.16 | 81.36 | 70.90 |
| w/o Sample Scheduling | 64.20 | 80.54 | 69.90 |
| w/o Differentiated Aggregation | 63.30 | 79.18 | 69.00 |
| w/o Counterfactual Consistency | 64.07 | 79.71 | 69.70 |
| w/o Topology-Consistent Repr | 62.09 | 82.25 | 69.10 |
| w/o Similarity Refinement | 52.70 | 74.84 | 60.40 |

---

## FBYG15K 80%

### 叠加视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Backbone | 66.25 | 85.85 | 73.40 |
| + Sample Scheduling | 65.76 | 85.89 | 73.00 |
| + Differentiated Aggregation | 69.64 | 88.04 | 76.30 |
| + Counterfactual Consistency | 69.24 | 87.86 | 75.90 |
| + Topology-Consistent Repr | 65.71 | 83.30 | 72.10 |
| Full Model | 75.49 | 89.15 | 80.50 |

### 移除视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Full Model | 75.49 | 89.15 | 80.50 |
| w/o Sample Scheduling | 74.38 | 88.66 | 79.60 |
| w/o Differentiated Aggregation | 73.44 | 87.37 | 78.40 |
| w/o Counterfactual Consistency | 73.71 | 87.59 | 78.90 |
| w/o Topology-Consistent Repr | 76.29 | 90.80 | 81.60 |
| w/o Similarity Refinement | 65.71 | 83.30 | 72.10 |

---

## DBP15K zh-en

### 叠加视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Backbone | 71.02 | 92.50 | 78.80 |
| + Sample Scheduling | 69.14 | 91.74 | 77.20 |
| + Differentiated Aggregation | 69.37 | 91.82 | 77.40 |
| + Counterfactual Consistency | 69.34 | 91.83 | 77.40 |
| + Topology-Consistent Repr | 73.76 | 93.65 | 80.80 |
| Full Model | 85.29 | 96.86 | 89.60 |

### 移除视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Full Model | 85.29 | 96.86 | 89.60 |
| w/o Sample Scheduling | 84.35 | 96.30 | 88.70 |
| w/o Differentiated Aggregation | 85.19 | 96.88 | 89.50 |
| w/o Counterfactual Consistency | 85.16 | 96.81 | 89.50 |
| w/o Topology-Consistent Repr | 80.27 | 95.87 | 85.90 |
| w/o Similarity Refinement | 73.76 | 93.65 | 80.80 |

---

## DBP15K ja-en

### 叠加视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Backbone | 70.89 | 93.23 | 78.90 |
| + Sample Scheduling | 70.11 | 93.50 | 78.50 |
| + Differentiated Aggregation | 70.11 | 93.50 | 78.50 |
| + Counterfactual Consistency | 70.04 | 93.42 | 78.40 |
| + Topology-Consistent Repr | 73.26 | 93.90 | 80.80 |
| Full Model | 85.07 | 97.55 | 89.80 |

### 移除视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Full Model | 85.07 | 97.55 | 89.80 |
| w/o Sample Scheduling | 84.27 | 97.35 | 89.10 |
| w/o Differentiated Aggregation | 85.07 | 97.55 | 89.80 |
| w/o Counterfactual Consistency | 84.73 | 97.35 | 89.40 |
| w/o Topology-Consistent Repr | 80.64 | 96.78 | 86.70 |
| w/o Similarity Refinement | 73.26 | 93.90 | 80.80 |

---

## DBP15K fr-en

### 叠加视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Backbone | 71.19 | 93.93 | 79.40 |
| + Sample Scheduling | 69.95 | 93.66 | 78.30 |
| + Differentiated Aggregation | 69.95 | 93.66 | 78.30 |
| + Counterfactual Consistency | 69.87 | 93.60 | 78.20 |
| + Topology-Consistent Repr | 74.08 | 95.11 | 81.60 |
| Full Model | 86.66 | 98.34 | 91.00 |

### 移除视角

| Variant | H@1 | H@10 | MRR |
|---------|----:|-----:|----:|
| Full Model | 86.66 | 98.34 | 91.00 |
| w/o Sample Scheduling | 85.99 | 98.23 | 90.60 |
| w/o Differentiated Aggregation | 86.66 | 98.34 | 91.00 |
| w/o Counterfactual Consistency | 86.39 | 98.23 | 90.80 |
| w/o Topology-Consistent Repr | 81.29 | 97.28 | 87.10 |
| w/o Similarity Refinement | 74.08 | 95.11 | 81.60 |

---

## 横向汇总（H@1）

| Variant | FBDB15K 20% | FBDB15K 50% | FBDB15K 80% | FBYG15K 20% | FBYG15K 50% | FBYG15K 80% | DBP15K zh-en | DBP15K ja-en | DBP15K fr-en |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| Backbone | 37.56 | 55.89 | 69.42 | 29.82 | 50.98 | 66.25 | 71.02 | 70.89 | 71.19 |
| + Sample Scheduling | 36.82 | 55.83 | 68.17 | 26.42 | 50.55 | 65.76 | 69.14 | 70.11 | 69.95 |
| + Differentiated Aggregation | 41.60 | 59.43 | 73.31 | 30.88 | 54.95 | 69.64 | 69.37 | 70.11 | 69.95 |
| + Counterfactual Consistency | 41.09 | 58.91 | 72.41 | 30.59 | 54.14 | 69.24 | 69.34 | 70.04 | 69.87 |
| + Topology-Consistent Repr | 42.33 | 58.42 | 69.81 | 33.16 | 52.70 | 65.71 | 73.76 | 73.26 | 74.08 |
| Full Model | 57.72 | 71.87 | 81.98 | 45.62 | 65.16 | 75.49 | 85.29 | 85.07 | 86.66 |
| w/o Sample Scheduling | 55.33 | 70.73 | 80.43 | 44.60 | 64.20 | 74.38 | 84.35 | 84.27 | 85.99 |
| w/o Differentiated Aggregation | 55.82 | 69.80 | 78.91 | 43.74 | 63.30 | 73.44 | 85.19 | 85.07 | 86.66 |
| w/o Counterfactual Consistency | 55.17 | 70.15 | 79.77 | 44.82 | 64.07 | 73.71 | 85.16 | 84.73 | 86.39 |
| w/o Topology-Consistent Repr | 52.35 | 69.83 | 82.26 | 36.52 | 62.09 | 76.29 | 80.27 | 80.64 | 81.29 |
| w/o Similarity Refinement | 42.33 | 58.42 | 69.81 | 33.16 | 52.70 | 65.71 | 73.76 | 73.26 | 74.08 |

_数值百分制（×100）。— 表示日志缺失。_
