import os
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ==========================================
# 1. 核心工具函数：复用你 data.py 里的优秀清洗逻辑
# ==========================================
def clean_uri_to_text(uri):
    """将 <http://dbpedia.org/ontology/birthDate> 转化为 'birth date'"""
    if uri is None or len(uri) == 0:
        return "none"
    # 截取最后的路径
    text = uri.strip().split('/')[-1].split('#')[-1].strip('>')
    # 处理 Freebase 多级路径
    text = text.replace('.', ' ')
    # 处理下划线和连字符
    text = text.replace('_', ' ').replace('-', ' ')
    # 处理驼峰命名 (如 birthDate -> birth Date)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text.lower().strip()


# ==========================================
# 2. 全局路径与模型配置
# ==========================================
# 绝对路径，指向你存放数据集的地方
DATA_DIR = "/data0/hwx/mmea_copy/data/mmkg/FBDB15K/norm"
# 推荐使用多语言或者 MPNet 模型
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

if __name__ == "__main__":
    # 使用空闲的 GPU（如果你后台跑着实验，这行保证它不会去抢卡，除非你手动用 CUDA_VISIBLE_DEVICES 指定）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading PLM model: {MODEL_NAME} on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # ==========================================
    # 3. 加载实体的 ID 映射表 (String URI -> Integer ID)
    # ==========================================
    print("Loading entity IDs...")
    ent2id = {}
    total_ents = 0
    # 读取 ent_ids_1 和 ent_ids_2
    for i in [1, 2]:
        ent_ids_path = os.path.join(DATA_DIR, f"ent_ids_{i}")
        if os.path.exists(ent_ids_path):
            with open(ent_ids_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        ent_id = int(parts[0])
                        ent_uri = parts[1]
                        ent2id[ent_uri] = ent_id
                        # 记录最大的 ID，用来初始化特征矩阵的行数
                        if ent_id > total_ents:
                            total_ents = ent_id

    # 实体总数 = 最大 ID + 1 (ID 从 0 开始)
    total_ents += 1 
    print(f"Total entities found: {total_ents}")

    # ==========================================
    # 4. 读取属性文件，并构造 No-Name Prompt 句子
    # ==========================================
    # 初始化所有实体为一个空描述（防止有些实体没属性导致报错）
    entity_sentences = ["An entity with no specific attributes."] * total_ents

    print("Parsing attributes and generating text prompts...")
    for attr_file in ['training_attrs_1', 'training_attrs_2']:
        attr_path = os.path.join(DATA_DIR, attr_file)
        if os.path.exists(attr_path):
            with open(attr_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    
                    ent_uri = parts[0]
                    # 如果该实体不在映射表里，跳过
                    if ent_uri not in ent2id:
                        continue

                    ent_id = ent2id[ent_uri]
                    
                    # 取出后面的所有属性 URI，并用写好的函数清洗为自然语言
                    attrs = []
                    for uri in parts[1:]:
                        clean_attr = clean_uri_to_text(uri)
                        if clean_attr != "none":
                            attrs.append(clean_attr)
                    
                    # 如果该实体有属性，则拼接成一句话
                    if len(attrs) > 0:
                        attrs_str = ", ".join(attrs)
                        sentence = f"An entity with the following attributes: {attrs_str}."
                        entity_sentences[ent_id] = sentence

    # ==========================================
    # 5. 送入 PLM 提取 768 维特征
    # ==========================================
    print(f"Start encoding {len(entity_sentences)} entities...")
    print("Example Prompt [0]:", entity_sentences[0])
    
    # 批量编码，batch_size 设为 256 加快提取速度
    embeddings = model.encode(
        entity_sentences, 
        batch_size=256, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )

    # ==========================================
    # 6. 保存为 .npy 文件供 main.py 训练时读取
    # ==========================================
    out_path = os.path.join(DATA_DIR, 'plm_attribute_features.npy')
    np.save(out_path, embeddings)
    print(f"✅ Success! PLM features saved to {out_path} with shape {embeddings.shape}")