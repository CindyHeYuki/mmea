import os
import json
import re

# 1. ⚠️ 修改路径配置为 FBYG15K 文件夹
data_dir = "/data0/hwx/mmea_copy/data/mmkg/FBYG15K/norm"
save_path = "/data0/hwx/mmea_copy/data/mmkg/FBYG15K/ent_name.json"

# 这里直接复用你为 FBDB15K 准备的那个 Freebase 翻译字典！
mid2name_file = "/data0/hwx/mmea_copy/data/mmkg/FBDB15K/FB15k_mid2name.txt" 

ent_names = []
mid_to_name_dict = {}

# 0. 先加载外部字典 (保持不变)
print("正在读取 mid2name 字典...")
with open(mid2name_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            mid = parts[0].strip()
            name = parts[1].strip()
            mid_to_name_dict[mid] = name

# 1. 处理 Freebase (ent_ids_1, 保持完全一致的防单词粘连逻辑)
hit_count = 0
with open(os.path.join(data_dir, "ent_ids_1"), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            ent_id = int(parts[0])
            full_mid = parts[1].strip() 
            
            if full_mid in mid_to_name_dict:
                real_name = mid_to_name_dict[full_mid]
                real_name = real_name.replace("_", " ").replace("-", " ")
                clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', real_name)
                name_tokens = [word.lower() for word in clean_name.split() if word]
                hit_count += 1
            else:
                mid_str = full_mid.split("/")[-1]
                name_tokens = list(mid_str) 
                
            ent_names.append([ent_id, name_tokens])

print(f"Freebase 处理完毕，共 {hit_count} 个实体成功匹配到真实英文名！")

# 2. 处理 YAGO (ent_ids_2, 专门针对 YAGO 的清洗逻辑)
print("正在处理 YAGO 实体...")
with open(os.path.join(data_dir, "ent_ids_2"), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            ent_id = int(parts[0])
            uri = parts[1]
            
            # YAGO 的实体通常是 <实体名> 的格式，比如 <Albert_Einstein>
            # 去掉前后的尖括号
            raw_name = uri.strip("<>")
            
            # 防御性截取：如果有路径符号 /，取最后一个部分
            if "/" in raw_name:
                raw_name = raw_name.split("/")[-1]
            
            # 去掉括号里的消歧义内容，例如 Washington_(state) -> Washington
            clean_name = re.sub(r'\(.*?\)', '', raw_name)
            
            # 按照下划线切割并转小写
            name_tokens = [word.lower() for word in clean_name.split("_") if word]
            
            ent_names.append([ent_id, name_tokens])

# 3. 排序并保存
ent_names.sort(key=lambda x: x[0])
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(ent_names, f, ensure_ascii=False, indent=4)

print(f"✅ 大功告成！FBYG15K 的实体名称文件已保存至：{save_path}")