import os
import json
import re

# 你的路径配置
data_dir = "/data0/hwx/mmea_copy/data/mmkg/FBDB15K/norm"
save_path = "/data0/hwx/mmea_copy/data/mmkg/FBDB15K/ent_name.json"
mid2name_file = "/data0/hwx/mmea_copy/data/mmkg/FBDB15K/FB15k_mid2name.txt" # 替换为你下载的文件路径

ent_names = []
mid_to_name_dict = {}

# 0. 先加载外部字典
print("正在读取 mid2name 字典...")
with open(mid2name_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            # 格式：/m/027rn -> George Washington
            mid = parts[0].strip()
            name = parts[1].strip()
            mid_to_name_dict[mid] = name

# 1. 处理 Freebase (用真实的英文名字替换机器码)
hit_count = 0
with open(os.path.join(data_dir, "ent_ids_1"), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            ent_id = int(parts[0])
            full_mid = parts[1].strip() # 比如 /m/027rn
            
            # 去字典里查真实名字
            if full_mid in mid_to_name_dict:
                real_name = mid_to_name_dict[full_mid]
                # 💡 关键修复：先把下划线和短横线替换成空格，防止单词粘连！
                real_name = real_name.replace("_", " ").replace("-", " ")
                
                clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', real_name)
                name_tokens = [word.lower() for word in clean_name.split() if word]
                hit_count += 1
            
            else:
                # 极少数找不到的，退化回原先的单字符模式
                mid_str = full_mid.split("/")[-1]
                name_tokens = list(mid_str) 
                
            ent_names.append([ent_id, name_tokens])

print(f"Freebase 处理完毕，共 {hit_count} 个实体成功匹配到真实英文名！")

# 2. 处理 DBPedia (保持不变)
with open(os.path.join(data_dir, "ent_ids_2"), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            ent_id = int(parts[0])
            uri = parts[1]
            if "resource/" in uri:
                raw_name = uri.split("resource/")[-1].strip(">")
            else:
                raw_name = uri.split("/")[-1].strip(">")
            
            clean_name = re.sub(r'\(.*?\)', '', raw_name)
            name_tokens = [word.lower() for word in clean_name.split("_") if word]
            ent_names.append([ent_id, name_tokens])

# 3. 排序并保存
ent_names.sort(key=lambda x: x[0])
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(ent_names, f, ensure_ascii=False, indent=4)

print(f"✅ 大功告成！完美的实体名称文件已保存至：{save_path}")