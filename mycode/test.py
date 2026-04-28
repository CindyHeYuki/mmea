
import torch
sd = torch.load('/data0/hwx/mmea_copy/data/mmkg/MEAformer/save/baseline_FBDB15K_norm_0.2_.pkl', map_location='cpu')
fusion_layers = set()
for k in sd.keys():
    if 'fusion.fusion_layer.' in k:
        # 提取层号: fusion_layer.0.xxx -> 0
        idx = k.split('fusion.fusion_layer.')[1].split('.')[0]
        fusion_layers.add(idx)
print('ckpt 中的 fusion_layer 索引:', sorted(fusion_layers))
print('fusion_layer 总层数:', len(fusion_layers))

# 顺便看 hidden_size
for k in ['multimodal_encoder.fusion.fusion_layer.0.attention.self.query.weight']:
    if k in sd:
        print(f'{k} shape: {sd[k].shape}')

# 还有 entity_emb 的 ent_num
ek = 'multimodal_encoder.entity_emb.weight'
if ek in sd:
    print(f'entity_emb shape: {sd[ek].shape}  (ent_num, hidden_size)')
