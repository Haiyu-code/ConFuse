import os
import h5py
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pickle
FEAT_DIR  = '/data2/wanghy/first/data/TCGA-UCEC/features/'
OUT_DIR   = '/data2/wanghy/first/UCEC/wsi_mean_feats/'
PKL_PATH  = '/data2/wanghy/first/UCEC/ucec_multimodal_v1.pkl'
os.makedirs(OUT_DIR, exist_ok=True)
# 加载PKL获取患者列表
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)
h5_paths_all = data['h5_paths']
patient_ids  = data['patient_id']
print(f"总患者数: {len(patient_ids)}")
# 对每个患者：读H5 → 计算均值特征 → 保存pt
missing = []
for pid, h5_list in tqdm(zip(patient_ids, h5_paths_all), 
                          total=len(patient_ids)):
    out_path = os.path.join(OUT_DIR, f"{pid}.pt")
    if os.path.exists(out_path):
        continue
    all_feats = []
    for h5_path in h5_list:
        try:
            with h5py.File(h5_path, 'r') as f:
                feat = f['features'][:]       # (1, n_patches, 1536)
                feat = feat.squeeze(0)        # (n_patches, 1536)
                all_feats.append(feat)
        except Exception as e:
            print(f"读取失败: {h5_path}, {e}")
            missing.append(pid)
            continue
    if all_feats:
        feats = np.concatenate(all_feats, axis=0)  # (total_patches, 1536)
        # 保存完整patch特征（用于ABMIL）
        torch.save({
            'features': torch.FloatTensor(feats),   # (n_patches, 1536)
            'pid': pid,
        }, out_path)
print(f"\n✅ 完成! 保存到: {OUT_DIR}")
print(f"缺失患者: {len(missing)}")
# 验证
saved = os.listdir(OUT_DIR)
print(f"已保存: {len(saved)} 个pt文件")
# 看一个
sample = torch.load(os.path.join(OUT_DIR, saved[0]), 
                     map_location='cpu', weights_only=False)
print(f"示例: {saved[0]}, shape={sample['features'].shape}")