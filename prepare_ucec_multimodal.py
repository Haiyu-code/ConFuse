import os
import h5py
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
FEAT_DIR  = '/data2/wanghy/first/data/TCGA-UCEC/features/'
CBIO_PAT  = '/data2/wanghy/first/data/ucec_tcga_pan_can_atlas_2018/data_clinical_patient.txt'
RNA_PATH  = '/data2/wanghy/first/data/TCGA-UCEC.star_fpkm-uq.tsv/TCGA-UCEC.star_fpkm-uq.tsv'
OUT_PKL   = '/data2/wanghy/first/UCEC/ucec_multimodal_v1.pkl'
# ============================================================
# 1. 生存 + 亚型数据
# ============================================================
pat = pd.read_csv(CBIO_PAT, sep='\t', comment='#', skiprows=4)
pat = pat[['PATIENT_ID','SUBTYPE','OS_MONTHS','OS_STATUS']].copy()
pat.columns = ['patient','subtype','os_months','os_status']
pat['os_event'] = (pat['os_status'] == '1:DECEASED').astype(int)
pat['os_days']  = pat['os_months'] * 30.4
pat = pat.dropna(subset=['subtype','os_months','os_status'])
pat = pat[pat['os_days'] > 0]
print(f"临床数据: {len(pat)} 例")
# ============================================================
# 2. WSI特征路径映射（患者→h5文件列表）
# ============================================================
h5_files = [f for f in os.listdir(FEAT_DIR) if f.endswith('.h5')]
pid2h5   = defaultdict(list)
for f in h5_files:
    pid = f[:12]  # TCGA-XX-XXXX
    pid2h5[pid].append(os.path.join(FEAT_DIR, f))
print(f"H5文件总数: {len(h5_files)}, 患者数: {len(pid2h5)}")
# ============================================================
# 3. RNA数据
# ============================================================
print("加载RNA数据...")
rna = pd.read_csv(RNA_PATH, sep='\t', index_col=0)
rna = rna.T
rna.index = rna.index.str[:12]
rna = rna[~rna.index.duplicated(keep='first')]
print(f"RNA: {rna.shape}")
# ============================================================
# 4. 四方合并（临床+WSI+RNA）
# ============================================================
gene_cols = rna.columns.tolist()
records   = []
for _, row in pat.iterrows():
    pid = row['patient']
    # 必须同时有WSI和RNA
    if pid not in pid2h5:
        continue
    if pid not in rna.index:
        continue
    records.append({
        'patient':   pid,
        'subtype':   row['subtype'],
        'os_days':   row['os_days'],
        'os_event':  row['os_event'],
        'h5_paths':  sorted(pid2h5[pid]),  # 多张切片按名排序
        'rna':       rna.loc[pid, gene_cols].values.astype(float),
    })
df_meta = pd.DataFrame(records)
print(f"\n三模态匹配: {len(df_meta)} 例")
print(f"死亡事件: {df_meta['os_event'].sum()} ({df_meta['os_event'].mean():.1%})")
print(f"\nSUBTYPE分布:")
print(df_meta['subtype'].value_counts())
cn_high_n = (df_meta['subtype']=='UCEC_CN_HIGH').sum()
print(f"\nCN_HIGH: {cn_high_n}人 ({cn_high_n/len(df_meta):.1%})")
# ============================================================
# 5. RNA预处理（log2 + Top320 + z-score）
# ============================================================
X_rna = np.stack(df_meta['rna'].values)
X_rna = np.log2(X_rna + 1)
var     = X_rna.var(axis=0)
top_idx = np.argsort(var)[::-1][:320]
X_rna   = X_rna[:, top_idx]
top_genes = [gene_cols[i] for i in top_idx]
X_rna   = (X_rna - X_rna.mean(axis=0)) / (X_rna.std(axis=0) + 1e-8)
print(f"\nRNA特征: {X_rna.shape}")
# ============================================================
# 6. 5折CV（按CN_HIGH分层）
# ============================================================
cn_label = (df_meta['subtype'] == 'UCEC_CN_HIGH').astype(int).values
skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits   = [{'train': tr.tolist(), 'test': te.tolist()}
             for tr, te in skf.split(X_rna, cn_label)]
print(f"5折CV: 每折测试~{len(splits[0]['test'])}人")
# ============================================================
# 7. 保存PKL
# ============================================================
data_out = {
    'x_omic':     X_rna,                                    # (N, 320)
    'h5_paths':   df_meta['h5_paths'].tolist(),              # 每人的h5路径列表
    'survtime':   df_meta['os_days'].values,
    'censorship': (1 - df_meta['os_event'].values).astype(int),
    'subtype':    df_meta['subtype'].values,
    'patient_id': df_meta['patient'].values,
    'top_genes':  top_genes,
    'splits':     splits,
}
with open(OUT_PKL, 'wb') as f:
    pickle.dump(data_out, f)
print(f"\n✅ PKL保存: {OUT_PKL}")
print(f"总样本: {len(df_meta)}")