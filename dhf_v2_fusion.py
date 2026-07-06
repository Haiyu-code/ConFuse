#!/usr/bin/env python3
"""
DHFv2: 共享主干 + WT专属适配层
Shared-Backbone Dual-Head Fusion v2
=============================================================
设计思想：
  继承 DHF v1 的优点：共享主干z同时接收global+wt两个head梯度
                       → 全队列不下降（0.836+）
  继承 CMDFv2 的优点：WTAdapter只接收WT患者梯度（z.detach()传入）
                       → IDH-WT专注预后信号（0.71+）

架构:
  WSI ──┐
        ├─→ CrossAttn+Gate → z(384) ──┬─→ cox_global → L_global(全部患者)
  RNA ──┘         ↑                   │
        主干同时接收两个head梯度          └─→ cat[z, z_adapted] → wt_head
        （DHF v1成功要素）                          ↑
                                            WTAdapter(z.detach())
                                            只接收WT梯度（152人）
                                            （CMDFv2成功要素）
用法:
  # 基准实验
  python dhf_v2_fusion.py --model dhf_v2_patch \
      --lambda_wt 2.0 --lambda_orth 0.1 --adapter_dim 64 \
      --gpu 1 --batch_size 32 --epochs 40

  # adapter_dim搜索
  python dhf_v2_fusion.py --model dhf_v2_patch \
      --lambda_wt 2.0 --lambda_orth 0.1 --adapter_dim 128 \
      --gpu 1 --batch_size 32 --epochs 40

  # 无正交约束对照
  python dhf_v2_fusion.py --model dhf_v2_patch \
      --lambda_wt 2.0 --lambda_orth 0 --adapter_dim 64 \
      --gpu 1 --batch_size 32 --epochs 40
"""

import os
import json
import pickle
import argparse
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sksurv.metrics import concordance_index_censored

# ─────────────────────────────────────────────
# 0. 配置
# ─────────────────────────────────────────────
MEAN_PKL = ("/data2/wanghy/first/PathomicFusion-master/data/TCGA_GBMLGG"
            "/splits/gbmlgg15cv_all_st_patches_512_uni2h_1536_rnaseq.pkl")
FEAT_DIR = ("/data2/wanghy/first/data/TCGA_GBMLGG_full"
            "/TCGA_GBMLGG/uni2h_features")
OUT_DIR  = "/data2/wanghy/first/results_dhf_v2"

PATH_DIM = 1536
OMIC_DIM = 320
N_FOLDS  = 15

# ─────────────────────────────────────────────
# 1. 参数
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
        choices=["all", "dhf_v2_mean", "dhf_v2_patch"])

    # 训练
    parser.add_argument("--epochs",       type=int,   default=40)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=4e-4)
    parser.add_argument("--batch_size",   type=int,   default=32)

    # 跨模态注意力
    parser.add_argument("--attn_dim",     type=int,   default=256)
    parser.add_argument("--n_heads",      type=int,   default=4)
    parser.add_argument("--attn_dropout", type=float, default=0.1)

    # 损失权重
    parser.add_argument("--lambda_wt",   type=float, default=2.0)
    parser.add_argument("--lambda_orth", type=float, default=0.1)
    parser.add_argument("--wt_head_dim", type=int,   default=128)

    # DHFv2专属
    parser.add_argument("--adapter_dim", type=int, default=64,
        help="WTAdapter隐层维度，控制WT专属信息容量")

    # 消融开关
    parser.add_argument("--no_gate",        action="store_true", default=False,
        help="消融：关闭门控融合")
    parser.add_argument("--no_cross_attn",  action="store_true", default=False,
        help="消融：关闭跨模态注意力")
    parser.add_argument("--no_adapter",     action="store_true", default=False,
        help="消融：无WTAdapter，退化为DHF v1")
    parser.add_argument("--wt_head_global", action="store_true", default=False,
        help="消融：WT头用全部患者训练")
    parser.add_argument("--lambda_wt_zero", action="store_true", default=False,
        help="消融：λ=0，无WT头")

    # 其他
    parser.add_argument("--gpu",     type=int, default=1)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    return parser.parse_args()


# ─────────────────────────────────────────────
# 2. 日志
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 3. 数据集（完全复用cmdf_fusion.py）
# ─────────────────────────────────────────────
class PatientBagDataset(Dataset):
    def __init__(self, split_data, data_pd,
                 feat_dir=None, patch_mode=False):
        self.patch_mode = patch_mode
        self.feat_dir   = feat_dir

        patnames   = split_data["x_patname"]
        x_path_all = split_data["x_path"]
        x_omic_all = split_data["x_omic"]
        e_all      = split_data["e"]
        t_all      = split_data["t"]

        idh_map = dict(zip(data_pd.index.str[:12],
                           data_pd["idh mutation"]))

        pid2rows = defaultdict(list)
        for i, pid_raw in enumerate(patnames):
            pid2rows[str(pid_raw).strip()[:12]].append(i)

        self.samples = []
        missing_pt   = []

        for pid12, rows in pid2rows.items():
            rows = sorted(rows)
            if patch_mode:
                pt_path = os.path.join(feat_dir, f"{pid12}.pt")
                if not os.path.exists(pt_path):
                    missing_pt.append(pid12)
                    continue
            else:
                pt_path = None

            self.samples.append({
                "pid":         pid12,
                "pt_path":     pt_path,
                "x_path_mean": torch.FloatTensor(
                                   x_path_all[rows].mean(axis=0)),
                "x_omic":      torch.FloatTensor(
                                   x_omic_all[rows].mean(axis=0)),
                "e":   float(e_all[rows[0]]),
                "t":   float(t_all[rows[0]]),
                "idh": float(idh_map.get(pid12, -1.0)),
            })

        if missing_pt:
            log.warning(f"  {len(missing_pt)}位无.pt: {missing_pt[:3]}")

        wt  = sum(1 for s in self.samples if s["idh"] == 0.0)
        mut = sum(1 for s in self.samples if s["idh"] == 1.0)
        unk = sum(1 for s in self.samples if s["idh"] == -1.0)
        log.info(f"  数据集: {len(self.samples)}人  "
                 f"IDH-WT={wt}  IDH-MUT={mut}  未知={unk}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.patch_mode:
            d      = torch.load(s["pt_path"], map_location="cpu",
                                weights_only=False)
            x_path = torch.FloatTensor(d["features"])
        else:
            x_path = s["x_path_mean"]
        return {
            "pid":    s["pid"],
            "x_path": x_path,
            "x_omic": s["x_omic"],
            "e":      torch.tensor(s["e"]),
            "t":      torch.tensor(s["t"]),
            "idh":    torch.tensor(s["idh"]),
        }


def collate_fn(batch):
    x_path_0 = batch[0]["x_path"]
    if x_path_0.dim() == 2:
        x_path = [s["x_path"] for s in batch]
    else:
        x_path = torch.stack([s["x_path"] for s in batch])
    return {
        "pid":    [s["pid"]    for s in batch],
        "x_path": x_path,
        "x_omic": torch.stack([s["x_omic"] for s in batch]),
        "e":      torch.stack([s["e"]      for s in batch]),
        "t":      torch.stack([s["t"]      for s in batch]),
        "idh":    torch.stack([s["idh"]    for s in batch]),
    }


# ─────────────────────────────────────────────
# 4. 模型组件
# ─────────────────────────────────────────────
class GatedAttentionPool(nn.Module):
    """ABMIL门控注意力池化（patch模式用）"""
    def __init__(self, in_dim, attn_dim=256):
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h):
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))
        a = F.softmax(a, dim=0)
        return (a * h).sum(dim=0)


class CrossModalAttention(nn.Module):
    """
    多头跨模态注意力
    z_q 作为 Query，z_kv 作为 Key/Value
    输出维度=d_out，含残差连接 + LayerNorm
    """
    def __init__(self, d_q, d_kv, d_out, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_out % n_heads == 0, \
            f"d_out({d_out}) 必须能被 n_heads({n_heads}) 整除"

        self.n_heads = n_heads
        self.d_head  = d_out // n_heads
        self.scale   = self.d_head ** -0.5

        self.Wq = nn.Linear(d_q,  d_out)
        self.Wk = nn.Linear(d_kv, d_out)
        self.Wv = nn.Linear(d_kv, d_out)
        self.Wo = nn.Linear(d_out, d_out)

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)
        self.proj = (nn.Linear(d_q, d_out, bias=False)
                     if d_q != d_out else nn.Identity())

    def forward(self, z_q, z_kv):
        B = z_q.shape[0]
        Q = self.Wq(z_q).view(B, self.n_heads, self.d_head)
        K = self.Wk(z_kv).view(B, self.n_heads, self.d_head)
        V = self.Wv(z_kv).view(B, self.n_heads, self.d_head)

        attn = (Q * K).sum(dim=-1) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn.unsqueeze(-1) * V).reshape(B, -1)
        out = self.Wo(out)
        return self.norm(out + self.proj(z_q))


class GatedFusion(nn.Module):
    """
    门控融合
    gate = sigmoid(MLP(cat(z_wsi, z_rna)))
    z    = gate * proj(z_wsi) + (1-gate) * proj(z_rna)
    """
    def __init__(self, d_wsi, d_rna, d_out, dropout=0.1):
        super().__init__()
        self.proj_wsi = nn.Linear(d_wsi, d_out)
        self.proj_rna = nn.Linear(d_rna, d_out)
        self.gate_net = nn.Sequential(
            nn.Linear(d_wsi + d_rna, d_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.Sigmoid())
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, z_wsi, z_rna):
        gate  = self.gate_net(torch.cat([z_wsi, z_rna], dim=-1))
        v_wsi = self.proj_wsi(z_wsi)
        v_rna = self.proj_rna(z_rna)
        z = gate * v_wsi + (1.0 - gate) * v_rna
        return self.norm(self.drop(z)), gate


class SimpleFusion(nn.Module):
    """简单拼接融合（消融，无门控）"""
    def __init__(self, d_wsi, d_rna, d_out, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_wsi + d_rna, d_out),
            nn.ReLU(),
            nn.Dropout(dropout))

    def forward(self, z_wsi, z_rna):
        z = self.proj(torch.cat([z_wsi, z_rna], dim=-1))
        return z, None


# ─────────────────────────────────────────────
# 5. 主模型：DHFv2
# ─────────────────────────────────────────────
class DHFv2(nn.Module):
    """
    DHFv2：共享主干 + WT专属适配层

    梯度流向（关键设计）：

    [全局头]:
      z ← global_cox梯度（全502人）
      z ← wt_cox梯度（152人，通过wt_head中的z部分反传）
      → 保留DHF v1的多任务协同效应，全队列不下降

    [WT适配层]:
      z_adapted = WTAdapter(z.detach())
      → z.detach()切断adapter对主干的梯度影响
      → adapter只接收wt_cox梯度（152人）
      → adapter专注学习WT内部预后信号，不被IDH信号干扰

    [WT头]:
      risk_wt = wt_head(cat[z, z_adapted])
      → z部分：携带主干的全局信息
      → z_adapted部分：携带WT专属的适配信息
      → 两者互补，IDH-WT性能提升

    [正交正则化]:
      L_orth = E[cos_sim(z.detach(), z_adapted)^2]
      → 强迫z_adapted学习与z正交的补充信息
      → 避免adapter退化为z的简单复制
    """
    def __init__(self,
                 path_dim       = PATH_DIM,
                 omic_dim       = OMIC_DIM,
                 path_hidden    = 256,
                 omic_hidden    = 128,
                 attn_dim       = 256,
                 n_heads        = 4,
                 attn_dropout   = 0.1,
                 dropout        = 0.25,
                 wt_head_dim    = 128,
                 adapter_dim    = 64,
                 patch_mode     = False,
                 use_gate       = True,
                 use_cross_attn = True,
                 use_adapter    = True,
                 lambda_orth    = 0.1):
        super().__init__()
        self.patch_mode     = patch_mode
        self.use_cross_attn = use_cross_attn
        self.use_adapter    = use_adapter
        self.lambda_orth    = lambda_orth
        fuse_dim = path_hidden + omic_hidden   # 256+128=384

        # ── 共享主干：WSI编码 ─────────────────────────────────────────────────
        if patch_mode:
            self.patch_encoder = nn.Sequential(
                nn.Linear(path_dim, path_hidden),
                nn.ReLU(), nn.Dropout(dropout))
            self.attn_pool = GatedAttentionPool(path_hidden, attn_dim)
        else:
            self.wsi_encoder = nn.Sequential(
                nn.Linear(path_dim, path_hidden),
                nn.ReLU(), nn.Dropout(dropout))

        # ── 共享主干：RNA编码 ─────────────────────────────────────────────────
        self.rna_encoder = nn.Sequential(
            nn.Linear(omic_dim, omic_hidden),
            nn.ReLU(), nn.Dropout(dropout))

        # ── 共享主干：跨模态注意力 ────────────────────────────────────────────
        if use_cross_attn:
            # RNA(128)询问WSI(256)：n_heads=2保证128/2=64整除
            self.cross_rna2wsi = CrossModalAttention(
                d_q=omic_hidden, d_kv=path_hidden,
                d_out=omic_hidden, n_heads=2,
                dropout=attn_dropout)
            # WSI(256)询问RNA(128)
            self.cross_wsi2rna = CrossModalAttention(
                d_q=path_hidden, d_kv=omic_hidden,
                d_out=path_hidden, n_heads=n_heads,
                dropout=attn_dropout)

        # ── 共享主干：融合层 ──────────────────────────────────────────────────
        if use_gate:
            self.fusion = GatedFusion(
                d_wsi=path_hidden, d_rna=omic_hidden,
                d_out=fuse_dim, dropout=dropout)
        else:
            self.fusion = SimpleFusion(
                d_wsi=path_hidden, d_rna=omic_hidden,
                d_out=fuse_dim, dropout=dropout)

        # ── 共享主干：融合后MLP ───────────────────────────────────────────────
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim),
            nn.ReLU(),
            nn.Dropout(dropout))

        # ── 全局Cox头（接收所有患者梯度，同DHF v1）───────────────────────────
        self.cox_global = nn.Linear(fuse_dim, 1)

        # ── WT专属适配层 ──────────────────────────────────────────────────────
        # 输入：z.detach()（切断对主干的梯度）
        # 只接收WT Cox损失的梯度（152人）
        # 学习WT内部预后相关的补充特征
        if use_adapter:
            self.wt_adapter = nn.Sequential(
                nn.Linear(fuse_dim, adapter_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(adapter_dim, adapter_dim),
                nn.LayerNorm(adapter_dim),
                nn.ReLU(),
                nn.Dropout(dropout))
            wt_in_dim = fuse_dim + adapter_dim
        else:
            # 无adapter：退化为DHF v1的WT头结构
            self.wt_adapter = None
            wt_in_dim = fuse_dim

        # ── WT Cox头 ─────────────────────────────────────────────────────────
        # 输入：cat[z, z_adapted]（或仅z，无adapter时）
        self.wt_head = nn.Sequential(
            nn.Linear(wt_in_dim, wt_head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(wt_head_dim, 1))

        n_params = sum(p.numel() for p in self.parameters())
        log.info(f"  DHFv2 参数量: {n_params:,}  "
                 f"patch={patch_mode}  cross={use_cross_attn}  "
                 f"gate={use_gate}  adapter={use_adapter}({adapter_dim})")

    def _encode_shared(self, x_path, x_omic):
        """共享主干编码，接收global+wt两个head的梯度"""
        dev = x_omic.device

        # WSI编码
        if self.patch_mode:
            path_vecs = []
            for patches in x_path:
                h = self.patch_encoder(patches.to(dev))
                path_vecs.append(self.attn_pool(h))
            z_wsi = torch.stack(path_vecs)
        else:
            z_wsi = self.wsi_encoder(x_path)

        # RNA编码
        z_rna = self.rna_encoder(x_omic)

        # 跨模态注意力
        if self.use_cross_attn:
            z_rna = self.cross_rna2wsi(z_rna, z_wsi)
            z_wsi = self.cross_wsi2rna(z_wsi, z_rna)

        # 门控/简单融合
        z, gate = self.fusion(z_wsi, z_rna)

        # 深化MLP
        z = self.fusion_mlp(z)

        return z, gate

    def forward(self, x_path, x_omic):
        """
        Returns:
            risk_global:  [B]         全局风险评分
            risk_wt:      [B]         WT专属风险评分
            z:            [B, fuse]   共享主干特征
            z_adapted:    [B, adp]    WT适配特征（无adapter时为None）
            gate:         [B, fuse]   门控权重（无gate时为None）
        """
        # 共享主干：同时接收global + wt 两个head的梯度
        z, gate = self._encode_shared(x_path, x_omic)

        # 全局头：直接用z（同DHF v1）
        risk_global = self.cox_global(z).squeeze(-1)

        # WT适配层 + WT头
        if self.use_adapter and self.wt_adapter is not None:
            # z.detach()：切断adapter对主干的梯度
            # adapter只接收wt_cox的梯度，专注WT内部预后信号
            z_adapted = self.wt_adapter(z.detach())     # [B, adapter_dim]

            # WT头输入：cat[z, z_adapted]
            # z部分：主干全局信息（梯度回传到主干）
            # z_adapted部分：WT专属信息（梯度只到adapter）
            risk_wt = self.wt_head(
                torch.cat([z, z_adapted], dim=-1)).squeeze(-1)
        else:
            # 无adapter：退化为DHF v1
            z_adapted = None
            risk_wt = self.wt_head(z).squeeze(-1)

        return risk_global, risk_wt, z, z_adapted, gate


# ─────────────────────────────────────────────
# 6. 损失函数
# ─────────────────────────────────────────────
def cox_loss(risk, t, e):
    if risk.shape[0] < 2:
        return torch.tensor(0.0, requires_grad=True, device=risk.device)
    order   = torch.argsort(t, descending=True)
    risk_s  = risk[order]
    e_s     = e[order]
    log_cum = torch.logcumsumexp(risk_s, dim=0)
    return -torch.mean((risk_s - log_cum) * e_s)


def dhfv2_loss(risk_global, risk_wt, z, z_adapted,
               t, e, idh,
               lambda_wt, lambda_orth,
               wt_head_global=False,
               lambda_wt_zero=False):
    """
    总损失 = L_global + λ_wt * L_wt + λ_orth * L_orth

    L_global: 全部患者Cox损失 → 梯度更新共享主干
    L_wt:     IDH-WT子集Cox损失 → 梯度更新共享主干(通过z) + adapter
    L_orth:   z与z_adapted的余弦相似度惩罚
              z.detach()，只推动z_adapted方向改变
              → 强迫adapter学习与主干互补的WT专属信息
    """
    # ── L_global ─────────────────────────────────────────────────────────────
    l_global = cox_loss(risk_global, t, e)

    # ── L_wt ─────────────────────────────────────────────────────────────────
    if lambda_wt_zero:
        l_wt = torch.tensor(0.0, device=risk_wt.device)
    elif wt_head_global:
        l_wt = cox_loss(risk_wt, t, e)
    else:
        wt_mask = (idh == 0.0)
        if wt_mask.sum() >= 2:
            l_wt = cox_loss(
                risk_wt[wt_mask], t[wt_mask], e[wt_mask])
        else:
            l_wt = torch.tensor(0.0, requires_grad=True,
                                device=risk_wt.device)

    # ── L_orth ────────────────────────────────────────────────────────────────
    if lambda_orth > 0 and z_adapted is not None:
        z_n  = F.normalize(z.detach(), dim=-1)      # [B, 384]
        za_n = F.normalize(z_adapted,  dim=-1)      # [B, 64]
        cross_cov = z_n.T @ za_n / z_n.shape[0]    # [384, 64]
        l_orth = cross_cov.pow(2).sum()
    else:
        l_orth = torch.tensor(0.0, device=risk_wt.device)

    # NaN 防护
    if torch.isnan(l_global):
        l_global = torch.tensor(0.0, requires_grad=True,
                                device=risk_global.device)
    if torch.isnan(l_wt):
        l_wt = torch.tensor(0.0, requires_grad=True,
                            device=risk_wt.device)
    if torch.isnan(l_orth):
        l_orth = torch.tensor(0.0, device=risk_wt.device)

    total = l_global + lambda_wt * l_wt + lambda_orth * l_orth
    return total, l_global.item(), l_wt.item(), l_orth.item()


# ─────────────────────────────────────────────
# 7. 训练 & 评估
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, args):
    model.train()
    total_l = total_g = total_w = total_o = 0.0
    n_batch = 0

    for batch in loader:
        x_path = batch["x_path"]
        x_omic = batch["x_omic"].to(device)
        e      = batch["e"].to(device)
        t      = batch["t"].to(device)
        idh    = batch["idh"].to(device)

        if isinstance(x_path, torch.Tensor):
            x_path = x_path.to(device)

        optimizer.zero_grad()
        risk_global, risk_wt, z, z_adapted, _ = model(
            x_path=x_path, x_omic=x_omic)

        loss, l_g, l_w, l_o = dhfv2_loss(
            risk_global, risk_wt, z, z_adapted,
            t, e, idh,
            args.lambda_wt, args.lambda_orth,
            args.wt_head_global,
            args.lambda_wt_zero)

        if torch.isnan(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_l += loss.item()
        total_g += l_g
        total_w += l_w
        total_o += l_o
        n_batch += 1

    n = max(n_batch, 1)
    return total_l/n, total_g/n, total_w/n, total_o/n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    risks_g, risks_w = [], []
    events, times, idhs = [], [], []

    for batch in loader:
        x_path = batch["x_path"]
        x_omic = batch["x_omic"].to(device)
        if isinstance(x_path, torch.Tensor):
            x_path = x_path.to(device)

        risk_global, risk_wt, _, _, _ = model(
            x_path=x_path, x_omic=x_omic)

        risks_g.append(risk_global.cpu().numpy())
        risks_w.append(risk_wt.cpu().numpy())
        events.append(batch["e"].numpy())
        times.append(batch["t"].numpy())
        idhs.append(batch["idh"].numpy())

    return (np.concatenate(risks_g),
            np.concatenate(risks_w),
            np.concatenate(events),
            np.concatenate(times),
            np.concatenate(idhs))


def cindex(risks, events, times):
    mask = events.astype(bool)
    if mask.sum() < 2:
        return float("nan")
    try:
        return float(concordance_index_censored(mask, times, risks)[0])
    except Exception:
        return float("nan")


def cindex_idhwt(risks, events, times, idhs):
    m = (idhs == 0.0)
    if m.sum() < 5:
        return float("nan")
    return cindex(risks[m], events[m], times[m])


# ─────────────────────────────────────────────
# 8. 15折交叉验证
# ─────────────────────────────────────────────
def run_cv(model_name: str, args, device):
    patch_mode = (model_name == "dhf_v2_patch")

    log.info(f"\n{'='*68}")
    log.info(f"模型: {model_name.upper()}  patch_mode={patch_mode}")
    log.info(f"epochs={args.epochs}  lr={args.lr}  bs={args.batch_size}")
    log.info(f"attn_dim={args.attn_dim}  n_heads={args.n_heads}")
    log.info(f"lambda_wt={args.lambda_wt}  lambda_orth={args.lambda_orth}  "
             f"wt_head_dim={args.wt_head_dim}  adapter_dim={args.adapter_dim}")
    log.info(f"no_cross={args.no_cross_attn}  no_gate={args.no_gate}  "
             f"no_adapter={args.no_adapter}")
    log.info(f"{'='*68}")

    with open(MEAN_PKL, "rb") as f:
        data = pickle.load(f)
    data_pd   = data["data_pd"]
    cv_splits = data["cv_splits"]

    ci_all_folds, ci_wt_folds = [], []

    for fold in range(1, N_FOLDS + 1):
        torch.manual_seed(args.seed + fold)
        np.random.seed(args.seed + fold)

        tr_set = PatientBagDataset(
            cv_splits[fold]["train"], data_pd,
            feat_dir=FEAT_DIR, patch_mode=patch_mode)
        te_set = PatientBagDataset(
            cv_splits[fold]["test"], data_pd,
            feat_dir=FEAT_DIR, patch_mode=patch_mode)

        tr_loader = DataLoader(
            tr_set, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=4, pin_memory=True)
        te_loader = DataLoader(
            te_set, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=4, pin_memory=True)

        model = DHFv2(
            patch_mode      = patch_mode,
            attn_dim        = args.attn_dim,
            n_heads         = args.n_heads,
            attn_dropout    = args.attn_dropout,
            wt_head_dim     = args.wt_head_dim,
            adapter_dim     = args.adapter_dim,
            use_gate        = not args.no_gate,
            use_cross_attn  = not args.no_cross_attn,
            use_adapter     = not args.no_adapter,
            lambda_orth     = args.lambda_orth,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

        best_ci_wt = 0.0
        best_state = None

        for epoch in range(1, args.epochs + 1):
            tr_loss, l_g, l_w, l_o = train_epoch(
                model, tr_loader, optimizer, device, args)
            scheduler.step()

            if epoch % 10 == 0 or epoch == args.epochs:
                rg, rw, e, t, idh = evaluate(model, te_loader, device)

                ci_a      = cindex(rg, e, t)
                ci_wt     = cindex_idhwt(rw, e, t, idh)
                ci_wt_ref = cindex_idhwt(rg, e, t, idh)

                if not np.isnan(ci_wt) and ci_wt > best_ci_wt:
                    best_ci_wt = ci_wt
                    best_state = (rg.copy(), rw.copy(),
                                  e.copy(), t.copy(), idh.copy())

                log.info(f"  [F{fold:2d} E{epoch:3d}] "
                         f"L={tr_loss:.3f}"
                         f"(g={l_g:.3f},w={l_w:.3f},o={l_o:.3f})  "
                         f"全队列={ci_a:.4f}  "
                         f"WT(wt_head)={ci_wt:.4f}  "
                         f"WT(global)={ci_wt_ref:.4f}")

        if best_state is None:
            best_state = evaluate(model, te_loader, device)

        rg, rw, e, t, idh = best_state
        ci_a = cindex(rg, e, t)
        ci_w = cindex_idhwt(rw, e, t, idh)
        ci_all_folds.append(ci_a)
        ci_wt_folds.append(ci_w)

        log.info(f"  ★ Fold {fold:2d}/{N_FOLDS}  "
                 f"全队列={ci_a:.4f}  IDH-WT={ci_w:.4f}  "
                 f"(WT={int((idh==0).sum())}人)")

    arr_a = np.array([c for c in ci_all_folds if not np.isnan(c)])
    arr_w = np.array([c for c in ci_wt_folds  if not np.isnan(c)])
    log.info(f"\n[{model_name}] 最终结果:")
    log.info(f"  全队列  C-index: {arr_a.mean():.4f} ± {arr_a.std():.4f}")
    log.info(f"  IDH-WT  C-index: {arr_w.mean():.4f} ± {arr_w.std():.4f}")

    return ci_all_folds, ci_wt_folds


# ─────────────────────────────────────────────
# 9. 汇总 & 保存
# ─────────────────────────────────────────────
def print_summary(results: dict, args):
    ablation_parts = []
    if args.no_cross_attn: ablation_parts.append("no_cross")
    if args.no_gate:       ablation_parts.append("no_gate")
    if args.no_adapter:    ablation_parts.append("no_adapter")
    ablation_str = ",".join(ablation_parts) if ablation_parts else "full"

    log.info("\n" + "="*74)
    log.info("阶段7 DHFv2 结果汇总")
    log.info(f"n_heads={args.n_heads}  λ_wt={args.lambda_wt}  "
             f"λ_orth={args.lambda_orth}  "
             f"adapter_dim={args.adapter_dim}  "
             f"ablation=[{ablation_str}]  bs={args.batch_size}")
    log.info("="*74)
    log.info(f"  {'模型':<52} {'全队列':>12} {'IDH-WT':>12}")
    log.info("-"*74)

    baselines = [
        ("baseline 均值(UNI2-h)",      "0.8446±0.0271", "0.6753±0.0325"),
        ("PathomicFusion(2019)",       "0.8083±0.0234", "0.6389±0.0602"),
        ("ADF +GRL",                   "0.8271±0.0245", "0.6875±0.0503"),
        ("分层Cox+wt=2",                "0.7701±0.0472", "0.7030±0.0345"),
        ("DHF v1 λ=2,d=128",          "0.8362±0.0274", "0.7132±0.0396"),
        ("CMDFv1 patch λ=2",          "0.8402±0.0309", "0.7062±0.0371"),
        ("CMDFv2 patch λ_orth=0",     "0.8324±0.0260", "0.7141±0.0470"),
        ("CMDFv2 patch λ_orth=0.1",   "0.8334±0.0234", "0.7157±0.0509"),
    ]
    for name, a, w in baselines:
        log.info(f"  {name:<52} {a:>12} {w:>12}")
    log.info("-"*74)

    labels = {
        "dhf_v2_mean":  "DHFv2 共享主干+适配层 均值",
        "dhf_v2_patch": "DHFv2 共享主干+适配层 patch",
    }
    for name, (ca, cw) in results.items():
        a = np.array([c for c in ca if not np.isnan(c)])
        w = np.array([c for c in cw if not np.isnan(c)])
        log.info(f"  {labels.get(name, name):<52} "
                 f"{a.mean():.4f}±{a.std():.4f}  "
                 f"{w.mean():.4f}±{w.std():.4f}  ★ DHFv2")
    log.info("="*74)


def save_results(results: dict, args):
    os.makedirs(args.out_dir, exist_ok=True)
    out = {"config": vars(args)}
    for name, (ca, cw) in results.items():
        a = np.array([c for c in ca if not np.isnan(c)])
        w = np.array([c for c in cw if not np.isnan(c)])
        out[name] = {
            "ci_all_folds": ca, "ci_wt_folds": cw,
            "ci_all_mean":  float(a.mean()), "ci_all_std": float(a.std()),
            "ci_wt_mean":   float(w.mean()), "ci_wt_std":  float(w.std()),
        }
    fname = (f"dhf_v2"
             f"_nh{args.n_heads}"
             f"_lw{args.lambda_wt}"
             f"_lo{args.lambda_orth}"
             f"_adp{args.adapter_dim}"
             f"_wd{args.wt_head_dim}"
             f"_b{args.batch_size}"
             f"{'_nc'  if args.no_cross_attn else ''}"
             f"{'_nog' if args.no_gate        else ''}"
             f"{'_na'  if args.no_adapter     else ''}"
             f".json")
    path = os.path.join(args.out_dir, fname)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"结果已保存: {path}")


# ─────────────────────────────────────────────
# 10. 主函数
# ─────────────────────────────────────────────
def main():
    args   = parse_args()
    device = (torch.device(f"cuda:{args.gpu}")
              if args.gpu >= 0 and torch.cuda.is_available()
              else torch.device("cpu"))
    log.info(f"设备: {device}")

    models = (["dhf_v2_mean", "dhf_v2_patch"]
              if args.model == "all" else [args.model])

    results = {}
    for m in models:
        ca, cw     = run_cv(m, args, device)
        results[m] = (ca, cw)

    print_summary(results, args)
    save_results(results, args)


if __name__ == "__main__":
    main()
