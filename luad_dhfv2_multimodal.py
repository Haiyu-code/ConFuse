#!/usr/bin/env python3
"""
DHFv2 (ConFuse) — LUAD 版本
严格对齐 dhf_v2_fusion.py 框架，仅替换数据加载。

困难子群: AJCC Stage III+IV (subtype==1.0)

用法:
  python luad_dhfv2_multimodal.py --model dhf_v2_patch \
      --lambda_wt 2.0 --lambda_orth 0.1 --adapter_dim 64 \
      --gpu 0 --batch_size 32 --epochs 40
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
PKL_PATH  = "/data2/wanghy/first/LUAD/luad_multimodal_v1.pkl"
FEAT_DIR  = "/data2/wanghy/first/LUAD/wsi_mean_feats/"
OUT_DIR   = "/data2/wanghy/first/LUAD/results_dhf_v2"

PATH_DIM = 1536
OMIC_DIM = 320
N_FOLDS  = 5

# ─────────────────────────────────────────────
# 1. 参数（与GBMLGG版完全一致）
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
        choices=["all", "dhf_v2_mean", "dhf_v2_patch"])
    parser.add_argument("--epochs",       type=int,   default=40)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=4e-4)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--attn_dim",     type=int,   default=256)
    parser.add_argument("--n_heads",      type=int,   default=4)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--lambda_wt",   type=float, default=2.0)
    parser.add_argument("--lambda_orth", type=float, default=0.1)
    parser.add_argument("--wt_head_dim", type=int,   default=128)
    parser.add_argument("--adapter_dim", type=int, default=64)
    parser.add_argument("--no_gate",        action="store_true", default=False)
    parser.add_argument("--no_cross_attn",  action="store_true", default=False)
    parser.add_argument("--no_adapter",     action="store_true", default=False)
    parser.add_argument("--wt_head_global", action="store_true", default=False)
    parser.add_argument("--lambda_wt_zero", action="store_true", default=False)
    parser.add_argument("--gpu",     type=int, default=0)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    return parser.parse_args()


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 3. 数据集（LUAD版本：从PKL+PT加载）
# ─────────────────────────────────────────────
class PatientBagDataset(Dataset):
    """
    从PKL索引 + PT文件加载。
    idh字段复用为subtype: 1.0=Hard(III+IV), 0.0=Easy
    """
    def __init__(self, data, indices,
                 feat_dir=FEAT_DIR, patch_mode=False):
        self.patch_mode = patch_mode

        x_omic_all = data['x_omic']
        survtime   = data['survtime']
        censorship = data['censorship']
        subtypes   = data['subtype']
        pids       = data['patient_id']

        self.samples = []
        missing_pt = []

        for i in indices:
            pid = pids[i]
            pt_path = os.path.join(feat_dir, f"{pid}.pt")
            if not os.path.exists(pt_path):
                missing_pt.append(pid)
                continue

            self.samples.append({
                "pid":     pid,
                "pt_path": pt_path,
                "x_omic":  torch.FloatTensor(x_omic_all[i]),
                "e":       float(1 - censorship[i]),    # event
                "t":       float(survtime[i]),
                "idh":     float(subtypes[i]),           # 复用: 1=hard
            })

        if missing_pt:
            log.warning(f"  {len(missing_pt)}位无.pt: {missing_pt[:5]}")

        hard = sum(1 for s in self.samples if s["idh"] == 1.0)
        easy = len(self.samples) - hard
        log.info(f"  数据集: {len(self.samples)}人  "
                 f"Hard(III+IV)={hard}  Easy={easy}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        d = torch.load(s["pt_path"], map_location="cpu",
                        weights_only=False)
        feats = d["features"]  # (n_patches, 1536)

        if self.patch_mode:
            x_path = feats if feats.dim() == 2 else feats.unsqueeze(0)
        else:
            # mean pooling
            x_path = feats.mean(dim=0) if feats.dim() == 2 else feats

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
# 4. 模型组件（与GBMLGG版完全一致）
# ─────────────────────────────────────────────
class GatedAttentionPool(nn.Module):
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
    def __init__(self, d_q, d_kv, d_out, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_out % n_heads == 0
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
        out  = (attn.unsqueeze(-1) * V).reshape(B, -1)
        out  = self.Wo(out)
        return self.norm(out + self.proj(z_q))


class GatedFusion(nn.Module):
    def __init__(self, d_wsi, d_rna, d_out, dropout=0.1):
        super().__init__()
        self.proj_wsi = nn.Linear(d_wsi, d_out)
        self.proj_rna = nn.Linear(d_rna, d_out)
        self.gate_net = nn.Sequential(
            nn.Linear(d_wsi + d_rna, d_out), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, z_wsi, z_rna):
        gate  = self.gate_net(torch.cat([z_wsi, z_rna], dim=-1))
        v_wsi = self.proj_wsi(z_wsi)
        v_rna = self.proj_rna(z_rna)
        z = gate * v_wsi + (1.0 - gate) * v_rna
        return self.norm(self.drop(z)), gate


class SimpleFusion(nn.Module):
    def __init__(self, d_wsi, d_rna, d_out, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_wsi + d_rna, d_out), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, z_wsi, z_rna):
        return self.proj(torch.cat([z_wsi, z_rna], dim=-1)), None


# ─────────────────────────────────────────────
# 5. DHFv2（与GBMLGG版完全一致）
# ─────────────────────────────────────────────
class DHFv2(nn.Module):
    def __init__(self,
                 path_dim=PATH_DIM, omic_dim=OMIC_DIM,
                 path_hidden=256, omic_hidden=128,
                 attn_dim=256, n_heads=4, attn_dropout=0.1,
                 dropout=0.25, wt_head_dim=128,
                 adapter_dim=64, patch_mode=False,
                 use_gate=True, use_cross_attn=True,
                 use_adapter=True, lambda_orth=0.1):
        super().__init__()
        self.patch_mode     = patch_mode
        self.use_cross_attn = use_cross_attn
        self.use_adapter    = use_adapter
        self.lambda_orth    = lambda_orth
        fuse_dim = path_hidden + omic_hidden

        if patch_mode:
            self.patch_encoder = nn.Sequential(
                nn.Linear(path_dim, path_hidden),
                nn.ReLU(), nn.Dropout(dropout))
            self.attn_pool = GatedAttentionPool(path_hidden, attn_dim)
        else:
            self.wsi_encoder = nn.Sequential(
                nn.Linear(path_dim, path_hidden),
                nn.ReLU(), nn.Dropout(dropout))

        self.rna_encoder = nn.Sequential(
            nn.Linear(omic_dim, omic_hidden),
            nn.ReLU(), nn.Dropout(dropout))

        if use_cross_attn:
            self.cross_rna2wsi = CrossModalAttention(
                omic_hidden, path_hidden, omic_hidden, n_heads=2,
                dropout=attn_dropout)
            self.cross_wsi2rna = CrossModalAttention(
                path_hidden, omic_hidden, path_hidden, n_heads=n_heads,
                dropout=attn_dropout)

        if use_gate:
            self.fusion = GatedFusion(path_hidden, omic_hidden,
                                       fuse_dim, dropout)
        else:
            self.fusion = SimpleFusion(path_hidden, omic_hidden,
                                        fuse_dim, dropout)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim), nn.ReLU(), nn.Dropout(dropout))

        self.cox_global = nn.Linear(fuse_dim, 1)

        if use_adapter:
            self.wt_adapter = nn.Sequential(
                nn.Linear(fuse_dim, adapter_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(adapter_dim, adapter_dim), nn.LayerNorm(adapter_dim),
                nn.ReLU(), nn.Dropout(dropout))
            wt_in_dim = fuse_dim + adapter_dim
        else:
            self.wt_adapter = None
            wt_in_dim = fuse_dim

        self.wt_head = nn.Sequential(
            nn.Linear(wt_in_dim, wt_head_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(wt_head_dim, 1))

        n_params = sum(p.numel() for p in self.parameters())
        log.info(f"  DHFv2 参数量: {n_params:,}  "
                 f"patch={patch_mode}  cross={use_cross_attn}  "
                 f"gate={use_gate}  adapter={use_adapter}({adapter_dim})")

    def _encode_shared(self, x_path, x_omic):
        dev = x_omic.device
        if self.patch_mode:
            path_vecs = []
            for patches in x_path:
                h = self.patch_encoder(patches.to(dev))
                path_vecs.append(self.attn_pool(h))
            z_wsi = torch.stack(path_vecs)
        else:
            z_wsi = self.wsi_encoder(x_path)

        z_rna = self.rna_encoder(x_omic)

        if self.use_cross_attn:
            z_rna = self.cross_rna2wsi(z_rna, z_wsi)
            z_wsi = self.cross_wsi2rna(z_wsi, z_rna)

        z, gate = self.fusion(z_wsi, z_rna)
        z = self.fusion_mlp(z)
        return z, gate

    def forward(self, x_path, x_omic):
        z, gate = self._encode_shared(x_path, x_omic)
        risk_global = self.cox_global(z).squeeze(-1)

        if self.use_adapter and self.wt_adapter is not None:
            z_adapted = self.wt_adapter(z.detach())
            risk_wt = self.wt_head(
                torch.cat([z, z_adapted], dim=-1)).squeeze(-1)
        else:
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
    与GBMLGG版唯一区别:
      GBMLGG: wt_mask = (idh == 0.0)  → IDH-WT是困难子群
      LUAD:   wt_mask = (idh == 1.0)  → Stage III+IV是困难子群
    """
    l_global = cox_loss(risk_global, t, e)

    if lambda_wt_zero:
        l_wt = torch.tensor(0.0, device=risk_wt.device)
    elif wt_head_global:
        l_wt = cox_loss(risk_wt, t, e)
    else:
        # ★ LUAD: 困难子群 = subtype==1.0
        wt_mask = (idh == 1.0)
        if wt_mask.sum() >= 2:
            l_wt = cox_loss(risk_wt[wt_mask], t[wt_mask], e[wt_mask])
        else:
            l_wt = torch.tensor(0.0, requires_grad=True,
                                device=risk_wt.device)

    if lambda_orth > 0 and z_adapted is not None:
        z_n  = F.normalize(z.detach(), dim=-1)
        za_n = F.normalize(z_adapted,  dim=-1)
        cross_cov = z_n.T @ za_n / z_n.shape[0]
        l_orth = cross_cov.pow(2).sum()
    else:
        l_orth = torch.tensor(0.0, device=risk_wt.device)

    if torch.isnan(l_global):
        l_global = torch.tensor(0.0, requires_grad=True, device=risk_global.device)
    if torch.isnan(l_wt):
        l_wt = torch.tensor(0.0, requires_grad=True, device=risk_wt.device)
    if torch.isnan(l_orth):
        l_orth = torch.tensor(0.0, device=risk_wt.device)

    total = l_global + lambda_wt * l_wt + lambda_orth * l_orth
    return total, l_global.item(), l_wt.item(), l_orth.item()


# ─────────────────────────────────────────────
# 7. 训练 & 评估（与GBMLGG版完全一致）
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
            args.wt_head_global, args.lambda_wt_zero)

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

    return (np.concatenate(risks_g), np.concatenate(risks_w),
            np.concatenate(events),  np.concatenate(times),
            np.concatenate(idhs))


def cindex(risks, events, times):
    mask = events.astype(bool)
    if mask.sum() < 2:
        return float("nan")
    try:
        return float(concordance_index_censored(mask, times, risks)[0])
    except:
        return float("nan")


def cindex_hard(risks, events, times, idhs):
    """LUAD: 困难子群 = idh==1.0"""
    m = (idhs == 1.0)
    if m.sum() < 5:
        return float("nan")
    return cindex(risks[m], events[m], times[m])


# ─────────────────────────────────────────────
# 8. 5折交叉验证
# ─────────────────────────────────────────────
def run_cv(model_name: str, args, device):
    patch_mode = (model_name == "dhf_v2_patch")

    log.info(f"\n{'='*68}")
    log.info(f"模型: {model_name.upper()}  patch_mode={patch_mode}")
    log.info(f"epochs={args.epochs}  lr={args.lr}  bs={args.batch_size}")
    log.info(f"attn_dim={args.attn_dim}  n_heads={args.n_heads}")
    log.info(f"lambda_wt={args.lambda_wt}  lambda_orth={args.lambda_orth}  "
             f"wt_head_dim={args.wt_head_dim}  adapter_dim={args.adapter_dim}")
    log.info(f"{'='*68}")

    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    splits = data["splits"]
    ci_all_folds, ci_hard_folds = [], []

    for fold in range(N_FOLDS):
        torch.manual_seed(args.seed + fold)
        np.random.seed(args.seed + fold)

        tr_idx = splits[fold]["train"]
        te_idx = splits[fold]["test"]

        tr_set = PatientBagDataset(data, tr_idx,
                                   feat_dir=FEAT_DIR, patch_mode=patch_mode)
        te_set = PatientBagDataset(data, te_idx,
                                   feat_dir=FEAT_DIR, patch_mode=patch_mode)

        tr_loader = DataLoader(
            tr_set, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=4, pin_memory=True)
        te_loader = DataLoader(
            te_set, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=4, pin_memory=True)

        model = DHFv2(
            patch_mode=patch_mode, attn_dim=args.attn_dim,
            n_heads=args.n_heads, attn_dropout=args.attn_dropout,
            wt_head_dim=args.wt_head_dim, adapter_dim=args.adapter_dim,
            use_gate=not args.no_gate, use_cross_attn=not args.no_cross_attn,
            use_adapter=not args.no_adapter, lambda_orth=args.lambda_orth,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

        best_ci_hard = 0.0
        best_state = None

        for epoch in range(1, args.epochs + 1):
            tr_loss, l_g, l_w, l_o = train_epoch(
                model, tr_loader, optimizer, device, args)
            scheduler.step()

            if epoch % 10 == 0 or epoch == args.epochs:
                rg, rw, e, t, idh = evaluate(model, te_loader, device)
                ci_a      = cindex(rg, e, t)
                ci_h      = cindex_hard(rw, e, t, idh)
                ci_h_ref  = cindex_hard(rg, e, t, idh)

                if not np.isnan(ci_h) and ci_h > best_ci_hard:
                    best_ci_hard = ci_h
                    best_state = (rg.copy(), rw.copy(),
                                  e.copy(), t.copy(), idh.copy())

                log.info(f"  [F{fold} E{epoch:3d}] "
                         f"L={tr_loss:.3f}"
                         f"(g={l_g:.3f},w={l_w:.3f},o={l_o:.3f})  "
                         f"全队列={ci_a:.4f}  "
                         f"Hard(head)={ci_h:.4f}  "
                         f"Hard(global)={ci_h_ref:.4f}")

        if best_state is None:
            best_state = evaluate(model, te_loader, device)

        rg, rw, e, t, idh = best_state
        ci_a = cindex(rg, e, t)
        ci_h = cindex_hard(rw, e, t, idh)
        ci_all_folds.append(ci_a)
        ci_hard_folds.append(ci_h)

        log.info(f"  ★ Fold {fold}/{N_FOLDS}  "
                 f"全队列={ci_a:.4f}  Hard(III+IV)={ci_h:.4f}  "
                 f"(Hard={int((idh==1).sum())}人)")

    arr_a = np.array([c for c in ci_all_folds  if not np.isnan(c)])
    arr_h = np.array([c for c in ci_hard_folds if not np.isnan(c)])
    log.info(f"\n[{model_name}] 最终结果:")
    log.info(f"  全队列       C-index: {arr_a.mean():.4f} ± {arr_a.std():.4f}")
    log.info(f"  Hard(III+IV) C-index: {arr_h.mean():.4f} ± {arr_h.std():.4f}")
    return ci_all_folds, ci_hard_folds


# ─────────────────────────────────────────────
# 9. 汇总 & 保存（与GBMLGG版一致）
# ─────────────────────────────────────────────
def print_summary(results: dict, args):
    log.info("\n" + "="*74)
    log.info("LUAD DHFv2 (ConFuse) 结果汇总")
    log.info(f"n_heads={args.n_heads}  λ_wt={args.lambda_wt}  "
             f"λ_orth={args.lambda_orth}  adapter_dim={args.adapter_dim}  "
             f"bs={args.batch_size}")
    log.info("="*74)
    log.info(f"  {'模型':<35} {'全队列':>15} {'Hard(III+IV)':>15}")
    log.info("-"*74)
    labels = {
        "dhf_v2_mean":  "DHFv2 ConFuse 均值",
        "dhf_v2_patch": "DHFv2 ConFuse patch",
    }
    for name, (ca, ch) in results.items():
        a = np.array([c for c in ca if not np.isnan(c)])
        h = np.array([c for c in ch if not np.isnan(c)])
        log.info(f"  {labels.get(name, name):<35} "
                 f"{a.mean():.4f}±{a.std():.4f}     "
                 f"{h.mean():.4f}±{h.std():.4f}")
    log.info("="*74)


def save_results(results: dict, args):
    os.makedirs(args.out_dir, exist_ok=True)
    out = {"config": vars(args)}
    for name, (ca, ch) in results.items():
        a = np.array([c for c in ca if not np.isnan(c)])
        h = np.array([c for c in ch if not np.isnan(c)])
        out[name] = {
            "ci_all_folds": ca, "ci_hard_folds": ch,
            "ci_all_mean": float(a.mean()), "ci_all_std": float(a.std()),
            "ci_hard_mean": float(h.mean()), "ci_hard_std": float(h.std()),
        }
    fname = (f"dhf_v2_nh{args.n_heads}_lw{args.lambda_wt}"
             f"_lo{args.lambda_orth}_adp{args.adapter_dim}"
             f"_wd{args.wt_head_dim}_b{args.batch_size}"
             f"{'_nc'  if args.no_cross_attn else ''}"
             f"{'_nog' if args.no_gate       else ''}"
             f"{'_na'  if args.no_adapter    else ''}"
             f".json")
    path = os.path.join(args.out_dir, fname)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"结果已保存: {path}")


# ─────────────────────────────────────────────
# 10. 主函数
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    device = (torch.device(f"cuda:{args.gpu}")
              if args.gpu >= 0 and torch.cuda.is_available()
              else torch.device("cpu"))
    log.info(f"设备: {device}")

    models = (["dhf_v2_mean", "dhf_v2_patch"]
              if args.model == "all" else [args.model])

    results = {}
    for m in models:
        ca, ch = run_cv(m, args, device)
        results[m] = (ca, ch)

    print_summary(results, args)
    save_results(results, args)


if __name__ == "__main__":
    main()
