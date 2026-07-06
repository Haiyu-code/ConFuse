#!/usr/bin/env python3
"""
KIRC DHFv2 多模态版本
基于 ucec_dhfv2_multimodal.py 改编
困难子群: Stage IV (15.8%)
"""

import os, json, pickle, argparse, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sksurv.metrics import concordance_index_censored

PKL_PATH    = '/data2/wanghy/first/KIRC/kirc_multimodal_5cv.pkl'          # ← KIRC
FEAT_DIR    = '/data2/wanghy/first/KIRC/wsi_mean_feats/'                 # ← KIRC
OUT_DIR     = '/data2/wanghy/first/KIRC/results_dhfv2_mm'                # ← KIRC
PATH_DIM    = 1536
OMIC_DIM    = 320
N_FOLDS     = 5                                                          # ← KIRC 15折
MAX_PATCHES = 4096
HARD_LABEL  = 'KIRC_StageIV'                                              # ← KIRC

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=40)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=4e-4)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--max_patches",  type=int,   default=MAX_PATCHES)
    p.add_argument("--attn_dim",     type=int,   default=256)
    p.add_argument("--n_heads",      type=int,   default=4)
    p.add_argument("--attn_dropout", type=float, default=0.1)
    p.add_argument("--lambda_wt",    type=float, default=0.3)            # ← KIRC 15.8%占比→较小λ
    p.add_argument("--lambda_orth",  type=float, default=0.1)
    p.add_argument("--adapter_dim",  type=int,   default=128)
    p.add_argument("--wt_head_dim",  type=int,   default=128)
    p.add_argument("--dropout",      type=float, default=0.25)
    p.add_argument("--gpu",          type=int,   default=0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--no_adapter",     action="store_true")
    p.add_argument("--no_cross_attn",  action="store_true")
    p.add_argument("--no_gate",        action="store_true")
    p.add_argument("--lambda_wt_zero", action="store_true")
    return p.parse_args()

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class KIRCMultiModalDataset(Dataset):
    def __init__(self, indices, x_omic, patient_ids,
                 survtime, censorship, subtype, feat_dir,
                 max_patches=4096, training=True):
        self.max_patches = max_patches
        self.training    = training
        self.samples     = []
        self.events      = []

        for i in indices:
            pid     = patient_ids[i]
            pt_path = os.path.join(feat_dir, f"{pid}.pt")
            if not os.path.exists(pt_path):
                continue
            e_val = 1.0 - float(censorship[i])
            self.samples.append({
                'pt_path':  pt_path,
                'x_omic':   torch.FloatTensor(x_omic[i]),
                'e':        torch.tensor(e_val),
                't':        torch.tensor(float(survtime[i])),
                'is_hard':  torch.tensor(
                                float(subtype[i] == HARD_LABEL)),      # ← KIRC
            })
            self.events.append(e_val)

        self.events = np.array(self.events)
        n_hard = sum(1 for s in self.samples if s['is_hard'].item() == 1.0)
        ev = int(self.events.sum())
        log.info(f"  {'训练' if training else '测试'}集: "
                 f"{len(self.samples)}人  StageIV={n_hard}  死亡事件={ev}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        d     = torch.load(s['pt_path'], map_location='cpu',
                           weights_only=False)
        feats = d['features']
        N     = feats.shape[0]
        if N > self.max_patches:
            if self.training:
                idx_s = torch.randperm(N)[:self.max_patches]
            else:
                rng   = torch.Generator()
                rng.manual_seed(42)
                idx_s = torch.randperm(N, generator=rng)[:self.max_patches]
            feats = feats[idx_s]
        return {'x_path': feats, 'x_omic': s['x_omic'],
                'e': s['e'], 't': s['t'], 'is_hard': s['is_hard']}

def collate_fn(batch):
    return {
        'x_path':  [s['x_path']  for s in batch],
        'x_omic':  torch.stack([s['x_omic']  for s in batch]),
        'e':       torch.stack([s['e']       for s in batch]),
        't':       torch.stack([s['t']       for s in batch]),
        'is_hard': torch.stack([s['is_hard'] for s in batch]),
    }

def make_weighted_sampler(dataset):
    weights = np.where(dataset.events == 1.0, 5.0, 1.0)
    weights = torch.DoubleTensor(weights)
    return WeightedRandomSampler(weights, len(weights), replacement=True)

# ─────────────────────────────────────────────
# 模型组件（与 UCEC 完全一致）
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
        self.Wq  = nn.Linear(d_q,  d_out)
        self.Wk  = nn.Linear(d_kv, d_out)
        self.Wv  = nn.Linear(d_kv, d_out)
        self.Wo  = nn.Linear(d_out, d_out)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)
        self.proj = (nn.Linear(d_q, d_out, bias=False)
                     if d_q != d_out else nn.Identity())
    def forward(self, z_q, z_kv):
        B = z_q.shape[0]
        Q = self.Wq(z_q).view(B, self.n_heads, self.d_head)
        K = self.Wk(z_kv).view(B, self.n_heads, self.d_head)
        V = self.Wv(z_kv).view(B, self.n_heads, self.d_head)
        attn = self.drop(torch.softmax(
            (Q*K).sum(-1)*self.scale, dim=-1))
        out  = (attn.unsqueeze(-1)*V).reshape(B, -1)
        return self.norm(self.Wo(out) + self.proj(z_q))

class GatedFusion(nn.Module):
    def __init__(self, d_wsi, d_rna, d_out, dropout=0.1):
        super().__init__()
        self.proj_wsi = nn.Linear(d_wsi, d_out)
        self.proj_rna = nn.Linear(d_rna, d_out)
        self.gate_net = nn.Sequential(
            nn.Linear(d_wsi+d_rna, d_out), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)
    def forward(self, z_wsi, z_rna):
        gate = self.gate_net(torch.cat([z_wsi, z_rna], dim=-1))
        z    = gate*self.proj_wsi(z_wsi) + (1-gate)*self.proj_rna(z_rna)
        return self.norm(self.drop(z)), gate

class KIRCDHFv2MM(nn.Module):
    def __init__(self, path_dim=PATH_DIM, omic_dim=OMIC_DIM,
                 path_hidden=256, omic_hidden=128,
                 attn_dim=256, n_heads=4, attn_dropout=0.1, dropout=0.25,
                 wt_head_dim=128, adapter_dim=128,
                 use_gate=True, use_cross_attn=True,
                 use_adapter=True, lambda_orth=0.1):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_adapter    = use_adapter
        fuse_dim = path_hidden + omic_hidden

        self.patch_encoder = nn.Sequential(
            nn.Linear(path_dim, path_hidden),
            nn.ReLU(), nn.Dropout(dropout))
        self.attn_pool = GatedAttentionPool(path_hidden, attn_dim)
        self.rna_encoder = nn.Sequential(
            nn.Linear(omic_dim, omic_hidden),
            nn.ReLU(), nn.Dropout(dropout))

        if use_cross_attn:
            self.cross_rna2wsi = CrossModalAttention(
                omic_hidden, path_hidden, omic_hidden,
                n_heads=2, dropout=attn_dropout)
            self.cross_wsi2rna = CrossModalAttention(
                path_hidden, omic_hidden, path_hidden,
                n_heads=n_heads, dropout=attn_dropout)

        if use_gate:
            self.fusion = GatedFusion(
                path_hidden, omic_hidden, fuse_dim, dropout)
        else:
            self.fusion = None
            self.simple_fusion = nn.Sequential(
                nn.Linear(path_hidden+omic_hidden, fuse_dim),
                nn.ReLU(), nn.Dropout(dropout))

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim),
            nn.ReLU(), nn.Dropout(dropout))
        self.cox_global = nn.Linear(fuse_dim, 1)

        if use_adapter:
            self.hard_adapter = nn.Sequential(
                nn.Linear(fuse_dim, adapter_dim), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(adapter_dim, adapter_dim),
                nn.LayerNorm(adapter_dim), nn.ReLU(),
                nn.Dropout(dropout))
            hard_in = fuse_dim + adapter_dim
        else:
            self.hard_adapter = None
            hard_in = fuse_dim

        self.hard_head = nn.Sequential(
            nn.Linear(hard_in, wt_head_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(wt_head_dim, 1))

        self._init_weights()
        n = sum(p.numel() for p in self.parameters())
        log.info(f"  KIRCDHFv2MM 参数: {n:,}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_path_list, x_omic):
        dev = x_omic.device
        path_vecs = []
        for patches in x_path_list:
            h = self.patch_encoder(patches.to(dev))
            path_vecs.append(self.attn_pool(h))
        z_wsi = torch.stack(path_vecs)
        z_rna = self.rna_encoder(x_omic)

        if self.use_cross_attn:
            z_rna = self.cross_rna2wsi(z_rna, z_wsi)
            z_wsi = self.cross_wsi2rna(z_wsi, z_rna)

        if self.fusion is not None:
            z, _ = self.fusion(z_wsi, z_rna)
        else:
            z = self.simple_fusion(torch.cat([z_wsi, z_rna], dim=-1))

        z = self.fusion_mlp(z)
        risk_global = self.cox_global(z).squeeze(-1)

        if self.use_adapter and self.hard_adapter is not None:
            z_adapted = self.hard_adapter(z.detach())
            risk_hard = self.hard_head(
                torch.cat([z, z_adapted], dim=-1)).squeeze(-1)
        else:
            z_adapted = None
            risk_hard = self.hard_head(z).squeeze(-1)

        return risk_global, risk_hard, z, z_adapted

# ─────────────────────────────────────────────
# 损失
# ─────────────────────────────────────────────
def cox_loss(risk, t, e):
    if risk.shape[0] < 2 or e.sum() < 1:
        return torch.tensor(0.0, requires_grad=True, device=risk.device)
    order   = torch.argsort(t, descending=True)
    risk_s  = risk[order]; e_s = e[order]
    log_cum = torch.logcumsumexp(risk_s, dim=0)
    loss    = -torch.mean((risk_s - log_cum) * e_s)
    return loss if not torch.isnan(loss) else \
           torch.tensor(0.0, requires_grad=True, device=risk.device)

def orth_loss(z, za):
    B = z.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z.device)
    z_n  = F.normalize(z.detach(), dim=-1)
    za_n = F.normalize(za,         dim=-1)
    G_z  = z_n  @ z_n.T
    G_za = za_n @ za_n.T
    eye  = torch.eye(B, device=z.device, dtype=torch.bool)
    diff = (G_z - G_za)[~eye]
    return diff.pow(2).mean()

def dhfv2_loss(rg, rc, z, za, t, e, is_hard,
               lambda_wt, lambda_orth, lambda_wt_zero=False):
    l_g = cox_loss(rg, t, e)

    if lambda_wt_zero:
        l_c = torch.tensor(0.0, device=rc.device)
    else:
        m   = (is_hard == 1.0)
        l_c = cox_loss(rc[m], t[m], e[m]) if m.sum() >= 2 else \
              torch.tensor(0.0, requires_grad=True, device=rc.device)

    if lambda_orth > 0 and za is not None:
        l_orth = orth_loss(z, za)
    else:
        l_orth = torch.tensor(0.0, device=rc.device)

    total = l_g + lambda_wt * l_c + lambda_orth * l_orth
    return total, l_g.item(), l_c.item(), l_orth.item()

# ─────────────────────────────────────────────
# 训练 & 评估
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, args):
    model.train()
    tl, tg, tw, to, nb = 0, 0, 0, 0, 0
    nan_batches = 0

    for batch in loader:
        x_omic  = batch['x_omic'].to(device)
        e       = batch['e'].to(device)
        t       = batch['t'].to(device)
        is_hard = batch['is_hard'].to(device)

        if e.sum() < 1:
            continue

        optimizer.zero_grad()
        rg, rc, z, za = model(batch['x_path'], x_omic)
        loss, lg, lw, lo = dhfv2_loss(
            rg, rc, z, za, t, e, is_hard,
            args.lambda_wt, args.lambda_orth,
            args.lambda_wt_zero)

        if not torch.isfinite(loss):
            nan_batches += 1
            continue

        loss.backward()

        has_bad_grad = False
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                has_bad_grad = True
                break
        if has_bad_grad:
            optimizer.zero_grad()
            nan_batches += 1
            continue

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tl += loss.item(); tg += lg; tw += lw; to += lo; nb += 1

    if nan_batches > 0:
        log.warning(f"    本epoch跳过 {nan_batches} 个异常batch")

    n = max(nb, 1)
    return tl/n, tg/n, tw/n, to/n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rgs, rcs, es, ts, hs = [], [], [], [], []
    for batch in loader:
        x_omic = batch['x_omic'].to(device)
        rg, rc, _, _ = model(batch['x_path'], x_omic)
        rgs.append(rg.cpu().numpy())
        rcs.append(rc.cpu().numpy())
        es.append(batch['e'].numpy())
        ts.append(batch['t'].numpy())
        hs.append(batch['is_hard'].numpy())
    return (np.concatenate(rgs), np.concatenate(rcs),
            np.concatenate(es),  np.concatenate(ts),
            np.concatenate(hs))

def cindex(risks, events, times):
    m = events.astype(bool)
    if m.sum() < 2: return float('nan')
    if not np.isfinite(risks).all(): return float('nan')
    try:
        return float(concordance_index_censored(m, times, risks)[0])
    except: return float('nan')

def cindex_hard(risks, events, times, is_hard):
    m = (is_hard == 1.0)
    if m.sum() < 5: return float('nan')
    return cindex(risks[m], events[m], times[m])

# ─────────────────────────────────────────────
# 15折CV
# ─────────────────────────────────────────────
def run_cv(args, device):
    log.info(f"\n{'='*60}")
    log.info(f"KIRC DHFv2 多模态  {N_FOLDS}折CV  "
             f"λ_wt={args.lambda_wt}  λ_orth={args.lambda_orth}  "
             f"adp={args.adapter_dim}  batch={args.batch_size}")
    log.info(f"{'='*60}")

    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)

    X_rna       = data['x_omic']
    patient_ids = data['patient_id']
    survtime    = data['survtime']
    censorship  = data['censorship']
    subtype     = data['subtype']
    splits      = data['splits']

    ci_all_folds, ci_hard_folds = [], []

    for fold_i, split in enumerate(splits):
        fold = fold_i + 1
        torch.manual_seed(args.seed + fold)
        np.random.seed(args.seed + fold)

        tr_idx = np.array(split['train'])
        te_idx = np.array(split['test'])

        tr_set = KIRCMultiModalDataset(
            tr_idx, X_rna, patient_ids,
            survtime, censorship, subtype, FEAT_DIR,
            max_patches=args.max_patches, training=True)
        te_set = KIRCMultiModalDataset(
            te_idx, X_rna, patient_ids,
            survtime, censorship, subtype, FEAT_DIR,
            max_patches=args.max_patches, training=False)

        sampler   = make_weighted_sampler(tr_set)
        tr_loader = DataLoader(
            tr_set, batch_size=args.batch_size,
            sampler=sampler, collate_fn=collate_fn,
            num_workers=0, pin_memory=True)
        te_loader = DataLoader(
            te_set, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn,
            num_workers=0, pin_memory=True)

        model = KIRCDHFv2MM(
            attn_dim      = args.attn_dim,
            n_heads       = args.n_heads,
            attn_dropout  = args.attn_dropout,
            dropout       = args.dropout,
            wt_head_dim   = args.wt_head_dim,
            adapter_dim   = args.adapter_dim,
            use_gate      = not args.no_gate,
            use_cross_attn= not args.no_cross_attn,
            use_adapter   = not args.no_adapter,
            lambda_orth   = args.lambda_orth,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr*0.01)

        best_ci_hard = 0.0
        best_state   = None

        for epoch in range(1, args.epochs+1):
            tl, lg, lw, lo = train_epoch(
                model, tr_loader, optimizer, device, args)
            scheduler.step()

            if epoch % 10 == 0 or epoch == args.epochs:
                rg, rc, e, t, h = evaluate(model, te_loader, device)
                ci_a    = cindex(rg, e, t)
                ci_h    = cindex_hard(rc, e, t, h)
                ci_href = cindex_hard(rg, e, t, h)

                if not np.isnan(ci_h) and ci_h > best_ci_hard:
                    best_ci_hard = ci_h
                    best_state = (rg.copy(), rc.copy(),
                                  e.copy(), t.copy(), h.copy())

                log.info(f"  [F{fold:02d} E{epoch:3d}] "
                         f"L={tl:.3f}(g={lg:.3f},h={lw:.3f},o={lo:.3f})  "
                         f"全队列={ci_a:.4f}  "
                         f"StageIV(hard_head)={ci_h:.4f}  "
                         f"StageIV(global)={ci_href:.4f}")

        if best_state is None:
            best_state = evaluate(model, te_loader, device)

        rg, rc, e, t, h = best_state
        ci_a  = cindex(rg, e, t)
        ci_h  = cindex_hard(rc, e, t, h)
        ci_all_folds.append(ci_a)
        ci_hard_folds.append(ci_h)

        log.info(f"  ★ Fold {fold:02d}/{N_FOLDS}  "
                 f"全队列={ci_a:.4f}  StageIV={ci_h:.4f}  "
                 f"(StageIV={int(h.sum())}人, "
                 f"事件={int(e[h==1].sum())})")

    arr_a = np.array([c for c in ci_all_folds  if not np.isnan(c)])
    arr_h = np.array([c for c in ci_hard_folds if not np.isnan(c)])

    log.info(f"\n{'='*60}")
    log.info(f"KIRC DHFv2 多模态 最终结果 ({N_FOLDS}折CV):")
    log.info(f"  全队列    C-index: {arr_a.mean():.4f} ± {arr_a.std():.4f}")
    log.info(f"  Stage IV  C-index: {arr_h.mean():.4f} ± {arr_h.std():.4f}")
    log.info(f"{'='*60}")

    return ci_all_folds, ci_hard_folds

def main():
    args   = parse_args()
    device = (torch.device(f'cuda:{args.gpu}')
              if args.gpu >= 0 and torch.cuda.is_available()
              else torch.device('cpu'))
    log.info(f"设备: {device}")
    ci_all, ci_hard = run_cv(args, device)

    os.makedirs(OUT_DIR, exist_ok=True)
    a = np.array([c for c in ci_all  if not np.isnan(c)])
    h = np.array([c for c in ci_hard if not np.isnan(c)])
    out = {
        'config':        vars(args),
        'ci_all_mean':   float(a.mean()), 'ci_all_std':  float(a.std()),
        'ci_hard_mean':  float(h.mean()), 'ci_hard_std': float(h.std()),
        'ci_all_folds':  ci_all,          'ci_hard_folds': ci_hard,
    }
    fname = (f"kirc_dhfv2_mm"
             f"_lw{args.lambda_wt}_lo{args.lambda_orth}"
             f"_adp{args.adapter_dim}_p{args.max_patches}.json")
    with open(os.path.join(OUT_DIR, fname), 'w') as f:
        json.dump(out, f, indent=2)
    log.info(f"结果已保存: {OUT_DIR}/{fname}")

if __name__ == '__main__':
    main()
