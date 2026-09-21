#!/usr/bin/env python3
"""Model, data, training, evaluation, and output code for two fixed pipelines.

Pipelines
---------
1. ``gbmlggmodel`` for the GBMLGG cohort.
2. ``ucecmodel`` for the UCEC cohort.

Use ``run.py`` as the command-line entry point.
"""

from __future__ import annotations

import json
import logging
import pickle
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dhfv2-release")


@dataclass
class GBMLGGConfig:
    data_pkl: Path
    out_dir: Path
    epochs: int = 40
    lr: float = 2e-4
    weight_decay: float = 4e-4
    batch_size: int = 32
    num_workers: int = 4
    path_dim: int = 1536
    omic_dim: int = 320
    path_hidden: int = 256
    omic_hidden: int = 128
    attn_dim: int = 256
    n_heads: int = 4
    attn_dropout: float = 0.1
    dropout: float = 0.25
    lambda_wt: float = 2.0
    lambda_orth: float = 0.1
    wt_head_dim: int = 128
    adapter_dim: int = 64
    no_gate: bool = False
    no_cross_attn: bool = False
    no_adapter: bool = False
    lambda_wt_zero: bool = False
    wt_head_global: bool = False
    gpu: int = 0
    seed: int = 42


@dataclass
class UCECConfig:
    data_pkl: Path
    feat_dir: Path
    out_dir: Path
    epochs: int = 40
    lr: float = 2e-4
    weight_decay: float = 4e-4
    batch_size: int = 16
    num_workers: int = 4
    max_patches: int = 4096
    event_weight: float = 5.0
    path_dim: int = 1536
    omic_dim: int = 320
    path_hidden: int = 256
    omic_hidden: int = 128
    attn_dim: int = 256
    dropout: float = 0.25
    gpu: int = 0
    seed: int = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(gpu: int) -> torch.device:
    if gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def cox_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood with Breslow-style risk sets."""
    if risk.numel() < 2 or event.sum() < 1:
        return risk.sum() * 0.0
    order = torch.argsort(time, descending=True)
    ordered_risk = risk[order]
    event_mask = event[order] > 0.5
    log_risk_set = torch.logcumsumexp(ordered_risk, dim=0)
    return -(ordered_risk[event_mask] - log_risk_set[event_mask]).mean()


def cindex(risk: np.ndarray, event: np.ndarray, time: np.ndarray) -> float:
    risk = np.asarray(risk)
    event = np.asarray(event).astype(bool)
    time = np.asarray(time)
    valid = np.isfinite(risk) & np.isfinite(time)
    risk, event, time = risk[valid], event[valid], time[valid]
    if len(risk) < 2 or event.sum() < 1:
        return float("nan")
    try:
        return float(concordance_index_censored(event, time, risk)[0])
    except Exception:
        return float("nan")


def metric_summary(values: list[float]) -> tuple[float, float]:
    array = np.asarray([value for value in values if np.isfinite(value)])
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std())


def json_config(config: GBMLGGConfig | UCECConfig) -> dict[str, Any]:
    output = asdict(config)
    for key, value in output.items():
        if isinstance(value, Path):
            output[key] = str(value)
    return output


def gradients_are_finite(model: nn.Module) -> bool:
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


class CrossModalAttention(nn.Module):
    """Cross-modal attention used by the original DHFv2 mean model."""

    def __init__(
        self,
        query_dim: int,
        key_value_dim: int,
        output_dim: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if output_dim % n_heads != 0:
            raise ValueError("output_dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = output_dim // n_heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(query_dim, output_dim)
        self.key = nn.Linear(key_value_dim, output_dim)
        self.value = nn.Linear(key_value_dim, output_dim)
        self.output = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        self.residual = (
            nn.Linear(query_dim, output_dim, bias=False)
            if query_dim != output_dim
            else nn.Identity()
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        batch_size = query.shape[0]
        q = self.query(query).view(batch_size, self.n_heads, self.head_dim)
        k = self.key(key_value).view(batch_size, self.n_heads, self.head_dim)
        v = self.value(key_value).view(batch_size, self.n_heads, self.head_dim)
        attention = torch.softmax((q * k).sum(dim=-1) * self.scale, dim=-1)
        attended = (self.dropout(attention).unsqueeze(-1) * v).reshape(batch_size, -1)
        return self.norm(self.output(attended) + self.residual(query))


class GatedFusion(nn.Module):
    def __init__(self, path_dim: int, omic_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.path_projection = nn.Linear(path_dim, output_dim)
        self.omic_projection = nn.Linear(omic_dim, output_dim)
        self.gate = nn.Sequential(
            nn.Linear(path_dim + omic_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self, path_vector: torch.Tensor, omic_vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate(torch.cat([path_vector, omic_vector], dim=-1))
        fused = gate * self.path_projection(path_vector) + (1.0 - gate) * self.omic_projection(
            omic_vector
        )
        return self.norm(self.dropout(fused)), gate


class ConcatFusion(nn.Module):
    def __init__(self, path_dim: int, omic_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(path_dim + omic_dim, output_dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(
        self, path_vector: torch.Tensor, omic_vector: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        return self.projection(torch.cat([path_vector, omic_vector], dim=-1)), None


# ---------------------------------------------------------------------------
# Pipeline 1: gbmlggmodel
# ---------------------------------------------------------------------------


class GBMLGGMeanDataset(Dataset):
    def __init__(self, split: dict[str, Any], data_pd: pd.DataFrame) -> None:
        patient_names = split["x_patname"]
        path_features = np.asarray(split["x_path"])
        omic_features = np.asarray(split["x_omic"])
        events = np.asarray(split["e"])
        times = np.asarray(split["t"])
        idh_map = dict(zip(data_pd.index.astype(str).str[:12], data_pd["idh mutation"]))

        patient_rows: dict[str, list[int]] = defaultdict(list)
        for row, raw_patient_id in enumerate(patient_names):
            patient_rows[str(raw_patient_id).strip()[:12]].append(row)

        self.samples: list[dict[str, Any]] = []
        for patient_id, rows in patient_rows.items():
            rows = sorted(rows)
            idh = float(idh_map.get(patient_id, -1.0))
            self.samples.append(
                {
                    "pid": patient_id,
                    "x_path": torch.as_tensor(path_features[rows].mean(axis=0), dtype=torch.float32),
                    "x_omic": torch.as_tensor(omic_features[rows].mean(axis=0), dtype=torch.float32),
                    "event": float(events[rows[0]]),
                    "time": float(times[rows[0]]),
                    "idh_wt": float(idh == 0.0),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "pid": sample["pid"],
            "x_path": sample["x_path"].clone(),
            "x_omic": sample["x_omic"].clone(),
            "event": torch.tensor(sample["event"], dtype=torch.float32),
            "time": torch.tensor(sample["time"], dtype=torch.float32),
            "idh_wt": torch.tensor(sample["idh_wt"], dtype=torch.float32),
        }


class GBMLGGModel(nn.Module):
    """Shared multimodal backbone with an IDH-WT-specific detached adapter."""

    def __init__(self, config: GBMLGGConfig) -> None:
        super().__init__()
        self.use_cross_attention = not config.no_cross_attn
        self.use_adapter = not config.no_adapter
        fusion_dim = config.path_hidden + config.omic_hidden

        self.wsi_encoder = nn.Sequential(
            nn.Linear(config.path_dim, config.path_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.rna_encoder = nn.Sequential(
            nn.Linear(config.omic_dim, config.omic_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        if self.use_cross_attention:
            rna_heads = 2 if config.omic_hidden % 2 == 0 else 1
            self.rna_to_wsi = CrossModalAttention(
                config.omic_hidden,
                config.path_hidden,
                config.omic_hidden,
                rna_heads,
                config.attn_dropout,
            )
            self.wsi_to_rna = CrossModalAttention(
                config.path_hidden,
                config.omic_hidden,
                config.path_hidden,
                config.n_heads,
                config.attn_dropout,
            )
        if config.no_gate:
            self.fusion: GatedFusion | ConcatFusion = ConcatFusion(
                config.path_hidden, config.omic_hidden, fusion_dim, config.dropout
            )
        else:
            self.fusion = GatedFusion(
                config.path_hidden, config.omic_hidden, fusion_dim, config.dropout
            )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim), nn.ReLU(), nn.Dropout(config.dropout)
        )
        self.global_head = nn.Linear(fusion_dim, 1)

        if self.use_adapter:
            self.wt_adapter = nn.Sequential(
                nn.Linear(fusion_dim, config.adapter_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.adapter_dim, config.adapter_dim),
                nn.LayerNorm(config.adapter_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            wt_input_dim = fusion_dim + config.adapter_dim
        else:
            self.wt_adapter = None
            wt_input_dim = fusion_dim
        self.wt_head = nn.Sequential(
            nn.Linear(wt_input_dim, config.wt_head_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.wt_head_dim, 1),
        )

    def forward(self, x_path: torch.Tensor, x_omic: torch.Tensor) -> dict[str, Any]:
        path_vector = self.wsi_encoder(x_path)
        omic_vector = self.rna_encoder(x_omic)
        if self.use_cross_attention:
            omic_vector = self.rna_to_wsi(omic_vector, path_vector)
            path_vector = self.wsi_to_rna(path_vector, omic_vector)
        representation, gate = self.fusion(path_vector, omic_vector)
        representation = self.fusion_mlp(representation)
        global_risk = self.global_head(representation).squeeze(-1)

        if self.wt_adapter is not None:
            adapted = self.wt_adapter(representation.detach())
            wt_input = torch.cat([representation, adapted], dim=-1)
        else:
            adapted = None
            wt_input = representation
        wt_risk = self.wt_head(wt_input).squeeze(-1)
        return {
            "global_risk": global_risk,
            "wt_risk": wt_risk,
            "representation": representation,
            "adapted": adapted,
            "gate": gate,
        }


def dhfv2_loss(
    output: dict[str, Any],
    time: torch.Tensor,
    event: torch.Tensor,
    idh_wt: torch.Tensor,
    config: GBMLGGConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    global_loss = cox_loss(output["global_risk"], time, event)
    if config.lambda_wt_zero:
        wt_loss = output["wt_risk"].sum() * 0.0
    elif config.wt_head_global:
        wt_loss = cox_loss(output["wt_risk"], time, event)
    else:
        wt_mask = idh_wt > 0.5
        wt_loss = (
            cox_loss(output["wt_risk"][wt_mask], time[wt_mask], event[wt_mask])
            if wt_mask.sum() >= 2
            else output["wt_risk"].sum() * 0.0
        )

    if config.lambda_orth > 0 and output["adapted"] is not None:
        base = F.normalize(output["representation"].detach(), dim=-1)
        adapted = F.normalize(output["adapted"], dim=-1)
        cross_covariance = base.T @ adapted / max(base.shape[0], 1)
        orthogonal_loss = cross_covariance.square().sum()
    else:
        orthogonal_loss = global_loss.new_zeros(())

    total = global_loss + config.lambda_wt * wt_loss + config.lambda_orth * orthogonal_loss
    return total, {
        "global": float(global_loss.detach()),
        "wt": float(wt_loss.detach()),
        "orth": float(orthogonal_loss.detach()),
    }


def train_dhfv2_epoch(
    model: GBMLGGModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: GBMLGGConfig,
) -> dict[str, float]:
    model.train()
    totals = defaultdict(float)
    valid_batches = 0
    skipped_batches = 0
    for batch in loader:
        x_path = batch["x_path"].to(device, non_blocking=True)
        x_omic = batch["x_omic"].to(device, non_blocking=True)
        event = batch["event"].to(device, non_blocking=True)
        time = batch["time"].to(device, non_blocking=True)
        idh_wt = batch["idh_wt"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(x_path, x_omic)
        loss, parts = dhfv2_loss(output, time, event, idh_wt, config)
        if not torch.isfinite(loss):
            skipped_batches += 1
            continue
        loss.backward()
        if not gradients_are_finite(model):
            optimizer.zero_grad(set_to_none=True)
            skipped_batches += 1
            continue
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals["loss"] += float(loss.detach())
        for key, value in parts.items():
            totals[key] += value
        valid_batches += 1
    denominator = max(valid_batches, 1)
    return {
        **{key: value / denominator for key, value in totals.items()},
        "skipped": float(skipped_batches),
    }


@torch.no_grad()
def evaluate_dhfv2(
    model: GBMLGGModel, loader: DataLoader, device: torch.device
) -> dict[str, Any]:
    model.eval()
    values: dict[str, list[Any]] = defaultdict(list)
    for batch in loader:
        output = model(
            batch["x_path"].to(device, non_blocking=True),
            batch["x_omic"].to(device, non_blocking=True),
        )
        values["global_risk"].append(output["global_risk"].cpu().numpy())
        values["wt_risk"].append(output["wt_risk"].cpu().numpy())
        values["event"].append(batch["event"].numpy())
        values["time"].append(batch["time"].numpy())
        values["idh_wt"].append(batch["idh_wt"].numpy())
        values["pid"].extend(batch["pid"])
    return {
        key: value if key == "pid" else np.concatenate(value)
        for key, value in values.items()
    }


def run_gbmlggmodel(config: GBMLGGConfig) -> dict[str, Any]:
    """Run cross-validation for gbmlggmodel."""
    seed_everything(config.seed)
    device = resolve_device(config.gpu)
    with config.data_pkl.open("rb") as handle:
        data = pickle.load(handle)
    data_pd = data["data_pd"]
    cv_splits = data["cv_splits"]
    if isinstance(cv_splits, dict):
        splits = [cv_splits[key] for key in sorted(cv_splits, key=lambda value: int(value))]
    else:
        splits = list(cv_splits)

    fold_results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    log.info("model=gbmlggmodel; folds=%d; device=%s", len(splits), device)
    log.info("Test folds are evaluated only after the fixed final epoch.")

    for fold, split in enumerate(splits, start=1):
        fold_seed = config.seed + fold
        seed_everything(fold_seed)
        train_set = GBMLGGMeanDataset(split["train"], data_pd)
        test_set = GBMLGGMeanDataset(split["test"], data_pd)
        generator = torch.Generator().manual_seed(fold_seed)
        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=generator,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
        )
        model = GBMLGGModel(config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
        )
        for epoch in range(1, config.epochs + 1):
            stats = train_dhfv2_epoch(model, train_loader, optimizer, device, config)
            scheduler.step()
            if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
                log.info(
                    "[GBMLGG F%02d E%03d] loss=%.4f global=%.4f wt=%.4f orth=%.4f skipped=%d",
                    fold,
                    epoch,
                    stats.get("loss", float("nan")),
                    stats.get("global", float("nan")),
                    stats.get("wt", 0.0),
                    stats.get("orth", 0.0),
                    int(stats["skipped"]),
                )

        prediction = evaluate_dhfv2(model, test_loader, device)
        wt_mask = prediction["idh_wt"] == 1.0
        full_ci = cindex(prediction["global_risk"], prediction["event"], prediction["time"])
        wt_ci = cindex(
            prediction["wt_risk"][wt_mask],
            prediction["event"][wt_mask],
            prediction["time"][wt_mask],
        )
        fold_results.append(
            {
                "fold": fold,
                "full_cindex": full_ci,
                "idh_wt_cindex": wt_ci,
                "test_n": len(prediction["pid"]),
                "idh_wt_n": int(wt_mask.sum()),
                "test_events": int(prediction["event"].sum()),
                "idh_wt_events": int(prediction["event"][wt_mask].sum()),
            }
        )
        for index, patient_id in enumerate(prediction["pid"]):
            prediction_rows.append(
                {
                    "dataset": "GBMLGG",
                    "model": "gbmlggmodel",
                    "fold": fold,
                    "patient_id": str(patient_id),
                    "global_risk": float(prediction["global_risk"][index]),
                    "idh_wt_risk": float(prediction["wt_risk"][index]),
                    "time": float(prediction["time"][index]),
                    "event": int(prediction["event"][index]),
                    "idh_wt": int(prediction["idh_wt"][index]),
                    "checkpoint_epoch": config.epochs,
                }
            )
        log.info("GBMLGG fold %02d: full=%.4f IDH-WT=%.4f", fold, full_ci, wt_ci)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_mean, full_std = metric_summary([row["full_cindex"] for row in fold_results])
    wt_mean, wt_std = metric_summary([row["idh_wt_cindex"] for row in fold_results])
    summary = {
        "dataset": "gbmlgg",
        "model": "gbmlggmodel",
        "evaluation_rule": "fixed final epoch; global head for full cohort; WT head for IDH-WT",
        "config": json_config(config),
        "full_mean": full_mean,
        "full_std": full_std,
        "idh_wt_mean": wt_mean,
        "idh_wt_std": wt_std,
        "fold_results": fold_results,
    }
    output_dir = config.out_dir / "gbmlggmodel"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"gbmlggmodel_seed{config.seed}_results.json"
    prediction_path = output_dir / f"gbmlggmodel_seed{config.seed}_oof_predictions.csv"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)
    log.info("Saved %s", result_path)
    log.info("Saved %s", prediction_path)
    return summary


# ---------------------------------------------------------------------------
# Pipeline 2: ucecmodel
# ---------------------------------------------------------------------------


def load_patch_features(path: Path) -> torch.Tensor:
    """Load a trusted .pt file containing a tensor or {'features': tensor}."""
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        if "features" not in value:
            raise KeyError(f"{path} does not contain a 'features' key")
        value = value["features"]
    features = torch.as_tensor(value, dtype=torch.float32)
    if features.ndim == 1:
        features = features.unsqueeze(0)
    if features.ndim != 2:
        raise ValueError(f"expected [patches, features] in {path}, got {tuple(features.shape)}")
    return features


class UCECPatchDataset(Dataset):
    def __init__(
        self,
        indices: Any,
        data: dict[str, Any],
        feat_dir: Path,
        max_patches: int,
        training: bool,
        seed: int,
    ) -> None:
        self.max_patches = max_patches
        self.training = training
        self.seed = seed
        self.samples: list[dict[str, Any]] = []
        missing: list[str] = []
        for raw_index in indices:
            index = int(raw_index)
            patient_id = str(data["patient_id"][index])
            feature_path = feat_dir / f"{patient_id}.pt"
            if not feature_path.is_file():
                missing.append(patient_id)
                continue
            self.samples.append(
                {
                    "pid": patient_id,
                    "feature_path": feature_path,
                    "x_omic": torch.as_tensor(data["x_omic"][index], dtype=torch.float32),
                    "event": 1.0 - float(data["censorship"][index]),
                    "time": float(data["survtime"][index]),
                    "cn_high": float(data["subtype"][index] == "UCEC_CN_HIGH"),
                }
            )
        if missing:
            log.warning("Missing %d UCEC feature files; examples=%s", len(missing), missing[:3])
        self.events = np.asarray([sample["event"] for sample in self.samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        features = load_patch_features(sample["feature_path"])
        if features.shape[0] > self.max_patches:
            generator = torch.Generator()
            if self.training:
                generator.manual_seed(torch.initial_seed() + index)
            else:
                generator.manual_seed(self.seed + index)
            selected = torch.randperm(features.shape[0], generator=generator)[: self.max_patches]
            features = features[selected]
        return {
            "pid": sample["pid"],
            "x_path": features,
            "x_omic": sample["x_omic"].clone(),
            "event": torch.tensor(sample["event"], dtype=torch.float32),
            "time": torch.tensor(sample["time"], dtype=torch.float32),
            "cn_high": torch.tensor(sample["cn_high"], dtype=torch.float32),
        }


def ucec_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pid": [item["pid"] for item in batch],
        "x_path": [item["x_path"] for item in batch],
        "x_omic": torch.stack([item["x_omic"] for item in batch]),
        "event": torch.stack([item["event"] for item in batch]),
        "time": torch.stack([item["time"] for item in batch]),
        "cn_high": torch.stack([item["cn_high"] for item in batch]),
    }


class GatedAttentionPool(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.v = nn.Linear(input_dim, hidden_dim)
        self.u = nn.Linear(input_dim, hidden_dim)
        self.weight = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attention = self.weight(torch.tanh(self.v(tokens)) * torch.sigmoid(self.u(tokens)))
        attention = torch.softmax(attention, dim=0)
        return torch.sum(attention * tokens, dim=0)


class UCECModel(nn.Module):
    """Independent RNA/WSI risk models combined by a learned scalar alpha."""

    def __init__(self, config: UCECConfig) -> None:
        super().__init__()
        self.patch_encoder = nn.Sequential(
            nn.Linear(config.path_dim, config.path_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.attention_pool = GatedAttentionPool(config.path_hidden, config.attn_dim)
        self.rna_encoder = nn.Sequential(
            nn.Linear(config.omic_dim, config.omic_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.rna_risk_head = nn.Linear(config.omic_hidden, 1)
        self.wsi_risk_head = nn.Linear(config.path_hidden, 1)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        self.apply(self._initialize_linear)

    @staticmethod
    def _initialize_linear(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, patch_bags: list[torch.Tensor], x_omic: torch.Tensor) -> torch.Tensor:
        path_vectors = []
        for patches in patch_bags:
            tokens = self.patch_encoder(patches.to(x_omic.device))
            path_vectors.append(self.attention_pool(tokens))
        path_vector = torch.stack(path_vectors)
        omic_vector = self.rna_encoder(x_omic)
        rna_risk = self.rna_risk_head(omic_vector).squeeze(-1)
        wsi_risk = self.wsi_risk_head(path_vector).squeeze(-1)
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * rna_risk + (1.0 - alpha) * wsi_risk


def train_ucec_epoch(
    model: UCECModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    valid_batches = 0
    skipped_batches = 0
    for batch in loader:
        x_omic = batch["x_omic"].to(device, non_blocking=True)
        event = batch["event"].to(device, non_blocking=True)
        time = batch["time"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        risk = model(batch["x_path"], x_omic)
        loss = cox_loss(risk, time, event)
        if not torch.isfinite(loss):
            skipped_batches += 1
            continue
        loss.backward()
        if not gradients_are_finite(model):
            optimizer.zero_grad(set_to_none=True)
            skipped_batches += 1
            continue
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach())
        valid_batches += 1
    return {
        "loss": total_loss / max(valid_batches, 1),
        "skipped": float(skipped_batches),
    }


@torch.no_grad()
def evaluate_ucec(
    model: UCECModel, loader: DataLoader, device: torch.device
) -> dict[str, Any]:
    model.eval()
    values: dict[str, list[Any]] = defaultdict(list)
    for batch in loader:
        risk = model(batch["x_path"], batch["x_omic"].to(device, non_blocking=True))
        values["risk"].append(risk.cpu().numpy())
        values["event"].append(batch["event"].numpy())
        values["time"].append(batch["time"].numpy())
        values["cn_high"].append(batch["cn_high"].numpy())
        values["pid"].extend(batch["pid"])
    return {
        key: value if key == "pid" else np.concatenate(value)
        for key, value in values.items()
    }


def run_ucecmodel(config: UCECConfig) -> dict[str, Any]:
    """Run cross-validation for ucecmodel."""
    seed_everything(config.seed)
    device = resolve_device(config.gpu)
    with config.data_pkl.open("rb") as handle:
        data = pickle.load(handle)
    splits = data["splits"]
    fold_results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    log.info("model=ucecmodel; folds=%d; device=%s", len(splits), device)
    log.info("Test folds are evaluated only after the fixed final epoch.")

    for fold, split in enumerate(splits, start=1):
        fold_seed = config.seed + fold
        seed_everything(fold_seed)
        train_set = UCECPatchDataset(
            split["train"], data, config.feat_dir, config.max_patches, True, fold_seed
        )
        test_set = UCECPatchDataset(
            split["test"], data, config.feat_dir, config.max_patches, False, fold_seed
        )
        if len(train_set) == 0 or len(test_set) == 0:
            raise RuntimeError(f"UCEC fold {fold} is empty after matching feature files")
        generator = torch.Generator().manual_seed(fold_seed)
        weights = np.where(train_set.events == 1.0, config.event_weight, 1.0)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=ucec_collate,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=generator,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=ucec_collate,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
        )
        model = UCECModel(config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
        )
        for epoch in range(1, config.epochs + 1):
            stats = train_ucec_epoch(model, train_loader, optimizer, device)
            scheduler.step()
            if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
                log.info(
                    "[UCEC F%02d E%03d] loss=%.4f alpha=%.4f skipped=%d",
                    fold,
                    epoch,
                    stats["loss"],
                    float(torch.sigmoid(model.alpha_logit).detach().cpu()),
                    int(stats["skipped"]),
                )

        prediction = evaluate_ucec(model, test_loader, device)
        cn_mask = prediction["cn_high"] == 1.0
        full_ci = cindex(prediction["risk"], prediction["event"], prediction["time"])
        cn_ci = cindex(
            prediction["risk"][cn_mask],
            prediction["event"][cn_mask],
            prediction["time"][cn_mask],
        )
        alpha = float(torch.sigmoid(model.alpha_logit).detach().cpu())
        fold_results.append(
            {
                "fold": fold,
                "full_cindex": full_ci,
                "cn_high_cindex": cn_ci,
                "fusion_alpha": alpha,
                "test_n": len(prediction["pid"]),
                "cn_high_n": int(cn_mask.sum()),
                "test_events": int(prediction["event"].sum()),
                "cn_high_events": int(prediction["event"][cn_mask].sum()),
            }
        )
        for index, patient_id in enumerate(prediction["pid"]):
            prediction_rows.append(
                {
                    "dataset": "UCEC",
                    "model": "ucecmodel",
                    "fold": fold,
                    "patient_id": str(patient_id),
                    "risk": float(prediction["risk"][index]),
                    "time": float(prediction["time"][index]),
                    "event": int(prediction["event"][index]),
                    "cn_high": int(prediction["cn_high"][index]),
                    "fusion_alpha": alpha,
                    "checkpoint_epoch": config.epochs,
                }
            )
        log.info("UCEC fold %02d: full=%.4f CN-HIGH=%.4f alpha=%.4f", fold, full_ci, cn_ci, alpha)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_mean, full_std = metric_summary([row["full_cindex"] for row in fold_results])
    cn_mean, cn_std = metric_summary([row["cn_high_cindex"] for row in fold_results])
    summary = {
        "dataset": "ucec",
        "model": "ucecmodel",
        "evaluation_rule": "fixed final epoch; same late-fusion risk for full cohort and CN-HIGH",
        "config": json_config(config),
        "full_mean": full_mean,
        "full_std": full_std,
        "cn_high_mean": cn_mean,
        "cn_high_std": cn_std,
        "fold_results": fold_results,
    }
    output_dir = config.out_dir / "ucecmodel"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"ucecmodel_seed{config.seed}_results.json"
    prediction_path = output_dir / f"ucecmodel_seed{config.seed}_oof_predictions.csv"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)
    log.info("Saved %s", result_path)
    log.info("Saved %s", prediction_path)
    return summary
