#!/usr/bin/env python3
"""Unified command-line launcher for gbmlggmodel and ucecmodel."""

from __future__ import annotations

import argparse
from pathlib import Path


MODELS = ("gbmlggmodel", "ucecmodel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one public survival model.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--data_pkl", type=Path, required=True)
    parser.add_argument(
        "--feat_dir",
        type=Path,
        help="WSI feature directory; required by ucecmodel only.",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("results"))

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=4e-4)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Defaults to 32 for gbmlggmodel and 16 for ucecmodel.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA index; use -1 for CPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--path_dim", type=int, default=1536)
    parser.add_argument("--omic_dim", type=int, default=320)
    parser.add_argument("--path_hidden", type=int, default=256)
    parser.add_argument("--omic_hidden", type=int, default=128)
    parser.add_argument("--attn_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)

    # gbmlggmodel options
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--lambda_wt", type=float, default=2.0)
    parser.add_argument("--lambda_orth", type=float, default=0.1)
    parser.add_argument("--wt_head_dim", type=int, default=128)
    parser.add_argument("--adapter_dim", type=int, default=64)
    parser.add_argument("--no_gate", action="store_true")
    parser.add_argument("--no_cross_attn", action="store_true")
    parser.add_argument("--no_adapter", action="store_true")
    parser.add_argument("--lambda_wt_zero", action="store_true")
    parser.add_argument("--wt_head_global", action="store_true")

    # ucecmodel options
    parser.add_argument("--max_patches", type=int, default=4096)
    parser.add_argument("--event_weight", type=float, default=5.0)
    args = parser.parse_args()

    if not args.data_pkl.is_file():
        parser.error(f"--data_pkl does not exist: {args.data_pkl}")
    if args.model == "ucecmodel" and (args.feat_dir is None or not args.feat_dir.is_dir()):
        parser.error("--feat_dir must point to a directory when --model ucecmodel")
    if args.batch_size is None:
        args.batch_size = 32 if args.model == "gbmlggmodel" else 16
    if min(args.epochs, args.batch_size, args.max_patches) < 1:
        parser.error("epochs, batch_size, and max_patches must be positive")
    if args.event_weight <= 0:
        parser.error("event_weight must be positive")
    return args


def main() -> None:
    args = parse_args()
    from main import GBMLGGConfig, UCECConfig, run_gbmlggmodel, run_ucecmodel

    common = {
        "data_pkl": args.data_pkl,
        "out_dir": args.out_dir,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "path_dim": args.path_dim,
        "omic_dim": args.omic_dim,
        "path_hidden": args.path_hidden,
        "omic_hidden": args.omic_hidden,
        "attn_dim": args.attn_dim,
        "dropout": args.dropout,
        "gpu": args.gpu,
        "seed": args.seed,
    }

    if args.model == "gbmlggmodel":
        run_gbmlggmodel(
            GBMLGGConfig(
                n_heads=args.n_heads,
                attn_dropout=args.attn_dropout,
                lambda_wt=args.lambda_wt,
                lambda_orth=args.lambda_orth,
                wt_head_dim=args.wt_head_dim,
                adapter_dim=args.adapter_dim,
                no_gate=args.no_gate,
                no_cross_attn=args.no_cross_attn,
                no_adapter=args.no_adapter,
                lambda_wt_zero=args.lambda_wt_zero,
                wt_head_global=args.wt_head_global,
                **common,
            )
        )
    else:
        assert args.feat_dir is not None
        run_ucecmodel(
            UCECConfig(
                feat_dir=args.feat_dir,
                max_patches=args.max_patches,
                event_weight=args.event_weight,
                **common,
            )
        )


if __name__ == "__main__":
    main()
