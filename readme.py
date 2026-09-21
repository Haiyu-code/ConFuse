# Multimodal survival models

The repository provides two model configurations through one consistent interface:

| Model | Cohort | Inputs |
|---|---|---|
| `gbmlggmodel` | GBMLGG | Mean WSI features and RNA features |
| `ucecmodel` | UCEC | WSI patch features and RNA features |

`main.py` contains the data, model, training, evaluation, and output implementation. `run.py` is the command-line entry point.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the CUDA build of PyTorch appropriate for the target server when GPU training is required.

## Data formats

### `gbmlggmodel`

The pickle passed through `--data_pkl` must contain:

- `data_pd`: pandas DataFrame indexed by patient ID and containing `idh mutation`.
- `cv_splits`: fold mapping/list. Every fold contains `train` and `test`; each split contains `x_patname`, `x_path`, `x_omic`, `e`, and `t`.

The WSI input is the mean feature stored in `x_path`, so this model does not require `--feat_dir`.

### `ucecmodel`

The pickle passed through `--data_pkl` must contain `x_omic`, `patient_id`, `survtime`, `censorship`, `subtype`, and `splits`. Every split contains `train` and `test` index arrays.

The directory passed through `--feat_dir` must contain one `<patient_id>.pt` per patient. Each file must be either a `[number_of_patches, path_dim]` tensor or a dictionary containing that tensor under `features`.

Only load trusted `.pkl` and `.pt` files because both formats may contain Python pickle payloads.

## Running

The commands use the same structure; only the selected model and its required data change.

```bash
python run.py \
  --model gbmlggmodel \
  --data_pkl /path/to/gbmlgg.pkl \
  --out_dir results \
  --gpu 0 \
  --seed 42
```

```bash
python run.py \
  --model ucecmodel \
  --data_pkl /path/to/ucec.pkl \
  --feat_dir /path/to/wsi_features \
  --out_dir results \
  --gpu 0 \
  --seed 42
```

Use `--gpu -1` for CPU execution. Run `python run.py --help` for all parameters.

## Outputs

The test fold is evaluated once after the fixed final epoch. No checkpoint is selected using test-fold performance.

```text
results/
├── gbmlggmodel/
│   ├── gbmlggmodel_seed42_results.json
│   └── gbmlggmodel_seed42_oof_predictions.csv
└── ucecmodel/
    ├── ucecmodel_seed42_results.json
    └── ucecmodel_seed42_oof_predictions.csv
```

Each JSON file contains configuration, fold-level metrics, and aggregate metrics. Each CSV contains patient-level out-of-fold predictions.

## Before publishing

1. Add a project-owner-approved `LICENSE` file.
2. Add authors, institution, contact, funding, paper link, and citation metadata.
3. Document how the cohorts and WSI features can be obtained without redistributing restricted patient data.
4. Re-run both models from a clean environment and archive the exact dependency versions.
5. Check that no patient identifiers, private paths, credentials, raw data, or OOF prediction files are committed.
