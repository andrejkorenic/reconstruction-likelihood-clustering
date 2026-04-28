# RLC API — CLI contract for downstream consumers

This document describes the **command-line contract** that downstream projects (`calcium_analysis`, future analyses) rely on. Communication happens through:

1. CLI flags passed to `run.py` (training) and `analyze.py` (post-training analysis).
2. Output files written to disk by those scripts, with documented shape/dtype/format guarantees.

The contract is enforced by `unit_tests/test_cli_contract.py`. If that test passes, the artefacts described below are guaranteed to be produced as specified.

> **No Python import API.** RLC is invoked as a subprocess only. Downstream consumers should never `from rlc import ...`. This keeps RLC's heavy PyTorch dependencies out of consumer environments and makes the contract version-stable across Python/PyTorch upgrades.

---

## Workflow

```
1. Train a model:        uv run python run.py [training flags]
                         → writes pretrained_models/<run_id>/checkpoint_best.pth
                                                            <model_name>.config

2. Run analyses:         uv run python analyze.py --dir <run_dir> [analysis flags]
                         → writes artefacts under <run_dir>/ or paths from flags
```

`<run_dir>` is the timestamp directory printed by `run.py` (line `MODEL_DIR: <path>` on stdout).

---

## `run.py` — training

**Required for any run:**

| Flag | Type | Notes |
|---|---|---|
| `--model_name` | choice | One of `vae`, `iwae`, `iwae_2level`, `hvae_2level`, `convhvae_2level`, `pixelhvae_2level`, `timeseries_vae`, `convvae`. |
| `--dataset_name` | str | Built-in: `dynamic_mnist`, `static_mnist`, `freyfaces`, `histopathologyGray`, `omniglot`, `caltech101silhouettes`, `cifar10`. Time-series benchmark: `ecg5000`, `synthetic_timeseries`. Bring-your-own time-series: `tabular_timeseries` (see [section below](#tabular-time-series--bring-your-own-data)). |

**Standard hyperparameters (most relevant):**

| Flag | Default | Meaning |
|---|---|---|
| `--prior` | `standard` | One of `standard`, `vampprior`, `exemplar_prior`. |
| `--number_components` | `500` | Number of pseudo-inputs (vampprior) or exemplars (exemplar_prior). For VampPrior on calcium we use 200. |
| `--z1_size`, `--z2_size` | 40 | Latent dims (z2 only used by 2-level models). |
| `--h_size` | 300 | Hidden layer width. |
| `--epochs` | 2000 | Training cap. |
| `--early_stopping_epochs` | 50 | Validation-loss patience. |
| `--warmup` | 100 | KL warm-up epochs. |
| `--lr` | 5e-4 | Adam learning rate. |
| `--batch_size` | 100 | Train batch size. |
| `--seed` | 42 | Master RNG seed. |

**TimeSeriesVAE-specific:** `--seq_len`, `--trend_poly`, `--reconstruction_wt`, `--reconstruction_dist` (default `beta`), `--plot_timesteps`.

**Auto-discover latent dim:** `--auto_z_size`, `--au_check_interval`, `--au_stability_count`, `--au_threshold`.

### Tabular time-series — bring your own data

`--dataset_name tabular_timeseries` is the unified entry point for arbitrary time-series in CSV/TSV/Parquet/NPY. No project-specific column conventions, no hardcoded splits.

**Path (one of):**

| Flag | Notes |
|---|---|
| `--ts_path` | Single file. The loader auto-splits into train/val/test per `--ts_split`. |
| `--ts_train_path` + `--ts_val_path` + `--ts_test_path` | Pre-split mode. All three required if any is set. Each file is format-resolved independently — mixing `.csv` + `.parquet` + `.npy` across the three is legal. |

**Format / parsing:**

| Flag | Default | Notes |
|---|---|---|
| `--ts_format` | `auto` | One of `{auto, csv, parquet, npy}`. `auto` reads from extension: `.csv`/`.tsv`→csv, `.parquet`/`.pq`→parquet, `.npy`→npy. |
| `--ts_csv_sep` | `,` | Separator for csv format. A `.tsv` file with the default `,` silently falls back to `\t`. |
| `--ts_value_cols` | (required for csv/parquet) | Sample-column selector. Regex if no comma (e.g. `'^t_\d+$'`); explicit comma-separated list if comma present (e.g. `'a,b,c'`). Ignored for npy. Output columns are sorted by name for determinism — for time order, prefer zero-padded names like `t_000, t_001, …`. |
| `--ts_label_col` | None | Optional label column. Strings are factorised to `0..K-1` ints (first-seen order) via `pd.factorize`. None → all-zero labels. Ignored for npy. |

**Split + normalisation:**

| Flag | Default | Notes |
|---|---|---|
| `--ts_split` | `0.8/0.1/0.1` | Train/val/test ratios as `a/b/c`, must sum to 1.0 (±1e-6). Used only in single-file mode. |
| `--ts_normalise` | `global_minmax` | One of `{global_minmax, per_sample_minmax, zscore, none}`. Statistics are computed from the train split only and applied identically to val and test (no test-set leakage). `none` is a passthrough — caller is responsible for the data being in whatever range the decoder likelihood expects (Beta needs `[0, 1]`). |

**Worked examples:**

```bash
# 1) Single-file CSV with regex column selection (calcium parquet style with t_NNN columns)
uv run python run.py --dataset_name tabular_timeseries \
    --ts_path data/segment1.parquet --ts_value_cols '^t_\d+$' \
    --model_name timeseries_vae --prior vampprior --number_components 200

# 2) TSV with explicit column list (no comma in --ts_value_cols means regex; comma means list)
uv run python run.py --dataset_name tabular_timeseries \
    --ts_path data/eeg.tsv --ts_value_cols 'ch1,ch2,ch3' --model_name timeseries_vae

# 3) Pre-split (3 files; each format auto-detected per file extension)
uv run python run.py --dataset_name tabular_timeseries \
    --ts_train_path data/train.csv --ts_val_path data/val.csv --ts_test_path data/test.csv \
    --ts_value_cols '^x_' --model_name timeseries_vae

# 4) With label column (unlocks --KNN / --classify in analyze.py)
uv run python run.py --dataset_name tabular_timeseries \
    --ts_path data/segment1.parquet --ts_value_cols '^t_\d+$' \
    --ts_label_col group --model_name timeseries_vae

# 5) Z-score normalisation (e.g. EEG-style mean-zero unit-variance)
uv run python run.py --dataset_name tabular_timeseries \
    --ts_path data/eeg.parquet --ts_value_cols '^ch_\d+$' \
    --ts_normalise zscore --model_name timeseries_vae

# 6) NPY input (raw (N, T) float array; --ts_value_cols and --ts_label_col are ignored)
uv run python run.py --dataset_name tabular_timeseries \
    --ts_path data/series.npy --ts_normalise none --model_name timeseries_vae
```

**Output artefacts (under `pretrained_models/<run_id>/<timestamp>/`):**

| File | Format | Contract |
|---|---|---|
| `checkpoint_best.pth` | torch state_dict | Best validation-loss checkpoint, used by `analyze.py --dir`. |
| `<model_name>.config` | torch-pickled `argparse.Namespace` | Captures every CLI flag used at training time. Reloaded by `analyze.py` to reconstruct the model architecture. |
| `train.log` | text | Per-epoch loss, RE, KL. |

`run.py` prints `MODEL_DIR: <abs_path>` to stdout — downstream scripts should parse this line to get the directory.

---

## `analyze.py` — post-training analysis

**Required for every invocation:**

| Flag | Type | Notes |
|---|---|---|
| `--dir` | str | Path to the timestamp directory containing `checkpoint_best.pth`. Trailing `/` optional. |

**Common settings:** `--no-cuda`, `--seed`, `--batch_size`, `--training_set_size`.

### Analysis flags

At least one of the following must be specified, otherwise `analyze.py` exits with an error listing valid flags:

| Flag | Type | Output | Contract |
|---|---|---|---|
| `--cluster` | bool | `<dir>/cluster_metrics.csv` + plots | LOO-kNN, K-Means, HDBSCAN over latents; t-SNE/UMAP scatters. Sub-flags: `--cluster_methods`, `--cluster_embeddings`. |
| `--recon_viz` | bool | `<dir>/recon_likelihood/*.png` | Best/worst reconstructions per class by `log p(x \| z)`. |
| `--generate` | bool | `<dir>/generated/*.png` | Reference-based generation grid (or time-series overlays for TS models). |
| `--KNN` | bool | stdout / log | KNN classification accuracy over latents (k = 3, 5, 7, 10, 20, 50, 75, 100, 200). |
| `--classify` | bool | `<dir>/classification/...` | Trains an MLP classifier on latents. Hparams: `--classify_hidden_units`, `--classify_lr`, `--classify_epochs`, `--classify_lambda`. |
| `--ood_scores` | bool | `<dir>/ood_scores.csv` (+ histogram if `--ood_dataset`) | Per-level ELBO decomposition (RE, KL1, KL2). Requires a hierarchical model. |
| `--cyclic_generation` | bool | `<dir>/cyclic_generation.png` | Linear interpolation in latent space between two test inputs. |
| `--export_latents` | bool | `<out_z>` (default `<dir>/z_mean.npy`) | See "export_latents contract" below. |
| `--export_pseudo_prototypes` | path | path arg | See "export_pseudo_prototypes contract" below. |
| `--ood_recon_nll` | path | path arg | Method E v1 — per-trace recon NLL via `--ood_K` MC samples. Lower nats = more in-distribution. |
| `--ood_pseudo_recon` | prefix | `<prefix>.nll.npy` + `<prefix>.cluster.npy` | Method E v2 — score every ROI against each VampPrior pseudo-input prototype, take max. |
| `--ts_path_override` | path | (modifier) | Encode an arbitrary csv/parquet/npy through the trained model instead of the training-time `--ts_path`. Applies to `--export_latents`, `--ood_recon_nll`, `--ood_pseudo_recon`. Format is auto-detected per the file extension (or trained-time `config.ts_format` if set). |
| `--out_z` | path | (modifier) | Override default `z_mean.npy` location for `--export_latents`. |
| `--ood_K` | int | (modifier, default 10) | MC samples for `--ood_recon_nll`. |
| `--ood_dataset` | str | (modifier) | Second dataset for `--ood_scores` comparison. |

### Output artefact contracts

#### `--export_latents`

Encodes every row of the input parquet/CSV in **original order** (no shuffle) using `model.q_z(x)`'s mean.

| Property | Contract |
|---|---|
| Path | `--out_z` if set, else `<dir>/z_mean.npy`. |
| Shape | `(N, z1_size)` where `N` = rows in source data, `z1_size` from training config. |
| Dtype | `float32`. |
| Stdout marker | `LATENTS_PATH: <abs_path>` (consumers should parse this). |
| Source | `--ts_path_override` if set; else `config.ts_path`; else `config.ts_train_path`. At least one must resolve. |
| Normalisation | Same global MinMax as training (`x = (x - x_min) / (x_max - x_min + 1e-7)`). |

#### `--export_pseudo_prototypes`

Decodes the K VampPrior pseudo-inputs through the encoder + decoder pipeline into raw waveforms.

| Property | Contract |
|---|---|
| Path | The path passed as the flag value (not derived). |
| Shape | `(K, D)` where `K = config.number_components`, `D = product(input_size)`. |
| Dtype | `float32`. |
| Value range | `[0, 1]` for Beta likelihood (TimeSeriesVAE default). |
| Stdout marker | `PSEUDO_PROTOTYPES_PATH: <abs_path>`. |
| Requires data loading | **No** — `analyze.py` short-circuits dataset loading when `--export_pseudo_prototypes` is the only requested op. |
| Fallback for non-VampPrior priors | K samples from N(0, 1) decoded through the decoder (deterministic via `torch.manual_seed(0)`). |

#### `--ood_recon_nll`

| Property | Contract |
|---|---|
| Path | The path passed as the flag value. |
| Shape | `(N,)` where `N` = rows in source data. |
| Dtype | `float32`. |
| Semantics | Per-trace negative log-likelihood (nats), averaged over `--ood_K` MC samples from `q(z\|x)`. Lower = more in-distribution. |
| Stdout marker | `RECON_NLL_PATH: <abs_path>`. |

#### `--ood_pseudo_recon`

| Property | Contract |
|---|---|
| Paths | `<prefix>.nll.npy` and `<prefix>.cluster.npy`. |
| `.nll.npy` shape/dtype | `(N,) float32` — `-max_k log p(x_n \| prototype_k)` over the K prototypes. |
| `.cluster.npy` shape/dtype | `(N,) int32` — argmax-k assignment for each ROI. |
| Stdout marker | `PSEUDO_RECON_PREFIX: <abs_path_prefix>`. |

---

## Stable consumer-side patterns

`calcium_analysis` invokes RLC like this:

```python
import subprocess, numpy as np

# Export decoded VampPrior prototypes for Method F
subprocess.run([
    "uv", "run", "python", "analyze.py",
    "--dir", str(checkpoint_dir),
    "--export_pseudo_prototypes", str(out_path),
    "--no-cuda",
], cwd=str(rlc_root), check=True)

prototypes = np.load(out_path)        # (K, D) float32, in [0, 1]
assert prototypes.dtype == np.float32 and prototypes.shape[0] == 200
```

This pattern is exercised end-to-end by `unit_tests/test_cli_contract.py`.
