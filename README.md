<a id="readme-top"></a>

<!-- ABOUT THE PROJECT -->
<h1 align="center">Revisiting Reconstruction Likelihood</h1>

[![Python](https://img.shields.io/badge/Python-3.13-yellow?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Version](https://img.shields.io/badge/Version-1.0.0-blue)


## Variational Autoencoders for Biological and Biomedical Data Clustering

This project investigates whether Variational Autoencoders (VAEs) trained purely for reconstruction implicitly encode meaningful cluster structure in their latent space — without any explicit clustering objective. We benchmark four VAE variants (standard VAE, IWAE, VampPrior, Exemplar VAE) on MNIST and evaluate their latent representations using t-SNE, UMAP, k-means, and HDBSCAN. We further explore **reconstruction likelihood** — the stochastic log p(x|z) — as a principled alternative to deterministic reconstruction error for unsupervised anomaly detection and cluster quality assessment.

The codebase is built on top of revamped version of [Exemplar VAE](https://arxiv.org/abs/2004.04795) (NeurIPS 2020) and extended with clustering analysis (LOO-kNN, K-Means, HDBSCAN), dimensionality reduction (t-SNE, UMAP).

<br/>

## 📌 Features

### Models

| Model | Description | Reference |
|---|---|---|
| **VAE** | Standard single-level VAE | [Kingma & Welling, 2014](https://arxiv.org/abs/1312.6114) |
| **IWAE** | Importance Weighted Autoencoder (K samples) | [Burda et al., 2015](https://arxiv.org/abs/1509.00519) |
| **IWAE 2-level** | IWAE with two stochastic hidden layers | [Burda et al., 2015](https://arxiv.org/abs/1509.00519) |
| **ConvVAE** | Single-level convolutional VAE (GatedConv encoder + transposed conv decoder) | — |
| **HVAE 2-level** | Two-level hierarchical VAE | — |
| **ConvHVAE** | Convolutional two-level hierarchical VAE | — |
| **PixelCNN HVAE** | PixelCNN decoder + hierarchical latents | [van den Oord et al., 2016](https://arxiv.org/abs/1606.05328) |
### Priors

| Prior | Description | Reference |
|---|---|---|
| **Standard** | Isotropic Gaussian N(0, I) | — |
| **VampPrior** | Variational mixture of posteriors with learnable pseudo-inputs | [Tomczak & Welling, 2018](https://arxiv.org/abs/1705.07120) |
| **Exemplar Prior** | Data-driven prior with leave-one-out masking | [Norouzi et al., 2020](https://arxiv.org/abs/2004.04795) |

### Additional capabilities

* **YAML configuration** — Annotated config files with CLI override support (`--config configs/vae_standard.yaml --epochs 50`).
* **Eleven datasets** — Dynamic MNIST, Static MNIST, Fashion MNIST, Omniglot, Caltech101 Silhouettes, HistopathologyGray, FreyFaces, SVHN, CIFAR-10, ECG5000, Synthetic Time Series. Most are downloaded automatically.
* **Clustering analysis** — LOO-kNN, K-Means, HDBSCAN on raw latent space with t-SNE and UMAP scatter plots (`--cluster` flag).
* **Posterior analysis** — Visualization of true vs variational posterior + SIR resampling for 2D latent models (`posterior_analysis.py`).
* **Adaptive LR scheduling** — `ReduceLROnPlateau` monitors validation loss; `BetaScheduler` ramps KL weight 0 → 1 during warm-up.
* **Slurm support** — Cluster experiment management with `--slurm_job_id` / `--slurm_task_id` for deterministic directory naming.
* **Robust checkpointing** — Atomic `state_dict` saves; best checkpoint loaded automatically for final evaluation.
* **Reconstruction likelihood visualization** — Per-class best/worst sample grids sorted by log p(x|z) (`--recon_viz` flag).
* **Post-training analysis** — Generation, KNN retrieval, classification, t-SNE/UMAP via `analyze.py`.

<br/>

### 📄 Citation

This project is based on the following paper. If you find the code useful, please consider citing the original work:

> Andrej Korenić, Ufuk Özkaya, Abdulkerim Çapar. Revisiting Reconstruction Likelihood: Variational Autoencoders for Biological and Biomedical Data Clustering. _bioRxiv_ 2026. doi: 10.64898/2026.04.09.717460

<p align="right"><a href="#readme-top">⏫</a></p>

<br/>

<!-- GETTING STARTED -->

## Getting Started

### 🔧 Installation

This project uses [uv](https://github.com/astral-sh/uv) for environment and dependency management (Python 3.13).

1. **Install uv** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. **Clone the repository:**
    ```bash
    git clone https://github.com/andrejkorenic/reconstruction-likelihood-clustering.git
    cd reconstruction-likelihood-clustering
    ```

3. **Create the environment and install dependencies:**
    ```bash
    uv sync
    ```

Datasets are downloaded automatically to the `datasets/` folder on first use (for those available via `torchvision`). For Omniglot, the data file is fetched automatically via `wget`. Static MNIST, Caltech101 Silhouettes, HistopathologyGray, and FreyFaces require manual download — see comments in `utils/load_data/data_loader_instances.py` for the expected file paths.

### 🧪 Testing

The test suite uses [pytest](https://pytest.org). Unit tests run in under 2 minutes; slow smoke tests train real models and take 15–30 minutes.

```bash
# Unit tests only (fast)
uv run pytest unit_tests/ -q

# Full suite including end-to-end smoke tests
uv run pytest unit_tests/ --run-slow -x -v
```

<p align="right"><a href="#readme-top">⏫</a></p>

<br/>

### 📖 Usage

#### YAML configuration (recommended)

Each experiment is described by a single YAML file with annotated parameters. CLI arguments override YAML values.

```bash
# Run from a config file
uv run python run.py --config configs/vae_standard.yaml

# Override specific values from the command line
uv run python run.py --config configs/iwae.yaml --K 50 --epochs 100

# Exemplar VAE
uv run python run.py --config configs/vae_exemplar.yaml
```

See `configs/` for all available configurations:

| Config file | Model | Prior |
|---|---|---|
| `configs/vae_standard.yaml` | VAE | Standard N(0,I) |
| `configs/vae_vampprior.yaml` | VAE | VampPrior |
| `configs/vae_exemplar.yaml` | VAE | Exemplar Prior |
| `configs/iwae.yaml` | IWAE (K=5) | Standard |
| `configs/iwae_k50.yaml` | IWAE (K=50) | Standard |
| `configs/vae_exemplar_paper.yaml` | VAE (paper settings) | Exemplar Prior |

#### CLI-only (without YAML)

```bash
# Standard VAE
uv run python run.py --model_name vae --dataset_name dynamic_mnist --prior standard

# VAE with VampPrior
uv run python run.py --model_name vae --prior vampprior --number_components 500

# IWAE with 50 importance samples
uv run python run.py --model_name iwae --prior standard --K 50

# Exemplar VAE with approximate prior
uv run python run.py --model_name vae --prior exemplar_prior \
    --number_components 25000 --approximate_prior
```

#### Loading a pretrained model

```bash
uv run python run.py --dir pretrained_models/{outer_folder}/{timestamp}/
```

#### Post-training analysis

```bash
# Cluster analysis (LOO-kNN, K-Means, HDBSCAN on raw/t-SNE/UMAP)
uv run python analyze.py --dir pretrained_models/{outer_folder}/{timestamp}/ --cluster

# Reconstruction likelihood grids (best/worst per class)
uv run python analyze.py --dir pretrained_models/{outer_folder}/{timestamp}/ --recon_viz

# Generate samples from the prior
uv run python analyze.py --dir pretrained_models/{outer_folder}/{timestamp}/ --generate

# KNN retrieval in latent space
uv run python analyze.py --dir pretrained_models/{outer_folder}/{timestamp}/ --KNN

# Posterior visualization (requires z1_size=2)
uv run python posterior_analysis.py --dir pretrained_models/.../

# Compute log-likelihood on a pretrained model (S=5000 importance samples)
uv run python run.py --dir pretrained_models/{outer_folder}/{timestamp}/ --ll

# Train without computing LL at the end (faster iteration)
uv run python run.py --config configs/vae_standard.yaml --no_ll
```

#### Slurm cluster experiments

[Slurm](https://slurm.schedmd.com/) is a job scheduler used on HPC clusters (e.g. university computing facilities). It lets you run many experiments in parallel as an "array job" — each gets a unique ID that can be used to name output directories so runs don't overwrite each other.

```bash
# In a Slurm array job (sbatch --array=0-9)
uv run python run.py --config configs/vae_exemplar.yaml \
    --slurm_job_id $SLURM_JOB_ID \
    --slurm_task_id $SLURM_ARRAY_TASK_ID \
    --seed $SLURM_ARRAY_TASK_ID
```

If you're running locally, you can ignore these flags entirely.

#### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--config` | — | YAML config file (CLI args override YAML values) |
| `--dir` | — | Load pretrained model from this directory |
| `--model_name` | `vae` | `vae`, `iwae`, `iwae_2level`, `hvae_2level`, `convhvae_2level`, `convvae` |
| `--prior` | `standard` | `standard`, `vampprior`, `exemplar_prior` |
| `--dataset_name` | `dynamic_mnist` | See supported datasets above |
| `--epochs` | 2000 | Maximum training epochs |
| `--warmup` | 100 | KL warm-up epochs (beta 0 → 1) |
| `--early_stopping_epochs` | 50 | Early stopping patience |
| `--lr` | 5e-4 | Learning rate |
| `--h_size` | 300 | Hidden layer size |
| `--z1_size` | 40 | Latent dimension (first level) |
| `--z2_size` | 40 | Latent dimension (second level, hierarchical models) |
| `--K` | 1 | Importance samples per data point (K>1 for IWAE) |
| `--ll` / `--no_ll` | auto | Compute log-likelihood: auto = skip during training, compute when loading |
| `--S` | 5000 | Importance samples for log-likelihood estimation |
| `--number_components` | 500 | Pseudo-inputs (VampPrior) or exemplar subset size |
| `--cluster` | off | Run clustering analysis on test set (LOO-kNN, K-Means, HDBSCAN, t-SNE, UMAP) |
| `--cluster_methods` | `knn kmeans hdbscan` | Which clustering methods to run (any subset) |
| `--cluster_embeddings` | `raw tsne umap` | Which embedding spaces to evaluate (any subset) |
| `--auto_z_size` | off | Enable adaptive latent dimension discovery (see below) |
| `--ood_scores` | off | Compute per-level ELBO decomposition for OOD detection (hierarchical models) |
| `--ood_dataset` | — | OOD dataset name for comparison (e.g., `fashion_mnist`) |

#### Adaptive latent size discovery

If you don't know the optimal latent dimension, start with a large `z1_size` and let the model find it:

```bash
uv run python run.py --model_name vae --dataset_name dynamic_mnist --z1_size 128 --auto_z_size
```

During training, Active Units (AU) are measured every `--au_check_interval` epochs. When AU stabilizes (same count for `--au_stability_count` consecutive readings), training pauses and asks:

```
+----------------------------------------------------------+
|  Latent space analysis complete                          |
|  Active units: 35 / 128                                  |
|  Recommended z1_size: 35                                 |
|                                                          |
|  Restart training with z1_size=35? [y/n]                 |
+----------------------------------------------------------+
```

- **y** — restarts training from scratch with the discovered `z1_size`
- **n** — continues training with the original `z1_size` (monitoring stops)

The discovered value is saved to `auto_z_size_result.json` in the model output directory.

| Argument | Default | Description |
|---|---|---|
| `--auto_z_size` | off | Enable adaptive latent dimension discovery |
| `--au_check_interval` | 5 | Compute AU every N epochs (after warmup) |
| `--au_stability_count` | 5 | Consecutive stable readings to trigger prompt |
| `--au_threshold` | 0.01 | Variance threshold for "active" dimension |

#### OOD score analysis

Hierarchical VAEs decompose the ELBO into per-level contributions. The lower latent level (z1) captures local features while the upper level (z2) captures semantics. OOD data often has in-distribution low-level features but anomalous high-level structure — the partial score **L>1 = RE - KL2** (skipping lower-level KL) is a better OOD discriminator than the full ELBO.

Inspired by [Hierarchical VAEs Know What They Don't Know](https://arxiv.org/abs/2102.08248).

```bash
# Compare HVAE trained on MNIST against Fashion-MNIST
uv run python analyze.py --dir pretrained_models/.../ \
    --ood_scores --ood_dataset fashion_mnist

# Inspect in-distribution score distribution only
uv run python analyze.py --dir pretrained_models/.../ --ood_scores
```

| Output | Description |
|---|---|
| `ood_scores.csv` | Per-sample RE, KL1, KL2, ELBO, L>1 for ID and OOD datasets |
| `ood_histograms.png` | Side-by-side histograms comparing score distributions |

Supported models: `hvae_2level`, `convhvae_2level`, `pixelcnn`, `iwae_2level`.

<p align="right"><a href="#readme-top">⏫</a></p>

<br/>

### 📂 Project Structure

```plaintext
reconstruction-likelihood-clustering/
    ├── configs/                         # YAML experiment configurations
    │   ├── vae_standard.yaml
    │   ├── vae_vampprior.yaml
    │   ├── vae_exemplar.yaml
    │   ├── vae_exemplar_paper.yaml
    │   ├── iwae.yaml
    │   └── iwae_k50.yaml
    ├── models/
    │   ├── BaseModel.py                 # Core VAE logic: loss, prior dispatch, generation
    │   ├── AbsModel.py                  # Single-level base (kl_loss, forward, p_x)
    │   ├── AbsHModel.py                 # Two-level hierarchical base (z1, z2)
    │   ├── VAE.py                       # Standard VAE encoder/decoder
    │   ├── IWAE.py                      # Importance Weighted Autoencoder
    │   ├── IWAE_2level.py               # IWAE with two stochastic layers
    │   ├── HVAE_2level.py               # Two-level hierarchical VAE
    │   ├── ConvVAE.py                   # Convolutional VAE (GatedConv encoder + transposed conv decoder)
    │   ├── convHVAE_2level.py           # Convolutional two-level HVAE
    │   └── PixelCNN.py                  # PixelCNN-based HVAE (planned)
    ├── utils/
    │   ├── load_data/                   # Dataset loaders (one class per dataset)
    │   ├── training.py                  # train_vae(), train_one_epoch()
    │   ├── evaluation.py                # evaluate_vae(), final_evaluation()
    │   ├── perform_experiment.py        # Training loop, scheduling, early stopping
    │   ├── create_model.py              # Model factory + output directory setup
    │   ├── load_model.py                # Load pretrained model from checkpoint
    │   ├── model_io.py                  # Model registry, checkpoint save/load
    │   ├── ood_scores.py                # OOD detection: per-level ELBO decomposition, CSV, histograms
    │   ├── clustering.py                # Latent space analysis: kNN, K-Means, HDBSCAN, t-SNE, UMAP
    │   ├── distributions.py             # Log-likelihoods: Bernoulli, Gaussian, logistic-256, Beta
    │   └── nn.py                        # GatedDense, NonLinear, and other layers
    ├── unit_tests/                      # pytest test suite (80 tests)
    ├── configs/                         # YAML experiment configurations
    ├── datasets/                        # Downloaded datasets (auto-created)
    ├── pretrained_models/               # Training outputs (auto-created)
    │   └── {dataset}_{prior}_model_name={model}/
    │       └── {run_id}/
    │           ├── {model}.config
    │           ├── checkpoint_best.pth
    │           └── reconstruction/
    ├── run.py                           # Main entry point
    ├── analyze.py                       # Post-training analysis (cluster, recon_viz, generate, KNN, classify, OOD)
    ├── posterior_analysis.py            # True vs variational posterior + SIR
    └── pyproject.toml                   # Project metadata and dependencies
```

<p align="right"><a href="#readme-top">⏫</a></p>

<br/>

<!-- RESULTS -->

## 📊 Results

### Paper Replication Benchmark (Dynamic MNIST)

All models use MLP encoder [784→300→300→40] with GatedDense activations, AdamNormGrad optimizer (lr=5e-4), 100-epoch KL warm-up, early stopping patience 50.

| Model | Prior | Test NLL ↓ | Val ELBO | Early Stop Epoch |
|---|---|---|---|---|
| VAE | Standard N(0,I) | -84.45 | 88.04 | 930 |
| IWAE K=5 | Standard | -83.37 | 86.19 | 983 |
| IWAE K=50 | Standard | -82.88 | 84.84 | 923 |
| VampPrior | 500 pseudo-inputs | **-82.29** | 85.30 | 934 |
| ExemplarVAE | 500 components, k=10 | -82.31 | 85.38 | 865 |

Test NLL estimated with S=5000 importance samples.

<p align="right"><a href="#readme-top">⏫</a></p>

<br/>

<!-- CONTRIBUTING -->

## 📢 Found a Bug? Let Us Know!

If you encounter any issues or have suggestions for improvements, please open an issue on GitHub. Before submitting, consider including:

1. **A clear description** of the problem and why it is unexpected.
2. **Reproduction steps** — the exact command you ran and the error output.
3. **Expected vs. actual behavior.**
4. **Environment details** — Python version, PyTorch version, OS.

<p align="right"><a href="#readme-top">⏫</a></p>

<br/>

## Typical Workflows

### Train only (default)

```bash
uv run python run.py --config configs/vae_standard.yaml
```

### Train + analysis (one command)

```bash
uv run python run.py --config configs/vae_standard.yaml --cluster --recon_viz --generate
```

### Analyze a pretrained model

```bash
uv run python analyze.py --dir pretrained_models/.../timestamp/ --cluster --recon_viz --generate
```

### LL estimation on pretrained model

```bash
uv run python run.py --dir pretrained_models/.../timestamp/ --ll
```

### Combine multiple analyses

```bash
uv run python analyze.py --dir pretrained_models/.../timestamp/ --cluster --KNN --ood_scores --ood_dataset fashion_mnist
```

### Flag availability

| Flag | `run.py` | `analyze.py` |
|------|----------|-------------|
| `--ll` | ✅ | ❌ |
| `--cluster` | ✅ | ✅ |
| `--recon_viz` | ✅ | ✅ |
| `--generate` | ✅ | ✅ |
| `--KNN` | ❌ | ✅ |
| `--classify` | ❌ | ✅ |
| `--ood_scores` | ❌ | ✅ |
| `--cyclic_generation` | ❌ | ✅ |

<p align="right"><a href="#readme-top">⏫</a></p>

<br/>

<!-- ISSUES AND ROADMAP -->

## 🚩 Known Issues and Roadmap

**Roadmap:**
- [ ] **PixelCNN HVAE** — planned implementation; not yet started.
- [ ] **IWAE decomposition** — split IWAE loss into rate vs distortion components for analysis (active vs inactive latent dimensions).

<!-- completed items removed -->

<p align="right"><a href="#readme-top">⏫</a></p>

<br/>

<!-- ACKNOWLEDGMENTS -->

## 🤝 Acknowledgments

We are deeply grateful to the following people and projects:

- Michele De Vita for his assistance in understanding the work of An and Cho, as well as his efforts to implement _reconstruction probability_. His [GitHub code](https://github.com/Michedev/VAE_anomaly_detection) and insightful discussions were instrumental in structuring this study.
- [Exemplar VAE](https://arxiv.org/abs/2004.04795) paper and its [original implementation](https://github.com/sajadn/Exemplar-VAE) for the foundational architecture and training code.
- [VampPrior](https://arxiv.org/abs/1705.07120) for the pseudo-input prior concept.
- Original IWAE implementation courtesy of [Yuri Burda's IWAE repository](https://github.com/yburda/iwae).
- [nbip/IWAE](https://github.com/nbip/IWAE) repository for inspiration on the posterior visualization approach.

<p align="right"><a href="#readme-top">⏫</a></p>

<br/>

<!-- LICENCE -->

## 🏷️ Licence

This project is released under the GPL-3.0 license (see [LICENSE.txt](LICENSE.txt) for the full text). The GPL-3.0 encourages open collaboration and knowledge sharing, allowing you to use, modify, and distribute this bundle freely as long as you adhere to its terms.

<p align="right"><a href="#readme-top">⏫</a></p>
