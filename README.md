# TRVelo: Decoupling RNA Velocity from Splicing Kinetics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-1.7-3C2179.svg)](https://www.pyg.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **TRVelo** is a deep learning-based framework for RNA velocity inference that shifts the paradigm from splicing kinetics to **transcriptional regulation**. By integrating Gene Regulatory Networks (GRNs) with total mRNA expression, TRVelo robustly reconstructs cellular dynamics and latent time, overcoming the sparsity limitations of traditional splicing-based methods.

---

## 📖 Introduction

Single-cell RNA velocity has revolutionized the analysis of cellular dynamic processes. However, conventional approaches rely heavily on the ratio of unspliced to spliced reads, a signal that is often stochastic and sparse.

**TRVelo** (Transcriptional Regulation Velocity) introduces a mechanistic alternative. It posits that the rate of change in gene expression is causally driven by the abundance of upstream Transcription Factors (TFs). By leveraging a **Graph Attention Network (GAT)** to model these regulatory interactions and a **Sine-based dynamical function** to capture complex expression profiles, TRVelo provides a robust, splicing-free solution for trajectory inference.

### Key Advantages

* **Splicing-Independent**: Operates directly on standard total expression matrices (e.g., from CellRanger), eliminating the need for complex velocyto/loom preprocessing.
* **Mechanistic & Interpretable**: Explicitly models velocity as a function of TF regulation, offering insights into the gene regulatory networks driving cell fate.
* **Flexible Dynamics**: Utilizes a flexible sine-wave function to model non-monotonic gene expression patterns (e.g., transient upregulation), surpassing rigid steady-state assumptions.
* **Global Latent Time**: Jointly infers a unified, globally consistent latent time across the entire transcriptome.

---

## 🧬 Methodology

The TRVelo framework consists of a dual-path deep learning architecture designed to decouple cell-specific temporal progression from gene-specific kinetic parameters.

### 1. Regulation-Driven Velocity Equation
Instead of modeling the degradation of unspliced RNA, TRVelo models the production of total RNA as driven by the weighted sum of transcription factor activities. The velocity for a target gene $g$ is formulated as:

$$
\frac{dE_g(t)}{dt} = \alpha \cdot \left( \sum_{j \in TFs} W_{j,g} \cdot E_j(t) \right) - \gamma_g E_g(t)
$$

Where:
* $W_{j,g}$ represents the regulatory strength of TF $j$ on gene $g$, learned via the GAT module constrained by prior GRN knowledge.
* $\gamma_g$ is the gene-specific degradation rate.

### 2. Sine-Based Temporal Dynamics
To capture multi-phase transcriptional dynamics (induction, peak, repression), we model the expression profile $E_g(t)$ over a latent time $t \in [0, 1]$ using a learnable sine function:

$$
E_g(t) = A_g \sin(\omega_g t + \phi_g) + u_g
$$

This formulation allows TRVelo to recover complex trajectories, including cyclic and transient patterns, which are challenging for traditional dynamical models.

---

## 🛠️ Installation

**System Requirements:** Ubuntu 22.04, CUDA 11.3

TRVelo is built on **PyTorch 1.11** and **PyTorch Geometric 1.7**.

### Step 1: Create Environment
```bash
conda create -n trvelo_env python=3.8
conda activate trvelo_env
```
### Step 2: Install PyTorch
```bash
# PyTorch 1.11 with CUDA 11.3
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
```
### Step 3: Install PyTorch Geometric
```bash
conda install pyg -c pyg
```
### Step 4: Install TRVelo
```bash
git clone https://github.com/LiangYu-Xidian/TRVelo.git
cd TRVelo
pip install -e . -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```

---

## 📂 Data Preparation

TRVelo requires standard **Total mRNA Expression** and a **Prior GRN**.
**Note:** You do *not* need separate spliced/unspliced matrices.

Ensure your data directory is structured as follows:
```text
TRVelo/
├── data/
│   ├── human_cd34_bone_marrow_processed.h5ad  	# AnnData object
├── TFs/
│   └── gene_attribute_edges.txt			    # GRN priors
```

---

## 🏃 Usage

The core training pipeline is encapsulated in `main.py`. The process involves **Initialization** (using Spearman correlation) and a **Multi-Stage Training** protocol (Pre-training followed by iterative refinement).

### Configuration

TRVelo exposes key hyperparameters in `configs/default.yaml`.

```yaml
# Model Architecture
embedding_dim: 128        # Dimension of gene/cell embeddings
hidden_dim: 256            # Hidden dimension for GAT and Residual MLPs
heads: 1                   # Number of attention heads in GAT
vel_num_layers: 1          # VelocityGAT layers
time_num_layers: 8         # LatentTime layers
dropout: 0.1               # Dropout rate

# Training
epochs: 300                # Epochs per phase
batch_size: 512            # Batch size
learning_rate: 1e-5        # LR (Pre-training)
learning_rate_train: 1e-5  # LR (fine-tuning phases)

# Loss Weights
lambda_velocity: 1e-9      # Velocity consistency loss
lambda_steady: 1e-9        # Steady-state constraints
lambda_scale: 0.1          # Scale factor regularization
lambda_smooth: 0.1         # Neighborhood smoothness

# Strategy Gates
velocity_start: 100        # Epoch velocity/steady loss starts (pretrain)
init_tf_end: 100           # Epoch TF init loss ends
init_self_end: 100         # Epoch self init loss ends
init_param_end: 300        # Epoch param init loss ends
scale_freeze_end: 500      # Epoch scale unfreezes
steady_in_phase3: false    # Keep steady loss in Train_III
smooth_in_phase1: false    # Enable smoothness in Train_I
```


### Quick Start — Load Pre-trained Weights

```python
import torch, scanpy as sc
from src.model import VelocityGAT, LatentTime

# 1. Load data
adata = sc.read('data/DentateGyrus_processed.h5ad')
n_genes, n_cells = adata.n_vars, adata.n_obs

# 2. Build model (fixed architecture)
in_channels = n_genes + 128        # genes + Node2Vec embedding
vel_model   = VelocityGAT(in_channels, num_TFs=37)    # 37 TFs for DentateGyrus
time_model  = LatentTime(n_cells, n_genes)

# 3. Load weights
vel_model.load_state_dict(torch.load('model/DentateGyrus_velocity_model_weights.pth'))
time_model.load_state_dict(torch.load('model/DentateGyrus_latent_time_model_weights.pth'))
vel_model.eval(); time_model.eval()
```

<<<<<<< HEAD
**Example: Human CD34+ bone marrow data (included):**

```bash
# run inference directly with pre-trained weights
python main.py --dataset human_cd34_bone_marrow --load-model ./model
```
=======
| Dataset | `num_TFs` | Weights |
|---------|-----------|---------|
| DentateGyrus | 37 | `DentateGyrus_velocity_model_weights.pth` / `..._latent_time_...` |
| endocrinogenesis_day15 | 29 | `endocrinogenesis_day15_velocity_model_weights.pth` / `..._latent_time_...` |
>>>>>>> 850b3de (Update Quick Start with load-model example)

## ✉️ Contact

For any questions, issues, or suggestions, please feel free to open an issue or contact the authors:

* **Liang Yu** (Xidian University)
* **Chenguang Zhao** (Fourth Military Medical University)

---

*This repository is the official implementation of TRVelo.*
