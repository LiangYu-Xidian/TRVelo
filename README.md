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
pip install torch-geometric==1.7.2 torch-scatter==2.0.9 torch-sparse==0.6.13 torch-cluster==1.6.0 torch-spline-conv==1.2.1
```
### Step 4: Install TRVelo
```bash
git clone https://github.com/LiangYu-Xidian/TRVelo.git
cd TRVelo
pip install -e .
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

TRVelo exposes key hyperparameters in `configs/default.yaml`, allowing users to control the balance between data fidelity, biological priors, and trajectory smoothness.

The configuration file is structured as follows:

```yaml
# configs/default.yaml

# --------------------------
# Model Architecture
# --------------------------
embedding_dim:          # Dimension of gene/cell embeddings
hidden_dim:             # Hidden dimension for GAT and Residual MLPs
heads:                  # Number of attention heads in GAT
dropout:                # Dropout rate for regularization

# --------------------------
# Training Hyperparameters
# --------------------------
seed:                   # Random seed for reproducibility
epochs:                 # Training epochs per phase
batch_size:             # Batch size for cell sampling
learning_rate:          # Initial learning rate (Pre-training)
learning_rate_train:    # Learning rate for fine-tuning phases

# --------------------------
# Physics & Loss Weights
# --------------------------
lambda_velocity:    # Weight for velocity consistency loss
lambda_steady:      # Weight for steady-state constraints
lambda_scale:       # Weight for scale factor regularization
lambda_smooth:      # Weight for neighborhood smoothness
```


### Quick Start
To train the model on a specific dataset located in `data/`, run:

```bash
python main.py --dataset [DATASET_NAME]
```

## ✉️ Contact

For any questions, issues, or suggestions, please feel free to open an issue or contact the authors:

* **Liang Yu** (Xidian University)
* **Chenguang Zhao** (Fourth Military Medical University)

---

*This repository is the official implementation of TRVelo.*
