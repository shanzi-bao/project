# Entropy Dynamics as a Diagnostic of Information Accumulation in Graph Neural Networks

Code and results for the L65 Mini-Project.

## Overview

We study predictive entropy from layer-wise linear probes as a practical diagnostic signal for representation dynamics in message-passing graph neural networks. After training a GNN for node classification, we freeze the model and fit an independent linear probe at each layer to obtain an entropy trajectory across depth. These trajectories distinguish depth-induced degradation (oversmoothing) from heterophily-driven structural mismatch, and provide a node-level uncertainty signal that complements existing smoothness diagnostics.

## Repository Structure
```
├── gcn.py                  # GCN model and training utilities
├── gat.py                  # GAT model (sparse attention) and training utilities
├── linear_probe.py         # Linear probing and AUC probing functions
├── my_datasets.py          # Cora/PubMed loader (Planetoid splits)
├── synthetic_dataset.py    # GraphUniverse synthetic graph loader
├── plot_multiseeds.py      # Multi-seed aggregation and plotting utilities
├── 1.ipynb                 # Main experiment notebook
├── graphuniverse/          # Synthetic graphs (homophily 0.9, 0.5, 0.1)
└── results/                # All experimental results (pkl) and figures (png)
    ├── gcn_h32_cora/
    ├── gcn_h32_pubmed/
    ├── sgat4h_relu_cora/
    ├── sgat4h_relu_pubmed/
    ├── gcn_h32_{high,mid,low}_homo_sy/
    └── sgat4h_relu_{high,mid,low}_homo_sy/
```

## Models

- **GCN**: Symmetric-normalised adjacency, ReLU, hidden dim 32, depths 1–20.
- **GAT**: Sparse multi-head attention (4 heads × 8 dim = 32), ReLU, depths 1–20.

## Datasets

| Dataset | Nodes | Classes | Split |
|---------|-------|---------|-------|
| Cora | 2,708 | 7 | Planetoid (140/500/1000) |
| PubMed | 19,717 | 3 | Planetoid (60/500/1000) |
| Synthetic (h=0.9/0.5/0.1) | 500 | 5 | GraphUniverse |

## Results

- `results/` — All experimental data and intermediate figures. Each model directory contains:
  - `probe_results.pkl`: Probe accuracy and entropy at every layer.
  - `trace_results.pkl`: Entropy trajectories split by correct/incorrect nodes.
  - `auc_results.pkl`: Per-layer entropy and confidence data for AUC and Spearman analysis.
- `results_final/` — Figures used in the paper.

## Reproducing

1. Open `experiments.ipynb` in Google Colab (GPU runtime).
2. Run setup cells (1–4), then skip to figure cells — pre-computed results are included.
3. To retrain from scratch, run cells 6–8 (~2 hours on GPU).

All experiments: 5 seeds (42, 123, 456, 789, 1024), mean ± std reported.

## Requirements

- Python 3.10+, PyTorch 2.x, PyTorch Geometric 2.x
- matplotlib, seaborn, scipy, numpy
