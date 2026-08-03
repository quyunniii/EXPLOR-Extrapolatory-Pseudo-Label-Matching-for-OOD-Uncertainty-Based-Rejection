# EXPLOR: Reliable OOD Virtual Screening with Extrapolatory Pseudo-Label Matching

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2406.01825-blue)](https://arxiv.org/abs/2406.01825)
[![ACM BCB 2026](https://img.shields.io/badge/ACM%20BCB%202026-Full%20Paper-green)](https://acm-bcb.org)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)

Official code for the paper:

> **Reliable OOD Virtual Screening with Extrapolatory Pseudo-Label Matching**  
> Accepted as a full paper at the 17th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (ACM-BCB 2026)  
> [ACM Digital Library](https://dl.acm.org/doi/10.1145/3807503.3819476)

---

## Overview

EXPLOR addresses the problem of **reliable out-of-distribution (OOD) detection for virtual screening** — identifying when a model should abstain from a prediction because a test sample lies outside the training distribution. This is especially important in biomedical and drug discovery settings, where overconfident predictions on OOD compounds can be dangerous.

The key idea is **Extrapolatory Pseudo-Label Matching**: the method generates extrapolated pseudo-OOD samples beyond the training distribution, assigns pseudo-labels based on a PCA projection ensemble, and trains a discriminator with a combined loss that ensures calibrated, reliable uncertainty estimates. At inference, confidence scores from both the PCA ensemble and the neural discriminator are combined for robust OOD rejection.

The full EXPLOR loss has three components:
- **`in_ce`** — binary cross-entropy per head matching loss on training (in-distribution) data
- **`in_prob`** — mean-matching term to keep predicted probabilities calibrated
- **`conf_ce`** — binary cross-entropy per head matching loss on high-confidence extrapolated (pseudo-OOD) data, weighted by `alpha`

---

## Repository Structure

```
.
├── run_explor.py        # Full EXPLOR training and evaluation script
├── Dbat_EXPLOR.ipynb    # Jupyter notebook: interactive demo and analysis
├── data/                # Dataset folder (see Data Preparation below)
│   ├── dataset1/
│   │   ├── train.csv
│   │   └── val_ood.csv
│   ├── dataset3/
│   └── dataset4/
└── results/
    └── EXPLOR_results.csv   # Auto-generated results table
```

---

## Installation

**Requirements:** Python 3.8+. A GPU (CUDA or Apple MPS) is used automatically if available.

```bash
# 1. Clone the repository
git clone https://github.com/quyunniii/Extrapolatory_Pseudo_Label_Matching_for_OOD_Uncertainty_Based_Rejection.git
cd Extrapolatory_Pseudo_Label_Matching_for_OOD_Uncertainty_Based_Rejection

# 2. (Optional but recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install torch numpy pandas scikit-learn xgboost tqdm
```

---

## Data Preparation

Place your datasets under a `data/` folder in the project root. Each dataset should have two CSV files:

```
data/
├── dataset1/
│   ├── train.csv       # In-distribution training data
│   └── val_ood.csv     # OOD validation/test data
├── dataset3/
│   ├── train.csv
│   └── val_ood.csv
└── dataset4/
    ├── train.csv
    └── val_ood.csv
```

**CSV format:**
- `train.csv`: columns are `[id, label, feature_0, feature_1, ...]`
- `val_ood.csv`: columns are `[id, label, ..., feature_0, feature_1, ...]` (for `dataset1`–`dataset4`, features start at column index 4; for other datasets, at index 2)

Labels should be binary (0/1).

---

## Quick Start

### Run the full pipeline

```bash
python run_explor.py
```

This will train and evaluate EXPLOR on all three default datasets (`dataset1`, `dataset3`, `dataset4`) with 5 random seeds, and save results to `results/EXPLOR_results.csv`.

### Run on specific datasets or seeds

```bash
# Run on a single dataset
python run_explor.py --datasets dataset1

# Run with a custom number of seeds
python run_explor.py --datasets dataset1 --seeds 10

# Run on multiple datasets
python run_explor.py --datasets dataset1 dataset3
```

**Arguments:**

| Argument | Description | Default |
|---|---|---|
| `--datasets` | Space-separated list of dataset folder names | `dataset1 dataset3 dataset4` |
| `--seeds` | Number of random seeds to average over | `5` |

### Interactive notebook

For a step-by-step walkthrough and visualizations, open the notebook:

```bash
jupyter notebook Dbat_EXPLOR.ipynb
```

---

## Key Hyperparameters

All hyperparameters are defined at the top of `run_explor.py` and can be edited directly:

| Parameter | Description | Default |
|---|---|---|
| `N_MODELS` | Number of models in the PCA projection ensemble | `1024` |
| `NDIRS` | PCA directions used per ensemble member | `64` |
| `NDIRS_TOTAL` | Total PCA directions computed | `128` |
| `ALPHA` | Weight on the extrapolatory pseudo-label loss (`conf_ce`) | `0.5` |
| `CONF_RANGE` | Range around 0/1 to define high-confidence pseudo-OOD samples | `0.1` |
| `CONF_THRESH` | Distance from 0.5 threshold for confident sample selection | `0.4` |
| `N_EXPAND_ROUNDS` | Number of extrapolation rounds to build the pseudo-OOD batch | `5` |
| `HIDDEN_SIZE` | Hidden layer size of the neural discriminator | `256` |
| `N_HIDDEN` | Number of hidden layers | `2` |
| `TR_ITER` | Training iterations | `20001` |
| `LR` | Learning rate | `5e-4` |
| `BATCH_SIZE` | Batch size | `256` |

---

## Outputs

Results are appended to `results/EXPLOR_results.csv` in the format:

| model | metric | d1_val | d3_val | d4_val |
|---|---|---|---|---|
| EXPLOR_full | AUROC | mean±se | mean±se | mean±se |
| EXPLOR_full | AUPRC_recall0.1 | ... | ... | ... |
| ... | | | | |

Metrics reported:
- **AUROC** — area under the ROC curve for OOD detection
- **AUPRC@r** — area under the precision-recall curve up to recall threshold `r` (normalized), for `r ∈ {0.1, 0.2, 0.3, 0.4, 1.0}`

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inbook{10.1145/3807503.3819476,
author = {Qu, Yunni and Vaduri, Bhargav and Jatoth, Karthikeya and Wellnitz, James and Dinh, Dzung and Veenbaas, Seth and Chapman, Jonathan and Tropsha, Alexander and Oliva, Junier},
title = {Reliable OOD Virtual Screening with Extrapolatory Pseudo-Label Matching},
year = {2026},
isbn = {9798400726538},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3807503.3819476},
booktitle = {Proceedings of the 17th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics},
articleno = {37},
numpages = {10}
}

```

---

## License

This project is released as open-source for research use. Please see the repository for license details.

---

## Contact

For questions about the code or paper, please open a [GitHub Issue](https://github.com/quyunniii/Extrapolatory_Pseudo_Label_Matching_for_OOD_Uncertainty_Based_Rejection/issues).
