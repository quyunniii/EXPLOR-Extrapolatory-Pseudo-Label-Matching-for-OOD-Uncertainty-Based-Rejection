"""
run_explor.py

Full EXPLOR model on datasets 1, 3, 4.

Loss: in_ce + in_prob + alpha * conf_ce
  - in_ce:    BCE on training data
  - in_prob:  mean-matching on training data  (|mean(sigmoid(logits)) - mean(labels)|)
  - conf_ce:  BCE on high-confidence expanded data

Results saved to: results/EXPLOR_results.csv (model row: EXPLOR_full)

Run:
  source mmelon_env/bin/activate
  python run_explor.py
  python run_explor.py --datasets dataset1 --seeds 5
"""

import os
import warnings
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from xgboost import XGBClassifier
from tqdm import tqdm

warnings.filterwarnings("ignore")

BASE_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "EXPLOR_results.csv")

DATASETS = ["dataset1", "dataset3", "dataset4"]
THRESHS  = [0.1, 0.2, 0.3, 0.4, 1.0]
N_SEEDS  = 5

# Hyperparameters
N_MODELS        = 1024
NDIRS           = 64
NDIRS_TOTAL     = 128
SPLIT_PERC      = 0.5
HIDDEN_SIZE     = 256
N_HIDDEN        = 2
TR_ITER         = 20001
TR_AVG_ITER     = 5000
BATCH_SIZE      = 256
LR              = 5e-4
ALPHA           = 0.5
CONF_RANGE      = 0.1
CONF_THRESH     = 0.4
N_EXPAND_ROUNDS = 5

DEVICE = (
    torch.device("mps")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


def auprc_threshs(confidences, Y, threshs, as_percentages=True):
    areas = []
    precisions, recalls, _ = metrics.precision_recall_curve(Y, confidences)
    for t in threshs:
        if t < 1.0:
            interp_prec_t = np.interp(t, recalls[::-1], precisions[::-1])
            recalls_thresh = np.concatenate(([t], recalls[recalls < t]))
            precisions_thresh = np.concatenate(([interp_prec_t], precisions[recalls < t]))
        else:
            recalls_thresh = recalls
            precisions_thresh = precisions
        area_t = metrics.auc(recalls_thresh, precisions_thresh)
        if as_percentages:
            area_t = area_t / t
        areas.append(area_t)
    return np.array(areas)


class PCAProjectionEnsemble:
    def __init__(self, Vpca, nmodels=256, ndirs=8, split_perc=0.5):
        self.Vpca       = Vpca
        self.nmodels    = nmodels
        self.ndirs      = ndirs
        self.split_perc = split_perc
        self._estimators = []
        self._dirs_inds  = []

    def _base_model(self):
        return XGBClassifier(
            n_estimators=40,
            subsample=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
        )

    def fit(self, X, Y):
        XV = X @ self.Vpca
        for _ in range(self.nmodels):
            in_range = np.random.rand(len(X)) <= self.split_perc
            rprm = np.random.permutation(self.Vpca.shape[1])
            di = rprm[: self.ndirs]
            self._dirs_inds.append(di)
            est = self._base_model()
            est.fit(XV[in_range][:, di], Y[in_range])
            self._estimators.append(est)

    def predictions(self, X):
        XV = X @ self.Vpca
        return np.stack(
            [e.predict(XV[:, di]) for di, e in zip(self._dirs_inds, self._estimators)],
            axis=-1,
        ).astype(np.float32)

    def mean_conf(self, X=None, preds=None):
        if preds is None:
            preds = self.predictions(X)
        return np.mean(preds, axis=-1)


class DatasetDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_size=512, nhidden_layers=2, out_size=256):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_size), nn.ELU()]
        for _ in range(nhidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ELU()]
        layers.append(nn.Linear(hidden_size, out_size))
        self.net = nn.Sequential(*layers)

    def get_features(self, x):
        return self.net(x)

    def predict_proba(self, x_np):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
            return torch.sigmoid(self.net(x)).cpu().numpy()


def full_loss(logits, labels, logits_conf, labels_conf, alpha=ALPHA):
    """Full EXPLOR loss: in_ce + in_prob + alpha * conf_ce."""
    in_ce    = F.binary_cross_entropy_with_logits(logits, labels)
    in_prob  = torch.mean(
        torch.abs(torch.sigmoid(logits).mean(dim=1, keepdim=True) - labels.mean(dim=1, keepdim=True))
    )
    conf_ce  = F.binary_cross_entropy_with_logits(logits_conf, labels_conf)
    return in_ce + in_prob + alpha * conf_ce


def load_data(dname):
    special   = {"dataset1", "dataset2", "dataset3", "dataset4"}
    ood_start = 4 if dname in special else 2

    train_df = pd.read_csv(os.path.join(BASE_DIR, dname, "train.csv"))
    X  = train_df.iloc[:, 2:].values.astype(np.float32)
    Y  = train_df.iloc[:, 1].values.astype(np.float32)

    ood_df = pd.read_csv(os.path.join(BASE_DIR, dname, "val_ood.csv"))
    Xv = ood_df.iloc[:, ood_start:].values.astype(np.float32)
    Yv = ood_df.iloc[:, 1].values.astype(np.float32)

    return X, Y, Xv, Yv


def preprocess(X, Xv):
    mu  = np.mean(X, axis=0, keepdims=True)
    Xc  = (X  - mu).astype(np.float32)
    Xvc = (Xv - mu).astype(np.float32)

    XtX = Xc.T @ Xc
    eigvals, eigvecs = np.linalg.eig(XtX)
    V1 = np.real(eigvecs[:, np.argsort(-eigvals)]).astype(np.float32)
    n_comp = min(NDIRS_TOTAL, V1.shape[1])
    Xp  = Xc  @ V1[:, :n_comp]
    Xvp = Xvc @ V1[:, :n_comp]

    XtX2 = Xp.T @ Xp
    eigvals2, eigvecs2 = np.linalg.eig(XtX2)
    Vpca = np.real(eigvecs2[:, np.argsort(-eigvals2)]).astype(np.float32)

    return Xp, Xvp, Vpca


def permute_expand(X):
    N, d = X.shape
    return np.stack([X[np.random.permutation(N), i] for i in range(d)], axis=-1).astype(np.float32)


def convex_expand(X):
    N = len(X)
    X1     = X[np.random.permutation(N)]
    alphas = np.random.rand(N, 1).astype(np.float32)
    return alphas * X + (1 - alphas) * X1


def scaling_expand(X):
    return (X * (1 + np.abs(np.random.randn(X.shape[0], 1)))).astype(np.float32)


def build_conf_batch(X, pca_proj):
    hi = 1.0 - CONF_RANGE / 2
    lo = CONF_RANGE / 2

    conf_expX, conf_exppreds = [], []
    for _ in range(N_EXPAND_ROUNDS):
        expX  = scaling_expand(X)
        preds = pca_proj.predictions(expX)
        mp    = pca_proj.mean_conf(preds=preds)
        hc    = np.logical_or(mp > hi, mp < lo)
        if hc.any():
            conf_expX.append(expX[hc])
            conf_exppreds.append(preds[hc])

    expX2  = scaling_expand(X)
    preds2 = pca_proj.predictions(expX2)
    mp2    = pca_proj.mean_conf(preds=preds2)
    keep2  = np.abs(mp2 - 0.5) > CONF_THRESH
    if keep2.any():
        conf_expX.append(expX2[keep2])
        conf_exppreds.append(preds2[keep2])

    return (
        np.concatenate(conf_expX,    axis=0).astype(np.float32),
        np.concatenate(conf_exppreds, axis=0).astype(np.float32),
    )


class BatchSampler:
    def __init__(self, X, Y, batch_size, device):
        self.X   = torch.tensor(X, dtype=torch.float32, device=device)
        self.Y   = torch.tensor(Y, dtype=torch.float32, device=device)
        self.N   = len(X)
        self.bsz = batch_size

    def __call__(self):
        idx = torch.randint(self.N, (self.bsz,))
        return self.X[idx], self.Y[idx]


def train_discriminator(X, train_preds, conf_X, conf_preds, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_size  = train_preds.shape[1]
    input_dim = X.shape[1]

    model     = DatasetDiscriminator(input_dim, HIDDEN_SIZE, N_HIDDEN, out_size).to(DEVICE)
    avg_model = DatasetDiscriminator(input_dim, HIDDEN_SIZE, N_HIDDEN, out_size).to(DEVICE)
    avg_model.load_state_dict(model.state_dict())

    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    batch      = BatchSampler(X, train_preds, BATCH_SIZE, DEVICE)
    batch_conf = BatchSampler(conf_X, conf_preds, BATCH_SIZE, DEVICE)

    T = 0
    model.train()
    for step in range(TR_ITER):
        xb,   ylb   = batch()
        xb_c, ylb_c = batch_conf()
        logits      = model.get_features(xb)
        logits_conf = model.get_features(xb_c)
        loss_val    = full_loss(logits, ylb, logits_conf, ylb_c)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        if step > 0 and step % TR_AVG_ITER == 0:
            with torch.no_grad():
                for aw, w in zip(avg_model.parameters(), model.parameters()):
                    aw.data.mul_(T / (T + 1.0)).add_(w.data / (T + 1.0))
            T += 1

    return model, avg_model


def run_dataset(dname, n_seeds=N_SEEDS):
    print(f"\n[EXPLOR_full] {dname}")

    X, Y, Xv, Yv = load_data(dname)
    Xp, Xvp, V   = preprocess(X, Xv)

    all_metrics = []
    for seed in tqdm(range(n_seeds), desc=f"  seeds ({dname})"):
        np.random.seed(seed)
        torch.manual_seed(seed)

        pca_proj    = PCAProjectionEnsemble(V, nmodels=N_MODELS, ndirs=NDIRS, split_perc=SPLIT_PERC)
        pca_proj.fit(Xp, Y)
        train_preds = pca_proj.predictions(Xp)

        conf_X, conf_preds = build_conf_batch(Xp, pca_proj)

        _, avg_model = train_discriminator(Xp, train_preds, conf_X, conf_preds, seed)

        base_conf = pca_proj.mean_conf(Xvp)
        disc_conf = avg_model.predict_proba(Xvp).mean(axis=-1)
        emoe_conf = (base_conf + disc_conf) / 2.0

        auroc  = metrics.roc_auc_score(Yv, emoe_conf)
        auprcs = auprc_threshs(emoe_conf, Yv, THRESHS)
        all_metrics.append(np.concatenate([[auroc], auprcs]))

    return np.array(all_metrics)


METRIC_NAMES = [
    "AUROC",
    "AUPRC_recall0.1",
    "AUPRC_recall0.2",
    "AUPRC_recall0.3",
    "AUPRC_recall0.4",
    "AUPRC_recall1.0",
]

COL_MAP = {"dataset1": "d1_val", "dataset3": "d3_val", "dataset4": "d4_val"}


def fmt(mean, se):
    return f"{mean*100:.2f}±{se*100:.2f}"


def update_results_csv(new_df):
    if os.path.exists(RESULTS_PATH):
        existing = pd.read_csv(RESULTS_PATH)
        mask = existing.set_index(["model", "metric"]).index.isin(
            new_df.set_index(["model", "metric"]).index
        )
        existing = existing[~mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved → {RESULTS_PATH}")
    print(combined[combined["model"] == "EXPLOR_full"].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds",    type=int,   default=N_SEEDS)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    col_order = ["model", "metric"] + [COL_MAP[d] for d in args.datasets if d in COL_MAP]

    collected = {}
    for dname in args.datasets:
        arr = run_dataset(dname, n_seeds=args.seeds)
        collected[dname] = arr
        means = arr.mean(axis=0)
        ses   = arr.std(axis=0, ddof=1) / np.sqrt(args.seeds)
        print(f"  {dname} AUROC: {fmt(means[0], ses[0])}")
        for i, t in enumerate(THRESHS):
            print(f"  {dname} AUPRC@{t}: {fmt(means[i+1], ses[i+1])}")

        rows = []
        for mi, mname in enumerate(METRIC_NAMES):
            row = {"model": "EXPLOR_full", "metric": mname, COL_MAP[dname]: fmt(means[mi], ses[mi])}
            rows.append(row)
        new_df = pd.DataFrame(rows)[[c for c in col_order if c in pd.DataFrame(rows).columns]]
        update_results_csv(new_df)


if __name__ == "__main__":
    main()
