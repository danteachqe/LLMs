#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

def main():
    # ─── Sample data ──────────────────────────────────────────────────────────
    # Replace this with loading your real dataset, e.g. pd.read_csv(...)
    data = [
        {"support_scores": [0.9, 0.8, 0.1], "chunk_labels": [1, 1, 0]},
        {"support_scores": [0.6, 0.4, 0.2], "chunk_labels": [1, 0, 0]},
        {"support_scores": [0.3, 0.7, 0.5], "chunk_labels": [0, 1, 1]},
        # … add more rows as needed …
    ]
    df = pd.DataFrame(data)

    # ─── Compute per-row metrics ─────────────────────────────────────────────
    aucs, aps, rhos = [], [], []
    for idx, row in df.iterrows():
        scores = row["support_scores"]
        labels = row["chunk_labels"]
        # Must have at least one positive and one negative label to compute AUC
        if len(scores) == len(labels) and len(set(labels)) > 1:
            try:
                aucs.append(roc_auc_score(labels, scores))
                aps.append(average_precision_score(labels, scores))
                rho = spearmanr(labels, scores).correlation
                rhos.append(rho)
            except Exception as e:
                print(f"Row {idx} skipped due to error: {e}")

    # ─── Report aggregated results ────────────────────────────────────────────
    if aucs:
        print(f"Mean ROC-AUC:           {np.mean(aucs):.3f}")
    if aps:
        print(f"Mean Average Precision: {np.mean(aps):.3f}")
    if rhos:
        print(f"Mean Spearman’s ρ:      {np.mean(rhos):.3f}")

if __name__ == "__main__":
    main()
