"""Equal-weight gene metrics; undefined correlations are reported explicitly."""

import numpy as np
import pandas as pd


def evaluate_genes(truth, prediction, genes):
    truth, prediction = np.asarray(truth, np.float64), np.asarray(
        prediction, np.float64
    )
    if (
        truth.ndim != 2
        or truth.shape != prediction.shape
        or truth.shape[1] != len(genes)
        or min(truth.shape) == 0
    ):
        raise ValueError("Expected aligned, nonempty cells-by-genes matrices")
    if len(set(genes)) != len(genes):
        raise ValueError("Evaluation gene names must be unique")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Predictions and targets must be finite")
    rows = []
    for j, gene in enumerate(genes):
        y, p = truth[:, j], prediction[:, j]
        sy, sp = float(y.std()), float(p.std())
        rows.append(
            dict(
                gene=gene,
                rmse=float(np.sqrt(np.mean((p - y) ** 2))),
                mae=float(np.mean(np.abs(p - y))),
                pcc=float(np.corrcoef(y, p)[0, 1]) if min(sy, sp) > 1e-8 else np.nan,
                std_truth=sy,
                std_prediction=sp,
            )
        )
    frame = pd.DataFrame(rows)
    valid = frame.pcc.notna()
    summary = dict(
        rmse=float(frame.rmse.mean()),
        mae=float(frame.mae.mean()),
        pcc=float(frame.loc[valid, "pcc"].mean()) if valid.any() else None,
        cells=len(truth),
        genes=len(genes),
        valid_pcc_genes=int(valid.sum()),
        undefined_pcc_genes=frame.loc[~valid, "gene"].tolist(),
    )
    return frame, summary
