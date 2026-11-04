"""Generate synthetic counts for an installation check, not a benchmark."""

import argparse

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from gist.io import fresh_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/demo")
    args = parser.parse_args()
    out = fresh_directory(args.output)
    rng = np.random.default_rng(42)
    rates = rng.gamma(2.0, 2.0, (4, 30))
    genes = [f"GENE{i:02d}" for i in range(30)]
    for label, cells in [("st", 96), ("sc", 128)]:
        types = rng.integers(0, 4, cells)
        counts = rng.poisson(
            rates[types] * rng.lognormal(0.0, 0.25, (cells, 1))
        ).astype(np.float32)
        data = ad.AnnData(
            sparse.csr_matrix(counts),
            obs=pd.DataFrame(index=[f"{label}_{i}" for i in range(cells)]),
            var=pd.DataFrame(index=genes),
        )
        if label == "st":
            data.obsm["spatial"] = rng.uniform(0, 100, (cells, 2))
        data.write_h5ad(out / f"{label}.h5ad")
    print(f"Synthetic inputs: {out}")


if __name__ == "__main__":
    main()
