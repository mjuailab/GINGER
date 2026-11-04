"""Prepare raw AnnData inputs without exposing held-out ST values to training."""

from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from .graph import build_graph
from .io import digest, fresh_directory, read_json, write_json


def validate_split(split):
    keys = ("train_genes", "val_genes", "test_genes")
    if any(not isinstance(split.get(k), list) for k in keys):
        raise ValueError(f"Split must contain lists: {keys}")
    genes = sum([split[k] for k in keys], [])
    if (
        len(split["train_genes"]) < 4
        or not split["val_genes"]
        or not split["test_genes"]
        or len(genes) != len(set(genes))
        or any(not isinstance(g, str) or not g for g in genes)
    ):
        raise ValueError(
            "Use at least four train genes and nonempty, disjoint validation/test genes"
        )
    return genes


def make_split(genes, seed=1234):
    genes = list(genes)
    if len(genes) < 10 or len(genes) != len(set(genes)):
        raise ValueError(
            "At least ten unique shared genes are needed for an 8:1:1 split"
        )
    order = np.random.default_rng(seed).permutation(genes).tolist()
    train, val = 8 * len(order) // 10, len(order) // 10
    return dict(
        train_genes=order[:train],
        val_genes=order[train : train + val],
        test_genes=order[train + val :],
    )


def make_masks(train_genes, seed=63410):
    n = len(train_genes)
    queries = np.array_split(np.tile(np.random.default_rng(seed).permutation(n), 3), 10)
    contexts = [
        dict(query=q.tolist(), observed=np.setdiff1d(np.arange(n), q).tolist())
        for q in queries
    ]
    masks = dict(seed=seed, train_genes=train_genes, contexts=contexts)
    validate_masks(masks, train_genes)
    return masks


def validate_masks(masks, train_genes):
    if masks.get("train_genes") != train_genes or not masks.get("contexts"):
        raise ValueError("Masks must use the exact training-gene order")
    n = len(train_genes)
    occurrences = np.zeros(n, int)
    for context in masks["contexts"]:
        q, o = context["query"], context["observed"]
        if (
            not q
            or not o
            or any(type(i) is not int for i in q + o)
            or sorted(q + o) != list(range(n))
        ):
            raise ValueError("Every mask must partition all training-gene indices")
        occurrences[q] += 1
    if np.any(occurrences == 0):
        raise ValueError("Every training gene must occur in a query mask")


def check_counts(matrix, label):
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{label}: raw counts must be finite and nonnegative")
    if not np.allclose(values, np.rint(values), atol=1e-5, rtol=0):
        raise ValueError(
            f"{label}: expected raw integer counts, not normalized expression"
        )


def read_counts(path, genes=None, layer=None, totals=False):
    data = ad.read_h5ad(path, backed="r")
    try:
        if not data.var_names.is_unique or not data.obs_names.is_unique:
            raise ValueError(
                f"{Path(path).name}: gene and cell identifiers must be unique"
            )
        names = data.var_names.tolist()
        genes = names if genes is None else genes
        lookup = {g: i for i, g in enumerate(names)}
        missing = [g for g in genes if g not in lookup]
        if missing:
            raise ValueError(f"{Path(path).name}: missing genes {missing[:10]}")
        columns = np.array([lookup[g] for g in genes])
        matrix = data.X if layer is None else data.layers[layer]
        counts = np.empty((data.n_obs, len(genes)), np.float32)
        libraries = np.empty(data.n_obs, np.float64) if totals else None
        for start in range(0, data.n_obs, 4096):
            block = matrix[start : start + 4096]
            check_counts(block, Path(path).name)
            if totals:
                libraries[start : start + block.shape[0]] = np.asarray(
                    block.sum(axis=1, dtype=np.float64)
                ).ravel()
            selected = block[:, columns]
            selected = (
                selected.toarray()
                if sparse.issparse(selected)
                else np.asarray(selected)
            )
            counts[start : start + len(selected)] = selected
        if len(counts) < 2:
            raise ValueError("At least two cells are required")
        return counts, data.obs_names.tolist(), libraries
    finally:
        data.file.close()


def correlations(counts, full_library, n_train):
    normalized = np.log1p(
        counts.astype(np.float64) * (10000.0 / np.maximum(full_library, 1))[:, None]
    )
    means = normalized.mean(0)
    variance = np.square(normalized).mean(0) - means**2
    inv = np.zeros_like(variance)
    valid = variance > 1e-12
    inv[valid] = 1.0 / np.sqrt(variance[valid])
    scaled = normalized * inv
    mean_scaled = means * inv
    return np.clip(
        scaled.T @ scaled[:, :n_train] / len(scaled)
        - mean_scaled[:, None] * mean_scaled[None, :n_train],
        -1,
        1,
    ).astype(np.float32)


def prepare(
    st_path,
    sc_path,
    output,
    split_path=None,
    masks_path=None,
    seed=1234,
    mask_seed=63410,
    st_layer=None,
    sc_layer=None,
    spatial_key="spatial",
    x_column="x_centroid",
    y_column="y_centroid",
):
    st, sc = ad.read_h5ad(st_path, backed="r"), ad.read_h5ad(sc_path, backed="r")
    try:
        if split_path:
            split = read_json(split_path)
        else:
            sc_names = set(sc.var_names)
            split = make_split([g for g in st.var_names if g in sc_names], seed)
        genes = validate_split(split)
        for label, data in [("ST", st), ("SC", sc)]:
            missing = set(genes) - set(data.var_names)
            if missing:
                raise ValueError(
                    f"{label} is missing split genes: {sorted(missing)[:10]}"
                )
        if spatial_key in st.obsm:
            coordinates = np.asarray(st.obsm[spatial_key])
        elif x_column in st.obs and y_column in st.obs:
            coordinates = st.obs[[x_column, y_column]].to_numpy()
        else:
            raise ValueError(
                f"Provide obsm[{spatial_key!r}] or obs columns {x_column!r}, {y_column!r}"
            )
    finally:
        st.file.close()
        sc.file.close()
    masks = (
        read_json(masks_path)
        if masks_path
        else make_masks(split["train_genes"], mask_seed)
    )
    validate_masks(masks, split["train_genes"])
    graph = build_graph(coordinates)
    st_counts, st_ids, _ = read_counts(st_path, genes, st_layer)
    sc_counts, sc_ids, sc_library = read_counts(sc_path, genes, sc_layer, totals=True)
    n_train, n_val = len(split["train_genes"]), len(split["val_genes"])
    prior = correlations(sc_counts, sc_library, n_train)
    out = fresh_directory(output)
    write_json(
        out / "split.json",
        {k: split[k] for k in ["train_genes", "val_genes", "test_genes"]},
    )
    write_json(out / "masks.json", masks)
    arrays = dict(
        st_train_counts=st_counts[:, :n_train],
        validation_counts=st_counts[:, n_train : n_train + n_val],
        test_counts=st_counts[:, n_train + n_val :],
        sc_counts=sc_counts,
        correlations=prior,
        edge_index=graph,
        normalization_factors=10000.0
        / np.maximum(st_counts[:, :n_train].sum(1, dtype=np.float64), 1),
        st_cell_ids=np.asarray(st_ids),
        sc_cell_ids=np.asarray(sc_ids),
    )
    for name, value in arrays.items():
        np.save(out / f"{name}.npy", value, allow_pickle=False)
    write_json(
        out / "manifest.json",
        dict(
            format_version=1,
            st_cells=len(st_ids),
            sc_cells=len(sc_ids),
            genes=len(genes),
            train_genes=n_train,
            validation_genes=n_val,
            test_genes=len(split["test_genes"]),
            inputs={
                "st": {
                    "name": Path(st_path).name,
                    "sha256": digest(st_path),
                    "layer": st_layer,
                },
                "sc": {
                    "name": Path(sc_path).name,
                    "sha256": digest(sc_path),
                    "layer": sc_layer,
                },
            },
            graph="Bidirectional Delaunay (QJ), self loops; coordinates only",
            normalization="CP10K/log1p; visible training genes define each library size",
            relation_normalization="CP10K/log1p; all genes in the supplied SC count matrix define the library size",
            files={p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()},
        ),
    )
    return out
