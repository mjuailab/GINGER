"""Mask-specific inputs; only one normalized reference context stays on device."""

from pathlib import Path

import numpy as np
import torch

from .graph import build_graph, incoming_index, incoming_neighborhood
from .io import digest, read_json
from .model import relation_weights


def features(raw, observed):
    visible = torch.zeros_like(raw, dtype=torch.float64)
    visible[:, observed] = raw[:, observed].double()
    factors = 10000.0 / visible.sum(-1).clamp_min(1.0)
    return torch.log1p(visible * factors[:, None]).float(), factors


class Runtime:
    def __init__(self, folder, device="cpu"):
        self.folder, self.device = Path(folder), torch.device(device)
        self.manifest = read_json(self.folder / "manifest.json")
        for name, expected in self.manifest["files"].items():
            if name == "test_counts.npy":
                continue
            if digest(self.folder / name) != expected:
                raise ValueError(
                    f"Prepared data changed: {name}; prepare into a new directory"
                )
        self.split = read_json(self.folder / "split.json")
        self.genes = sum(
            [self.split[k] for k in ["train_genes", "val_genes", "test_genes"]], []
        )
        self.cell_ids = np.load(self.folder / "st_cell_ids.npy").tolist()
        self.contexts = read_json(self.folder / "masks.json")["contexts"]
        self.raw = self.tensor("st_train_counts")
        self.sc_raw = self.tensor("sc_counts")
        self.correlations = self.tensor("correlations")
        self.initialize(self.tensor("edge_index"))
        self.val_ids = list(range(self.g, self.g + len(self.split["val_genes"])))
        self.test_ids = list(range(self.g + len(self.val_ids), len(self.genes)))

    def tensor(self, name):
        return torch.from_numpy(np.load(self.folder / f"{name}.npy")).to(self.device)

    def initialize(self, graph):
        self.n, self.g = self.raw.shape
        self.n_sc = len(self.sc_raw)
        self.incoming = incoming_index(graph, self.n)
        self.observed = [c["observed"] for c in self.contexts] + [list(range(self.g))]
        occurrences = np.bincount(
            [j for c in self.contexts for j in c["query"]], minlength=self.g
        )
        self.gene_weights = torch.tensor(
            1.0 / np.maximum(occurrences, 1), dtype=torch.float32, device=self.device
        )
        self.active_context = None
        self.profile = self.sc = self.relation = self.inverse_variance = None

    @classmethod
    def for_imputation(
        cls,
        st_path,
        sc_path,
        train_genes,
        target_genes,
        device="cpu",
        st_layer=None,
        sc_layer=None,
        spatial_key="spatial",
        x_column="x_centroid",
        y_column="y_centroid",
    ):
        import anndata as ad
        from .prepare import correlations, read_counts

        if not target_genes or len(set(target_genes)) != len(target_genes):
            raise ValueError("Target genes must be a nonempty unique list")
        if set(target_genes) & set(train_genes):
            raise ValueError(
                "Imputation targets must not include observed training genes"
            )
        rt = cls.__new__(cls)
        rt.device = torch.device(device)
        rt.genes = train_genes + target_genes
        raw, rt.cell_ids, _ = read_counts(st_path, train_genes, st_layer)
        reference, _, libraries = read_counts(sc_path, rt.genes, sc_layer, totals=True)
        st = ad.read_h5ad(st_path, backed="r")
        try:
            if spatial_key in st.obsm:
                coordinates = np.asarray(st.obsm[spatial_key])
            else:
                coordinates = st.obs[[x_column, y_column]].to_numpy()
        finally:
            st.file.close()
        rt.raw = torch.from_numpy(raw).to(rt.device)
        rt.sc_raw = torch.from_numpy(reference).to(rt.device)
        rt.correlations = torch.from_numpy(
            correlations(reference, libraries, len(train_genes))
        ).to(rt.device)
        rt.contexts = []
        rt.initialize(torch.from_numpy(build_graph(coordinates)).to(rt.device))
        return rt

    @torch.no_grad()
    def activate(self, context):
        if self.active_context == context:
            return
        self.profile = self.sc = self.relation = self.inverse_variance = None
        observed = self.observed[context]
        profile = torch.empty_like(self.sc_raw)
        for start in range(0, self.n_sc, 4096):
            raw = self.sc_raw[start : start + 4096].double()
            factors = 10000.0 / raw[:, observed].sum(1).clamp_min(1.0)
            profile[start : start + len(raw)] = torch.log1p(
                raw * factors[:, None]
            ).float()
        masked = torch.zeros_like(profile[:, : self.g])
        masked[:, observed] = profile[:, observed]
        variance = masked[:, observed].double().var(0, correction=0).float()
        valid = variance > 1e-8
        if not valid.any():
            raise ValueError(
                "No variable observed SC genes; check counts, reference and masks"
            )
        self.inverse_variance = torch.where(
            valid, variance.clamp_min(0.01).reciprocal(), 0.0
        )
        self.relation = relation_weights(self.correlations[:, observed], valid)
        self.profile, self.sc, self.active_context = profile, masked, context

    def inputs(self, cells, context):
        self.activate(context)
        nodes, edge, roots = incoming_neighborhood(self.incoming, cells)
        st, factors = features(self.raw[nodes], self.observed[context])
        return st, edge, roots, self.sc, factors[roots]

    def batch(self, model, cells, context, aux):
        st, edge, roots, sc, factors = self.inputs(cells, context)
        query = self.contexts[context]["query"]
        target = torch.log1p(
            self.raw[cells][:, query].double() * factors[:, None]
        ).float()
        full = torch.log1p(self.raw[cells].double() * factors[:, None]).float()
        return model.losses(
            st,
            edge,
            roots,
            sc,
            self.profile[:, query],
            target,
            self.observed[context],
            self.gene_weights[query],
            aux,
            self.relation[query],
            self.inverse_variance,
            st_full_target=full,
            sc_full_target=self.profile[aux, : self.g],
        )

    def truth(self, partition):
        name = (
            "validation_counts.npy" if partition == "validation" else "test_counts.npy"
        )
        if digest(self.folder / name) != self.manifest["files"][name]:
            raise ValueError(f"Ground truth changed: {name}")
        counts = np.load(self.folder / name).astype(np.float64)
        factors = 10000.0 / self.raw.double().sum(-1).clamp_min(1.0).cpu().numpy()
        return np.log1p(counts * factors[:, None]).astype(np.float32)


@torch.no_grad()
def predict(model, rt, ids, batch=512):
    if batch < 1 or not ids:
        raise ValueError("Prediction batch size and target count must be positive")
    model.eval()
    rt.activate(-1)
    output = np.empty((rt.n, len(ids)), np.float32)
    zsc, values = model.SC_encoder(rt.sc), rt.profile[:, ids]
    for start in range(0, rt.n, batch):
        cells = torch.arange(start, min(start + batch, rt.n), device=rt.device)
        st, edge, roots, sc, _ = rt.inputs(cells, -1)
        result = model.transfer_cells(
            model.ST_encoder(st, edge)[roots],
            zsc,
            values,
            st[roots],
            sc,
            rt.relation[ids],
            rt.inverse_variance,
        )
        if not torch.isfinite(result["soft"]).all():
            raise FloatingPointError("Nonfinite predictions")
        output[start : start + len(cells)] = result["soft"].cpu().numpy()
    return output
