"""Gene-aware spatial reference transfer with the selected pre-DCA architecture."""

import torch
from torch import nn
from torch.nn import functional as F
from .layers import GATEncoder, MLPEncoder, MLPDecoder, configure_sc_depth
from .losses import mmd_loss, imputation_loss, imputation_mse_loss


def retrieve(query, reference, values, k=100, temperature=2.0):
    if not 1 <= k <= len(reference):
        raise ValueError("top-k must be between 1 and the number of SC cells")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    neg_dist_matrix = -torch.cdist(query, reference, p=2)
    (val, indices) = torch.topk(neg_dist_matrix, k=k, dim=1)
    weights = (val / temperature).softmax(dim=1)
    expression = values[indices]
    soft = (weights[:, :, None] * expression).sum(1)
    hard = values[indices[:, 0]]
    return dict(soft=soft, hard=hard, indices=indices, weights=weights)


def relation_weights(correlations, valid):
    assert valid.any(), "No variable observed SC genes"
    uniform = valid.float() / valid.sum()
    weights = correlations.square() * valid[None]
    denominator = weights.sum(1, keepdim=True)
    return torch.where(
        denominator > 1e-12, weights / denominator.clamp_min(1e-12), uniform[None]
    )


@torch.no_grad()
def observed_distance(st_observed, sc_observed, indices, inverse_variance, relation):
    discrepancy = (st_observed[:, None] - sc_observed[indices]).square()
    discrepancy.mul_(inverse_variance[None, None])
    return discrepancy @ relation.T


class GIST(nn.Module):

    def __init__(
        self, genes, hidden=128, latent=32, topk=100, temperature=2.0, strength=0.1
    ):
        super().__init__()
        if (
            genes < 2
            or hidden < 1
            or latent < 1
            or topk < 1
            or temperature <= 0
            or strength < 0
        ):
            raise ValueError("Invalid model dimensions or transfer settings")
        self.ST_encoder = GATEncoder(
            genes, hidden, latent, num_heads=1, n_layers=1, dropout=0.0, residual=True
        )
        self.ST_decoder = MLPDecoder(genes, hidden, latent)
        self.SC_encoder = MLPEncoder(genes, hidden, latent, dropout=0.0)
        self.SC_decoder = MLPDecoder(genes, hidden, latent)
        self.topk, self.temperature, self.strength = topk, temperature, strength
        configure_sc_depth(self, 2, 128)

    def encode(self, st, edge, roots, sc):
        return self.ST_encoder(st, edge)[roots], self.SC_encoder(sc)

    def transfer_cells(
        self,
        st_latent,
        sc_latent,
        values,
        st_observed=None,
        sc_observed=None,
        relation=None,
        inverse_variance=None,
    ):
        output = retrieve(st_latent, sc_latent, values, self.topk, self.temperature)
        base = output["weights"]
        output["base_weights"] = base
        if self.strength == 0:
            output["weights"] = base[:, :, None].expand(-1, -1, values.shape[1])
            return output
        distance = observed_distance(
            st_observed, sc_observed, output["indices"], inverse_variance, relation
        )
        weights = (
            base.clamp_min(1e-30).log()[:, :, None] - self.strength * distance
        ).softmax(1)
        output["weights"] = weights
        output["soft"] = (weights * values[output["indices"]]).sum(1)
        return output

    def losses(
        self,
        st,
        edge,
        roots,
        sc,
        values,
        target,
        observed,
        gene_weights,
        sc_aux,
        relation,
        inverse_variance,
        st_full_target=None,
        sc_full_target=None,
    ):
        (zst, zsc) = self.encode(st, edge, roots, sc)
        zaux = zsc[sc_aux]
        (st_fit, sc_fit) = (st[roots][:, observed], sc[sc_aux][:, observed])
        if st_full_target is None or sc_full_target is None:
            raise ValueError("Full training-gene reconstruction targets are required")
        if st_full_target.shape != (
            len(roots),
            st.shape[1],
        ) or sc_full_target.shape != (len(sc_aux), sc.shape[1]):
            raise ValueError(
                "Reconstruction targets must match the training-gene input shape"
            )
        recon_st = F.mse_loss(self.ST_decoder(zst), st_full_target)
        recon_sc = F.mse_loss(self.SC_decoder(zaux), sc_full_target)
        consistency = (
            1 - F.cosine_similarity(self.SC_decoder(zst)[:, observed], st_fit)
        ).mean()
        consistency = (
            consistency
            + (
                1 - F.cosine_similarity(self.ST_decoder(zaux)[:, observed], sc_fit)
            ).mean()
        )
        output = self.transfer_cells(
            zst, zsc, values, st_fit, sc[:, observed], relation, inverse_variance
        )
        prediction = output["soft"]
        terms = dict(
            recon_st=recon_st,
            recon_sc=recon_sc,
            mmd=mmd_loss(zst, zaux),
            imputation=imputation_loss(prediction, target, gene_weights),
            imputation_mse=imputation_mse_loss(prediction, target, gene_weights),
            consistency=consistency,
        )
        return (terms, output)
