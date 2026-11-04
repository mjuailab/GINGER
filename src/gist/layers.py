"""Encoder and decoder layers adapted from stAI; see THIRD_PARTY_NOTICES.md."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv


class MLPEncoder(nn.Module):

    def __init__(self, d_input, d_hidden, d_latent, dropout=0.0):
        super(MLPEncoder, self).__init__()
        self.d_input = d_input
        self.d_latent = d_latent
        self.d_hidden = d_hidden
        self.model = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0, std=0.1)
                torch.nn.init.normal_(module.bias, mean=0, std=0.1)

    def forward(self, x):
        z = self.model(x)
        return z


class MLPDecoder(nn.Module):

    def __init__(self, d_input, d_hidden, d_latent):
        super(MLPDecoder, self).__init__()
        self.d_input = d_input
        self.d_latent = d_latent
        self.d_hidden = d_hidden
        self.model = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.ELU(), nn.Linear(d_hidden, d_input)
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0, std=0.1)
                torch.nn.init.normal_(module.bias, mean=0, std=0.1)

    def forward(self, z):
        x = self.model(z)
        return x


class GATLayer(nn.Module):

    def __init__(self, d_input, num_heads=4, n_layers=1, dropout=0.0, residual=True):
        super(GATLayer, self).__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.conv_layers = nn.ModuleList()
        self.n_layers = n_layers
        for i in range(self.n_layers):
            self.conv_layers.append(
                GATConv(d_input, d_input, heads=num_heads, concat=False)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        if self.residual:
            resid = x
        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = self.dropout(F.leaky_relu(x))
        if self.residual:
            x += resid
        return x


class GATEncoder(nn.Module):

    def __init__(
        self,
        d_input,
        d_hidden,
        d_latent,
        num_heads=1,
        n_layers=1,
        dropout=0.0,
        residual=True,
    ):
        super().__init__()
        self.pre_fc = nn.Sequential(
            nn.Linear(d_input, d_hidden, bias=True),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent, bias=True),
        )
        self.post_fc = nn.Linear(d_latent, d_latent, bias=True)
        self.gat_layer = GATLayer(
            d_input=d_latent,
            num_heads=num_heads,
            n_layers=n_layers,
            dropout=dropout,
            residual=residual,
        )

    def forward(self, x, e):
        z = self.pre_fc(x)
        z = self.gat_layer(z, e)
        z = self.post_fc(z)
        return z


def configure_sc_depth(model, hidden_layers, second_hidden=None):
    assert hidden_layers in (1, 2)
    model.sc_hidden_layers = hidden_layers
    if hidden_layers == 1:
        return
    layers = list(model.SC_encoder.model)
    assert (
        len(layers) == 4
        and isinstance(layers[0], nn.Linear)
        and isinstance(layers[3], nn.Linear)
    )
    hidden_width = layers[0].out_features
    assert layers[3].in_features == hidden_width
    second_hidden = hidden_width if second_hidden is None else second_hidden
    assert second_hidden in (64, 128)
    model.sc_second_hidden = second_hidden
    extra = nn.Linear(hidden_width, second_hidden)
    torch.nn.init.normal_(extra.weight, mean=0, std=0.1)
    torch.nn.init.normal_(extra.bias, mean=0, std=0.1)
    output = layers[3]
    if second_hidden != hidden_width:
        output = nn.Linear(second_hidden, layers[3].out_features)
        torch.nn.init.normal_(output.weight, mean=0, std=0.1)
        torch.nn.init.normal_(output.bias, mean=0, std=0.1)
    model.SC_encoder.model = nn.Sequential(
        layers[0],
        layers[1],
        layers[2],
        extra,
        nn.ELU(),
        nn.Dropout(layers[2].p),
        output,
    )
