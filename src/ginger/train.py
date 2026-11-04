"""Validation-selected training with held-out test evaluation as a separate step."""

import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .data import Runtime, predict
from .io import digest, fresh_directory, read_json, save_checkpoint, setup, write_json
from .metrics import evaluate_genes
from .model import GINGER

DEFAULTS = dict(
    seed=3407,
    hidden=128,
    latent=32,
    topk=100,
    temperature=2.0,
    strength=0.1,
    batch_cells=512,
    sc_aux_batch=512,
    prediction_batch=512,
    lr=1e-4,
    max_epochs=2000,
    eval_every=10,
    patience=10,
    min_delta=0.002,
    reference_rmse=None,
    loss_weights=dict(
        recon_st=0.1,
        recon_sc=0.1,
        mmd=1.0,
        consistency=1.0,
        imputation=1.0,
        imputation_mse=1.0,
    ),
)


def configuration(path=None):
    values = read_json(path) if path else {}
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    cfg = dict(DEFAULTS, **values)
    for key in [
        "hidden",
        "latent",
        "topk",
        "batch_cells",
        "sc_aux_batch",
        "prediction_batch",
        "max_epochs",
        "eval_every",
        "patience",
    ]:
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ["lr", "temperature"]:
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ["strength", "min_delta"]:
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if cfg["reference_rmse"] is not None and (
        not math.isfinite(cfg["reference_rmse"]) or cfg["reference_rmse"] <= 0
    ):
        raise ValueError("reference_rmse must be positive or null")
    if set(cfg["loss_weights"]) != set(DEFAULTS["loss_weights"]):
        raise ValueError("Specify all six loss weights")
    if any(not math.isfinite(w) or w < 0 for w in cfg["loss_weights"].values()):
        raise ValueError("Loss weights must be finite and nonnegative")
    return cfg


def make_model(genes, cfg, device):
    return GINGER(
        genes,
        **{k: cfg[k] for k in ["hidden", "latent", "topk", "temperature", "strength"]},
    ).to(device)


def update_monitor(state, value, epoch, cfg):
    improved = state.get("best") is None or value > state["best"]
    if improved:
        state.update(best=value, best_epoch=epoch)
    if state.get("anchor") is None or value > state["anchor"] + cfg["min_delta"]:
        state.update(anchor=value, bad=0)
    else:
        state["bad"] += 1
    state["stop"] = state["bad"] >= cfg["patience"]
    return improved


def train(data, output, config=None, device="cpu"):
    cfg = configuration(config)
    setup(cfg["seed"])
    rt = Runtime(data, device)
    if cfg["topk"] > rt.n_sc:
        raise ValueError(
            f"topk={cfg['topk']} exceeds the {rt.n_sc} reference cells; set topk explicitly"
        )
    out = fresh_directory(output)
    setup(cfg["seed"])
    model = make_model(rt.g, cfg, rt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    shuffle = torch.Generator().manual_seed(cfg["seed"] + 100)
    sc_shuffle = torch.Generator().manual_seed(cfg["seed"] + 200)
    validation = rt.truth("validation")
    rt.activate(-1)
    reference = cfg["reference_rmse"]
    if reference is None:
        mean = rt.profile[:, rt.val_ids].mean(0).cpu().numpy()
        _, baseline = evaluate_genes(
            validation, np.broadcast_to(mean, validation.shape), rt.split["val_genes"]
        )
        reference = baseline["rmse"]
    if reference <= 0:
        raise ValueError(
            "Validation reference RMSE is zero; supply a positive fixed reference scale"
        )
    versions = {
        p: importlib.metadata.version(p)
        for p in [
            "torch",
            "torch-geometric",
            "numpy",
            "scipy",
            "pandas",
            "anndata",
            "h5py",
        ]
    }
    write_json(out / "config.json", cfg)
    write_json(out / "split.json", rt.split)
    write_json(
        out / "run.json",
        dict(
            versions=versions,
            device=str(rt.device),
            reference_rmse=reference,
            reference_policy=(
                "User-supplied fixed scale"
                if cfg["reference_rmse"] is not None
                else "SC mean validation RMSE, fixed before training"
            ),
            prepared_manifest_sha256=digest(Path(data) / "manifest.json"),
            source_sha256={
                p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
        ),
    )
    state = {}
    for epoch in range(1, cfg["max_epochs"] + 1):
        started = time.monotonic()
        model.train()
        totals = {k: 0.0 for k in cfg["loss_weights"]}
        batches = list(
            torch.randperm(rt.n, generator=shuffle).split(cfg["batch_cells"])
        )
        order = torch.randperm(len(rt.contexts), generator=shuffle).tolist()
        for step, cells in enumerate(batches):
            cells = cells.to(rt.device)
            context = order[step % len(order)]
            aux = torch.randperm(rt.n_sc, generator=sc_shuffle)[
                : cfg["sc_aux_batch"]
            ].to(rt.device)
            optimizer.zero_grad(set_to_none=True)
            terms, _ = rt.batch(model, cells, context, aux)
            loss = sum(cfg["loss_weights"][k] * v for k, v in terms.items())
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite loss at epoch {epoch}, batch {step}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            for key, value in terms.items():
                totals[key] += float(value.detach()) * len(cells) / rt.n
        metrics = None
        if epoch % cfg["eval_every"] == 0 or epoch == cfg["max_epochs"]:
            prediction = predict(model, rt, rt.val_ids, cfg["prediction_batch"])
            frame, metrics = evaluate_genes(
                validation, prediction, rt.split["val_genes"]
            )
            if metrics["pcc"] is None:
                raise ValueError(
                    "All validation PCC values are undefined; inspect reference and inputs"
                )
            value = 0.5 * metrics["pcc"] - 0.5 * metrics["rmse"] / reference
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite validation selection score")
            metrics["joint_score"] = value
            if update_monitor(state, value, epoch, cfg):
                save_checkpoint(
                    out / "best.pt",
                    dict(
                        format_version=1,
                        model=model.state_dict(),
                        config=cfg,
                        epoch=epoch,
                        train_genes=rt.split["train_genes"],
                        prepared_manifest_sha256=digest(Path(data) / "manifest.json"),
                    ),
                )
                frame.to_csv(out / "validation_per_gene.csv", index=False)
                write_json(
                    out / "selection.json",
                    dict(
                        epoch=epoch,
                        validation=metrics,
                        reference_rmse=reference,
                        checkpoint_sha256=digest(out / "best.pt"),
                    ),
                )
        record = dict(
            epoch=epoch,
            seconds=time.monotonic() - started,
            losses=totals,
            total_loss=sum(cfg["loss_weights"][k] * v for k, v in totals.items()),
            validation=metrics,
            monitor=state.copy(),
        )
        with (out / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(
            json.dumps(
                {"epoch": epoch, "loss": record["total_loss"], "validation": metrics}
            ),
            flush=True,
        )
        if state.get("stop"):
            break
    write_json(
        out / "status.json",
        dict(
            complete=True,
            epochs=epoch,
            best_epoch=state["best_epoch"],
            test_truth_used=False,
        ),
    )
    return out


def load_model(checkpoint, device="cpu", prepared=None):
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if bundle.get("format_version") != 1:
        raise ValueError("Expected a GINGER release checkpoint (format_version=1)")
    if prepared is not None:
        if (
            digest(Path(prepared) / "manifest.json")
            != bundle["prepared_manifest_sha256"]
        ):
            raise ValueError("Checkpoint and prepared data do not match")
    model = make_model(len(bundle["train_genes"]), bundle["config"], device)
    model.load_state_dict(bundle["model"], strict=True)
    return model, bundle
