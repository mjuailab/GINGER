"""Command-line preparation, training, inference and evaluation."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def input_arguments(parser):
    parser.add_argument(
        "--st", required=True, help="ST AnnData with raw counts and 2D coordinates"
    )
    parser.add_argument(
        "--sc", required=True, help="SC reference AnnData with raw counts"
    )
    parser.add_argument("--st-layer", help="Raw-count layer; default: X")
    parser.add_argument("--sc-layer", help="Raw-count layer; default: X")
    parser.add_argument("--spatial-key", default="spatial")
    parser.add_argument("--x-column", default="x_centroid")
    parser.add_argument("--y-column", default="y_centroid")


def input_options(args):
    return {
        k: getattr(args, k)
        for k in ["st_layer", "sc_layer", "spatial_key", "x_column", "y_column"]
    }


def prediction_artifacts(output, predictions, genes, cells, metadata):
    import numpy as np
    from .io import digest, fresh_directory, write_json

    out = fresh_directory(output)
    np.save(out / "prediction.npy", predictions, allow_pickle=False)
    write_json(out / "genes.json", genes)
    write_json(out / "cell_ids.json", cells)
    metadata.update(
        prediction_sha256=digest(out / "prediction.npy"),
        genes_sha256=digest(out / "genes.json"),
        cell_ids_sha256=digest(out / "cell_ids.json"),
        scale="log1p(CP10K), library defined by observed training genes",
    )
    write_json(out / "prediction.json", metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ginger", description="GINGER spatial transcriptomics imputation"
    )
    from . import __version__

    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser(
        "prepare", help="Prepare a gene-holdout benchmark from raw AnnData"
    )
    input_arguments(p)
    p.add_argument("--output", required=True)
    p.add_argument("--split", help="JSON with train_genes, val_genes and test_genes")
    p.add_argument("--masks", help="Optional saved training masks")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--mask-seed", type=int, default=63410)

    p = commands.add_parser(
        "train", help="Train and select a checkpoint using validation genes"
    )
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config")
    p.add_argument("--device", default="cpu")

    p = commands.add_parser(
        "predict", help="Predict held-out genes without loading their ST truth"
    )
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--partition", choices=["validation", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cpu")

    p = commands.add_parser(
        "evaluate", help="Score a saved prediction on the matching gene partition"
    )
    p.add_argument("--data", required=True)
    p.add_argument(
        "--prediction", required=True, help="Directory written by ginger predict"
    )
    p.add_argument("--output", required=True)

    p = commands.add_parser(
        "impute", help="Impute genes absent from ST, using a trained checkpoint"
    )
    input_arguments(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--genes", required=True, help="Text file with one target gene per line"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cpu")

    p = commands.add_parser(
        "summarize", help="Aggregate fold metrics with sample standard deviations"
    )
    p.add_argument(
        "--evaluations",
        nargs="+",
        required=True,
        help="Evaluation directories, one per split",
    )
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        from .prepare import prepare

        prepare(
            args.st,
            args.sc,
            args.output,
            args.split,
            args.masks,
            args.seed,
            args.mask_seed,
            **input_options(args),
        )
        print(f"Prepared data: {args.output}")
    elif args.command == "train":
        from .train import train

        train(args.data, args.output, args.config, args.device)
    elif args.command in ("predict", "impute"):
        from .data import Runtime, predict
        from .io import digest, setup
        from .train import load_model

        model, bundle = load_model(
            args.checkpoint,
            args.device,
            prepared=args.data if args.command == "predict" else None,
        )
        setup(bundle["config"]["seed"])
        metadata = dict(
            checkpoint_sha256=digest(args.checkpoint), epoch=bundle["epoch"]
        )
        if args.command == "predict":
            rt = Runtime(args.data, args.device)
            ids = rt.val_ids if args.partition == "validation" else rt.test_ids
            metadata.update(
                partition=args.partition,
                prepared_manifest_sha256=digest(Path(args.data) / "manifest.json"),
            )
        else:
            genes = [
                line.strip()
                for line in Path(args.genes).read_text().splitlines()
                if line.strip()
            ]
            rt = Runtime.for_imputation(
                args.st,
                args.sc,
                bundle["train_genes"],
                genes,
                args.device,
                **input_options(args),
            )
            ids = list(range(rt.g, len(rt.genes)))
            metadata.update(
                st_sha256=digest(args.st),
                sc_sha256=digest(args.sc),
                partition="unmeasured",
            )
        predictions = predict(model, rt, ids, args.batch_size)
        prediction_artifacts(
            args.output, predictions, [rt.genes[i] for i in ids], rt.cell_ids, metadata
        )
        print(
            f"Saved {predictions.shape[0]} cells × {predictions.shape[1]} genes: {args.output}"
        )
    elif args.command == "evaluate":
        evaluate(args.data, args.prediction, args.output)
    else:
        summarize(args.evaluations, args.output)


def evaluate(data, prediction, output):
    import numpy as np
    from .io import digest, fresh_directory, read_json, write_json
    from .metrics import evaluate_genes

    data, prediction = Path(data), Path(prediction)
    info = read_json(prediction / "prediction.json")
    if info.get("prepared_manifest_sha256") != digest(data / "manifest.json"):
        raise ValueError("Prediction does not belong to this prepared dataset")
    partition = info["partition"]
    if partition not in ("test", "validation"):
        raise ValueError("Only held-out benchmark predictions can be evaluated")
    split = read_json(data / "split.json")
    genes = split["test_genes" if partition == "test" else "val_genes"]
    for name in ["prediction.npy", "genes.json", "cell_ids.json"]:
        key = {
            "prediction.npy": "prediction_sha256",
            "genes.json": "genes_sha256",
            "cell_ids.json": "cell_ids_sha256",
        }[name]
        if digest(prediction / name) != info[key]:
            raise ValueError(f"Prediction artifact changed: {name}")
    if read_json(prediction / "genes.json") != genes:
        raise ValueError("Prediction gene order does not match the split")
    if (
        read_json(prediction / "cell_ids.json")
        != np.load(data / "st_cell_ids.npy").tolist()
    ):
        raise ValueError("Prediction cell order does not match the data")
    manifest = read_json(data / "manifest.json")
    for name in [
        f"{partition}_counts.npy",
        "normalization_factors.npy",
        "st_cell_ids.npy",
        "split.json",
    ]:
        if digest(data / name) != manifest["files"][name]:
            raise ValueError(f"Prepared artifact changed: {name}")
    counts = np.load(data / f"{partition}_counts.npy").astype(np.float64)
    factors = np.load(data / "normalization_factors.npy")
    truth = np.log1p(counts * factors[:, None]).astype(np.float32)
    frame, metrics = evaluate_genes(
        truth, np.load(prediction / "prediction.npy"), genes
    )
    out = fresh_directory(output)
    frame.to_csv(out / "per_gene.csv", index=False)
    write_json(
        out / "metrics.json",
        dict(
            metrics,
            partition=partition,
            prepared_manifest_sha256=digest(data / "manifest.json"),
            split_sha256=digest(data / "split.json"),
            cell_ids_sha256=digest(data / "st_cell_ids.npy"),
            prediction_sha256=info["prediction_sha256"],
            scale=info["scale"],
        ),
    )
    print({k: metrics[k] for k in ["rmse", "mae", "pcc", "undefined_pcc_genes"]})


def summarize(evaluations, output):
    import numpy as np
    import pandas as pd
    from .io import fresh_directory, read_json, write_json

    if len(evaluations) < 2:
        raise ValueError(
            "At least two splits are needed for a sample standard deviation"
        )
    records = [read_json(Path(p) / "metrics.json") for p in evaluations]
    if len({r["split_sha256"] for r in records}) != len(records):
        raise ValueError("Each evaluation must represent a different split")
    for key in ["partition", "scale", "cells", "cell_ids_sha256"]:
        if len({r[key] for r in records}) != 1:
            raise ValueError(f"Cannot aggregate incompatible {key}")
    out = fresh_directory(output)
    summary = dict(
        splits=len(records),
        partition=records[0]["partition"],
        cells=records[0]["cells"],
        aggregation="Equal gene means, then equal split means; sample SD (ddof=1)",
        scale=records[0]["scale"],
    )
    for metric in ["rmse", "mae", "pcc"]:
        values = [r[metric] for r in records]
        summary[metric] = (
            dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)))
            if all(v is not None for v in values)
            else dict(mean=None, std=None)
        )
    summary["undefined_pcc_genes_by_split"] = [
        r["undefined_pcc_genes"] for r in records
    ]
    pd.DataFrame(
        [
            dict(
                split=i,
                **{k: r[k] for k in ["rmse", "mae", "pcc", "genes", "valid_pcc_genes"]},
            )
            for i, r in enumerate(records)
        ]
    ).to_csv(out / "per_split.csv", index=False)
    write_json(out / "summary.json", summary)
    print(summary)
