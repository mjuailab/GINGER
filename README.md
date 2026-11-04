# GIST

**Gene-aware spatial transcriptomics imputation using a single-cell RNA-seq reference.**

GIST predicts unmeasured genes in spatial transcriptomics (ST) data from a matched scRNA-seq (SC) reference. A spatial graph encoder and an SC encoder learn a shared representation. For each ST cell, GIST retrieves nearby reference cells and computes target-gene-specific transfer weights using SC gene correlations and observed expression differences.

This repository contains the selected pre-DCA model, a portable training pipeline, fixed gene splits, and a synthetic example. Raw research data and pretrained checkpoints are not bundled.

## Installation

Python 3.10 and PyTorch 2.5.1 are the tested versions. Run the commands below from the repository root after cloning or downloading it.

```bash
conda env create -f environment.yml
conda activate gist
gist --help
```

Alternatively, use an existing Python 3.10 environment:

```bash
python -m pip install torch==2.5.1
python -m pip install -r requirements.txt
```

Install a PyTorch 2.5.1 build compatible with your CUDA runtime for GPU execution. `--device cpu` works for the synthetic example; use `--device cuda:0` for research datasets. The pipeline uses PyTorch Geometric's standard GAT operations and does not require optional neighbor-sampling extensions.

## Quick start

This small synthetic example checks data preparation, training, prediction and evaluation. Its scores are not research results.

```bash
python examples/make_demo.py --output data/demo

gist prepare \
  --st data/demo/st.h5ad \
  --sc data/demo/sc.h5ad \
  --output data/demo/prepared

gist train \
  --data data/demo/prepared \
  --config configs/demo.json \
  --output outputs/demo/train \
  --device cpu

gist predict \
  --data data/demo/prepared \
  --checkpoint outputs/demo/train/best.pt \
  --output outputs/demo/prediction \
  --device cpu

gist evaluate \
  --data data/demo/prepared \
  --prediction outputs/demo/prediction \
  --output outputs/demo/evaluation
```

Output directories must be empty or absent. To repeat a run, choose a new directory. Training starts fresh; automatic resume is not implemented.

## Input data

Provide two `.h5ad` files with **raw, nonnegative integer counts** in `.X`, or select count layers with `--st-layer counts --sc-layer counts`.

| Input | Required content |
|---|---|
| ST | Cells × genes; unique `.obs_names` and `.var_names`; 2D coordinates in `.obsm["spatial"]` or `.obs[["x_centroid", "y_centroid"]]` |
| SC | Cells × genes; unique identifiers; all observed and target genes; at least `topk` cells |
| Split | JSON lists named `train_genes`, `val_genes`, `test_genes`; disjoint and in the intended order |

Use matching gene identifiers in ST and SC. Resolve duplicates and perform dataset-specific QC before preparation. No cell-type labels, images or pretrained gene embeddings are required. Keep the full SC gene matrix when it was used to compute the original relation prior: its library totals affect SC correlations.

`prepare` aligns genes by name, preserves ST cell order, builds the graph and stores training inputs separately from validation/test counts. Without `--split`, it creates an 8:1:1 random gene holdout with seed 1234. See [data and split details](docs/data.md).

## Training on your data

```bash
gist prepare \
  --st data/st_counts.h5ad \
  --sc data/sc_counts.h5ad \
  --split splits/brca/fold00.json \
  --masks splits/brca/masks00.json \
  --output data/brca/fold00

gist train \
  --data data/brca/fold00 \
  --config configs/brca_fold00.json \
  --output outputs/brca/fold00/train \
  --device cuda:0
```

The BRCA command requires the matching BRCA input data. For a new dataset, omit `--split` and `--masks` or provide your own files, and use `configs/default.json`. Do not use BRCA's fixed validation reference scales for a different dataset.

Default training uses Adam (`1e-4`), ST/SC auxiliary batches of 512, at most 2,000 epochs, validation every 10 epochs, and early stopping after 10 evaluations without sufficient improvement. Checkpoints maximize:

```text
0.5 × validation mean gene PCC − 0.5 × validation mean gene RMSE / fixed reference RMSE
```

The default reference scale is the validation RMSE of the SC mean, fixed before training. BRCA configs contain the historical fixed reference values. Test genes never select checkpoints. Undefined validation PCC genes are named explicitly; training fails if every validation PCC is undefined.

For remote servers, launch long training runs in a persistent session, for example:

```bash
tmux new -s gist-training
# Run the training command inside this session; detach with Ctrl-b, then d.
```

Memory use grows with the number of SC cells, genes, ST batch size and top-k. Counts and one normalized SC context reside on the selected device; exact SC distance search is performed for every batch. Reduce `batch_cells` and `sc_aux_batch` when necessary and record the change. `prediction_batch` controls validation inference; `predict --batch-size` controls standalone inference. Full study runtime and peak memory have not been benchmarked for this cleaned release.

## Prediction and evaluation

Use the `predict` and `evaluate` commands from the quick start, replacing the paths. `predict --partition validation` selects validation genes; the default is `test`. Predictions are saved before test truth is opened by the separate evaluation command.

| Artifact | Meaning |
|---|---|
| `train/best.pt` | Selected model weights, architecture settings, training-gene order and data fingerprint |
| `train/config.json`, `run.json` | Hyperparameters, reference scale, software versions and source hashes |
| `train/training.jsonl` | Epoch losses, validation metrics and stopping state |
| `train/selection.json` | Selected epoch, validation scores and checkpoint hash |
| `prediction/prediction.npy` | Float32 **ST cells × target genes**, in native log1p(CP10K) units |
| `prediction/cell_ids.json`, `genes.json` | Exact row and column labels |
| `evaluation/per_gene.csv` | RMSE, MAE, PCC and standard deviations for each gene |
| `evaluation/metrics.json` | Gene means and explicit undefined-PCC gene lists |

CP10K library sizes use only the visible training genes. During self-supervision, each mask has its own visible-gene denominator; inference uses all training genes. Predictions are not raw counts. Evaluation computes each metric across cells for each gene, then averages genes equally. RMSE is **mean per-gene RMSE**, not the RMSE of a flattened matrix. All genes remain in RMSE and MAE; PCC averages finite values and reports every undefined gene.

## Imputing genes that were never measured in ST

After training, create a text file with one target gene per line. The target genes must be present in SC, and the ST input must contain the checkpoint's observed training genes. Target expression is not required in ST.

```bash
gist impute \
  --st data/st_observed_only.h5ad \
  --sc data/sc_full_counts.h5ad \
  --checkpoint outputs/brca/fold00/train/best.pt \
  --genes data/target_genes.txt \
  --output outputs/brca/unmeasured \
  --device cuda:0
```

Use the same reference and preprocessing as training when applying the model to the same tissue. The checkpoint fixes the observed gene set; adding validation genes as inputs requires retraining with a new split. Unmeasured-gene predictions have no measured ST ground truth and cannot be scored by `gist evaluate`.

## Validation

```bash
python -m unittest discover -s tests -v
```

The model was checked against the selected research implementation for identical initialization, all six loss terms, transfer outputs and parameter gradients on a fixed CPU input. See [validation scope](docs/validation.md) for the end-to-end checks and their limits. This release does not claim a new full-dataset reproduction or new performance numbers.
