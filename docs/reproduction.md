# Reproduction scope

## Selected model

The release is based on the completed BRCA pre-DCA gene-aware model used in the GIST figures, originally stored under the experiment name `brca_dim32_mask10_sc3linear_strength010_5fold_20261017`, condition `relation_strength010`. Source hashes are recorded in [source_provenance.json](source_provenance.json).

The cleaned model preserves parameter names, initialization order, encoder/decoder calculations, exact top-k retrieval, gene-specific transfer and six loss terms. The portable runtime adopts the later experiment's one-active-SC-context allocation to avoid caching all ten normalized reference contexts on the GPU. Original experiments, predictions and checkpoints remain separate from this release.

## BRCA

The original five gene splits, ten masks per split, seed 3407, network dimensions, optimizer settings, loss weights and fold-specific validation reference RMSE values are bundled. For fold `00`, use:

```bash
gist prepare --st data/brca_st_counts.h5ad --sc data/brca_sc_full_counts.h5ad \
  --split splits/brca/fold00.json --masks splits/brca/masks00.json \
  --output data/brca/fold00
gist train --data data/brca/fold00 --config configs/brca_fold00.json \
  --output outputs/brca/fold00/train --device cuda:0
gist predict --data data/brca/fold00 --checkpoint outputs/brca/fold00/train/best.pt \
  --output outputs/brca/fold00/prediction --device cuda:0
gist evaluate --data data/brca/fold00 --prediction outputs/brca/fold00/prediction \
  --output outputs/brca/fold00/evaluation
```

Repeat with `01` through `04`, changing the split, mask, config and output paths together. Raw input provenance, QC, cell/gene order and full SC library totals must match the original study. A complete BRCA retraining has not been run for this cleaned release. Hardware, PyTorch/CUDA and graph construction versions can affect numerical reproducibility.

## Lung and TNBC

The fixed split assignments are provided for reuse. Default training recreates a ten-mask schedule with `--mask-seed 63410`, incremented by the split index. These split files alone are not a complete frozen reproduction of every historical Lung/TNBC run: exact QC/reference matrices, historical prior normalization and dataset-specific selection settings must also be matched. The default reference-RMSE policy is documented explicitly rather than borrowing BRCA's scales.

## Research artifacts and publication metadata

This repository contains executable source and small split/configuration artifacts. It does not include raw study data, large checkpoints, baseline implementations, experiment queues, diagnostic sweeps or saved figure analyses. Download locations for the exact processed study inputs and pretrained release weights have not been provided. The synthetic example is runnable without those resources.

Paper metadata, the final project license and a public repository URL remain for the authors to supply. No remote GitHub repository is assumed by the commands above. The release contains no fabricated citation, performance table or link to unavailable weights.
