# Data preparation

## Required files

Supply ST and SC AnnData files with raw counts, matching gene identifiers and unique cell identifiers. A count layer can be selected explicitly. The reader supports dense and sparse `.h5ad` matrices. Spatial coordinates may be provided in `obsm['spatial']` or two observation columns selected with `--x-column` and `--y-column`.

The repository does not download data, perform gene-name remapping, deduplicate genes, select reference cell types, or apply dataset-specific QC. Those decisions must be made explicitly before `gist prepare`. Inputs must contain at least four spatial cells for Delaunay triangulation and a valid two-dimensional coordinate array. Triangulation failures are surfaced; there is no silent kNN fallback.

## Splits

```json
{
  "train_genes": ["GENE_A", "GENE_B", "GENE_C", "GENE_D"],
  "val_genes": ["GENE_E"],
  "test_genes": ["GENE_F"]
}
```

This illustrates the schema only. Use enough variable genes and cells for meaningful training and validation. Every benchmark split gene must occur in both input matrices; ordering in the JSON defines the model and output axes.

| Included panel | Repeated split specification |
|---|---|
| BRCA | Original 313-gene panel; seeds 1234–1238; 250/31/32 genes; exact ten training masks bundled |
| Lung | Original 497-gene panel; seeds 1234–1238; 397/49/51 genes |
| TNBC | Selected 1,418-gene SC/ST detection-filtered panel; original seeds 2026102600–2026102604; 1,134/142/142 genes |

The TNBC selection used 42,512 TNBC SC cells from GSE176078 and 4,649 ST cells in core 1. Genes were required to be detected in at least 5% of both SC and core-1 ST cells before splitting. Consequently, this panel is conditional on measured ST detection; the splits do not establish unbiased performance over all unmeasured genes or over other cores. The data matrices themselves are not bundled.

## Normalization contract

Do not supply pre-normalized or log-transformed `.X` without selecting a raw-count layer. Preparation rejects negative, nonfinite and noninteger counts. Training/evaluation use visible-train-library CP10K/log1p. Relation priors use the library totals of the full supplied SC count matrix. Removing SC genes before preparation can therefore change the relation prior even when the requested output panel is unchanged.

For historical BRCA reproduction, the SC prior used the full 37,143-gene reference after the historical cell QC (44,863 cells). A reference restricted to the 313-gene panel is insufficient to reproduce that prior. Original ST ordering contained 167,780 cells. Exact data, QC and cell order must match, not just filenames.

## Prepared directory

Preparation creates training counts, SC counts, relation priors, graph edges, masks and splits, plus separate validation/test count arrays. `manifest.json` records input filenames and SHA-256 values, normalization definitions and artifact hashes. It contains no machine-specific absolute paths. Runtime opens training arrays and validation counts; the separate evaluation command opens test counts after predictions have been saved.

Each prepared directory is self-contained and can be moved together with its contents. Checkpoints and predictions verify its manifest. Use a new directory to change data or splits; do not edit prepared arrays in place.
