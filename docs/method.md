# Method and implementation

GIST combines a one-layer spatial graph encoder with a three-linear-layer SC encoder. Both decoders reconstruct the entire outer training-gene panel. The GAT uses complete incoming neighborhoods for each root cell; edges are not restricted to cells in the current minibatch. With one message-passing layer this preserves full-graph root outputs.

## Normalization and gene information

For each cell, expression is `log1p(count × 10000 / max(visible_count_sum, 1))`. Only visible outer training genes contribute to this sum. Masked genes are zeroed for all ST graph nodes and SC encoder inputs. SC target values and full training-gene reconstruction targets use the same visible-gene denominator.

The gene relation prior is Pearson correlation between SC target and observed genes after CP10K/log1p normalization using all genes in the supplied SC matrix. This is a separate normalization from the mask-specific transfer values. ST validation/test expression does not enter the prior. Squared correlations are normalized across variable observed SC genes, with a uniform fallback when the row sum is zero. SC variance is computed in the current visible-gene normalization; variances below `1e-8` are excluded and the inverse-variance denominator is floored at `0.01`.

## Gene-aware transfer

Let `b_ij = softmax_j(-||z_ST_i - z_SC_j|| / 2)` on the selected top-100 SC cells. For target gene `g`:

```text
a_gh = rho(g,h)^2 / sum_h rho(g,h)^2
D_ijg = sum_h a_gh × (ST_ih − SC_jh)^2 / max(var_SC_h, 0.01)
w_ijg = softmax_j(log(max(b_ij, 1e-30)) − 0.1 × D_ijg)
prediction_ig = sum_j w_ijg × normalized_SC_jg
```

`h` ranges over variable observed genes. The observed-expression distance is fixed with respect to model gradients. Gradients pass through selected latent distances and their softmax weights; the discrete top-k indices are not differentiable. All SC latent representations are recomputed at every training batch.

## Objective

```text
L = 0.1 × ST reconstruction MSE
  + 0.1 × SC reconstruction MSE
  + 1.0 × five-RBF MMD
  + 1.0 × masked-gene (1 − PCC)
  + 1.0 × masked-gene MSE
  + 1.0 × cross-decoder cosine consistency
```

Reconstruction targets include all outer training genes. Cosine consistency uses only visible genes. Masked imputation losses use inverse query-occurrence weights. Constant target genes are excluded from the PCC loss, while constant predictions remain in it. MMD uses the original biased five-bandwidth kernel and a finite bandwidth floor. Adam updates are clipped to gradient norm 1.

New ten-mask schedules are formed by splitting three repetitions of a seeded gene permutation into ten chunks: each training gene is queried three times. BRCA's historical masks are supplied separately because they retained an earlier seven-mask schedule and added three masks.

## Checkpoint selection

Every strict improvement in the joint validation score saves `best.pt`. Patience resets only when the score exceeds its anchor by `min_delta=0.002`. Patience counts validation evaluations, not epochs. The generic pipeline fixes an SC-mean validation RMSE before training; BRCA configs preserve its historical fixed baseline scales. Validation PCC uses finite gene correlations and names undefined genes; test metrics never influence selection.

## Scope

The selected implementation uses no diffusion, DCA count likelihood, gene sequence embeddings, image features or reference cell-type labels. The model produces point estimates from reference expression. Similarity between predictions and tissue structure is not independent evidence of biological validity. Evaluation on held-out measured genes and prediction of genuinely unmeasured genes are distinct uses.
