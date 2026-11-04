# Validation scope

The cleaned model was compared against the selected research implementation on CPU with seed 3407, 250 training genes, latent dimension 32 and top-100 retrieval. Parameter initialization, all six loss terms, transfer outputs and every parameter gradient were bitwise identical for the checked input.

The included unit tests cover masked-input independence, full-graph/minibatch GAT agreement, transfer weights and gradient flow, explicit undefined-PCC handling, split/mask validity, SC correlation construction and checkpoint/patience behavior.

The synthetic CLI example passed preparation, two training epochs, checkpoint loading, prediction and evaluation on 96 ST cells, 128 SC cells and 30 genes. Separate inference with all ST target columns removed matched the held-out predictions within `1e-5` absolute/relative tolerance. Package installation and CLI execution outside the repository root also passed.

Seven unit tests passed. All 15 saved gene splits and five original BRCA mask sets passed validation. Checks used Python 3.10, PyTorch 2.5.1+cu118 on CPU, PyTorch Geometric 2.6.1, NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, AnnData 0.11.4 and h5py 3.16.0. Generated verification data, checkpoints and build files were removed after checking.

These checks validate software behavior. They do not replace a full study rerun, establish new model performance or guarantee numerical identity across different hardware and dependency versions. No long-running research training is launched as part of repository cleanup.
