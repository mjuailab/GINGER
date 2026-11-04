# Third-party attribution

`src/gist/layers.py` adapts the `MLPEncoder`, `MLPDecoder`, `GATLayer` and `GATEncoder` implementations from **stAI**:

- Repository: https://github.com/gszou99/stAI
- Inspected local upstream commit: `3376cc16cc6d8461edafc0aeb4519b92d18474b7`
- Original file: `stAI/layers.py`
- Changes relative to upstream: ReLU activations changed to ELU in the selected research implementation; the second SC hidden layer retained. Unused classifiers and attention helpers removed; formatting and imports cleaned.

The selected local research implementation and source-file hashes are recorded in [source_provenance.json](docs/source_provenance.json). Losses and transfer behavior are preserved from that implementation. PyTorch Geometric supplies `GATConv` as an installed dependency; its source is not vendored.

No project-wide license has been assigned to this release. No LICENSE file was found in the inspected stAI checkout. Upstream attribution is retained; it is not a substitute for a license grant. The authors must establish the applicable distribution terms before assigning this repository a license.
