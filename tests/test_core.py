import unittest

import numpy as np
import torch

from ginger.data import features
from ginger.graph import build_graph, incoming_index, incoming_neighborhood
from ginger.metrics import evaluate_genes
from ginger.model import GINGER, relation_weights
from ginger.prepare import correlations, make_masks, make_split, validate_split
from ginger.train import update_monitor


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_masked_counts_do_not_affect_inputs_or_size_factors(self):
        raw = torch.rand(12, 10)
        observed = [0, 2, 5, 8]
        x, factors = features(raw, observed)
        changed = raw.clone()
        changed[:, [1, 3, 4, 6, 7, 9]] += 10000
        y, other_factors = features(changed, observed)
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        torch.testing.assert_close(factors, other_factors, rtol=0, atol=0)
        self.assertEqual(int(torch.count_nonzero(x[:, [1, 3, 4, 6, 7, 9]])), 0)

    def test_minibatch_gat_matches_full_graph(self):
        coords = np.random.default_rng(3).random((40, 2))
        edges = torch.from_numpy(build_graph(coords))
        model = GINGER(12, topk=4).eval()
        features_ = torch.randn(40, 12)
        cells = torch.tensor([2, 11, 20, 32])
        nodes, local_edges, roots = incoming_neighborhood(
            incoming_index(edges, 40), cells
        )
        with torch.no_grad():
            full = model.ST_encoder(features_, edges)[cells]
            local = model.ST_encoder(features_[nodes], local_edges)[roots]
        torch.testing.assert_close(full, local, rtol=1e-5, atol=1e-6)

    def test_transfer_weights_and_gradient_path(self):
        model = GINGER(10, topk=5)
        st = torch.randn(6, 32, requires_grad=True)
        sc = torch.randn(20, 32, requires_grad=True)
        values = torch.rand(20, 3)
        relation = relation_weights(torch.rand(3, 10), torch.ones(10, dtype=torch.bool))
        output = model.transfer_cells(
            st,
            sc,
            values,
            torch.rand(6, 10),
            torch.rand(20, 10),
            relation,
            torch.ones(10),
        )
        torch.testing.assert_close(output["weights"].sum(1), torch.ones(6, 3))
        expected = (output["weights"] * values[output["indices"]]).sum(1)
        torch.testing.assert_close(output["soft"], expected)
        output["soft"].sum().backward()
        self.assertGreater(float(st.grad.norm()), 0)
        self.assertGreater(float(sc.grad.norm()), 0)
        self.assertTrue(torch.isfinite(st.grad).all())

    def test_metrics_keep_all_errors_and_report_undefined_pcc(self):
        truth = np.array([[0, 0, 1], [1, 1, 1], [2, 2, 1]], float)
        pred = np.array([[0, 1, 2], [1, 1, 2], [2, 1, 2]], float)
        frame, summary = evaluate_genes(truth, pred, ["a", "b", "c"])
        self.assertEqual(summary["undefined_pcc_genes"], ["b", "c"])
        self.assertAlmostEqual(summary["pcc"], 1)
        self.assertAlmostEqual(summary["rmse"], (np.sqrt(2 / 3) + 1) / 3)
        self.assertEqual(len(frame), 3)

    def test_splits_and_masks(self):
        split = make_split([f"g{i}" for i in range(313)], 1234)
        self.assertEqual(
            [len(split[k]) for k in ["train_genes", "val_genes", "test_genes"]],
            [250, 31, 32],
        )
        validate_split(split)
        masks = make_masks(split["train_genes"])
        counts = np.bincount(
            [i for c in masks["contexts"] for i in c["query"]], minlength=250
        )
        np.testing.assert_array_equal(counts, np.full(250, 3))
        split["test_genes"][0] = split["train_genes"][0]
        with self.assertRaises(ValueError):
            validate_split(split)

    def test_correlations_match_direct_pearson(self):
        counts = np.random.default_rng(2).poisson(3, (50, 12)).astype(np.float32)
        libraries = counts.sum(1) + 100
        actual = correlations(counts, libraries, 8)
        norm = np.log1p(counts.astype(float) * (10000 / libraries)[:, None])
        expected = np.corrcoef(norm, rowvar=False)[:, :8]
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

    def test_patience_and_best_checkpoint_are_distinct(self):
        state = {}
        cfg = dict(min_delta=0.002, patience=2)
        self.assertTrue(update_monitor(state, 0.1, 10, cfg))
        self.assertTrue(update_monitor(state, 0.101, 20, cfg))
        self.assertEqual(state["bad"], 1)
        self.assertFalse(update_monitor(state, 0.101, 30, cfg))
        self.assertTrue(state["stop"])
        self.assertEqual(state["best_epoch"], 20)


if __name__ == "__main__":
    unittest.main()
