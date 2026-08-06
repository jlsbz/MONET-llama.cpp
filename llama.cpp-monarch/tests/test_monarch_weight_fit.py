#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_DIR))

from monarch_reference import materialize_monarch_dense_reference  # noqa: E402
from monarch_weight_fit import fit_square_weight_rank1_blocks  # noqa: E402


class TestMonarchWeightFit(unittest.TestCase):
    def test_recovers_exact_rank1_block_pairs(self) -> None:
        rng = np.random.default_rng(7)
        block_size = 2
        width = block_size**2
        blocks = np.empty((block_size, block_size, block_size, block_size), dtype=np.float32)
        for source in range(block_size):
            for destination in range(block_size):
                u = rng.standard_normal(block_size).astype(np.float32)
                v = rng.standard_normal(block_size).astype(np.float32)
                blocks[source, destination] = np.outer(u, v)

        dense = blocks.transpose(0, 2, 1, 3).reshape(width, width)
        hf_weight = dense.T
        left, right, perm, metrics = fit_square_weight_rank1_blocks(
            hf_weight, block_size=block_size, power_iterations=20, seed=3
        )
        materialized = materialize_monarch_dense_reference(left, right, perm)

        np.testing.assert_allclose(materialized, dense, rtol=1e-5, atol=1e-5)
        self.assertLess(metrics["relative_frobenius_squared_error"], 1e-10)

    def test_rejects_unsupported_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_blocks == block_size"):
            fit_square_weight_rank1_blocks(
                np.eye(8, dtype=np.float32), block_size=2
            )


if __name__ == "__main__":
    unittest.main()
