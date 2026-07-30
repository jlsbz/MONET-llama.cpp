#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_DIR))

from monarch_reference import (  # noqa: E402
    materialize_monarch_dense_reference,
    monarch_linear_reference,
)


class TestMonarchReference(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(7)
        self.left = generator.normal(size=(2, 3, 3)).astype(np.float32)
        self.right = generator.normal(size=(2, 3, 3)).astype(np.float32)
        self.permutation = np.array([0, 3, 1, 4, 2, 5], dtype=np.int64)

    def test_block_forward_matches_independent_dense_materialization(self) -> None:
        inputs = np.arange(24, dtype=np.float32).reshape(4, 6) / 10

        actual = monarch_linear_reference(
            inputs,
            self.left,
            self.right,
            self.permutation,
        )
        dense = materialize_monarch_dense_reference(
            self.left,
            self.right,
            self.permutation,
        )

        np.testing.assert_allclose(actual, inputs @ dense, rtol=1e-6, atol=1e-6)

    def test_one_block_matches_explicit_right_then_left_multiplication(self) -> None:
        inputs = np.array([[1.0, -2.0]], dtype=np.float32)
        right = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        left = np.array([[[2.0, 0.0], [1.0, -1.0]]], dtype=np.float32)
        permutation = np.array([0, 1], dtype=np.int64)

        actual = monarch_linear_reference(inputs, left, right, permutation)
        expected = (inputs @ right[0]) @ left[0]

        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_identity_factors_expose_gather_permutation_semantics(self) -> None:
        identity = np.stack([np.eye(2), np.eye(2)]).astype(np.float32)
        permutation = np.array([2, 0, 3, 1], dtype=np.int32)
        inputs = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)

        actual = monarch_linear_reference(
            inputs,
            identity,
            identity,
            permutation,
        )

        np.testing.assert_array_equal(actual, inputs[:, permutation])

    def test_preserves_leading_dimensions_and_uses_float32_minimum(self) -> None:
        inputs = np.arange(24, dtype=np.float16).reshape(2, 2, 6)

        actual = monarch_linear_reference(
            inputs,
            self.left.astype(np.float16),
            self.right.astype(np.float16),
            self.permutation,
        )

        self.assertEqual(actual.shape, inputs.shape)
        self.assertEqual(actual.dtype, np.float32)

    def test_rejects_wrong_input_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "last dimension"):
            monarch_linear_reference(
                np.zeros((2, 5), dtype=np.float32),
                self.left,
                self.right,
                self.permutation,
            )

    def test_rejects_non_floating_or_non_finite_inputs(self) -> None:
        with self.assertRaisesRegex(TypeError, "floating dtype"):
            monarch_linear_reference(
                np.zeros((2, 6), dtype=np.int32),
                self.left,
                self.right,
                self.permutation,
            )

        inputs = np.zeros((2, 6), dtype=np.float32)
        inputs[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            monarch_linear_reference(
                inputs,
                self.left,
                self.right,
                self.permutation,
            )


if __name__ == "__main__":
    unittest.main()
