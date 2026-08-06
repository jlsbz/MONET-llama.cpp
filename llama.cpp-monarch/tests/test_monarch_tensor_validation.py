#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_DIR))

from monarch_tensor_validation import (  # noqa: E402
    extract_monarch_fields,
    hf_layer_to_gguf_base,
    validate_monarch_arrays,
    validate_monarch_layer_for_architecture,
    validate_monarch_metadata,
)


class TestMonarchLayerMapping(unittest.TestCase):
    def test_maps_supported_attention_projections(self) -> None:
        expected = {
            "model.layers.0.self_attn.q_proj": "blk.0.attn_q",
            "model.layers.7.self_attn.k_proj": "blk.7.attn_k",
            "model.layers.18.self_attn.v_proj": "blk.18.attn_v",
            "model.layers.31.self_attn.o_proj": "blk.31.attn_output",
        }

        for layer_name, gguf_name in expected.items():
            with self.subTest(layer_name=layer_name):
                self.assertEqual(hf_layer_to_gguf_base(layer_name), gguf_name)

    def test_rejects_unsupported_or_partial_layer_names(self) -> None:
        unsupported = (
            "model.layers.0.mlp.gate_proj",
            "prefix.model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.q_proj.weight",
        )

        for layer_name in unsupported:
            with self.subTest(layer_name=layer_name):
                self.assertIsNone(hf_layer_to_gguf_base(layer_name))

    def test_requires_a_string_layer_name(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a string"):
            hf_layer_to_gguf_base(0)  # type: ignore[arg-type]

    def test_llama_accepts_all_attention_projections(self) -> None:
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            with self.subTest(projection=projection):
                validate_monarch_layer_for_architecture(
                    f"model.layers.0.self_attn.{projection}",
                    "llama",
                )

    def test_qwen2_accepts_only_square_q_and_o(self) -> None:
        for projection in ("q_proj", "o_proj"):
            with self.subTest(projection=projection):
                validate_monarch_layer_for_architecture(
                    f"model.layers.0.self_attn.{projection}",
                    "qwen2",
                )

        for projection in ("k_proj", "v_proj"):
            with self.subTest(projection=projection):
                with self.assertRaisesRegex(ValueError, "does not support"):
                    validate_monarch_layer_for_architecture(
                        f"model.layers.0.self_attn.{projection}",
                        "qwen2",
                    )

    def test_rejects_architectures_without_runtime_support(self) -> None:
        with self.assertRaisesRegex(ValueError, "not enabled"):
            validate_monarch_layer_for_architecture(
                "model.layers.0.self_attn.q_proj",
                "mistral",
            )


class TestMonarchObjectExtraction(unittest.TestCase):
    def test_accepts_target_module_name_alias(self) -> None:
        obj = {
            "target_module_name": "model.layers.0.self_attn.q_proj",
            "L": object(),
            "R": object(),
            "perm": object(),
        }

        layer_name, left, right, permutation = extract_monarch_fields(obj)

        self.assertEqual(layer_name, obj["target_module_name"])
        self.assertIs(left, obj["L"])
        self.assertIs(right, obj["R"])
        self.assertIs(permutation, obj["perm"])

    def test_requires_a_mapping_and_all_required_fields(self) -> None:
        with self.assertRaisesRegex(TypeError, "expected a mapping"):
            extract_monarch_fields(["not", "a", "mapping"])

        with self.assertRaisesRegex(KeyError, "layer_name"):
            extract_monarch_fields({"L": 1, "R": 2, "perm": 3})

        with self.assertRaisesRegex(KeyError, "'R', 'perm'"):
            extract_monarch_fields(
                {
                    "layer_name": "model.layers.0.self_attn.q_proj",
                    "L": 1,
                }
            )

    def test_requires_a_non_empty_string_layer_name(self) -> None:
        required = {"L": 1, "R": 2, "perm": 3}

        with self.assertRaisesRegex(TypeError, "must be a string"):
            extract_monarch_fields({"layer_name": 1, **required})

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            extract_monarch_fields({"layer_name": " ", **required})


class TestMonarchArrayValidation(unittest.TestCase):
    layer_name = "model.layers.0.self_attn.q_proj"

    def setUp(self) -> None:
        self.left = np.zeros((2, 4, 4), dtype=np.float32)
        self.right = np.ones((2, 4, 4), dtype=np.float16)
        self.permutation = np.arange(8, dtype=np.int64)

    def test_accepts_valid_factors_and_permutation(self) -> None:
        validate_monarch_arrays(
            self.layer_name,
            self.left,
            self.right,
            self.permutation[::-1],
        )

    def test_rejects_incompatible_factor_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank-3"):
            validate_monarch_arrays(
                self.layer_name,
                self.left[0],
                self.right,
                self.permutation,
            )

        with self.assertRaisesRegex(ValueError, "shapes must match"):
            validate_monarch_arrays(
                self.layer_name,
                self.left,
                np.zeros((1, 4, 4), dtype=np.float32),
                self.permutation,
            )

        with self.assertRaisesRegex(ValueError, "must be square"):
            validate_monarch_arrays(
                self.layer_name,
                np.zeros((2, 4, 3), dtype=np.float32),
                np.zeros((2, 4, 3), dtype=np.float32),
                self.permutation,
            )

    def test_rejects_non_floating_or_non_finite_factors(self) -> None:
        with self.assertRaisesRegex(TypeError, "L must use a floating dtype"):
            validate_monarch_arrays(
                self.layer_name,
                self.left.astype(np.int32),
                self.right,
                self.permutation,
            )

        left = self.left.copy()
        left[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "L contains NaN"):
            validate_monarch_arrays(
                self.layer_name,
                left,
                self.right,
                self.permutation,
            )

    def test_rejects_invalid_permutation_shape_or_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be rank-1"):
            validate_monarch_arrays(
                self.layer_name,
                self.left,
                self.right,
                self.permutation.reshape(2, 4),
            )

        with self.assertRaisesRegex(ValueError, "perm length"):
            validate_monarch_arrays(
                self.layer_name,
                self.left,
                self.right,
                self.permutation[:-1],
            )

        with self.assertRaisesRegex(TypeError, "integer dtype"):
            validate_monarch_arrays(
                self.layer_name,
                self.left,
                self.right,
                self.permutation.astype(np.float32),
            )

    def test_rejects_duplicate_or_out_of_range_permutation_entries(self) -> None:
        duplicate = self.permutation.copy()
        duplicate[-1] = duplicate[-2]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_monarch_arrays(
                self.layer_name,
                self.left,
                self.right,
                duplicate,
            )

        out_of_range = self.permutation.copy()
        out_of_range[-1] = len(out_of_range)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_monarch_arrays(
                self.layer_name,
                self.left,
                self.right,
                out_of_range,
            )

    def test_rejects_unsupported_layer_before_conversion(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported Monarch layer name"):
            validate_monarch_arrays(
                "model.layers.0.mlp.gate_proj",
                self.left,
                self.right,
                self.permutation,
                source="fixture.pt",
            )


class TestMonarchMetadataValidation(unittest.TestCase):
    def test_accepts_matching_or_absent_metadata(self) -> None:
        validate_monarch_metadata(
            {}, width=4096, block_size=64, num_blocks=64
        )
        validate_monarch_metadata(
            {
                "in_features": 4096,
                "out_features": np.asarray(4096, dtype=np.int64),
                "block_size": 64,
                "num_blocks": 64,
                "bias": None,
            },
            width=4096,
            block_size=64,
            num_blocks=64,
        )

    def test_rejects_inconsistent_dimensions(self) -> None:
        for field, value, message in (
            ("in_features", 2048, "in_features"),
            ("out_features", 2048, "out_features"),
            ("block_size", 32, "block_size"),
            ("num_blocks", 32, "num_blocks"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    validate_monarch_metadata(
                        {field: value}, width=4096, block_size=64, num_blocks=64
                    )

    def test_rejects_fitted_bias(self) -> None:
        with self.assertRaisesRegex(ValueError, "fitted bias"):
            validate_monarch_metadata(
                {"bias": np.zeros(8, dtype=np.float32)},
                width=8,
                block_size=4,
                num_blocks=2,
            )


if __name__ == "__main__":
    unittest.main()
