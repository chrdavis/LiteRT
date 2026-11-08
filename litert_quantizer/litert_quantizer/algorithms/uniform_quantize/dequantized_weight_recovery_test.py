# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import dequantized_weight_recovery
from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig


def _construct_blockwise_dequantized_weights(
    quantized_weights: np.ndarray,
    scale: np.ndarray,
    quantized_dimension: int,
    block_size: int,
) -> np.ndarray:
  """Constructs blockwise dequantized weights from quantized weights and scale.

  Args:
    quantized_weights: The quantized weights of shape S.
    scale: The scale array of shape where S[quantized_dimension] is replaced by
      S[quantized_dimension] // block_size.
    quantized_dimension: The dimension along which blockwise quantization is
      performed.
    block_size: The size of each block.

  Returns:
    The dequantized weights of shape S.
  """
  assert scale.shape == common_utils.get_blockwise_shape(
      quantized_weights.shape, quantized_dimension, block_size
  ), f"Scale shape {scale.shape} does not match expected blockwise shape."
  expanded_shape = []
  for i, dim in enumerate(quantized_weights.shape):
    if i == quantized_dimension:
      expanded_shape.append(dim // block_size)
      expanded_shape.append(block_size)
    else:
      expanded_shape.append(dim)
  qw_expanded = quantized_weights.reshape(expanded_shape)
  scale_expanded = np.expand_dims(scale, quantized_dimension + 1)
  dequant_expanded = qw_expanded * scale_expanded
  return dequant_expanded.reshape(quantized_weights.shape)


def _round_to_bfloat16(values: np.ndarray) -> np.ndarray:
  """Rounds values to bfloat16 precision, keeping the float32 dtype."""
  bits = np.asarray(values, dtype=np.float32).view(np.uint32)
  # Round to nearest even, then drop the 16 mantissa bits bfloat16 lacks.
  rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
  return rounded.view(np.float32)


class DequantizedWeightRecoveryTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._dummy_quantized_weights = np.array([
        [1, -2, 3, 4],
        [6, 7, -6, 5],
        [2, -6, -7, -4],
    ])
    self._dummy_quantized_weights_4x4 = np.array([
        [1, -2, 3, -4],
        [3, 1, 4, 1],
        [2, 3, 1, -1],
        [1, 4, 2, 1],
    ])
    self._dummy_op_info = qtyping.OpInfo(
        op=None,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=0,
        op_quant_config=qtyping.OpQuantizationConfig(),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="per-tensor-recovery",
          quantized_dimension=None,
          scale=np.array([0.1875]).reshape(1, 1),
      ),
      dict(
          testcase_name="channel0-recovery",
          quantized_dimension=0,
          scale=np.array([0.1875, 1e-4, 12.3]).reshape(3, 1),
      ),
      dict(
          testcase_name="channel1-recovery",
          quantized_dimension=1,
          scale=np.array([0.003, 1.234, 12.65, 2.24e-4]).reshape(1, 4),
      ),
  )
  def test_tensor_zp_scale_from_2d_dequantized_symmetric_weights_success(
      self, quantized_dimension, scale
  ):
    dequant_vals = scale * self._dummy_quantized_weights
    zp, recovered_scale = (
        dequantized_weight_recovery.get_zp_scale_from_dequantized_symmetric_weights(
            dequant_vals, quantized_dimension
        )
    )
    self.assertEqual(recovered_scale.shape, scale.shape)
    self.assertSequenceAlmostEqual(recovered_scale.flatten(), scale.flatten())
    # Zero point should be zero for symmetric quantization.
    self.assertEqual(np.sum(zp), 0)
    self.assertEqual(zp.shape, scale.shape)

  @parameterized.named_parameters(
      dict(
          testcase_name="per-tensor-recovery",
          quantized_dimension=None,
          scale=np.array([0.1875]).reshape(1, 1),
      ),
      dict(
          testcase_name="channel0-recovery",
          quantized_dimension=0,
          scale=np.array([0.1875, 1e-4, 12.3]).reshape(3, 1, 1),
      ),
      dict(
          testcase_name="channel1-recovery",
          quantized_dimension=1,
          scale=np.array([0.003, 1.234]).reshape(1, 2, 1),
      ),
  )
  def test_tensor_zp_scale_from_3d_dequantized_symmetric_weights_success(
      self, quantized_dimension, scale
  ):
    dequant_vals = scale * self._dummy_quantized_weights.reshape(3, 2, 2)
    zp, recovered_scale = (
        dequantized_weight_recovery.get_zp_scale_from_dequantized_symmetric_weights(
            dequant_vals, quantized_dimension
        )
    )
    with self.subTest("shapes_match"):
      self.assertEqual(recovered_scale.shape, scale.shape)
      self.assertEqual(zp.shape, scale.shape)
    with self.subTest("scale_value_match"):
      self.assertSequenceAlmostEqual(recovered_scale.flatten(), scale.flatten())
    with self.subTest("zp_is_zero"):
      # Zero point should be zero for symmetric quantization.
      self.assertEqual(np.sum(zp), 0)

  @parameterized.named_parameters(
      dict(
          testcase_name="blockwise-dim0-recovery",
          quantized_dimension=0,
          block_size=2,
          scale=np.array([
              [0.1, 0.2, 0.3, 0.4],
              [0.5, 0.6, 0.7, 0.8],
          ]),
      ),
      dict(
          testcase_name="blockwise-dim1-recovery",
          quantized_dimension=1,
          block_size=2,
          scale=np.array([
              [0.1, 0.2],
              [0.3, 0.4],
              [0.5, 0.6],
              [0.7, 0.8],
          ]),
      ),
  )
  def test_tensor_zp_scale_from_2d_dequantized_symmetric_weights_blockwise_success(
      self, quantized_dimension, block_size, scale
  ):
    dequant_vals = _construct_blockwise_dequantized_weights(
        self._dummy_quantized_weights_4x4,
        scale,
        quantized_dimension,
        block_size,
    )
    zp, recovered_scale = (
        dequantized_weight_recovery.get_zp_scale_from_dequantized_symmetric_weights(
            dequant_vals, quantized_dimension, block_size
        )
    )
    with self.subTest(name="shapes_match"):
      self.assertEqual(recovered_scale.shape, scale.shape)
      self.assertEqual(zp.shape, scale.shape)
    with self.subTest(name="scale_value_match"):
      self.assertSequenceAlmostEqual(recovered_scale.flatten(), scale.flatten())
    with self.subTest(name="zp_is_zero"):
      self.assertEqual(np.sum(zp), 0)

  def test_channelwise_recovery_survives_bfloat16_rounded_weights(self):
    """Recovers scales that bfloat16 cannot represent exactly.

    QAT stores `q * scale` in bfloat16 while the scale itself stays fp32. When
    the scale needs more mantissa bits than bfloat16 has, every level rounds
    independently and the gaps between adjacent levels stop being equal.
    Estimating the scale from the smallest gap then always underestimates it.
    """
    # A full set of 4-bit levels, so each channel pins down its own scale.
    codes = np.tile(np.arange(-8, 8, dtype=np.float32), (4, 1))
    scales = np.array(
        [0.0083849, 0.0187500, 0.0412350, 0.0061095], dtype=np.float32
    )
    dequant_vals = _round_to_bfloat16(codes * scales[:, np.newaxis])

    _, recovered = (
        dequantized_weight_recovery.get_zp_scale_from_dequantized_symmetric_weights(
            dequant_vals, quantized_dimension=0, num_bits=4
        )
    )
    recovered = recovered.flatten()

    with self.subTest(name="premise_scales_are_not_bfloat16_representable"):
      # Otherwise the levels stay evenly spaced and the test proves nothing.
      self.assertFalse(np.any(_round_to_bfloat16(scales) == scales))

    with self.subTest(name="premise_smallest_gap_is_misleading"):
      levels = np.sort(np.abs(dequant_vals), axis=1)
      gaps = np.diff(levels, axis=1)
      smallest_gap = np.where(gaps > 0, gaps, np.inf).min(axis=1)
      self.assertGreater(np.abs(smallest_gap / scales - 1).max(), 0.01)

    with self.subTest(name="scale_value_match"):
      # Seeding from the gaps alone lands at 6.4e-3 here; the least-squares
      # refit averages the rounding noise away and lands at 1.2e-3.
      self.assertLess(np.abs(recovered / scales - 1).max(), 2.5e-3)

    with self.subTest(name="weights_are_reconstructed"):
      per_channel_scale = recovered[:, np.newaxis]
      requantized = np.rint(dequant_vals / per_channel_scale)
      err = np.abs(requantized * per_channel_scale - dequant_vals)
      # Every weight lands within 5% of a quantization step of its level.
      self.assertLess((err.max(axis=1) / recovered).max(), 0.05)

  def test_channelwise_recovery_repairs_sparse_lattice_group(self):
    """Recovers a group whose gaps are all multiples of the step.

    With levels {0, 2s, 7s} both the smallest gap (2s) and the median gap
    (3.5s) overestimate the step, so neither the gap seed nor the
    least-squares refit around it reconstructs the group losslessly.
    """
    step = np.float32(0.01303)
    dense = np.arange(-8, 8, dtype=np.float32) * np.float32(0.02)
    sparse = np.zeros(16, dtype=np.float32)
    sparse[3] = 2 * step
    sparse[11] = -7 * step
    dequant_vals = np.stack([dense, sparse])

    _, recovered = (
        dequantized_weight_recovery.get_zp_scale_from_dequantized_symmetric_weights(
            dequant_vals, quantized_dimension=0, num_bits=4
        )
    )
    recovered = recovered.flatten()

    with self.subTest(name="dense_group_scale_match"):
      self.assertAlmostEqual(float(recovered[0]), 0.02, delta=1e-7)

    with self.subTest(name="sparse_group_scale_match"):
      self.assertAlmostEqual(float(recovered[1]), float(step), delta=1e-7)

    with self.subTest(name="weights_are_reconstructed"):
      per_channel_scale = recovered[:, np.newaxis]
      requantized = np.clip(np.rint(dequant_vals / per_channel_scale), -8, 7)
      np.testing.assert_allclose(
          requantized * per_channel_scale, dequant_vals, rtol=0, atol=1e-6
      )

  def test_get_tensor_quant_params_blockwise_sparse_lattice_block(self):
    """A sparse block passes the lossless validation (Gemma 4 26B regression).

    Mirrors the block of a Gemma 4 26B-A4B down_proj weight that made the
    recovery fail: 30 zeros, 2s and 7s in one block of 32.
    """
    block_size = 32
    step = np.float32(0.01303)
    dense = np.array([1, -2, 3, -4] * 8, dtype=np.float32) * np.float32(0.1)
    sparse = np.zeros(block_size, dtype=np.float32)
    sparse[12] = 2 * step
    sparse[17] = 7 * step
    dequant_vals = np.vstack([dense, sparse])

    tensor_quant_config = qtyping.TensorQuantizationConfig(
        num_bits=4,
        granularity=qtyping.QuantGranularity.BLOCKWISE_32,
    )
    tensor_quant_params = dequantized_weight_recovery.get_tensor_quant_params(
        self._dummy_op_info, tensor_quant_config, dequant_vals
    )

    with self.subTest(name="scale_match"):
      np.testing.assert_allclose(
          tensor_quant_params.scale.flatten(),
          np.array([0.1, step], dtype=np.float32),
          rtol=1e-6,
      )

    with self.subTest(name="codes_match"):
      codes = np.asarray(tensor_quant_params.quantized_data)
      self.assertEqual(codes[1, 12], 2)
      self.assertEqual(codes[1, 17], 7)
      self.assertEqual(np.count_nonzero(codes[1]), 2)

  def test_compute_scale_from_diffs_picks_min_or_median_gap(self):
    """Picks the median gap on dense rows and the minimum gap on sparse rows.

    Calls the helper directly because the least-squares refit that follows it
    in `get_zp_scale_from_dequantized_symmetric_weights` would hide these edge
    cases.
    """
    diffs = np.array(
        [
            # Dense with rounding noise: median 0.01 <= 1.35 * min 0.0099.
            [0.0, 0.0099, 0.01, 0.0101],
            # Sparse: median 0.05 > 1.35 * min 0.01, so the minimum wins.
            [0.0, 0.01, 0.05, 0.05],
            # Even count: the median averages the two middle gaps.
            [0.02, 0.02, 0.03, 0.03],
            # A single positive gap.
            [0.0, 0.0, 0.02, 0.0],
            # No positive gap falls back to `min_scale`.
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    scales = dequantized_weight_recovery._compute_scale_from_diffs(  # pylint: disable=protected-access
        diffs, min_scale=1e-9
    )

    np.testing.assert_allclose(
        scales, [0.01, 0.01, 0.025, 0.02, 1e-9], rtol=1e-6
    )

  @parameterized.named_parameters(
      dict(testcase_name="negative_dimension", quantized_dimension=-1),
      dict(testcase_name="too_large_dimension", quantized_dimension=2),
  )
  def test_tensor_zp_scale_from_2d_dequantized_symmetric_weights_raises_error_for_invalid_quantized_dimension(
      self, quantized_dimension
  ):
    dequant_vals = self._dummy_quantized_weights * 1.02
    with self.assertRaisesRegex(
        ValueError, "quantized_dimension must be 0, 1, a sequence, or None. Got"
    ):
      dequantized_weight_recovery.get_zp_scale_from_dequantized_symmetric_weights(
          dequant_vals, quantized_dimension
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="tensor-recovery-tensor-quant",
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=4,
              granularity=qtyping.QuantGranularity.TENSORWISE,
          ),
          scale=np.array([0.1875]).reshape(1, 1),
      ),
      dict(
          testcase_name="channel-recovery-channel-quant",
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=4,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          scale=np.array([0.1875, 1e-4, 12.3]).reshape(3, 1),
      ),
      dict(
          testcase_name="channel-recovery-excessive-bits",
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=8,  # int4 is enough for the sample weights.
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          scale=np.array([0.1875, 1e-4, 12.3]).reshape(3, 1),
      ),
      dict(
          testcase_name="multiaxis-channel-recovery",
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=4,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
              quantized_dimensions=[0, 1],
          ),
          scale=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape(2, 3, 1),
      ),
  )
  def test_get_tensor_quant_params_success_with_dequantized_weights(
      self, tensor_quant_config, scale
  ):
    dequant_vals = scale * self._dummy_quantized_weights
    op_info = qtyping.OpInfo(
        op=None,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=0,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=tensor_quant_config
        ),
    )
    tensor_quant_params = dequantized_weight_recovery.get_tensor_quant_params(
        op_info, tensor_quant_config, dequant_vals
    )

    if tensor_quant_config.granularity is qtyping.QuantGranularity.TENSORWISE:
      self.assertIsNone(tensor_quant_params.quantized_dimension)
      self.assertIsNone(tensor_quant_params.quantized_dimensions)
    elif tensor_quant_config.quantized_dimensions is not None:
      self.assertIsNone(tensor_quant_params.quantized_dimension)
      self.assertEqual(
          tensor_quant_params.quantized_dimensions,
          tuple(tensor_quant_config.quantized_dimensions),
      )
    else:
      self.assertEqual(tensor_quant_params.quantized_dimension, 0)
      self.assertIsNone(tensor_quant_params.quantized_dimensions)

    recovered_scale = tensor_quant_params.scale
    self.assertEqual(recovered_scale.shape, scale.shape)
    self.assertSequenceAlmostEqual(recovered_scale.flatten(), scale.flatten())

    # Zero point should be zero for symmetric quantization.
    recovered_zp = tensor_quant_params.zero_point
    self.assertEqual(np.sum(recovered_zp), 0)
    self.assertEqual(recovered_zp.shape, scale.shape)

  @parameterized.named_parameters(
      dict(
          testcase_name="single_axis",
          quantized_dimensions=None,
          weight_shape=(2, 32),
          # FULLY_CONNECTED blockwise quantized dim is 1. Weight shape (2, 32)
          # with block_size=32 -> 1 block per row -> scale shape (2, 1).
          scale=np.array([[0.1], [0.5]], dtype=np.float32),
          expected_quantized_dimension=1,
      ),
      dict(
          testcase_name="multiaxis",
          quantized_dimensions=(0, 1),
          weight_shape=(2, 3, 1, 64),
          # Weight shape (2, 3, 1, 64) with block_size=32 -> 2 blocks per row
          # -> scale shape (2, 3, 1, 2).
          scale=np.array(
              [
                  [[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.6]]],
                  [[[0.7, 0.8]], [[0.9, 1.0]], [[1.1, 1.2]]],
              ],
              dtype=np.float32,
          ),
          expected_quantized_dimension=None,
      ),
  )
  def test_get_tensor_quant_params_success_with_blockwise_dequantized_weights(
      self,
      quantized_dimensions,
      weight_shape,
      scale,
      expected_quantized_dimension,
  ):
    block_size = 32
    # Generate dummy quantized weights: a deterministic pattern of length 32
    # (block_size), repeated to fill `weight_shape`.
    pattern = np.array([1, -2, 3, -4] * 8, dtype=np.float32)
    quantized_weights = np.resize(pattern, weight_shape)
    dequant_vals = _construct_blockwise_dequantized_weights(
        quantized_weights,
        scale,
        quantized_weights.ndim - 1,  # Blocks are along the last dimension.
        block_size,
    )
    tensor_quant_config = qtyping.TensorQuantizationConfig(
        num_bits=4,
        granularity=qtyping.QuantGranularity.BLOCKWISE_32,
        quantized_dimensions=quantized_dimensions,
    )
    op_info = qtyping.OpInfo(
        op=None,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=0,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=tensor_quant_config
        ),
    )
    tensor_quant_params = dequantized_weight_recovery.get_tensor_quant_params(
        op_info, tensor_quant_config, dequant_vals
    )

    with self.subTest(name="metadata_match"):
      self.assertEqual(
          tensor_quant_params.quantized_dimension, expected_quantized_dimension
      )
      self.assertEqual(
          tensor_quant_params.quantized_dimensions, quantized_dimensions
      )
      self.assertEqual(tensor_quant_params.block_size, block_size)

    with self.subTest(name="scale_match"):
      recovered_scale = tensor_quant_params.scale
      self.assertEqual(recovered_scale.shape, scale.shape)
      self.assertSequenceAlmostEqual(
          recovered_scale.flatten(), scale.flatten(), places=5
      )

    with self.subTest(name="zero_point_match"):
      recovered_zp = tensor_quant_params.zero_point
      self.assertEqual(np.sum(recovered_zp), 0)
      self.assertEqual(recovered_zp.shape, scale.shape)

  def test_get_tensor_quant_params_success_with_qsv(self):

    # Fall back to naive_min_max_quantize.py for non-weight tensors.
    tensor_quant_params = dequantized_weight_recovery.get_tensor_quant_params(
        self._dummy_op_info,
        tensor_quant_config=qtyping.TensorQuantizationConfig(
            num_bits=8,
            granularity=qtyping.QuantGranularity.TENSORWISE,
        ),
        tensor_qsv={
            "min": np.array([-1]),
            "max": np.array([1]),
        },
    )

    self.assertIsNone(tensor_quant_params.quantized_dimension)
    recovered_scale = tensor_quant_params.scale
    self.assertEqual(recovered_scale.shape, (1,))
    self.assertSequenceAlmostEqual(recovered_scale.flatten(), [1 / 127])

    # Zero point should be zero for symmetric quantization.
    recovered_zp = tensor_quant_params.zero_point
    self.assertEqual(np.sum(recovered_zp), 0)
    self.assertEqual(recovered_zp.shape, (1,))

  @parameterized.named_parameters(
      dict(
          testcase_name="recovery_on_wrong_dimension",
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=4,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          scale=np.array([0.003, 1.234, 12.65, 2.24e-4]).reshape(1, 4),
          expected_error_pattern=(
              "Max unique values in any group is 4 \\(limit: 16\\)"
          ),
      ),
      dict(
          testcase_name="tensor_recovery_for_channel_quantization",
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=4,
              granularity=qtyping.QuantGranularity.TENSORWISE,
          ),
          scale=np.array([0.1875, 1e-2, 12.3]).reshape(3, 1),
          expected_error_pattern=(
              "Max unique values in any group is 12 \\(limit: 16\\)"
          ),
      ),
      dict(
          testcase_name="insufficient_bits",
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=2,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          scale=np.array([0.1875, 1e-2, 12.3]).reshape(3, 1),
          expected_error_pattern=(
              "Max unique values in any group is 4 \\(limit: 4\\)"
          ),
      ),
      dict(
          testcase_name="insufficient_bits_multiaxis",
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=2,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
              quantized_dimensions=[0, 1],
          ),
          scale=np.array([0.1875, 1e-2, 12.3]).reshape(3, 1, 1),
          expected_error_pattern=(
              "Max unique values in any group is 4 \\(limit: 4\\)"
          ),
      ),
      dict(
          testcase_name="exceed_limit_tensorwise",
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=2,
              granularity=qtyping.QuantGranularity.TENSORWISE,
          ),
          scale=np.array([1.0, 1.0, 1.0]).reshape(3, 1),
          expected_error_pattern=(
              "Detected a quantization group with 11 unique values, which"
              " exceeds the limit of 4"
          ),
      ),
  )
  def test_get_tensor_quant_params_raises_error_big_recovery_error(
      self, tensor_quant_config, scale, expected_error_pattern
  ):
    op_info = qtyping.OpInfo(
        op=None,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=0,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=tensor_quant_config
        ),
    )
    dequant_vals = scale * self._dummy_quantized_weights
    with self.assertRaisesRegex(RuntimeError, expected_error_pattern):
      dequantized_weight_recovery.get_tensor_quant_params(
          op_info, tensor_quant_config, dequant_vals
      )

  def test_get_tensor_transformation_params_wrapper_adds_tensor_name_on_error(
      self,
  ):
    class DummyTensor:

      def __init__(self, name, buffer_idx, type_code):
        self.name = name.encode("utf-8")
        self.buffer = buffer_idx
        self.type = type_code
        self.shape = (3, 4)

    class DummyBuffer:

      def __init__(self, data):
        self.data = data

    tensor_name = "test_tensor_fail_recovery"
    float32_code = tfl_flatbuffer_utils.TENSOR_TYPE_TO_CODE["FLOAT32"]
    tensor = DummyTensor(tensor_name, 0, float32_code)

    # Use deterministic data that will fail recovery.
    deterministic_data = np.arange(12, dtype=np.float32).reshape(3, 4) * 0.1
    buffers = [DummyBuffer(deterministic_data.tobytes())]
    graph_info = qtyping.GraphInfo(subgraph_tensors=[tensor], buffers=buffers)

    tensor_quant_config = qtyping.TensorQuantizationConfig(
        num_bits=2,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
    )
    op_info = qtyping.OpInfo(
        op=self._dummy_op_info.op,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=0,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=tensor_quant_config,
        ),
    )

    tensor_quant_params_cache = common_utils.TensorQuantParamsCache()

    expected_pattern = (
        f"Failed to get quantization parameters for tensor: {tensor_name}. "
        "Error: Failed to recover weights. "
        "Original error: Failed to recover the original quantized values"
    )

    with self.assertRaisesRegex(ValueError, expected_pattern):
      common_utils._get_tensor_transformation_params_wrapper(
          tensor=tensor,
          is_inbounding_tensor=True,
          op_info=op_info,
          graph_info=graph_info,
          tensor_name_to_qsv={},
          get_tensor_quant_params_fn=dequantized_weight_recovery.get_tensor_quant_params,
          tensor_quant_params_cache=tensor_quant_params_cache,
      )


if __name__ == "__main__":
  absltest.main()
