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

"""Recovers quantized weights from dequantized weights (often from QAT)."""

from collections.abc import Sequence
import dataclasses
from typing import Any, Optional

import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from litert_quantizer.algorithms.utils import common_utils

ALGORITHM_KEY = "dequantized_weight_recovery"
_TFLOpName = qtyping.TFLOperationName
_QuantTransformation = qtyping.QuantTransformation
_IntType = uniform_quantize_tensor.IntType


def _validate_recovered_weights(
    original_vals: np.ndarray,
    quant_vals: np.ndarray,
    quant_params: qtyping.UniformQuantParams,
    tol: float = 1e-4,
):
  """Validates if requantized weights are close enough to the original ones.

  Args:
    original_vals: Original values before quantization.
    quant_vals: Quantized values.
    quant_params: Quantization parameters.
    tol: Tolerance for the difference between original and recovered values.

  Raises:
    RuntimeError: If the maximum difference between original and recovered
    values exceeds the tolerance.
  """

  recovered_vals = uniform_quantize_tensor.uniform_dequantize(
      quant_vals, quant_params
  )
  diff = np.ravel(np.abs(recovered_vals - original_vals))
  max_diff = diff.max()
  if max_diff > tol:
    raise RuntimeError(
        "Failed to recover the original quantized values from dequantized"
        f" values. Max diff between recovered and original values: {max_diff}"
        f" (tolerance: {tol})"
    )


def _compute_scale_from_diffs(
    diffs: np.ndarray, min_scale: float
) -> np.ndarray:
  """Computes scale from diffs, using median for dense and min for sparse."""
  tolerance = 1e-9
  num_rows, num_cols = diffs.shape
  if num_rows == 0 or num_cols == 0:
    return np.full(num_rows, min_scale, dtype=diffs.dtype)

  # np.nanmin / np.nanmedian along axis 1 fall back to a Python-level loop per
  # row, which takes minutes for the ~8M blocks of a large MoE expert tensor.
  # Sorting each chunk of rows once and reading the min / median by index
  # gives the same values in one vectorized pass: gaps that are not positive
  # sort last as +inf and are excluded through the per-row count.
  min_diffs = np.empty(num_rows, dtype=diffs.dtype)
  median_diffs = np.empty(num_rows, dtype=diffs.dtype)
  chunk_rows = max(1, (1 << 22) // max(num_cols, 1))
  for start in range(0, num_rows, chunk_rows):
    end = min(start + chunk_rows, num_rows)
    chunk = diffs[start:end]
    valid = chunk > tolerance
    cnt = np.count_nonzero(valid, axis=1)
    sorted_chunk = np.sort(np.where(valid, chunk, np.inf), axis=1)
    rows = np.arange(end - start)
    lo = np.maximum(0, (cnt - 1) // 2)
    hi = np.maximum(0, cnt // 2)
    min_diffs[start:end] = np.where(cnt > 0, sorted_chunk[:, 0], min_scale)
    median_diffs[start:end] = np.where(
        cnt > 0,
        0.5 * (sorted_chunk[rows, lo] + sorted_chunk[rows, hi]),
        min_scale,
    )

  # If median is significantly larger than min (> 1.35x), intermediate levels
  # are missing (sparse lattice, e.g. in small channels or unit tests).
  # In that case, min_diffs is the true fundamental grid step.
  # Otherwise, variations are just floating-point rounding noise around the
  # same step, so median_diffs cancels the noise and rejects single-element
  # outliers.
  use_min = median_diffs > 1.35 * min_diffs
  chosen_diffs = np.where(use_min, min_diffs, median_diffs)
  return np.maximum(chosen_diffs, min_scale)


def _refit_scales(
    abs_vals_2d: np.ndarray,
    scales_1d: np.ndarray,
    min_scale: float,
    *,
    num_bits: int | None = None,
    max_iters: int = 2,
    max_rel_change: float = 0.25,
    chunk_elements: int = 1 << 22,
) -> np.ndarray:
  """Refines per-group scales with a closed-form least-squares fit.

  `_compute_scale_from_diffs` derives the scale from a single gap between
  adjacent levels. That is unbiased but noisy, because the dequantized weights
  are usually stored at reduced precision (e.g. bf16 widened to fp32), which
  perturbs every level independently.

  Once the seed scale is close enough to assign each weight to the right integer
  level `q`, the scale that minimizes the reconstruction error has a closed
  form, and it averages that perturbation over the whole group:

      S = sum(q * |W|) / sum(q * q)

  Note this is computed on absolute values, which is equivalent to using signed
  values since `q` and `W` always share a sign.

  Args:
    abs_vals_2d: Absolute weight values, shaped (num_groups, group_size). Each
      row is one quantization group (channel, block, or the whole tensor).
    scales_1d: Seed scale per group, from `_compute_scale_from_diffs`.
    min_scale: Lower bound on the returned scales.
    num_bits: Bit width, used to bound the integer codes. `None` disables the
      bound.
    max_iters: Maximum refinement passes. Assigning codes and refitting is a
      fixpoint iteration; in practice it converges after the first pass, and the
      second only confirms it. Each pass is a single linear scan, cheap next to
      the sort that produced the seed.
    max_rel_change: Reject a refit that moves a scale by more than this fraction
      of the seed. A good seed moves by a few percent at most, so a large move
      means the seed was too far off to assign codes reliably, and the fit would
      be meaningless. In that case the seed is kept.
    chunk_elements: Target number of elements to materialize at a time.

  Returns:
    The refined scales, same shape and dtype as `scales_1d`.
  """
  num_groups, group_size = abs_vals_2d.shape
  if num_groups == 0 or group_size == 0:
    return scales_1d

  code_limit = None if num_bits is None else float(1 << (num_bits - 1))

  # Bound the size of each materialized block. Chunk over rows when groups are
  # small and numerous, and over columns when a single group is very large
  # (per-tensor quantization of a big weight).
  rows_per_chunk = max(1, min(num_groups, chunk_elements // max(group_size, 1)))
  cols_per_chunk = group_size
  if rows_per_chunk == 1 and group_size > chunk_elements:
    cols_per_chunk = chunk_elements

  scales = scales_1d.astype(np.float64, copy=True)
  for _ in range(max_iters):
    numerator = np.zeros(num_groups, dtype=np.float64)
    denominator = np.zeros(num_groups, dtype=np.float64)
    inv_scales = np.reciprocal(scales)

    for row_start in range(0, num_groups, rows_per_chunk):
      row_end = min(row_start + rows_per_chunk, num_groups)
      inv_block = inv_scales[row_start:row_end, np.newaxis]
      for col_start in range(0, group_size, cols_per_chunk):
        col_end = min(col_start + cols_per_chunk, group_size)
        block = abs_vals_2d[row_start:row_end, col_start:col_end]
        # Reuse the buffer: |W|/S becomes the rounded codes in place.
        codes = np.multiply(block, inv_block)
        np.rint(codes, out=codes)
        if code_limit is not None:
          np.minimum(codes, code_limit, out=codes)
        numerator[row_start:row_end] += np.einsum("ij,ij->i", codes, block)
        denominator[row_start:row_end] += np.einsum("ij,ij->i", codes, codes)

    # Groups that quantize entirely to zero carry no information; keep the seed.
    refit = np.divide(
        numerator, denominator, out=scales.copy(), where=denominator > 0
    )
    accept = np.abs(refit - scales) <= max_rel_change * scales
    refit = np.maximum(np.where(accept, refit, scales), min_scale)

    converged = np.allclose(refit, scales, rtol=1e-9, atol=0.0)
    scales = refit
    if converged:
      break

  return scales.astype(scales_1d.dtype, copy=False)


def _repair_sparse_groups(
    abs_vals_2d: np.ndarray,
    scales_1d: np.ndarray,
    min_scale: float,
    *,
    num_bits: int | None,
    max_rel_err: float = 0.05,
    chunk_elements: int = 1 << 22,
) -> np.ndarray:
  """Re-derives the scale of groups whose gap seed missed the true step.

  The gap seed assumes the smallest gap between adjacent levels is one
  quantization step. In a sparse group that only holds a few levels, every gap
  can be a multiple of the step (e.g. levels {0, 2s, 7s}: gaps 2s and 5s), so
  the seed, and the least-squares refit around it, reconstruct the group with
  an error of a sizeable fraction of a step. Such groups are rare (one in
  ~200k blocks of Gemma 4 26B) but fail the lossless validation.

  For groups whose reconstruction error exceeds `max_rel_err` of a step, the
  scale is re-derived as max|w| / k for the smallest k in [1, 2^(num_bits-1)]
  that reproduces every value within `max_rel_err` of a step. Any such k is a
  lossless representation; the smallest one keeps the largest scale.

  Args:
    abs_vals_2d: Absolute weight values, shaped (num_groups, group_size).
    scales_1d: Current scale per group.
    min_scale: Lower bound on the returned scales.
    num_bits: Bit width; `None` disables the repair.
    max_rel_err: Reconstruction error, as a fraction of a step, above which a
      group is repaired (and below which a candidate scale is accepted).
    chunk_elements: Target number of elements to materialize at a time.

  Returns:
    The scales with repaired groups, same shape and dtype as `scales_1d`.
  """
  if num_bits is None:
    return scales_1d
  num_groups, group_size = abs_vals_2d.shape
  if num_groups == 0 or group_size == 0:
    return scales_1d
  code_limit = float(1 << (num_bits - 1))
  rows_per_chunk = max(1, min(num_groups, chunk_elements // max(group_size, 1)))
  scales = scales_1d.astype(np.float64, copy=True)
  bad_rows = []
  for row_start in range(0, num_groups, rows_per_chunk):
    row_end = min(row_start + rows_per_chunk, num_groups)
    block = abs_vals_2d[row_start:row_end]
    chunk_scales = scales[row_start:row_end, np.newaxis]
    codes = np.rint(block / chunk_scales)
    np.minimum(codes, code_limit, out=codes)
    err = np.max(np.abs(codes * chunk_scales - block), axis=1)
    bad = np.flatnonzero(err > max_rel_err * scales[row_start:row_end])
    if bad.size:
      bad_rows.extend((bad + row_start).tolist())
  for row in bad_rows:
    group = np.asarray(abs_vals_2d[row], dtype=np.float64)
    group_max = float(group.max())
    if group_max <= 0.0:
      continue
    for k in range(1, int(code_limit) + 1):
      candidate = group_max / k
      codes = np.rint(group / candidate)
      if codes.max() > code_limit:
        continue
      if np.max(np.abs(codes * candidate - group)) <= max_rel_err * candidate:
        scales[row] = max(candidate, min_scale)
        break
  return scales.astype(scales_1d.dtype, copy=False)


def _get_scale(arr: np.ndarray, min_scale: float) -> float:
  """Helper function to calculate scale from a 1D array of absolute values."""
  # Reduce to unique values first: `arr` may be the whole tensor, and appending
  # to it before deduplicating would copy all of it.
  unique_vals = np.unique(arr)
  # Make sure the values include zero (symmetric quantization).
  if unique_vals.size and unique_vals[0] != 0:
    unique_vals = np.concatenate(
        (np.zeros(1, dtype=unique_vals.dtype), unique_vals)
    )
  if unique_vals.size > 1:
    diffs = np.diff(unique_vals)[np.newaxis, :]
    # Cast to float to ensure return type consistency.
    return float(_compute_scale_from_diffs(diffs, min_scale)[0])
  return min_scale


def _check_unique_values(
    tensor_content: np.ndarray,
    quantized_dimension: int | Sequence[int] | None,
    *,
    block_size: int,
    num_bits: int,
) -> tuple[int, int]:
  """Checks if the number of unique values in any quantization group exceeds the limit.

  Args:
    tensor_content: The tensor content to check.
    quantized_dimension: The dimension(s) along which the tensor is quantized.
    block_size: The block size for blockwise quantization.
    num_bits: The number of bits for quantization.

  Returns:
    A tuple (max_found, limit) where:
      max_found: The maximum number of unique values found in any group (may
        stop early if limit is exceeded).
      limit: The maximum allowed number of unique values (2^num_bits).
  """
  # The maximum number of unique values representable by the target bit width.
  limit = 1 << num_bits

  if quantized_dimension is None:
    # Per-tensor quantization: the entire tensor shares a single scale.
    return np.unique(tensor_content).size, limit

  if isinstance(quantized_dimension, int) and block_size == 0:
    # Single-axis channelwise quantization: check each slice along the axis.
    max_found = 0
    for i in range(tensor_content.shape[quantized_dimension]):
      slices: list[Any] = [slice(None)] * tensor_content.ndim
      slices[quantized_dimension] = i
      vec = tensor_content[tuple(slices)]
      num_unique = np.unique(vec).size
      max_found = max(max_found, num_unique)
      if max_found > limit:
        break
    return max_found, limit

  if isinstance(quantized_dimension, Sequence) and block_size == 0:
    # Multi-axis channelwise quantization: flatten the quantized dimensions
    # into rows so each row contains all elements of one quantization channel.
    num_channels = int(
        np.prod([tensor_content.shape[d] for d in quantized_dimension])
    )
    reshaped_2d = tensor_content.reshape(num_channels, -1)
  else:
    # Blockwise quantization: reshape so each row is one quantization block.
    reshaped_2d = common_utils.reshape_to_blocks(
        tensor_content, quantized_dimension, block_size
    )

  if reshaped_2d.size == 0:
    return 0, limit

  max_found = 0
  # Count distinct values on sorted chunks instead of np.unique per row,
  # which is a Python-level loop over millions of blocks.
  chunk_rows = max(1, (1 << 20) // max(reshaped_2d.shape[1], 1))
  for start in range(0, reshaped_2d.shape[0], chunk_rows):
    sorted_chunk = np.sort(reshaped_2d[start : start + chunk_rows], axis=1)
    row_unique = 1 + np.count_nonzero(
        sorted_chunk[:, 1:] != sorted_chunk[:, :-1], axis=1
    )
    max_found = max(max_found, int(row_unique.max()))
    if max_found > limit:
      break
  return max_found, limit


def get_zp_scale_from_dequantized_symmetric_weights(
    dequant_vals: np.ndarray,
    quantized_dimension: int | Sequence[int] | None = None,
    block_size: int = 0,
    min_scale: float = 1e-9,
    num_bits: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Calculates scale and zero point from dequantized and symmetric weights.

  Handles per-tensor, per-channel (axis), multi-axis, and blockwise
  quantization.

  The scale is seeded from the spacing between adjacent weight levels and then
  refined by a least-squares fit over every weight in the group. See
  `_refit_scales` for why the seed alone is not accurate enough.

  Args:
      dequant_vals: The dequantized weight values (numpy array).
      quantized_dimension: The dimension(s) along which quantization was
        performed (0, 1, a sequence for multi-axis), or None for per-tensor
        quantization.
      block_size: The block size for blockwise quantization.
      min_scale: The minimum allowed scale value.
      num_bits: The target bit width. Used to bound the integer codes while
        refining the scale. `None` disables the bound.

  Returns:
      A tuple containing:
          - zero_points: Zero points (all zeros for symmetric quantization).
          - scales: Scales.

  Raises:
      ValueError: If `quantized_dimension` is invalid.
  """

  # Use absolute values for symmetric quantization.
  dequant_vals = np.abs(dequant_vals)

  if quantized_dimension is None:
    # Per-tensor quantization: One scale for the entire tensor.
    flat_vals = np.ravel(dequant_vals)
    scales_1d = np.array([_get_scale(flat_vals, min_scale)])
    scales_1d = _refit_scales(
        flat_vals[np.newaxis, :], scales_1d, min_scale, num_bits=num_bits
    )
    scales = scales_1d.reshape(1, 1)
    return np.zeros_like(scales, dtype=np.int32), scales

  if not isinstance(quantized_dimension, Sequence) and (
      quantized_dimension not in (0, 1)
  ):
    raise ValueError(
        "quantized_dimension must be 0, 1, a sequence, or None. Got"
        f" {quantized_dimension}"
    )

  # Reshape `dequant_vals` into a 2D array `(num_groups, elements_per_group)` so
  # each row is one quantization group (channel or block) sharing a single
  # scale, and compute `target_shape` to reshape the recovered 1D scales back to
  # match `dequant_vals.ndim`.
  if block_size > 0:
    # Blockwise quantization: single and multi-axis.
    reshaped_2d = common_utils.reshape_to_blocks(
        dequant_vals, quantized_dimension, block_size
    )
    target_shape = common_utils.get_blockwise_shape(
        dequant_vals.shape, quantized_dimension, block_size
    )
  elif isinstance(quantized_dimension, Sequence):
    # Multi-axis channelwise quantization.
    num_channels = int(
        np.prod([dequant_vals.shape[d] for d in quantized_dimension])
    )
    reshaped_2d = dequant_vals.reshape(num_channels, -1)
    target_shape = tuple(
        dequant_vals.shape[d] if d in quantized_dimension else 1
        for d in range(dequant_vals.ndim)
    )
  else:
    # Per-channel: Move quantized_dimension to axis 0.
    perm = [quantized_dimension] + [
        i for i in range(dequant_vals.ndim) if i != quantized_dimension
    ]
    transposed = np.transpose(dequant_vals, perm)
    # Flatten all other dimensions.
    reshaped_2d = transposed.reshape(
        dequant_vals.shape[quantized_dimension], -1
    )
    target_shape = [1] * dequant_vals.ndim
    target_shape[quantized_dimension] = dequant_vals.shape[quantized_dimension]
    target_shape = tuple(target_shape)

  # Append 0 to each row to handle symmetric quantization.
  zeros = np.zeros((reshaped_2d.shape[0], 1), dtype=reshaped_2d.dtype)
  reshaped_2d = np.hstack([reshaped_2d, zeros])

  # Sort each row.
  sorted_2d = np.sort(reshaped_2d, axis=1)

  # Diff each row.
  diffs = np.diff(sorted_2d, axis=1)

  scales_1d = _compute_scale_from_diffs(diffs, min_scale)

  # Release the sort temporaries before refining, so the refit does not add
  # to peak memory.
  del sorted_2d, diffs

  scales_1d = _refit_scales(
      reshaped_2d, scales_1d, min_scale, num_bits=num_bits
  )
  scales_1d = _repair_sparse_groups(
      reshaped_2d, scales_1d, min_scale, num_bits=num_bits
  )

  scales = scales_1d.reshape(target_shape)

  # Zero points are all zeros for symmetric quantization.
  zero_points = np.zeros_like(scales, dtype=np.int32)
  return zero_points, scales


def get_tensor_quant_params(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_content: Optional[np.ndarray] = None,
    tensor_qsv: Optional[dict[str, Any]] = None,
) -> qtyping.UniformQuantParams:
  """Get the quantization parameters for a tensor.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    tensor_quant_config: The quantization config for the tensor.
    tensor_content: The content of the tensor.
    tensor_qsv: A dictionary containing the min/max of the tensor.

  Returns:
    The quantization parameters for the tensor.

  Raises:
    ValueError: If the tensor is not a symmetric weight tensor.
    RuntimeError: If weight recovery validation fails.
  """
  # Fallback to naive_min_max_quantize.py for non-weight tensors.
  if tensor_content is None:
    return naive_min_max_quantize.get_tensor_quant_params(
        op_info, tensor_quant_config, tensor_content, tensor_qsv
    )

  is_blockwise = uniform_quantize_tensor.is_blockwise(
      tensor_quant_config.granularity
  )
  block_size = 0
  if is_blockwise:
    block_size = uniform_quantize_tensor.extract_block_size_from_granularity(
        tensor_quant_config.granularity
    )

  if not tensor_quant_config.symmetric:
    raise ValueError(
        "Only symmetric weights are supported for dequantized weight recovery."
    )

  quantized_dim = common_utils.get_weight_quantized_dim(
      op_info, tensor_content, tensor_quant_config.granularity
  )
  zp, scale = get_zp_scale_from_dequantized_symmetric_weights(
      dequant_vals=tensor_content,
      quantized_dimension=quantized_dim,
      block_size=block_size,
      num_bits=tensor_quant_config.num_bits,
  )
  is_multi_axis = isinstance(quantized_dim, Sequence)
  quant_params = qtyping.UniformQuantParams(
      scale=scale,
      zero_point=zp,
      num_bits=tensor_quant_config.num_bits,
      symmetric=tensor_quant_config.symmetric,
      quantized_dimension=(quantized_dim if not is_multi_axis else None),
      quantized_dimensions=(quantized_dim if is_multi_axis else None),
      block_size=block_size,
  )
  quantized_vars = uniform_quantize_tensor.uniform_quantize(
      tensor_content, quant_params, is_blockwise_quant=is_blockwise
  )
  if not op_info.op_quant_config.skip_checks:
    try:
      _validate_recovered_weights(tensor_content, quantized_vars, quant_params)
    except RuntimeError as e:
      max_found, limit = _check_unique_values(
          tensor_content,
          quantized_dim,
          block_size=block_size,
          num_bits=tensor_quant_config.num_bits,
      )
      extra_msg = (
          (
              f"Detected a quantization group with {max_found} unique values, "
              f"which exceeds the limit of {limit} for"
              f" {tensor_quant_config.num_bits}-bit quantization. This suggests"
              " the input tensor is NOT dequantized (fake-quantized) weights."
              " Please verify if you are using a QAT checkpoint."
          )
          if max_found > limit
          else (
              f"Max unique values in any group is {max_found} (limit: {limit})."
              " The recovery failed despite reasonable unique value count."
              " Check if the weights are symmetric or if tolerance is too"
              " tight."
          )
      )
      msg = f"Failed to recover weights. Original error: {e}. {extra_msg}"
      raise RuntimeError(msg) from e
  return dataclasses.replace(quant_params, quantized_data=quantized_vars)


def calibrate(
    tfl_op: Any,
    graph_info: qtyping.GraphInfo,
    tensor_content_map: dict[str, np.ndarray],
    inputs_to_ignore: Optional[list[int]] = None,
    outputs_to_ignore: Optional[list[int]] = None,
) -> dict[str, qtyping.QSV]:
  """Collect quantization statistics variable (QSV, e.g., min/max) for the op.

  Args:
    tfl_op: The tfl operation.
    graph_info: Graph information needed to perform quantization for the op.
    tensor_content_map: A map of tensor name to tensor content.
    inputs_to_ignore: Input tensor indices to ignore.
    outputs_to_ignore: Output tensor indices to ignore.

  Returns:
    A dictionary with key as tensor name and value as the collected QSV.
  """
  # Reuse the min/max calibration algorithm from naive_min_max_quantize.py since
  # only weights need to be handled differently.
  return naive_min_max_quantize.min_max_calibrate(
      tfl_op,
      graph_info,
      tensor_content_map,
      inputs_to_ignore,
      outputs_to_ignore,
  )


def init_qsvs(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    inputs_to_ignore: Optional[list[int]] = None,
    outputs_to_ignore: Optional[list[int]] = None,
) -> qtyping.QSV:
  """Initialize the QSVs.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    inputs_to_ignore: Input tensor indices to ignore.
    outputs_to_ignore: Output tensor indices to ignore.

  Returns:
    QSVs.
  """
  # Reuse the min/max calibration algorithm from naive_min_max_quantize.py since
  # only weights need to be handled differently.
  return naive_min_max_quantize.init_qsvs(
      op_info, graph_info, inputs_to_ignore, outputs_to_ignore
  )
