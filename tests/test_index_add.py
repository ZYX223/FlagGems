# Copyright 2026 FlagOS Contributors
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

import importlib
import random
import time

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    DIM_LIST = [1]
else:
    DIM_LIST = [0, 1]

random.seed(time.time() // 100)


CONTIGUOUS_SUFFIX_CASES = [
    ((1024, 4), 0),
    ((1, 2048, 8), 1),
    ((2, 8, 2048, 16), 2),
    ((2, 8, 2048, 32), 2),
    ((1024, 64), 0),
]


def _make_repeated_index(index_len):
    index_range = max(index_len // 2, 1)
    return torch.arange(index_len, device=flag_gems.device) % index_range


def _run_flag_gems_index_add(inp, dim, index, src, inplace, alpha=1):
    if inplace:
        result = flag_gems.index_add_(inp, dim, index, src, alpha=alpha)
        assert result is inp
        return result
    return flag_gems.index_add(inp, dim, index, src, alpha=alpha)


def _get_active_index_add_module():
    module = importlib.import_module(flag_gems.index_add.__module__)
    if module.index_add is not flag_gems.index_add:
        raise AssertionError("resolved a duplicate index_add backend module")
    return module


def _get_active_generated_index_add_module():
    module = _get_active_index_add_module()
    if not hasattr(module, "_index_add_func"):
        pytest.skip("active backend does not use the generated index_add kernel")
    return module


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
def test_generated_index_add_reuses_rank_wrapper_on_noncurrent_device(inplace):
    """One rank-specialized wrapper must safely launch on every operand device."""
    if torch_device_fn.device_count() < 2:
        pytest.skip("requires two visible devices")

    module = _get_active_generated_index_add_module()
    module._index_add_func.overloads.clear()
    original_device = torch_device_fn.current_device()
    device_type = torch.device(flag_gems.device).type

    try:
        torch_device_fn.set_device(0)
        for target_device_index in (0, 1):
            target_device = torch.device(device_type, target_device_index)
            inp = torch.zeros(
                (1, 64, 1), device=target_device, dtype=torch.float32
            )
            index_len = 32
            src = torch.ones(
                (1, index_len, 1), device=target_device, dtype=torch.float32
            )
            index = torch.arange(index_len, device=target_device)

            result = _run_flag_gems_index_add(inp, 1, index, src, inplace)
            consumed = result + 1.0

            expected = torch.ones((1, 64, 1), dtype=torch.float32)
            expected[:, :index_len, :] = 2.0
            torch.testing.assert_close(consumed.cpu(), expected, rtol=0, atol=0)
            assert torch_device_fn.current_device() == 0

        assert tuple(module._index_add_func.overloads) == ("3",)
    finally:
        module._index_add_func.overloads.clear()
        torch_device_fn.set_device(original_device)


@pytest.mark.index_add
@pytest.mark.index_add_
def test_generated_index_add_offsets_are_int64():
    """Large tensor metadata must not truncate in generated address arithmetic."""
    module = _get_active_generated_index_add_module()
    code = module.generate_code(
        (None, None, torch.empty((1, 2, 3))),
        "_index_add_wrapper",
        "_index_add_jit_function",
        module.IndentedBuffer(),
    ).getvalue()

    for parameter in (
        "N",
        "inp_numel",
        "inp_stride_dim",
        "inp_shape_dim",
        "src_shape_dim",
        "delta",
        "src_stride_0",
        "src_shape_0",
    ):
        assert f"{parameter}: tl.int64" in code
    assert "logical_offsets = offsets" in code
    assert "pre_idx = (logical_offsets // pre_cal)" in code
    assert "dim_idx = (logical_offsets % pre_cal // inp_stride_dim)" in code
    assert "input_idx = (logical_offsets +" in code


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_noncontiguous_source_uses_logical_offsets(inplace):
    """Destination coordinates come from logical, not storage, source offsets."""
    inp = torch.zeros((1, 64, 8), dtype=torch.float32, device=flag_gems.device)
    source_backing = torch.randn(
        (1, 32, 24), dtype=torch.float32, device=flag_gems.device
    )
    src = source_backing[:, :, 8:16]
    index = torch.arange(32, device=flag_gems.device) * 2
    assert src.shape == (1, 32, 8)
    assert src.stride() == (32 * 24, 24, 1)
    assert src.storage_offset() == 8
    assert not src.is_contiguous()

    reference_inp = utils.to_reference(inp)
    reference = torch.index_add(
        reference_inp,
        1,
        utils.to_reference(index),
        utils.to_reference(src),
    )
    result = _run_flag_gems_index_add(inp, 1, index, src, inplace)

    torch.testing.assert_close(result, reference, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon",
    reason="Hygon multi-device autotuning must not capture HIP Graphs",
)
@pytest.mark.index_add
@pytest.mark.index_add_
def test_hygon_index_add_autotuners_use_event_timing():
    module = _get_active_index_add_module()
    for kernel_name in (
        "_index_add_contiguous_suffix_flat_kernel",
        "_index_add_contiguous_suffix_fp16_flat_kernel",
        "_index_add_contiguous_suffix_tile_kernel",
    ):
        tuner = getattr(module, kernel_name).fn
        protocol = tuner.benchmark_protocol
        assert protocol.requested_mode.value == "event"
        assert protocol.resolved_mode.value == "event"
        assert protocol.implementation == "triton_do_bench"


@pytest.mark.index_add
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_add(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max
    index = torch.randperm(index_len, device=flag_gems.device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_out = torch.index_add(ref_inp, dim, ref_index, ref_src, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.index_add(inp, dim, index, src, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype=dtype, reduce_dim=dim)


@pytest.mark.index_add
@pytest.mark.parametrize("shape, dim", CONTIGUOUS_SUFFIX_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_add_contiguous_suffix(shape, dim, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    index = _make_repeated_index(inp.size(dim))
    src = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_out = torch.index_add(ref_inp, dim, ref_index, ref_src, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.index_add(inp, dim, index, src, alpha=alpha)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.index_add_
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_add_(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max
    index = torch.randperm(index_len, device=flag_gems.device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_inp.index_add_(dim, ref_index, ref_src, alpha=alpha)
    with flag_gems.use_gems():
        inp.index_add_(dim, index, src, alpha=alpha)

    utils.gems_assert_close(inp, ref_inp, dtype=dtype, reduce_dim=dim)


@pytest.mark.index_add_
@pytest.mark.parametrize("shape, dim", CONTIGUOUS_SUFFIX_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_add_inplace_contiguous_suffix(shape, dim, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    index = _make_repeated_index(inp.size(dim))
    src = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_inp.index_add_(dim, ref_index, ref_src, alpha=alpha)
    with flag_gems.use_gems():
        inp.index_add_(dim, index, src, alpha=alpha)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_invalid_index(inplace):
    shape = (2, 4, 8)
    dim = 1
    inp = torch.zeros(shape, device=flag_gems.device)
    src = torch.ones((2, 2, 8), device=flag_gems.device)
    index = torch.tensor([0, shape[dim]], device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with (
        flag_gems.use_gems(),
        pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"),
    ):
        if inplace:
            inp.index_add_(dim, index, src)
        else:
            torch.index_add(inp, dim, index, src)

    utils.gems_assert_equal(inp, ref_inp)
