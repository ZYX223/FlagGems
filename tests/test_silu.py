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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


SILU_ASCEND_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
SILU_EDGE_CASES = ["tail", "non_contiguous", "empty", "special_values"]
ASCEND_ONLY = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend-only SiLU coverage"
)


def _make_silu_edge_input(case, dtype):
    if case == "tail":
        return torch.linspace(
            -8.0,
            8.0,
            4097,
            dtype=torch.float32,
            device=flag_gems.device,
        ).to(dtype)
    if case == "non_contiguous":
        return torch.randn(
            (257, 259), dtype=dtype, device=flag_gems.device
        ).transpose(0, 1)
    if case == "empty":
        return torch.empty((0, 17), dtype=dtype, device=flag_gems.device)
    if case == "special_values":
        return torch.tensor(
            [
                float("-inf"),
                -20.0,
                -1.0,
                -0.0,
                0.0,
                1.0,
                20.0,
                float("inf"),
                float("nan"),
            ],
            dtype=dtype,
            device=flag_gems.device,
        )
    raise AssertionError(f"unsupported SiLU edge case: {case}")


@pytest.mark.silu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_silu(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.nn.functional.silu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.silu(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.silu
@pytest.mark.silu_
@ASCEND_ONLY
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("case", SILU_EDGE_CASES)
@pytest.mark.parametrize("dtype", SILU_ASCEND_DTYPES)
def test_silu_edge_cases(case, dtype, inplace):
    res_inp = _make_silu_edge_input(case, dtype)
    ref_inp = utils.to_reference(res_inp.clone(), True)
    original_stride = res_inp.stride()
    original_storage_ptr = res_inp.untyped_storage().data_ptr()
    native_out = None
    if case == "non_contiguous" and not inplace:
        native_out = torch.nn.functional.silu(res_inp)

    ref_out = torch.nn.functional.silu(ref_inp, inplace=inplace)
    selected_ops = ["silu_"] if inplace else ["silu"]
    with flag_gems.use_gems(include=selected_ops):
        res_out = torch.nn.functional.silu(res_inp, inplace=inplace)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    if native_out is not None:
        assert res_out.layout == native_out.layout
        assert res_out.stride() == native_out.stride()
        assert res_out.is_contiguous() == native_out.is_contiguous()
    if inplace:
        assert res_out.untyped_storage().data_ptr() == original_storage_ptr
        assert res_out.stride() == original_stride


@pytest.mark.silu
@pytest.mark.silu_backward
@ASCEND_ONLY
@pytest.mark.parametrize("dtype", SILU_ASCEND_DTYPES)
def test_silu_forward_autograd(dtype):
    res_inp = torch.linspace(
        -4.0,
        4.0,
        4097,
        dtype=torch.float32,
        device=flag_gems.device,
    ).to(dtype)
    res_inp.requires_grad_(True)
    res_grad_out = torch.linspace(
        0.25,
        1.25,
        4097,
        dtype=torch.float32,
        device=flag_gems.device,
    ).to(dtype)
    ref_inp = utils.to_reference(res_inp.detach(), True).requires_grad_(True)
    ref_grad_out = utils.to_reference(res_grad_out, True)

    ref_out = torch.nn.functional.silu(ref_inp)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad_out)
    with flag_gems.use_gems(include=["silu", "silu_backward"]):
        res_out = torch.nn.functional.silu(res_inp)
        (res_grad,) = torch.autograd.grad(res_out, res_inp, res_grad_out)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.silu_
@ASCEND_ONLY
@pytest.mark.parametrize("dtype", SILU_ASCEND_DTYPES)
def test_silu_inplace_leaf_autograd_error(dtype):
    res_inp = torch.randn(
        (4097,), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_inp = utils.to_reference(res_inp.detach(), True).requires_grad_(True)

    with pytest.raises(RuntimeError):
        torch.nn.functional.silu(ref_inp, inplace=True)
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems(include=["silu_"]):
            torch.nn.functional.silu(res_inp, inplace=True)


@pytest.mark.silu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_silu_(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp.clone(), True)

    ref_out = torch.nn.functional.silu(ref_inp, inplace=True)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.silu(res_inp, inplace=True)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.silu_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_silu_backward(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)

    ref_in_grad = torch.ops.aten.silu_backward(ref_grad, ref_inp)
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.silu_backward(res_grad, res_inp)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
