# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os

import pytest
import torch
import torch_npu
import triton
import triton.backends.ascend.runtime
import triton.backends.ascend.runtime.autotuner as ascend_autotuner
import triton.language as tl


@triton.autotune(
    configs=[],
    key={"x": "n_elements"},
    hints={
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {"x": "BLOCK_SIZE_SUB"},
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    }
)
@triton.jit
def add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_SIZE_SUB: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK_SIZE
    loops1 = (BLOCK_SIZE + BLOCK_SIZE_SUB - 1) // BLOCK_SIZE_SUB
    for loop in range(0, loops1):
        x0 = offset + loop * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE_SUB)
        mask = x0 < n_elements
        x = tl.load(x_ptr + x0, mask)
        y = tl.load(y_ptr + x0, mask)
        output = x + y
        tl.store(output_ptr + x0, output)


def add_torch(x, y):
    return x + y


def add_autotune(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    add_kernel[lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)](x, y, output, n_elements)
    return output


@pytest.mark.autotune
@pytest.mark.parametrize('size', [2048, ])
def test_add(size: int):
    x = torch.rand(size, device="npu")
    y = torch.rand(size, device="npu")

    output_torch = add_torch(x, y)
    output_triton = add_autotune(x, y)
    assert torch.allclose(output_triton, output_torch)


@pytest.mark.autotune
def test_add_no_reduction_axes():
    try:
        @triton.autotune(
            configs=[],
            key={"x": "n_elements"},
            hints={
                "split_params": {"x": "BLOCK_SIZE"},
                "tiling_params": {"x": "BLOCK_SIZE_SUB"},
                "low_dim_axes": ["x"],
            }
        )
        @triton.jit
        def add_kernel_exception():
            pass
    except ValueError as e:
        assert "reduction_axes must be a list" in str(e)


@pytest.mark.autotune
def test_add_no_low_dim_axes():
    try:
        @triton.autotune(
            configs=[],
            key={"x": "n_elements"},
            hints={
                "split_params": {"x": "BLOCK_SIZE"},
                "tiling_params": {"x": "BLOCK_SIZE_SUB"},
                "reduction_axes": [],
            }
        )
        @triton.jit
        def add_kernel_exception():
            pass
    except ValueError as e:
        assert "low_dim_axes must be a list" in str(e)


@pytest.mark.autotune
def test_add_no_tiling_params():
    try:
        @triton.autotune(
            configs=[],
            key={"x": "n_elements"},
            hints={
                "split_params": {"x": "BLOCK_SIZE"},
                "low_dim_axes": ["x"],
                "reduction_axes": [],
            }
        )
        @triton.jit
        def add_kernel_exception():
            pass
    except ValueError as e:
        assert "tiling_params must be a dict" in str(e)


@pytest.mark.autotune
def test_add_no_split_params():
    try:
        @triton.autotune(
            configs=[],
            key={"x": "n_elements"},
            hints={
                "tiling_params": {"x": "BLOCK_SIZE_SUB"},
                "low_dim_axes": ["x"],
                "reduction_axes": [],
            }
        )
        @triton.jit
        def add_kernel_exception():
            pass
    except ValueError as e:
        assert "split_params must be a dict" in str(e)


@pytest.mark.autotune
def test_add_no_keyname():
    try:
        @triton.autotune(
            configs=[],
            key={"x0": "n_elements"},
            hints={
                "tiling_params": {"x": "BLOCK_SIZE_SUB"},
                "low_dim_axes": ["x"],
                "reduction_axes": [],
            }
        )
        @triton.jit
        def add_kernel_exception():
            pass
    except ValueError as e:
        assert "All keys in 'key' must be valid axis names" in str(e)


def test_expand_explicit_configs_with_hints():
    base_configs = [
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
    ]

    @triton.autotune(
        configs=base_configs,
        key=["n_elements"],
        hints={
            "GROUP_M": [2, 4, 8],
        }
    )
    @triton.jit
    def kernel_with_group_hint(
            x_ptr,
            output_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
            GROUP_M: tl.constexpr,
    ):
        pass

    assert len(kernel_with_group_hint.configs) == 12
    actual_triplets = {
        (cfg.kwargs["BLOCK_SIZE"], cfg.kwargs["GROUP_M"], cfg.num_stages)
        for cfg in kernel_with_group_hint.configs
    }
    expected_triplets = {
        (128, 2, 1), (128, 2, 2),
        (128, 4, 1), (128, 4, 2),
        (128, 8, 1), (128, 8, 2),
        (256, 2, 1), (256, 2, 2),
        (256, 4, 1), (256, 4, 2),
        (256, 8, 1), (256, 8, 2),
    }
    assert actual_triplets == expected_triplets


def test_expand_hints_multibuffer_maps_to_num_stages():
    @triton.autotune(
        configs=[triton.Config({"BLOCK_SIZE": 128})],
        key=["n_elements"],
        hints={
            "GROUP_M": [2],
            "multibuffer": [True, False],
        }
    )
    @triton.jit
    def kernel_with_multibuffer_alias(
            x_ptr,
            output_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
            GROUP_M: tl.constexpr,
    ):
        pass

    assert len(kernel_with_multibuffer_alias.configs) == 2
    assert {cfg.num_stages for cfg in kernel_with_multibuffer_alias.configs} == {1, 2}
    assert all("multibuffer" not in cfg.kwargs for cfg in kernel_with_multibuffer_alias.configs)
    assert kernel_with_multibuffer_alias.config_hints == {
        "GROUP_M": [2],
        "num_stages": [2, 1],
    }


def test_expand_hints_explicit_num_stages_precedes_multibuffer():
    @triton.autotune(
        configs=[triton.Config({"BLOCK_SIZE": 128})],
        key=["n_elements"],
        hints={
            "GROUP_M": [2, 4],
            "num_stages": [1],
            "multibuffer": [True, False],
        }
    )
    @triton.jit
    def kernel_num_stages_precedence(
            x_ptr,
            output_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
            GROUP_M: tl.constexpr,
    ):
        pass

    assert len(kernel_num_stages_precedence.configs) == 2
    assert {cfg.num_stages for cfg in kernel_num_stages_precedence.configs} == {1}
    assert kernel_num_stages_precedence.config_hints == {
        "GROUP_M": [2, 4],
        "num_stages": [1],
    }


def test_expand_explicit_configs_with_mixed_hints():
    base_configs = [
        triton.Config({"BLOCK_SIZE": 128}),
    ]

    @triton.autotune(
        configs=base_configs,
        key=["n_elements"],
        hints={
            "GROUP_M": [2, 4],
            "num_stages": [1, 2],
            "enable_ubuf_saving": [True, False],
        }
    )
    @triton.jit
    def kernel_with_mixed_hints(
            x_ptr,
            output_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
            GROUP_M: tl.constexpr,
    ):
        pass

    assert len(kernel_with_mixed_hints.configs) == 8
    assert {cfg.num_stages for cfg in kernel_with_mixed_hints.configs} == {1, 2}
    assert {cfg.kwargs["GROUP_M"] for cfg in kernel_with_mixed_hints.configs} == {2, 4}
    assert {cfg.kwargs["enable_ubuf_saving"] for cfg in kernel_with_mixed_hints.configs} == {True, False}


def test_expand_hints_coexist_with_axis_hints():
    base_configs = [
        triton.Config({"BLOCK_SIZE": 128, "BLOCK_SIZE_SUB": 32}),
        triton.Config({"BLOCK_SIZE": 256, "BLOCK_SIZE_SUB": 64}),
    ]

    @triton.autotune(
        configs=base_configs,
        key={"x": "n_elements"},
        hints={
            "split_params": {"x": "BLOCK_SIZE"},
            "tiling_params": {"x": "BLOCK_SIZE_SUB"},
            "low_dim_axes": ["x"],
            "reduction_axes": [],
            "GROUP_M": [2, 4],
        }
    )
    @triton.jit
    def kernel_with_axis_hints(
            x_ptr,
            output_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
            BLOCK_SIZE_SUB: tl.constexpr,
            GROUP_M: tl.constexpr,
    ):
        pass

    assert len(kernel_with_axis_hints.configs) == 8
    assert {cfg.num_stages for cfg in kernel_with_axis_hints.configs} == {1, 2}
    assert kernel_with_axis_hints.hints == {
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {"x": "BLOCK_SIZE_SUB"},
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    }
    assert kernel_with_axis_hints.config_hints == {
        "GROUP_M": [2, 4],
        "num_stages": [1, 2],
    }


def test_expand_hints_invalid_key():
    with pytest.raises(ValueError, match="Invalid hints keys for config expansion"):
        @triton.autotune(
            configs=[triton.Config({"BLOCK_SIZE": 128})],
            key=["n_elements"],
            hints={
                "INVALID_KEY": [1, 2],
            }
        )
        @triton.jit
        def kernel_invalid_hint_key(
                x_ptr,
                output_ptr,
                n_elements,
                BLOCK_SIZE: tl.constexpr,
        ):
            pass


def test_expand_hints_invalid_value_container():
    with pytest.raises(ValueError, match="must be a non-empty list/tuple"):
        @triton.autotune(
            configs=[triton.Config({"BLOCK_SIZE": 128})],
            key=["n_elements"],
            hints={
                "GROUP_M": 4,
            }
        )
        @triton.jit
        def kernel_invalid_hint_value(
                x_ptr,
                output_ptr,
                n_elements,
                BLOCK_SIZE: tl.constexpr,
                GROUP_M: tl.constexpr,
        ):
            pass


def test_expand_hints_invalid_multibuffer_values():
    with pytest.raises(ValueError, match="must contain only boolean values"):
        @triton.autotune(
            configs=[triton.Config({"BLOCK_SIZE": 128})],
            key=["n_elements"],
            hints={
                "multibuffer": [1, 0],
            }
        )
        @triton.jit
        def kernel_invalid_multibuffer_hint(
                x_ptr,
                output_ptr,
                n_elements,
                BLOCK_SIZE: tl.constexpr,
        ):
            pass


def test_expand_hints_invalid_compile_option_value():
    with pytest.raises(ValueError, match="Invalid value for 'num_stages'"):
        @triton.autotune(
            configs=[triton.Config({"BLOCK_SIZE": 128})],
            key=["n_elements"],
            hints={
                "num_stages": [3],
            }
        )
        @triton.jit
        def kernel_invalid_compile_hint(
                x_ptr,
                output_ptr,
                n_elements,
                BLOCK_SIZE: tl.constexpr,
        ):
            pass


def test_expand_hints_require_explicit_configs():
    with pytest.raises(ValueError, match="Config expansion hints require explicit configs"):
        @triton.autotune(
            configs=[],
            key=["n_elements"],
            hints={
                "GROUP_M": [2, 4],
            }
        )
        @triton.jit
        def kernel_require_configs(
                x_ptr,
                output_ptr,
                n_elements,
                GROUP_M: tl.constexpr,
        ):
            pass


def test_non_simt_num_stages_candidates_priority():
    tuner = object.__new__(ascend_autotuner.AutoTilingTuner)
    tuner.user_specified_num_stages = None
    tuner.user_specified_multibuffer = None

    assert tuner._get_non_simt_num_stages_candidates() == [1, 2]

    tuner.user_specified_multibuffer = True
    assert tuner._get_non_simt_num_stages_candidates() == [2]

    tuner.user_specified_num_stages = 1
    assert tuner._get_non_simt_num_stages_candidates() == [1]


def test_expand_simt_num_warps_configs_default_candidates():
    tuner = object.__new__(ascend_autotuner.AutoTilingTuner)
    tuner.user_specified_warps = None
    tuner.print_autotuning = False

    expanded_configs = tuner._expand_simt_num_warps_configs(
        [triton.Config({"BLOCK_SIZE": 128})]
    )

    assert len(expanded_configs) == 4
    assert [cfg.num_warps for cfg in expanded_configs] == [8, 16, 32, 64]
