#!/usr/bin/env python3
"""
Pytest tests for ubtuner decorator.

Tests cover:
1. UBTuner decorator detection and error handling
2. Decorator order validation (autotune outside ubtuner)
3. Two-chain kernel UB overflow scenarios
"""
import os
import shutil
import sys
import pytest

import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

import triton
import triton.language as tl

from triton.backends.ascend.runtime.ubtuner import ubtuner

# Set environment before imports
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

# Skip all tests if NPU is not available
pytestmark = pytest.mark.skipif(
    torch_npu is None or not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="requires torch_npu runtime"
)

BLOCK = int(os.environ.get("BLOCK", "20480"))


# =============================================================================
# Test: Decorator order validation
# =============================================================================
class TestDecoratorOrder:
    """Test that decorator order is correctly validated."""

    def test_autotune_outside_ubtuner_allowed(self):
        """Test: @autotune outside @ubtuner is allowed."""

        @triton.autotune(
            configs=[triton.Config(kwargs={'N': 512})],
            key=['N'],
        )
        @ubtuner(key=['N'])
        @triton.jit
        def kernel_allowed(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
            offs = tl.arange(0, N)
            a = tl.load(a_ptr + offs)
            b = tl.load(b_ptr + offs)
            r = a * b
            tl.store(out_ptr + offs, r)

        # Should not raise any error when defining
        assert kernel_allowed is not None

    def test_ubtuner_outside_autotune_raises_error(self):
        """Test: @ubtuner outside @autotune should raise ValueError."""

        with pytest.raises(ValueError) as exc_info:
            @ubtuner(key=['N'])
            @triton.autotune(
                configs=[triton.Config(kwargs={'N': 512})],
                key=['N'],
            )
            @triton.jit
            def kernel_disallowed(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
                offs = tl.arange(0, N)
                a = tl.load(a_ptr + offs)
                b = tl.load(b_ptr + offs)
                r = a * b
                tl.store(out_ptr + offs, r)

        assert "Cannot apply @ubtuner decorator" in str(exc_info.value)
        assert "@autotune" in str(exc_info.value)

    def test_ubtuner_with_intermediate_decorator(self):
        """Test: @ubtuner with intermediate decorator should work when autotune is outermost."""

        def some_decorator(fn):
            return fn

        @triton.autotune(
            configs=[triton.Config(kwargs={'N': 512})],
            key=['N'],
        )
        @some_decorator
        @ubtuner(key=['N'])
        @triton.jit
        def kernel_with_intermediate(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
            offs = tl.arange(0, N)
            a = tl.load(a_ptr + offs)
            b = tl.load(b_ptr + offs)
            r = a * b
            tl.store(out_ptr + offs, r)

        assert kernel_with_intermediate is not None


if __name__ == "__main__":
    # Run pytest when executed directly
    sys.exit(pytest.main([__file__, "-v"]))