from __future__ import annotations

import os
import math

import torch
import torch.nn.functional as F

from .fxp_utils import (
    FXP_FIXED_WIDTH,
    FXP_OUTPUT_FIXED_WIDTH,
    fxp_exp as _fxp_exp,
    fxp_ln as _fxp_ln,
    fxp_softplus as _fxp_softplus,
    fxp_silu as _fxp_silu,
)

_USE_FXP_EXP = 1
_USE_FXP_SOFTPLUS = 1
_USE_FXP_SILU = 1
_USE_FXP_LN = 1


# -----------------------------------------------------------------------------
# Torch-level units (for Python code paths)
# -----------------------------------------------------------------------------

def fxp_exp(x: torch.Tensor, fixed_width: int = FXP_FIXED_WIDTH, output_fixed_width: int = FXP_OUTPUT_FIXED_WIDTH) -> torch.Tensor:
    return _fxp_exp(x, fixed_width=fixed_width, output_fixed_width=output_fixed_width)


def fxp_ln(x: torch.Tensor, fixed_width: int = FXP_FIXED_WIDTH, output_fixed_width: int = FXP_OUTPUT_FIXED_WIDTH) -> torch.Tensor:
    return _fxp_ln(x, fixed_width=fixed_width, output_fixed_width=output_fixed_width)


def fxp_softplus(x: torch.Tensor, fixed_width: int = FXP_FIXED_WIDTH, output_fixed_width: int = FXP_OUTPUT_FIXED_WIDTH) -> torch.Tensor:
    return _fxp_softplus(x, fixed_width=fixed_width, output_fixed_width=output_fixed_width)


def fxp_silu(x: torch.Tensor, fixed_width: int = FXP_FIXED_WIDTH, output_fixed_width: int = FXP_OUTPUT_FIXED_WIDTH) -> torch.Tensor:
    return _fxp_silu(x, fixed_width=fixed_width, output_fixed_width=output_fixed_width)


def exp(x: torch.Tensor) -> torch.Tensor:
    """Drop-in exp() for quamba code paths (env-toggleable)."""
    return fxp_exp(x) if _USE_FXP_EXP else torch.exp(x)


def ln(x: torch.Tensor) -> torch.Tensor:
    """Drop-in ln() for quamba code paths (env-toggleable)."""
    return fxp_ln(x) if _USE_FXP_LN else torch.log(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    """Drop-in softplus() for quamba code paths (env-toggleable).

    - If QUAMBA_FXP_SOFTPLUS=1: use fixed-point (fxp_exp+fxp_ln).
    - Else: defer to torch.nn.functional.softplus (baseline).
    """
    if _USE_FXP_SOFTPLUS:
        return fxp_softplus(x)
    return F.softplus(x)


def silu(x: torch.Tensor) -> torch.Tensor:
    """Drop-in silu() for quamba code paths (env-toggleable)."""
    return fxp_silu(x) if _USE_FXP_SILU else F.silu(x)


# -----------------------------------------------------------------------------
# Triton-level units (for SSD kernels)
# -----------------------------------------------------------------------------

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    from packaging import version  # type: ignore

    _TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON3 = False
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    # Prefer mamba_ssm's reference softplus for the baseline path to avoid
    # unintended numerical deltas when FXP is disabled.
    try:  # pragma: no cover
        from mamba_ssm.ops.triton.softplus import softplus as _mamba_softplus_tl  # type: ignore

        _MAMBA_SOFTPLUS_AVAILABLE = True
    except Exception:  # pragma: no cover
        _mamba_softplus_tl = None  # type: ignore
        _MAMBA_SOFTPLUS_AVAILABLE = False

    @triton.jit
    def fxp_exp_tl(x, fixed_width: tl.constexpr = FXP_FIXED_WIDTH, output_fixed_width: tl.constexpr = FXP_OUTPUT_FIXED_WIDTH):
        scale_factor = 1 << fixed_width
        x_int = (x * scale_factor).to(tl.int32)
        y_int = x_int + (x_int >> 1) - (x_int >> 4)
        u = y_int >> fixed_width
        v_int = y_int - (u << fixed_width)
        approx_2v_int = v_int + (1 << fixed_width)
        total_shift = u - fixed_width + output_fixed_width
        out_int = tl.where(total_shift >= 0, approx_2v_int << total_shift, approx_2v_int >> (-total_shift))
        return out_int.to(tl.float32) * (1.0 / (1 << output_fixed_width))


    @triton.jit
    def fxp_ln_tl(x, fixed_width: tl.constexpr = FXP_FIXED_WIDTH, output_fixed_width: tl.constexpr = FXP_OUTPUT_FIXED_WIDTH):
        """Base ln approximation (no slope correction)."""
        x = tl.maximum(x, 1e-9)
        scale_factor = 1 << fixed_width
        x_int = (x * scale_factor).to(tl.int32)
        x_f = x_int.to(tl.float32)
        # msb_pos ~= floor(log2(x_int)) via float log.
        inv_ln2 = 1.4426950408889634  # 1 / ln(2)
        msb_pos = (tl.math.log(x_f) * inv_ln2).to(tl.int32)
        w = msb_pos - fixed_width
        k_int = tl.where(w >= 0, x_int >> w, x_int << (-w))

        t = k_int - (1 << fixed_width)
        log2k_approx_int = t

        z_int = (w << fixed_width) + log2k_approx_int
        ln_approx_intermediate = (z_int >> 1) + (z_int >> 3) + (z_int >> 4)
        scale_shift = output_fixed_width - fixed_width
        out_int = tl.where(scale_shift >= 0, ln_approx_intermediate << scale_shift, ln_approx_intermediate >> (-scale_shift))
        return out_int.to(tl.float32) * (1.0 / (1 << output_fixed_width))

    @triton.jit
    def fxp_ln_slopefix_w0_tl(x, fixed_width: tl.constexpr = FXP_FIXED_WIDTH, output_fixed_width: tl.constexpr = FXP_OUTPUT_FIXED_WIDTH):
        """ln approximation with slope correction only for normalized w==0 (x in [1,2))."""
        x = tl.maximum(x, 1e-9)
        scale_factor = 1 << fixed_width
        x_int = (x * scale_factor).to(tl.int32)
        x_f = x_int.to(tl.float32)
        inv_ln2 = 1.4426950408889634  # 1 / ln(2)
        msb_pos = (tl.math.log(x_f) * inv_ln2).to(tl.int32)
        w = msb_pos - fixed_width
        k_int = tl.where(w >= 0, x_int >> w, x_int << (-w))

        t = k_int - (1 << fixed_width)
        t_alpha = t + (t >> 1) - (t >> 4) + (t >> 6)
        log2k_approx_int = tl.where(w == 0, t_alpha, t)

        z_int = (w << fixed_width) + log2k_approx_int
        ln_approx_intermediate = (z_int >> 1) + (z_int >> 3) + (z_int >> 4)
        scale_shift = output_fixed_width - fixed_width
        out_int = tl.where(scale_shift >= 0, ln_approx_intermediate << scale_shift, ln_approx_intermediate >> (-scale_shift))
        return out_int.to(tl.float32) * (1.0 / (1 << output_fixed_width))



    @triton.jit
    def fxp_softplus_tl(x, fixed_width: tl.constexpr = FXP_FIXED_WIDTH, output_fixed_width: tl.constexpr = FXP_OUTPUT_FIXED_WIDTH):
        exp_x = fxp_exp_tl(x, fixed_width=fixed_width, output_fixed_width=output_fixed_width)
        return fxp_ln_slopefix_w0_tl(exp_x + 1.0, fixed_width=output_fixed_width, output_fixed_width=output_fixed_width)

    @triton.jit
    def fxp_silu_tl(x, fixed_width: tl.constexpr = FXP_FIXED_WIDTH, output_fixed_width: tl.constexpr = FXP_OUTPUT_FIXED_WIDTH):
        """Fixed-point SiLU approximation for Triton tensors.

        Mirrors fxp_utils.fxp_silu() (masking x < -4 to 0.0).
        """
        compute_mask = x >= -4.0
        a = tl.abs(x)
        # IMPORTANT: never apply slope-fix on ln(|x|) branch (can hurt SiLU accuracy).
        ln_a = fxp_ln_tl(a + 1e-9, fixed_width=fixed_width, output_fixed_width=output_fixed_width)
        exp_neg_a = fxp_exp_tl(-a, fixed_width=fixed_width, output_fixed_width=output_fixed_width)
        # Apply slope-fix only on ln(1+exp(.)) path (and only when normalized w==0).
        ln_1p_exp_neg_a = fxp_ln_slopefix_w0_tl(exp_neg_a + 1.0, fixed_width=output_fixed_width, output_fixed_width=output_fixed_width)
        ratio = fxp_exp_tl(ln_a - ln_1p_exp_neg_a, fixed_width=output_fixed_width, output_fixed_width=output_fixed_width)
        out = tl.where(x < 0, x + ratio, ratio)
        return tl.where(compute_mask, out, 0.0)


    # Reference softplus used when FXP softplus is disabled.
    if _TRITON3:
        @triton.jit
        def _softplus_ref_tl(x):
            return tl.where(x <= 20.0, tl.math.log(tl.math.exp(x) + 1.0), x)

    else:
        @triton.jit
        def _softplus_ref_tl(x):
            return tl.where(x <= 20.0, tl.math.log1p(tl.exp(x)), x)


    def get_exp_tl():
        """Return the exp function to use in Triton kernels (tl.exp vs fxp_exp_tl)."""
        return fxp_exp_tl if _USE_FXP_EXP else tl.exp


    def get_softplus_tl():
        """Return the softplus function to use in Triton kernels."""
        if _USE_FXP_SOFTPLUS:
            print("do _USE_FXP_SOFTPLUS")
            return fxp_softplus_tl
        if _MAMBA_SOFTPLUS_AVAILABLE:
            print("do _MAMBA_SOFTPLUS_AVAILABLE")
            return _mamba_softplus_tl
        return _softplus_ref_tl


    @triton.jit
    def _silu_ref_tl(x):
        return x * tl.sigmoid(x)

    def get_silu_tl():
        """Return the SiLU(x)=x*sigmoid(x) function to use in Triton kernels."""
        if _USE_FXP_SILU:
            return fxp_silu_tl
        return _silu_ref_tl

else:  # pragma: no cover
    # Keep imports from failing in environments without Triton.
    def get_exp_tl():  # type: ignore[no-redef]
        raise RuntimeError("Triton not available")

    def get_softplus_tl():  # type: ignore[no-redef]
        raise RuntimeError("Triton not available")

    def get_silu_tl():  # type: ignore[no-redef]
        raise RuntimeError("Triton not available")
