from __future__ import annotations

import os
import math
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MPL_AVAILABLE = True
except Exception:
    plt = None  # type: ignore[assignment]
    MPL_AVAILABLE = False


FXP_FIXED_WIDTH = 8
FXP_OUTPUT_FIXED_WIDTH = 8


def fxp_exp(x: "torch.Tensor", fixed_width: int = FXP_FIXED_WIDTH, output_fixed_width: int = FXP_OUTPUT_FIXED_WIDTH) -> "torch.Tensor":
    """Fixed-point exp approximation (vectorized, pure PyTorch bitwise ops).

    This matches the exp core used in `test_naf_tensor.py` but defaults to fixed_width=8/out=8.
    """
    scale_factor = 1 << fixed_width
    x_int = (x * scale_factor).to(torch.int64)  # torch-style trunc toward 0
    y_int = x_int + torch.bitwise_right_shift(x_int, 1) - torch.bitwise_right_shift(x_int, 4)
    u = torch.bitwise_right_shift(y_int, fixed_width)
    v_int = y_int - torch.bitwise_left_shift(u, fixed_width)
    approx_2v_int = v_int + (1 << fixed_width)
    total_shift = u - fixed_width + output_fixed_width

    pos = total_shift >= 0
    left_shifted = torch.bitwise_left_shift(approx_2v_int, total_shift)
    right_shifted = torch.bitwise_right_shift(approx_2v_int, -total_shift)
    output_int = torch.where(pos, left_shifted, right_shifted)
    return (output_int.to(torch.float32) / float(1 << output_fixed_width)).to(dtype=x.dtype)


def fxp_ln(x: "torch.Tensor", fixed_width: int = FXP_FIXED_WIDTH, output_fixed_width: int = FXP_OUTPUT_FIXED_WIDTH) -> "torch.Tensor":
    """Fixed-point ln approximation (vectorized, pure PyTorch bitwise ops).

    Base variant: log2(k) ~= (k-1) with no slope correction. This is used by
    generic ln callers (e.g., SiLU) to preserve legacy behavior.
    """
    x_clamped = torch.clamp(x, min=1e-9)
    scale_factor = 1 << fixed_width
    x_int = (x_clamped * scale_factor).to(torch.int64)  # torch-style trunc toward 0

    # msb_pos = floor(log2(x_int)) for x_int > 0
    msb_pos = torch.log2(x_int.to(torch.float32)).floor().to(torch.int64)
    w = msb_pos - fixed_width

    # k_int = x_int >> w (if w>=0) else x_int << (-w)
    k_int = torch.where(
        w >= 0,
        torch.bitwise_right_shift(x_int, w),
        torch.bitwise_left_shift(x_int, -w),
    )

    t = k_int - (1 << fixed_width)  # (k-1) in Q(fixed_width), k in [1,2)
    log2k_approx_int = t
    z_int = torch.bitwise_left_shift(w, fixed_width) + log2k_approx_int
    ln_approx_intermediate = (
        torch.bitwise_right_shift(z_int, 1)
        + torch.bitwise_right_shift(z_int, 3)
        + torch.bitwise_right_shift(z_int, 4)
    )

    scale_shift = output_fixed_width - fixed_width
    if scale_shift >= 0:
        output_int = torch.bitwise_left_shift(ln_approx_intermediate, scale_shift)
    else:
        output_int = torch.bitwise_right_shift(ln_approx_intermediate, -scale_shift)

    return (output_int.to(torch.float32) / float(1 << output_fixed_width)).to(dtype=x.dtype)


def fxp_ln_slopefix_w0(
    x: "torch.Tensor", fixed_width: int = FXP_FIXED_WIDTH, output_fixed_width: int = FXP_OUTPUT_FIXED_WIDTH
) -> "torch.Tensor":
    """Fixed-point ln with slope correction for the normalized w==0 branch only.

    This matches `test_naf_tensor2.py`:
      - x = 2^w * k, k in [1,2)
      - log2(k) ~= alpha*(k-1) when w==0, alpha ~= 1 + 1/2 - 1/16 + 1/64
      - ln(x) ~= ln2_approx * (w + log2(k))

    IMPORTANT: We only use this from softplus (and only for its internal ln),
    because applying it inside other composites (e.g. SiLU) regressed accuracy.
    """
    x_clamped = torch.clamp(x, min=1e-9)
    scale_factor = 1 << fixed_width
    x_int = (x_clamped * scale_factor).to(torch.int64)  # torch-style trunc toward 0

    msb_pos = torch.log2(x_int.to(torch.float32)).floor().to(torch.int64)
    w = msb_pos - fixed_width

    k_int = torch.where(
        w >= 0,
        torch.bitwise_right_shift(x_int, w),
        torch.bitwise_left_shift(x_int, -w),
    )

    t = k_int - (1 << fixed_width)  # (k-1) in Q(fixed_width), k in [1,2)
    t_alpha = t + torch.bitwise_right_shift(t, 1) - torch.bitwise_right_shift(t, 4) + torch.bitwise_right_shift(t, 6)
    log2k_approx_int = torch.where(w == 0, t_alpha, t)

    z_int = torch.bitwise_left_shift(w, fixed_width) + log2k_approx_int
    ln_approx_intermediate = (
        torch.bitwise_right_shift(z_int, 1)
        + torch.bitwise_right_shift(z_int, 3)
        + torch.bitwise_right_shift(z_int, 4)
    )

    scale_shift = output_fixed_width - fixed_width
    if scale_shift >= 0:
        output_int = torch.bitwise_left_shift(ln_approx_intermediate, scale_shift)
    else:
        output_int = torch.bitwise_right_shift(ln_approx_intermediate, -scale_shift)

    return (output_int.to(torch.float32) / float(1 << output_fixed_width)).to(dtype=x.dtype)


def fxp_softplus(x: "torch.Tensor", fixed_width: int = FXP_FIXED_WIDTH, output_fixed_width: int = FXP_OUTPUT_FIXED_WIDTH) -> "torch.Tensor":
    """Fixed-point softplus using fxp_exp + fxp_ln: ln(1 + exp(x))."""
    exp_x = fxp_exp(x, fixed_width=fixed_width, output_fixed_width=output_fixed_width)
    # Slope fix is only enabled for the ln(1+exp(.)) path, and only for w==0.
    return fxp_ln_slopefix_w0(exp_x + 1.0, fixed_width=output_fixed_width, output_fixed_width=output_fixed_width)


def fxp_silu(x: "torch.Tensor", fixed_width: int = FXP_FIXED_WIDTH, output_fixed_width: int = FXP_OUTPUT_FIXED_WIDTH) -> "torch.Tensor":
    """Fixed-point silu using the stable identity:

    For a = |x|:
      silu(x) = a/(1+exp(-a))            if x>=0
              = x + a/(1+exp(-a))       if x<0
    We approximate division via exp(ln(a) - ln(1+exp(-a))).
    """
    # Keep the small optimization from the prototype: values < -4 are close to 0 anyway.
    compute_mask = (x != 0) & (x >= -4)
    out = torch.zeros_like(x)
    x_sel = x[compute_mask]
    if x_sel.numel() == 0:
        return out

    a = torch.abs(x_sel)
    # IMPORTANT: never apply slope-fix on ln(|x|) branch (can hurt SiLU accuracy).
    ln_a = fxp_ln(a + 1e-9, fixed_width=fixed_width, output_fixed_width=output_fixed_width)
    exp_neg_a = fxp_exp(-a, fixed_width=fixed_width, output_fixed_width=output_fixed_width)
    # Apply slope-fix only on ln(1+exp(.)) path (and only when normalized w==0).
    ln_1p_exp_neg_a = fxp_ln_slopefix_w0(
        exp_neg_a + 1.0, fixed_width=output_fixed_width, output_fixed_width=output_fixed_width
    )
    ratio = fxp_exp(ln_a - ln_1p_exp_neg_a, fixed_width=output_fixed_width, output_fixed_width=output_fixed_width)
    silu_sel = torch.where(x_sel < 0, x_sel + ratio, ratio)
    out[compute_mask] = silu_sel.to(out.dtype)
    return out


@dataclass(frozen=True)
class ErrorStats:
    mean_abs: float
    max_abs: float
    mean_rel: float
    max_rel: float


def _error_stats(y_hat: "torch.Tensor", y_ref: "torch.Tensor") -> ErrorStats:
    y_hat = y_hat.detach().to(torch.float32).cpu()
    y_ref = y_ref.detach().to(torch.float32).cpu()
    abs_err = (y_hat - y_ref).abs()
    rel_err = abs_err / (y_ref.abs() + 1e-12)
    return ErrorStats(
        mean_abs=float(abs_err.mean().item()),
        max_abs=float(abs_err.max().item()),
        mean_rel=float(rel_err.mean().item()),
        max_rel=float(rel_err.max().item()),
    )


def _plot_curve(x: "torch.Tensor", y_hat: "torch.Tensor", y_ref: "torch.Tensor", *, title: str, save_path: str, yscale: str = "linear") -> None:
    if not MPL_AVAILABLE:
        raise RuntimeError("matplotlib is required for plotting")
    x_np = x.detach().to(torch.float32).cpu().numpy()
    y_hat_np = y_hat.detach().to(torch.float32).cpu().numpy()
    y_ref_np = y_ref.detach().to(torch.float32).cpu().numpy()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x_np, y_ref_np, "b--", linewidth=2, label=f"torch {title}")
    ax.plot(x_np, y_hat_np, "r-", linewidth=1, label=f"fxp {title}")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_yscale(yscale)
    ax.legend()
    ax.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_single_test(name: str, x: "torch.Tensor", y_hat: "torch.Tensor", y_ref: "torch.Tensor", *, plot_dir: str, yscale: str = "linear") -> None:
    stats = _error_stats(y_hat, y_ref)
    print(f"\n[{name}]")
    print(f"abs_err: mean={stats.mean_abs:.6g} max={stats.max_abs:.6g}")
    print(f"rel_err: mean={stats.mean_rel:.6g} max={stats.max_rel:.6g}")
    if MPL_AVAILABLE:
        _plot_curve(x, y_hat, y_ref, title=name, save_path=os.path.join(plot_dir, f"{name}.png"), yscale=yscale)
        print(f"plot: {os.path.join(plot_dir, f'{name}.png')}")
    else:
        print("plot: skipped (matplotlib not available)")


def run_fxp_unit_tests(*, device: str = "cpu", plot_dir: str = "./fxp_plots") -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required")

    dev = torch.device(device)
    dtype = torch.float32

    # exp
    x = torch.arange(-5.0, 5.001, 0.01, device=dev, dtype=dtype)
    _run_single_test("fxp_exp", x, fxp_exp(x), torch.exp(x), plot_dir=plot_dir, yscale="log")

    # ln
    x = torch.arange(0.1, 8.001, 0.01, device=dev, dtype=dtype)
    _run_single_test("fxp_ln", x, fxp_ln(x), torch.log(x), plot_dir=plot_dir)

    # softplus
    x = torch.arange(-10.0, 5.001, 0.01, device=dev, dtype=dtype)
    _run_single_test("fxp_softplus", x, fxp_softplus(x), F.softplus(x), plot_dir=plot_dir)

    # silu
    x = torch.arange(-8.0, 8.001, 0.01, device=dev, dtype=dtype)
    _run_single_test("fxp_silu", x, fxp_silu(x), F.silu(x), plot_dir=plot_dir)


if __name__ == "__main__":
    # Keep it simple: default CPU test, plot to ./fxp_plots
    run_fxp_unit_tests()
