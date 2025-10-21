#!/usr/bin/env python3
"""
Adaptive-fit comparison demo (uniform offsets vs Chebyshev vs physical grid).

Panels:
A) Uniform OFFSETS around x0 (via grid=offsets)
B) Chebyshev default around x0 (spacing="auto") + optional widened spacing overlay
C) Physical ABSOLUTE x grid (use_physical_grid=True)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---- DerivKit (adjust import path if needed) ----
from derivkit.adaptive.adaptive_fit import AdaptiveFitDerivative

# ---------- target & noise ----------
def f_clean(x, a=1.7, b=-0.2):
    x = np.asarray(x, float)
    return a * x + b  # true slope is 'a'

def make_noisy(f, sigma=0.02, seed=42):
    rng = np.random.default_rng(seed)
    def g(x):
        x = np.asarray(x, float)
        return f(x) + rng.normal(0.0, sigma, size=x.shape)
    return g

# ---------- small helper: value ± error ----------
def format_pm(val: float, err: float) -> str:
    err = abs(float(err))
    if err == 0 or not np.isfinite(err):
        return f"{val:.6g} ± 0"
    leading = int(f"{err:.1e}".split("e")[0].replace(".", "").lstrip("0")[:1] or "1")
    sig_err = 2 if leading == 1 else 1
    exp = int(np.floor(np.log10(err)))
    decimals = max(0, -exp + (sig_err - 1))
    val_r  = round(val,  decimals)
    err_r  = round(err,  decimals)
    return f"{val_r:.{decimals}f} ± {err_r:.{decimals}f}"

# ---------- Derivative helpers ----------
def slope_uniform_offsets(f, x0, *, H=0.12, n_points=25, order=1):
    """Evenly-spaced OFFSETS in [-H, H] around x0 (use physical ABS grid)."""
    t = np.linspace(-H, H, n_points)      # OFFSETS
    x_abs = x0 + t                        # convert to ABSOLUTE x
    est = AdaptiveFitDerivative(f, x0).differentiate(
        order,
        n_points=len(x_abs),
        spacing=x_abs,
        use_physical_grid=True,
    )
    return float(np.asarray(est).ravel()[0])

def slope_chebyshev_default(f, x0, *, n_points=25, spacing="auto", order=1):
    """Chebyshev OFFSETS around x0 (the new default path)."""
    est = AdaptiveFitDerivative(f, x0).differentiate(
        order, n_points=n_points, spacing=spacing
    )
    return float(np.asarray(est).ravel()[0])

def slope_physical_grid(f, x0, x_abs, *, order=1):
    """User-provided ABSOLUTE x grid (non-uniform allowed)."""
    x_abs = np.asarray(x_abs, float)
    est = AdaptiveFitDerivative(f, x0).differentiate(
        order,
        n_points=int(len(x_abs)),
        spacing=x_abs,
        use_physical_grid=True,
    )
    return float(np.asarray(est).ravel()[0])

def main():
    # knobs
    x0        = 0.30
    n_points  = 25                # try 35–40 to trigger the advisory
    noise     = 0.01
    order     = 1
    true_a, true_b = 1.7, -0.2

    # optional: numeric half-width to *widen* Chebyshev when n_points is large
    # (only used when n_points >= 30 and spacing is "auto")
    cheby_spacing_wide = 0.25     # <-- half-width H; tune as you like

    # target (noisy linear)
    f = make_noisy(lambda x: f_clean(x, a=true_a, b=true_b), sigma=noise, seed=42)

    # visualization domain
    delta_vis = 0.01
    W_plot = (n_points - 1) * delta_vis
    xx = np.linspace(x0 - W_plot, x0 + W_plot, 800)

    # grids per panel
    H_uniform = 0.12
    t_uniform = np.linspace(-H_uniform, H_uniform, n_points)
    x_uniform = x0 + t_uniform

    # Custom physical (absolute) grid, more clustered near x0
    t_phys = np.array([-0.10, -0.06, -0.03, -0.015, -0.007, -0.003,
                        0.0,
                        0.003, 0.007, 0.015, 0.03, 0.06, 0.10], dtype=float)
    x_phys = x0 + t_phys

    # compute derivatives
    d_uniform   = slope_uniform_offsets(f, x0, H=H_uniform, n_points=n_points, order=order)

    # Chebyshev default (auto)
    d_cheby     = slope_chebyshev_default(f, x0, n_points=n_points, spacing="auto", order=order)

    # If many points with auto spacing, also evaluate a widened Chebyshev
    d_cheby_wide = None
    if (n_points >= 30):
        try:
            d_cheby_wide = slope_chebyshev_default(
                f, x0, n_points=n_points, spacing=cheby_spacing_wide, order=order
            )
        except Exception:
            # if your AdaptiveFitDerivative expects spacing to be interpreted as half-width, this works.
            # if not, just set cheby_spacing_wide to a percentage string like "10%" instead.
            pass

    d_physical  = slope_physical_grid(f, x0, x_abs=x_phys, order=order)
    true_slope  = true_a

    # quick “errors” vs truth (demo labels)
    e_uniform  = abs(d_uniform - true_slope)
    e_cheby    = abs(d_cheby   - true_slope)
    e_physical = abs(d_physical - true_slope)
    e_cheby_w  = abs(d_cheby_wide - true_slope) if d_cheby_wide is not None else None

    # samples for dots
    x_vis = np.linspace(x0 - W_plot, x0 + W_plot, n_points)
    y_vis = f(x_vis)
    y_uniform = f(x_uniform)
    y_phys    = f(x_phys)

    # plots (plain colors)
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.0), constrained_layout=True)

    # Panel A: uniform OFFSETS
    ax = axs[0]
    ax.plot(xx, f_clean(xx, a=true_a, b=true_b), "--", lw=1.6, color="0.4", label="true $f(x)$")
    ax.scatter(x_uniform, y_uniform, s=70, color="C2", label="samples", zorder=3)
    ax.scatter([x0], [f(x0)], s=110, color="C3", edgecolor="none", label="$x_0$", zorder=4)
    ax.plot(xx, f(x0) + d_uniform * (xx - x0), lw=2.2, color="C0", label="slope @ $x_0$")
    ax.set_title("A · uniform OFFSETS (grid=…)")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.text(0.02, 0.96, f"true ≈ {true_slope:.3f}\nest. = {format_pm(d_uniform, e_uniform)}",
            transform=ax.transAxes, va="top", ha="left")
    ax.legend(frameon=True, fontsize=9, loc="lower right")

    # Panel B: Chebyshev (default) + widened overlay if applicable
    ax = axs[1]
    ax.plot(xx, f_clean(xx, a=true_a, b=true_b), "--", lw=1.6, color="0.4", label="true $f(x)$")
    ax.scatter(x_vis, y_vis, s=70, color="C2", label="samples (viz)", zorder=3)
    ax.scatter([x0], [f(x0)], s=110, color="C3", edgecolor="none", label="$x_0$", zorder=4)
    ax.plot(xx, f(x0) + d_cheby * (xx - x0), lw=2.2, color="C0", label='Chebyshev slope (auto)')
    if d_cheby_wide is not None:
        ax.plot(xx, f(x0) + d_cheby_wide * (xx - x0), lw=2.0, ls="--", color="C1",
                label=f'Chebyshev slope (H={cheby_spacing_wide:g})')
    ax.set_title('B · Chebyshev (spacing="auto")')
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    label_b = f"true ≈ {true_slope:.3f}\n" \
              f"auto = {format_pm(d_cheby, e_cheby)}"
    if d_cheby_wide is not None and e_cheby_w is not None:
        label_b += f"\nwide = {format_pm(d_cheby_wide, e_cheby_w)}"
    ax.text(0.02, 0.96, label_b, transform=ax.transAxes, va="top", ha="left")
    ax.legend(frameon=True, fontsize=9, loc="lower right")

    # Panel C: physical ABSOLUTE grid
    ax = axs[2]
    ax.plot(xx, f_clean(xx, a=true_a, b=true_b), "--", lw=1.6, color="0.4", label="true $f(x)$")
    ax.scatter(x_phys, y_phys, s=70, color="C2", label="user grid", zorder=3)
    ax.scatter([x0], [f(x0)], s=110, color="C3", edgecolor="none", label="$x_0$", zorder=4)
    ax.plot(xx, f(x0) + d_physical * (xx - x0), lw=2.2, color="C0", label="slope @ $x_0$")
    ax.set_title("C · physical grid (absolute x)")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.text(0.02, 0.96, f"true ≈ {true_slope:.3f}\nest. = {format_pm(d_physical, e_physical)}",
            transform=ax.transAxes, va="top", ha="left")
    ax.legend(frameon=True, fontsize=9, loc="lower right")

    # save & print
    outdir = Path("plots"); outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "adaptive_fit_compare.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {outfile}")

    print("\nResults (value ± demo-error vs truth):")
    print(f"A · uniform OFFSETS   : {format_pm(d_uniform,  e_uniform)}")
    if d_cheby_wide is None:
        print(f"B · Chebyshev default : {format_pm(d_cheby,    e_cheby)}")
    else:
        print(f"B · Chebyshev default : {format_pm(d_cheby,    e_cheby)}   "
              f"(widened: {format_pm(d_cheby_wide, e_cheby_w)})")
    print(f"C · physical ABS grid : {format_pm(d_physical, e_physical)}")

    # advisory
    if n_points >= 30:
        print("\nAdvisory:")
        print("  You’re using many Chebyshev points. If you increase n_points, widen the")
        print("  half-width 'spacing' proportionally to keep the fit well-conditioned.")
        print(f"  (Here we also showed a widened Chebyshev with H={cheby_spacing_wide:g}.)")

if __name__ == "__main__":
    main()
