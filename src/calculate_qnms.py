#!/usr/bin/env python3
"""
calculate_qnms.py

Robust, self-contained script to compute (approximate) quasinormal mode (QNM)
frequencies for a static, magnetically charged black hole immersed in quintessence.

This file is intentionally self-contained to avoid dependency problems when
reviewers run the repository. It uses a 1st-order WKB estimate as a simple,
reproducible approximation and performs robust checks for numerical stability.

Physics conventions & metric (geometric units G=c=1):
  f(r) = 1 - 2M/r + Q^2 / r^2 - c_q / r^(3*omega_q + 1)
  where Q^2 = 2*pi * Q_m^2  (magnetic charge relation used in the paper)

Notes:
 - The script prints clear messages instead of producing NaNs.
 - The 1st-order WKB is approximate; higher-order methods give better accuracy.
"""

import argparse
import math
import numpy as np
import sys

# -------------------------
# Metric and potentials
# -------------------------
def f_metric(r, M, Q_m, c_q, omega_q):
    """Metric function f(r). Returns np.nan for non-positive r."""
    r = float(r)
    if r <= 0:
        return np.nan
    Q2 = 2.0 * math.pi * (Q_m**2)   # Q^2 as used in the paper
    exp = 3.0 * omega_q + 1.0
    # protect against negative/zero power domain issues (r>0 always)
    quint_term = c_q / (r**exp) if c_q != 0 else 0.0
    return 1.0 - 2.0*M/r + Q2 / (r**2) - quint_term

def df_dr_numeric(r, M, Q_m, c_q, omega_q, h=1e-6):
    """Numerical derivative of f(r)."""
    return (f_metric(r + h, M, Q_m, c_q, omega_q) - f_metric(r - h, M, Q_m, c_q, omega_q)) / (2.0*h)

def effective_potential_scalar(r, l, M, Q_m, c_q, omega_q):
    """Scalar-field effective potential V(r) = f(r) * [l(l+1)/r^2 + f'(r)/r]."""
    r = float(r)
    fr = f_metric(r, M, Q_m, c_q, omega_q)
    if not np.isfinite(fr):
        return np.nan
    dfr = df_dr_numeric(r, M, Q_m, c_q, omega_q)
    if not np.isfinite(dfr):
        return np.nan
    return fr * ((l*(l+1)) / (r**2) + dfr / r)

# -------------------------
# Horizon & peak finding
# -------------------------
def find_outer_horizon(M, Q_m, c_q, omega_q, r_min=1e-6, r_max=500.0, N=20000):
    """
    Find the largest positive root of f(r)=0 in [r_min, r_max] by scanning and bisection.
    Returns r_h (float) or None if no horizon found (naked).
    """
    rs = np.linspace(r_min, r_max, N)
    frs = np.array([f_metric(r, M, Q_m, c_q, omega_q) for r in rs])
    # mask invalids
    valid = np.isfinite(frs)
    if not np.any(valid):
        return None
    # find sign changes
    s = np.sign(frs)
    # mark intervals where sign changes (exclude zeros due to sampling)
    idx = np.where((s[:-1] * s[1:] < 0) & np.isfinite(frs[:-1]) & np.isfinite(frs[1:]))[0]
    if idx.size == 0:
        return None
    # largest index corresponds to outermost root
    i = idx[-1]
    a, b = rs[i], rs[i+1]
    fa, fb = f_metric(a, M, Q_m, c_q, omega_q), f_metric(b, M, Q_m, c_q, omega_q)
    # bisection refine
    for _ in range(60):
        m = 0.5*(a+b)
        fm = f_metric(m, M, Q_m, c_q, omega_q)
        if not np.isfinite(fm):
            break
        if fa * fm <= 0:
            b = m; fb = fm
        else:
            a = m; fa = fm
    return 0.5*(a+b)

def find_potential_maximum(l, M, Q_m, c_q, omega_q, rmax=200.0):
    """
    Robustly find the radius r0 where the scalar effective potential reaches its maximum.
    Procedure:
      - compute outer horizon r_h (if exists)
      - set sampling domain from just outside horizon to rmax
      - sample V(r) on a fine grid and pick the maximum (mask invalid evaluations)
    Returns (r0, V0) or raises RuntimeError if not found or potential is not suitable.
    """
    r_h = find_outer_horizon(M, Q_m, c_q, omega_q)
    if r_h is None:
        r_start = 2.5  # fallback minimal start (a safe zone outside horizon for typical M=1)
    else:
        r_start = max(r_h * 1.05, 2.1)  # start slightly outside horizon, but not below ~2.1

    if r_start >= rmax:
        raise RuntimeError("Sampling interval invalid: starting radius >= rmax. Try increasing rmax.")

    # sampling grid
    Nsamp = 6000
    rs = np.linspace(r_start, rmax, Nsamp)
    Vs = np.array([effective_potential_scalar(rr, l, M, Q_m, c_q, omega_q) for rr in rs])

    # mask invalid (nan/inf)
    mask = np.isfinite(Vs)
    if not np.any(mask):
        raise RuntimeError("All potential evaluations failed in sampling range. Try smaller c_q or Q_m.")

    rs_ok = rs[mask]
    Vs_ok = Vs[mask]

    # require at least one positive peak (single barrier assumption)
    Vmax = np.max(Vs_ok)
    if Vmax <= 0.0:
        raise RuntimeError("No positive potential barrier found in sampling range; WKB is not applicable.")

    idx = int(np.argmax(Vs_ok))
    r0 = float(rs_ok[idx])
    V0 = float(Vs_ok[idx])
    return r0, V0

# -------------------------
# WKB calculation
# -------------------------
def second_derivative(fun, x0, h=1e-3):
    """Central second derivative (robust). Returns nan if invalid."""
    f0 = fun(x0)
    fp = fun(x0 + h)
    fm = fun(x0 - h)
    if not (np.isfinite(f0) and np.isfinite(fp) and np.isfinite(fm)):
        return float('nan')
    return (fp - 2.0*f0 + fm) / (h*h)

def calculate_wkb(V0, V0_second_derivative, n):
    """
    1st-order WKB estimate:
      omega^2 = V0 - i (n + 1/2) sqrt(-2 V0'')
    returns complex omega (or complex(nan) if invalid)
    """
    if not (np.isfinite(V0) and np.isfinite(V0_second_derivative)):
        return complex(float('nan'), float('nan'))
    if V0_second_derivative >= 0.0:
        # need negative curvature at peak
        return complex(float('nan'), float('nan'))
    # compute
    root_term = np.sqrt(-2.0 * V0_second_derivative)
    omega2 = V0 - 1j * (n + 0.5) * root_term
    try:
        omega = np.sqrt(omega2)
    except Exception:
        return complex(float('nan'), float('nan'))
    # pick branch with negative imaginary part (damped)
    if omega.imag > 0:
        omega = -omega
    return omega

# -------------------------
# Main CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Calculate QNMs (1st-order WKB) for magnetically charged BH + quintessence")
    parser.add_argument('--mass', '-M', type=float, default=1.0, help='Black hole mass M (default 1)')
    parser.add_argument('--charge', '-Q', type=float, default=0.0, help='Magnetic charge Q_m (default 0)')
    parser.add_argument('--quintessence', '-c', type=float, default=0.0, help='Quintessence amplitude c_q (default 0)')
    parser.add_argument('--omega_q', '-w', type=float, default=-0.6666667, help='Quintessence EOS parameter ω_q (default -2/3)')
    parser.add_argument('--multipole', '-l', type=int, default=2, help='Angular multipole ℓ (default 2)')
    parser.add_argument('--overtone', '-n', type=int, default=0, help='Overtone number n (default 0)')
    parser.add_argument('--rmax', type=float, default=200.0, help='Maximum radius for sampling (default 200)')
    args = parser.parse_args()

    M = float(args.mass)
    Q_m = float(args.charge)
    c_q = float(args.quintessence)
    omega_q = float(args.omega_q)
    l = int(args.multipole)
    n = int(args.overtone)

    print("="*60)
    print("QNM Calculator (1st-order WKB) — Magnetically charged BH + Quintessence")
    print("="*60)
    print(f"Parameters: M={M}, Q_m={Q_m}, c_q={c_q}, ω_q={omega_q}, ℓ={l}, n={n}")
    print("Note: This script uses a simplified 1st-order WKB approximation for demonstration.")
    print("      For publication-level accuracy see higher-order WKB/AIM/Leaver as described in the paper.")
    print("="*60)

    # 1) find potential maximum
    print("Step 1/3: Locating potential maximum (sampling)...")
    try:
        r0, V0 = find_potential_maximum(l, M, Q_m, c_q, omega_q, rmax=args.rmax)
    except RuntimeError as e:
        print("  ERROR:", e)
        print("  Suggestion: try smaller --quintessence and/or smaller --charge or larger --rmax.")
        return

    print(f"  Found potential maximum at r0 = {r0:.6f} (V0 = {V0:.6e})")

    # 2) compute second derivative
    print("Step 2/3: Estimating curvature at r0...")
    Vfun = lambda r: effective_potential_scalar(r, l, M, Q_m, c_q, omega_q)
    Vpp = second_derivative(Vfun, r0, h=1e-3)
    if not np.isfinite(Vpp):
        print("  ERROR: second derivative evaluation failed (NaN/Inf).")
        print("  Suggestion: try slightly different step (increase rmax) or smaller c_q/Q_m.")
        return
    print(f"  V''(r0) = {Vpp:.6e}")

    # 3) compute WKB frequency
    print("Step 3/3: Computing 1st-order WKB frequency...")
    omega = calculate_wkb(V0, Vpp, n)
    if not (np.isfinite(omega.real) and np.isfinite(omega.imag)):
        print("  ERROR: WKB produced invalid frequency (likely V'' sign incorrect or method not applicable).")
        print("  Suggestion: try smaller --quintessence or --charge, or use higher-order methods.")
        return

    # print results
    print("="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Complex Frequency Mω = {omega.real:.6f} - i{abs(omega.imag):.6f}")
    print(f"Oscillation Frequency Re(ω) = {omega.real:.6f} (1/M)")
    print(f"Damping Rate Im(ω) = {omega.imag:.6f} (1/M)  (negative => damping)")
    print("="*60)

    # optional comparison to Schwarzschild benchmark
    if (abs(Q_m) < 1e-12) and (abs(c_q) < 1e-12) and (l == 2) and (n == 0):
        known = complex(0.37367, -0.08896)
        err_re = abs(omega.real - known.real) / abs(known.real) * 100.0
        err_im = abs(omega.imag - known.imag) / abs(known.imag) * 100.0
        print("Comparison to Schwarzschild (ℓ=2, n=0):")
        print(f"  Known: {known.real:.5f} - i{abs(known.imag):.5f}")
        print(f"  Error: Re(ω) = {err_re:.2f}%, Im(ω) = {err_im:.2f}%")
        print("(Reminder: 1st-order WKB is approximate; higher-order methods are more accurate.)")
        print("="*60)

if __name__ == "__main__":
    main()
