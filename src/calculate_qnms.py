import numpy as np

def f(r, M, Q_m, c_q, omega_q):
    """
    Metric function f(r).
    Note: for omega_q = -2/3, the quintessence term behaves ~ +c_q * r.
    """
    Q_squared = 2 * np.pi * Q_m**2
    # guard against r <= 0
    r = np.asarray(r, dtype=float)
    if np.any(r <= 0):
        return np.nan
    charge_term = Q_squared / r**2
    # c_q / r**(3*omega_q+1)  (for omega_q=-2/3, exponent ≈ -1, so this is c_q / r**(-1) = c_q * r)
    exp = 3.0*omega_q + 1.0
    quintessence_term = c_q / (r**exp)
    return 1.0 - (2.0*M)/r + charge_term - quintessence_term

def df_dr(r, M, Q_m, c_q, omega_q):
    """Numerical derivative of f using a safe, small step."""
    h = 1e-5
    return (f(r + h, M, Q_m, c_q, omega_q) - f(r - h, M, Q_m, c_q, omega_q)) / (2.0*h)

def effective_potential_scalar(r, l, M, Q_m, c_q, omega_q):
    """
    Effective potential for scalar perturbations V(r) = f(r) * [ l(l+1)/r^2 + f'(r)/r ].
    Returns NaN if any ingredient is invalid.
    """
    fr = f(r, M, Q_m, c_q, omega_q)
    if not np.isfinite(fr):
        return np.nan
    dfr = df_dr(r, M, Q_m, c_q, omega_q)
    if not np.isfinite(dfr):
        return np.nan
    return fr * ( (l*(l+1)) / (r**2) + dfr / r )

def find_outer_horizon(M, Q_m, c_q, omega_q, r_min=1e-3, r_max=200.0, N=20000):
    """
    Find the largest root of f(r)=0 in [r_min, r_max] by scanning.
    Returns r_h (outer horizon) or None if not found.
    """
    rs = np.linspace(r_min, r_max, N)
    frs = f(rs, M, Q_m, c_q, omega_q)
    # look for sign changes
    sign = np.sign(frs)
    idx = np.where((sign[:-1] * sign[1:] < 0) & np.isfinite(frs[:-1]) & np.isfinite(frs[1:]))[0]
    if idx.size == 0:
        return None
    # take the largest r where a sign change occurs (outer horizon)
    i = idx[-1]
    # refine root by bisection
    a, b = rs[i], rs[i+1]
    for _ in range(60):
        m = 0.5*(a+b)
        fa, fm = f(a, M, Q_m, c_q, omega_q), f(m, M, Q_m, c_q, omega_q)
        if not (np.isfinite(fa) and np.isfinite(fm)):  # fallback
            break
        if fa*fm <= 0:
            b = m
        else:
            a = m
    return 0.5*(a+b)

def find_potential_maximum(l, M, Q_m, c_q, omega_q, rmax=200.0):
    """
    Robustly find the peak of V(r):
    - start just outside the outer horizon
    - sample V on a grid and pick the max
    """
    r_h = find_outer_horizon(M, Q_m, c_q, omega_q)
    if r_h is None:
        # no horizon found; assume safe inner bound
        r_start = 2.5
    else:
        r_start = max(r_h*1.1, 2.1)  # 10% outside horizon, but not below ~2M
    r_end = rmax

    # sample on a grid
    rs = np.linspace(r_start, r_end, 5000)
    Vs = np.array([effective_potential_scalar(r, l, M, Q_m, c_q, omega_q) for r in rs])
    # mask invalid values
    mask = np.isfinite(Vs)
    if not np.any(mask):
        raise RuntimeError("All potential evaluations failed. Try smaller quintessence or charge.")
    rs, Vs = rs[mask], Vs[mask]

    # require some positive barrier
    if np.max(Vs) <= 0.0:
        raise RuntimeError("No positive potential barrier found; WKB not applicable for these parameters.")

    # argmax
    j = int(np.argmax(Vs))
    r0 = rs[j]
    V0 = Vs[j]
    return r0, V0

def second_derivative_central(fun, x0, h):
    """Second derivative with robust central difference."""
    f0 = fun(x0)
    fp = fun(x0 + h)
    fm = fun(x0 - h)
    if not (np.isfinite(f0) and np.isfinite(fp) and np.isfinite(fm)):
        return np.nan
    return (fp - 2.0*f0 + fm) / (h**2)

def calculate_wkb(V0, V0_second_derivative, n):
    """
    1st-order WKB frequency.
    Adds guards to avoid NaNs when curvature sign is wrong.
    """
    if not (np.isfinite(V0) and np.isfinite(V0_second_derivative)):
        return np.nan + 1j*np.nan
    # We need a local maximum -> V0'' < 0
    if V0_second_derivative >= 0:
        return np.nan + 1j*np.nan
    omega_square = V0 - 1j * (n + 0.5) * np.sqrt(-2.0 * V0_second_derivative)
    omega = np.sqrt(omega_square)
    # choose the decaying mode (negative imaginary part)
    if omega.imag > 0:
        omega = -omega
    return omega
