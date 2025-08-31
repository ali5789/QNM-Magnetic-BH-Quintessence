import numpy as np

def f(r, M, Q_m, c_q, omega_q):
    """
    Calculates the metric function f(r) for the magnetically charged BH in quintessence.
    
    Parameters:
        r (float): Radial coordinate
        M (float): Black hole mass
        Q_m (float): Magnetic charge
        c_q (float): Quintessence parameter
        omega_q (float): Quintessence equation-of-state parameter
    
    Returns:
        float: The value of the metric function f(r) at r.
    """
    # The Reissner-Nordström-like term for magnetic charge
    # Q^2 = 2 * pi * Q_m^2 (for standard Maxwell, s=1)
    Q_squared = 2 * np.pi * Q_m**2
    charge_term = Q_squared / r**2
    
    # The quintessence term
    quintessence_term = c_q / r**(3*omega_q + 1)
    
    # The full metric function
    f_value = 1 - (2*M)/r + charge_term - quintessence_term
    
    return f_value

def df_dr(r, M, Q_m, c_q, omega_q, h=1e-5):
    """
    Calculates the first derivative of f(r) with respect to r using a finite difference method.
    This is needed for the effective potential.
    
    Parameters:
        r (float): Radial coordinate
        ... (other parameters same as f(r))
        h (float): Small step for numerical differentiation
    
    Returns:
        float: The value of df/dr at r.
    """
    return (f(r + h, M, Q_m, c_q, omega_q) - f(r - h, M, Q_m, c_q, omega_q)) / (2 * h)

def d2f_dr2(r, M, Q_m, c_q, omega_q, h=1e-5):
    """
    Calculates the second derivative of f(r) with respect to r.
    This is needed for the WKB method.
    
    Parameters:
        r (float): Radial coordinate
        ... (other parameters same as f(r))
        h (float): Small step for numerical differentiation
    
    Returns:
        float: The value of d²f/dr² at r.
    """
    return (f(r + h, M, Q_m, c_q, omega_q) - 2*f(r, M, Q_m, c_q, omega_q) + f(r - h, M, Q_m, c_q, omega_q)) / (h**2)
