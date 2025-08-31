import numpy as np
from scipy.optimize import minimize_scalar

def calculate_wkb(V0, V0_second_derivative, n):
    """
    Calculates the complex QNM frequency using the 1st-order WKB formula.
    This is a simplified version. A full 6th-order implementation is more complex,
    but this gives a result very close to the known Schwarzschild value and is perfect for testing.
    
    Parameters:
        V0 (float): The maximum value of the potential.
        V0_second_derivative (float): The second derivative of the potential at the maximum.
        n (int): The overtone number (n=0 is the fundamental mode).
    
    Returns:
        complex: The complex quasinormal mode frequency ω.
    """
    # The standard first-order WKB formula
    omega_square = V0 - 1j * (n + 0.5) * np.sqrt(-2 * V0_second_derivative)
    omega = np.sqrt(omega_square)
    
    # We need to choose the root with negative imaginary part (decaying mode)
    if omega.imag > 0:
        omega = -omega
    return omega

def find_potential_maximum(l, M, Q_m, c_q, omega_q, r_min=2.5, r_max=50.0):
    """
    Finds the radius where the effective potential is at its maximum.
    This is a critical step for the WKB method.
    
    Parameters:
        l (int): Angular momentum number
        ... (other parameters same as geometry.f())
        r_min, r_max (float): The range to search for the maximum.
    
    Returns:
        tuple: (r_maximum, V_maximum)
    """
    from potentials import effective_potential_scalar # Import here to avoid circular imports
    
    # Define a function to minimize: -V(r) because we want the maximum of V(r)
    def neg_potential(r):
        return -effective_potential_scalar(r, l, M, Q_m, c_q, omega_q)
    
    # Use SciPy to find the minimum of -V(r), which is the maximum of V(r)
    result = minimize_scalar(neg_potential, bounds=(r_min, r_max), method='bounded')
    
    if result.success:
        r0 = result.x
        V0 = -result.fun
        return r0, V0
    else:
        raise RuntimeError(f"Could not find potential maximum: {result.message}")
