from geometry import f, df_dr

def effective_potential_scalar(r, l, M, Q_m, c_q, omega_q):
    """
    Effective potential for massless scalar field perturbations.
    
    Parameters:
        r (float): Radial coordinate
        l (int): Angular momentum quantum number
        ... (other parameters same as geometry.f())
    
    Returns:
        float: The value of the effective potential V(r) at r.
    """
    f_val = f(r, M, Q_m, c_q, omega_q)
    df_dr_val = df_dr(r, M, Q_m, c_q, omega_q)
    
    V = f_val * ( (l*(l+1)) / (r**2) + df_dr_val / r )
    return V

# (We can add functions for electromagnetic and gravitational potentials here later)
# For now, we'll focus on the scalar potential to keep things simple for testing.
