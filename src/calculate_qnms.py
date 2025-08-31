#!/usr/bin/env python3
"""
Main script to calculate QNMs for a magnetically charged BH with quintessence.
This is a simplified version for demonstration and reproducibility.
"""

import argparse
import numpy as np
from wkb import find_potential_maximum, calculate_wkb
from geometry import d2f_dr2, df_dr

def main():
    parser = argparse.ArgumentParser(description="Calculate QNMs using WKB approximation")
    parser.add_argument('--mass', '-M', type=float, default=1.0, help='Black hole mass')
    parser.add_argument('--charge', '-Q', type=float, default=0.0, help='Magnetic charge Q_m')
    parser.add_argument('--quintessence', '-c', type=float, default=0.0, help='Quintessence parameter c_q')
    parser.add_argument('--omega_q', '-w', type=float, default=-0.6667, help='EOS parameter ω_q (default: -2/3)')
    parser.add_argument('--multipole', '-l', type=int, default=2, help='Angular number ℓ (default: 2)')
    parser.add_argument('--overtone', '-n', type=int, default=0, help='Overtone number n (default: 0)')
    
    args = parser.parse_args()

    print("="*50)
    print("QNM Calculator: Magnetically Charged BH + Quintessence")
    print("="*50)
    print(f"Parameters: M={args.mass}, Q_m={args.charge}, c_q={args.quintessence}, ω_q={args.omega_q}")
    print(f"Mode: l={args.multipole}, n={args.overtone} (scalar perturbations)")
    print("Method: 1st-order WKB approximation")
    print("="*50)

    # 1. Find the peak of the potential
    print("Step 1/3: Finding potential maximum...")
    try:
        r0, V0 = find_potential_maximum(args.multipole, args.mass, args.charge, args.quintessence, args.omega_q)
        print(f"   Found at r0 = {r0:.6f}")
        print(f"   V(r0) = {V0:.6f}")
    except RuntimeError as e:
        print(f"   Error: {e}")
        return

    # 2. Calculate the second derivative of the potential at r0 (simplified)
    print("Step 2/3: Calculating potential curvature...")
    # For a more accurate WKB, we need d²V/dr² at r0. For this test, we approximate it.
    h = 1e-3
    from potentials import effective_potential_scalar
    V_plus = effective_potential_scalar(r0 + h, args.multipole, args.mass, args.charge, args.quintessence, args.omega_q)
    V_minus = effective_potential_scalar(r0 - h, args.multipole, args.mass, args.charge, args.quintessence, args.omega_q)
    V0_second_derivative = (V_plus - 2*V0 + V_minus) / (h**2)
    print(f"   V''(r0) = {V0_second_derivative:.6f}")

    # 3. Calculate the QNM frequency
    print("Step 3/3: Calculating QNM frequency...")
    omega = calculate_wkb(V0, V0_second_derivative, args.overtone)

    print("="*50)
    print("RESULTS:")
    print("="*50)
    print(f"Complex Frequency Mω = {omega.real:.6f} {np.sign(omega.imag)} i{abs(omega.imag):.6f}")
    print(f"Oscillation Frequency Re(ω) = {omega.real:.6f} (1/M)")
    print(f"Damping Rate Im(ω) = {omega.imag:.6f} (1/M)")
    print("="*50)

    # Compare to known Schwarzschild value if parameters are default
    if args.charge == 0.0 and args.quintessence == 0.0 and args.multipole == 2 and args.overtone == 0:
        known_schwarzschild = 0.37367 - 1j*0.08896
        error_real = abs(omega.real - known_schwarzschild.real) / known_schwarzschild.real * 100
        error_imag = abs(omega.imag - known_schwarzschild.imag) / abs(known_schwarzschild.imag) * 100
        print(f"Comparison to Schwarzschild (l=2,n=0):")
        print(f"   Known value: {known_schwarzschild.real:.5f} - i{abs(known_schwarzschild.imag):.5f}")
        print(f"   Error: Re(ω) = {error_real:.2f}%, Im(ω) = {error_imag:.2f}%")
        print("(A small error is expected due to the numerical approximation.)")
        print("="*50)

if __name__ == "__main__":
    main()
