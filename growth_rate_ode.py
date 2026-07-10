"""
Growth-rate ODE system for flat LambdaCDM
==========================================

State vector: y = [Omega_m, f], independent variable N = ln(a)

    dOmega_m/dN = -3 * Omega_m * (1 - Omega_m)
    df/dN       = (3/2)*Omega_m - f*[ 0.5*(4 - 3*Omega_m) + f ]

Both equations were derived in the write-up above from the continuity/Euler/
Poisson equations for pressureless matter. This script:
  1. Integrates the system forward in N with scipy.integrate.solve_ivp
  2. Finds all equilibrium points (dy/dN = 0) analytically with sympy,
     since the equations are simple polynomials in (Omega_m, f)
  3. Classifies each equilibrium via the eigenvalues of the Jacobian
  4. Plots f(a) and a phase portrait with trajectories + equilibria
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Right-hand side of the ODE system
# ---------------------------------------------------------------------
def rhs(N, y):
    """y = [Omega_m, f]; returns [dOmega_m/dN, df/dN]."""
    Om, f = y
    dOm_dN = -3.0 * Om * (1.0 - Om)
    df_dN = 1.5 * Om - f * (0.5 * (4.0 - 3.0 * Om) + f)
    return [dOm_dN, df_dN]


# ---------------------------------------------------------------------
# 2. Equilibrium points, found analytically (exact, not root-finding)
# ---------------------------------------------------------------------
def find_equilibria():
    Om, f = sp.symbols('Omega_m f', real=True)
    eq1 = -3 * Om * (1 - Om)
    eq2 = sp.Rational(3, 2) * Om - f * (sp.Rational(1, 2) * (4 - 3 * Om) + f)

    solutions = sp.solve([sp.Eq(eq1, 0), sp.Eq(eq2, 0)], [Om, f], dict=True)

    # Jacobian for stability classification
    J = sp.Matrix([eq1, eq2]).jacobian([Om, f])

    results = []
    for sol in solutions:
        Om_val, f_val = sol[Om], sol[f]
        J_at_point = J.subs(sol)
        eigenvals = list(J_at_point.eigenvals().keys())
        eigenvals_num = [complex(sp.N(ev)) for ev in eigenvals]

        if all(ev.real < 0 for ev in eigenvals_num):
            kind = "stable node (attractor)"
        elif all(ev.real > 0 for ev in eigenvals_num):
            kind = "unstable node (repeller)"
        else:
            kind = "saddle point"

        results.append({
            "Omega_m": Om_val,
            "f": f_val,
            "eigenvalues": eigenvals_num,
            "type": kind,
        })
    return results


# ---------------------------------------------------------------------
# 3. Integrate a trajectory with solve_ivp
# ---------------------------------------------------------------------
def integrate_trajectory(Om0, f0, N_span=(-8, 0), n_points=400):
    """
    N = ln(a); N_span=(-8, 0) covers a = e^-8 (deep matter domination)
    to a = 1 (today).
    """
    N_eval = np.linspace(N_span[0], N_span[1], n_points)
    sol = solve_ivp(
        rhs, N_span, [Om0, f0],
        t_eval=N_eval,
        method="RK45",
        rtol=1e-9, atol=1e-11,
    )
    return sol


if __name__ == "__main__":
    # ---- Equilibria ----
    print("Equilibrium points of (Omega_m, f):\n")
    equilibria = find_equilibria()
    for eq in equilibria:
        print(f"  (Omega_m, f) = ({eq['Omega_m']}, {eq['f']})")
        print(f"    eigenvalues: {[round(e.real, 3) for e in eq['eigenvalues']]}")
        print(f"    classification: {eq['type']}\n")

    # ---- Trajectory: start deep in matter domination, Omega_m ~ 1, f ~ 1 ----
    a_today_Om0 = 0.3   # Omega_m today, e.g. Planck-like flat LCDM
    sol = integrate_trajectory(Om0=1 - 1e-6, f0=1 - 1e-6, N_span=(-8, 0))
    a_vals = np.exp(sol.t)

    # A second trajectory perturbed off the attractor, to show convergence
    sol2 = integrate_trajectory(Om0=1 - 1e-6, f0=0.5, N_span=(-8, 0))

    # ---- Plot 1: f(a) and Omega_m(a) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(a_vals, sol.y[0], label=r"$\Omega_m(a)$", color="tab:blue")
    axes[0].plot(a_vals, sol.y[1], label=r"$f(a)$", color="tab:red")
    axes[0].set_xlabel("scale factor a")
    axes[0].set_title("Solution along N = ln a")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ---- Plot 2: phase portrait (Omega_m, f) with equilibria marked ----
    Om_grid, f_grid = np.meshgrid(np.linspace(-0.1, 1.1, 22), np.linspace(-2.5, 1.5, 22))
    dOm, df = rhs(0, [Om_grid, f_grid])
    axes[1].streamplot(Om_grid, f_grid, dOm, df, color="lightgray", density=1.1)
    axes[1].plot(sol.y[0], sol.y[1], color="tab:red", lw=2, label="trajectory (near attractor)")
    axes[1].plot(sol2.y[0], sol2.y[1], color="tab:orange", lw=2, ls="--", label="perturbed trajectory")

    for eq in equilibria:
        Om_v, f_v = float(eq["Omega_m"]), float(eq["f"])
        marker = "o" if "stable node" in eq["type"] else ("s" if "saddle" in eq["type"] else "^")
        axes[1].scatter(Om_v, f_v, s=90, marker=marker, edgecolor="k", zorder=5)
        axes[1].annotate(f"({Om_v:.1f}, {f_v:.1f})\n{eq['type'].split(' ')[0]}",
                          (Om_v, f_v), textcoords="offset points", xytext=(8, 8), fontsize=8)

    axes[1].set_xlabel(r"$\Omega_m$")
    axes[1].set_ylabel(r"$f$")
    axes[1].set_title("Phase portrait")
    axes[1].legend(fontsize=8, loc="lower left")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/growth_rate_phase_portrait.png", dpi=150)
    print("Saved plot to growth_rate_phase_portrait.png")
