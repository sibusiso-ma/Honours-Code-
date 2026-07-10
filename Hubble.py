# Hubble parameter vs z
# Sibusiso Mathebula
# 24/06/2026


import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



# Cosmological parameters
H0 = 67.36
Omega_m = 0.3153
Omega_r = 9.2e-5
Omega_k=0.0
Omega_l = 0.6847

def Hubble(z):
    return H0 * np.sqrt(
        Omega_r * (1 + z)**4 +
        Omega_m * (1 + z)**3 +
        Omega_k * (1 + z)**2 +
        Omega_l
    )

# Redshift range
z = np.linspace(0, 5, 1000)

# H(z)
Hz = Hubble(z)

# Plot
plt.figure(figsize=(8,5))
plt.plot(z, Hz)
plt.xlabel("Redshift z")
plt.ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
plt.title("Hubble Parameter vs Redshift")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(np.log10(z+1),np.log10( Hz))
plt.xlabel("Redshift z")
plt.ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
plt.title("Hubble Parameter vs Redshift in log scale")
plt.show()



import numpy as np

# Cosmology parameters
Om = 0.3
Ol = 0.7
Ok = 0.0

c = 299792.458      # km/s
H0 = 70.0           # km/s/Mpc

def E(z):
   
    return Hubble(z) / H0

def f(z):
    return c / E(z)

def comoving_trapezoidal(z, N=1000):
    z_vals = np.linspace(0, z, N+1)
    h = z / N

    integral = (h/2) * (
        f(z_vals[0]) +
        2*np.sum(f(z_vals[1:N])) +
        f(z_vals[N])
    )

    return  integral

print(comoving_trapezoidal(1.0))

def comoving_midpoint(z, N=1000):
    h = z / N
    z_mid = np.linspace(h/2, z - h/2, N)

    integral = h * np.sum(f(z_mid))

    return  integral

print(comoving_midpoint(1.0))

z_values = np.linspace(0, 5, 1000)

chi_trap = []
chi_mid = []

N = 200  # subdivisions for integration

for z in z_values:
    chi_trap.append(comoving_trapezoidal(z, N))
    chi_mid.append(comoving_midpoint(z, N))

chi_trap = np.array(chi_trap)
chi_mid = np.array(chi_mid)

plt.figure(figsize=(8,5))

plt.plot(z_values, chi_trap, label="Trapezoidal Rule")
plt.plot(z_values, chi_mid, label="Midpoint Rule", linestyle="--")

plt.xlabel("Redshift z")
plt.ylabel("Comoving Distance $\chi (z)$ [Mpc]")
plt.title("Comoving Distance vs Redshift")
plt.grid(True)
plt.legend()

plt.show()

error = np.abs(chi_trap - chi_mid)

plt.figure(figsize=(8,5))

plt.plot(z_values, error)

plt.xlabel("Redshift z")
plt.ylabel("|Difference| [Mpc]")
plt.title("Difference Between Trapezoidal and Midpoint Methods")
plt.grid(True)

plt.show()

# Low-z approximation
chi_lowz =  c*z_values

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(z_values, chi_trap,
         label='Numerical (Trapezoidal)',
         linewidth=2)

plt.plot(z_values, chi_lowz,
         '--',
         label=r'Low-z Approximation: $\chi=\frac{cz}{H_0}$')

plt.xlabel("Redshift z")
plt.ylabel("Comoving Distance $\chi(z)$ [Mpc]")
plt.title("Comoving Distance vs Redshift")

plt.legend()
plt.grid(True)

plt.show()


def Omega_mass(z):
    return (Omega_m*((1+z)**3))/(E(z)**2)

def Growth_rate(z):
    return ((Omega_mass(z))**(0.55))


plt.figure(figsize=(8,5))
plt.plot(z_values, Growth_rate(z_values))
plt.xlabel("Redshift z")
plt.ylabel(r"$f(z)$ (dimensionelss)")
plt.title("Growth rate vs Redshift")
plt.show()

def trapezoidal_integral(func, a, b, N):
    z = np.linspace(a, b, N+1)
    y = func(z)
    h = (b-a)/N

    integral = h*(0.5*y[0] + np.sum(y[1:-1]) + 0.5*y[-1])
    return integral    # <-- Missing return statement


def integrand(z):
    return Growth_rate(z)/(1+z)

# Growth factor
def growth_factor(z, N=1000):
    I = trapezoidal_integral(integrand, 0, z, N)
    return np.exp(-I)

# Generate values
z_values = np.linspace(0, 5, 200)
D_values = np.array([growth_factor(z) for z in z_values])

# Plot
plt.figure(figsize=(8,5))
plt.plot(z_values, D_values)
plt.xlabel("Redshift z")
plt.ylabel("Growth factor D(z)")
plt.title("Growth Factor using Trapezoidal Rule")
plt.grid(True)
plt.show()



    
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

##################################################
# Autonomous system
##################################################

def system(N, y):
    f, Om = y

    df = 1.5*Om - 0.5*f*(4 - 3*Om + f)
    dOm = -3*(1-Om)*Om

    return [df, dOm]

##################################################
# Event functions
##################################################

# f' = 0
def event_df(N, y):
    f, Om = y
    return 1.5*Om - 0.5*f*(4 - 3*Om + f)

event_df.terminal = False
event_df.direction = 0


# Omega_m' = 0
def event_dOm(N, y):
    f, Om = y
    return -3*(1-Om)*Om

event_dOm.terminal = False
event_dOm.direction = 0


##################################################
# Initial conditions
##################################################

f0 = 1.0
Om0 = 0.999

y0 = [f0, Om0]

##################################################
# Integrate
##################################################

sol = solve_ivp(
    system,
    [0,6],
    y0,
    method='RK45',
    dense_output=True,
    rtol=1e-9,
    atol=1e-12,
    events=[event_df,event_dOm]
)

##################################################
# Plot
##################################################

N = sol.t
f = sol.y[0]
Om = sol.y[1]

plt.figure(figsize=(7,5))
plt.plot(N,f,label="Growth rate $f$")
plt.plot(N,Om,label=r"$\Omega_m$")
plt.xlabel(r"$N=\ln a$")
plt.grid(True)
plt.legend()
plt.show()


##################################################
# Phase portrait
##################################################

plt.figure(figsize=(6,6))
plt.plot(Om,f)

plt.scatter([1,0],[1,0],
            c='red',s=80,label='Critical points')

plt.xlabel(r'$\Omega_m$')
plt.ylabel(r'$f$')
plt.grid(True)
plt.legend()
plt.show()


##################################################
# Event outputs
##################################################

print("f'=0 events")
print(sol.t_events[0])

print()

print("Omega'=0 events")
print(sol.t_events[1])

##################################################
# Convert N -> a
##################################################

N = sol.t
a = np.exp(N)

f = sol.y[0]
Om = sol.y[1]

##################################################
# Growth rate vs scale factor
##################################################

plt.figure(figsize=(7,5))
plt.plot(a, f, lw=2)
plt.xlabel("Scale factor $a$")
plt.ylabel("Growth rate $f$")
plt.xscale('log')          # optional
plt.grid(True)
plt.tight_layout()
plt.show()

##################################################
# Matter density parameter vs scale factor
##################################################

plt.figure(figsize=(7,5))
plt.plot(a, Om, lw=2)
plt.xlabel("Scale factor $a$")
plt.ylabel(r"$\Omega_m$")
plt.xscale('log')          # optional
plt.grid(True)
plt.tight_layout()
plt.show()

##################################################
# Both quantities together
##################################################

plt.figure(figsize=(7,5))
plt.plot(a, f, label=r"$f$")
plt.plot(a, Om, label=r"$\Omega_m$")
plt.xlabel("Scale factor $a$")
plt.xscale('log')          # optional
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Cosmological parameters
Omega_m0 = 0.315
Omega_L0 = 0.685


# Coupled ODE system
def growth_system(x, y):
    """
    x = ln(a)
    y[0] = f
    y[1] = Omega_m
    """

    f, Om = y

    df_dx = (3/2)*Om - f*(0.5*(4 - 3*Om) + f)

    dOm_dx = -3*(1-Om)*Om

    return [df_dx, dOm_dx]


# Initial conditions
a_i = 0.01
x_i = np.log(a_i)

a_final = 1
x_final = np.log(a_final)

f_i = 1.0
Om_i = 0.999     # deep matter domination


# Solve ODE
solution = solve_ivp(
    growth_system,
    [x_i, x_final],
    [f_i, Om_i],
    dense_output=True,
    rtol=1e-10,
    atol=1e-10
)


# Create scale factor array
a = np.linspace(a_i, 1, 500)
x = np.log(a)


# Numerical solution
f_num, Om_num = solution.sol(x)


# Approximation f = Omega_m^0.55
Om_LCDM = (Omega_m0*a**(-3)) / (
    Omega_m0*a**(-3) + Omega_L0
)

f_approx = Om_LCDM**0.55


# Plot
plt.figure(figsize=(8,5))

plt.plot(
    a,
    f_num,
    label="Numerical solution",
    linewidth=2
)

plt.plot(
    a,
    f_approx,
    "--",
    label=r"$f=\Omega_m^{0.55}$",
    linewidth=2
)

plt.xlabel("Scale factor $a$")
plt.ylabel("Growth rate $f(a)$")

plt.title("$\Omega_m (a) $ and Growth rate in ΛCDM")

plt.legend()
plt.grid(True)

plt.xlim(0.01,1)

plt.show()

# Mass density
plt.figure(figsize=(8,5))

plt.plot(
    a,
    f_num,
    label="$\Omega_m$",
    linewidth=2
)

plt.xlabel("Scale factor $a$")
plt.ylabel("Mass density $\Omega_m(a)$")

plt.title("Mass density $\Omega_m(a)$ evolution in ΛCDM")

plt.legend()
plt.grid(True)

plt.xlim(0.01,1)

plt.show()



plt.figure(figsize=(8,5))
plt.plot(
    a,
    f_approx,
    "--",
    label=r"$f=\Omega_m^{0.55}$",
    linewidth=2
)

plt.xlabel("Scale factor $a$")
plt.ylabel("Growth rate $f(a)$")

plt.title("Growth rate evolution in ΛCDM")

plt.legend()
plt.grid(True)

plt.xlim(0.01,1)

plt.show()



