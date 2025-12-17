import numpy as np
import matplotlib.pyplot as plt
import time



def Cd_sphere(Re):
    """
    Sphere drag coefficient including drag crisis behavior.
    Valid from creeping flow through post-critical regime (~Re <= 1e7).
    """

    Re = np.maximum(Re, 1e-12)

    t1 = 24 / Re
    t2 = (2.6 * (Re/5)) / (1 + (Re/5)**1.52)
    t3 = (0.411 * (Re / 263000)**-7.94) / (1 + (Re / 263000)**-8.00)
    t4 = (Re**0.80) / 461000

    return t1 + t2 + t3 + t4


def air_dynamic_viscosity(T):
    """
    Dynamic viscosity of air using Sutherland's law.

    Parameters
    ----------
    T : float or array
        Temperature in Kelvin

    Returns
    -------
    mu : float or array
        Dynamic viscosity in Pa*s
    """
    # Reference values for air
    T0 = 273.15       # K
    mu0 = 1.716e-5   # Pa*s
    C = 111.0        # K (Sutherland constant)

    mu = mu0 * (T / T0)**(3/2) * (T0 + C) / (T + C)
    return mu

def air_density(P,T):
    R = 287.05 #J/(kg·K)
    rho = P/(R*T)

    return rho


def He_density(P,T):
    rho_He_std = 0.1785 #kg/m3 @ STP 
    b = rho_He_std*273.15/101325 # b = rho*T/P = M/R (Molar Mass/ Gass Constant)
    rho_He = b*P/T
    return rho_He



'''
D = 3 #Meters
A = np.pi*D**2/4
u = 5 #m/s

rho = 1.1 #kg/m3

T_test = 0  #C

T_in = T_test + 273.15 

nu = air_dynamic_viscosity(T_in)

Re = u*D / nu

Cd = Cd_sphere(Re)

f_D = 0.5*Cd* rho*(u**2)*A



u_vec = np.linspace(0,10,100)
Re_vec = u_vec*D/nu
Cd_vec = Cd_sphere(Re_vec)

f_D_vec = 0.5*Cd_vec*rho*(u_vec**2)*A


print("Dynamic Viscocity [m2/s2]",nu)
print("Reynolds Number ", Re)

print("Cd - Morrison Estimate ",Cd)

print("DragForce",f_D)

# Plot
plt.figure()
plt.plot(u_vec, f_D_vec)
plt.xlabel("Velocity (m/s)")
plt.ylabel("Drag Force (N)")
plt.title("Drag Force vs Velocity")
plt.show()

'''