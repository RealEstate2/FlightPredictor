from herbie import Herbie
import xarray as xr
import numpy as np
from func_lookup import Cd_sphere, air_dynamic_viscosity, air_density

# ======================================================
# USER INPUTS
# ======================================================



LAT = 40.554707
LON = -105.157157
if LON < 0:
    LON += 360

SEARCH_RADIUS = 0.10   # degrees (~11 km box around LAT/LON)


P = 825 #mb
PRESSURE_LEVEL = f"{P} mb"   # main level for horizontal flow
P = P*100 #Pa


R_AIR = 287.05              # J/(kg*K)

# ======================================================
# SPATIAL WINDOW
# ======================================================

LAT_MIN = LAT - SEARCH_RADIUS
LAT_MAX = LAT + SEARCH_RADIUS
LON_MIN = LON - SEARCH_RADIUS
LON_MAX = LON + SEARCH_RADIUS

# ======================================================
# VARIABLES TO PULL (PRESSURE-LEVEL)
# ======================================================

VARIABLES = [
    f"TMP:{PRESSURE_LEVEL}",    # temperature at level
    f"UGRD:{PRESSURE_LEVEL}",   # u-wind at level
    f"VGRD:{PRESSURE_LEVEL}",   # v-wind at level
    f"HGT:{PRESSURE_LEVEL}",    # geopotential height at level
    f"VVEL:{PRESSURE_LEVEL}",              # vertical velocity at 200 mb (just as extra info)
]

# ======================================================
# DOWNLOAD HRRR PRESSURE-LEVEL DATA
# ======================================================

H = Herbie(TIME, model="hrrr", product="prs", overwrite=True)

datasets = []
for var in VARIABLES:
    print(f"Pulling {var} ...")
    ds = H.xarray(var)
    datasets.append(ds)

dataset = xr.merge(datasets, compat="override")

# ======================================================
# SUBSET TO LOCAL REGION
# ======================================================

dataset = dataset.where(
    (dataset.latitude >= LAT_MIN) &
    (dataset.latitude <= LAT_MAX) &
    (dataset.longitude >= LON_MIN) &
    (dataset.longitude <= LON_MAX),
    drop=True
)

if dataset.sizes["x"] == 0 or dataset.sizes["y"] == 0:
    raise RuntimeError("No grid points found in local window. Increase SEARCH_RADIUS.")

# ======================================================
# FIND NEAREST GRID INDEX (2-D lat/lon → nearest y,x)
# ======================================================

lat2d = dataset["latitude"].values
lon2d = dataset["longitude"].values

dist2 = (lat2d - LAT)**2 + (lon2d - LON)**2
iy, ix = np.unravel_index(np.argmin(dist2), lat2d.shape)

pt = dataset.isel(y=iy, x=ix)

# ======================================================
# EXTRACT VARIABLES (standard cfgrib names)
# ======================================================
# Most Herbie/cfgrib HRRR-prs files use:
#   t  = temperature (K)
#   u  = u-wind (m/s)
#   v  = v-wind (m/s)
#   w  = vertical velocity (Pa/s)
#   gh = geopotential (m^2/s^2)

T = float(pt["t"].squeeze())      # K
u = float(pt["u"].squeeze())      # m/s
v = float(pt["v"].squeeze())      # m/s
gh = float(pt["gh"].squeeze())    # m^2/s^2  (for HGT:800 mb)
z = gh / 9.81                     # m, approximate geometric height
omega = float(pt["w"].squeeze())
# Convert pressure level string to Pascals
P = float(PRESSURE_LEVEL.split()[0]) * 100.0  # Pa

# Density from your helper (assumed ρ(P,T) with P in Pa, T in K)
rho = air_density(P, T)

w = -omega / (rho*9.81) 

wind_speed = np.sqrt(u**2 + v**2 + w**2)

# ======================================================
# ATMOSPHERIC PROPERTIES (ρ, μ)
# ======================================================


# Dynamic viscosity from your helper
# NOTE: if air_dynamic_viscosity expects °C, use (T - 273.15)
mu = air_dynamic_viscosity(T)

# ======================================================
# FLUID DYNAMICS: Re, Cd, Drag
# ======================================================


# Balloon / drag properties
# ------------------- # BALLOON/DRAG PARAMETERS # ------------------- 
d_burst = 6.03504 #Meters 
M_balloon = 0.6 #Kg 
V_He = 2 #m3 #Properties 


rho_He_std = 0.1785 #kg/m3 @ STP 
b = rho_He_std*273.15/101325 # b = rho*T/P = M/R (Molar Mass/ Gass Constant)
rho_He = b*P/T

M_He = V_He *rho_He


D_BALLOON =  (V_He*6/(np.pi))**(1/3)   # m
A_BALLOON = np.pi * (D_BALLOON / 2.0)**2

Re = rho * wind_speed * D_BALLOON / mu
Cd = Cd_sphere(Re)

F_drag = 0.5 * rho * wind_speed**2 * Cd * A_BALLOON

# ======================================================
# PLACEHOLDERS FOR NEXT STEPS
# ======================================================

buoyancy_force = rho*9.81*V_He
weight_force = M_He*9.81
payload_weight = 5  # Newtons
total_mass = (M_He + M_balloon+ payload_weight/9.81)
net_force = buoyancy_force - weight_force - payload_weight
acceleration_buouyancy = net_force / total_mass
acceleration_drag = F_drag/total_mass
velocity_next = None
position_next = None
time_step = None

#Calculations
wind_norm = np.array([u,v,w]/wind_speed)
a_wind = wind_norm*acceleration_drag
a_net = a_wind
a_net[2] += acceleration_buouyancy



# ======================================================
# SNAPSHOT OUTPUT
# ======================================================

print("\n====== LOCAL COLUMN (one step) ======")
print(f"Level          : {PRESSURE_LEVEL}")
print(f"Altitude (MSL) : {z:.1f} m")
print(f"T              : {T:.2f} K")
print(f"P              : {P:.1f} Pa (from level)")
print(f"ρ (density)    : {rho:.4f} kg/m³")
print(f"μ (viscosity)  : {mu:.3e} Pa·s")
print(f"u, v, w           : {u:.2f}, {v:.2f},{w:.2f} m/s")
print(f"|V|            : {wind_speed:.2f} m/s")
print("wind Direction : ", wind_norm)


print("\n====== FLOW / DRAG ======")
print(f"Diameter       : {D_BALLOON} m")
print(f"Re             : {Re:.3e}")
print(f"Cd             : {Cd}")
print(f"F_drag         : {F_drag:.3f} N")
print("a_drag          :",acceleration_drag)

print("\n====== PLACEHOLDER STATE ======")
print("total mass = ",total_mass)
print("buoyancy_force =", buoyancy_force)
print("weight_force   =", weight_force)
print("net_force      =", net_force)
print("a_buoyancy     =", acceleration_buouyancy)

print("net acceleration =",a_net)
print("velocity_next  =", velocity_next)
print("position_next  =", position_next)
