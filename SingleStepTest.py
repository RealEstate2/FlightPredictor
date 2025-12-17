# ===========================================
# HIGH ALTITUDE BALLOON FLIGHT MODEL (Skeleton)
# ===========================================

import numpy as np
from func_lookup import He_density, air_density, air_dynamic_viscosity, Cd_sphere
from atmosphere import standardAtmosphere

from herbie import Herbie
import xarray as xr


# ====================================================
# 1. USER INPUTS
# ====================================================
def input_parameters():
    return {
        "launch_time": "2025-11-29 12:00",
        "latitude": 40.0547,
        "longitude": -105.2571,
        "altitude": 1500,
        "balloon_mass": 0.6,
        "burst_diameter": 6.0,
        "helium_volume": 2.0,
        "payload_mass": 0.5,
        "time_step": 1.0
    }


# ====================================================
# 2. INITIAL CONDITIONS @ SURFACE
# ====================================================
class BalloonState:

    def __init__(self, params):
        self.params = params
        self.init_state()


    def init_state(self):

        p = self.params

        self.timestep = 1 #s
        self.t_sim = 0
        self.t_real = p["launch_time"]

        self.position = np.array([0.0, 0.0, p["altitude"]])
        self.velocity = np.zeros(3)

        P, T, rho, g = standardAtmosphere.qualities(p["altitude"])

        self.P_air = P*1000
        self.T_air = T
        self.rho_air = rho
        self.g = g

        self.rho_He = He_density(self.P_air, self.T_air)
        self.volume_He = p["helium_volume"]
        self.diameter  = (6*self.volume_He/np.pi)**(1/3)
        self.CSA = np.pi/4*self.diameter**2
        self.mass_He = self.rho_He * self.volume_He

        self.F_buoyant = self.rho_air * self.g * self.volume_He

        self.mass = (
            p["balloon_mass"]
            + p["payload_mass"]
            + self.mass_He
        )

        self.F_weight = self.mass * self.g
        self.Net_Lift = self.F_buoyant - self.F_weight

    def report(self):

        speed = np.linalg.norm(self.velocity)

        print("\n========= BALLOON FLIGHT REPORT =========")
        print(f"Simulated Time    : {self.t_sim:10.1f} s")
        print(f"Altitude          : {self.position[2]:10.2f} m")

        print("\n--- Atmospheric State ---")
        print(f"Temperature       : {self.T_air:10.2f} K")
        print(f"Pressure          : {self.P_air:10.2f} Pa")
        print(f"Density           : {self.rho_air:10.4f} kg/m³")
        print(f"Gravity           : {self.g:10.5f} m/s²")

        print("\n--- Balloon System ---")
        print(f"Helium Density    : {self.rho_He:10.4f} kg/m³")
        print(f"Helium Volume     : {self.volume_He:10.3f} m³")
        print("Ballon Diameter   :",)
        print(f"Helium Mass       : {self.mass_He:10.3f} kg")
        print(f"Total Mass        : {self.mass:10.3f} kg")

        print("\n--- Forces ---")
        print(f"Buoyancy Force    : {self.F_buoyant:10.3f} N")
        print(f"Weight Force      : {self.F_weight:10.3f} N")
        print(f"Net Vertical      : {self.Net_Lift:10.3f} N")

        print("\n--- Kinematics ---")
        print(f"Velocity Vector   : {self.velocity}")
        print(f"Speed             : {speed:10.3f} m/s")
        print(f"Position Vector   : {self.position}")

        print("========================================")



# ====================================================
# 3. ATMOSPHERE LOOKUP (MODEL STUB)
# ====================================================
# ====================================================
# HRRR FIELD CACHE
# ====================================================

_HRRR_FIELD_CACHE = {}

def load_hrrr_pressure_field(TIME, P_pa, lat0, lon0, box_deg=0.25):

    if lon0 < 0:
        lon0 += 360

    discrete_levels = [
        50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300,
        325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575,
        600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850,
        875, 900, 925, 950, 975, 1000
    ]

    P_mb = snap_to_nearest(P_pa / 100, discrete_levels)
    PRESSURE_LEVEL = f"{P_mb} mb"
    KEY = (TIME, P_mb, box_deg)

    # ✅ Load once
    if KEY not in _HRRR_FIELD_CACHE:
        print(f"📡 Downloading HRRR pressure field {P_mb} mb @ {TIME}")

        H = Herbie(TIME, model="hrrr", product="prs")

        vars = [
            f"TMP:{PRESSURE_LEVEL}",
            f"UGRD:{PRESSURE_LEVEL}",
            f"VGRD:{PRESSURE_LEVEL}",
            f"VVEL:{PRESSURE_LEVEL}",
            f"HGT:{PRESSURE_LEVEL}"
        ]

        ds = xr.merge([H.xarray(v) for v in vars], compat="override")

        # Spatial window
        ds = ds.where(
            (ds.latitude  >= lat0 - box_deg) &
            (ds.latitude  <= lat0 + box_deg) &
            (ds.longitude >= lon0 - box_deg) &
            (ds.longitude <= lon0 + box_deg),
            drop=True
        )

        _HRRR_FIELD_CACHE[KEY] = ds

    return _HRRR_FIELD_CACHE[KEY]

def sample_field(field, lat, lon):

    if lon < 0:
        lon += 360

    lat2d = field.latitude.values
    lon2d = field.longitude.values

    dist2 = (lat2d - lat)**2 + (lon2d - lon)**2
    iy, ix = np.unravel_index(np.argmin(dist2), lat2d.shape)

    p = field.isel(y=iy, x=ix)

    return {
        "T": float(p.t),
        "u": float(p.u),
        "v": float(p.v),
        "w": float(p.w),
        "z": float(p.gh)
    }


def pos_to_coords(state,params):
    lat_i = params["latitude"]
    long_i = params["latitude"]

def snap_to_nearest(value, levels):
    return min(levels, key=lambda x: abs(x - value))


# ====================================================
# 4. FORCE MODELS
# ====================================================
def compute_drag(balloon, air_vel, air_T):
    #Compute Relative Velocity
    vel_balloon = balloon.velocity
    vel_air = air_vel

    relative_vel = vel_air - vel_balloon
    print("relative Velocity",relative_vel)
    rel_speed  = np.linalg.norm(relative_vel)
    direction = relative_vel/rel_speed
    print("Direction",direction)

    #Compute Fluid Properties
    mu = air_dynamic_viscosity(air_T )

    Re = balloon.diameter * rel_speed / mu
    
    Cd = Cd_sphere(Re)
    print("Cd Sphere",Cd)
    D_mag = 0.5*Cd*balloon.rho_air*balloon.CSA*rel_speed**2
    print("D_Mag",D_mag)

    Drag = D_mag *direction

    return Drag


# ====================================================
# 5. ACCELERATION
# ====================================================

def compute_acceleration(forces, balloon):
    return (forces / balloon.mass)


# ====================================================
# 6. RK4 INTEGRATOR (Clean Core)
# ====================================================
# ====================================================
# 6. RK4 INTEGRATOR (WORKING CORE)
# ====================================================

def rk4_step(balloon, a_net):
    dt = 0.5

    # --- current state vectors ---
    x0 = balloon.position.copy()
    v0 = balloon.velocity.copy()

    # --- derivatives ---
    def dxdt(v): return v
    def dvdt(): return a_net   # acceleration is constant per step for now

    # ---- RK4 VECTORS ----
    k1x = dxdt(v0)
    k1v = dvdt()

    k2x = dxdt(v0 + 0.5 * dt * k1v)
    k2v = dvdt()

    k3x = dxdt(v0 + 0.5 * dt * k2v)
    k3v = dvdt()

    k4x = dxdt(v0 + dt * k3v)
    k4v = dvdt()

    # ---- UPDATE STATE ----
    balloon.position += dt/6 * (k1x + 2*k2x + 2*k3x + k4x)
    balloon.velocity += dt/6 * (k1v + 2*k2v + 2*k3v + k4v)

    # Advance time
    balloon.t_sim += dt

    return balloon


'''
# ====================================================
# 7. PROPERTY UPDATES
# ====================================================

def recompute_properties(state, atmosphere, params):
    """
    Allow balloon expansion, pressure change, gas density update.
    """
    pass


# ====================================================
# 8. TERMINATION CONDITION
# ====================================================

def termination_condition(state, atmosphere):
    """
    Stop at burst altitude, ground contact, or condition flags.
    """
    return False

'''

# ====================================================
# 9. MAIN SIMULATION LOOP
# ====================================================

def simulate():
    params = input_parameters()
    bloon = BalloonState(params)

    print("StartingPressure",bloon.P_air)

    lat_i = bloon.params["latitude"]
    long_i = bloon.params["longitude"]


    field = load_hrrr_pressure_field(
    bloon.t_real,
    bloon.P_air,
    bloon.params["latitude"],
    bloon.params["longitude"],
    box_deg=0.3)

    atm = sample_field(field, lat_i, long_i)
    
    w_vel = atm["w"]/(bloon.rho_air*bloon.g)

    wind_vel = np.array([atm["u"],atm["v"],w_vel])

    Drag = compute_drag(bloon,wind_vel,atm["T"])

    F_Net = Drag + np.array([0,0,bloon.Net_Lift])

    a_Net = compute_acceleration(F_Net,bloon)

    rk4_step(bloon,a_Net)

    print("HRRR Field Sample:")
    print("Temp:", atm["T"])
    print("U:", atm["u"])
    print("V:", atm["v"])
    print("Omega:", atm["w"])
    print("Height:", atm["z"])
    print("Drag Vector", Drag)
    print("Net Force",F_Net)
    print("Net Acceleration", a_Net)

    return bloon


# ====================================================
# EXECUTION
# ====================================================

if __name__ == "__main__":
    balloon = simulate()
    balloon.report()
    

    
