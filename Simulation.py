# ===========================================
# HIGH ALTITUDE BALLOON FLIGHT MODEL
# ===========================================

import numpy as np
from func_lookup import He_density, air_density, air_dynamic_viscosity, Cd_sphere
from HRRRSlicer import Slice_HRRR
from GFSSlicer_o import Slice_GFS

from atmosphere import standardAtmosphere
import matplotlib.pyplot as plt

from herbie import Herbie
import xarray as xr
import time


import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

from line_profiler import LineProfiler

lp = LineProfiler()

# ====================================================
# 1. USER INPUTS
# ====================================================
#CSU-CHILL COordinates
#"latitude": 40.4462,
#"longitude": -104.6379,


def input_parameters():
    timestep = 0.7 #s
    max_time = 3600*8 #s
    iter_max = int(np.floor(max_time/timestep))

    return {
        "launch_time": "2025-12-18 12:00",
        "latitude": 40.4462,
        "longitude": -104.6379,
        "altitude": 1500,
        "balloon_mass": 0.6,
        "burst_diameter": 6.0,
        #"helium_volume": 2,
        "neck_lift": 2,
        "payload_mass": 1.85,
        "time_step": timestep,
        "iter_max":iter_max,
        "DataSet":"gfs",
        "parachute_cd":1,
        "parachute_mass":0.1,
        "parachute_diameter":1.22
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

        self.timestep = p["time_step"] #s
        self.t_sim = 0
        self.t_real = p["launch_time"]
        self.iter_max = p["iter_max"]

        self.lat = p["latitude"]
        self.long = p["longitude"]
        self.position = np.array([0.0, 0.0, p["altitude"]])

        self.parachute_cd = p["parachute_cd"]
        self.parachute_mass = p["parachute_mass"]
        self.parachute_diameter = p["parachute_diameter"]
        self.parachute_area = self.parachute_diameter**2*np.pi/4

        self.velocity = np.zeros(3)
        
        #Atmospheric Conditions
        P, T, rho, g = standardAtmosphere.qualities(p["altitude"])

        self.P_air = P*1000
        self.T_air = T
        self.rho_air = rho
        self.g = g

        self.rho_He = He_density(self.P_air, self.T_air)
        rho_diff = (self.rho_air - self.rho_He)

        #When Neck lift is attained
        self.neck_lift = p["neck_lift"] #In units Kg of Force
        total_force = (self.neck_lift+p["balloon_mass"])


        self.volume_He = total_force/rho_diff

        #When Fill is Controlled
        #self.volume_He = p["helium_volume"]

        self.diameter  = (6*self.volume_He/np.pi)**(1/3)
        self.CSA = np.pi/4*self.diameter**2
        self.mass_He = self.volume_He*self.rho_He

        self.F_buoyant = self.rho_air * self.g * self.volume_He

        self.mass = (
            p["balloon_mass"]
            + p["payload_mass"]
            + self.mass_He
            + self.parachute_mass
        )

        self.F_weight = self.mass * self.g
        self.Net_Lift = self.F_buoyant - self.F_weight 

        self.burst_time = None
        self.landing_time = None

        self.burst_pos = None
        self.land_pos = None

    def update_state(self,atmosphere):
        # Atmospheric Conditions at current altitude [m]
        P_kPa, T, rho, g = standardAtmosphere.qualities(self.position[2])

        # Convert and store
        self.P_air  = P_kPa * 1000.0   # Pa
        self.T_air  = T
        self.rho_air = rho
        self.g       = g

        #self.T_air = atmosphere.temp
        #self.rho_air = self.P_air/(287*self.T_air)

        # Helium properties (assuming He_density(P [Pa], T [K]))
        self.rho_He = He_density(self.P_air, self.T_air)
        self.volume_He = self.mass_He / self.rho_He
        self.diameter  = (6 * self.volume_He / np.pi)**(1/3)
        self.CSA       = np.pi / 4 * self.diameter**2

        # Forces
        self.F_buoyant = self.rho_air * self.g * self.volume_He
        self.F_weight  = self.mass * self.g
        self.Net_Lift  = self.F_buoyant - self.F_weight

    def update_mass(self):
        p = self.params
        self.mass = (
            p["balloon_mass"]/2
            + p["payload_mass"]
            + self.parachute_mass
        )

    def report(self):
        

        print("\n========= PRE FLIGHT REPORT =========")
        print("\n--- Initial Atmospheric State ---")
        print(f"Temperature       : {self.T_air:10.2f} K")
        print(f"Pressure          : {self.P_air:10.2f} Pa")
        print(f"Density           : {self.rho_air:10.4f} kg/m³")
        print(f"Gravity           : {self.g:10.5f} m/s²")

        print("\n--- Balloon System ---")
        print(f"Helium Density    : {self.rho_He:10.4f} kg/m³")
        print(f"Helium Volume     : {self.volume_He:10.3f} m³")
        print( "Balloon Neck Lift :",self.neck_lift, "kgF")
        print( "Ballon Diameter i :",self.diameter, "m")
        print(f"Helium Mass       : {self.mass_He:10.3f} kg")
        print( "Balloon Mass      :",self.params["balloon_mass"])
        print( "Payload Mass      :",self.params["payload_mass"]+self.params["parachute_mass"])

        print(f"Total Mass        : {self.mass:10.3f} kg")

        print("\n--- Forces ---")
        print(f"Buoyancy Force    : {self.F_buoyant:10.3f} N")
        print(f"Weight Force      : {self.F_weight:10.3f} N")
        print(f"Net Vertical      : {self.Net_Lift:10.3f} N")

        print("========================================")
    


class AtmosphereState:
    def __init__(self, params):
        self.params = params
        self.sample_init()

    def sample_init(self):
        p = self.params
        LAT = p["latitude"]
        LONG = p["longitude"]
        ALT_I = p["altitude"]
        DT_I = p["launch_time"]

        dataset= p["DataSet"]
        if  dataset == "gfs":
            u,v,w,z,T,agl = Slice_GFS(LAT,LONG,ALT_I,0,DT_I)
        elif dataset == "hrrr":
            u,v,w,z,T,agl = Slice_HRRR(LAT,LONG,ALT_I,0,DT_I)
        else:
            print("No Dataset Selected")

        self.uvel = u
        self.vvel = v
        self.wvel = w
        self.gh = z
        self.temp = T
        self.agl = agl

        
    
    def sample_update(self,balloon):

        LAT,LONG,ALT,dt=balloon.lat,balloon.long,balloon.position[2],balloon.t_sim
        p = self.params
        DT_I = p["launch_time"]
        

        dataset= p["DataSet"]
        if  dataset == "gfs":
            u,v,w,z,T,agl = Slice_GFS(LAT,LONG,ALT,dt,DT_I)
        elif dataset == "hrrr":
            u,v,w,z,T,agl = Slice_HRRR(LAT,LONG,ALT,dt,DT_I)
        

        self.uvel = u
        self.vvel = v
        self.wvel = w
        self.gh = z
        self.temp = T
        self.agl = agl

def compute_lift(balloon, atm):
    """
    Compute net force and acceleration acting on the balloon.

    Inputs:
        balloon : BalloonState instance
        atm     : AtmosphereState instance (returned by Slice or similar)

    Returns:
        F_net : np.array(3,)
        a_net : np.array(3,)
    """

    # --- Build air velocity vector ---
    w_vel = -1*atm.wvel / (balloon.rho_air * balloon.g)    # Pa/s → m/s
    wind_vel = np.array([atm.uvel, atm.vvel, w_vel])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    # --- Relative velocity ---
    v_rel = (wind_vel - balloon.velocity)
    speed = np.linalg.norm(v_rel) + 1e-12      # avoid div by zero
    direction = v_rel / speed

    # --- Fluid properties ---
    mu = air_dynamic_viscosity(atm.temp)

    # --- Reynolds number & Drag ---
    Re = balloon.diameter * speed / mu
    Cd = Cd_sphere(Re)

    D_mag = 0.5 * Cd * balloon.rho_air * balloon.CSA * speed**2
    F_drag = D_mag * direction

    # --- Buoyancy ---
    F_buoy = np.array([0, 0, balloon.rho_air * balloon.g * balloon.volume_He])

    # --- Weight ---
    F_weight = np.array([0, 0, -balloon.mass * balloon.g])

    # --- Net force ---
    F_net = F_drag + F_buoy + F_weight

    # --- Acceleration ---
    a_net = F_net / balloon.mass

    return F_net, a_net

def compute_fall(balloon, atm):
    # --- Build air velocity vector ---
    w_vel = -1*atm.wvel / (balloon.rho_air * balloon.g)    # Pa/s → m/s
    wind_vel = np.array([atm.uvel, atm.vvel, w_vel])

    # --- Relative velocity ---
    v_rel = wind_vel-balloon.velocity 
    speed = np.linalg.norm(v_rel) + 1e-12      # avoid div by zero
    direction = v_rel / speed

    Cd = balloon.parachute_cd
    D_mag = 0.5 * Cd * balloon.rho_air * balloon.parachute_diameter * speed**2

    F_drag = D_mag * direction

    # --- Weight ---
    F_weight = np.array([0, 0, -balloon.mass * balloon.g])

    # --- Net force ---
    F_net = F_drag + F_weight

    # --- Acceleration ---
    a_net = F_net / balloon.mass

    return F_net, a_net


def rk4_step(balloon, a_net):
    dt = balloon.timestep

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



def relative_to_latlon(lat0_deg, lon0_deg, dx, dy):
    """
    Convert relative Cartesian motion (meters) to lat/lon coordinates.

    Parameters:
        lat0_deg : float
            Initial latitude  (degrees)
        lon0_deg : float
            Initial longitude (degrees)
        dx : float
            Eastward displacement (meters)
        dy : float
            Northward displacement (meters)

    Returns:
        lat_deg, lon_deg : float
            Updated latitude and longitude in degrees
    """

    # Earth's radius (meters)
    R = 6371000.0

    # Convert base to radians
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)

    # Latitude update
    lat = lat0 + dy / R

    # Longitude update (scaled by latitude)
    lon = lon0 + dx / (R * np.cos(lat0))

    # Convert back to degrees
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)

    return lat_deg, lon_deg


def Simulation(params):

    #Establish Initial Conditions
    balloon = BalloonState(params)
    atmosphere = AtmosphereState(params)

    burst_d = balloon.params["burst_diameter"]

    #print("Starting Altitude Entered:" , params["altitude"])
    #print("StartingPressure: ATM Model",balloon.P_air)
    #print("Starting Alttitude HRRR:", atmosphere.gh)


    #Ready Simulation to be Ran
    iter_max = balloon.iter_max
    #Empty position matricies
    pos_log = np.zeros([3,iter_max])
    coord_log = np.zeros([3,iter_max])

    #Set Initial Position
    pos_log = np.array(balloon.position)
    #Set Initial Coordinates
    lat_0,long_0 = balloon.params["latitude"],balloon.params["longitude"]
    coord_log= np.array([lat_0,long_0,balloon.params["altitude"]])

    #balloon.report()
    if balloon.Net_Lift < 0:
        balloon.burst_time = 0
        balloon.landing_time = 0

        balloon.burst_pos = np.array([lat_0,long_0,balloon.params["altitude"]])
        balloon.land_pos = np.array([lat_0,long_0,balloon.params["altitude"]])
        
        print("[Simulation Failed]: Lift < 0")
        return balloon

    #Set Initial Velocity
    vel_log = np.array(balloon.velocity)

    
    time_i = time.time()
    #Ascent
    for iter in range(1,iter_max):
        if balloon.diameter > burst_d :
            break
        
        #Compute Current Set of Forces
        F_net, a_net = compute_lift(balloon,atmosphere)

        #Update Position During Step Based on Velocity and Acceleration
        rk4_step(balloon,a_net)

        #Log Position After Step
        pos_log = np.vstack((pos_log,balloon.position))
        vel_log = np.vstack((vel_log,balloon.velocity))

        #Change is position from this iteration to last
        d_pos= pos_log[iter] - pos_log[iter - 1]
        dx = d_pos[0]
        dy = d_pos[1]

        #Update Latitude and Longitude Change to Balloon 
        balloon.lat,balloon.long = relative_to_latlon(balloon.lat,balloon.long,dx,dy)
        #Store Relative Motion in Coord Log
        coord_log = np.vstack((coord_log,[balloon.lat,balloon.long,balloon.position[2]]))

        #Recompute Atmospheric Conditions
        balloon.t_sim = iter* balloon.timestep
        atmosphere.sample_update(balloon)
        balloon.update_state(atmosphere)


    #print("================Burst Report===================")
    #print("Average Ascent[m/s]:",(balloon.position[2]-balloon.params["altitude"])/balloon.t_sim)
    #print("Max Altitude[m]    :",balloon.position[2])
    #print("Burst Coords       :",balloon.lat,balloon.long)
    #print("Burst Time[s]      :",balloon.t_sim )
    #hr = np.floor(balloon.t_sim/3600)
    #min = np.floor((balloon.t_sim-hr*3600)/60)
    #sec = np.floor((balloon.t_sim-hr*3600-min*60))
    #print("Burst Time[Hr-M-S] :",hr,"-",min,"-",sec)
    #print("Final Pressure[Pa] :",balloon.P_air)
    #print("Final Diameter     :",balloon.diameter)

    iter_burst = iter
    balloon.burst_time = balloon.t_sim
    balloon.burst_pos =  balloon.lat,balloon.long,balloon.position[2]
    balloon.update_mass()
    #Descent
    for iter in range(iter_burst,iter_max):
        if atmosphere.agl < 0:
            break
        #print(atmosphere.agl)

        #Compute Current Set of Forces
        F_net, a_net = compute_fall(balloon,atmosphere)

        #Update Position During Step Based on Velocity and Acceleration
        rk4_step(balloon,a_net)

        #Log Position After Step
        pos_log = np.vstack((pos_log,balloon.position))
        vel_log = np.vstack((vel_log,balloon.velocity))

        #Change is position from this iteration to last
        d_pos= pos_log[iter] - pos_log[iter - 1]
        dx = d_pos[0]
        dy = d_pos[1]

        #Update Position Change to Balloon Coordinate Change
        balloon.lat,balloon.long = relative_to_latlon(balloon.lat,balloon.long,dx,dy)
        #Store Relative Motion in Coord Log
        coord_log = np.vstack((coord_log,[balloon.lat,balloon.long,balloon.position[2]]))

        #Recompute Atmospheric Conditions
        balloon.t_sim = iter* balloon.timestep
        atmosphere.sample_update(balloon)
        balloon.update_state(atmosphere)

    #print("==============Landing Report===================")
    #print("Average Descent[m/s] :",(balloon.position[2]-burst_pos[2])/(balloon.t_sim-burst_time))
    #print("Final Altitude[m]    :",balloon.position[2])
    #print("Burst Coords         :",balloon.lat,balloon.long)
    #print("Impact Velocity [m/s]:",balloon.velocity)
    #print("Final Pressure[Pa]   :",balloon.P_air)
    #print("Landing Time[s]      :",balloon.t_sim )
    #hr = np.floor(balloon.t_sim/3600)
    #min = np.floor((balloon.t_sim-hr*3600)/60)
    #sec = np.floor((balloon.t_sim-hr*3600-min*60))
    #print("Burst Time[Hr-M-S]   :",hr,"-",min,"-",sec)

    time_f = time.time()
    balloon.landing_time = time_f
    #print("Time To Run[s]:",time_f-time_i)
    balloon.land_pos =  balloon.lat,balloon.long,balloon.position[2]
    
    return balloon,coord_log

if __name__ == "__main__":
    params = input_parameters()

    # profile the whole Simulation function
    lp_wrapper = lp(Simulation)
    balloon = lp_wrapper(params)

    lp.print_stats()



  