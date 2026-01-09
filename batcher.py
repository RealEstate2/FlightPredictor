import tkinter
from Simulation import Simulation
import numpy as np
import time
import matplotlib.pyplot as plt
import copy


import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt


def input_parameters():
    timestep = 0.7 #s
    max_time = 3600*8 #s
    iter_max = int(np.floor(max_time/timestep))

    return {
        "launch_time": "2025-10-18 13:00",
        "latitude": 40.4462,
        "longitude": -104.6379,
        "altitude": 1500,
        "balloon_mass": 0.6,
        "burst_diameter": 6.0,
        #"helium_volume": 2,
        "neck_lift": 2.1,
        "payload_mass": 1.8,
        "time_step": timestep,
        "iter_max":iter_max,
        "DataSet":"gfs",
        "parachute_cd":1,
        "parachute_mass":0.1,
        "parachute_diameter":1.22
    }


#Number of randomly distributed points.
n_sims = 25

p_ideal = input_parameters()

#Establish which variables we want to randomize
neck_lift_i = p_ideal["neck_lift"]
payload_mass_i = p_ideal["payload_mass"]
lat_i = p_ideal["latitude"]
long_i = p_ideal["longitude"]

neck_lift_std = 0.1
payload_mass_std = .005
lat_std = 0.0005
long_std = 0.0005

diff_ideal = 0.174 #difference in lift vs mass

results_log = []
param_log = []
timer = []

print("Running ",n_sims,"simulations..." )
tic_full = time.time()
for i in range(n_sims):
    param = p_ideal.copy()

    #Neck lift distribution and 
    neck_lift_r = np.random.normal(loc=neck_lift_i,scale=neck_lift_std)
    #difference distribution
    diff_r = np.random.normal(loc= diff_ideal,scale=payload_mass_std)
    payload_mass_r = neck_lift_r-diff_r

    #payload_mass_r = np.random.normal(loc=payload_mass_i,scale=payload_mass_std)
    lat_r = np.random.normal(loc=lat_i,scale=lat_std)
    long_r = np.random.normal(loc=long_i,scale=long_std)

    param["neck_lift"] = neck_lift_r
    param["payload_mass"] = payload_mass_r
    param["latitude"] = lat_r
    param["longitude"] = long_r
    
    #print("This Is Run:",i)
    print("neck lift [kg]:",param["neck_lift"])
    print("payload mass [kg]:",param["payload_mass"])
    #print("latitude [dd]:",lat_r)
    #print("longitude [dd]:",long_r)

    tic = time.time()
    results,log = Simulation(param)
    results_log.append(results)
    param_log.append(copy.deepcopy(param))
    toc = time.time()

    timer.append(toc-tic)
    print("time to run sim",i," : ",timer[i])

print("time to run sim",i," : ",timer[i])
import csv
import numpy as np

# --- Great-circle distance (meters) ---
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = np.deg2rad(lat1); phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlmb = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def _to_float_or_nan(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _get_pos(obj, attr_name):
    """
    Returns (lat, lon, alt?) safely from e.g. obj.burst_pos or obj.land_pos.
    Your code shows burst_pos = [lat, lon] (no altitude). We'll handle both cases.
    """
    pos = getattr(obj, attr_name, None)
    if pos is None:
        return (np.nan, np.nan, np.nan)

    try:
        # list/tuple/np array
        lat = float(pos[0]) if len(pos) > 0 else np.nan
        lon = float(pos[1]) if len(pos) > 1 else np.nan
        alt = float(pos[2]) if len(pos) > 2 else np.nan
        return (lat, lon, alt)
    except Exception:
        return (np.nan, np.nan, np.nan)

def saveCSV(results, params_list, out_csv="flight_batch_summary.csv"):
    """
    Save important params + outcomes for each simulation result to a CSV.

    results: list of Simulation() outputs (each has burst_time, landing_time, burst_pos, land_pos, etc.)
    params_list: list of dicts used for each sim (your randomized params per run)
    """

    if len(results) != len(params_list):
        raise ValueError(f"results and params_list must be same length. Got {len(results)} vs {len(params_list)}")

    # Define exactly what “important stuff” is (you can add/remove fields here)
    param_fields = [
        "launch_time",
        "latitude", "longitude", "altitude",
        "balloon_mass", "payload_mass",
        "neck_lift",
        "burst_diameter",
        "time_step", "iter_max",
        "DataSet",
        "parachute_cd", "parachute_mass", "parachute_diameter",
    ]

    # CSV columns
    fieldnames = (
        ["sim_id"]
        + [f"p_{k}" for k in param_fields]
        + [
            "burst_time_s",
            "landing_time_s",
            "burst_lat", "burst_lon", "burst_alt_m",
            "land_lat",  "land_lon",  "land_alt_m",
            "drift_m",   # great-circle distance from launch to landing
        ]
    )

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (r, p) in enumerate(zip(results, params_list)):
            # launch reference (from this run's params)
            launch_lat = _to_float_or_nan(p.get("latitude", np.nan))
            launch_lon = _to_float_or_nan(p.get("longitude", np.nan))

            # positions (from result object)
            b_lat, b_lon, b_alt = _get_pos(r, "burst_pos")
            l_lat, l_lon, l_alt = _get_pos(r, "land_pos")

            # times (assumed seconds if numeric; otherwise NaN)
            t_burst_s = _to_float_or_nan(getattr(r, "burst_time", np.nan))
            t_land_s  = _to_float_or_nan(getattr(r, "landing_time", np.nan))

            # drift (great-circle from launch -> landing)
            drift_m = np.nan
            if np.isfinite(launch_lat) and np.isfinite(launch_lon) and np.isfinite(l_lat) and np.isfinite(l_lon):
                drift_m = float(haversine_m(launch_lat, launch_lon, l_lat, l_lon))

            row = {"sim_id": i}

            # write params (with prefix)
            for k in param_fields:
                row[f"p_{k}"] = p.get(k, np.nan)

            # write outputs
            row.update({
                "burst_time_s": t_burst_s,
                "landing_time_s": t_land_s,
                "burst_lat": b_lat, "burst_lon": b_lon, "burst_alt_m": b_alt,
                "land_lat":  l_lat, "land_lon":  l_lon, "land_alt_m":  l_alt,
                "drift_m": drift_m,
            })

            writer.writerow(row)

    print(f"Saved CSV -> {out_csv}")
    return out_csv
       


saveCSV(np.array(results_log), np.array(param_log))