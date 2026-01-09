from Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import time

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

from Utilities.GreatCircle import great_circle_distance

import csv


def input_parameters():
    timestep = 0.7 #s
    max_time = 3600*8 #s
    iter_max = int(np.floor(max_time/timestep))

    lift = 2.5
    diff = 0.15

    return {
        "launch_time": "2026-01-01 13:00",
        "latitude": 40.4462,
        "longitude": -104.6379,
        "altitude": 1500,
        "balloon_mass": 0.6,
        "burst_diameter": 5.9,
        #"helium_volume": 2,
        "neck_lift": lift,
        "payload_mass": lift-diff,
        "time_step": timestep,
        "iter_max": iter_max,
        "DataSet": "gfs",
        "parachute_cd": 0.7,
        "parachute_mass": 0.1,
        "parachute_diameter": 1.22
    }


# Real balloon landing (comparison target)
lat_i  = 39.417
long_i = -101.8234

params = input_parameters()

tic = time.time()
results, coords = Simulation(params)
toc = time.time()
print("TTR [s]:",toc-tic)

coords = np.asarray(coords)
if coords.ndim != 2 or coords.shape[1] < 3:
    raise ValueError(f"Expected coords as Nx3 [lat, lon, alt]. Got shape {coords.shape}")

# --- take every 50th sample ---
coords_thinned = coords[::500]

# Header
if coords_thinned.shape[1] == 2:
    header = "lat,lon"
elif coords_thinned.shape[1] == 3:
    header = "lat,lon,alt"
else:
    header = ",".join([f"col{i}" for i in range(coords_thinned.shape[1])])

np.savetxt(
    "coords.csv",
    coords_thinned,
    delimiter=",",
    header=header,
    comments=""
)

lat = coords[:, 0].astype(float)
lon = coords[:, 1].astype(float)
alt = coords[:, 2].astype(float)

# Build time array
dt = float(params["time_step"])
t_s = np.arange(len(lat)) * dt
t_hr = t_s / 3600.0

# Compute along-track distance (cumulative)
# Uses your great_circle_distance (assumed meters). If it returns km, adjust labels below.
dist_step = np.zeros(len(lat))
for i in range(1, len(lat)):
    dist_step[i] = great_circle_distance(lat[i-1], lon[i-1], lat[i], lon[i])
dist_cum = np.cumsum(dist_step)

# Identify burst index as max altitude
i_burst = int(np.nanargmax(alt))
burst_lat, burst_lon, burst_alt = lat[i_burst], lon[i_burst], alt[i_burst]

# Identify landing index as last point (your existing logic)
i_land = len(lat) - 1
land_lat, land_lon, land_alt = lat[i_land], lon[i_land], alt[i_land]

# Error distance vs real landing
distance_err = great_circle_distance(lat_i, long_i, land_lat, land_lon)
print("Distance error to real landing:", distance_err)
print("Final altitude (ground level-ish):", land_alt)

print("Max Altitude:",results.burst_pos[2])

# -----------------------------
# Plot 1: Altitude vs Time
# -----------------------------
plt.figure()
plt.plot(t_hr, alt)
plt.scatter(t_hr[i_burst], alt[i_burst], marker="x", s=80)
plt.scatter(t_hr[i_land], alt[i_land], marker="o", s=80)
plt.xlabel("Time (hours)")
plt.ylabel("Altitude (m)")
plt.title("Altitude Profile vs Time (burst marked with X, landing with O)")
plt.grid(True)

# -----------------------------
# Plot 2: Altitude vs Along-Track Distance
# -----------------------------
plt.figure()
# If great_circle_distance is meters, this is km:
dist_km = dist_cum / 1000.0
plt.plot(dist_km, alt)
plt.scatter(dist_km[i_burst], alt[i_burst], marker="x", s=80)
plt.scatter(dist_km[i_land], alt[i_land], marker="o", s=80)
plt.xlabel("Along-track distance (km)")
plt.ylabel("Altitude (m)")
plt.title("Altitude Profile vs Distance (burst marked with X, landing with O)")
plt.grid(True)

# -----------------------------
# Plot 3: Ground Track Map (colored by altitude)
# -----------------------------
tiler = cimgt.OSM()
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(1, 1, 1, projection=tiler.crs)

# Map bounds with padding
pad = 0.4
extent = [
    float(np.nanmin(lon) - pad),
    float(np.nanmax(lon) + pad),
    float(np.nanmin(lat) - pad),
    float(np.nanmax(lat) + pad),
]
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.add_image(tiler, 9)

ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.coastlines(resolution="10m")

# Track colored by altitude
sc = ax.scatter(
    lon, lat,
    c=alt,
    s=10,
    transform=ccrs.PlateCarree()
)
plt.colorbar(sc, ax=ax, label="Altitude (m)")

# Launch marker
ax.scatter(params["longitude"], params["latitude"], marker="*", s=180, transform=ccrs.PlateCarree())
ax.text(params["longitude"], params["latitude"], "Launch", transform=ccrs.PlateCarree())

# Burst marker
ax.scatter(burst_lon, burst_lat, marker="X", s=140, transform=ccrs.PlateCarree())
ax.text(burst_lon, burst_lat, "Burst", transform=ccrs.PlateCarree())

# Predicted landing marker
ax.scatter(land_lon, land_lat, marker="o", s=140, transform=ccrs.PlateCarree())
ax.text(land_lon, land_lat, "Pred Landing", transform=ccrs.PlateCarree())

# Real landing marker (comparison)
ax.scatter(long_i, lat_i, marker="s", s=140, transform=ccrs.PlateCarree())
ax.text(long_i, lat_i, "Real Landing", transform=ccrs.PlateCarree())

ax.set_title("Ground Track (colored by altitude) + Launch/Burst/Landing markers")

plt.show()
