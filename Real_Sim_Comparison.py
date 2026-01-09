from Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

from Utilities.GreatCircle import great_circle_distance
import csv


def input_parameters():
    timestep = 0.7 # s
    max_time = 3600 * 8 # s
    iter_max = int(np.floor(max_time / timestep))

    lift = 2.2
    diff = 0.172

    return {
        "launch_time": "2025-10-18 13:00",
        "latitude": 40.4462,
        "longitude": -104.6379,
        "altitude": 1500,
        "balloon_mass": 0.6,
        "burst_diameter": 5.9,
        "neck_lift": lift,
        "payload_mass": lift - diff,
        "time_step": timestep,
        "iter_max": iter_max,
        "DataSet": "gfs",
        "parachute_cd": 0.7,
        "parachute_mass": 0.1,
        "parachute_diameter": 1.22
    }


# -----------------------------
# Helpers
# -----------------------------
def parse_timestamp(s):
    """
    Example: '10/18/2025 9:02'
    If your file sometimes has seconds, we try a couple formats.
    """
    s = str(s).strip()
    fmts = [
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %I:%M %p",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None


def decimate_to_target(arr, target_n):
    """
    Keep about target_n evenly-spaced samples from a 1D or 2D array along axis 0.
    """
    n = len(arr)
    if n <= target_n:
        return arr
    idx = np.linspace(0, n - 1, target_n).astype(int)
    return arr[idx]


def load_real_flight_csv(path, target_points=2000):
    """
    Loads real flight CSV with columns:
    timestamp, latitude (dd), longitude (dd), altitude (m), ...

    Returns:
      t_hr_real, lat_real, lon_real, alt_real
      dist_km_real
    Downsamples to ~target_points for plotting.
    """
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # pull and parse
            ts = parse_timestamp(row.get("timestamp", ""))
            if ts is None:
                continue

            try:
                lat = float(str(row.get("latitude (dd)", "")).strip())
                lon = float(str(row.get("longitude (dd)", "")).strip())
                alt = float(str(row.get("altitude (m)", "")).strip())
            except Exception:
                continue

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue

            rows.append((ts, lat, lon, alt))

    if len(rows) < 2:
        raise ValueError(f"Not enough valid rows found in {path}")

    # sort by time
    rows.sort(key=lambda x: x[0])

    ts = np.array([r[0] for r in rows], dtype=object)
    lat = np.array([r[1] for r in rows], dtype=float)
    lon = np.array([r[2] for r in rows], dtype=float)
    alt = np.array([r[3] for r in rows], dtype=float)

    # time axis in hours since start
    t0 = ts[0]
    t_s = np.array([(t - t0).total_seconds() for t in ts], dtype=float)
    t_hr = t_s / 3600.0

    # along-track distance
    dist_step = np.zeros(len(lat))
    for i in range(1, len(lat)):
        dist_step[i] = great_circle_distance(lat[i-1], lon[i-1], lat[i], lon[i])
    dist_km = np.cumsum(dist_step) / 1000.0

    # decimate for plotting
    # keep alignment by decimating a stacked array
    stacked = np.column_stack([t_hr, lat, lon, alt, dist_km])
    stacked_d = decimate_to_target(stacked, target_points)

    t_hr_d  = stacked_d[:, 0]
    lat_d   = stacked_d[:, 1]
    lon_d   = stacked_d[:, 2]
    alt_d   = stacked_d[:, 3]
    dist_km_d = stacked_d[:, 4]

    return t_hr_d, lat_d, lon_d, alt_d, dist_km_d


# -----------------------------
# Run simulation
# -----------------------------
params = input_parameters()

tic = time.time()
results, coords = Simulation(params)
toc = time.time()
print("TTR [s]:", toc - tic)

coords = np.asarray(coords)
if coords.ndim != 2 or coords.shape[1] < 3:
    raise ValueError(f"Expected coords as Nx3 [lat, lon, alt]. Got shape {coords.shape}")

lat = coords[:, 0].astype(float)
lon = coords[:, 1].astype(float)
alt = coords[:, 2].astype(float)

# sim time axis
dt = float(params["time_step"])
t_s = np.arange(len(lat)) * dt
t_hr = t_s / 3600.0

# sim distance axis
dist_step = np.zeros(len(lat))
for i in range(1, len(lat)):
    dist_step[i] = great_circle_distance(lat[i-1], lon[i-1], lat[i], lon[i])
dist_km = np.cumsum(dist_step) / 1000.0

# sim burst & landing
i_burst = int(np.nanargmax(alt))
burst_lat, burst_lon, burst_alt = lat[i_burst], lon[i_burst], alt[i_burst]
i_land = len(lat) - 1
land_lat, land_lon, land_alt = lat[i_land], lon[i_land], alt[i_land]

print("Predicted landing:", land_lat, land_lon, "alt:", land_alt)
print("Predicted burst altitude:", np.nanmax(alt))


# -----------------------------
# Load real flight (decimated)
# -----------------------------
real_path = "Flight_10-18-25_telem.csv"   # <-- set your filename here
t_hr_r, lat_r, lon_r, alt_r, dist_km_r = load_real_flight_csv(real_path, target_points=9100)

# real burst & landing (from recorded alt)
i_burst_r = int(np.nanargmax(alt_r))
burst_lat_r, burst_lon_r, burst_alt_r = lat_r[i_burst_r], lon_r[i_burst_r], alt_r[i_burst_r]
i_land_r = len(lat_r) - 1
land_lat_r, land_lon_r, land_alt_r = lat_r[i_land_r], lon_r[i_land_r], alt_r[i_land_r]

print("real burst altitude;",burst_alt_r)

# landing error vs real landing in this file
distance_err = great_circle_distance(land_lat_r, land_lon_r, land_lat, land_lon)
print("Landing error vs REAL landing [m]:", distance_err)


# -----------------------------
# Plot 1: Altitude vs Time (sim + real)
# -----------------------------
plt.figure()
plt.plot(t_hr, alt, label="Simulated", linewidth=2)
plt.plot(t_hr_r, alt_r, label="Real flight", linewidth=2)


plt.xlabel("Time (hours)")
plt.ylabel("Altitude (m)")
plt.title("Altitude vs Time: Simulated + Real Flight")
plt.grid(True)
plt.legend()


# -----------------------------
# Plot 2: Altitude vs Distance (sim + real)
# -----------------------------
plt.figure()
plt.plot(dist_km, alt, label="Simulated", linewidth=2)
plt.plot(dist_km_r, alt_r, label="Real flight", linewidth=2)


plt.xlabel("Along-track distance (km)")
plt.ylabel("Altitude (m)")
plt.title("Altitude vs Distance: Simulated + Real Flight")
plt.grid(True)
plt.legend()


# -----------------------------
# Plot 3: Ground Track Map (both colored by altitude)
# -----------------------------
tiler = cimgt.OSM()
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(1, 1, 1, projection=tiler.crs)

# include both tracks in bounds
pad = 0.4
lon_all = np.concatenate([lon, lon_r])
lat_all = np.concatenate([lat, lat_r])

extent = [
    float(np.nanmin(lon_all) - pad),
    float(np.nanmax(lon_all) + pad),
    float(np.nanmin(lat_all) - pad),
    float(np.nanmax(lat_all) + pad),
]
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.add_image(tiler, 9)

ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.coastlines(resolution="10m")

# Sim: altitude-colored scatter (one colormap)
sc_sim = ax.scatter(
    lon, lat,
    c=alt,
    s=8,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    alpha=0.7
)

# Real: altitude-colored scatter (different colormap)
sc_real = ax.scatter(
    lon_r, lat_r,
    c=alt_r,
    s=10,
    transform=ccrs.PlateCarree(),
    cmap="plasma",
    alpha=0.7
)

# Colorbars (two scales)
cb1 = plt.colorbar(sc_sim, ax=ax, pad=0.08, fraction=0.046)
cb1.set_label("Altitude (m)")
cb2 = plt.colorbar(sc_real, ax=ax, pad=0.08, fraction=0.046)
#cb2.set_label("Real Altitude (m)")

# Markers: sim
#ax.scatter(params["longitude"], params["latitude"], marker="*", s=180, transform=ccrs.PlateCarree())
#ax.text(params["longitude"], params["latitude"], "Launch (sim)", transform=ccrs.PlateCarree())

ax.scatter(burst_lon, burst_lat, marker="X", s=140, transform=ccrs.PlateCarree())
ax.text(burst_lon, burst_lat-.05, "Burst (sim)", transform=ccrs.PlateCarree())

ax.scatter(land_lon, land_lat, marker="o", s=140, transform=ccrs.PlateCarree())
ax.text(land_lon +.03, land_lat-.03, f"Landing (sim)", transform=ccrs.PlateCarree())

# Markers: real
ax.scatter(lon_r[0], lat_r[0], marker="^", s=140, transform=ccrs.PlateCarree())
ax.text(lon_r[0]+.03, lat_r[0]+.03, "Launch", transform=ccrs.PlateCarree())

ax.scatter(burst_lon_r, burst_lat_r, marker="X", s=140, transform=ccrs.PlateCarree())
ax.text(burst_lon_r+.03, burst_lat_r+.03, "Burst (real)", transform=ccrs.PlateCarree())

ax.scatter(land_lon_r, land_lat_r, marker="s", s=140, transform=ccrs.PlateCarree())
ax.text(land_lon_r+.03, land_lat_r+.03, "Landing (real)", transform=ccrs.PlateCarree())

ax.set_title("Ground Track: Sim + Real")

plt.show()
