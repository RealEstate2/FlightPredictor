from herbie import Herbie
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# TIME
# ---------------------------------------------------
H = Herbie("2025-11-29 00:00", model="hrrr",product = "")

# ---------------------------------------------------
# LOCATION (convert lon from [-180,180] → [0,360])
# ---------------------------------------------------
LAT = 40.554707
LON = -105.157157

if LON < 0:
    LON = LON + 360

R = .1

LAT_MIN = LAT - R
LAT_MAX = LAT + R
LON_MIN = LON - R
LON_MAX = LON + R

# ---------------------------------------------------
# VARIABLES
# ---------------------------------------------------
VARIABLES = [
    "TMP:2 m",
    "UGRD:10 m",
    "VGRD:10 m",
    "PRES:surface"
]

# ---------------------------------------------------
# LOAD VARIABLES
# ---------------------------------------------------
datasets = []
for var in VARIABLES:
    print(f"Pulling {var} ...")
    ds = H.xarray(var)
    datasets.append(ds)

# ---------------------------------------------------
# MERGE SAFELY
# ---------------------------------------------------
dataset = xr.merge(datasets, compat="override")


# ---------------------------------------------------
# SPATIAL FILTER
# ---------------------------------------------------
dataset = dataset.where(
    (dataset.latitude >= LAT_MIN) &
    (dataset.latitude <= LAT_MAX) &
    (dataset.longitude >= LON_MIN) &
    (dataset.longitude <= LON_MAX),
    drop=True
)

# ---------------------------------------------------
# VALIDATE DATA (before plotting)
# ---------------------------------------------------
if dataset.dims["x"] == 0 or dataset.dims["y"] == 0:
    raise RuntimeError("Dataset EMPTY after filtering. Check lat/lon bounds.")

# ---------------------------------------------------
# DERIVED FIELDS
# ---------------------------------------------------
dataset["T_C"] = dataset["t2m"] - 273.15
dataset["wind_speed"] = np.sqrt(dataset["u10"]**2 + dataset["v10"]**2)

# ---------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# ────────────────
# Temperature
# ────────────────
temp = axs[0].pcolormesh(
    dataset.longitude,
    dataset.latitude,
    dataset.T_C,
    shading="auto"
)
axs[0].set_title("Temperature (°C)")
plt.colorbar(temp, ax=axs[0])

# ────────────────
# Pressure
# ────────────────

pres = axs[1].pcolormesh(
    dataset.longitude,
    dataset.latitude,
    dataset.sp,
    shading="auto"
)

axs[1].set_title("Surface Pressure (Pa)")
plt.colorbar(pres, ax=axs[1])

# ────────────────
# Wind Vectors
# ────────────────
skip = 5
axs[2].quiver(
    dataset.longitude[::skip, ::skip],
    dataset.latitude[::skip, ::skip],
    dataset.u10[::skip, ::skip],
    dataset.v10[::skip, ::skip],
    scale=400
)
axs[2].set_title("Wind Field (10 m)")

# ────────────────
# Final polish
# ────────────────
for ax in axs:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")

plt.show()
