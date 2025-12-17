# AtmoSlicer.py  (FASTER version)
import numpy as np
import os
from datetime import datetime, timedelta
from herbie import Herbie
import xarray as xr
from atmosphere import standardAtmosphere

# ===============================
# Config
# ===============================
DATA_DIR = "hrrr_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ===============================
# In-memory caches (hot path)
# ===============================
_DATASET_CACHE = {}    # (time_str, level_hPa) -> xr.Dataset
_LAST_IJ       = {}    # id(ds) -> (iy, ix) last nearest index
_LATLON_CACHE  = {}    # id(ds) -> (lat2d, lon2d)
_ARRAY_CACHE   = {}    # (id(ds), var_name) -> np.ndarray

# ===============================
# Helpers
# ===============================
def _nearest_pressures(P_hPa):
    levels = np.array([
        1000, 975, 950, 925, 900, 875, 850, 825, 800,
        775, 750, 725, 700, 675, 650, 625, 600,
        575, 550, 525, 500, 475, 450, 425, 400,
        375, 350, 325, 300, 275, 250, 225, 200,
        175, 150, 125, 100
    ])
    i = np.argsort(np.abs(levels - P_hPa))
    return levels[i[0]], levels[i[1]]

def _time_from_base(base_time_str, dt_sec):
    t0 = datetime.fromisoformat(base_time_str)
    t = t0 + timedelta(seconds=float(dt_sec))

    # round to nearest hour
    if t.minute >= 30:
        t = t.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        t = t.replace(minute=0, second=0, microsecond=0)

    return t.strftime("%Y-%m-%d %H:%M")

# ===============================
# Fast slice fetcher (disk + RAM cache)
# ===============================
def _fetch_slice(time_str, level_hPa):
    key = (time_str, level_hPa)
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]

    fname = f"{DATA_DIR}/hrrr_{time_str.replace(':','').replace(' ','_')}_{level_hPa}.nc"

    if os.path.exists(fname):
        ds = xr.open_dataset(fname)
        _DATASET_CACHE[key] = ds
        return ds

    print(f"📡 Fetching HRRR {level_hPa} mb @ {time_str}")
    H = Herbie(time_str, model="hrrr", product="prs")
    lvl = f"{level_hPa} mb"

    # Pull variables individually and merge
    ds = xr.merge([
        H.xarray(f"UGRD:{lvl}"),
        H.xarray(f"VGRD:{lvl}"),
        H.xarray(f"VVEL:{lvl}"),
        H.xarray(f"TMP:{lvl}"),
        H.xarray(f"HGT:{lvl}")
    ], compat="override")

    # Normalize variable names
    rename = {}
    for v in ds.data_vars:
        name = v.lower()
        if "ugrd" in name: rename[v] = "u"
        elif "vgrd" in name and "vvel" not in name: rename[v] = "v"
        elif "vvel" in name: rename[v] = "w"
        elif "tmp"  in name: rename[v] = "t"
        elif "hgt"  in name: rename[v] = "gh"
    ds = ds.rename(rename)

    ds.to_netcdf(fname)
    _DATASET_CACHE[key] = ds
    return ds

# ===============================
# Nearest grid index (local search + cache)
# ===============================
def _nearest_ij(ds, lat, lon, search_radius=15):
    """
    Find nearest grid point to (lat, lon).

    Uses a fast local search around the last used index for this dataset.
    Falls back to full-domain search on first use.
    """
    ds_id = id(ds)

    # Cache lat/lon fields
    if ds_id in _LATLON_CACHE:
        lat2d, lon2d = _LATLON_CACHE[ds_id]
    else:
        lat2d = ds["latitude"].values
        lon2d = ds["longitude"].values
        _LATLON_CACHE[ds_id] = (lat2d, lon2d)

    ny, nx = lat2d.shape

    # First time for this dataset: full search
    if ds_id not in _LAST_IJ:
        dist2 = (lat2d - lat)**2 + (lon2d - lon)**2
        iy, ix = np.unravel_index(np.argmin(dist2), lat2d.shape)
        _LAST_IJ[ds_id] = (iy, ix)
        return iy, ix

    # Local window search around last index
    iy0, ix0 = _LAST_IJ[ds_id]
    r = search_radius

    i_min = max(0, iy0 - r)
    i_max = min(ny, iy0 + r + 1)
    j_min = max(0, ix0 - r)
    j_max = min(nx, ix0 + r + 1)

    sub_lat = lat2d[i_min:i_max, j_min:j_max]
    sub_lon = lon2d[i_min:i_max, j_min:j_max]

    dist2 = (sub_lat - lat)**2 + (sub_lon - lon)**2
    sub_iy, sub_ix = np.unravel_index(np.argmin(dist2), dist2.shape)

    iy = i_min + sub_iy
    ix = j_min + sub_ix

    _LAST_IJ[ds_id] = (iy, ix)
    return iy, ix

# ===============================
# Cached array getter
# ===============================
def _get_array(ds, var_name):
    key = (id(ds), var_name)
    arr = _ARRAY_CACHE.get(key)
    if arr is None:
        arr = ds[var_name].values
        _ARRAY_CACHE[key] = arr
    return arr

# ===============================
# Interp
# ===============================
def _lerp(v1, p1, v2, p2, p):
    return v1 + (v2 - v1) * (p - p1) / (p2 - p1)

# ===============================
# Public API
# ===============================
def Slice_HRRR(lat, lon, alt_m, dt_sec, launch_time_str):
    """
    Returns: u, v, w, z, T at (lat, lon, alt, time)
    """

    # ---- Time ----
    valid_time = _time_from_base(launch_time_str, dt_sec)

    # ---- Altitude → pressure ----
    P_kPa, _, _, _ = standardAtmosphere.qualities(alt_m)
    P_target = P_kPa * 10.0  # kPa → hPa

    # ---- Two nearest pressure levels ----
    P1, P2 = _nearest_pressures(P_target)

    # ---- Fetch (RAM cached) ----
    ds1 = _fetch_slice(valid_time, P1)
    ds2 = _fetch_slice(valid_time, P2)

    # ---- Lon convention ----
    if lon < 0:
        lon = lon + 360.0

    # ---- Nearest grid cell (local search) ----
    iy, ix = _nearest_ij(ds1, lat, lon)

    # ---- FAST scalar access (cached NumPy arrays) ----
    a1 = {
        "u":  _get_array(ds1, "u"),
        "v":  _get_array(ds1, "v"),
        "w":  _get_array(ds1, "w"),
        "gh": _get_array(ds1, "gh"),
        "t":  _get_array(ds1, "t"),
    }
    a2 = {
        "u":  _get_array(ds2, "u"),
        "v":  _get_array(ds2, "v"),
        "w":  _get_array(ds2, "w"),
        "gh": _get_array(ds2, "gh"),
        "t":  _get_array(ds2, "t"),
    }

    out = {}
    for name in ("u", "v", "w", "gh", "t"):
        out[name] = _lerp(
            a1[name][iy, ix], P1,
            a2[name][iy, ix], P2,
            P_target
        )

    return out["u"], out["v"], out["w"], out["gh"], out["t"]


# ===============================
# Optional quick test
# ===============================
"""
LAT  = 40.0547
LON  = -105.2571
ALT  = 1600
TIME = "2025-11-29 12:00"

u,v,w,z,T = Slice(LAT, LON, ALT, 0, TIME)
print("U, V, W(Pa/s), Z, T:", u, v, w, z, T)
"""
