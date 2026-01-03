import numpy as np
import os
from datetime import datetime, timedelta
from herbie import Herbie
import xarray as xr
from atmosphere import standardAtmosphere

# ===============================
# Config
# ===============================
DATA_DIR = "gfs_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ===============================
# In-memory caches (hot path)
# ===============================
_DATASET_CACHE = {}
_LAST_IJ       = {}
_LATLON_CACHE  = {}
_ARRAY_CACHE   = {}

# ===============================
# Helpers
# ===============================
def _nearest_pressures(P_hPa):
    levels = np.array([
        1000, 975, 950, 925, 900, 850, 800,
        750, 700, 650, 600, 550, 500, 450, 400,
        350, 300, 250, 200,150, 100, 70, 50, 40,  
        30, 20, 15, 10, 7, 5, 3, 2, 1
        
    ])
    i = np.argsort(np.abs(levels - P_hPa))
    return levels[i[0]], levels[i[1]]

def _nearest_time(hhmm):
    times = np.array([
        0000, 400,800, 1200, 1800
        
    ])
    i = np.argsort(np.abs(times - hhmm))
    return times[i[0]], times[i[1]]


def _time_from_base(base_time_str, dt_sec):
    t0 = datetime.fromisoformat(base_time_str)
    t = t0 + timedelta(seconds=float(dt_sec))
    if t.minute >= 30:
        t = t.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        t = t.replace(minute=0, second=0, microsecond=0)
    return t.strftime("%Y-%m-%d %H:%M")

# ===============================
# Fetcher (GFS safe)
# ===============================
def _fetch_slice(time_str, level_hPa):

    key = (time_str, level_hPa)
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]

    fname = f"{DATA_DIR}/gfs_{time_str.replace(':','').replace(' ','_')}_{level_hPa}.nc"
    if os.path.exists(fname):
        ds = xr.open_dataset(fname)
        _DATASET_CACHE[key] = ds
        return ds
    

    print(f"🌍 Fetching GFS {level_hPa} mb @ {time_str}")

    H = Herbie(
    time_str,
    model="gfs",
    product="pgrb2.0p25",
    priority=["aws", "google", "nomads"],
    validate_timeout=5,
    )

    lvl = f"{level_hPa} mb"

    ds_list = [
        H.xarray(f"UGRD:{lvl}"),
        H.xarray(f"VGRD:{lvl}"),
        H.xarray(f"VVEL:{lvl}"),
        H.xarray(f"TMP:{lvl}"),
        H.xarray(f"HGT:{lvl}")
    ]

    # Normalize list of datasets
    flat = []
    for item in ds_list:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)

    # Merge safely
    ds = xr.merge(flat, compat="override")

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

    # Normalize coordinate names if needed
    if "lat" in ds.coords:
        ds = ds.rename({"lat":"latitude"})
    if "lon" in ds.coords:
        ds = ds.rename({"lon":"longitude"})

    ds.to_netcdf(fname)
    _DATASET_CACHE[key] = ds
    return ds


def _nearest_ij(ds, lat, lon, search_radius=15):
    """
    Find nearest index for BOTH:
    - Curvilinear grids (HRRR style lat[y,x])
    - Rectilinear grids (GFS style lat[y], lon[x])
    """

    ds_id = id(ds)

    # ---- Load or cache lat/lon ----
    if ds_id in _LATLON_CACHE:
        lat_arr, lon_arr = _LATLON_CACHE[ds_id]
    else:
        lat_arr = ds["latitude"].values
        lon_arr = ds["longitude"].values
        _LATLON_CACHE[ds_id] = (lat_arr, lon_arr)

    # ======================================
    # CASE 1: GFS STYLE (1D coordinate axes)
    # ======================================
    if lat_arr.ndim == 1 and lon_arr.ndim == 1:
        iy = np.argmin(np.abs(lat_arr - lat))
        ix = np.argmin(np.abs(lon_arr - lon))
        return iy, ix
    
    '''
    # ======================================
    # CASE 2: HRRR STYLE (2D curvilinear grid)
    # ======================================
    lat2d, lon2d = lat_arr, lon_arr
    ny, nx = lat2d.shape
    
    if ds_id not in _LAST_IJ:
        dist2 = (lat2d - lat)**2 + (lon2d - lon)**2
        iy, ix = np.unravel_index(np.argmin(dist2), lat2d.shape)
        _LAST_IJ[ds_id] = (iy, ix)
        return iy, ix

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
    '''
    return iy, ix


def _get_array(ds, var_name):
    key = (id(ds), var_name)
    if key not in _ARRAY_CACHE:
        _ARRAY_CACHE[key] = ds[var_name].values
    return _ARRAY_CACHE[key]

def _lerp(v1, p1, v2, p2, p):
    return v1 + (v2 - v1) * (p - p1) / (p2 - p1)

def Slice_GFS(lat, lon, alt_m, dt_sec, launch_time_str):

    valid_time = _time_from_base(launch_time_str, dt_sec)
    P_kPa, _, _, _ = standardAtmosphere.qualities(alt_m)
    P_target = P_kPa * 10.0

    P1, P2 = _nearest_pressures(P_target)

    ds1 = _fetch_slice(valid_time, P1)
    ds2 = _fetch_slice(valid_time, P2)

    if lon < 0:
        lon += 360.0

    iy, ix = _nearest_ij(ds1, lat, lon)

    a1 = {k:_get_array(ds1,k) for k in ("u","v","w","gh","t")}
    a2 = {k:_get_array(ds2,k) for k in ("u","v","w","gh","t")}

    out = {}
    for name in a1:
        out[name] = _lerp(
            a1[name][iy,ix], P1,
            a2[name][iy,ix], P2,
            P_target
        )

    return out["u"], out["v"], out["w"], out["gh"], out["t"]
