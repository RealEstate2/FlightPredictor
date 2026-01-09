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

_TIMES = np.array([0, 600, 1200, 1800], dtype=int)

def _times(base_time_str, dt_sec):
    """
    Returns:
        t_str    : actual time at base + dt_sec, "YYYY-MM-DD HH:MM"
        t_lo_str : bracket time at or below t, same format
        t_hi_str : bracket time at or above t, same format
    """
    t0 = datetime.fromisoformat(base_time_str)
    t  = t0 + timedelta(seconds=float(dt_sec))

    hhmm = t.hour * 100 + t.minute
    idx = int(np.searchsorted(_TIMES, hhmm, side="left"))

    # Lower bracket code/date
    if idx > 0:
        lo_code = int(_TIMES[idx - 1])
        lo_day  = t.date()
    else:
        lo_code = int(_TIMES[-1])
        lo_day  = (t - timedelta(days=1)).date()

    # Upper bracket code/date
    if idx < len(_TIMES):
        hi_code = int(_TIMES[idx])
        hi_day  = t.date()
    else:
        hi_code = int(_TIMES[0])
        hi_day  = (t + timedelta(days=1)).date()

    def build_timestr(day, code):
        hh = code // 100
        mm = code % 100
        return datetime.combine(day, datetime.min.time()).replace(hour=hh, minute=mm) \
                       .strftime("%Y-%m-%d %H:%M")

    t_str    = t.strftime("%Y-%m-%d %H:%M")
    t_lo_str = build_timestr(lo_day, lo_code)
    t_hi_str = build_timestr(hi_day, hi_code)

    return t_str, t_lo_str, t_hi_str

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

def _hhmm_to_dt(date_obj, hhmm_code: int):
    hh = hhmm_code // 100
    mm = hhmm_code % 100
    return datetime.combine(date_obj, datetime.min.time()).replace(hour=hh, minute=mm)

def _fetch_surface(time_str):
    """
    Fetch surface terrain height using HGT:surface (reliable for GFS pgrb2.0p25).
    Saves into DATA_DIR to avoid Herbie default paths.
    """
    key = (time_str, "surface")
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]

    fname = f"{DATA_DIR}/gfs_{time_str.replace(':','').replace(' ','_')}_surface.nc"
    if os.path.exists(fname):
        ds = xr.open_dataset(fname)
        _DATASET_CACHE[key] = ds
        return ds

    print(f"🌍 Fetching GFS surface (terrain) @ {time_str}")

    H = Herbie(
        time_str,
        model="gfs",
        product="pgrb2.0p25",
        priority=["aws", "google", "nomads"],
        validate_timeout=10,
        save_dir=DATA_DIR,   # <-- IMPORTANT: keep files local & predictable
    )

    # Terrain in these files is usually "HGT:surface"
    ds_list = H.xarray("HGT:surface")

    # Normalize list
    flat = []
    if isinstance(ds_list, list):
        flat.extend(ds_list)
    else:
        flat.append(ds_list)

    ds = xr.merge(flat, compat="override")

    # Normalize coordinate names
    if "lat" in ds.coords:
        ds = ds.rename({"lat": "latitude"})
    if "lon" in ds.coords:
        ds = ds.rename({"lon": "longitude"})

    # Normalize variable name to 'orog' (terrain height in meters)
    rename = {}
    for v in ds.data_vars:
        if "hgt" in v.lower():
            rename[v] = "orog"
    ds = ds.rename(rename)

    ds.to_netcdf(fname)
    _DATASET_CACHE[key] = ds
    return ds


def Slice_GFS(lat, lon, alt_m, dt_sec, launch_time_str):

    t_str, t_lo, t_hi = _times(launch_time_str, dt_sec)

    t     = datetime.fromisoformat(t_str)
    lo_dt = datetime.fromisoformat(t_lo)
    hi_dt = datetime.fromisoformat(t_hi)

    denom = (hi_dt - lo_dt).total_seconds()
    wt = 0.0 if denom == 0 else (t - lo_dt).total_seconds() / denom

    # Pressure target from standard atmosphere (your existing approach)
    P_kPa, _, _, _ = standardAtmosphere.qualities(alt_m)
    P_target = P_kPa * 10.0  # kPa -> hPa

    P1, P2 = _nearest_pressures(P_target)

    # Fetch the 4 pressure corners (existing)
    ds1_t1 = _fetch_slice(t_lo, P1)
    ds2_t1 = _fetch_slice(t_lo, P2)
    ds1_t2 = _fetch_slice(t_hi, P1)
    ds2_t2 = _fetch_slice(t_hi, P2)

    # Fetch surface terrain at the two bracket times (NEW)
    dsS_lo = _fetch_surface(t_lo)
    dsS_hi = _fetch_surface(t_hi)

    # Lon normalization
    if lon < 0:
        lon += 360.0

    # Same grid indices (use pressure slice grid)
    iy, ix = _nearest_ij(ds1_t1, lat, lon)

    # Pressure weight
    wp = 0.0 if P2 == P1 else (P_target - P1) / (P2 - P1)

    out = {}
    for name in ("u", "v", "w", "gh", "t"):
        v11 = _get_array(ds1_t1, name)[iy, ix]
        v21 = _get_array(ds2_t1, name)[iy, ix]
        v12 = _get_array(ds1_t2, name)[iy, ix]
        v22 = _get_array(ds2_t2, name)[iy, ix]

        v_tlo = (1.0 - wp) * v11 + wp * v21
        v_thi = (1.0 - wp) * v12 + wp * v22
        out[name] = (1.0 - wt) * v_tlo + wt * v_thi

    # --- NEW: terrain height + AGL geopotential height ---
    orog_lo = _get_array(dsS_lo, "orog")[iy, ix]
    orog_hi = _get_array(dsS_hi, "orog")[iy, ix]
    orog = (1.0 - wt) * orog_lo + wt * orog_hi   # time interp only
    out["orog"] = orog
    out["gh_agl"] = out["gh"] - orog

    return out["u"], out["v"], out["w"], out["gh"], out["t"],  out["gh_agl"]
