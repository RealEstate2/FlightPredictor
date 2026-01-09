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
# Cache tuning (IMPORTANT)
# ===============================
# Coarser bins -> higher cache hit rate + less memory
T_BIN_S   = 30.0   # cache resolution in seconds
ALT_BIN_M = 50.0   # cache resolution in meters
LL_BIN_DP = 4      # lat/lon rounding decimals

# Cap sample cache size to avoid runaway RAM
MAX_SAMPLE_CACHE = 200_000

# ===============================
# In-memory caches (hot path)
# ===============================
_DATASET_CACHE   = {}   # (time_str, level_hPa) or (time_str, "surface") -> xarray dataset
_LATLON_CACHE    = {}   # id(ds) -> (lat_arr, lon_arr)
_ARRAY_CACHE     = {}   # (id(ds), var_name) -> numpy array

# New caches
_IJ_CACHE        = {}   # (id(ds), lat_bin, lon_bin) -> (iy, ix)
_PRESSURE_CACHE  = {}   # alt_bin -> (P1, P2, wp, P_target)
_SAMPLE_CACHE    = {}   # (t_lo, t_hi, t_bin, lat_bin, lon_bin, alt_bin) -> (u,v,w,gh,t,agl)

# ===============================
# Helpers
# ===============================
def _nearest_pressures(P_hPa):
    levels = np.array([
        1000, 975, 950, 925, 900, 850, 800,
        750, 700, 650, 600, 550, 500, 450, 400,
        350, 300, 250, 200, 150, 100, 70, 50, 40,
        30, 20, 15, 10, 7, 5, 3, 2, 1
    ], dtype=float)
    i = np.argsort(np.abs(levels - P_hPa))
    return float(levels[i[0]]), float(levels[i[1]])

_TIMES = np.array([0, 600, 1200, 1800], dtype=int)

def _times(base_time_str, dt_sec):
    t0 = datetime.fromisoformat(base_time_str)
    t  = t0 + timedelta(seconds=float(dt_sec))

    hhmm = t.hour * 100 + t.minute
    idx = int(np.searchsorted(_TIMES, hhmm, side="left"))

    if idx > 0:
        lo_code = int(_TIMES[idx - 1])
        lo_day  = t.date()
    else:
        lo_code = int(_TIMES[-1])
        lo_day  = (t - timedelta(days=1)).date()

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

    return (
        t.strftime("%Y-%m-%d %H:%M"),
        build_timestr(lo_day, lo_code),
        build_timestr(hi_day, hi_code),
    )

def _bin_key(lat, lon, alt_m, dt_sec):
    """Quantize inputs so repeated calls hit cache."""
    lat_b = round(float(lat), LL_BIN_DP)
    lon_b = round(float(lon), LL_BIN_DP)
    alt_b = int(float(alt_m) // ALT_BIN_M)
    t_b   = int(float(dt_sec) // T_BIN_S)
    return lat_b, lon_b, alt_b, t_b

def _get_array(ds, var_name):
    key = (id(ds), var_name)
    if key not in _ARRAY_CACHE:
        _ARRAY_CACHE[key] = ds[var_name].values
    return _ARRAY_CACHE[key]

def _cache_guard():
    # cheap RAM safety
    if len(_SAMPLE_CACHE) > MAX_SAMPLE_CACHE:
        _SAMPLE_CACHE.clear()

# ===============================
# Fetchers
# ===============================
def _fetch_slice(time_str, level_hPa):
    key = (time_str, float(level_hPa))
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
        save_dir=DATA_DIR,
    )
    H.download(overwrite=False)
    lvl = f"{level_hPa} mb"
    ds_list = [
        H.xarray(f"UGRD:{lvl}"),
        H.xarray(f"VGRD:{lvl}"),
        H.xarray(f"VVEL:{lvl}"),
        H.xarray(f"TMP:{lvl}"),
        H.xarray(f"HGT:{lvl}")
    ]

    flat = []
    for item in ds_list:
        flat.extend(item) if isinstance(item, list) else flat.append(item)

    ds = xr.merge(flat, compat="override")

    rename = {}
    for v in ds.data_vars:
        name = v.lower()
        if "ugrd" in name: rename[v] = "u"
        elif "vgrd" in name and "vvel" not in name: rename[v] = "v"
        elif "vvel" in name: rename[v] = "w"
        elif "tmp"  in name: rename[v] = "t"
        elif "hgt"  in name: rename[v] = "gh"
    ds = ds.rename(rename)

    if "lat" in ds.coords:
        ds = ds.rename({"lat":"latitude"})
    if "lon" in ds.coords:
        ds = ds.rename({"lon":"longitude"})

    ds.to_netcdf(fname)
    _DATASET_CACHE[key] = ds
    return ds

def _fetch_surface(time_str):
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
        save_dir=DATA_DIR,
    )

    ds_list = H.xarray("HGT:surface")

    flat = []
    flat.extend(ds_list) if isinstance(ds_list, list) else flat.append(ds_list)

    ds = xr.merge(flat, compat="override")

    if "lat" in ds.coords:
        ds = ds.rename({"lat":"latitude"})
    if "lon" in ds.coords:
        ds = ds.rename({"lon":"longitude"})

    rename = {}
    for v in ds.data_vars:
        if "hgt" in v.lower():
            rename[v] = "orog"
    ds = ds.rename(rename)

    ds.to_netcdf(fname)
    _DATASET_CACHE[key] = ds
    return ds

# ===============================
# Indexing
# ===============================
def _nearest_ij(ds, lat, lon):
    ds_id = id(ds)

    if ds_id in _LATLON_CACHE:
        lat_arr, lon_arr = _LATLON_CACHE[ds_id]
    else:
        lat_arr = ds["latitude"].values
        lon_arr = ds["longitude"].values
        _LATLON_CACHE[ds_id] = (lat_arr, lon_arr)

    # GFS: 1D lat[y], lon[x]
    if lat_arr.ndim == 1 and lon_arr.ndim == 1:
        iy = int(np.argmin(np.abs(lat_arr - lat)))
        ix = int(np.argmin(np.abs(lon_arr - lon)))
        return iy, ix

    # If you ever enable 2D lat/lon for HRRR-like grids, you can add it here.
    raise RuntimeError("Unexpected GFS grid shape: latitude/longitude are not 1D.")

def _nearest_ij_cached(ds, lat_b, lon_b):
    key = (id(ds), lat_b, lon_b)
    if key in _IJ_CACHE:
        return _IJ_CACHE[key]
    iy, ix = _nearest_ij(ds, lat_b, lon_b)
    _IJ_CACHE[key] = (iy, ix)
    return iy, ix

# ===============================
# Pressure selection caching
# ===============================
def _pressure_for_alt(alt_m):
    alt_b = int(float(alt_m) // ALT_BIN_M)
    if alt_b in _PRESSURE_CACHE:
        return _PRESSURE_CACHE[alt_b]

    # compute target pressure once per altitude bin
    P_kPa, _, _, _ = standardAtmosphere.qualities(alt_b * ALT_BIN_M)
    P_target = float(P_kPa) * 10.0  # kPa -> hPa

    P1, P2 = _nearest_pressures(P_target)
    wp = 0.0 if P2 == P1 else (P_target - P1) / (P2 - P1)

    _PRESSURE_CACHE[alt_b] = (P1, P2, wp, P_target)
    return _PRESSURE_CACHE[alt_b]

# ===============================
# Main public function
# ===============================
def Slice_GFS(lat, lon, alt_m, dt_sec, launch_time_str):
    """
    Returns: u, v, w, gh, t, agl (AGL is gh - terrain)
    CACHED at binned (time, lat, lon, alt) resolution.
    """

    # normalize lon to [0, 360)
    lon = float(lon)
    if lon < 0:
        lon += 360.0

    # bin keys for caching
    lat_b, lon_b, alt_b, t_b = _bin_key(lat, lon, alt_m, dt_sec)

    # bracket times based on *actual* dt, not binned dt (brackets matter)
    t_str, t_lo, t_hi = _times(launch_time_str, dt_sec)

    # sample cache key includes brackets + bins
    sample_key = (t_lo, t_hi, t_b, lat_b, lon_b, alt_b)
    if sample_key in _SAMPLE_CACHE:
        return _SAMPLE_CACHE[sample_key]

    _cache_guard()

    # time interpolation weight (still using real dt)
    t     = datetime.fromisoformat(t_str)
    lo_dt = datetime.fromisoformat(t_lo)
    hi_dt = datetime.fromisoformat(t_hi)
    denom = (hi_dt - lo_dt).total_seconds()
    wt = 0.0 if denom == 0 else (t - lo_dt).total_seconds() / denom

    # pressure selection from cached altitude bin
    P1, P2, wp, _ = _pressure_for_alt(alt_m)

    # fetch corner datasets
    ds1_t1 = _fetch_slice(t_lo, P1)
    ds2_t1 = _fetch_slice(t_lo, P2)
    ds1_t2 = _fetch_slice(t_hi, P1)
    ds2_t2 = _fetch_slice(t_hi, P2)

    dsS_lo = _fetch_surface(t_lo)
    dsS_hi = _fetch_surface(t_hi)

    # nearest indices (cached)
    iy, ix = _nearest_ij_cached(ds1_t1, lat_b, lon_b)

    # interpolate u,v,w,gh,t across pressure and time
    out = {}
    for name in ("u", "v", "w", "gh", "t"):
        v11 = _get_array(ds1_t1, name)[iy, ix]
        v21 = _get_array(ds2_t1, name)[iy, ix]
        v12 = _get_array(ds1_t2, name)[iy, ix]
        v22 = _get_array(ds2_t2, name)[iy, ix]

        v_tlo = (1.0 - wp) * v11 + wp * v21
        v_thi = (1.0 - wp) * v12 + wp * v22
        out[name] = (1.0 - wt) * v_tlo + wt * v_thi

    # terrain and AGL
    orog_lo = _get_array(dsS_lo, "orog")[iy, ix]
    orog_hi = _get_array(dsS_hi, "orog")[iy, ix]
    orog = (1.0 - wt) * orog_lo + wt * orog_hi

    gh_agl = out["gh"] - orog

    ret = (
        float(out["u"]),
        float(out["v"]),
        float(out["w"]),
        float(out["gh"]),
        float(out["t"]),
        float(gh_agl),
    )

    _SAMPLE_CACHE[sample_key] = ret
    return ret
