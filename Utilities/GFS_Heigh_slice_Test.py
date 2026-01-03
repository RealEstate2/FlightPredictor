from herbie import Herbie
import xarray as xr
import numpy as np
import time

# =========================
# SETTINGS
# =========================
TIME    = "2025-12-05 12:00"
MODEL   = "gfs"
PRODUCT = "pgrb2.0p25"

# Optional targets (set to None to skip)
TARGET_ALT_M   = 10000.0   # altitude -> pressure query (meters)
TARGET_P_HPA   = 500.0     # pressure -> altitude query (hPa)

# Optional location selection (set to None to do domain-mean)
# Example: (40.585, -105.084)  # Fort Collins-ish
TARGET_LATLON = (40.585, -105.084)

print("📡 Model:", MODEL)
print("🕒 Time:", TIME)
print("📦 Product:", PRODUCT)

# =========================
# HELPERS
# =========================
def ensure_list(x):
    return x if isinstance(x, list) else [x]

def pick_isobaric_cube(ds_list):
    """
    Pick best isobaric cube:
      1) Prefer isobaricInhPa
      2) Else use isobaricInPa
    Returns: (dataset, pressure_coord_name, pressure_units_str)
    """
    # Prefer hPa cube
    for dsi in ds_list:
        if "isobaricInhPa" in dsi.coords or "isobaricInhPa" in dsi.dims:
            return dsi, "isobaricInhPa", "hPa"

    # Fall back to Pa cube (convert later)
    for dsi in ds_list:
        if "isobaricInPa" in dsi.coords or "isobaricInPa" in dsi.dims:
            return dsi, "isobaricInPa", "Pa"

    raise ValueError("No isobaric cube found (neither isobaricInhPa nor isobaricInPa).")

def pick_height_var(ds):
    """
    Pick geopotential height variable name.
    Common: 'gh' (cfgrib), 'HGT' (some decodes).
    Otherwise: first data_var.
    """
    for cand in ("gh", "HGT", "hgt", "z", "GeopotentialHeight"):
        if cand in ds.data_vars:
            return cand
    if len(ds.data_vars) == 0:
        raise ValueError("Dataset has no data variables to use as height.")
    return list(ds.data_vars)[0]

def to_hpa(p_vals, units):
    p = np.asarray(p_vals, dtype=float)
    if units.lower() == "hpa":
        return p
    if units.lower() == "pa":
        return p / 100.0
    # unknown, assume hPa
    return p

def squeeze_time_like(da):
    # Remove singletons safely across common time dims
    for d in ["time", "step", "valid_time"]:
        if d in da.dims and da.sizes.get(d, 1) == 1:
            da = da.squeeze(d, drop=True)
    return da

def select_point_or_mean(da, pcoord_name, latlon=None):
    """
    Return a 1D profile vs pressure:
    - If latlon is provided, nearest-neighbor select.
    - Else domain-mean over horizontal dims.
    """
    da = squeeze_time_like(da)

    if latlon is not None:
        lat, lon = latlon
        # try common coord naming
        if "latitude" in da.coords and "longitude" in da.coords:
            return da.sel(latitude=lat, longitude=lon, method="nearest")
        if "lat" in da.coords and "lon" in da.coords:
            return da.sel(lat=lat, lon=lon, method="nearest")
        raise ValueError("Could not find lat/lon coordinate names to select a point.")
    else:
        horiz_dims = [d for d in da.dims if d not in (pcoord_name, "time", "step", "valid_time")]
        if horiz_dims:
            return da.mean(dim=horiz_dims, skipna=True)
        return da

def altitude_at_pressure(p_hpa_target, p_hpa_levels, z_profile):
    """
    z_profile is 1D heights aligned with p_hpa_levels.
    Interpolate z at target pressure, using log(p) interpolation.
    """
    p = np.asarray(p_hpa_levels, dtype=float)
    z = np.asarray(z_profile, dtype=float)

    # Sort by decreasing pressure (common) but interpolation in log(p) expects monotonic x
    idx = np.argsort(p)  # increasing pressure
    p_sorted = p[idx]
    z_sorted = z[idx]

    lp = np.log(p_sorted)
    lp_t = np.log(float(p_hpa_target))

    # out of range -> nan
    if lp_t < lp.min() or lp_t > lp.max():
        return np.nan

    return float(np.interp(lp_t, lp, z_sorted))

def pressure_at_altitude(z_m_target, p_hpa_levels, z_profile):
    """
    Invert z(p)->p(z): interpolate log(p) at target z.
    """
    p = np.asarray(p_hpa_levels, dtype=float)
    z = np.asarray(z_profile, dtype=float)

    # Sort by height increasing
    idx = np.argsort(z)
    z_sorted = z[idx]
    lp_sorted = np.log(p[idx])

    if float(z_m_target) < z_sorted.min() or float(z_m_target) > z_sorted.max():
        return np.nan

    lp_t = np.interp(float(z_m_target), z_sorted, lp_sorted)
    return float(np.exp(lp_t))

# =========================
# LOAD: HGT (geopotential height)
# =========================
H = Herbie(TIME, model=MODEL, product=PRODUCT)

ds_list_hgt = ensure_list(H.xarray("HGT"))
print(f"\nNote: Herbie returned {len(ds_list_hgt)} dataset(s) for HGT (multiple hypercubes is normal).")

iso_ds, pcoord_name, p_units = pick_isobaric_cube(ds_list_hgt)
zvar = pick_height_var(iso_ds)

print("\n✅ Selected isobaric cube:")
print("   pressure coord:", pcoord_name, f"({p_units})")
print("   height var    :", zvar)
print("   dims          :", dict(iso_ds.sizes))
print("   coords        :", list(iso_ds.coords))
print("   vars          :", list(iso_ds.data_vars))

# Extract pressure levels (convert to hPa if needed)
p_levels_raw = iso_ds[pcoord_name].values
p_levels_hpa = to_hpa(p_levels_raw, p_units)

# Extract height field and reduce to a 1D profile (point or mean)
z_field = iso_ds[zvar]
z_profile_da = select_point_or_mean(z_field, pcoord_name, latlon=TARGET_LATLON)
z_profile_da = squeeze_time_like(z_profile_da)

# Ensure the profile is 1D over pressure coordinate
if pcoord_name not in z_profile_da.dims or z_profile_da.ndim != 1:
    raise ValueError(f"Expected a 1D profile over '{pcoord_name}', got dims={z_profile_da.dims}")

z_profile = z_profile_da.values

print("\n========== PRESSURE LEVELS (hPa) ==========")
print(p_levels_hpa)

print("\n========== ALTITUDE OF PRESSURE SURFACES ==========")
label = "nearest-point" if TARGET_LATLON is not None else "domain-mean"
print(f"(using {label} profile)")
for p, z in zip(p_levels_hpa, z_profile):
    print(f"{float(p):7.1f} hPa  ->  {float(z):9.2f} m (geopotential)")

# =========================
# QUERIES
# =========================

if TARGET_ALT_M is not None:
    tic = time.time()
    p_at_z = pressure_at_altitude(TARGET_ALT_M, p_levels_hpa, z_profile)
    toc = time.time()

    print("\n========== QUERY: altitude -> pressure ==========")
    print(f"z = {TARGET_ALT_M} m  ->  p ≈ {p_at_z:.2f} hPa (model-inverted)")
    print("Call Time[s]:", toc-tic)

print("\n✅ Done.")
