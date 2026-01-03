from herbie import Herbie
import xarray as xr

# -------- SETTINGS --------
TIME    = "2025-12-05 12:00"
MODEL   = "gfs"
PRODUCT = "pgrb2.0p25"   # GFS

print("📡 Model:", MODEL)
print("🕒 Time:", TIME)

H = Herbie(TIME, model=MODEL, product=PRODUCT)

# --- Load temperature AND geopotential height on isobaric surfaces ---
# TMP gives you temp; HGT gives you altitude of the pressure surfaces.
ds_list_tmp = H.xarray("TMP")
ds_list_hgt = H.xarray("HGT")

def normalize_to_dataset(ds_or_list):
    if isinstance(ds_or_list, list):
        return xr.merge(ds_or_list, compat="override")
    return ds_or_list

ds_tmp = normalize_to_dataset(ds_list_tmp)
ds_hgt = normalize_to_dataset(ds_list_hgt)

# Merge everything together
ds = xr.merge([ds_tmp, ds_hgt], compat="override")

print("\n========== DATASET STRUCTURE ==========\n")
print(ds)

print("\n========== ALTITUDES (FROM HGT) ==========\n")

if "isobaricInhPa" in ds.coords and "HGT" in ds.data_vars:
    levels = ds["isobaricInhPa"].values

    # Find horizontal dims automatically (lat/lon naming varies sometimes)
    hgt = ds["HGT"]
    horiz_dims = [d for d in hgt.dims if d not in ("isobaricInhPa", "time", "step", "valid_time")]

    # Reduce to a single representative height per level (domain mean)
    if horiz_dims:
        hgt_profile = hgt.mean(dim=horiz_dims, skipna=True)
    else:
        hgt_profile = hgt

    # If there’s still a time/step axis, squeeze it out
    hgt_profile = hgt_profile.squeeze()

    print("✅ Altitude of pressure surfaces (domain-mean), meters:")
    for p, z in zip(levels, hgt_profile.values):
        # HGT in GFS is usually geopotential meters (gpm) ~ meters
        print(f"{p:7.1f} hPa  ->  {float(z):8.1f} m")

else:
    print("❌ Couldn't find both 'isobaricInhPa' coord and 'HGT' variable.")
    print("Coordinates:", list(ds.coords))
    print("Variables:", list(ds.data_vars))

    # --- Fallback: Standard Atmosphere approximation (rough) ---
    print("\n========== ALTITUDES (STANDARD ATMOSPHERE APPROX) ==========\n")
    if "isobaricInhPa" in ds.coords:
        levels = ds["isobaricInhPa"].values
        p0 = 1013.25  # hPa
        # z ≈ 44330*(1-(p/p0)^(0.1903))  (tropospheric approx)
        z_est = 44330.0 * (1.0 - (levels / p0) ** 0.1903)

        print("⚠️ Approx altitude from pressure only (rough), meters:")
        for p, z in zip(levels, z_est):
            print(f"{p:7.1f} hPa  ->  {float(z):8.1f} m")
    else:
        print("❌ No 'isobaricInhPa' dimension found.")
