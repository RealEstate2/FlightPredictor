from herbie import Herbie
import xarray as xr

# -------- SETTINGS --------
TIME    = "2025-12-05 12:00"
MODEL   = "gfs"
PRODUCT = "pgrb2.0p25"   # GFS

print("📡 Model:", MODEL)
print("🕒 Time:", TIME)

H = Herbie(TIME, model=MODEL, product=PRODUCT)

# Load variable list
ds_list = H.xarray("TMP")

# Normalize list → Dataset
if isinstance(ds_list, list):
    ds = xr.merge(ds_list, compat="override")
else:
    ds = ds_list

print("\n========== DATASET STRUCTURE ==========\n")
print(ds)

# ---- Extract pressure levels ----
print("\n========== PRESSURE LEVELS ==========\n")

if "isobaricInhPa" in ds.coords:
    levels = ds["isobaricInhPa"].values
    print("✅ Available pressure levels [hPa]:")
    print(levels)
else:
    print("❌ No 'isobaricInhPa' dimension found.")
    print("Coordinates:", list(ds.coords))
