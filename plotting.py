import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

# -----------------------------
# Helpers
# -----------------------------
def _has(df, col):
    return col in df.columns

def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce")

def _finite_series(s):
    s = pd.to_numeric(s, errors="coerce")
    return s[np.isfinite(s)]

def prominent_indices(df, top_n=5):
    """
    Choose 'prominent' sims based on what columns exist:
      - farthest drift
      - longest landing time
      - earliest burst time
    """
    idx = set()

    if _has(df, "drift_m"):
        drift = _num(df, "drift_m")
        finite = drift[np.isfinite(drift)]
        if len(finite):
            idx |= set(drift.sort_values().tail(top_n).index)

    if _has(df, "landing_time_s"):
        tland = _num(df, "landing_time_s")
        finite = tland[np.isfinite(tland)]
        if len(finite):
            idx |= set(tland.sort_values().tail(top_n).index)

    if _has(df, "burst_time_s"):
        tburst = _num(df, "burst_time_s")
        finite = tburst[np.isfinite(tburst)]
        if len(finite):
            idx |= set(tburst.sort_values().head(top_n).index)

    return sorted(idx)

# -----------------------------
# Map plotting
# -----------------------------
def plot_map_from_csv(df, zoom=10, top_n=5, pad_deg=0.5):
    # Required columns
    needed = ["burst_lat", "burst_lon", "land_lat", "land_lon"]
    if not all(_has(df, c) for c in needed):
        print("Map skipped: missing one of", needed)
        return

    # Launch (prefer per-row launch columns if present)
    launch_lat = None
    launch_lon = None
    if _has(df, "p_latitude") and _has(df, "p_longitude"):
        launch_lat = float(_num(df, "p_latitude").iloc[0])
        launch_lon = float(_num(df, "p_longitude").iloc[0])
    elif _has(df, "launch_lat") and _has(df, "launch_lon"):
        launch_lat = float(_num(df, "launch_lat").iloc[0])
        launch_lon = float(_num(df, "launch_lon").iloc[0])

    b_lat = _num(df, "burst_lat")
    b_lon = _num(df, "burst_lon")
    l_lat = _num(df, "land_lat")
    l_lon = _num(df, "land_lon")

    # Build bounds from available burst/land points
    lats = pd.concat([b_lat, l_lat], ignore_index=True)
    lons = pd.concat([b_lon, l_lon], ignore_index=True)
    good = np.isfinite(lats) & np.isfinite(lons)

    if not np.any(good):
        print("Map skipped: no finite burst/land points")
        return

    lats_g = lats[good].astype(float)
    lons_g = lons[good].astype(float)

    extent = [
        float(lons_g.min() - pad_deg),
        float(lons_g.max() + pad_deg),
        float(lats_g.min() - pad_deg),
        float(lats_g.max() + pad_deg),
    ]

    # Prominent sims
    prom_idx = prominent_indices(df, top_n=top_n)

    # --- Make map ---
    tiler = cimgt.OSM()
    fig = plt.figure(figsize=(12, 9))
    ax = plt.axes(projection=tiler.crs)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_image(tiler, zoom)

    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.coastlines(resolution="10m")

    # Plot all points
    # Bursts as x, landings as o
    for i in range(len(df)):
        if np.isfinite(b_lat.iloc[i]) and np.isfinite(b_lon.iloc[i]):
            ax.scatter(float(b_lon.iloc[i]), float(b_lat.iloc[i]),
                       marker="x", s=25, transform=ccrs.PlateCarree())
        if np.isfinite(l_lat.iloc[i]) and np.isfinite(l_lon.iloc[i]):
            ax.scatter(float(l_lon.iloc[i]), float(l_lat.iloc[i]),
                       marker="o", s=18, transform=ccrs.PlateCarree())

    # Launch marker (if available)
    if launch_lat is not None and launch_lon is not None and np.isfinite(launch_lat) and np.isfinite(launch_lon):
        ax.scatter(launch_lon, launch_lat, marker="*", s=180,
                   transform=ccrs.PlateCarree())
        ax.text(launch_lon, launch_lat, "Launch", transform=ccrs.PlateCarree())

    # Highlight prominent sims + label
    # Use sim_id column if present, else index
    for i in prom_idx:
        label = str(int(df.loc[i, "sim_id"])) if _has(df, "sim_id") else str(i)

        if np.isfinite(b_lat.loc[i]) and np.isfinite(b_lon.loc[i]):
            ax.scatter(float(b_lon.loc[i]), float(b_lat.loc[i]),
                       marker="X", s=110, transform=ccrs.PlateCarree())
            ax.text(float(b_lon.loc[i]), float(b_lat.loc[i]), label,
                    transform=ccrs.PlateCarree())

        if np.isfinite(l_lat.loc[i]) and np.isfinite(l_lon.loc[i]):
            ax.scatter(float(l_lon.loc[i]), float(l_lat.loc[i]),
                       marker="o", s=110, transform=ccrs.PlateCarree())
            ax.text(float(l_lon.loc[i]), float(l_lat.loc[i]), label,
                    transform=ccrs.PlateCarree())

    ax.set_title("Burst (x) and Landing (o) Map — prominent sims labeled")
    plt.show()

# -----------------------------
# Bulk plotting (hist/scatters)
# -----------------------------
def plot_csv_bundle(csv_path="flight_batch_summary.csv", top_n=5):
    df = pd.read_csv(csv_path)
    print("Loaded:", csv_path)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    drift_m = _num(df, "drift_m") if _has(df, "drift_m") else None
    t_burst_s = _num(df, "burst_time_s") if _has(df, "burst_time_s") else None
    t_land_s  = _num(df, "landing_time_s") if _has(df, "landing_time_s") else None

    drift_km = (drift_m / 1000.0) if drift_m is not None else None
    t_burst_min = (t_burst_s / 60.0) if t_burst_s is not None else None
    t_land_min  = (t_land_s / 60.0) if t_land_s is not None else None

    # 1) Drift distribution
    if drift_km is not None:
        dk = _finite_series(drift_km)
        if len(dk):
            plt.figure()
            plt.hist(dk, bins=20)
            plt.xlabel("Drift distance (km)")
            plt.ylabel("Count")
            plt.title("Landing drift distribution")

            plt.figure()
            plt.plot(np.sort(dk), marker="o", linestyle="None")
            plt.xlabel("Sorted index")
            plt.ylabel("Drift distance (km)")
            plt.title("Sorted drift distances")

    # 2) Times
    if t_burst_min is not None:
        tb = _finite_series(t_burst_min)
        if len(tb):
            plt.figure()
            plt.hist(tb, bins=20)
            plt.xlabel("Burst time (min)")
            plt.ylabel("Count")
            plt.title("Burst time distribution")

    if t_land_min is not None:
        tl = _finite_series(t_land_min)
        if len(tl):
            plt.figure()
            plt.hist(tl, bins=20)
            plt.xlabel("Landing time (min)")
            plt.ylabel("Count")
            plt.title("Landing time distribution")

    # 3) Drift vs time
    if drift_km is not None and t_land_min is not None:
        dk = pd.to_numeric(drift_km, errors="coerce")
        tl = pd.to_numeric(t_land_min, errors="coerce")
        m = np.isfinite(dk) & np.isfinite(tl)
        if np.any(m):
            plt.figure()
            plt.scatter(tl[m], dk[m])
            plt.xlabel("Landing time (min)")
            plt.ylabel("Drift distance (km)")
            plt.title("Drift vs landing time")

    if drift_km is not None and t_burst_min is not None:
        dk = pd.to_numeric(drift_km, errors="coerce")
        tb = pd.to_numeric(t_burst_min, errors="coerce")
        m = np.isfinite(dk) & np.isfinite(tb)
        if np.any(m):
            plt.figure()
            plt.scatter(tb[m], dk[m])
            plt.xlabel("Burst time (min)")
            plt.ylabel("Drift distance (km)")
            plt.title("Drift vs burst time")

    # 4) Quick “top lists”
    if drift_km is not None:
        dk = pd.to_numeric(drift_km, errors="coerce")
        top = dk.sort_values(ascending=False).head(10)
        if len(top):
            print("\nTop 10 drift (km):")
            simcol = "sim_id" if _has(df, "sim_id") else None
            out = pd.DataFrame({
                "sim_id": df.loc[top.index, simcol].values if simcol else top.index.values,
                "drift_km": top.values
            })
            print(out.to_string(index=False))

    if t_land_min is not None:
        tl = pd.to_numeric(t_land_min, errors="coerce")
        top = tl.sort_values(ascending=False).head(10)
        if len(top):
            print("\nTop 10 landing time (min):")
            simcol = "sim_id" if _has(df, "sim_id") else None
            out = pd.DataFrame({
                "sim_id": df.loc[top.index, simcol].values if simcol else top.index.values,
                "t_land_min": top.values
            })
            print(out.to_string(index=False))

    #plt.show()

    # 5) Map (separate window)
    plot_map_from_csv(df, zoom=10, top_n=top_n, pad_deg=0.5)

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    plot_csv_bundle("flight_batch_summary.csv", top_n=20)
