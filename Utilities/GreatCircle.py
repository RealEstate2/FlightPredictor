import numpy as np

R_EARTH = 6_371_000.0  # meters

def great_circle_distance(lat1, lon1, lat2, lon2, R=R_EARTH):
    """
    lat/lon in radians
    returns distance in meters
    """
    lat1=np.deg2rad(lat1)
    lon1=np.deg2rad(lon1)
    lat2=np.deg2rad(lat2)
    lon2=np.deg2rad(lon2)



    dlat = (lat2 - lat1)
    dlon = (lon2 - lon1)

    a = (
        np.sin(dlat / 2)**2 +
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


lat_i = 39.41699

long_i = -101.9234

lat_f = 40.01287
long_f = -102.60405

dist= great_circle_distance(lat_i,long_i,lat_f,long_f,R_EARTH)

print("Distance from i to f [km]:",dist/1000)
