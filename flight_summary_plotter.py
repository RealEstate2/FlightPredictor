import pandas as pd
from Utilities.GreatCircle import great_circle_distance

csv_path = "flight_batch_summary.csv"

df = pd.read_csv(csv_path)
print("Loaded:", csv_path)


len = len(df)
print("Rows:", len)
print("Columns:", list(df.columns))


land_lat = pd.to_numeric(df["land_lat"])
land_lon = pd.to_numeric(df["land_lon"]) 
burst_lon = pd.to_numeric(df["burst_lat"])
burst_lat = pd.to_numeric(df["burst_lon"])

lift = pd.to_numeric(df['p_neck_lift'])
mass = pd.to_numeric(df['p_payload_mass'])

target_lat =  39.417
 
target_lon =  -101.9234

radius = 10 #km

count = 0

for n in range(len):
    dist_n = great_circle_distance(lat1=land_lat[n],lon1=land_lon[n],lat2=target_lat,lon2=target_lon)/1000
    
    if dist_n < radius:
        count += 1

        print("run",n)
        print("distance from ideal:", dist_n)
        print("lat:",land_lat[n],"  long:",land_lon[n])
        print("lift:", lift[n])
        print("mass:",mass[n])
        print("differnce:",lift[n]-mass[n])

        
print(count)




