Process Flow

1. Enter Initial Parameters:
Location - Latitude, Longitude (Decimal Degrees)
Altitude - Meters
Ballonn Mass - kg
Balloon Burst Diameter - m
Helium Fill Volume - m3

2. Computer Initial Parameters from Surface Conditions:
Air Pressure - milibar
Air Teperature - K
Air Dynamic Viscocity - m2/s2
Air Density - kg/m3
Helium Density - kg/m3
Helium Mass - kg
Balloon Diameter - m
Balloon CSA - m2

Total Flight Mass - kg
Buoyancy Force - N
Weight Force - N
Net Lift - N
Acceleration Buoyancy - m/s2

3. Lookup Atmospheric Data (Interpolate between Data Points [Latitude Longitude Altitude])
Temperature - K
Omega Vertical Pressure Gradient - Pa/s
u Wind Velocity - m/s
v Wind Velocity - m/s
w Wind Velocity - m/s
|V| Wind Velocity - m/s
Ground Distance - m

--- Compute Force Variables ---
Reynolds Number(D,|V|,mu)
Cd(Re)
F_Drag(Cd,rho_air,V_rel,A,)

--- Compute Acceleration ---
Total Force - F_Drag[Dx,Dy,Dz] + Net Lift[0,0,Lz]
Acceleration - Total Force / Total Mass

4. Integrate (RK4)
-- First Convert to System of Std Diff Equations --
X_dot = V
V_dot = A

Use a Timestep to Integrate

5. Recompute Properties
Air Pressure - milibar
Helium Density - kg/m3
Helium Mass - kg
Balloon Diameter - m
Balloon CSA - m2

Buoyancy Force - N
Weight Force - N
Net Lift - N
Acceleration Buoyancy - m/s2

6. Repeat from Step 3 until Conditions are met


Func Lookup File
Sutherland's Dynamic Viscocity Law

CD for sphere Morrison-Style Drag
Faith A. Morrison — Data Correlation for Drag Coefficient for Sphere (2013, Michigan Technological University)
https://galileoandeinstein.phys.virginia.edu/more_stuff/Applets/Projectile/DataCorrelationForSphereDrag2013.pdf?utm_source=chatgpt.com


CD Estimate for Parachute, RocketMan Parchutes do not give estimate for the Cd
https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/rktvrecv.html