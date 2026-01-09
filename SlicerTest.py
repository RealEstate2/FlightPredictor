from GFSSlicer_b import Slice_GFS,_fetch_slice
import time
from line_profiler import LineProfiler

lp = LineProfiler()


LAT = 40.4462
LONG = -104.6379
ALT = 4000
TIME = "2025-12-10 12:00"
hPa = 800


tic = time.time()
_fetch_slice(TIME,hPa)
toc = time.time()
print("TTR[s]:",(toc-tic))



#tic = time.time()
#u,v,w,z,T = Slice_GFS(LAT,LONG,ALT,0,TIME)
#toc = time.time()
#print("TTR[s]:",(toc-tic))




#print("U-Vel[m/s]:)",u)
#print("V-Vel[m/s]:)",v)
#print("w-grad[pa/s]:)",u)
#print("Geopotential Altitude[m]:)",z)
#print("Temperature[K]:)",T)



lp.add_function(_fetch_slice)
lp.run('_fetch_slice(TIME,hPa)')
lp.print_stats()


