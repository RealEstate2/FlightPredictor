import bisect
import math

class standardAtmosphere:
    _segments = [0, 11, 20, 32, 47, 51, 71, 84.852] #geopotential altitude segments in km
    _lapseRates = [-6.5, 0, 1, 2.8, 0, -2.8, -2, 0] #lapse rate @ each segment altitude
    _temperatures = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946] #temperature @ each segment altitude
    _pressures = [101.325, 22.632142, 5.474889, 0.868019, 0.110906, 0.066939, 0.003956, 0.000373] #pressure @ each segment altitude

    def qualities(Zh):
        Z = Zh/1000
        g = 9.80665 * (6356.766 / (6356.766 + Z)) ** 2
        H = 6356.766 * Z / (6356.766 + Z)
        i = bisect.bisect(standardAtmosphere._segments, H) - 1
        t = standardAtmosphere._temperatures[i] + standardAtmosphere._lapseRates[i] * (H - standardAtmosphere._segments[i])
        if standardAtmosphere._lapseRates[i] != 0:
            p = standardAtmosphere._pressures[i] * (standardAtmosphere._temperatures[i] / t) ** (34.163195 / standardAtmosphere._lapseRates[i])
        else:
            p = standardAtmosphere._pressures[i] * math.exp(-34.163195 * (H - standardAtmosphere._segments[i]) / standardAtmosphere._temperatures[i])
        ρ = p * 3.483676 / t
        return p, t, ρ, g