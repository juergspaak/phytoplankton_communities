"""
@author: J. W. Spaak, jurg.spaak@unamur.be

create different fluctuating incoming light functions
for illustrations of the different fluctuating types see: plot_ap_I_In_fluct.py
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import simps

try:
    from generate_species import lambs, dlam
except ImportError: # to allow importing in a submodule
    from phytoplankton_communities.generate_species import lambs, dlam


###############################################################################
# Sun light data
try:
    sun_data = pd.read_csv("sunlight.csv")
except (OSError,FileNotFoundError):
    sun_data = pd.read_csv("phytoplankton_communities/sunlight.csv")
# units: [sun_spectrum] = (mu mol photons) * m^-2 * s^-1 * nm^-1
sun_spectrum = {}    
for key in ["usual", "direct full", "blue sky", "clouded"]:
    x = sun_data["lambs, " + key]
    y = sun_data[key]
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    sun_spectrum[key] = interp1d(x,y)(lambs)
    sun_spectrum[key] = sun_spectrum[key]/simps(sun_spectrum[key],dx = dlam)

def sun_light(lux = [20,200], period = 10, sky = "blue sky"):
    """returns whithe light that changes total intensity over time
    
    lux: array of shape (2), minimal and maximal lux of the incoming light
    period: float, length of period
    
    Returns: I_in(t), function returning incoming light at given time t"""
    

    def I_in(t):
        "gives white light at time t"
        
        alpha = (np.cos(2*t*np.pi/period)+1)/2
        lux_t = alpha*lux[0] + (1-alpha)*lux[1]
        return lux_t*sun_spectrum[sky]
        
    return I_in
    
###############################################################################
# background absorption
try:
    background = pd.read_csv("background_absorption.csv")
except (OSError,FileNotFoundError):
    background = pd.read_csv("phytoplankton_communities/"
                           "background_absorption.csv")


k_water = background["lambs, water"], background["water"]
ind = np.isfinite(k_water[0])
k_water = interp1d(k_water[0][ind], k_water[1][ind])(lambs)

k_gli_tri = interp1d(background["lambs, GliTri"], background["GliTri"])(lambs)

# different background absorptions depending on the environment
# units: [k_BG_rel] = cm^-1
k_BG = {"ocean": k_water/100,
        "baltic sea": (k_water + k_gli_tri)/100,
        "peat lake": (k_water + k_gli_tri/k_gli_tri[50]*2)/100,
        "clear": k_water*0}

# mixing depth in meters depending on the environment
# units: [zm] = cm
zm = 100

if __name__ == "__main__":# examples    
    import matplotlib.pyplot as plt
    
    # incoming light spectra
    plt.figure()
    for key in sun_spectrum.keys():
        plt.plot(lambs, sun_spectrum[key], label = key)
    plt.legend(loc = "lower center")
    plt.title("Sunlight spectrum")
    
    # effects of background absorption
    plt.figure()
    