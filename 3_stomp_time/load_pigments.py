"""
@author: J. W. Spaak, jurg.spaak@unamur.be
contains two functions to load/generate pigments
random_pigments: generates n random pigments
real_pigments: loads n predefined (in nature occuring) pigments
random_pigment and realpigments have the same 'in' and 'out' parameters"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

lambs, dlam = np.linspace(400,700,101, retstep = True)

def random_pigments(n):
    """randomly generates `n` absorption spectra
    
    n: number of absorption pigments to create
    
    Returns: pigs, a list of functions. Each function in pigs (`pigs[i]`) is
        the absorption spectrum of this pigment.
    `pig[i](lam)` = sum_peak(gamma*e^(-(lam-peak)**2/sigma)"""
    #number of peaks for each pigment:
    npeak = 2+np.random.randint(5,size = n)
    # location of peaks
    peak = [np.random.uniform(400,700,(1,npeak[i])) for i in range(n)]
    # shape of peack
    sigma = [np.random.uniform(100,900, size = (1,npeak[i])) for i in range(n)]
    # magnitude of peak
    gamma = [np.random.uniform(0,1, size = (1,npeak[i])) for i in range(n)]
    pigs = np.empty((n,101))
    for i in range(n):
        # pigs[i](lam) = sum_peak(gamma*e^(-(lam-peak)**2/sigma)
        pigs[i] = np.sum(gamma[i]*np.exp(-(lambs[:,np.newaxis]-peak[i])**2
                                    /sigma[i]), axis = 1)*10**-8
    return pigs

# load pigments from küpper
path = "../../2_data/3. Different pigments/"

gp_data = pd.read_csv(path + "gp_krueger.csv")
absorptivity = pd.read_csv(path+"absorptivity_Krueger.csv")
names_pigments = list(absorptivity["Pigment"])

a = gp_data.iloc[::3,2:].values
xp = gp_data.iloc[1::3, 2:].values
sig = gp_data.iloc[2::3,2: ].values

kuepper = np.nansum(a*np.exp(-0.5*((xp-lambs.reshape(-1,1,1))/sig)**2),-1).T
kuepper *= 1e-8*absorptivity.iloc[:,1].reshape(-1,1) 

# load additional pogments from vaious authors
df_pigs = pd.read_csv(path + "additional_pigments.csv") 
add_pigs = np.empty((df_pigs.shape[-1]//2, len(lambs)))
names_pigments.extend(df_pigs.columns[1::2])

for i,pigment in enumerate(df_pigs.columns[1::2]):
    add_pigs[i] = interp1d(df_pigs["lambs, "+pigment], df_pigs[pigment])(lambs)

# multiply with absorptivity
add_pigs /= np.nanmax(add_pigs, axis = 1, keepdims = True)
ref_chla = np.nanmax(kuepper[4])
absorptivity = ref_chla*np.array([2.0, 1, 1.5, 0.8,0.8])[:,np.newaxis]
    
pigments = np.append(kuepper, absorptivity*add_pigs, axis = 0)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.plot(pigments.T)
    