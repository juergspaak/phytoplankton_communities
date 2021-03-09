"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Contains functions that generate coexisting communities with changing incoming
light

For examples see the sim* files and at the end of the file
"""

import numpy as np
from scipy.integrate import simps
import warnings

try:
    from generate_species import dlam
    import I_in_functions as I_in_m
    from nfd_definitions import numerical_NFD
except ImportError: # to allow importing in a submodule
    from phytoplankton_communities.generate_species import dlam
    import  phytoplankton_communities.I_in_functions as I_in_m
    from phytoplankton_communities.nfd_definitions import numerical_NFD


def NFD_phytoplankton(phi, l, k_photo, k_abs,  equi = True, 
                      I_in = 50*I_in_m.sun_spectrum["blue sky"],
                      k_BG = np.array([0]), zm = 100):
    """Compute the NFD parameters for a community
    
    Parameters
    ----------
    fitness: array (shape = m,n)
        Base fitness, \phi/l for each species
    k_spec: array (shape = len(lambs), m, n)
        Absorption spectrum of the species
    equi: array (shape = m,n)
        Equilibrium density of all species. NFD will only be computed of
        species that have positive equilibrium density.
    I_in: array (shape = len(lambs),m)
        Incoming lights at which equilibrium must be computed
    runs: int
        Number of iterations to find equilibrium
    k_BG: float
        background absorptivity
        
    Returns
    -------
    equis: array, (shape = m,n)
        Equilibrium densitiy of all species, that reached equilibrium
    unfixed: array, (shape = n)
        Boolean array indicating in which communitiy an equilibrium was found
    """
    if equi is True:
        pos = np.full(phi.shape, True, bool).T
    else:
        pos = equi.T>0
    ND, FD = np.full((2,) + phi.shape, np.nan)
    I_in_f = lambda t: I_in
    for i in range(len(l)):
        try:
            n_spec = int(sum(pos[i]))
            if n_spec == 1:
                ND[pos[i],i] = 1
                FD[pos[i],i] = 0
                continue
            pars = dict(N_star = equi[pos[i],i]*np.ones((n_spec, n_spec)))
            pars = numerical_NFD.NFD_model(multi_growth,n_spec = n_spec,
                pars = pars,
                args = (0, I_in_f, k_photo[:,pos[i],i], k_abs[:,pos[i],i],
                    phi[pos[i],i],l[[i]],zm, k_BG, "per_cap"))
            ND[pos[i],i] = pars["ND"]
            FD[pos[i],i] = pars["FD"]
        except numerical_NFD.InputError:
            pass
    return ND, FD
        
def multi_growth(N,t,I_in, k_photo, k_abs, phi,l,zm = 1,k_BG = 0, linear = False):
    if linear == "per_cap":
        k_photo = k_photo.reshape(k_photo.shape + (1,))
        k_abs = k_abs.reshape(k_abs.shape + (1,))
        phi = phi.reshape(phi.shape + (1,))
    if linear:
        N = N.reshape(-1,len(l))

    # sum(N_j*k_j(lambda))
    tot_abs = zm*(np.nansum(N*k_abs, axis = 1, keepdims = True) + k_BG)
    if np.any(tot_abs<=0):
        growth = phi*simps(k_photo*I_in(t).reshape(-1,1,1),
                           dx = dlam, axis = 0)
    else:
        # growth part
        growth = phi*simps(k_photo/tot_abs*(1-np.exp(-tot_abs))\
                           *I_in(t).reshape(-1,1,1),dx = dlam, axis = 0)
    
    if linear == "per_cap":
        return (growth-l).reshape(-1)
    elif linear:
        return (N*(growth-l)).reshape(-1)
    else:
        return N*(growth -l)

def multispecies_equi(fitness, k_photo, k_abs,
                      I_in = 50*I_in_m.sun_spectrum["blue sky"],
                      k_BG = np.array([0]), zm = 100, runs = 5000):
    """Compute the equilibrium density for several species with its pigments
    
    Compute the equilibrium density for the species with the parameters
    fitness and k_spec for the the incoming light I_in (constant over time)
    
    Parameters
    ----------
    fitness: array (shape = m,n)
        Base fitness, \phi/l for each species
    k_spec: array (shape = len(lambs), m, n)
        Absorption spectrum of the species
    I_in: array (shape = len(lambs),m)
        Incoming lights at which equilibrium must be computed
    runs: int
        Number of iterations to find equilibrium
    k_BG: float
        background absorptivity
        
    Returns
    -------
    equis: array, (shape = m,n)
        Equilibrium densitiy of all species, that reached equilibrium
    unfixed: array, (shape = n)
        Boolean array indicating in which communitiy an equilibrium was found
    """
    k_BG = k_BG.reshape(-1,1,1)
    # starting densities for iteration, shape = (npi, itera)
    equis = np.full(fitness.shape, 1e7) # start of iteration
    equis_fix = np.zeros(equis.shape)

    # k_spec(lam), shape = (len(lam), richness, ncom)
    abs_photo = k_photo.copy()
    k_abs = k_abs.copy()
    I_in.shape = -1,1,1
    unfixed = np.full(fitness.shape[-1], True, dtype = bool)
    n = 20
    i = 0
    
    #print(equis.shape, equis_fix.shape, fitness.shape, np.sum(unfixed), abs_points.shape)
    while np.any(unfixed) and i<runs:          
        # sum_i(N_i*sum_j(a_ij*k_j(lam)), shape = (len(lam),itera)
        tot_abs = zm*(np.sum(equis*k_abs, axis = 1,
                             keepdims = True) + k_BG)
        if (tot_abs == 0).any():
            raise
        # N_j*k_j(lam), shape = (npi, len(lam), itera)
        all_abs = equis*abs_photo
        # N_j*k_j(lam)/sum_j(N_j*k_j)*(1-e^(-sum_j(N_j*k_j))), shape =(npi, len(lam), itera)
        y_simps = all_abs/tot_abs*(I_in*(1-np.exp(-tot_abs)))
        # fit*int(y_simps)
        equis = fitness*simps(y_simps, dx = dlam, axis = 0)
        # remove rare species
        equis[equis<1] = 0
        if np.any(np.isnan(equis)):
            print(i, "error occured")
            return(equis, tot_abs)
        if i % n==n-2:
            # to check stability of equilibrium in next run
            equis_old = equis.copy()
        if i % n==n-1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # division by 0 is triggered
                stable = np.logical_or(equis == 0, #no change or already 0
                                   np.abs((equis-equis_old)/equis)<1e-3)
            cond = np.logical_not(np.prod(stable, 0)) #at least one is unstable
            equis_fix[:,unfixed] = equis #copy the values
            # prepare for next runs
            unfixed[unfixed] = cond
            equis = equis[:,cond]
            k_abs = k_abs[...,cond]
            abs_photo = abs_photo[...,cond]
            fitness = fitness[:,cond]
        i+=1
    return equis_fix, unfixed
            

if __name__ == "__main__":
    # short example, can be used for debugging
    present_species = np.arange(5)
    n_com = 20
    fac = 1.1
    l_period = 10
    t_const = [0,0.5]
    randomized_spectra = 0
    #(richness_equi, EF_biovolume, r_pig_equi, r_pig_start, prob_spec, 
    #        prob_pig, n_fix) = fluctuating_richness()