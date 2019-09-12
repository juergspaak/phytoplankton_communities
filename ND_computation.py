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
    from generate_species import gen_com, n_diff_spe, lambs, dlam
    import I_in_functions as I_in_m
    from differential_functions import own_ode
    from nfd_definitions import numerical_NFD
except ImportError: # to allow importing in a submodule
    from phytoplankton_communities.generate_species import gen_com, n_diff_spe
    from phytoplankton_communities.generate_species import lambs, dlam
    import  phytoplankton_communities.I_in_functions as I_in_m
    from  phytoplankton_communities.differential_functions import own_ode
    from phytoplankton_communities.nfd_definitions import numerical_NFD
    
def find_survivors(equi, species_id):
    # compute the average amount of each species in the communities
    return [np.sum(species_id[equi>0] == i)/equi.shape[-1] 
                    for i in range(n_diff_spe)]

def pigment_richness(equi, alpha):

    return np.sum(np.sum(equi*alpha, axis = -2)>0, axis = -2)

def NFD_phytoplankton(phi, l, k_spec,  equi = True, 
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
                args = (0, I_in_f, k_spec[:,pos[i],i], phi[pos[i],i],
                        l[[i]],zm, k_BG, "per_cap"))
            ND[pos[i],i] = pars["ND"]
            FD[pos[i],i] = pars["FD"]
        except numerical_NFD.InputError:
            pass
    return ND, FD
        
def multi_growth(N,t,I_in, k_spec, phi,l,zm = 1,k_BG = 0, linear = False):
    if linear == "per_cap":
        k_spec = k_spec.reshape(k_spec.shape + (1,))
        phi = phi.reshape(phi.shape + (1,))
    if linear:
        N = N.reshape(-1,len(l))
    # growth rate of the species
    # sum(N_j*k_j(lambda))
    tot_abs = zm*(np.nansum(N*k_spec, axis = 1, keepdims = True) + k_BG)
    if np.any(tot_abs==0):
        growth = phi*simps(k_spec*I_in(t).reshape(-1,1,1), dx = dlam, axis = 0)
    else:
        # growth part
        growth = phi*simps(k_spec/tot_abs*(1-np.exp(-tot_abs))\
                           *I_in(t).reshape(-1,1,1),dx = dlam, axis = 0)
    
    if linear == "per_cap":
        return (growth-l).reshape(-1)
    elif linear:
        return (N*(growth-l)).reshape(-1)
    else:
        return N*(growth -l)

def multispecies_equi(fitness, k_spec, I_in = 50*I_in_m.sun_spectrum["blue sky"],
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
    abs_points = k_spec.copy()
    I_in.shape = -1,1,1
    unfixed = np.full(fitness.shape[-1], True, dtype = bool)
    n = 20
    i = 0
    
    #print(equis.shape, equis_fix.shape, fitness.shape, np.sum(unfixed), abs_points.shape)
    while np.any(unfixed) and i<runs:          
        # sum_i(N_i*sum_j(a_ij*k_j(lam)), shape = (len(lam),itera)
        tot_abs = zm*(np.sum(equis*abs_points, axis = 1,
                             keepdims = True) + k_BG)
        # N_j*k_j(lam), shape = (npi, len(lam), itera)
        all_abs = equis*abs_points
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
            abs_points = abs_points[...,cond]
            fitness = fitness[:,cond]
        i+=1
    return equis_fix, unfixed

def mono_equi_fun(fitness, k_spec, I_in = 50*I_in_m.sun_spectrum["blue sky"],
                      k_BG = np.array([0]), zm = 100, runs = 5000):
    
    # first estimate: assume species absorb all light
    equi_old = fitness/zm * simps(I_in, dx = dlam)
    equi = 2*equi_old
    i = 0
    while np.amax(np.abs(equi_old-equi)/equi_old) > 1e-3 and i < runs:
        equi_old = equi
        equi = fitness/zm * simps(I_in.reshape(-1,1,1)
                    *(1-np.exp(-zm*equi_old*k_spec)), dx = dlam, axis = 0)
        i += 1
        
    
    return equi

def constant_richness(present_species = np.arange(5), n_com = 100, fac = 3,
    I_in = I_in_m.sun_light()(0), 
    randomized_spectra = 0, k_BG = np.array([0]),zm = 100, _iteration = 0,
    species = None):
    """Computes the number of coexisting species in fluctuating incoming light
    
    Returns the richness, biovolume, pigment richness and some other parameters
    at equilibrium. Each of the parameters is returned for each timepoint in
    `t_const` and for the fluctuating time, this is indicated by the value `t`
    
    Parameters:
    present_species: list of integeters
        Id. of present species
    n_com: int
        Number of communities to generate
    fac: float
        maximal factor by which traits of species differ
    l_period: float
        Lenght of period of fluctuating incoming light
    I_in: callable
        Must return an array of shape (len(lambs),). Incoming light at time t
    t_const: array-like
        Times at which species richness must be computed for constant light
    randomized_spectra: float in [0,1]
        percentage by which pigments differ between species
    k_BG: float
        background absorptivity
    _iteration: int
        Should not be set manually, tracks how often function already called
        itself
    
    
        
    Returns:
    richness_equi: array (shape = t)
        species richness at equilibrium
    EF_biovolume: array (shape = 5,t)
        biovolume, i.e. sum of all species abundancies. given are the 
        5,25,50,75,95 percentiles of the communities
    r_pig_equi array, (shape = t)
        pigment richness at equilibrium
    r_pig_start: int
        pigment richness at start of the communtiy (without comp. exclusion)
    prob_spec: array (shape = 5,`t`)
        Probability of finding a community with i species at equilibrium
    prob_pig:
            similar to prob_spec for pigments
    n_fix:
        Number of communities that reached equilibrium in all constant light
        cases.
    """
    ###########################################################################
    # find potentially interesting communities       
    if species is None:
        # generate species and communities
        phi,l,k_spec,alpha, feasible = gen_com(present_species, fac, n_com,
                        I_ins = np.array([I_in]))

    else:
        phi,l,k_spec,alpha, feasible = species
    
    if not feasible:
        return None
        
    if randomized_spectra>0:
        # slightly change the spectra of all species
        # equals interspecific variation of pigments
        eps = randomized_spectra
        k_spec *= np.random.uniform(1-eps, 1+eps, k_spec.shape)
    # compute the equilibria densities for the different light regimes
    equi, unfixed = multispecies_equi(phi/l, k_spec, 
        I_in.view(), runs = 5000*(1+_iteration),k_BG=k_BG, zm = zm)
    # consider only communities, where algorithm found equilibria (all regimes)
    fixed = np.logical_not(unfixed)
    equi = equi[..., fixed]
    phi = phi[:, fixed]
    l = l[fixed]
    k_spec = k_spec[..., fixed]
    alpha = alpha[...,fixed]
    
    # no communities left, call function again with higher precision
    if np.sum(fixed) == 0 and _iteration < 5:
        return constant_richness(present_species, n_com,fac, l_period,I_in,
                 randomized_spectra, k_BG, _iteration+1)
    elif np.sum(fixed) == 0: # increased presicion doesn't suffice, return nan
        return None

    EF = np.nansum(equi, axis = 0) # ecosystem funcitoning at equilibrium
    # compute ND and FD
    ND, FD = NFD_phytoplankton(phi, l, k_spec,  equi, I_in = I_in,
                      k_BG = k_BG, zm = zm)
    sort_index = np.argsort(equi, axis = 0)
    equi_s = np.sort(equi, axis = 0)[:5]
    
    # sort ND and FD according to equilibrium dens
    ND = ND[sort_index, np.arange(sum(fixed))][-5:][::-1]
    FD = FD[sort_index, np.arange(sum(fixed))][-5:][::-1]
    
    # species richness
    richness = np.sum(equi>0, axis = 0)
    r_pig_equi = pigment_richness(equi, alpha)
    # compute pigment richness at the beginning (the same in all communities)
    r_pig_start = pigment_richness(np.ones(equi.shape), alpha)
    # compute complementarity and selection effect
    mono_equi = mono_equi_fun(phi/l, k_spec, I_in, zm)
    delta_RY = equi/mono_equi - 1/len(phi)
    complementarity = len(phi)*np.mean(delta_RY, axis = 0)*np.mean(mono_equi,
                         axis = 0)
    selection = np.sum(delta_RY * mono_equi) - complementarity
    
    # extend lenght of output
    if len(present_species)<5:
        ND = np.append(ND, np.full((5-len(ND), ND.shape[-1]), np.nan)
                    , axis = 0)
        FD = np.append(FD, np.full((5-len(FD), ND.shape[-1]), np.nan)
                    , axis = 0)
        equi_s = np.append(equi_s, np.full((5-len(equi_s), ND.shape[-1])
                ,np.nan), axis = 0)
    return np.array((r_pig_start, r_pig_equi, richness, selection,
            complementarity, EF, *equi_s, *ND, *FD)).T
            

if __name__ == "__main__":
    # short example, can be used for debugging
    present_species = np.arange(5)
    n_com = 20
    fac = 1.1
    l_period = 10
    t_const = [0,0.5]
    randomized_spectra = 0
    present_species = np.arange(5)
    n_com = 100
    fac = 3
    l_period = 10
    I_in = I_in_m.sun_light()(0)
    t_const = [0,0.5]
    randomized_spectra = 0
    k_BG = np.array([0])
    zm = 100
    _iteration = 0
    species = None
    #(richness_equi, EF_biovolume, r_pig_equi, r_pig_start, prob_spec, 
    #        prob_pig, n_fix) = fluctuating_richness()