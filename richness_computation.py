"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Contains functions that generate coexisting communities with changing incoming
light
"""
import numpy as np
from scipy.integrate import simps

from pigments import lambs, dlam
from generate_species import gen_com, n_diff_spe
import I_in_functions as I_in

from differential_functions import own_ode
                    
def I_in_t(I_in1, I_in2, period):
    # returns light at given time t, fluctuates between I_in1, I_in2
    def fun(t):
        t_rel = (t%period)/period
        part_1 = 2*np.abs(t_rel-0.5)
        return part_1*I_in1+(1-part_1)*I_in2
    return fun
    
def find_survivors(equi, species_id):
    # compute the average amount of each species in the communities
    return [np.sum(species_id[equi>0] == i)/equi.shape[-1] 
                    for i in range(n_diff_spe)]

def pigment_richness(equi, alpha):
    return np.mean(np.sum(np.sum(equi*alpha, axis = -2)>0, axis = -2),-1)

# standard incoming light fluctuation    
I_in_ref = I_in.fluc_nconst([I_in.I_in_def(40,450,50), 
                            I_in.I_in_def(40,650,50)],10)

def multispecies_equi(fitness, k_spec, I_in = I_in.I_in_def(40),runs = 5000, 
                      k_BG = 0):
    """Compute the equilibrium density for several species with its pigments
    
    Computes `itera` randomly selected communities, each community contains
    at most len(`pigs`) different species. Returns equilibrium densities
    for each community.
    
    Parameters
    ----------
    pigs : list of functions
        Each element `pig` of pigs must be the absorption spectrum of that 
        pigment. Each `pig` must be a function that returns a float
    itera : int, optional
        number of generated communities
    runs : int, optional
        number of iterations to find equilibrium
    av_fit: float, optional
        Average fitness of all species
    pow_fit: float, optional
        Fitness of each species ill be in [1/pow_fit, pow_fit]*av_fit
    per_fix: Bool, optional
        Percent of fixed species is printed if True
    sing_pig: Bool, optional
        Determines if species have only one pigment. If False, species 
        absorption spectrum will be a sum of different pigments        
        
    Returns
    -------
    equis:
        Equilibrium densitiy of all species, that reached equilibrium      
    """
    # starting densities for iteration, shape = (npi, itera)
    equis = np.full(fitness.shape, 1e7) # start of iteration
    equis_fix = np.zeros(equis.shape)

    # k_spec(lam), shape = (len(lam), richness, ncom)
    abs_points = k_spec.copy()
    I_in = I_in[:,np.newaxis]
    unfixed = np.full(fitness.shape[-1], True, dtype = bool)
    n = 20
    i = 0
    #print(equis.shape, equis_fix.shape, fitness.shape, np.sum(unfixed), abs_points.shape)
    while np.sum(unfixed)/equis.shape[-1]>0.01 and i<runs:          
        # sum_i(N_i*sum_j(a_ij*k_j(lam)), shape = (len(lam),itera)
        tot_abs = np.einsum('ni,lni->li', equis, abs_points)
        # N_j*k_j(lam), shape = (npi, len(lam), itera)
        all_abs = equis*abs_points #np.einsum('ni,li->nli', equis, abs_points)
        # N_j*k_j(lam)/sum_j(N_j*k_j)*(1-e^(-sum_j(N_j*k_j))), shape =(npi, len(lam), itera)
        y_simps = all_abs*(I_in/tot_abs*(1-np.exp(-tot_abs)))[:,np.newaxis]
        # fit*int(y_simps)
        equis = fitness*simps(y_simps, dx = dlam, axis = 0)
        # remove rare species
        equis[equis<1] = 0
        if i % n==n-2:
            # to check stability of equilibrium in next run
            equis_old = equis.copy()
        if i % n==n-1:
            
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
    #return only communities that found equilibrium
    return equis_fix, unfixed

def fluctuating_richness(present_species = np.arange(5),
    n_com = 100,fac = 3, l_period = 10, I_in = I_in_ref, t_const = [0,0.5],
     no_super = True, randomized_spectra = 0,k_BG = 0, iteration = 0):
    """Computes the number of coexisting species
    
    Parameters:
    r_spec: int
        richness of species in the regional community
    n_com: int
        Number of communities to generate
    fac: float
        maximal factor by which traits of species differ
    l_period: float
        Lenght of period of fluctuating incoming light
    I_in: callable
        Must return an array of shape (101,). Incoming light at time t
    t_const: array-like
        Times at which species richness must be computed for constant light
    randomized_spectra: float
        amout by which pigments spectra differ from species
    allow_shortcut: boolean
        If True, when for all constant light cases the same species coexist,
        then fluctuating incoming light is not computed
    
    Returns:
    ret_mat: array, shape (len(t_const)+2, 10)
        ret_mat[i,j] Percentages of communities that have j coexisting species
        in the incoming light situation j. j in range(len(t_const)) means
        I_in(t_const[j]*period) as incoming light. j=len(t_const) is the 
        maximum of all constant incoming lights and the last one is the
        fluctuation incoming light
    intens_const:
        Intensity of outcoming light for the constant incoming light cases
    intens_fluct:
        Intensity of outcoming light for the fluctuating incoming light case"""
    ###########################################################################
    # find potentially interesting communities
             
    # generate species and communities
    [phi,l],k_spec,alpha = gen_com(present_species, fac, n_com, case = 2,
                    I_ins = np.array([I_in(t*l_period) for t in t_const]),
                    no_super=no_super)
    # compute pigment richness at the beginning (the same in all communities)
    r_pig_start = pigment_richness(1, alpha)
    if randomized_spectra>0:
        # slightly change the spectra of all species
        # equals interspecific variation of pigments
        eps = randomized_spectra
        k_spec *= np.random.uniform(1-eps, 1+eps, k_spec.shape)
    
    # compute the equilibria densities for the different light regimes
    equi = np.empty((len(t_const),) + phi.shape)
    unfixed = np.empty((len(t_const),phi.shape[-1]))
    
    for i,t in list(enumerate(t_const)):
        equi[i], unfixed[i] = multispecies_equi(phi/l, k_spec, 
            I_in(t*l_period), runs = 5000*(1+iteration),k_BG=k_BG)
    # consider only communities, where algorithm found equilibria (all regimes)
    fixed = np.logical_not(np.sum(unfixed, axis = 0))
    equi = equi[..., fixed]
    phi = phi[:, fixed]
    l = l[:, fixed]
    k_spec = k_spec[..., fixed]
    alpha = alpha[...,fixed]
    n_fix = np.sum(fixed)
    
    # no communities left, call function again with higher precision
    if np.sum(fixed) == 0 and iteration < 5:
        return fluctuating_richness(present_species, n_com,fac, l_period,I_in,
                 t_const,no_super,randomized_spectra, k_BG, iteration+1)
    elif np.sum(fixed) == 0: # increased presicion doesn't suffice, return nan
        return (np.full(len(t_const)+1,np.nan), np.full((len(t_const)+1,5),np.nan),
                np.full(len(t_const)+1,np.nan),r_pig_start, 
                np.full((len(t_const)+1,6),np.nan), 
                np.full((len(t_const)+1,len(alpha)+1),np.nan),0)
        

    ###########################################################################
    # return values for constant cases
    # richness in constant lights
    richness_equi = np.zeros(len(t_const)+1)
    richness_equi[:-1] = np.mean(np.sum(equi>0, axis = 1),axis = 1)
            
    r_pig_equi = np.zeros(len(t_const)+1)
    r_pig_equi[:-1] = pigment_richness(equi[:,np.newaxis], alpha)

    # Compute EF, biovolume
    EF_biovolume = np.full((len(t_const)+1,5),np.nan)
    EF_biovolume[:len(t_const)] = np.percentile(np.sum(equi, axis = 1),
                                [5,25,50,75,95], axis = -1).T

    # find the probabilities of coexisting species
    prob_spec = np.zeros((len(t_const)+1,6))
    nr_coex = np.sum(equi>0,axis = 1)
    # probabilities to fine 0 to 5 species
    prob_spec[:-1] = np.mean(nr_coex[...,np.newaxis] ==[0,1,2,3,4,5],axis = -2)
    
    prob_pig = np.zeros((len(t_const)+1,len(alpha)+1))
    nr_pig = np.sum(np.sum(equi[:,np.newaxis]*alpha,axis=-2)>0,axis = -2)
    prob_pig[:-1] = np.mean(nr_pig[...,np.newaxis] == np.arange(len(alpha)+1),
            axis = 1)
    
    ###########################################################################
    # Prepare computation for fluctuating incoming light
    # set 0 all species that did not survive in any of the cases
    dead = np.sum((equi>0), axis = 0)==0
    phi[dead] = 0
    l[dead] = 1 # to avoid division by 0
    k_spec[:,dead] = 0

    # maximal richness over all environments in one community
    max_spec = np.amax(np.sum(equi>0, axis = 1))
    # sort them accordingly to throw rest away
    com_ax = np.arange(equi.shape[-1])
    spec_sort = np.argsort(np.amax(equi,axis = 0), axis = 0)[-max_spec:]
    phi = phi[spec_sort, com_ax]         
    l = l[spec_sort, com_ax]
    equi = equi[np.arange(len(t_const)).reshape(-1,1,1),spec_sort, com_ax]
    k_spec = k_spec[np.arange(len(lambs)).reshape(-1,1,1),spec_sort, com_ax]
    alpha = alpha[:,spec_sort, com_ax]
       
    ###########################################################################
    # take average densitiy over all lights for the starting density
    start_dens = np.mean(equi, axis = 0)

    def multi_growth(N,t,I_in, k_spec, phi,l):
        # growth rate of the species
        # sum(N_j*k_j(lambda))
        tot_abs = np.einsum("sc,lsc->lc", N, k_spec)[:,np.newaxis]+k_BG
        # growth part
        growth = phi*simps(k_spec/tot_abs*(1-np.exp(-tot_abs))\
                           *I_in(t).reshape(-1,1,1),dx = dlam, axis = 0)
        return N*(growth-l)
    
    n_period = 100 # number of periods to simulate in one run
    
    undone = np.arange(phi.shape[-1])
    # compute 100 periods, 10 timepoints per period
    time = np.linspace(0,l_period*n_period,n_period*10)
    phit,lt,k_spect = phi.copy(), l.copy(), k_spec.copy()
    # to save the solutions found
    sols = np.empty((10,)+phi.shape)
    
    # simulate densities
    counter = 1 # to avoid infinite loops

    while len(undone)>0 and counter <1000:
        sol = own_ode(multi_growth,start_dens, time[[0,-1]], 
                      I_in, k_spect, phit,lt,steps = len(time))
        
        # determine change in densities, av at end and after finding equilibria
        av_end = np.average(sol[-10:], axis = 0) 
        av_start = np.average(sol[-110:-100], axis = 0) 
        
        # relative difference in start and end
        rel_diff = np.nanmax(np.abs((av_end-av_start)/av_start),axis = 0)
        # communities that still change "a lot"
        unfixed = rel_diff>0.005
        
        # save equilibria found
        sols[...,undone] = sol[-10:]
        
        # select next communities
        undone = undone[unfixed]
        phit,lt,k_spect = phi[:, undone], l[:, undone], k_spec[...,undone]
        start_dens = sol[-1,:,unfixed].T
        # remove very rare species
        start_dens[start_dens<start_dens.sum(axis = 0)/10000] = 0
        counter += 1

    #######################################################################
    # preparing return values for richnesses computation
    EF_biovolume[-1] = np.percentile(np.sum(sols, axis = (0,1)),
                                [5,25,50,75,95], axis = -1)/len(sols)
    # find average of coexisting species through time
    richness_equi[-1] = np.mean(np.sum(sols[-1]>0,axis = 0))
    r_pig_equi[-1] = pigment_richness(sols[-1], alpha)
    
    # find the probabilities of coexisting species
    nr_coex = np.sum(sols[-1]>0,axis = 0)
    # probabilities to fine 0 to 5 species
    prob_spec[-1] = np.mean(nr_coex[:,np.newaxis] == [0,1,2,3,4,5],axis = 0)
    # include cases with more than 5 species in the last entry
    prob_spec[:,-1] = 1-np.sum(prob_spec,axis = 1)
    
    nr_pig = np.sum(np.sum(sols[-1]*alpha,axis=-2)>0,axis = -2)
    prob_pig[-1] = np.mean(nr_pig[...,np.newaxis] == np.arange(len(alpha)+1),
            axis = 0)
    
    return (richness_equi, EF_biovolume, r_pig_equi, r_pig_start, prob_spec, 
            prob_pig, n_fix)
            

if __name__ == "__main__":
    present_species = np.arange(5)
    n_com = 20
    fac = 1.1
    l_period = 10
    I_in = I_in_ref
    t_const = [0,0.5]
    randomized_spectra = 0
    (richness_equi, EF_biovolume, r_pig_equi, r_pig_start, prob_spec, 
            prob_pig) = fluctuating_richness()