"""
@author: J.W.Spaak

Debug functions
Contain functions to check, that the other functions work correctly
"""
import numpy as np
import matplotlib.pyplot as plt

# import all files for Debugfunctions
import numerical_communities as numcom
import numerical_r_i as numri
import analytical_communities as anacom
import analytical_r_i as anari

# create the species
# numerical species with saturating carbon uptake
spec_sat, carb_sat, I_r_sat = numcom.gen_species(numcom.sat_carbon_par, 1000)
# numerical species with photoinhibition
spec_inh, carb_inh,I_r_inh=numcom.gen_species(numcom.photoinhibition_par,1000)
# analytical species with saturating carbon uptake
spec_ana = anacom.gen_species(1000)
I_r_ana = I_r_sat.copy()

    
def plot_rel_diff(values1, values2, ylabel):
    """plot the full percentile curve of relative differences"""
    rel_diff = np.sort(np.abs((values1-values2)/values1).ravel())
    fig,ax = plt.subplots()
    ax.plot(np.linspace(0,100,rel_diff.size),rel_diff)
    ax.semilogy()
    ax.set_ylabel("relative difference in "+ylabel)
    ax.set_xlabel("percentiles")

def plot_abs_diff(values1, values2, ylabel):
    """plot the full percentile curve of relative differences"""
    rel_diff = np.sort(np.abs(values1-values2).ravel())
    fig,ax = plt.subplots()
    ax.plot(np.linspace(0,100,rel_diff.size),rel_diff)
    ax.semilogy()
    ax.set_ylabel("absolute difference in "+ylabel)
    ax.set_xlabel("percentiles")

###############################################################################
# Check that similar functions give similar results in *_communities

def same_equilibria():
    """ to check that the equilibrium function of both files are the same"""
    I_in = np.random.uniform(*I_r_sat, spec_sat.shape[-1])
    equi_num = numcom.equilibrium(spec_sat, carb_sat, I_in, 'simple')
    equi_ana = anacom.equilibrium(spec_sat, I_in, 'simple')
    plot_rel_diff(equi_num, equi_ana, "equilibria")

    
def same_find_balance():
    """ to check that the equilibrium function of both files are the same"""
    bal_num = numcom.find_balance(spec_sat, carb_sat, I_r_sat)
    bal_ana = anacom.find_balance(spec_sat, I_r_sat)
    plot_rel_diff(bal_num, bal_ana, "balance") 
    
###############################################################################
# Show that the assumption I_out = 0 is a good one

def I_out_zero_approx():
    """solves growth rates with assuming I_out =0 and compares to exact"""
    # first axis is for invader/resident, second for the two species
    from scipy.integrate import odeint
    from numerical_r_i import dwdt as dwdt
    
    P = 25 #period length
    C = numcom.equilibrium(spec_sat, carb_sat, np.random.uniform(50,200))
    start_dens = np.array([np.ones(C.shape), C])
    E = np.random.uniform(10,200)
    t = np.linspace(0,P,50)
    sol = odeint(dwdt, start_dens.reshape(-1), t,args=(spec_sat, E, carb_sat))
    sol.shape = len(t),2,2,-1
    rep = np.random.randint(spec_sat.shape[-1])
    #### Resident check:
    k,l = spec_sat[[0,-1]]
    equi = anacom.equilibrium(spec_sat, E, "simple")
    W_rt = equi-(equi-C)*np.exp(-l*t[:, None, None])
    fig,ax = plt.subplots()
    plt.plot(t,W_rt[:, 0,rep],'.')
    plt.plot(t, sol[:, 1,0,rep])
    plt.plot(t,W_rt[:, 1,rep],'.')
    plt.plot(t, sol[:, 1,1,rep])
    ax.set_ylabel("density of resident", fontsize=14)
    ax.set_xlabel("time", fontsize=14)
    ### invader:
    fig,ax = plt.subplots()
    plt.plot(t,sol[:,0,0,rep])
    plt.plot(t,sol[:,0,1,rep])
    ax.set_ylabel("density of invader", fontsize=14)
    ax.set_xlabel("time", fontsize=14)
    W_it = ana_inv_growth(C,E,spec_sat, t.reshape(-1,1,1))
    plt.plot(t, W_it[:,0, rep], '.')
    plt.plot(t, W_it[:,1, rep], '.')

###############################################################################
# Further proofs of functioning code
    
def plot_carbon_uptake():
    """show the different carbon uptake functions"""
    I_r = [0,1000]
    data_sat = carb_sat(spec_sat, np.linspace(*I_r, 100), 'generating')
    data_inh = carb_inh(spec_inh, np.linspace(*I_r, 100), 'generating')
    fig,ax = plt.subplots()
    ax.set_ylabel(r'$p(I)$')
    ax.set_xlabel(r'$I$')
    for k in range(5):
        i = np.random.randint(2)
        plt.plot(np.linspace(*I_r, 100),data_sat[i,k], '--')
        plt.plot(np.linspace(*I_r, 100),data_inh[i,k], ',')
        
###############################################################################
# Compare anayltical solutions to numerical solutions

def r_i_one_period():
    """compare growth rate in one period"""
    # find parameters
    P = 25 #period length
    C = numcom.equilibrium(spec_sat, carb_sat, np.random.uniform(50,200, spec_sat.shape[-1]))
    
    E = np.random.uniform(50,200, (1,spec_sat.shape[-1]))
    
    # numerical r_i
    num_ri = numri.r_i(C,E, spec_sat,P,carb_sat)
    # analytical r_i
    W_it = ana_inv_growth(C,E,spec_sat, P)
    ana_ri = np.log(W_it)/P

    plot_rel_diff(ana_ri, num_ri, "growth in one period")
    plot_abs_diff(ana_ri, num_ri, "growth in one period")
    
def compare_bound_growth_averaged():
    """averaged boundary growth over several lights"""
    itera = 625
    np.random.seed(0) # take the same randomseed to be able to compare them
    ana_ri = ana_bound_growth(spec_sat, I_r_sat, 25,itera)
    np.random.seed(0) # take the same randomseed to be able to compare them
    num_ri = numri.bound_growth(spec_sat, carb_sat, I_r_sat, 25, itera)
    np.random.seed(1) # compare with different seed
    num_ri2 = numri.bound_growth(spec_sat, carb_sat, I_r_sat, 25, itera)
    plot_rel_diff(ana_ri, num_ri, "analytical and numerical r_i")
    # high variance in data
    plot_rel_diff(num_ri2, num_ri, "numerica with different seed")
    
###############################################################################
# Compare anayltical solutions to numerical solutions

def compare_averaged_analytical_ri():
    ave_ri = ana_bound_growth(spec_sat, I_r_sat, 25, int(1e6))
    I_r_mp = I_r_sat[:,np.newaxis]*np.ones(spec_sat.shape[-1])
    envi, comp, stor, mp_ri = anari.mp_approx_r_i(spec_sat, 25, I_r_mp)
    plot_rel_diff(ave_ri, mp_ri[[1,0]], "test")
    plot_abs_diff(ave_ri, mp_ri[[1,0]], "test")
    
   
###############################################################################
# Help functions,not directly related to any comparison, used by many functions
    
def ana_bound_growth(species, I_r ,P, num_iterations= 10000):
    """equivalent function to numri.bound_growth"""
    
    if I_r.ndim == 1: # species are allowed to have different light ranges
        I_r = np.ones((1,species.shape[-1]))*I_r[:,np.newaxis]
    
    Im, IM = I_r #light range

    acc_rel_I = int(np.sqrt(num_iterations)) # number of simulated lights
    # relative distribution of incoming light in previous time period
    rel_I_prev = np.sort(np.random.random((acc_rel_I,1))) #light in prev period
    
    # Effective light, linear transformation
    I_prev = (IM-Im)*rel_I_prev+Im
    # equilibrium densitiy of resident in previous period
    dens_prev = anacom.equilibrium(species, I_prev, "partial")

    # save the growth rates in the periods, different order than in original
    r_is = np.empty((acc_rel_I, acc_rel_I,2,species.shape[-1]))
    for i in range(acc_rel_I): #compute the growth rates for each period
        rel_I_now = np.random.random((acc_rel_I,1)) #light in current period
        I_now = (IM-Im)*rel_I_now+Im
        
        r_i_save = ana_inv_growth(dens_prev[i], I_now, species, P)
        r_is[i] = np.log(r_i_save)/P
        if i%100 == 99:
            print(100*(i+1)/acc_rel_I, " percent done")
        
    # return the average of the growth rates, average over axis = 1 represents
    # average over light in current period, axis 0 to previous period
    return np.average(r_is, axis = (0,1))

def ana_inv_growth(C,E,species, t):
    k,l = species[[0,-1]]
    equi = anacom.equilibrium(species, E) # mode depends on E
    inv = [0,1] # invader indeces
    res = [1,0] # resident inceces
    abso = k[inv]*np.take(equi,inv,-2)/(k[res]*np.take(equi,res,-2))# relative absorption
    W_rt = equi-(equi-C)*np.exp(-l*t)
    W_it = np.exp((np.take(abso,inv,-2)-1)*l[inv]*t)*\
        (np.take(W_rt,res, -2)/C[res])**(np.take(abso,inv,-2)*l[inv]/l[res])
    return W_it

"""functions to include:
    
    compare different r_i functions
    compare r_i analytical with r_i numerical with sat_carbon
    Adapt photo_inh pars to fit -sat pars to compare solutions"""