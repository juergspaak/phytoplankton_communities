"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Create randomized species according to methods
"""

import pandas as pd
import numpy as np
from scipy.integrate import simps
import warnings

# load pigment absorption spectra
try:
    df = pd.read_csv("pigments.csv")
    # which pigmentation type contains which pigment
    pig_spe_id = pd.read_csv("Pigment_algae_table.csv", index_col = 0)
except (FileNotFoundError,OSError):
    df = pd.read_csv("phytoplankton_communities/pigments.csv")
    # which pigmentation type contains which pigment
    pig_spe_id = pd.read_csv("phytoplankton_communities/"
                             "Pigment_algae_table.csv", index_col = 0)
    
pigment_order =["Chl a", "Chl b", "Chl c", "Peridinin", "Fucoxanthin", "19'-BF",
                 "19'-HF", "beta-carotene", "Phycocyanobilin", "Phycoerythrobilin",
                 "Phycourobilin", "alpha-carotene", "Alloxanthin", "Zeaxanthin",
                 "Diadinoxanthin"]

lambs = df["lambda"].values
dlam = lambs[1]-lambs[0]                     
pigment_names = df.columns[1:] # remove wavelength
pigments = df[pigment_order].values.T
pigments[pigments<0] = 0

species_names = pig_spe_id.columns[2:]
species_pigments = pig_spe_id.loc[pigment_order, species_names].values
species_pigments[np.isnan(species_pigments)] = 0


# photosynthetic active pigments
photo = pig_spe_id.Photosynthetic[pigment_order].values == 1

# from Finkel 2010
size_range = 10**np.array([0,4]) # range of phytoplankton volumes
# maximum and minimal value of photosynthetic efficiency
phi_mean = 2e6 # taken from stomp/landaon
tot_abs_mean = 2.0e-7 # average integrated absorbtion
mean_size = np.prod(np.sqrt(size_range)) # reference size

noise = [1,1]

                 
n_diff_spe = len(species_names) # number of different species
 
def gen_com(present_species, fac, n_com_org = 100, I_ins = None, 
            k_BG = 0, zm = 100, run = 0, photoprotection = True):
    """Generate random species
    
    Generate species with random absorption spectrum according to the table
    of vd Hoek. Species are ensured to survive in monoculture with incoming
    light equal to I_ins
    
    Input:
        present_species: list
            A list containing the identities of each species
        fac: float
            Maximal factor of deviation from parameters in Stomp et al.
        n_com: integer
            number of communities to generate. Actual number of communities
            might differ slightly from this number
        I_ins: `None` or list of incoming lights
            Lights at which each species must be able to survive
            
    Return:
        para: [phi,l]
            The photosynthetic efficiency and loss rate of the species
        k_spec: array with shape (len(lambs), len(present_species),n_com)
            The absorption spectrum of each species in each community
        alphas: array with shape (len(pigments), len(present_species),n_com)
            Concentration of each pigment for each species in each community
            
    units of the returned variables are:
        [phi]: fl * (mu mol photons)^-1
        [l]: h^-1
        [k_spec]: cm^2 * fl^-1
        [alphas]: unitless"""
    if run == 2:
        return None, None, None, None, False
    # internally generate to many species, as some will not survive
    n_com = n_com_org*10
    
    # species richness to be generated
    r_spec = len(present_species)

    
    # check input
    if max(present_species)>=n_diff_spe:
        raise ValueError("maximum of `present_species` must be at most"+
                         "{} entries".format(n_diff_spe))
        
    species_size = np.exp(np.random.uniform(*np.log(size_range),
                                            (r_spec, n_com)))
    
    phi_exponent = -0.06 # Finkel 2010
    # photosynthetic efficiency for each species
    # unit: [phi] = fl * (mumol photons)^-1

    phi = phi_mean*(species_size/mean_size)**phi_exponent
    phi = phi*np.random.uniform(*noise, species_size.shape)
    
    
    # loss rate of the community
    # unit: [l] = h^-1
    l = 0.003*(0.015/0.003)**np.random.uniform(0,1,n_com)

    # Malerba 2010, Finkel 2010/2004, Key 2010
    tot_abs_exponent = np.random.uniform(.23,0.58, species_size.shape)
    tot_abs = tot_abs_mean*(species_size/mean_size)**tot_abs_exponent
    tot_abs = tot_abs*np.random.uniform(*noise, species_size.shape)


    # concentration of each pigment for each species
    # unit: [alphas] = unitless
    alphas_photo = np.random.uniform(1,2,(sum(photo),r_spec,n_com)) *\
                species_pigments[photo][:,present_species, np.newaxis]

    # compute absorption spectrum of each species
    # unit: [k_spec] = cm^2 * fl^-1
    k_photo = np.einsum("pl,psc->lsc",pigments[photo], alphas_photo)
    
    # Total absorption of each species should be equal (similar to Stomp)
    int_abs = simps(k_photo, dx = dlam, axis = 0)
    k_photo = k_photo/int_abs*tot_abs
    
    # change pigment concentrations accordingly
    alphas_photo = alphas_photo/int_abs*tot_abs
    
    # photoprotection
    alphas_prot = np.random.uniform(1,2,(sum(~photo),r_spec,n_com)) *\
                species_pigments[~photo][:,present_species, np.newaxis]
    k_prot = np.einsum("pl,psc->lsc",pigments[~photo], alphas_prot)
    int_abs = simps(k_prot, dx = dlam, axis = 0)

    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # division by 0 is triggered
        k_prot = k_prot/int_abs*tot_abs/4 # 20% non-photosyntetic absorption
        alphas_prot = alphas_prot/int_abs*tot_abs/4
        k_prot[np.isnan(k_prot)] = 0
        alphas_prot[np.isnan(alphas_prot)] = 0
    if photoprotection:
        k_abs = k_prot+k_photo
    else:
        k_abs = k_photo # no photo protection pigments
    # check survivability in monoculture
    if not(I_ins is None):
        surv = mono_culture_survive(phi/l,k_photo, I_ins,k_BG,zm)
        n_surv = min(n_com_org, sum(surv))
        
        # in some unprobable cases this might generate less than n_com species
        if n_surv == 0:
            return gen_com(present_species, fac, n_com_org, I_ins, k_BG, zm,
                           run+1)
        
        # choose from each species n_surv that survived in all light conditions
        spec_id = np.arange(n_com)[surv][:n_surv]
    else:
        spec_id = np.arange(n_com_org)
    
        
        
    # remove species that would not survive
    phi,l = phi[..., spec_id], l[spec_id]
    species_size = species_size[...,spec_id]
    k_photo = k_photo[..., spec_id]
    alphas_photo = alphas_photo[..., spec_id]
    k_abs = k_abs[..., spec_id]
    alphas_prot = alphas_prot[..., spec_id]
    tot_abs = tot_abs[...,spec_id]
    if photoprotection:
        alphas = np.empty((len(photo), ) + alphas_photo.shape[1:])
        alphas[photo] = alphas_photo
        alphas[~photo] = alphas_prot
    else:
        alphas = alphas_photo
    
    return phi,l, k_photo, k_abs, alphas, species_size, tot_abs

def mono_culture_survive(par, k_photo, I_ins, k_BG = 0 ,zm = 100):
    """check whether each species could survive in monoculture
    
    par: phi/l
    k_spec: absorption spectrum
    I_ins: Incoming lights at which species must survive
    
    Returns:
        Surv: boolean array with same shape as par, indicating which species
        survive in all light conditions"""
    # light condition
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # division by 0 is triggered
        light = np.where(zm*k_BG == 0, I_ins, 
                         I_ins*(1-np.exp(-k_BG*zm))/(k_BG*zm))
    light.shape = -1,len(lambs),1,1
    # initial growth rate
    init_growth = par*simps(light*k_photo,dx = dlam,axis = 1)-1
    # initial growth rate must be larger than 0 for all lights
    survive = np.all(init_growth>0,axis = (0,1))
    return survive
    

if __name__ == "__main__":
    # For illustration plot the absorption spectrum of some random species
    import matplotlib.pyplot as plt
    import I_in_functions as I_inf
    from matplotlib.cm import viridis
    
    colors = viridis(np.linspace(0,1, len(pigment_order)))
    
    # Absorption spectrum of all pigments
    fig = plt.figure(figsize=(9,9))
    for i, name in enumerate(pigment_order):
        plt.plot(lambs,pigments[i], label = name,
                 color = colors[i], linestyle = [":", "-", "--"][i%3],
                 linewidth = 3)
    plt.xlabel("nm")
    plt.legend(labels = pigment_order)
    fig.savefig("Pigment_absorption_spectra.pdf")
    # plot the absorption spectrum of random species
    plt.figure()
    phi,l, k_photo, k_abs, alphas, a,b  = gen_com(np.random.randint(11,size = 5),4,1000,
                        50*I_inf.sun_spectrum["blue sky"], I_inf.k_BG["ocean"],
                        photoprotection=True)
    plt.plot(lambs, k_photo[...,0])
    plt.plot(lambs, k_abs[...,0], '--')
    plt.title("example species absorption spectra")