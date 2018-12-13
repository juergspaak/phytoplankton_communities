"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Plot a figure used in the main text,
for more information check the description in the main text

Plot the absorption spectrum of all pigments involved
Plot the example absorption spectrum of two coexisting species
"""

import numpy as np
import matplotlib.pyplot as plt


from generate_species import gen_com, pigments, pigment_names, lambs
import richness_computation as rc
from I_in_functions import sun_spectrum

# fix randomness
np.random.seed(hash(5))

I_in = 40*sun_spectrum["direct full"]

phi,l,k_spec,alpha,found = gen_com([13,15], 3, 50,I_ins = I_in)

equi,unfixed = rc.multispecies_equi(phi/l,k_spec, I_in)

# remove all species where we have not found equilibrium
equi = equi*(1-unfixed)

index = np.argmax(np.sum(equi>0, axis = 0))
pig_colors = ["green", "darkolivegreen", "lime", "yellowgreen",
              "purple", "brown", "cyan", "red", "orange"]

def plot_pigments(array, ax, ls = '-', lw = 2):
    for i in range(len(pig_colors)):
        if max(array[i])>0:
            ax.plot(lambs, array[i], ls,color = pig_colors[i], 
                label = i, linewidth = lw)

fig = plt.figure(figsize = (8,6))
ax_pig = fig.add_subplot(211)
plot_pigments(pigments, ax_pig)
ax_pig.legend(pigment_names, ncol = 3, fontsize = 10)

ax_ex1 = fig.add_subplot(223)
ax_ex2 = fig.add_subplot(224)
ax_ex2.get_shared_y_axes().join(ax_ex1,ax_ex2)



# plot the absorption spectrum of the species
ax_ex1.plot(lambs,10**9*(k_spec)[:,0, index], linewidth = 2, color = "black",
        label = "Prochlorophyta")
# add the decomposition of the absorption spectra
plot_pigments(10**9*alpha[:,0,index, np.newaxis]*pigments,ax_ex1, ls = '-'
              , lw = 1)

# plot the absorption spectrum of the species
ax_ex2.plot(lambs,10**9*(k_spec)[:,1, index], linewidth = 2, color = "black",
        label = "Dinophyta")
# add the decomposition of the absorption spectra
plot_pigments(10**9*alpha[:,1,index, np.newaxis]*pigments,ax_ex2, ls = '-'
              , lw = 1)

fs = 12

# labels and legend
ax_pig.set_xlabel("Wavelength [nm]", fontsize = fs)
ax_ex1.set_xlabel("Wavelength [nm]", fontsize = fs)
ax_ex2.set_xlabel("Wavelength [nm]", fontsize = fs)

ax_pig.set_ylabel(r"Absorptiom [$cm^{-1}mM^{-1}$]", fontsize = fs)
ax_ex1.set_ylabel(r"Absorption [$10^{-9}cm^2fl^{-1}$]", 
                    fontsize = fs)

# add titles
ax_pig.set_title("A", loc = "left")
ax_ex1.set_title("B", loc = "left")
ax_ex2.set_title("C", loc = "left")
fig.tight_layout()

fig.savefig("Figure, absorption_spectra.pdf")