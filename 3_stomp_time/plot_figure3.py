"""
@author: J.W. Spaak, jurg.spaak@unamur.be

Plots the regression of pigment richness, real data, purly random and 
model prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import linregress


# contains information about biodiversity and pigment richness
datas_biodiv = {}
# contains information about EF and pigment richness
datas_EF = {}

# folder where all datas are saved
folder = "../../2_data/5. EF/"

# Datas of Estrada
###############################################################################
c_est = "red" # color for all estrada data
estrada = pd.read_csv(folder+"estrada2.csv", delimiter = ",",
                      engine = "python")
estrada = estrada.convert_objects(convert_numeric = True)
estrada = estrada[estrada.Salinity<16]

datas_biodiv["estrada"] = [estrada['SP Pigments detected by HPLC'],
             estrada['SM Phyto-plankton taxa'], c_est]

datas_EF["estrada, chl a"] = [estrada['SP Pigments detected by HPLC'].values,
         np.nansum(estrada.iloc[:,8:-3].values, axis = 1), c_est,'^']

###############################################################################
c_lab = "blue"
# Datas of Striebel
striebel = pd.read_csv(folder+"Striebel,lab.csv",delimiter = ",")
datas_biodiv["striebel, exp"] = [striebel["Pigment richness"].values, 
             np.exp(striebel["ln taxon richness"].values), c_lab]
datas_EF["striebel, exp"] = [striebel["Pigment richness"].values, 
             np.exp(striebel["ln wet mass"].values), c_lab]

# datas of striebel field
c_fie = "cyan"
striebel_field_pigs = pd.read_csv(folder+"Striebel,field,pigments.csv",
                                  delimiter= ",")
striebel_field_spec = pd.read_csv(folder+"Striebel,field,species.csv",
                                  delimiter= ",")                    
striebel_field_spec = striebel_field_spec.convert_objects(convert_numeric = 1)
r_pig = np.nansum(striebel_field_pigs.iloc[:,1:-2]>0,axis = 1)
datas_biodiv["striebel, field"] = [r_pig                
                ,np.nansum(striebel_field_spec.iloc[:,1:-1]>0,axis = 1), c_fie]
datas_EF["striebel, field"] = [r_pig, 
           1e-9*np.nansum(striebel_field_spec.iloc[:,1:-1],axis = 1), c_fie]
datas_EF["striebel, field, pigments"] = [r_pig, striebel_field_pigs["Gesamt"].values, 
         c_fie, '^']
         
###############################################################################
c_fietz = "orange"
# Datas of Striebel
fietz = pd.read_csv(folder+"Fietz.csv",delimiter = ",", engine = "python")
r_pig_fietz = np.sum(fietz.iloc[:,3:28]>0, axis = 1).values
datas_biodiv["Fietz"] = [r_pig_fietz, np.sum(fietz.iloc[:,28:].values>0, axis = 1),
                            c_fietz]
datas_EF["Fietz"] = [r_pig_fietz, 1e3*np.nansum(fietz.iloc[:,28:].values, axis = 1)
             ,c_fietz]

# datas of striebel field
c_fie = "cyan"
striebel_field_pigs = pd.read_csv(folder+"Striebel,field,pigments.csv",
                                  delimiter= ",")
striebel_field_spec = pd.read_csv(folder+"Striebel,field,species.csv",
                                  delimiter= ",")                    
striebel_field_spec = striebel_field_spec.convert_objects(convert_numeric = 1)
r_pig = np.nansum(striebel_field_pigs.iloc[:,1:-2]>0,axis = 1)
datas_biodiv["striebel, field"] = [r_pig                
                ,np.nansum(striebel_field_spec.iloc[:,1:-1]>0,axis = 1), c_fie]
datas_EF["striebel, field"] = [r_pig, 
           1e-9*np.nansum(striebel_field_spec.iloc[:,1:-1],axis = 1), c_fie]
datas_EF["striebel, field, pigments"] = [r_pig, striebel_field_pigs["Gesamt"].values, 
         c_fie, '^']


###############################################################################
# datas of Spaak
c_sta = "lime"
c_coe = "green"
t = "240"
def medians(x_val, y_val):
    # compute averages
    x_val, y_val = spaak_data[x_val], spaak_data[y_val]
    x_range = np.arange(min(x_val), max(x_val)+1)
    return x_range, np.array([np.nanmedian(y_val[x_val==x]) for x in x_range])
    
spaak_data = pd.read_csv("data/data_EF_time[0, 0, 0].csv")
datas_biodiv["spaak, t="+t] = [*medians("r_pig, start","r_spec, t="+t),c_sta]
datas_biodiv["spaak, equi"] = [*medians("r_pig, start","r_spec, equi"),c_coe]

for col in spaak_data.columns:
    if col[:2] == "EF":
        spaak_data[col] *=1e-9               
                        
pig_range, EF_equi = medians("r_pig, start", "EF, equi")
pig_range, EF_t15 = medians("r_pig, start", "EF, t="+t)
datas_EF["spaak, equi"] = [pig_range, EF_equi, c_coe]
datas_EF["spaak, t="+t] = [pig_range, EF_t15, c_sta]


###############################################################################
# plot boxes
def boxs(x_val, y_val, x_range,ax,color):
    # compute averages
    x_val, y_val = spaak_data[x_val], spaak_data[y_val]
    def_col = dict(color= color)
    ax.boxplot([y_val[x_val==x] for x in x_range], boxprops = def_col,
               whiskerprops = def_col, capprops = def_col,
               medianprops = def_col, showfliers = False)

def plot_results(dictionary, ylabel, ax_org, twinx = False, legend = True):
    ax_org.set_ylabel(ylabel)
    if twinx:
        ax_cop = ax_org.twinx()
        ax_cop.set_ylabel("Total pigment concentration")
    handles = []
    
    for i,key in enumerate(sorted(dictionary.keys())):
        ax = ax_org
        x,y,col = dictionary[key][:3]
        handles.append(mpatches.Patch(color = col, label = key))
        try:
            marker = dictionary[key][3]
            ax = ax_cop
        except IndexError:
            marker = '.'
        
        x,y = x[np.isfinite(x*np.log(y))], y[np.isfinite(x*np.log(y))]
        
        ax.plot(x,y,linewidth = 0,marker = marker, label = key, color = col)
        ax.semilogy()
        y = np.log(y)
            
        
        slope, intercept,r,p,stderr = linregress(x,y)
        ran = np.array([min(x), max(x)])
        y_linregres = intercept + ran*slope

        y_linregres = np.exp(y_linregres)
        ax.plot(ran, y_linregres, color = col)
    if legend:
        ax_org.legend(handles = handles,loc = "best", numpoints = 1)
    ax_org.set_xlabel("Pigment richness")
        
fig, ax = plt.subplots(1,2,figsize = (9,9), sharex = True)
ax[0].set_title("A")
ax[1].set_title("B")

pig_range = range(1,23)
boxs("r_pig, start", "r_spec, t="+t,pig_range,ax[0], c_sta)
boxs("r_pig, start", "r_spec, equi",pig_range,ax[0], c_coe)

boxs("r_pig, start", "EF, t="+t,pig_range,ax[1], c_sta)
boxs("r_pig, start", "EF, equi",pig_range,ax[1], c_coe)

plot_results(datas_biodiv, "Species richness",ax[0])
print("new")
plot_results(datas_EF,r"Biovolume $[fl ml^{-1}]$",ax[1], True, False)
ax[0].set_xlim(3.5,23.5)
plt.xticks(range(4,23,2),range(4,23,2))
fig.savefig("Figure, biodiv-EF.pdf")

