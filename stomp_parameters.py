# -*- coding: utf-8 -*-
"""
Generates parameters for the stomp model
"""
from scipy.interpolate import interp1d
import math
from scipy.integrate import quad, odeint
import numpy as np
import matplotlib.pyplot as plt

absor_val = np.load("npz,absorption.npz") #loads the absorption curves of the cyanos


k_red = interp1d(absor_val['x_red'], 10**-9*absor_val['y_red'], 'cubic')
k_green = interp1d(absor_val['x_green'], 10**-9*absor_val['y_green'], 'cubic')
k = lambda lam: np.array([k_green(lam), k_red(lam)])

l = np.array([0.014,0.014])  #specific loss rate [h^-1]
l /=3600            #change to [s^-1]

phi = 10**6*np.array([1.6,1.6])   # photosynthetic efficiency [fl*mumol^-1]
zm = 7.7          #total depth [m]
N = np.array([1,1]) # density [fl*cm^-3]
I_in_prev = lambda t,l: 1
int_I_in = 40  # light over entire spectrum [mumol ph*m^-2*s^-1]
I_in = lambda t,l: I_in_prev(t,l)*int_I_in/300

def growth(N, t, absor = 'both' ):
    if absor == 'both':
        abs_fun = lambda lam: sum(N*k(lam))
    else: 
        abs_fun = lambda lam: (N*k(lam))[absor]
    #plotter(abs_fun,400,700)
    integrand = lambda lam, col: I_in(t,lam)*k(lam)[col]/(abs_fun(lam))*\
                            (1-math.exp(-abs_fun(lam)*zm))
    gamma0 = quad(lambda lam: integrand(lam,0), 400,700)[0]
    gamma1 = quad(lambda lam: integrand(lam,1), 400,700)[0]
    gamma = np.array([gamma0,gamma1])
    return (phi/zm*gamma-l)*N

time = np.linspace(0,1000,50)

N_time = odeint(growth, np.array([0,1]),time, args = (1,))
plt.plot(time,N_time)