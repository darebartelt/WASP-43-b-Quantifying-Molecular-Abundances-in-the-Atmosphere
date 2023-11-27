#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:09:41 2023

@author: darebartelt
"""
import numpy as np
import os
os.environ['pRT_input_data_path'] = '/Users/darebartelt/Documents/Research/input_data'
from petitRADTRANS import Radtrans
import petitRADTRANS.nat_cst as nc
import pylab as plt

atmosphere = Radtrans(line_species = ['CO2_main_iso'],
                      rayleigh_species = ['H2', 'He'],
                      continuum_opacities = ['H2-H2', 'H2-He'],
                      wlen_bords_micron = [1.2, 2.8],
                      mode = 'lbl')

pressures = np.logspace(-10, 2, 130)
atmosphere.setup_opa_structure(pressures)

R_pl = 0.93*nc.r_jup_mean 
gravity = 5100 #find in m/s^2
P0 = 0.01

temperature = 1426.7 * np.ones_like(pressures) #equilibrium temp? this is the only one I could find

mass_fractions = {}
mass_fractions['H2'] = 0.74 * np.ones_like(temperature)
mass_fractions['He'] = 0.24 * np.ones_like(temperature)
mass_fractions['CO2_main_iso'] = 0.001 * np.ones_like(temperature)

MMW = 2.33 * np.ones_like(temperature)

atmosphere.calc_transm(temperature, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0)

plt.rcParams['figure.figsize'] = (10, 6)

xarr = nc.c/atmosphere.freq/1e-4
yarr = ((atmosphere.transm_rad/100)/4.5936e8)**2
#print(xarr)

plt.plot(xarr, yarr) #divide by the radius of the star in meters, and then square it
#/4.5936e8)**2
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit radius ($\rm R_{Jup}$)')
plt.show()
plt.clf()
#save as a text file with two columns; one for each axis
f = open('W43_CO2.txt','w')
n = len(xarr)
for i in range(n):
    f.write(f'{xarr[i]:.7f}\t{yarr[i]:.7f}\n') #writing the two arrays into the file
f.close()#closing the file when done writing
