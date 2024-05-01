#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 12:55:26 2024

@author: darebartelt
All code, except segments explicitly stated as written by someone else, were written by me.
"""

'''
This code has two main purposes: 1) shift a model in wavelength depending
 the Doppler shift of the planet, and 2) cross-correlate the shifted model with the data.
The cross correlation will be used to detect the planet's atmosphere by correlating
with a model containing key gases that are found in the atmosphere. For this planet, we are
cross-correlating models of water, methane, carbon monoxide, and carbon dioxide.
'''

# Import Functions
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from scipy import constants
from scipy import interpolate
from astropy.stats import sigma_clipped_stats

#I did not write lines 25-42
def get_rot_ker(vsini, wStar):
	'''
	This function calculates how much the lines in the model are broadened due to 
	the rotation of the planet. You can leave it here with no edits.
	'''
	nx, = wStar.shape
	dRV = np.mean(2.0*(wStar[1:]-wStar[0:-1])/(wStar[1:]+wStar[0:-1]))*2.998E5
	nker = 401
	hnker = (nker-1)//2
	rker = np.zeros(nker)
	for ii in range(nker):
		ik = ii - hnker
		x = ik*dRV / vsini
		if np.abs(x) < 1.0:
			y = np.sqrt(1-x**2)
			rker[ii] = y
	rker /= rker.sum()
	return rker


def cross_corr_data(Vsys, Kp, scale, cs_p, wlgrid, data_arr,data_scale,ph,Rvel, nPCA):
    '''This is the primary function of this code, as it takes the data and model and performs the
    cross-correlation

	Variables:
	Vsys - the systemic velocity of the star+planet system through the Milky Way, in km/s
	Kp - the orbital velocity of the planet around the star, in km/s
	scale - for now, leave this set to 1 (you can use it to make models stronger for
		the purposes of testing)
	cs_p - coefficients describing how much the model needs to be broadened to account
		for the planet's rotation
	wlgrid - defines the wavelengths of each data point
	data_arr - the planet transit data, with tellurics and stellar signals removed
	data_scale - the tellurics and stellar signals which were removed from the transit data.
		An array in the same shape as data_arr.
	ph - the orbital phases at which the planet was observed. The center of transit is at a 
    phase of 0 and an orbit goes from phases of 0 to 1.
	Rvel - the velocity of the star relative to the Earth at each time an exposure was
		observed.
    nPCA - the number of components removed in the data reduction (Principal Component Analysis)
    process '''

	#These arrays are all the same length as the planet transit data array (data_arr)
    Norders = np.shape(data_arr)[0] #number of orders
    Nexposures = np.shape(data_arr)[1] #number of exposures
    Npixels = np.shape(data_arr)[2] #number of pixels
    
	#Calculates the radial velocity of the planet as a function of phase
    Radv = Vsys + Rvel + Kp*np.sin(2*np.pi*ph)

	#Calculates the wavelength shift (lambda_observed-lambda_emitted) for each radial velocity 
    #calculated above
    dl_l = Radv*1E3 /constants.c
	
    total = 0 #this variable will be equal to the total cross-correlation

	#This loops over the number of Norders
    for i in range(Norders):

		#####PART 1 OF MAIN FUNCTION: Shift the model in wavelength

		#Within this loop, select the ith index of the wavelength grid (wlgrid[i,:])
        selected_wavelength = wlgrid[i,:]
		#Define an array of zeros using numpy.zeros that is of size
		# number of exposures x number of pixels per order
        zeroarr = np.zeros((Nexposures, Npixels))

		#This loops over the number of exposures
        for j in range(Nexposures):

			#For each exposure, the wavelength shift calculated above shifts
            #the selected portion of thethe wavelength grid to account for the planet's motion
            shifted_waves = selected_wavelength * (1.0 - dl_l[j])
			#Use the function interpolate.splev to interpolate the model to the right
			#wavelengths. The correct syntax is shown below, but you will have to make
			#sure the variable names match. Shifted_waves should be the shifted wavelengths
			#which you calculate on the line above this.
            Depth_p = interpolate.splev(shifted_waves, cs_p, der=0) * scale #I did not write this line
            
			#Saves Depth_p as the jth row of the empty array of zeros 
            zeroarr[j,:] = Depth_p


		#I did not write lines 112-117
        #using numpy.linalg.svd to perform principal component analysis and separate out components
        fData=(1.-zeroarr)*data_scale[i,] 
        u,ss,vh=np.linalg.svd(fData,full_matrices=False)
        ss[0:nPCA]=0.
        W=np.diag(ss)
        A=np.dot(u,np.dot(W,vh))
        g_unnormalized=A

		##This is where the cross-correlation is performed. This method is from 
        #Brogi and Line (2019).
        for k in range(Nexposures):
            #For the jth exposure, take the ith row of g_unnormalized and subtract the 
            #mean of this row. 
            g = g_unnormalized[k,:] - np.mean(g_unnormalized[k,:]) #creates the template spectrum
            sf2 = (1/Npixels)*np.sum(data_arr[i,k,:]**2) #calculates the variance of the data
            sg2 = (1/Npixels)*np.sum(g**2) #calculates the variance of the model
            R = (1/Npixels)*np.sum(data_arr[i,k,:]*g) #calculates the cross-covariance
            ccoeff = R/np.sqrt(sf2*sg2)  #cross-correlation to Log-L mapping
            #Add this value to the total ccoeff variable 
            total += ccoeff


    return total #returns the cross-correlated data

#################

#These lines read in the necessary files from the data reduction
wl_data, data_arr=pickle.load(open('PCA_4_clean_data.pic','rb')) #PCA has been performed to remove tellurics and stellar signals
wl_data, data_scale=pickle.load(open('PCA_4_noise.pic','rb'))#contains the tellurics and the stellar signal
ph = pickle.load(open('/Users/darebartelt/Documents/Research/01:29:2024 Data/phi_phstrip.pic','rb'))[0]    # Time-resolved phases
Rvel =pickle.load(open('/Users/darebartelt/Documents/Research/01:29:2024 Data/Vbary_phstrip.pic','rb'))[0] # Time-resolved Earth-star velocity
Kp=211.  #orbital velocity of planet
scale=1. #setting as 1
vsini=5.812 #speed of the planet's rotation 
nPCA = 4 #number of components removed by Principal Component Analysis

#loading in the desired molecular model
model=np.loadtxt('/Users/darebartelt/Documents/Research/IGRINS_transit/W43_H2O_10e-3.txt')
#I did not write lines 150-151
wl_model=model[:,0]
rprs_model=model[:,1]

#This is an array of Kp (planet orbital velocity) values that
#spans a range from the expected Kp above minus 50 to Kp+50, with a stepsize of 5
kparr = np.linspace(0,300,61)

#SThis is an array of Vsys (star systemic velocity) values that
#span a range from -50 to +50 with a stepsize of 2
varr = np.linspace(-50,50,51)
#number of steps = final-initial/stepsize

#This is a 2D array with dimensions (number of Kp values) x (number of Vsys values)
twodarr = np.zeros((len(kparr),len(varr)))

##This section is taking the model for the planet's absorption and convolving it with
#equations that account for broadening due to the planet's rotation and the behavior
#of the instrument. Leave it as is, and skip ahead!
#I did not write lines 170-181
#convolving model with instrument broadening and broadening due to planetary rotation
ker_rot=get_rot_ker(vsini, wl_model)
model_conv_rot = np.convolve(rprs_model,ker_rot,mode='same')

xker = np.arange(41)-20
modelres=np.mean(wl_model[1:]/np.diff(wl_model))
scalefactor=modelres/45000.
sigma = scalefactor/(2.* np.sqrt(2.0*np.log(2.0)))  #nominal 

yker = np.exp(-0.5 * (xker / sigma)**2.0)
yker /= yker.sum()
rprs_conv_final = np.convolve(model_conv_rot,yker,mode='same')
coeff_spline = interpolate.splrep(wl_model,rprs_conv_final,s=0.0)

#For each value of Kp and Vsys, we are calling the cross_corr_data function to 
#calculate the cross-correlation for that Kp and Vsys
for i in range(len(kparr)):
    for j in range(len(varr)):

		#Save this cross-correlation value into the appropriate place in the 2D array
		#you set up above to hold cross-correlation values.
        twodarr[i,j] = cross_corr_data(varr[j], kparr[i], scale, coeff_spline, wl_data, data_arr,data_scale,ph,Rvel,nPCA)
        
        #This print function exists to ensure that the function is progressing, as it can take
        #time to run. This tells me what indices I'm at for Kp and Vsys
        print(i,j)

        
#Calculates the maximum cross-correlation value in the full array and determines at what
#values of Kp and Vsys it occurs.
maxval = np.unravel_index(np.argmax(twodarr),twodarr.shape)
print('The max value is', maxval)

#Did not write lines 204-205
#calculating 3 sigma clipped median to get signal-to-noise
mean, med, sc = sigma_clipped_stats(twodarr,sigma_lower=3.,sigma_upper=3.) 
sigmaarr=(twodarr-med)/sc


#Plotting the cross-correlation array using matplotlib.pyplot.imshow,
#and using the "extent" keyword in the imshow documentation online to set the x- and y-
#axis limits.
#matplotlib.pyplot.scatter is used to plot an x shaped marker on top of this image indicating
#the values of Kp and Vsys that produced the maximum cross-correlation signal.
plt.figure()
plt.imshow(twodarr,cmap='viridis',origin='lower',extent=(np.min(varr),np.max(varr),np.min(kparr),np.max(kparr)),aspect=0.5,zorder=0)
c=plt.colorbar()
c.set_label('CCF',fontsize=15)
c.ax.tick_params(labelsize=15,width=2,length=6)
plt.scatter(varr[maxval[1]],kparr[maxval[0]],s=80,color='red',marker='x',zorder=4)
plt.axvline(x=-3.696,color='white',linestyle='--',linewidth=2,zorder=3) #marks the literature value of Vsys
plt.axhline(y=211,color='white',linestyle='--',linewidth=2,zorder=3) #marks the literature value of Kp
plt.xlabel('Vsys',fontsize=20)
plt.ylabel('Kp',fontsize=20)
plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
plt.show()
#This plot shows the detection significance of the cross-correlation
plt.figure()
plt.imshow(sigmaarr,cmap='viridis',origin='lower',extent=(np.min(varr),np.max(varr),np.min(kparr),np.max(kparr)),aspect=0.5,zorder=0)
c=plt.colorbar()
c.set_label('Significance (sigma)',fontsize=15)
c.ax.tick_params(labelsize=15,width=2,length=6)
plt.scatter(varr[maxval[1]],kparr[maxval[0]],s=80,color='red',marker='x',zorder=4)
plt.axvline(x=-3.696,color='white',linestyle='--',linewidth=2,zorder=3)
plt.axhline(y=211,color='white',linestyle='--',linewidth=2,zorder=3)
plt.xlabel('Vsys',fontsize=20)
plt.ylabel('Kp',fontsize=20)
plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
plt.title('Significance for H2O',fontsize = 20)
plt.show()

#np.savez('W43_H2O_2024.npz',kparr,varr,twodarr,sigmaarr) 
