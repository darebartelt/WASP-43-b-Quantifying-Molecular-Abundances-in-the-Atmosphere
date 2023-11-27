#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:32:02 2023

@author: darebartelt
"""

'''
This python script will take a 3-dimensional data cube (number of orders x 
number of exposures x number of pixels per order) and perform a cross-correlation with a
model. The cross correlation will be used to detect the planet's atmosphere by correlating
with a model containing key gases that are found in the atmosphere.
'''

# Import Functions
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from scipy import constants
from scipy import interpolate
from astropy.stats import sigma_clipped_stats

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
    '''This function is the meat of the code. It takes in the data and the model and 
	performs the cross-correlation. Use the comments below to fill it out.

	Variables:
	Vsys - the systemic velocity of the star+planet system through the Milky Way, in km/s
	Kp - the orbital velocity of the planet around the star, in km/s
	scale - for now, leave this set to 1 (you can use it to make models stronger for
		the purposes of testing)
	cs_p - coefficients describing how much the model needs to be broadened to account
		for the planet's rotation
	wlgrid - defines the wavelengths of each data point
	data_arr - the planet transit data, with tellurics and the star removed
	data_scale - the tellurics and star which were removed from the transit data.
		An array in the same shape as data_arr.
	ph - the orbital phases at which the planet was observed. Values are unitless, where
		the center of transit is at a phase of 0 and an orbit goes from phases of 0 to 1.
	Rvel - the velocity of the star relative to the Earth at each time an exposure was
		observed.'''

	#This code is going to do two main things: 1) shift a model in wavelength depending
	#on the Doppler shift of the planet, and 2) cross-correlate the shifted model with
	#the data.

	#define 3 variables which have the same values as the length of each dimension of
	#data_arr. Dimension 0 = number of orders, dimension 1 = number of exposures,
	#dimension 2 = number of pixels per order
    Norders = np.shape(data_arr)[0]
    Nexposures = np.shape(data_arr)[1]
    Npixels = np.shape(data_arr)[2]
    
	#Calculate the radial velocity of the planet as a function of phase. Use the provided
	#equation.
    Radv = Vsys + Rvel + Kp*np.sin(2*np.pi*ph)

	#Use problem 5 from the problem set to calculate the wavelength shift 
	#(lambda_observed-lambda_emitted) for each radial velocity calculated above
    dl_l = Radv*1E3 /constants.c
	#Define a variable that will be equal to the total cross-correlation
    total = 0

	#Create a loop over the number of orders (i.e. for i in range(Norders))
    for i in range(Norders):

		#####PART 1 OF MAIN FUNCTION: Shift the model in wavelength

		#Within this loop, select the ith index of the wavelength grid (wlgrid[i,:])
        selected_wavelength = wlgrid[i,:]
		#Define an array of zeros using numpy.zeros that is of size
		# number of exposures x number of pixels per order
        zeroarr = np.zeros((Nexposures, Npixels))

		#Set up a second loop within the first, which loops over the number of exposures
		#(i.e. for j in range(exposures))
        for j in range(Nexposures):

			#For each exposure, use the wavelength shift calculated above
			#(lambda_observed - lambda_emitted) to shift the selected portion of the
			#wavelength grid to account for the planet's motion
            shifted_waves = selected_wavelength * (1.0 - dl_l[j])
			#Use the function interpolate.splev to interpolate the model to the right
			#wavelengths. The correct syntax is shown below, but you will have to make
			#sure the variable names match. Shifted_waves should be the shifted wavelengths
			#which you calculate on the line above this.
            Depth_p = interpolate.splev(shifted_waves, cs_p, der=0) * scale
            
			#Save Depth_p as the jth row of the empty array of zeros you created before 
			#this loop
            zeroarr[j,:] = Depth_p


		#The next few lines are some operations we perform which I will explain later.
		#Leave these lines as-is, and replace gTemp with whatever is the name of the array
		#you defined to be size number of exposures x number of pixels per order above.

		#NOTE: These lines should be within the loop over the orders, but outside the 
		#loop over the exposures
        fData=(1.-zeroarr)*data_scale[i,] 
        u,ss,vh=np.linalg.svd(fData,full_matrices=False)
        ss[0:nPCA]=0.
        W=np.diag(ss)
        A=np.dot(u,np.dot(W,vh))
        g_unnormalized=A

		#####PART 2 OF MAIN FUNCTION: Cross-correlate the shifted model with the data
		#Now we're going to do the actual cross-correlation! You can use a lot of your
		#code from before here. But first, make a loop that goes through each exposure
		#again
        for k in range(Nexposures):
            g = g_unnormalized[k,:] - np.mean(g_unnormalized[k,:])
            sf2 = (1/Npixels)*np.sum(data_arr[i,k,:]**2)
            sg2 = (1/Npixels)*np.sum(g**2) 
            R = (1/Npixels)*np.sum(data_arr[i,k,:]*g) 
            ccoeff = R/np.sqrt(sf2*sg2) 
            total += ccoeff

			#For the jth exposure, take the ith row of g_unnormalized.
			#Subtract the mean of this row. With the mean subtracted, this variable
			#becomes g(n-s) from the Brogi and Line (2019) paper.

			#Use the un-numbered equation from Brogi and Line (2019) to calculate sg^2 
			#using the g(n-s) you defined above.

			#Use the un-numbered equation from Brogi and Line (2019) to calculate sf^2.
			#For this equation, the variable that is defined as f(n) in the paper is going
			#to be part of the data_arr you feed in above. Specifically, if at this point
			#we're looping over the ith order and the jth exposure, use data_arr[i,j,:]

			#Use the remaining equations in Brogi and Line (2019), following your code
			#you've already written, to calculate the cross-correlation for this order
			#and exposure

			#Add this value to the total ccoeff variable you defined above


    return total

#################

#Here we are reading in the data files I provided you
#need to have three separate ph and Rvel functions for each night
wld1, da1=pickle.load(open('/Users/darebartelt/Documents/Research/IGRINS_transit/PCA_4_clean_data.pic','rb')) #change these file names based on run_pipeline results
wld1, ds1=pickle.load(open('/Users/darebartelt/Documents/Research/IGRINS_transit/PCA_4_noise.pic','rb'))
wld2, da2=pickle.load(open('/Users/darebartelt/Documents/Research/WASP43b_Obs_Data/20230408/PCA_4_clean_data.pic','rb')) #change these file names based on run_pipeline results
wld2, ds2=pickle.load(open('/Users/darebartelt/Documents/Research/WASP43b_Obs_Data/20230408/PCA_4_noise.pic','rb'))
wld3, da3=pickle.load(open('/Users/darebartelt/Documents/Research/WASP43b_Obs_Data/20230426/PCA_4_clean_data.pic','rb')) #change these file names based on run_pipeline results
wld3, ds3=pickle.load(open('/Users/darebartelt/Documents/Research/WASP43b_Obs_Data/20230426/PCA_4_noise.pic','rb'))
ph1 = pickle.load(open('/Users/darebartelt/Documents/Research/IGRINS_transit/phi_phstrip.pic','rb'))[0]      # Time-resolved phases
Rvel1 =pickle.load(open('/Users/darebartelt/Documents/Research/IGRINS_transit/Vbary_phstrip.pic','rb'))[0] # Time-resolved Earth-star velocity
ph2 = pickle.load(open('/Users/darebartelt/Documents/Research/WASP43b_Obs_Data/20230408/phi_phstrip.pic','rb'))[0]      # Time-resolved phases
Rvel2 =pickle.load(open('/Users/darebartelt/Documents/Research/WASP43b_Obs_Data/20230408/Vbary_phstrip.pic','rb'))[0] # Time-resolved Earth-star velocity
ph3 = pickle.load(open('/Users/darebartelt/Documents/Research/WASP43b_Obs_Data/20230426/phi_phstrip.pic','rb'))[0]      # Time-resolved phases
Rvel3 =pickle.load(open('/Users/darebartelt/Documents/Research/WASP43b_Obs_Data/20230426/Vbary_phstrip.pic','rb'))[0] # Time-resolved Earth-star velocity
Kp=211.  #orbital velocity of planet
scale=1. #nuisance parameter, ignore it.
vsini=5.812 #speed of the planet's rotation 
nPCA = 4


#Here we're loading in the model I provided you.
model=np.loadtxt('W43_CO_10e-3.txt')
wl_model=model[:,0]
rprs_model=model[:,1]

#Set up three empty arrays.
#First: using np.linspace, an array of Kp (planet orbital velocity) values that
#spans a range from the expected Kp above minus 50 (using the value listed above) 
#to Kp+50, with a stepsize of 5
kparr = np.linspace(0,300,61)

#Second: using np.linspace, an array of Vsys (star systemic velocity) values that
#span a range from -50 to +50 with a stepsize of 2
varr = np.linspace(-60,60,61)
#number of steps = final-initial/stepsize

#Third: Using np.zeros, a 2D array with dimensions (number of Kp values) x 
#(number of Vsys values)
twodarr = np.zeros((len(kparr),len(varr)))


##This section is taking the model for the planet's absorption and convolving it with
#equations that account for broadening due to the planet's rotation and the behavior
#of the instrument. Leave it as is, and skip ahead!
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

#Set up two nested loops that go over the arrays of Kp values and Vsys values

for i in range(len(kparr)):
    for j in range(len(varr)):
		#For each value of Kp and Vsys:
		#Call the cross_corr_data function to calculate the cross-correlation for that
		#Kp and Vsys. Remember, the inputs in order are:
		#1. The Vsys value you're interested in (pulled from the array)
		#2. The Kp value you're interested in (pulled from the array)
		#3. coeff_spline as defined above (just leave this, you don't need to edit it)
		#4. The wavelength array from the data (wl_data, defined above)
		#5. The planet transit data (data_arr, defined above)
		#6. The telluric noise removed from the transit data (data_scale, defined above)
		#7. The phases of each exposure (ph, defined above)
		#8. The velocity of the star relative to Earth for each exposure (Rvel, defined above)

		#Save this cross-correlation value into the appropriate place in the 2D array
		#you set up above to hold cross-correlation values.
        
        c1 = cross_corr_data(varr[j], kparr[i], scale, coeff_spline, wld1, da1, ds1, ph1, Rvel1, nPCA)
        c2 = cross_corr_data(varr[j], kparr[i], scale, coeff_spline, wld2, da2, ds2, ph2, Rvel2, nPCA)
        c3 = cross_corr_data(varr[j], kparr[i], scale, coeff_spline, wld3, da3, ds3, ph3, Rvel3, nPCA)
        twodarr[i,j] = c1 + c2 + c3 

		#This function can take a long time to run, so I would put some sort of print
		#statement here that for each value you loop through prints something, just so
		#you can make sure the function is making progress. For example, print the
		#indices you're at in both loops.
        print(i,j)

        

#Now, outside the loop:
#Calculate the maximum cross-correlation value in the full array and determine at what
#values of Kp and Vsys it occurs.
maxval = np.unravel_index(np.argmax(twodarr),twodarr.shape)
print('The max value is', maxval)

mean, med, sc = sigma_clipped_stats(twodarr,sigma_lower=3.,sigma_upper=3.) #here CCFarr is whatever you named your array that comes out of the CCF function.
sigmaarr=(twodarr-med)/sc


#Make a plot showing the cross-correlation array using matplotlib.pyplot.imshow,
#and using the "extent" keyword in the imshow documentation online to set the x- and y-
#axis limits.
#Use matplotlib.pyplot.scatter to plot an x shaped marker on top of this image indicating
#the values of Kp and Vsys that produced the maximum cross-correlation signal.
plt.figure()
plt.imshow(twodarr,cmap='viridis',origin='lower',extent=(np.min(varr),np.max(varr),np.min(kparr),np.max(kparr)),aspect=0.5,zorder=0)
c=plt.colorbar()
c.set_label('CCF',fontsize=15)
c.ax.tick_params(labelsize=15,width=2,length=6)
plt.scatter(varr[maxval[1]],kparr[maxval[0]],s=80,color='red',marker='x',zorder=4)
plt.axvline(x=-3.696,color='white',linestyle='--',linewidth=2,zorder=3)
plt.axhline(y=211,color='white',linestyle='--',linewidth=2,zorder=3)
plt.xlabel('Vsys',fontsize=20)
plt.ylabel('Kp',fontsize=20)
plt.tick_params(labelsize=20,axis="both",top=True,right=True,width=2,length=8,direction='in')
#plt.title('CCF for Nights 1-3',fontsize = 20)
#plt.ylim(bottom = 0)
plt.show()

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
#plt.ylim(bottom = 0)
#plt.title('Significance for Nights 1-3',fontsize = 20)
plt.show()

#np.savez('W43_CO2_combined.npz',kparr,varr,twodarr,sigmaarr)
