## Import functions
#IMPORTANT NOTE: This code won't run on your computer because you don't have the ability
#to parallelize on a personal laptop (and you don't have all the functions anyway).
#Just do your best to fill it out and then we'll go over it.
import pymultinest
import math, os
import pdb
import numpy as np
import pickle
from fm import *
import pickle
import time
from scipy import constants
from numba import jit
from astropy.io import fits
import sys
from joblib import Parallel, delayed

def LOOP(i,Norders,Nexposures,Npixels,data_scale,data_arr,wlgrid, dl_l, cs_p,I, nPCA):
    '''
    This function is very similar to the function you wrote to perform
    cross-correlation. Instead, it calculates a log-likelihood, which is not exactly
    the same quantity but can also be used to measure how good of a fit a model is.
    The only difference is that this loop function starts by selecting a single order
    and then only performs the part of the cross-correlation calculation within the
    first loop from your code you wrote originally.

    Variables:
    i - order number
    Ndet - number of orders total
    Nphi - number of exposures
    Npix - number of pixels per order
    data_scale - the tellurics and star which were removed from the transit data
    data_arr - the planet transit data, with tellurics and the star removed
    wlgrid - defines the wavelengths of each data point
    dl_l - calculated velocity shift (previously, this was inside the cross-correlation
    loop. Now it will be somewhere else in the code.)
    cs_p - coefficients describing how much the model needs to be broadened to account
    for the planet's rotation
    '''

    #Define a variable that will be equal to the total log-likelihood (set it equal to zero for now)
    loglike = 0
    #This is very similar to what we set up for the cross-correlation before, 
    #just using a slightly different value now

    #As in the set-up for calculating cross-correlations, 
    #select the ith index of the wavelength grid (the index corresponding to number of orders)
    selected_wavelength = wlgrid[i,:]
    #Define an array of zeros using numpy.zeros that is of size
    # number of exposures x number of pixels per order
    zeroarr = np.zeros((Nexposures, Npixels))
    # Set up a loop over the number of exposures (i.e., for j in range(exposures))
    for j in range(Nexposures):
        #For each exposure, use the wavelength shift you previously calculated
        #that is called "dl_l" to shift the selected portion of the wavelength
        #grid to account for the planet's motion
        shifted_waves = selected_wavelength * (1.0 - dl_l[j])
        #Using the function interpolate.splev to interpolate the model to the right
        #wavelengths. The correct syntax is shown below, but make sure your variable
        #names match as before.
        Depth_p = interpolate.splev(shifted_waves, cs_p, der=0)
        #Save Depth_p as the jth row of the empty array of zeros you created before
        #this loop
        zeroarr[j,:] = Depth_p
    #The next few lines are some operations we perform which I will explain later.
    #Leave these lines as-is, and replace gTemp with whatever is the name of the array
    #you defined to be size number of exposures x number of pixels per order above.
    fData=(1.-zeroarr)*data_scale[i,]
    u,ss,vh=np.linalg.svd(fData.T,full_matrices=False)  #decompose
    ss[0:nPCA]=0.
    W=np.diag(ss)
    A=np.dot(u,np.dot(W,vh))
    g_unnormalized=A.T

    #Now doing the cross-correlation again, just saving a slightly different value
    #Mke a loop that goes through each exposure again 
    for k in range(Nexposures):
        #For the jth exposure, take the jth row of g_unnormalized.
        #Subtract the mean of this row. With the mean subtracted, this variable
        #becomes g(n-s) from the Brogi and Line (2019) paper.
        # g = g_unnormalized[k,:] - np.mean(g_unnormalized[k,:])
        # #Use the un-numbered equation from Brogi and Line (2019) to calculate sf^2.
        # #For this equation, the variable that is defined as f(n) in the paper is going
        # #to be part of the data_arr you feed in above. Specifically, if at this point
        # #we're looping over the ith order and the jth exposure, use data_arr[i,j,:]
        # sf2 = (1/Npixels)*np.sum(data_arr[i,k,:]**2)
        # #Use the un-numbered equation from Brogi and Line (2019) to calculate sg^2
        # #using the g(n-s) you defined above.
        # sg2 = (1/Npixels)*np.sum(g**2) 
        # R = (1/Npixels)*np.sum(data_arr[i,k,:]*g) 
        # pdb.set_trace()
        # #Continue to use the equations in Brogi and Line (2019), but instead of
        # #calculating the cross-correlation as the last step, use equation 9 to 
        # #calculate the log-likelihood.
        # logL= -(Npixels/2)*np.log(sf2 - (2*R) +sg2)
        # #Add this value to the total log likelihood variable you defined above
        # loglike += logL #change!!

        gVec=g_unnormalized[k,].copy()
        gVec-=(gVec.dot(I))/float(Npixels)  #mean subtracting here...
        sg2=(gVec.dot(gVec))/float(Npixels)
        fVec=data_arr[i,k,].copy() # already mean-subtracted
        sf2=(fVec.dot(fVec))/Npixels
        R=(fVec.dot(gVec))/Npixels # cross-covariance
        #CC=R/np.sqrt(sf2*sg2) # cross-correlation   
        #CCF+=CC
        loglike+=(-0.5*Npixels * np.log(sf2+sg2-2.0*R))

    #Return the log likelihood as the function output
    return loglike   


#This is the outside part of the log-likelihood calculation, which is very similar
#to the outer loop of the cross-correlation function. We set it up in this way because
#it makes it easy to run in parallel on a computer cluster if we divide up the orders.
def log_likelihood_PCA(Vsys, Kp, dphi, cs_p, wlgrid, data_arr,data_scale, ph, Rvel, nPCA):
    '''
    This function takes in the data and model and performs the outer loop of your
    standard cross-correlation calculation

    Variables:
    Vsys - the systemic velocity of the star+planet system through the Milky Way, in km/s
    Kp - the orbital velocity of the planet around the star, in km/s
    dphi - An offset from the expected mid-transit time (effectively accounts for
    small, unexpected differences in the orbital phase at different times)
    cs_p - coefficients describing how much the model needs to be broadened to account
        for the planet's rotation
    wlgrid - defines the wavelengths of each data point
    data_arr - the planet transit data, with tellurics and the star removed
    data_scale - the tellurics and star which were removed from the transit data.
        An array in the same shape as data_arr.
    '''

    #define 3 variables which have the same values as the length of each dimension of
    #data_arr. Dimension 0 = number of orders, dimension 1 = number of exposures,
    #dimension 2 = number of pixels per order
    Norders = np.shape(data_arr)[0]
    Nexposures = np.shape(data_arr)[1]
    Npixels = np.shape(data_arr)[2]
    #Use numpy.ones to set up an array of ones that is the same length as the number of 
    #pixels per order
    ones = np.ones((Npixels))
    #Calculate the radial velocity of the planet as a function of phase. Use the provided
    #equation. However, instead of just using the phase phi, include the phase offset
    # - so use (phi+dphi)
    Radv = Vsys + Rvel + Kp*np.sin(2*np.pi*(ph-dphi))
    #Use problem 5 from the problem set to calculate the wavelength shift 
    #(lambda_observed-lambda_emitted) for each radial velocity calculated above
    dl_l = Radv*1E3 /constants.c

    #This line separates the individual orders to feed them into the other half of the
    #calculation in the other function above. Don't change it, but check the definition
    #of the function above to make sure you use the same variable names in the code
    #here (j, Ndet, Nphi, etc.)
    logL_i=Parallel(n_jobs=42)(delayed(LOOP)(j,Norders,Nexposures,Npixels, data_scale,data_arr,wlgrid, dl_l, cs_p,ones,nPCA) for j in range(Norders))
    logL=np.sum(logL_i)
    
    return logL

#################


#This function sets up priors for the fitting code we're using (pymultinest).
#The format for a single parameter is: cube[0] = x + y*cube[0], where x is the
#lower limit and x+y is the upper limit. Each time you set a new parameter,
#change the number in brackets (e.g., cube[1], cube[2],...) and put each
#on a new line
def prior(cube, ndim, nparams):
        #following the format described above, set up the following parameters:
        #Water abundance, ranging from -12 to -1
        cube[0] = -12 + 11.*cube[0]
        #CO2 abundance, ranging from -12 to -1
        cube[1] = -12 + 11.*cube[1]
        #Temperature, ranging from 500 to 3000
        cube[2] = 500 + 2500.*cube[2]
        #A dummy parameter that scales the radius, ranging from 0.5 to 1.5
        cube[3] = 0.5 + cube[3]
        #A pressure where clouds block our view of the atmosphere, ranging from -6 to 0
        cube[4] = -6 + 6.*cube[4]
        #The Kp of the planet signal, ranging from 50 less than the predicted value to 50 more
        cube[5] = (211-50) + (211+100)*cube[5]
        #The Vsys of the planet signal, ranging from 25 less than the predicted value to 25 more
        cube[6] = (-4 - 25) + (46)*cube[6]
        #The uncertainty in phase (dphi), ranging from -0.01 to 0.01
        cube[7] = -0.01 + 0.02*cube[7]



#This function runs the pymultinest. You won't have to edit it except the single
#marked line.
def loglike(cube, ndim, nparams):
    print('****************')
    t0=time.time()

    logH2O,  logCO2=cube[0],cube[1]
    Tiso, xRp, logPc,Kp,Vsys,dphi=cube[2],cube[3],cube[4],cube[5],cube[6],cube[7]
    scale=1.
  
    xx=np.array([logH2O, logCO2, Tiso,   xRp, logPc]) 
 
    wno,Depth=fx_trans(xx)
    Depth=Depth[::-1]
    wl_model=1E4/wno[::-1]
    t1=time.time()

    #In the line below, replace 3.3 with the speed of the planet's rotation for WASP-43
    ker_rot=get_rot_ker(5.812, wl_model)
    Depth_conv_rot = np.convolve(Depth,ker_rot,mode='same')

    xker = np.arange(41)-20
    sigma = 5.5/(2.* np.sqrt(2.0*np.log(2.0)))
    yker = np.exp(-0.5 * (xker / sigma)**2.0)
    yker /= yker.sum()
    Depth_conv = np.convolve(Depth_conv_rot,yker,mode='same')*scale
    cs_p = interpolate.splrep(wl_model,Depth_conv,s=0.0) 
    t2=time.time()

    logL_1=log_likelihood_PCA(Vsys, Kp,dphi, cs_p, wld1, da1, ds1, ph1, Rvel1, 5)
    logL_2=log_likelihood_PCA(Vsys, Kp,dphi, cs_p, wld2, da2, ds2, ph2, Rvel2, 6)
    logL_3=log_likelihood_PCA(Vsys, Kp,dphi, cs_p, wld3, da3, ds3, ph3, Rvel3, 5)
     
    logL_M = logL_1 + logL_2 + logL_3


    print(logH2O, logCO2,Kp,Vsys, scale)
     
    loglikelihood=logL_M

    t3=time.time()
    print('TOTAL: ',t3-t0)
    print('logL ',t3-t2)
    print('Conv/Interp: ',t2-t1)
    print('GPU RT: ',t1-t0)
    return loglikelihood


# =============================================================================
# #Read in your data files the same way you did before (just use one night for now)
# wl_data, data_arr=pickle.load(open('./0408/PCA_6_clean_data.pic','rb'))
# wl_data, data_scale=pickle.load(open('./0408/PCA_6_noise.pic','rb'))
# ph = pickle.load(open('./0408/phi_phstrip.pic','rb'))[0]
# Rvel =pickle.load(open('./0408/Vbary_phstrip.pic','rb'))[0]
# =============================================================================
wld1, da1=pickle.load(open('/Users/darebartelt/Documents/Research/IGRINS_transit/PCA_5_clean_data.pic','rb')) 
wld1, ds1=pickle.load(open('/Users/darebartelt/Documents/Research/IGRINS_transit/PCA_5_noise.pic','rb'))

wld2, da2=pickle.load(open('/Users/darebartelt/Documents/Research/WASP43b_Obs_Data/20230408/PCA_6_clean_data.pic','rb')) 
wld2, ds2=pickle.load(open('/Users/darebartelt/Documents/Research/WASP43b_Obs_Data/20230408/PCA_6_noise.pic','rb'))

wld3, da3=pickle.load(open('/Users/darebartelt/Documents/Research/WASP43b_Obs_Data/20230426/PCA_5_clean_data.pic','rb')) 
wld3, ds3=pickle.load(open('/Users/darebartelt/Documents/Research/WASP43b_Obs_Data/20230426/PCA_5_noise.pic','rb'))

ph1 = pickle.load(open('/Users/darebartelt/Documents/Research/IGRINS_transit/phi_phstrip.pic','rb'))[0]      
Rvel1 =pickle.load(open('/Users/darebartelt/Documents/Research/IGRINS_transit/Vbary_phstrip.pic','rb'))[0] 

ph2 = pickle.load(open('/Users/darebartelt/Documents/Research/WASP43b_Obs_Data/20230408/phi_phstrip.pic','rb'))[0]      
Rvel2=pickle.load(open('/Users/darebartelt/Documents/Research/WASP43b_Obs_Data/20230408/Vbary_phstrip.pic','rb'))[0] 

ph3 = pickle.load(open('/Users/darebartelt/Documents/Research/WASP43b_Obs_Data/20230426/phi_phstrip.pic','rb'))[0]   
Rvel3=pickle.load(open('/Users/darebartelt/Documents/Research/WASP43b_Obs_Data/20230426/Vbary_phstrip.pic','rb'))[0] 

#You don't need to worry about the stuff below here; I'll edit it.
outfile='PMN_WASP43_H2O_CO2_0408.pic' 
n_params=8 #number of params (count the number of indecies in cube..it's too dumb to figure it out on its own)
Nlive=500 #number of live points (100-1000 is good....the more the better, but the "slower")
res=False #if the job crashes or you get the boot, you can change this to "true" to restart it from where it left off
pymultinest.run(loglike, prior, n_params, outputfiles_basename='./chains/firstrun_0408_H2O_CO2/H2O_CO2_',resume=res, verbose=True,n_live_points=Nlive)
a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename='./chains/firstrun_0408_H2O_CO2/H2O_CO2_')
s = a.get_stats()
output=a.get_equal_weighted_posterior()
pickle.dump(output,open(outfile,"wb"))





