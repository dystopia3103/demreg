#import numpy as np
#
#def dem_reg_map(sigmaa,sigmab,U,W,data,err,reg_tweak,nmu=500):
#    """
#    dem_reg_map
#    computes the regularization parameter
#    
#    Inputs

#    sigmaa: 
#        gsv vector
#    sigmab: 
#        gsv vector
#    U:      
#        gsvd matrix
#    V:      
#        gsvd matrix
#    data:   
#        dn data
#    err:    
#        dn error
#    reg_tweak: 
#        how much to adjust the chisq each iteration
#
##    Outputs
##
#    opt:
#        regularization paramater
#
#    """
# 
#
#   nf=data.shape[0]
#    nreg=sigmaa.shape[0]
#
#    arg=np.zeros([nreg,nmu])
#    discr=np.zeros([nmu])
#
#    sigs=sigmaa[:nf]/sigmab[:nf]
#    maxx=max(sigs)
#    # minx=min(sigs)**2.0*1E-2
#    ## Useful to make the lower limit smaller?
#    minx=min(sigs)**2.0*1E-4
#
#    ## Range from original non-map code
#    # maxx=max(sigs)*1E3
#    # minx=max(sigs)*1E-15
#
#    step=(np.log(maxx)-np.log(minx))/(nmu-1.)
#    mu=np.exp(np.arange(nmu)*step)*minx
#    for kk in np.arange(nf):
#        coef=data@U[kk,:]
#        for ii in np.arange(nmu):
#            arg[kk,ii]=(mu[ii]*sigmab[kk]**2*coef/(sigmaa[kk]**2+mu[ii]*sigmab[kk]**2))**2
#    
#    discr=np.sum(arg,axis=0)-np.sum(err**2)*reg_tweak
#  
#    opt=mu[np.argmin(np.abs(discr))]
#    # print(opt)
#
#    return opt


#%timeit
#cprofile



import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def dem_reg_map(sigmaa, sigmab, U, W, data, err, reg_tweak, nmu=500):
    nf = data.shape[0]
    nreg = sigmaa.shape[0]

    arg = np.zeros((nreg, nmu))
    discr = np.zeros(nmu)

    sigs = sigmaa[:nf] / sigmab[:nf]
    maxx = np.max(sigs)
    minx = np.min(sigs)**2.0 * 1e-4

    step = (np.log(maxx) - np.log(minx)) / (nmu - 1.0)
    mu = np.exp(np.arange(nmu) * step) * minx

    for kk in prange(nf):
        coef = 0.0
        # Equivalent to coef = data @ U[kk, :]
        for jj in range(U.shape[1]):
            coef += data[jj] * U[kk, jj]

        for ii in range(nmu):
            denom = sigmaa[kk]**2 + mu[ii] * sigmab[kk]**2
            val = mu[ii] * sigmab[kk]**2 * coef / denom
            arg[kk, ii] = val * val

    for ii in prange(nmu):
        s = 0.0
        for kk in range(nreg):
            s += arg[kk, ii]
        discr[ii] = s - np.sum(err**2) * reg_tweak

    opt = mu[np.argmin(np.abs(discr))]
    return opt