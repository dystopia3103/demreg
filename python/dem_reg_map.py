import numpy as np
from numba import njit

@njit  # No parallel or fastmath
def dem_reg_map(sigmaa, sigmab, U, W, data, err, reg_tweak, nmu=500):
    """
    dem_reg_map
    computes the regularization parameter
    
    Inputs

    sigmaa: 
        gsv vector
    sigmab: 
        gsv vector
    U:      
        gsvd matrix
    V:      
        gsvd matrix
    data:   
        dn data
    err:    
        dn error
    reg_tweak: 
        how much to adjust the chisq each iteration

    Outputs

    opt:
        regularization paramater

    """

    nf = data.shape[0]
    nreg = sigmaa.shape[0]

    arg = np.zeros((nreg, nmu))
    discr = np.zeros(nmu)

    sigs = sigmaa[:nf] / sigmab[:nf]
    maxx = np.max(sigs)
    minx = np.min(sigs)**2.0 * 1e-4

    step = (np.log(maxx) - np.log(minx)) / (nmu - 1.0)
    mu = np.exp(np.arange(nmu) * step) * minx

    for kk in range(nf):
        coef = 0.0
        for jj in range(U.shape[1]):
            coef += data[jj] * U[kk, jj]
        for ii in range(nmu):
            val = mu[ii] * sigmab[kk]**2 * coef / (sigmaa[kk]**2 + mu[ii] * sigmab[kk]**2)
            arg[kk, ii] = val * val

    # Sum over first axis manually (Numba prefers explicit loops)
    for ii in range(nmu):
        s = 0.0
        for kk in range(nreg):
            s += arg[kk, ii]
        discr[ii] = s - np.sum(err**2) * reg_tweak

    opt_index = 0
    min_abs_discr = abs(discr[0])
    for ii in range(1, nmu):
        abs_val = abs(discr[ii])
        if abs_val < min_abs_discr:
            min_abs_discr = abs_val
            opt_index = ii

    return mu[opt_index]
