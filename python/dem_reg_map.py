import numpy as np

def dem_reg_map(sigmaa,sigmab,U,W,data,err,reg_tweak,nmu=500):
    return dem_reg_map_vectorized(sigmaa,sigmab,U,W,data,err,reg_tweak,nmu=500)
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
 

    nf=data.shape[0]
    nreg=sigmaa.shape[0]

    arg=np.zeros([nreg,nmu])
    discr=np.zeros([nmu])

    sigs=sigmaa[:nf]/sigmab[:nf]
    maxx=max(sigs)
    # minx=min(sigs)**2.0*1E-2
    ## Useful to make the lower limit smaller?
    minx=min(sigs)**2.0*1E-4

    ## Range from original non-map code
    # maxx=max(sigs)*1E3
    # minx=max(sigs)*1E-15

    step=(np.log(maxx)-np.log(minx))/(nmu-1.)
    mu=np.exp(np.arange(nmu)*step)*minx
    for kk in np.arange(nf):
        coef=data@U[kk,:]
        for ii in np.arange(nmu):
            arg[kk,ii]=(mu[ii]*sigmab[kk]**2*coef/(sigmaa[kk]**2+mu[ii]*sigmab[kk]**2))**2
    
    discr=np.sum(arg,axis=0)-np.sum(err**2)*reg_tweak
  
    opt=mu[np.argmin(np.abs(discr))]
    # print(opt)

    return opt

def dem_reg_map_vectorized(sigmaa, sigmab, U, W, data, err, reg_tweak, nmu=500):
    nf = data.shape[0]

    sigs = sigmaa[:nf] / sigmab[:nf]

    maxx = np.max(sigs)
    minx = np.min(sigs) ** 2.0 * 1E-4
    step = (np.log(maxx) - np.log(minx)) / (nmu - 1.)
    mu = np.exp(np.arange(nmu) * step) * minx

    # Vectorized computation
    coef = data @ U[:nf, :].T

    mu_2d = mu[np.newaxis, :]
    sigmaa_2d = sigmaa[:nf, np.newaxis]
    sigmab_2d = sigmab[:nf, np.newaxis]
    coef_2d = coef[:, np.newaxis]

    numerator = mu_2d * sigmab_2d ** 2 * coef_2d
    denominator = sigmaa_2d ** 2 + mu_2d * sigmab_2d ** 2
    arg = (numerator / denominator) ** 2

    discr = np.sum(arg, axis=0) - np.sum(err ** 2) * reg_tweak

    opt = mu[np.argmin(np.abs(discr))]

    return opt
