# -*- coding: utf-8 -*-
cimport cython
from cython_gsl cimport *

import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sampler(features,
            rankings,
            np.ndarray[np.float_t, ndim=1] prior_shape,
            np.ndarray[np.float_t, ndim=1] prior_rate,
            int repl = 10000,
            int burn = 50,
            old_trace = None):
    """
    Executes a Markov Chain Monte Carlo simulation to estimate the coefficients of a 
    Plackett-Luce(PL) regression model by using an augmented variable Gibbs sampling scheme.
    
    PL regression coefficients are sampled in a two step process. For every ranking i = 1,...,N,
    exponential random variables named Z are drawn for each ordinal position except the last-place finisher. 
    Next, discrete random variables for p = 1,...,P for each ordinal position except the last-place finisher.
    These random variables enable the computation of sufficient statistics for the parameters of the full 
    conditional distribution of the PL regression coefficients. This full conditional distribution is a Gamma
    random variable with shape and rate parameters.
        - features is a list of 2-dim arrays containing input variables (i.e. independent variables, covariates, etc.)
          taking ONLY non-negative values. the list index corresponds to an observation in the dataset. the row index
          of each array indexes an object to be ranked and the column index of each array corresponds to a feature
          of that object.
        - rankings is a list of 1-dim arrays that contain the ordinal-ranking vector for each observation in the
          dataset, each ranking vector is an array of indices whose index corresponds to the rank of that individual. 
          for instance, rankings[i][3] returns the k = 1,...,K  index of the individual that finished 4th in the
          i-th observation
        - prior_shape is a 1-dim array of P elements, shape parameters for the PL regression coefficients prior distribution
        - prior_rate is a 1-dim array of P elements, rate parameters for the PL regression coefficients prior distribution
        - repl is an integer of the desired number of mcmc samples
        - burn is an integer representing the number of samples in the burn-in phase
        - old trace(optional) is an array of MCMC samples from a previous sampler
    """                                
    cdef:
        gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        Py_ssize_t i,ii,iii,j,p,rep
        int N = len(features)
        int P = features[0].shape[1]
        int n        
        float Z_rate = 0.0
        float Z = 0.0
        float sum_pi = 0.0
        float u
        np.ndarray[np.float_t, ndim=1] flat_features = np.concatenate(map(lambda x,y: x[y[::-1]].flatten(), features, rankings))
        np.ndarray[np.float_t, ndim=1] feature_sums = np.concatenate(map(lambda x,y: x[y[::-1]].cumsum(0).flatten(), features, rankings))
        np.ndarray[np.int_t, ndim=1] num_items = np.array(map(len,rankings), dtype=np.int)
        np.ndarray[np.float_t, ndim=1] lam = np.empty(P, dtype=np.float)
        np.ndarray[np.float_t, ndim=1] posterior_shape = np.empty(P, dtype=np.float)
        np.ndarray[np.float_t, ndim=1] posterior_rate = np.empty(P, dtype=np.float)
        np.ndarray[np.float_t, ndim=1] pi = np.empty(P, dtype=np.float)
        np.ndarray[np.float_t, ndim=2] trace = np.vstack((np.zeros((0,P)) if old_trace is None else old_trace, np.zeros((repl, P), dtype=np.float)))
        int T = 0 if old_trace is None else old_trace.shape[0]
        
    gsl_rng_set(r,np.random.randint(2**32))        
    
    if old_trace is None:
        for p in xrange(P):
            lam[p] = gsl_ran_gamma(r,prior_shape[p],1./prior_rate[p])
    else:
        lam[:] = trace[T-1]
    
    for rep in xrange(repl+burn):
        
        posterior_shape[:] = prior_shape[:]
        posterior_rate[:] = prior_rate[:]
        
        j = 0
        for i in xrange(N):
            n = num_items[i]
            Z_rate = 0.0
            for p in xrange(P):
                Z_rate += flat_features[j*P+p] * lam[p]
            for ii in xrange(1,n):
                sum_pi = 0.0
                for p in xrange(P):
                    pi[p] = flat_features[(j+ii)*P+p] * lam[p]
                    sum_pi += pi[p]
                Z_rate += sum_pi
                Z = gsl_ran_exponential(r,1./Z_rate)
                for p in xrange(P):
                    posterior_rate[p] += feature_sums[(j+ii)*P+p] * Z
                iii = 0
                u = gsl_ran_flat(r,0.0,sum_pi) - pi[iii]
                while u > 0.0:
                    iii += 1
                    u -= pi[iii]
                posterior_shape[iii] += 1.
            j += n
            
        for p in xrange(P):
            lam[p] = gsl_ran_gamma(r,posterior_shape[p],1./posterior_rate[p])
        
        if rep >= burn:
            trace[T+rep-burn,:] = lam
            
#        if rep % 1000 == 0:
#            print "rep #{}:{}".format(rep,lam)
            
    return trace