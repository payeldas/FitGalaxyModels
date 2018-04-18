""" PROBABILITY DISTRIBUTION FUNCTIONS 
This module collates a number of probability distribution functions that could form an extended distribution function (EDF).

Author:
    Payel Das

To do:
    Implement more probability distribution functions as they come my way.
"""

import numpy as np

## GAUSSIAN DISTRIBUTION FUNCTION
    # x    - input                                  [vector]
    # pars - parameters of distribution (mu,sigma)  [vector]
    # Returns distribution function density
def gaussDf(x,pars):
    
    x = np.asarray(x)

    # Parameters
    mu,sig = pars
        
    return (1./sig/np.sqrt(2.*np.pi) * np.exp(-((x-mu)**2/(2.*sig**2))))

## LOGNORMAL DISTRIBUTION FUNCTION
    # x - input                                                     [vector]
    # pars - parameters of distribution (mu, sigma of Gaussian)     [vector]
    # Returns distribution function density
def lognormDf(x,pars):
    
    x = np.asarray(x)
    
    # Parameters
    mu,sig = pars
                
    # DF
    df         = np.zeros_like(x)
    index      = (x-mu)<0
    df[np.asarray(index)]  = 0.
    df[np.asarray(~index)] = np.exp(-0.5*((np.log(x[np.asarray(~index)]-mu))/sig)**2)/((x[np.asarray(~index)]-mu)*sig*np.sqrt(2.*np.pi))
        
    return df
    
## LOGNORMAL DISTRIBUTION FUNCTION (ALTERNATIVE FORMAT)
    # x    - input                                                     [vector]
    # pars - parameters of distribution (xmax, xpeak)                  [vector]
    # Returns distribution function density
def lognorm2Df(x,pars):
    
    x = np.asarray(x)
    
    # Parameters
    xmax,xpeak = pars
                
    # DF
    g          = -np.log(xmax-x)
    sig        = np.sqrt(-np.log(xmax-xpeak))
    df         = np.zeros_like(x)
    index      = x>xmax
    df[np.asarray(index)]  = 0.
    df[np.asarray(~index)] = np.exp(g[np.asarray(~index)])*np.exp(-g[np.asarray(~index)]**2./(2.*sig**2.))/(sig*np.sqrt(2.*np.pi))
        
    return df
    
## LOGNORMAL DISTRIBUTION FUNCTION (ANOTHER ALTERNATIVE FORMAT)
    # x    - input                                                     [vector]
    # pars - parameters of distribution (xmax, xpeak, mu)              [vector]
    # Returns distribution function density
def lognorm3Df(x,pars):
    
    x = np.asarray(x)
    
    # Parameters
    xmax,xpeak,mu = pars
                
    # DF
    g          = -np.log(xmax-x) + mu
    sig        = np.sqrt(-np.log(xmax-xpeak) + mu)
    df         = np.zeros_like(x)
    index      = x>xmax
    df[np.asarray(index)]  = 0.
    df[np.asarray(~index)] = np.exp(g[np.asarray(~index)])*np.exp(-g[np.asarray(~index)]**2./(2.*sig**2.))/(sig*np.sqrt(2.*np.pi))
        
    return df    
    
## BROKEN POWER-LAW SPATIAL DISTRIBUTION FUNCTION WITH FIXED AXIAL RATIO
    # x - Galactocentric Cartesian (kpc) [vector]
    # y - Galactocentric Cartesian (kpc) [vector]
    # z - Galactocentric Cartesian (kpc) [vector]
    # Returns broken power-law density
def bplDf(x,y,z,pars):
    
    # Extract parameters
    q,alpha,beta,rb = pars
      
    # Evaluate density  
    R           = np.sqrt(x**2.+y**2.)
    rq          = np.sqrt(R**2.+(z/q)**2.)
    df          = np.zeros_like(x)
    index       = rq<rb
    df[np.asarray(index)]  = (rq[np.asarray(index)]/rb)**-alpha
    df[np.asarray(~index)] = (rq[np.asarray(~index)]/rb)**-beta
    
    return df
    
## SINGLE POWER-LAW SPATIAL DISTRIBUTION FUNCTION WITH VARIABLE AXIS RATIO
    # x - Galactocentric Cartesian (kpc) [vector]
    # y - Galactocentric Cartesian (kpc) [vector]
    # z - Galactocentric Cartesian (kpc) [vector]
    # Returns broken power-law density
def splDf(x,y,z,pars):
    
    # Extract parameters
    q0,qinf,r0,alpha = pars
      
    # Evaluate density  
    Rproj = np.sqrt(x**2.+y**2.)
    r     = np.sqrt(x**2.+y**2.+z**2.)
    qr    = qinf - (qinf-q0)*np.exp(1-(np.sqrt(r**2.+r0**2.))/r0)
    df    = (Rproj**2. + z**2./qr**2)**(-alpha/2.)
    
    return df
    
## EINASTO SPATIAL DISTRIBUTION FUNCTION
    # x - Galactocentric Cartesian (kpc) [vector]
    # y - Galactocentric Cartesian (kpc) [vector]
    # z - Galactocentric Cartesian (kpc) [vector]
    # Returns Einasto density
def einastoDf(x,y,z,pars):
    
    # Extract parameters
    q,n,rs = pars
    
    # Evaluate density
    R  = np.sqrt(x**2.+y**2.)
    rq = np.sqrt(R**2.+(z/q)**2.)
    df = np.zeros_like(x)
    dn = 3.*n - 1./3. + 0.0079/n                # Graham et al. 2006
    df = np.exp(-dn * ((rq/rs)**(1./n) - 1))
    
    return df

## KROUPA, TOUT, AND GILMORE (1993) INITIAL MASS FUNCTION 
    # mass - mass (solar masses) [scalar or vector]
    # Returns initial mass function
def imf(mass):
    
    imf = np.zeros_like(mass)
    index = ((0.08<=mass) & (mass<0.5))
    imf[index] = 0.035*mass[index]**(-1.3)
  
    index = ((0.5<=mass) & (mass<1.0))
    imf[index] = 0.019*mass[index]**(-2.2)
        
    index = (1.0<=mass)
    imf[index] = 0.019*mass[index]**(-2.7)
    
    return imf