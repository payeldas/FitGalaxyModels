# FitGalaxyModels
A suite of routines fitting distribution functions to data for the Milky Way Galaxy. DOCSTRINGS give more detailed informaiton.

## AstroMethods.py
""" ASTRO METHODS
This module collates useful methods for astronomy.
To do:
	Implement whatever new astronomy-related methods that come my way.
"""

## CalcEDFMoments.py 
"""  MOMENTS OF AN EXTENDED DISTRIBUTION FUNCTION
This module calculates the density moment of an extended distribution function,
assuming a cylindrically symmetric system.
To do:
    Calculate velocity and chemistry moments.
    Calculate density, velocity, and chemistry moments by age.
"""

## CoordTrans.py
""" COORDINATE TRANSFORMATIONS 
This module contains methods that carry out coordinate transformations for left-handed coordinate systems.
To do:
    Nothing I can think of at present.
"""
	
## CreateMockCat.py
"""CLASSES FOR CREATING MOCK CATALOGUES GIVEN A GALAXY MODEL
This module calculates the mock catalogues for a galaxy model that is either a 
SF*DF (equatorial coordinates) or an SF*EDF (equatorial coordinates, 
metallicity, alpha, age).
To do:
    Check sizes of input arrays and whether the galaxy model supplied takes the
    right size array.
"""
	
## DistFunc.py 
""" PROBABILITY DISTRIBUTION FUNCTIONS 
This module collates a number of probability distribution functions that could form an extended distribution function (EDF).
To do:
    Implement more probability distribution functions as they come my way.
"""

## MilkyWayEDF.py 
"""
MILKY WAY EDF 
This module calculates the EDF probabilities given the actions (Jr, Jz, Lz), 
[Fe/H], [a/Fe], and age.
Example:
    Initialize as mwedf = MilkyWayEDF()
    Returns EDF prob = mwedf(coords)
To do:
    Nothing (I think).
"""
