# FitGalaxyModels
A suite of routines fitting distribution functions to data for the Milky Way Galaxy. DOCSTRINGS give more detailed informaiton.

## AstroMethods.py
This module collates useful methods for astronomy.

## CalcEDFMoments.py 
This module calculates the density moment of an extended distribution function,
assuming a cylindrically symmetric system.

## CoordTrans.py
This module contains methods that carry out coordinate transformations for left-handed coordinate systems.
	
## CreateMockCat.py
This module calculates the mock catalogues for a galaxy model that is either a 
SF*DF (equatorial coordinates) or an SF*EDF (equatorial coordinates, 
metallicity, alpha, age).
	
## DistFunc.py 
This module collates a number of probability distribution functions that could form an extended distribution function (EDF).

## MilkyWayEDF.py 
This module calculates the EDF probabilities given the actions (Jr, Jz, Lz), 
[Fe/H], [a/Fe], and age.\\
Example:\\
    Initialize as mwedf = MilkyWayEDF()\\
    Returns EDF prob = mwedf(coords)\\
