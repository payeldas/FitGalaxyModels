""" ASTRO METHODS
This module collates useful methods for astronomy.

Author:
	Payel Das

To do:
	Implement whatever new astronomy-related methods that come my way.
"""
import numpy as np

def appMag(absmag,s):
	""" APPARENT MAGNITUDE OF STAR GIVEN DISTANCE AND ABSOLUTE MAGNITUDE

	Arguments:
		absmag - absolute magnitudes (mag) [vector]
		s      - distances (kpc)     [vector]

	Returns:
		Apparent magntiude (mag) [vector]
	"""
	
	return (absmag + 5*np.log10(s) + 10)
 
def absMag(appmag,s):
	""" ABSOLUTE MAGNITUDE OF STAR GIVEN DISTANCE AND ABSOLUTE MAGNITUDE

	Arguments:
		appmag - apparent magnitudes (mag) [vector]
		s      - distances (kpc)     [vector]

	Returns:
		Absolute magntiude (mag) [vector]
	"""
	
	return (appmag - 5*np.log10(s) - 10)
 
def luminosity(absmag,absmag_sun):
    """ LUMINOSITY GIVE ABSOLUTE MAGNITUDE AND ABSOLUTE MAGNITUDE OF SUN

    Arguments:
        absmag     - absolute magnitude (mag) [vector]
        absmag_sun - absolute magnitude of sun (mag) [scalar]

    Returns:
        Luminosity (Lsun) [vector]
    """     

    return(10**(-0.4*(absmag-absmag_sun)))

def dist(appmag,absmag):
	"""DISTANCE GIVEN APPARENT AND ABSOLUTE MAGNITUDES

	Arguments:
		appmag - apparent magnitudes (mag) [vector]
		absmag - absolute magnitudes (mag) [vector]

	Returns:
		distance (kpc) [vector]
	"""
	
	return (10.**((appmag-absmag-10.)/5.))