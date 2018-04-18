"""CLASSES FOR CREATING MOCK CATALOGUES GIVEN A GALAXY MODEL
This module calculates the mock catalogues for a galaxy model that is either a 
SF*DF (equatorial coordinates) or an SF*EDF (equatorial coordinates, 
metallicity, alpha, age).
    
Author:
    Payel Das
    
To do:
    Check sizes of input arrays and whether the galaxy model supplied takes the
    right size array.
"""
import numpy as np
import agama

""" CLASS CREATING MOCK CATALOGUE AT ANY SKY POSITION
"""
class AnySkyPos:
    
    def  __init__(self,obsSfdf,minObs,maxObs):
        
        """CLASS CONSTRUCTOR 

        Arguments:        
            obsSfdf - SF times DF or SF times EDF     [func]
            minObs  - minima for real observations   (rad,rad,kpc,km/s,mas,mas or rad,rad,kpc,km/s,mas,mas,dex,dex,Gyr [vector])
            maxObs  - maxima for real observations   (rad,rad,kpc,km/s,mas,mas or rad,rad,kpc,km/s,mas,mas,dex,dex,Gyr [vector])
        
        Returns:
            None.
            
        """
        
        self.obsSfdf = obsSfdf
        self.minObs  = minObs
        self.maxObs  = maxObs
        
        print(obsSfdf(np.vstack((minObs,maxObs))))
        
        # Number of coordinate dimensions
        nvar      = len(minObs)
        self.nvar = np.copy(nvar)
        
        # For scaling transformations
        scale = np.max([np.abs(minObs),np.abs(maxObs)],0)
        print("Scalings:")
        print(scale)
        print(" ")
        self.scale = np.copy(scale)
    
    def transCoords(self,star):
        
        """TRANSFORM OBSERVED COORDINATES 
        
        Arguments:
            star - star's coordinates 
                (ra,dec,s,vlos,muracosdec,mudec, [matrix]) for SF*DF OR
                (ra,dec,s,vlos,muracosdec,mudec,feh,ah,age, [matrix]) for SF*EDF
        
        Returns:
            Transformed coordinates (-, [matrix])
            
        """
        
        star = np.atleast_2d(star)
        
        # Make local copies
        nvar  = np.copy(self.nvar)
        scale = np.copy(self.scale) 
        
        # Transformations (none for ra, dec, and chemistry parameters)
        rap  = star[:,0]
        decp = star[:,1]
        if (nvar==6): # Assume (ra,dec,s,vlos,muracosdec,mudec)
            sp      = np.log(star[:,2])
            vrp     = np.tanh(star[:,3]/scale[3])
            murap   = np.tanh(star[:,4]/scale[4])
            mudecp  = np.tanh(star[:,5]/scale[5])
            starp   = np.column_stack((rap,decp,sp,vrp,murap,mudecp))
        if (nvar==9): # Assume (ra,dec,s,vlos,muracosdec,mudec,feh,ah,age)
            sp      = np.log(star[:,2])
            vrp     = np.tanh(star[:,3]/scale[3])
            murap   = np.tanh(star[:,4]/scale[4])
            mudecp  = np.tanh(star[:,5]/scale[5])
            fehp    = star[:,6]
            ahp     = star[:,7]
            agep    = star[:,8]
            starp   = np.column_stack((rap,decp,sp,vrp,murap,mudecp,fehp,ahp,agep))
         
        return(starp)
        
    def invtransCoords(self,starp):
        
        """INVERSE TRANSFORM OBSERVED COORDINATES 
        
        Arguments:
            starp - Transformed star coordinates (-, [matrix])
        
        Returns:
            Inverse transformed coordinates - star's coordinates
                    (ra,dec,s,vlos,muracosdec,mudec, [vector]) for SF*DF OR
                    (ra,dec,s,vlos,muracosdec,mudec,feh,ah,age, [matrix]) for SF*EDF
            
        """

        starp = np.atleast_2d(starp)
        
        # Make local copies
        nvar  = np.copy(self.nvar)
        scale = np.copy(self.scale)
    
        # Transformations (none for ra and dec)
        ra  = starp[:,0]
        dec = starp[:,1]
        if (nvar==6): # Assume (ra,dec,s,vr,muracosdec,mudec)
            s      = np.exp(starp[:,2])
            vr     = scale[3]*np.arctanh(starp[:,3])
            mura   = scale[4]*np.arctanh(starp[:,4])
            mudec  = scale[5]*np.arctanh(starp[:,5])
            star   = np.column_stack((ra,dec,s,vr,mura,mudec))
        if (nvar==9): # Assume (ra,dec,s,vr,muracosdec,mudec,feh,ah,age)
            s      = np.exp(starp[:,2])
            vr     = scale[3]*np.arctanh(starp[:,3])
            mura   = scale[4]*np.arctanh(starp[:,4])
            mudec  = scale[5]*np.arctanh(starp[:,5])
            feh    = starp[:,6]
            ah     = starp[:,7]
            age    = starp[:,8]
            star   = np.column_stack((ra,dec,s,vr,mura,mudec,feh,ah,age))
            
        return(star)
        
    def jacDet(self,star):
        
        """JACOBIAN DETERMINANT OF TRANSFORMATIONS
        
        Arguments:
            star - star's coordinates 
                (ra,dec,s,vlos,muracosdec,mudec, [vector]) for SF*DF OR
                (ra,dec,s,vlos,muracosdec,mudec,feh,ah,age, [matrix]) for SF*EDF
              
        Returns:
            Jacobian determinant (-,[vector])
            
        """
        
        star = np.atleast_2d(star)
        
        # Make local copies
        nvar  = np.copy(self.nvar)
        scale = np.copy(self.scale)
        
        if (nvar==4):  # Assume (ra,dec,s,feh)
            jacobian   = star[:,2]
        if (nvar==6): # Assume (ra,dec,s,vr,muracosdec,mudec)
            jacobian   = star[:,2]/\
                         ((1.-(np.tanh(star[:,3]/scale[3]))**2.)/scale[3])/\
                         ((1.-(np.tanh(star[:,4]/scale[4]))**2.)/scale[4])/\
                         ((1.-(np.tanh(star[:,5]/scale[5]))**2.)/scale[5])
        if (nvar==7): # Assume (ra,dec,s,vr,muracosdec,mudec,feh)
            jacobian   = star[:,2]/\
                         ((1.-(np.tanh(star[:,3]/scale[3]))**2.)/scale[3])/\
                         ((1.-(np.tanh(star[:,4]/scale[4]))**2.)/scale[4])/\
                         ((1.-(np.tanh(star[:,5]/scale[5]))**2.)/scale[5])
        if (nvar==9): # Assume (ra,dec,s,vr,muracosdec,mudec,feh,ah,age)
            jacobian   = star[:,2]/\
                         ((1.-(np.tanh(star[:,3]/scale[3]))**2.)/scale[3])/\
                         ((1.-(np.tanh(star[:,4]/scale[4]))**2.)/scale[4])/\
                         ((1.-(np.tanh(star[:,5]/scale[5]))**2.)/scale[5])
            
        return(jacobian)
        
    def integrand(self,starp):
        
        """INTEGRAND IN TRANSFORMED COORDINATES
        
        Arguments:
            starp - Transformed star coordinates (-, [matrix])
      
        Returns:
            Integrand (-,[vector])
            
        """
        
        starp = np.atleast_2d(starp)
                
        # Make local copies
        nvar    = np.copy(self.nvar)
                        
        # Inverse transform 
        star    = self.invtransCoords(starp)

        # Jacobian determinant of transformations imposed here
        jacdet  = self.jacDet(star)
        
        # Construct total star vector and multiply further by the Jacobian 
        # determinant for the Equatorial coordinates to Cartesian coordinates
        # transformation
        if (nvar==6):  # Assume (ra,dec,s,vr,muracosdec,mudec)
            jacdet *= star[:,2]**4. * np.cos(star[:,1])
        if (nvar==9):  # Assume (ra,dec,s,vr,muracosdec,mudec,feh,ah,age)
            jacdet *= star[:,2]**4. * np.cos(star[:,1])
        
        # Integrand
        integrand = jacdet*self.obsSfdf(star)
        index     = integrand<0.
        if (np.any(index)):
            print("Negative integrand.")
            print (star[index,:])
            print(integrand[index])
        index     = np.isnan(integrand)
        if (np.any(index)):
            print("NaN integrand.")
            print (star[index,:])
            print(integrand[index])
        index     = np.isinf(index)
        if (np.any(index)):
            print("Infinite integrand.")
            print (star[index,:])
            print(integrand[index])
        return(integrand)
      
    def genSamples(self,nsamp):
        
        """GENERATES SAMPLES IN OBSERVED COORDINATES
        
        Arguments:
            nsamp - number of samples to generate (-, [scalar])
      
        Returns:
            Samples - (ra,dec,s,vlos,muracosdec,mudec, [matrix]) for SF*DF OR
                      (ra,dec,s,vlos,muracosdec,mudec,feh,ah,age, [matrix]) for SF*EDF
            
        """
        
        # Make local copies
        nvar   = np.copy(self.nvar)
        minObs = np.copy(self.minObs)
        maxObs = np.copy(self.maxObs)
        
        print("Generating samples for "+str(nvar)+" dimensions:")
        print(" ")
        
        # Calculate in transformed coordinates
        tobsmin = self.transCoords(np.array([minObs]))
        tobsmax = self.transCoords(np.array([maxObs]))
        print("Transformed minima:")
        print(tobsmin[0])
        print("Transformed maxima:")
        print(tobsmax[0])
        print(" ")
        
        # Calculate samples in transformed coordinates
        samplep,val,err,_ = agama.sampleNdim(self.integrand,nsamp,tobsmin[0],tobsmax[0])

        print("Sampling error:")
        print(err)
            
        # Transform back to observed coordinates  
        samples           = self.invtransCoords(samplep) 
        print("...done.")
        print(" ")
                    
        return(samples)
        
""" CLASS CREATING MOCK CATALOGUE AT GIVEN SKY POSITIONS
"""
class GivenSkyPos:
    
    def  __init__(self,obsSfdf,obs,obsinfofile):
        
        """CLASS CONSTRUCTOR 

        Arguments:        
            obsSfdf     - SF times DF or SF times EDF     [func]
            obs         - observed coordinates (rad,rad,kpc,km/s,mas,mas or rad,rad,kpc,km/s,mas,mas,dex,dex,Gyr [vector])
            obsinfofile - temporary file for storing observation number [str]
        Returns:
            None.
            
        """

        self.obsSfdf     = obsSfdf
        self.obs         = np.copy(obs)
        self.obsinfofile = obsinfofile
        
        # Number of dimensions being sampled
        nobs,nvar = np.shape(obs)
        nvar     -= 2
        self.nvar = np.copy(nvar)
        
        # For scaling tanh transformations
        scale      = np.max(np.abs(self.obs[:,2]),0)
        self.scale = scale
        print("Scalings:")
        print(scale)
        print(" ")
        
        return
        
    def transCoords(self,star):
        
        """TRANSFORM OBSERVED COORDINATES 
        
        Arguments:
            star - star's coordinates 
                (s,vlos,muracosdec,mudec, [matrix]) for SF*DF OR
                (s,vlos,muracosdec,mudec,feh,ah,age, [matrix]) for SF*EDF
        
        Returns:
            Transformed coordinates (-, [matrix])
            
        """
        
        # Make local copies
        nvar  = self.nvar
        scale = self.scale        
        if (nvar==4): # Assume (s,vr,muracosdec,mudec)
            sp      = np.log(star[:,0])
            vrp     = np.tanh(star[:,1]/scale[1])
            murap   = np.tanh(star[:,2]/scale[2])
            mudecp  = np.tanh(star[:,3]/scale[3])
            starp   = np.column_stack((sp,vrp,murap,mudecp))
        if (nvar==7): # Assume (s,vr,muracosdec,mudec,feh,ah,age)
            sp      = np.log(star[:,0])
            vrp     = np.tanh(star[:,1]/scale[1])
            murap   = np.tanh(star[:,2]/scale[2])
            mudecp  = np.tanh(star[:,3]/scale[3])
            fehp    = star[:,4]
            ahp     = star[:,5]
            agep    = star[:,6]
            starp   = np.column_stack((sp,vrp,murap,mudecp,fehp,ahp,agep))
        
        return(starp)
        
    def invtransCoords(self,starp):
        
        """INVERSE TRANSFORM OBSERVED COORDINATES 
        
        Arguments:
            starp - Transformed star coordinates (-, [matrix])
        
        Returns:
            Inverse transformed coordinates - star's coordinates
                    (s,vlos,muracosdec,mudec, [vector]) for SF*DF OR
                    (s,vlos,muracosdec,mudec,feh,ah,age, [matrix]) for SF*EDF
            
        """
        
        # Make local copies
        nvar  = self.nvar
        scale = self.scale
    
        if (nvar==4): # Assume (s,vr,muracosdec,mudec)
            s      = np.exp(starp[:,0])
            vr     = scale[1]*np.arctanh(starp[:,1])
            mura   = scale[2]*np.arctanh(starp[:,2])
            mudec  = scale[3]*np.arctanh(starp[:,3])
            star   = np.column_stack((s,vr,mura,mudec))
        if (nvar==7): # Assume (s,vr,muracosdec,mudec,feh,ah,age)
            s      = np.exp(starp[0])
            vr     = scale[1]*np.arctanh(starp[:,1])
            mura   = scale[2]*np.arctanh(starp[:,2])
            mudec  = scale[3]*np.arctanh(starp[:,3])
            feh    = starp[:,4]
            ah     = starp[:,5]
            age    = starp[:,6]
            star   = np.column_stack((s,vr,mura,mudec,feh,ah,age))
         
        return(star)
        
    def jacDet(self,star):
        
        """JACOBIAN DETERMINANT OF TRANSFORMATIONS
        
        Arguments:
            star - star's coordinates 
                (s,vlos,muracosdec,mudec, [vector]) for SF*DF OR
                (s,vlos,muracosdec,mudec,feh,ah,age, [matrix]) for SF*EDF
              
        Returns:
            Jacobian determinant (-,[vector])
            
        """

        # Make local copies
        scale = self.scale
        
        jacobian   = star[:,0]/\
                        ((1.-(np.tanh(star[:,1]/scale[1]))**2.)/scale[1])/\
                        ((1.-(np.tanh(star[:,2]/scale[2]))**2.)/scale[2])/\
                        ((1.-(np.tanh(star[:,3]/scale[3]))**2.)/scale[3])
            
        return(np.abs(jacobian))
        
    def integrand(self,starp):
        
        """INTEGRAND IN TRANSFORMED COORDINATES
        
        Arguments:
            starp - Transformed star coordinates (-, [matrix])
      
        Returns:
            Integrand (-,[vector])
            
        """
        
        # Make local copies
        nvar  = self.nvar
                
        # Read in observations
        obsinfo  = np.loadtxt(self.obsinfofile)
        
        # Inverse transformations in equatorial coordinates
        star = self.invtransCoords(starp)
        
         # Jacobian determinant of transformations
        jacdet  = self.jacDet(star)
        
        # Construct total star vector and multiply further by the determinant
        if (nvar==4):  # Assume (s,vr,mura,mudec)
            jacdet *= star[:,0]**4.
            wefeh   = np.column_stack((star[:,0]*0.+obsinfo[0,0]/180.*np.pi,star[:,0]*0.+obsinfo[1,0]/180.*np.pi,star[:,0],star[:,1],star[:,2],star[:,3]))
        if (nvar==7):  # Assume (s,vr,mura,mudec,feh,ah,age)
            jacdet *= star[:,0]**4.
            wefeh   = np.column_stack((star[:,0]*0.+obsinfo[0,0]/180.*np.pi,star[:,0]*0.+obsinfo[1,0]/180.*np.pi,star[:,0],star[:,1],star[:,2],star[:,3],star[:,4],star[:,5],star[:,6]))
    
        # Integrand
        return(jacdet*self.sfdf(wefeh))
      
    def genSamples(self):
        
        """GENERATES SAMPLES AT OBSERVED COORDINATES
        
        Arguments:
            nsamp - number of samples to generate (-, [scalar])
      
        Returns:
            Samples - (ra,dec,s,vlos,muracosdec,mudec, [matrix]) for SF*DF OR
                      (ra,dec,s,vlos,muracosdec,mudec,feh,ah,age, [matrix]) for SF*EDF
            
        """
        
        # Make local copies
        nvar  = np.copy(self.nvar)
        obs   = np.copy(self.Obs)
        
        print("Generating samples for "+str(nvar)+" dimensions:")
        print(" ")
        
        # Calculate minima and maxima of observations
        obsmin     = np.array([np.min(obs[:,2:],0)])
        obsmax     = np.array([np.max(obs[:,2:],0)])
        print("Observed minima:")
        print(obsmin)
        print("Observed maxima:")
        print(obsmax)
        print(" ")
        
        # Calculate in transformed coordinates, assuming first two coordinates are sky positions
        tobsmin = self.transCoords(obsmin)
        tobsmax = self.transCoords(obsmax)
        print("Transformed minima:")
        print(tobsmin[0])
        print("Transformed maxima:")
        print(tobsmax[0])
        print(" ")
        
        # Calculate samples in transformed coordinates at each coordinate of 
        # the observed star, and transform back
        nsamp,tmp = np.shape(obs)
        print("Generating "+str(nsamp)+" samples:")
        samplesall = np.zeros_like(obs)
        for jsamp in range(nsamp):
            print("Sample "+str(jsamp+1)+":")
            np.savetxt(self.obsinfofile,np.column_stack((obs[jsamp,:])))
            samplep,val,err,_ = agama.sampleNdim(self.integrand,2,tobsmin[0],tobsmax[0])
            
            # Transform back to sensible coordinates  
            samples               = self.invtransCoords(samplep)      
            samplesall[jsamp,0:2] = obs[jsamp,0:2]
            if (nvar==2):
                samplesall[jsamp,2]   = samples[1,0]
                samplesall[jsamp,3]   = samples[1,1]
            if (nvar==4):
                samplesall[jsamp,2]   = samples[1,0]
                samplesall[jsamp,3]   = samples[1,1]
                samplesall[jsamp,4]   = samples[1,2]
                samplesall[jsamp,5]   = samples[1,3]
            if (nvar==5):
                samplesall[jsamp,2]   = samples[1,0]
                samplesall[jsamp,3]   = samples[1,1]
                samplesall[jsamp,4]   = samples[1,2]
                samplesall[jsamp,5]   = samples[1,3]
                samplesall[jsamp,6]   = samples[1,4]
            print(samplesall[jsamp,:])
            print(" ")
            
            with open("mocksamples.txt",'a') as filehandle:
                    np.savetxt(filehandle,
                               np.column_stack([obs[jsamp,0],obs[jsamp,1],samples[1,0],samples[1,1],samples[1,2],samples[1,3]]),fmt='%12.5f')  
            
        return(samplesall)