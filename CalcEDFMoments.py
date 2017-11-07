"""  MOMENTS OF AN EXTENDED DISTRIBUTION FUNCTION
This module calculates the density moment of an extended distribution function,
assuming a cylindrically symmetric system.
    
Author:
    Payel Das
    
To do:
    Calculate velocity and chemistry moments.
    Calculate density, velocity, and chemistry moments by age.
"""
import numpy as np
import CoordTrans as ct
import vegas

class EdfMoments:
    
    def  __init__(self,edf,af,vscale,maxeval,niter):
        
        """CLASS CONSTRUCTOR 

        Arguments:        
            edf     - Extended Distribution Function (EDF) object [object]
            af      - action finder object [object]
            vscale  - velocity scale for transformation of velocity coordinates (km/s, [scalar])
            maxeval - maximum number of evaluations for Vegas integrals (-, [integer scalar])
            niter   - number of iterations for Vegas integrals (-, [integer scalar])

        Returns:
            None.
        """
        
        self.edf     = edf
        self.af      = af
        self.vscale  = np.copy(vscale)
        self.maxeval = np.copy(maxeval)
        self.niter   = np.copy(niter)
        
        return

    def moments(self,R,z,intmin,intmax):
        
        """ EDF MOMENTS INTEGRATING OVER AGE

        Arguments:    
            R             - R coordinates
            z             - z coordinate
            intmin/intmax - integration limits
        
        Returns:
            Density moments at R,z.
            
        """
        
        vscale  = np.copy(self.vscale)
        maxeval = np.copy(self.maxeval)
        niter   = np.copy(self.niter)
        
        # Integrand with scaled velocities
        @vegas.batchintegrand
        def integrand(scaledstar): 
        
            # Velocities
            vx     = vscale*np.arctanh(scaledstar[:,0])
            vy     = vscale*np.arctanh(scaledstar[:,1])
            vz     = vscale*np.arctanh(scaledstar[:,2])
        
            # Chemistry
            feh    = scaledstar[:,3]
            ah     = scaledstar[:,4]
            age    = scaledstar[:,5]
            
            # Transformation Jacobian
            jac    = vscale**3./(1.-(np.tanh(vx/vscale))**2.)/\
                                (1.-(np.tanh(vy/vscale))**2.)/\
                                (1.-(np.tanh(vz/vscale))**2.)
                                         
            # npts
            npts = len(vx)

            # Get other coordinates
            Rvec    = np.tile(R,npts)
            phivec  = np.tile(0.,npts)
            zvec    = np.tile(z,npts)  
            xp      = np.column_stack((Rvec,phivec,zvec))
            xc      = ct.PolarToCartesian(xp)
            wc      = np.column_stack((xc[:,0],xc[:,1],xc[:,2],vx,vy,vz))
            acts    = self.af(wc)
            xi      = np.column_stack((feh,ah,age))
      
            # Calculate probability
            prob                 = self.edf(acts,xi)
            prob[np.isnan(prob)] = 0.
        
            return(jac*prob)
   
        # Create integration object
        integ = vegas.Integrator([[intmin[0],intmax[0]],
                                  [intmin[1],intmax[1]],
                                  [intmin[2],intmax[2]],
                                  [intmin[3],intmax[3]],
                                  [intmin[4],intmax[4]],
                                  [intmin[5],intmax[5]]])
                                   
        # Calculate moments   
        print("At polar coordinates [R,z] = ["+str(R)+","+str(z)+"]:")
                
        # Train integration object
        integ(integrand,nitn=5,neval=1000)      
                
        # Calculate integration
        result = integ(integrand,nitn=niter,neval=maxeval)
        val    = result.mean
        err    = result.sdev
           
        print("     Density moment and error:")
        print("     "+str(val)+", "+str(err))               
                    
        return(val)