"""
MILKY WAY EDF 

This module calculates the EDF probabilities given the actions (Jr, Jz, Lz), 
[Fe/H], [a/Fe], and age.

Example:
    Initialize as mwedf = MilkyWayEDF()
    Returns EDF prob = mwedf(coords)
    
Author:
    Payel Das
    
To do:
    Nothing (I think).

"""
import numpy as np
import vegas
from scipy.integrate import quad

class MilkyWayEDF:
    
    def  __init__(self,gp,sfr_par,discchem_par,halochem_par,thickdiscedf_par,
                  thindiscedf_par,stellarhaloedf_par,Fhalo,solpos,maxeval,
                  niter,norm=None):
                      
        """ CLASS CONSTRUCTOR

        Arguments:
            gp                 - Galactic potential instance [object]
            sfr_par            - star formation rate parameters [dict]
                tauF - thin disc star formation rate decay constant (Gyr, [scalar])
                taus - early star formation growth timescale (Gyr, [scalar])
                taum - age of the Galaxy (Gyr, [scalar]) 
            discchem_par - abundance parameters [dict]
                LzF    - z-component of angular momentum at solar radius (km/s kpc, [scalar])
                gZm    - metallicity at birth of Galaxy (dex, [scalar])
                gZLz   - Lz metallicity gradient at zero eccentricity (dex/(km/s kpc), [scalar])
                gZJz   - Jz metallicity gradient (dex/(km/s kpc), [scalar])
                aZ     - dispersion of metallicities at each Lz and Jz (-, [scalar])
                gam    - alpha at birth of Galaxy (dex, [scalar])
                gaLz   - Lz alpha gradient at zero eccentricity (dex/(km/s kpc), [scalar])
                gaJz   - Jz alpha gradient (dex/(km/s kpc), [scalar])
                aa     - dispersion of alpha at each Lz and Jz (-, [scalar])
            halochem_par       - halo chemical evolution parameters [dict]
                bZ       - halo metallicity at scale action (dex, [scalar])
                cZ       - dependence of halo metallicity on total action (dex/(km/s kpc), [scalar])
                sigZ     - dispersion in halo metallicities at each total action (dex, [scalar])
                balpha   - halo alpha abundance at scale action (dex, [scalar])
                calpha   - dependence of halo alpha abundance on total action (dex/(km/s kpc), [scalar])
                sigalpha - dispersion in halo alpha abundance at each total action (dex, [scalar])
                btau     - halo age at scale action (Gyr, [scalar])
                ctau     - dependence of halo ago on total action (Gyr/(km/s kpc), [scalar])
                sigtau   - dispersion in halo ages at each total action (Gyr, [scalar])   
                J0       - break action splitting inner and outer halo (km/s kpc, [scalar])
            thickdiscedf_par    - thick disc DF parameters    [dict]
                Lz0     - scale angular momentum determining the suppression of retrograde orbits (kpc km/s, [scalar])       
                Rsmin   - scale radius of oldest stars
                Rsmax   - scale radius of youngest stars
                sigmar0tau0 - radial velocity dispersion at solar radius (km/s, [scalar]) 
                sigmaz0tau0 - vertical velocity dispersion at solar radius (km/s, [scalar]) 
                zeta    - scale length of radial velocity dispersion as ratio of scale radius (-, [scalar])
                Rsigmar - R velocity dispersion scale length of the youngest stars at the Galactic center (kpc, [scalar])
                Rsigmaz - z velocity dispersion scale length of the youngest stars at the Galactic center (kpc, [scalar])                
                tausig  - parameter controlling velocity dispersions of stars born today (Gyr, [scalar])
            thindiscedf_par     - thin disc DF parameters [dict]
                Lz0     - scale angular momentum determining the suppression of retrograde orbits (kpc km/s, [scalar])        
                Rsmin   - scale radius of oldest stars
                Rsmax   - scale radius of youngest stars
                sigmar0tau0 - radial velocity dispersion at solar radius (km/s, [scalar]) 
                sigmaz0tau0 - vertical velocity dispersion at solar radius (km/s, [scalar]) 
                Rsigmar - R velocity dispersion scale length of the youngest stars at the Galactic center (kpc, [scalar])
                Rsigmaz - z velocity dispersion scale length of the youngest stars at the Galactic center (kpc, [scalar])
                betar   - growth of radial velocity dispersions with age
                betaz   - growth of vertical velocity dispersions with age
                tausig  - parameter controlling velocity dispersions of stars born today
            stellarhalodf_par - stellar halo DF parameters [dict] STOPP
                Jcore	   - action defining central core (kpc km/s, [scalar])
                alpha	   - minimum inner halo slope (-, [scalar])
                beta	   - minimum outer halo slope (-, [scalar])
                ar	   - inner halo weight on Jr (-, [scalar])
                aphi	   - inner halo weight on Lz (-, [scalar])
                az	   - inner halo weight on Jz (-, [scalar])
                br	   - outer halo weight on Jr (-, [scalar])
                bphi	   - outer halo weight on Lz (-, [scalar])
                bz	   - outer halo weight on Jz (-, [scalar])
                Jcutoff - halo action cut off (kpc km/s, [scalar])
                ncutoff - fall-off of cut off (-, [scalar])
            Fthick  - weight on thick disc in DF (-, [scalar])
            solpos  - Solar position and velocity (kpc,kpc,km/s,km/s,km/s, [vector])
            maxeval - maximum number of evaluations for Vegas normalization integrals [scalar]
            niter   - number of iterations for Vegas normalization integrals [integer scalar]
            norm    - optional array of normalizations [vector]
        
        Returns:
            Nothing
        """        
        
        # Share objects and variables
        self.gp                 = gp
        self.sfr_par            = sfr_par
        self.discchem_par       = discchem_par
        self.halochem_par       = halochem_par
        self.thickdiscedf_par   = thickdiscedf_par
        self.thindiscedf_par    = thindiscedf_par
        self.stellarhaloedf_par = stellarhaloedf_par   
        self.Fhalo              = Fhalo
        self.solpos             = np.copy(solpos)
        self.maxeval            = maxeval
        self.niter              = niter
       
        # Thick disc, thin disc, and stellar halo EDF normalizations
        if (norm is not None):
            discedfnorm,stellarhaloedfnorm=norm
        else:
            Jscale             = 3000.
            discedfnorm        = self.CalcDiscEDFNorm(Jscale)
            Jscale             = 10000.
            stellarhaloedfnorm = self.CalcStellarHaloEDFNorm(Jscale)

        self.discedfnorm         = np.copy(discedfnorm)
        self.stellarhaloedfnorm  = np.copy(stellarhaloedfnorm)
        
        return
        
    def __call__(self,acts,xi):
          
        """ EDF PROBABILITY

        Arguments:
            acts   - actions [Jr, Jz, Lz], (km/s kpc, km/s kpc, km/s kpc [matrix])
            xi     - chemistry parameters [[Fe/H], [a/Fe], age], (dex, dex, Gyr, [vector])
    
        Returns:
            EDF probability [vector]
        """
    
        # Complete EDF probability
        prob    = self.CompleteEDF(acts,xi)
        
        return(prob)
        
    def CalcFreq(self,Lz):
            
        """ CALCULATE FREQUENCIES OF CIRCULAR ORBIT WITH ANGULAR MOMENTUM Lz GIVEN POTENTIAL
            
        Arguments:
            Lz - z-component of angular momentum (km/s kpc, [vector])
                
        Returns:
            Kappa, nu, and omega frequencies (/s, [matrix])
        """
        
        Lz = np.atleast_1d(Lz)
        
        Rc           = self.gp.Rcirc(L=Lz)
        xc           = np.column_stack((Rc,Rc*0.,Rc*0.))
        force, deriv = self.gp.forceDeriv(xc) # Returns force and force derivatives at (x, y, z)
        kappa        = np.sqrt(-deriv[:,0] - 3.*force[:,0]/xc[:,0])
        nu           = np.sqrt(-deriv[:,2])
        omega        = np.sqrt(-force[:,0]/xc[:,0])
        
        return(kappa,nu,omega)
       
    def CalcStellarAbundance(self,Lz,Jz,par):
        
        """ STELLAR ABUNDANCE
        
        Arguments:
            Lz  - z-component of angular momentum (km/s kpc, [vector])
            Jz  - vertical action (km/s kpc, [vector])
            par - stellar elemental abundance parameters [dict] 
                LzF   - z-component of angular momentum at solar radius (km/s kpc, [scalar])
                gm    - abundance at birth of Galaxy (dex, [scalar])
                gLz   - Lz abundance gradient at zero eccentricity (dex/(km/s kpc), [scalar])
                gJz   - Jz abundance gradient (dex/(km/s kpc), [scalar])      
        Returns:
            Stellar abundance (dex, [vector])
        """
        
        # Stellar elemental abundance-Lz relation for stars born now
        g0 = par["gm"] + par["gLz"]*np.log(np.abs(Lz)/par["LzF"])
        
        # Mean stellar elemental abundance at each Lz
        g  = g0 + par["gJz"]*np.log(np.abs(Jz)/par["LzF"])
   
        return(g)
                
    ## CALCULATE METALLICITY, ALPHA, OR AGE GIVEN THE TOTAL ACTION    
    # Jt   - total actions
    # par  - dictionary of chemical evolution parameters 
        # b    - chemistry observable value at scale action
        # c    - dependence of chemistry observable on total action
        # J0   - halo scale action
    # Returns chemistry observable
    def CalcHaloChemistry(self,Jt,par):
        
        """HALO CHEMISTRY (METALLICITY, ALPHA, OR AGE GIVEN THE TOTAL ACTION)
        
        Arguments:
            Jt  - total action (km/s kpc, [vector])
            par - halo chemical evolution parameters [dict]
                bZ       - halo metallicity at scale action (dex, [scalar])
                cZ       - dependence of halo metallicity on total action (dex/(km/s kpc), [scalar])
                sigZ     - dispersion in halo metallicities at each total action (dex, [scalar])
                balpha   - halo alpha abundance at scale action (dex, [scalar])
                calpha   - dependence of halo alpha abundance on total action (dex/(km/s kpc), [scalar])
                sigalpha - dispersion in halo alpha abundance at each total action (dex, [scalar])
                btau     - halo age at scale action (Gyr, [scalar])
                ctau     - dependence of halo ago on total action (Gyr/(km/s kpc), [scalar])
                sigtau   - dispersion in halo ages at each total action (Gyr, [scalar])   
                J0       - break action splitting inner and outer halo (km/s kpc, [scalar])

        Returns:
            Metallicity, alpha, or age (-,[vector])
            
        """
    
        return(par["b"] + par["c"]*np.log(Jt/par["J0"]))
        
    def StarFormationRate(self,tau,par):
            
        """ STAR FORMATION RATE 

        Arguments:            
            tau - age (Gyr, [vector])
            par - star formation rate parameters [dict]
                tauF - thin disc star formation rate decay constant (Gyr, [scalar])
                taus - early star formation growth timescale (Gyr, [scalar])
                taum - age of the Galaxy (Gyr, [scalar]) 
             
        Returns:
            Star formation rate (1/Gyr, [vector])
        """
        
        tau = np.atleast_1d(tau)
        sfr = np.zeros_like(tau)
        
        # For ages close to the age of the Galaxy, let star formation rate be zero
        index      = (par["taum"]-tau) > 0.001
        sfr[index] = np.exp(tau[index]/par["tauF"] - par["taus"]/(par["taum"]-tau[index]))
        
        return(sfr)
        
    def CalcStarFormationRateNorm(self,par):
        
        """ NORMALIZATION OF STAR FORMATION RATE

        Arguments:        
            par - star formation rate parameters [dict]
                tauF - thin disc star formation rate decay constant (Gyr, [scalar])
                taus - early star formation growth timescale (Gyr, [scalar])
                taum - age of the Galaxy (Gyr, [scalar]) 
                
        Returns:
            Normalization of star formation rate model.
        """
        
        val,err = quad(self.StarFormationRate,0.,par["taum"],par)
        
        return(1./val)
        
    def DiscChemicalEvolutionModel(self,acts,xi,par):
        
        """ DISC CHEMICAL EVOLUTION MODEL

        Arguments:            
            acts - actions Jr, Jz, Lz (km/s kpc, km/s kpc, km/s kpc, [matrix])
            xi  - metallicity, alpha-abundance, age (dex,dex,Gyr, [matrix])
            par - disc chemical evolution parameters [dict]
                gZLz    - stellar radial metallicity gradient at solar radius (dex/kpc, [scalar])
                gZm    - stellar metallicity at birth of Galaxy (dex, [scalar])
                tauZF  - stellar metallicity enrichment timescale (Gyr, [scalar])
                aZ     - dependence of dispersion of ISM metallicities at each Lz on age
                LzF    - z-component of angular momentum at solar radius
                gaLz    - stellar radial alpha abundance gradient at solar radius (dex/kpc, [scalar])
                gam    - stellar alpha abundance at birth of Galaxy (dex, [scalar])
                tauaF  - stellar alpha abundance enrichment timescale (Gyr, [scalar])
                aalpha - dependence of dispersion of ISM alpha abundances at each Lz on age
    
        Returns:
        Product of probabilities of [Fe/H] and [a/Fe] (-,[vector]).
        
        """
        
        acts = np.atleast_2d(acts)
        xi   = np.atleast_2d(xi)        
        
        # Actions
        Jz  = acts[:,1]
        Lz  = acts[:,2]
        
        # Chemistry variables
        feh = xi[:,0]
        afe = xi[:,1]
        
        # Stellar metallicity probability
        
        # Create dictionary of stellar metallicity parameters
        stellar_metallicity_par = dict(LzF  = par["LzF"],
                                       gm   = par["gZm"],
                                       gLz  = par["gZLz"],
                                       gJz  = par["gZJz"])
        gZ   = self.CalcStellarAbundance(Lz,Jz,stellar_metallicity_par)
        fZ   = np.exp(-(gZ-feh)**2./(2.*par["aZ"]**2.))/np.sqrt(2.*np.pi*par["aZ"]**2.)
        
        # Stellar alpha probability
        
        # Create dictionary of stellar alpha parameters
        stellar_alpha_par = dict(LzF  = par["LzF"],
                                 gm   = par["gam"],
                                 gLz  = par["gaLz"],
                                 gJz  = par["gaJz"])
                    
        ga   = self.CalcStellarAbundance(Lz,Jz,stellar_alpha_par)       
        fa   = np.exp(-(ga-afe)**2./(2.*par["aalpha"]**2.))/np.sqrt(2.*np.pi*par["aalpha"]**2.)
        
        return(fZ*fa)
        
    def HaloChemicalEvolutionModel(self,acts,xi,par):
        
        """ STELLAR HALO CHEMICAL EVOLUTION MODEL
        
        Arguments:        
            acts - actions Jr, Jz, Lz (km/s kpc, km/s kpc, km/s kpc, [matrix])
            xi   - metallicity, alpha-abundance, age (dex, dex, Gyr, [matrix])
            par  - halo chemical evolution parameters [dict]
                bZ       - halo metallicity at scale action (dex, [scalar])
                cZ       - dependence of halo metallicity on total action (dex/(km/s kpc), [scalar])
                sigZ     - dispersion in halo metallicities at each total action (dex, [scalar])
                balpha   - halo alpha abundance at scale action (dex, [scalar])
                calpha   - dependence of halo alpha abundance on total action (dex/(km/s kpc), [scalar])
                sigalpha - dispersion in halo alpha abundance at each total action (dex, [scalar])
                btau     - halo age at scale action (Gyr, [scalar])
                ctau     - dependence of halo ago on total action (Gyr/(km/s kpc), [scalar])
                sigtau   - dispersion in halo ages at each total action (Gyr, [scalar])   
                J0       - break action splitting inner and outer halo (km/s kpc, [scalar])
        Returns:
            Product of probabilities of [Fe/H], [a/Fe], and age.
        """

        acts = np.atleast_2d(acts)
        xi   = np.atleast_2d(xi)        
        
        # Actions
        Jr  = acts[:,0]
        Jz  = acts[:,1]
        Lz  = acts[:,2]
        
        # Chemistry variables
        feh = xi[:,0]
        afe = xi[:,1]
        tau = xi[:,2] 
        
        # Total action
        Jt  = np.sqrt(Jr**2. + Jz**2. + Lz**2.)
        
        # Metallicity
        halo_metallicity_par = dict(b  = par["bZ"],
                                    c  = par["cZ"],
                                    J0 = par["J0"])
        gZ = self.CalcHaloChemistry(Jt,halo_metallicity_par)
        
        # Metallicity probability
        fZ = np.exp(-(gZ-feh)**2./(2.*par["sigZ"]**2.))/np.sqrt(2.*np.pi*par["sigZ"]**2.)
        
        # Alpha abundance
        halo_alpha_par = dict(b  = par["balpha"],
                              c  = par["calpha"],
                              J0 = par["J0"])
        ga = self.CalcHaloChemistry(Jt,halo_alpha_par)
               
        # Alpha probability
        fa = np.exp(-(ga-afe)**2./(2.*par["sigalpha"]**2.))/np.sqrt(2.*np.pi*par["sigalpha"]**2.)
        
        # Age
        halo_age_par = dict(b  = par["btau"],
                            c  = par["ctau"],
                            J0 = par["J0"])
        gt = self.CalcHaloChemistry(Jt,halo_age_par)
               
        # Age probability
        ft = np.exp(-(gt-tau)**2./(2.*par["sigtau"]**2.))/np.sqrt(2.*np.pi*par["sigtau"]**2.)
        
        return(fZ*fa*ft)
          
    def PseudoIsoDF(self,acts,tau,par):
         
        """ PSEUDO-ISOTHERMAL DF OF A COEVAL POPULATION AT SOME ACTIONS
         
        Arguments:
            acts - actions Jr, Jz, Lz (km/s kpc, km/s kpc, km/s kpc, [matrix])
            tau  - age of coeval population (Gyr, [vector])
            par  - pseudo-isothermal parameters  [dict]
                Lz0     - scale angular momentum determining the suppression of retrograde orbits (kpc km/s, [scalar])       
                Rsmin   - scale radius of oldest stars
                Rsmax   - scale radius of youngest stars
                sigmar0tau0 - radial velocity dispersion at solar radius (km/s, [scalar]) 
                sigmaz0tau0 - vertical velocity dispersion at solar radius (km/s, [scalar]) 
                Rsigmar - R velocity dispersion scale length of the youngest stars at the Galactic center (kpc, [scalar])
                Rsigmaz - z velocity dispersion scale length of the youngest stars at the Galactic center (kpc, [scalar])                
                betar   - growth of radial velocity dispersions with age (-, [scalar])
                betaz   - growth of vertical velocity dispersions with age (-, [scalar])
                taum    - age of Galaxy (Gyr, [scalar])
                tausig  - parameter controlling velocity dispersion of stars born today (Gyr, [scalar])
                tauT    - age of thin-thick disc separation (Gyr, [scalar])                                                
        Returns:    
        Pseudo-isothermal df probabilities.
        """
        
        acts = np.atleast_2d(acts)
        tau  = np.atleast_1d(tau)        
     
        # Actions
        Jr  = acts[:,0]
        Jz  = acts[:,1]
        Lz  = acts[:,2]
        
        # Frequencies
        kappa,nu,omega = self.CalcFreq(Lz)
        
        # Radius of circular orbit
        Rc             = self.gp.Rcirc(L=Lz)
        
        # Surface brightness
        nfallmax = 2.
        nfallmin = 1.
        nfall    = nfallmax - (nfallmax-nfallmin)*tau/par["taum"]       
        Rs       = par["Rsmin"]+(par["Rsmax"]-par["Rsmin"])*tau/par["taum"]  # Radial migration
        Rsprime  = (nfall/(nfall-1))**(1/nfall)*Rs
        fsb      = omega/kappa**2. * (nfall/Rsprime) * \
                                     (Rc/Rsprime)**(nfall-1) * \
                                     np.exp(-Rc**nfall/Rsprime**nfall)

        # Velocity dispersion (heating with age)
        sigmar0  = par["sigmar0tau0"]*((tau+par["tausig"])/(par["tauT"]+par["tausig"]))**par["betar"]
        sigmaz0  = par["sigmaz0tau0"]*((tau+par["tausig"])/(par["tauT"]+par["tausig"]))**par["betaz"]
        sigmar   = sigmar0*np.exp((self.solpos[0]-Rc)/par["Rsigmar"])
        sigmaz   = sigmaz0*np.exp((self.solpos[0]-Rc)/par["Rsigmaz"])
        
        # EDF probability        
        frot    = 1. + np.tanh(Lz/par["Lz0"])
        fsigmar = kappa/(sigmar**2)*np.exp(-kappa*Jr/sigmar**2.)
        fsigmaz = nu/(sigmaz**2)*np.exp(-nu*Jz/sigmaz**2.)
        prob    = 1./(8.*np.pi) * frot * fsb * fsigmar * fsigmaz  
        
        return(prob)
        
    def ThickDiscEDF(self,acts,xi):
        
        """ THICK DISC EDF (not normalized)
    
        Arguments:        
            acts - actions Jr,Jz,Lz (km/s kpc, km/s kpc, km/s kpc, [matrix])
            xi   - chemistry variables [Fe/H], [a/Fe], age (dex, dex, Gyr, [matrix])
    
        Returns:
            Thick disc EDF probabilities.
        
        """
        
        # Copy objects
        sfr_par          = self.sfr_par
        discchem_par     = self.discchem_par
        thickdiscedf_par = self.thickdiscedf_par
        
        # Actions
        acts = np.atleast_2d(acts)
        
        # Chemical parameters
        xi   = np.atleast_2d(xi)
        tau  = xi[:,2]
             
        # Disc chemical evolution model
        edfprobdiscchem = self.DiscChemicalEvolutionModel(acts,xi,discchem_par)
            
        # Thick disc EDF
        ftauthk        = self.StarFormationRate(tau,sfr_par)
        index          = (tau <= sfr_par["tauT"])
        ftauthk[index] = 0.
        edfprobthk     = self.PseudoIsoDF(acts,tau,thickdiscedf_par)*\
                         edfprobdiscchem*ftauthk
        
        return(edfprobthk)
        
    def ThinDiscEDF(self,acts,xi):
        
        """ THIN DISC EDF (not normalized)
    
        Arguments:        
            acts - actions Jr,Jz,Lz (km/s kpc, km/s kpc, km/s kpc, [matrix])
            xi   - chemistry variables [Fe/H], [a/Fe], age (dex, dex, Gyr, [matrix])
    
        Returns:
            Thin disc EDF probabilities.
        
        """
        
        # Copy objects
        sfr_par         = self.sfr_par
        discchem_par    = self.discchem_par
        thindiscedf_par = self.thindiscedf_par
        
        # Actions
        acts = np.atleast_2d(acts)
        
        # Chemical parameters
        xi      = np.atleast_2d(xi)
        tau     = xi[:,2]
             
        # Disc chemical evolution model
        edfprobdiscchem = self.DiscChemicalEvolutionModel(acts,xi,discchem_par)

        # Thin disc EDF
        ftauthn        = self.StarFormationRate(tau,sfr_par)
        index          = (tau > sfr_par["tauT"])
        ftauthn[index] = 0.
        edfprobthn     = self.PseudoIsoDF(acts,tau,thindiscedf_par)*\
                         edfprobdiscchem*ftauthn
                         
        return(edfprobthn)
        
    def StellarHaloEDF(self,acts,xi):
        
        """ STELLAR HALO EDF (not normalized)
    
        Arguments:        
            acts - actions Jr,Jz,Lz (km/s kpc, km/s kpc, km/s kpc, [matrix])
            xi   - chemistry variables [Fe/H], [a/Fe], age (dex, dex, Gyr, [matrix])
    
        Returns:
            Stellar halo EDF probabilities.
        
        """
        
        # Copy objects
        df_par       = self.stellarhaloedf_par
        halochem_par = self.halochem_par
        
        acts = np.atleast_2d(acts)
                
        # Actions
        Jr = acts[:,0]
        Jz = acts[:,1]
        Lz = acts[:,2]
                                       
        # Phase-space distribution function
        hJ  = df_par["Jcore"] + df_par["ar"]*Jr + df_par["aphi"]*np.abs(Lz) + df_par["az"]*Jz
        gJ  = df_par["br"]*Jr + df_par["bphi"]*np.abs(Lz) + df_par["bz"]*Jz
        fps = ((1.+df_par["J0"]/hJ)**df_par["alpha"])/((1.+gJ/df_par["J0"])**df_par["beta"])
        if (df_par["Jcutoff"]>0.):
            fps *= np.exp(-(gJ/df_par["Jcutoff"])**df_par["ncutoff"])
        fps *= (1./(8*np.pi**3))
        
        # Total EDF probability              
        prob = fps * self.HaloChemicalEvolutionModel(acts,xi,halochem_par)
    
        return(prob)
        
    def CalcDiscEDFNorm(self,Jscale):
        
        """ NORMALIZATION FOR TOTAL DISC EDF 
    
        Arguments:        
            Jscale - scale for actions (km/s kpc, [scalar])
    
        Returns:
            Total disc normalization.
        
        """
    
        # Transformed disc EDF
        @vegas.batchintegrand
        def TransDiscEDF(scaledstar):
             
             # Transform back coordinates
             Jr  = Jscale*np.arctanh(scaledstar[:,0])
             Jz  = Jscale*np.arctanh(scaledstar[:,1])            
             Lz  = Jscale*np.arctanh(scaledstar[:,2])
             feh = scaledstar[:,3]
             afe  = scaledstar[:,4]
        
             # Transformation Jacobian
             jac = Jscale**3./(1.-(np.tanh(Jr/Jscale))**2.)/\
                              (1.-(np.tanh(Jz/Jscale))**2.)/\
                              (1.-(np.tanh(Lz/Jscale))**2.)
                                                                                             
             # Create untransformed coordinate arrays
             acts = np.column_stack((Jr,Jz,Lz))
             xi   = np.column_stack((feh,afe,scaledstar[:,5]))
             
             return (jac*8.*np.pi**3.*(self.ThinDiscEDF(acts,xi)+self.ThickDiscEDF(acts,xi)))
        
        # Limits
        fehmin = -2.0
        fehmax = 1.0
        afemin = -1.0
        afemax = 1.0
        
        # Create integration object
        integ = vegas.Integrator([[0.,1.],
                                  [0.,1.],
                                  [0.,1.],
                                  [fehmin,fehmax],
                                  [afemin,afemax],
                                  [0.,self.sfr_par["taum"]]])
                                  
        # Train integration object
        integ(TransDiscEDF,nitn=5,neval=1000)      
                
        # Calculate integration
        result = integ(TransDiscEDF,nitn=self.niter,neval=self.maxeval)
        val    = result.mean
        err    = result.sdev
               
        print("Disc EDF normalization and error: "+str(np.round(val,5))+"+/"+str(np.round(err,4)))
        pererr = err/val*100.
        print("% error = "+str(np.round(pererr,2)))
        
        return(1./val)
        
    def CalcStellarHaloEDFNorm(self,Jscale):
        
        """ NORMALIZATION FOR STELLAR HALO EDF 
    
        Arguments:        
            Jscale - scale for actions (km/s kpc, [scalar])
    
        Returns:
            Stellar halo normalization.
        """
        
        # Copy objects
        halochem_par = self.halochem_par
        
        # Transformed stellar halo EDF
        @vegas.batchintegrand
        def TransStellarHaloEDF(scaledstar):
             
             # Transform back coordinates
             Jr  = Jscale*np.arctanh(scaledstar[:,0])
             Jz  = Jscale*np.arctanh(scaledstar[:,1])
             Lz  = Jscale*np.arctanh(scaledstar[:,2])
             feh = scaledstar[:,3]
             afe = scaledstar[:,4]
                                            
             jac = Jscale**3./(1.-(np.tanh(Jr/Jscale))**2.)/\
                              (1.-(np.tanh(Jz/Jscale))**2.)/\
                              (1.-(np.tanh(Lz/Jscale))**2.)
                                                
             # Create untransformed coordinate arrays
             acts = np.column_stack((Jr,Jz,Lz))
             xi   = np.column_stack((feh,afe,scaledstar[:,5]))             
             return (jac*self.StellarHaloEDF(acts,xi))
        
        # y and age limits
        fehmax = -1.8+3.*halochem_par["sigZ"]
        fehmin = -2.0-3.*halochem_par["sigZ"]
        afemax  = 0.3+3.*halochem_par["sigalpha"]
        afemin  = 0.0-3.*halochem_par["sigalpha"]
        agemin = 10.0-3.*halochem_par["sigtau"]
        
        # Create integration object
        integ = vegas.Integrator([[0.,1.],
                                  [0.,1.],
                                  [-1.,1.],
                                  [fehmin,fehmax],
                                  [afemin,afemax],
                                  [agemin,self.sfr_par["taum"]]])
                                  
        # Train integration object
        integ(TransStellarHaloEDF,nitn=5,neval=1000)      
                
        # Calculate integration
        result = integ(TransStellarHaloEDF,nitn=self.niter,neval=self.maxeval)
        val    = result.mean
        err    = result.sdev
                            
        print("Stellar halo EDF normalization and error: "+str(np.round(val,5))+"+/"+str(np.round(err,4)))
        pererr = err/val*100.
        print("% error = "+str(np.round(pererr,2)))
       
        return(1./val)
            
    def CompleteEDF(self,acts,xi):
        
        """ MILKY WAY EDF FOR EACH COEVAL POPULATION  

        Arguments:        
            acts - actions Jr,Jz,Lz, (km/s kpc, km/s kpc, km/s kpc, [matrix])
            xi   - chemistry variables [Fe/H], [a/Fe], age (dex, dex, Gyr, [matrix])
        Returns:
            Complete EDF probabilities (-, [vector])

        """
        
        # Actions
        acts  = np.atleast_2d(acts)
        
        # Chemical parameters
        xi    = np.atleast_2d(xi)
        
        # Disc EDF
        edfprobdisc = self.discedfnorm*(self.ThinDiscEDF(acts,xi)+self.ThickDiscEDF(acts,xi))
                            
        # Stellar halo EDF
        edfprobhalo = self.stellarhaloedfnorm*self.StellarHaloEDF(acts,xi)
                                    
        edfprob     = edfprobdisc+self.Fhalo*edfprobhalo
                 
        return(edfprob)