#include "fargo3d.h"

void InitSizes(dgratio)
  real dgratio;
{

  int i,j,k;
  real r, omega, h;
  
  real *rho  = Density->field_cpu;
  real *cs   = Energy->field_cpu;
  real *vphi = Vx->field_cpu;
  real *vr   = Vy->field_cpu;
  
  real rhog, rhod;
  real vk;
  
  i = j = k = 0;
  
  for (k=0; k<Nz+2*NGHZ; k++) {
    for (j=0; j<Ny+2*NGHY; j++) {
      for (i=0; i<Nx+2*NGHX; i++) {
	
        r     = Ymed(j);
        h = ASPECTRATIO*pow(r/R0,FLARINGINDEX);
        omega = sqrt(G*MSTAR/r/r/r);                       //Keplerian frequency
        rhog  = SIGMA0*pow(r/R0,-SIGMASLOPE);              //Gas surface density

        if (Fluidtype == GAS) {
          rho[l]   = rhog;
          vphi[l]  = omega*r*sqrt(1.0 + pow(ASPECTRATIO,2)*pow(r/R0,2*FLARINGINDEX)*
                (2.0*FLARINGINDEX - 1.0 - SIGMASLOPE));
          vr[l]    = 0.0;
#ifdef ISOTHERMAL
          cs[l] = h*sqrt(G*MSTAR/r);
#else
	        cs[l] = rho[l]*h*h*G*MSTAR/r/(GAMMA-1.0);
#endif

	      }
	
        if (Fluidtype == DUST) {
          rho[l]  = rhog*dgratio;
          vphi[l] = omega*r;
          vr[l]   = 0.0;
          cs[l]   = 0.0;
        }
	
	      vphi[l] -= OMEGAFRAME*r;
	
      }
    }
  }
}

void CondInit() {
  
  int id_gas = 0;
  int feedback = YES;
  int id_dust;
  FILE *output;
  char name[256], dust_name[MAXNAMELENGTH];
  real unit_length, unit_mass, sum, xi=0.0;
  real dust_Stokes, dust_inv_Stokes;
  real dust_size, dust_size_codeunits, dust_internaldensity_codeunits, InvStokesOverRhoGas;

  //We first create the gaseous fluid and store it in the array Fluids[]
  Fluids[id_gas] = CreateFluid("gas",GAS);

  //We now select the gas fluid
  SelectFluid(id_gas);

  //and fill its fields
  InitSizes(xi);

  /* We now repeat the process for the dust fluids (loop) */
#if defined(EPSTEINDRAG)
  unit_length = UNITOFLENGTHAU*1.49597871e11;     // unit of length in meters
  unit_mass   = UNITOFMASSMSUN*MSTAR_MKS;         // unit of mass in kg
  
  /* Convert dust's internal density from g/cm^3 to code units */
  dust_internaldensity_codeunits = DUSTINTERNALRHO*1.0e3*pow(unit_length, 3.)/unit_mass;
  
  /* Write a file dustsizes.dat with size and mass ratio (wrt gas) of
     all dust fluids */
  sprintf (name, "%sdustsizes.dat", OUTPUTDIR);
  output = fopen_prs (name, "w");
  fprintf (output, "# dust_id \t dust size [meters] \t dust/gas ratio \n");

  /* Compute sum over dust fluids of s^4-p where s = size of each dust
     fluid and -p the power-law exponent of the dust's size
     distribution. This is required to compute the dust-to-gas density
     ratio for each dust fluid via a size distribution. */
  sum = 0.0;
  for (id_dust = 1; id_dust<NFLUIDS; id_dust++) {
    if (NFLUIDS > 2)
      dust_size = DUSTSIZEMIN*exp((id_dust-1.0)/(NFLUIDS-2.0)*log(DUSTSIZEMAX/DUSTSIZEMIN));
    else
      dust_size = DUSTSIZEMIN;
    sum += pow(dust_size/unit_length,4.0+DUSTSLOPEDIST);
  }
#endif

#if defined(STOKESNUMBER)
  /* Write a file duststokesnb.dat with Stokes numbers of all dust
     fluids */
  sprintf (name, "%sduststokesnb.dat", OUTPUTDIR);
  output = fopen_prs (name, "w");
  fprintf (output, "# dust_id \t Stokes number \n");
#endif
  
#if defined(NODUSTFEEDBACK)
  feedback = NO;
#endif

  //Loop over dust fluids
  for (id_dust = 1; id_dust<NFLUIDS; id_dust++) {
    sprintf(dust_name,"dust%d",id_dust); //We assign different names to the dust fluids

    Fluids[id_dust]  = CreateFluid(dust_name, DUST);
    SelectFluid(id_dust);

#if defined(EPSTEINDRAG)
    /* Assign size to each dust fluid */
    if (NFLUIDS > 2)
      dust_size = DUSTSIZEMIN*exp((id_dust-1.0)/(NFLUIDS-2.0)*log(DUSTSIZEMAX/DUSTSIZEMIN));
    else
      dust_size = DUSTSIZEMIN;
    
    /* Convert dust's size from meters to code units */
    dust_size_codeunits = dust_size/unit_length;
    
    /* Dust-to-gas initial density ratio */
    xi = EPSILON * pow(dust_size_codeunits,4.0+DUSTSLOPEDIST) / sum;
    fprintf (output, "%d\t%lg\t%lg\n",id_dust,dust_size,xi);
    InitSizes(xi);

#ifdef Z   //3D
    /* contains rho_gas x c_s / Omega */
    InvStokesOverRhoGas = sqrt(8.0/M_PI)/dust_size_codeunits/dust_internaldensity_codeunits;
#else      //2D
    /* contains Sigma_gas / Stokes number */
    InvStokesOverRhoGas = 2.0/M_PI/dust_size_codeunits/dust_internaldensity_codeunits;
#endif

    /*We now fill the collision matrix (Feedback from dust included)
      Note: ColRate() moves the collision matrix to the device.
      If feedback=NO, gas does not feel the drag force.*/
    ColRate(InvStokesOverRhoGas, id_gas, id_dust, feedback);
#endif
    
#if defined(STOKESNUMBER)
    /* Dust-to-gas initial density ratio */
    xi = EPSILON;
    InitSizes(xi);

    /* Assign Stokes number to each dust fluid */
    if (NFLUIDS > 2)
      dust_Stokes = STOKESMIN*exp((id_dust-1.0)/(NFLUIDS-2.0)*log(STOKESMAX/STOKESMIN));
    else
      dust_Stokes = STOKESMIN;
    fprintf (output, "%d\t%lg\n",id_dust,dust_Stokes);

    /* Inverse Stokes number */
    dust_inv_Stokes = 1./dust_Stokes;

    /*We now fill the collision matrix (Feedback from dust included)
      Note: ColRate() moves the collision matrix to the device.
      If feedback=NO, gas does not feel the drag force.*/
    ColRate(dust_inv_Stokes, id_gas, id_dust, feedback);
#endif

  }

#if defined(EPSTEINDRAG) || defined(STOKESNUMBER)
  fclose (output);
#endif

}
