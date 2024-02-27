#include "fargo3d.h"

void _CondInit() {
  OUTPUT(Density);
  OUTPUT(Energy);
  OUTPUT(Vx);
  OUTPUT(Vy);

  #ifdef BETACOOLING 
  OUTPUT2D(OmegaOverBeta);
  real *OoB = OmegaOverBeta->field_cpu;
  #endif
  
  int i,j,k;
  real r, omega;
  real soundspeed;
  
  real *vphi = Vx->field_cpu;
  real *vr   = Vy->field_cpu;
  real *rho  = Density->field_cpu;

  real rhog, rhod;
  real vk;
  
  
#ifdef ADIABATIC
  real *e   = Energy->field_cpu;
#endif
#ifdef ISOTHERMAL
  real *cs   = Energy->field_cpu;
#endif

  i = j = k = 0;
  
  for (j=0; j<Ny+2*NGHY; j++) {
    for (i=0; i<Nx+2*NGHX; i++) {
      
      r = Ymed(j);
      omega = sqrt(G*MSTAR/r/r/r);
      rhog  = SIGMA0*pow(r/R0,-SIGMASLOPE);
      rhod  = rhog*EPSILON;   
      soundspeed  = ASPECTRATIO*pow(r/R0,FLARINGINDEX)*omega*r;
      
      
  if (Fluidtype == GAS) {
    rho[l] = rhog;
    #ifdef ISOTHERMAL
          cs[l] = soundspeed;
    #endif
    #ifdef ADIABATIC
          e[l] = GAMMA*pow(soundspeed,2)*rho[l]/(GAMMA-1.0);
    #endif
    #ifdef BETACOOLING
          OoB[l2D] = omega/BETA;
    #endif

    vphi[l] = omega*r*sqrt(1.0+pow(ASPECTRATIO,2)*pow(r/R0,2*FLARINGINDEX)*
        (2.0*FLARINGINDEX - 1.0 - SIGMASLOPE));
    vphi[l] -= OMEGAFRAME*r;
    //vphi[l] *= (1.+ASPECTRATIO*NOISE*(drand48()-.5));
    vr[l]    = -1.5*ALPHA*pow(ASPECTRATIO, 2)*pow(r/R0,2*FLARINGINDEX-0.5);

  }

  if (Fluidtype == DUST) {
	  rho[l]  = rhod;
	  vphi[l] = omega*r;
	  vr[l]   = (-1.5*ALPHA+(2*FLARINGINDEX-1-SIGMASLOPE)/INVSTOKES1)*pow(ASPECTRATIO, 2)*pow(r/R0,2*FLARINGINDEX-0.5);
	  #ifdef ADIABATIC
      e[l]  = 0.0;
    #endif
    #ifdef ISOTHERMAL
      cs[l]  = 0.0;
    #endif
    vphi[l] -= OMEGAFRAME*r;
	}
    }
  } 
}

void CondInit() {
  
  int id_gas = 0;
  int feedback = NO;
  //We first create the gaseous fluid and store it in the array Fluids[]
  Fluids[id_gas] = CreateFluid("gas",GAS);

  //We now select the fluid
  SelectFluid(id_gas);

  //and fill its fields
  _CondInit();

  //We repeat the process for the dust fluids
  char dust_name[MAXNAMELENGTH];
  int id_dust;

  for(id_dust = 1; id_dust<NFLUIDS; id_dust++) {
    sprintf(dust_name,"dust%d",id_dust); //We assign different names to the dust fluids

    Fluids[id_dust]  = CreateFluid(dust_name, DUST);
    SelectFluid(id_dust);
    _CondInit();

  }

  /*We now fill the collision matrix (Feedback from dust included)
   Note: ColRate() moves the collision matrix to the device.
   If feedback=NO, gas does not feel the drag force.*/
  
  ColRate(INVSTOKES1, id_gas, 1, feedback);

}