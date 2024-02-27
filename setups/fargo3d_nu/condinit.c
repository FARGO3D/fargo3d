#include "fargo3d.h"

void Init() {
  
  OUTPUT(Density);
  OUTPUT(Energy);
  OUTPUT(Vx);
  OUTPUT(Vy);

  int i,j,k;
  real r, h, omega;
  real soundspeed;
  
  real *vphi   = Vx->field_cpu;
  real *vr     = Vy->field_cpu;
  real *vtheta = Vz->field_cpu;
  real *rho  = Density->field_cpu;
  
#ifdef ADIABATIC
  real *e   = Energy->field_cpu;
#endif
#ifdef ISOTHERMAL
  real *cs   = Energy->field_cpu;
#endif

  i = j = k = 0;
  for (k=0; k<Nz+2*NGHZ; k++) {
    for (j=0; j<Ny+2*NGHY; j++) {
      r = Ymed(j);
      h = ASPECTRATIO*pow(r/R0,FLARINGINDEX);
      real r3 = r*r*r;
      omega = sqrt(G*MSTAR/r3);
      // vertically isothermal
      soundspeed  = h*r*omega;
      for (i=0; i<Nx+2*NGHX; i++) {
        vr[l]     = 0.0;
        vtheta[l] = 0.0;
        #ifdef CYLINDRICAL
          rho[l] = SIGMA0*pow(r/R0,-SIGMASLOPE)*exp(-pow(Zmed(k)/h,2.0)/2.0)/(ZMAX-ZMIN);
        #else
          real xi = SIGMASLOPE+1.+FLARINGINDEX;
          real beta = 1.-2.*FLARINGINDEX;
          if (FLARINGINDEX == 0.0) {
            rho[l] = SIGMA0/sqrt(2.0*M_PI)/(R0*ASPECTRATIO)*pow(r/R0,-xi)* \
              pow(sin(Zmed(k)),-beta-xi+1./(h*h));
          } else {
            rho[l] = SIGMA0/sqrt(2.0*M_PI)/(R0*ASPECTRATIO)*pow(r/R0,-xi)* \
              pow(sin(Zmed(k)),-xi-beta)*         \
              exp((1.-pow(sin(Zmed(k)),-2.*FLARINGINDEX))/2./FLARINGINDEX/(h*h));
          }
          vphi[l] = omega*r*sqrt(pow(sin(Zmed(k)),-2.*FLARINGINDEX)-(beta+xi)*h*h);
          vphi[l] -= OMEGAFRAME*r*sin(Zmed(k));
          #endif
          #ifdef ISOTHERMAL
                cs[l] = soundspeed;
          #endif
          #ifdef ADIABATIC
                e[l] = pow(soundspeed,2)*rho[l]/(GAMMA-1.0);
          #endif

          vphi[l]  *= (1.+ASPECTRATIO*NOISE*(drand48()-.5));
          vr[l]    += soundspeed*NOISE*(drand48()-.5);
          vtheta[l]+= soundspeed*NOISE*(drand48()-.5);
      }
    } 
  }
}

void CondInit() {
   Fluids[0] = CreateFluid("gas",GAS);
   SelectFluid(0);
   Init();
}
