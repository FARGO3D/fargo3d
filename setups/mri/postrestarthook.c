#include "fargo3d.h"

void PostRestartHook() {
  
  srand48(ArrayNb);

  int i,j,k;

  real r,denslocal;
  real* rho = Density->field_cpu;
  real* vx = Vx->field_cpu;
  real* vy = Vy->field_cpu;
  real* vz = Vz->field_cpu;
  real* bx = Bx->field_cpu;
  real* by = By->field_cpu;
  real* bz = Bz->field_cpu;
  real* cs = Energy->field_cpu;

  printf("Executing POSTRESTARHOOK %d\n",ArrayNb);

  for (k=0;k<Nz+2*NGHZ;k++) {
    for (j=0; j<Ny+2*NGHY;j++) {
      for (i=0; i<Nx+2*NGHX; i++) {
	r = Ymed(j);
	denslocal = SIGMA0/(ZMAX-ZMIN)*pow(r/R0,-SIGMASLOPE);
	vy[l] = (drand48()-.5)*0.1*vy[l];
	vz[l] = (drand48()-.5)*0.1*vz[l];
	rho[l] = denslocal;
      }
    }
  }
}
