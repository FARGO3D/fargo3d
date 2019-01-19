#include "fargo3d.h"

void _init_stockholm() {
  
  INPUT(Density);
  OUTPUT2D(Density0);
#ifdef ADIABATIC
  INPUT(Energy);
  OUTPUT2D(Energy0);
#endif
#ifdef X
  INPUT(Vx);
  OUTPUT2D(Vx0);
#endif
#ifdef Y
  INPUT(Vy);
  OUTPUT2D(Vy0);
#endif
#ifdef Z
  INPUT(Vz);
  OUTPUT2D(Vz0);
#endif
  
  int i,j,k;
  
  i = j = k = 0;
  
#ifdef X
  real* vx  = Vx->field_cpu;
  real* vx0 = Vx0->field_cpu;
#endif
#ifdef Y
  real* vy  = Vy->field_cpu;
  real* vy0 = Vy0->field_cpu;
#endif
#ifdef Z
  real* vz  = Vz->field_cpu;
  real* vz0 = Vz0->field_cpu;
#endif
#ifdef ADIABATIC
  real* e    = Energy->field_cpu;
  real* e0   = Energy0->field_cpu;
#endif
  real* rho  = Density->field_cpu;
  real* rho0 = Density0->field_cpu;
  
#ifdef Z
  for (k=0; k<Nz+2*NGHZ; k++) {
#endif
#ifdef Y
    for (j=0; j<Ny+2*NGHY; j++) {
#endif
#ifdef ADIABATIC
      e0[l2D]   = e[l];
#endif
#ifdef X
      vx0[l2D]  = vx[l];
#endif
#ifdef Y
      vy0[l2D]  = vy[l];
#endif
#ifdef Z
      vz0[l2D]  = vz[l];
#endif
      rho0[l2D] = rho[l];
#ifdef Y
    }
#endif
#ifdef Z
  }
#endif
}

void init_stockholm() {

  static boolean init = TRUE;

  if (init) MULTIFLUID(_init_stockholm());
  
  init = FALSE;
}
