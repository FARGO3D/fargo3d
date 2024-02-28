//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void SubStep2_a_cpu (real dt) {

//<USER_DEFINED>
  INPUT(Density);
  INPUT(Pressure);
#if XDIM
#if COLLISIONPREDICTOR
  INPUT(Vx_half);
#else
  INPUT(Vx);
#endif
  OUTPUT(Mpx);
#endif
#if YDIM
#if COLLISIONPREDICTOR
  INPUT(Vy_half);
#else
  INPUT(Vy);
#endif
  OUTPUT(Mpy);
#endif
#if ZDIM
#if COLLISIONPREDICTOR
  INPUT(Vz_half);
#else
  INPUT(Vz);
#endif
  OUTPUT(Mpz);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* rho    = Density->field_cpu;
  real* p      = Pressure->field_cpu;
#if XDIM
#if COLLISIONPREDICTOR
  real* vx     = Vx_half->field_cpu;
#else
  real* vx     = Vx->field_cpu;
#endif
  real* pres_x = Mpx->field_cpu;
#endif
#if YDIM
#if COLLISIONPREDICTOR
  real* vy     = Vy_half->field_cpu;
#else
  real* vy     = Vy->field_cpu;
#endif
  real* pres_y = Mpy->field_cpu;
#endif
#if ZDIM
#if COLLISIONPREDICTOR
  real* vz     = Vz_half->field_cpu;
#else
  real* vz     = Vz->field_cpu;
#endif
  real* pres_z = Mpz->field_cpu;
#endif
  int pitch    = Pitch_cpu;
  int stride   = Stride_cpu;
  int size_x   = XIP;
  int size_y   = Ny+2*NGHY-1;
  int size_z   = Nz+2*NGHZ-1;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
#if XDIM
  int llxp;
  real dvx;
#endif
#if YDIM
  int llyp;
  real dvy;
#endif
#if ZDIM
  int llzp;
  real dvz;
#endif
//<\INTERNAL>

//<CONSTANT>
//real GAMMA(1);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;

#if ZDIM
  for(k=0; k<size_z; k++) {
#endif
#if YDIM
    for(j=0; j<size_y; j++) {
#endif
#if XDIM
      for(i=0; i<size_x; i++) {
#endif
//<#>

	ll = l;
#if XDIM
	llxp = lxp;
#endif
#if YDIM
	llyp = lyp;
#endif
#if ZDIM
	llzp = lzp;
#endif

#if XDIM
	dvx = vx[llxp]-vx[ll];
	if (dvx < 0.0) {
	  pres_x[ll] = CVNR*CVNR*rho[ll]*dvx*dvx;
#if STRONG_SHOCK
	  pres_x[ll] -= CVNL*rho[ll]*sqrt(GAMMA*fabs(p[ll]/rho[ll]))*dvx;
#endif
	}
	else {
	  pres_x[ll] = 0.0;
	}
#endif
#if YDIM
	dvy = vy[llyp]-vy[ll];
	if (dvy < 0.0) {
	  pres_y[ll] = CVNR*CVNR*rho[ll]*dvy*dvy;
#if STRONG_SHOCK
	  pres_y[ll] -= CVNL*rho[ll]*sqrt(GAMMA*fabs(p[ll]/rho[ll]))*dvy;
#endif
	}
	else {
	  pres_y[ll] = 0.0;
	}
#endif
#if ZDIM
	dvz = vz[llzp]-vz[ll];
	if (dvz < 0.0) {
	  pres_z[ll] = CVNR*CVNR*rho[ll]*dvz*dvz;
#if STRONG_SHOCK
	  pres_z[ll] -= CVNL*rho[ll]*sqrt(GAMMA*fabs(p[ll]/rho[ll]))*dvz;
#endif
	}
	else {
	  pres_z[ll] = 0.0;
	}
#endif
//<\#>
#if XDIM
      }
#endif
#if YDIM
    }
#endif
#if ZDIM
  }
#endif
//<\MAIN_LOOP>
}
