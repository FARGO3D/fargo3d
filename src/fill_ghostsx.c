//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void Fill_GhostsX_cpu () {

//<USER_DEFINED>
  INPUT(Density);
  //#if ADIABATIC
  INPUT(Energy);
  //#endif
#if XDIM
  INPUT(Vx);
#endif
#if YDIM
  INPUT(Vy);
#endif
#if ZDIM
  INPUT(Vz);
#endif
#if MHD
  INPUT(Bx);
  INPUT(By);
  INPUT(Bz);
#endif
  OUTPUT(Density);
  //#if ADIABATIC
  OUTPUT(Energy);
  //#endif
#if XDIM
  OUTPUT(Vx);
#endif
#if YDIM
  OUTPUT(Vy);
#endif
#if ZDIM
  OUTPUT(Vz);
#endif
#if MHD
  OUTPUT(Bx);
  OUTPUT(By);
  OUTPUT(Bz);
#endif
//<\USER_DEFINED>

//<INTERNAL>
  int i;
  int j;
  int k;
  int lghost1;
  int lcopy1;
  int lghost2;
  int lcopy2;
//<\INTERNAL>

//<EXTERNAL>
  real* rho = Density->field_cpu;
#if XDIM
  real* vx = Vx->field_cpu;
#endif
#if YDIM
  real* vy = Vy->field_cpu;
#endif
#if ZDIM
  real* vz = Vz->field_cpu;
#endif
  //#if ADIABATIC
  real* energy = Energy->field_cpu;
  //#endif
#if MHD
  real* bx = Bx->field_cpu;
  real* by = By->field_cpu;
  real* bz = Bz->field_cpu;
#endif
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int nx = Nx;
  int size_x = NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
//<\EXTERNAL>

//<MAIN_LOOP>
    i = j = k = 0;
#if ZDIM
    for (k = 0; k < size_z; k++) {
#endif
#if YDIM
      for (j = 0; j < size_y; j++) {
#endif
	for (i = 0; i < size_x; i++) {
//<#>
	  lghost1 = l;
          lcopy1 = l + nx;
          lcopy2 = lghost1 + NGHX;
          lghost2 = lcopy2 + nx;

	  rho[lghost1]    = rho[lcopy1];
	  rho[lghost2]    = rho[lcopy2];
#if YDIM
	  vy[lghost1]     = vy[lcopy1];
	  vy[lghost2]     = vy[lcopy2];
#endif
#if XDIM
	  vx[lghost1]     = vx[lcopy1];
	  vx[lghost2]     = vx[lcopy2];
#endif
#if ZDIM
	  vz[lghost1]     = vz[lcopy1];
	  vz[lghost2]     = vz[lcopy2];
#endif
	  //#if ADIABATIC
	  energy[lghost1]   = energy[lcopy1];
	  energy[lghost2]   = energy[lcopy2];
	  //#endif
#if MHD
	  bx[lghost1]     = bx[lcopy1];
	  bx[lghost2]     = bx[lcopy2];
	  by[lghost1]     = by[lcopy1];
	  by[lghost2]     = by[lcopy2];
	  bz[lghost1]     = bz[lcopy1];
	  bz[lghost2]     = bz[lcopy2];
#endif
//<\#>
	}
#if YDIM
      }
#endif
#if ZDIM
    }
#endif
//<\MAIN_LOOP>
}
