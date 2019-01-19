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
  //#ifdef ADIABATIC
  INPUT(Energy);
  //#endif
#ifdef X
  INPUT(Vx);
#endif
#ifdef Y
  INPUT(Vy);
#endif
#ifdef Z
  INPUT(Vz);
#endif
#ifdef MHD
  INPUT(Bx);
  INPUT(By);
  INPUT(Bz);
#endif
  OUTPUT(Density);
  //#ifdef ADIABATIC
  OUTPUT(Energy);
  //#endif
#ifdef X
  OUTPUT(Vx);
#endif
#ifdef Y
  OUTPUT(Vy);
#endif
#ifdef Z
  OUTPUT(Vz);
#endif
#ifdef MHD
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
#ifdef X  
  real* vx = Vx->field_cpu;
#endif
#ifdef Y
  real* vy = Vy->field_cpu;
#endif
#ifdef Z
  real* vz = Vz->field_cpu;
#endif
  //#ifdef ADIABATIC
  real* energy = Energy->field_cpu;
  //#endif
#ifdef MHD
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
#ifdef Z
    for (k = 0; k < size_z; k++) {
#endif
#ifdef Y
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
#ifdef Y
	  vy[lghost1]     = vy[lcopy1];
	  vy[lghost2]     = vy[lcopy2];
#endif
#ifdef X
	  vx[lghost1]     = vx[lcopy1];
	  vx[lghost2]     = vx[lcopy2];
#endif	 
#ifdef Z
	  vz[lghost1]     = vz[lcopy1];
	  vz[lghost2]     = vz[lcopy2];
#endif	 
	  //#ifdef ADIABATIC
	  energy[lghost1]   = energy[lcopy1];
	  energy[lghost2]   = energy[lcopy2];
	  //#endif
#ifdef MHD
	  bx[lghost1]     = bx[lcopy1];
	  bx[lghost2]     = bx[lcopy2];
	  by[lghost1]     = by[lcopy1];
	  by[lghost2]     = by[lcopy2];
	  bz[lghost1]     = bz[lcopy1];
	  bz[lghost2]     = bz[lcopy2];
#endif
//<\#>
	}
#ifdef Y
      }
#endif	 
#ifdef Z
    }
#endif
//<\MAIN_LOOP>
}
