//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void momenta_y_cpu() {

//<USER_DEFINED>
  INPUT(Density);
#if YDIM
  INPUT(Vy_temp);
  OUTPUT(Mmy);
  OUTPUT(Mpy);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* rho = Density->field_cpu;
#if YDIM
  real* vy = Vy_temp->field_cpu;
  real* mmy = Mmy->field_cpu;
  real* mpy = Mpy->field_cpu;
#endif
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ;
//<\EXTERNAL>

//<INTERNAL>
  int i; //Reserved variables
  int j; //for the topology
  int k; //of the kernels
  int ll;
  int llyp;
//<\INTERNAL>

//<MAIN_LOOP>

  i = j = k = 0;

#if ZDIM
  for (k=0; k<size_z; k++) {
#endif
#if YDIM
    for (j=0; j<size_y; j++) {
#endif
#if XDIM
      for (i=0; i<size_x; i++) {
#endif
//<#>
#if YDIM
	ll = l;
	llyp = lyp;
	mmy[ll] = vy[ll] * rho[ll];
	mpy[ll] = vy[llyp] * rho[ll];
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
