//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void mon_maxwell_cpu () {

//<USER_DEFINED>
  INPUT(Density);
  INPUT(Bx);
  INPUT(By);
  OUTPUT(Slope);
//<\USER_DEFINED>


//<EXTERNAL>
  real* dens = Density->field_cpu;
  real* bx = Bx->field_cpu;
  real* by = By->field_cpu;
  real* interm = Slope->field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
//<\INTERNAL>

//<CONSTANT>
// real Sxi(Nx);
// real Syk(Nz+2*NGHZ);
// real InvVj(Ny+2*NGHY);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;

#if ZDIM
  for (k=0; k<size_z; k++) {
#endif
#if YDIM
    for (j=0; j<size_y; j++) {
#endif
#if XDIM
      for (i=0; i<size_x; i++ ) {
#endif
//<#>
	ll = l;
	interm[ll] = .25*(bx[ll]+bx[lxp])*(by[ll]+by[lyp])/MU0*Vol(i,j,k);
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
