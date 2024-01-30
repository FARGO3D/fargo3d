//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void DivideByRho_cpu(Field *Q) {

//<USER_DEFINED>
  INPUT(Density);
  INPUT(Q);
  OUTPUT(DivRho);
//<\USER_DEFINED>


//<EXTERNAL>
  real* q      = Q->field_cpu;
  real* divrho = DivRho->field_cpu;
  real* rho    = Density->field_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
  int stride = Stride_cpu;
  int pitch  = Pitch_cpu;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
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
	ll = l;
	divrho[ll] = q[ll]/rho[ll];
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
