//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void CorrectVtheta_cpu(real domega) {

//<USER_DEFINED>
  INPUT(Vx);
  OUTPUT(Vx);
//<\USER_DEFINED>

//<EXTERNAL>
  real* vt = Vx->field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
//<\INTERNAL>

//<CONSTANT>
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;

#if ZDIM
  for(k=0; k < size_z; k++) {
#endif
#if YDIM
    for (j = 0; j < size_y; j++) {
#endif
#if XDIM
      for (i = 0; i < size_x; i++) {
#endif
//<#>
#if SPHERICAL
	vt[l] -= domega*ymed(j)*sin(zmed(k));
#endif
#if CYLINDRICAL
	vt[l] -= domega*ymed(j);
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
