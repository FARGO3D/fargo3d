//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void CloseBoundaries_cpu(Field *Vy, Field *Vz){

//<USER_DEFINED>
  INPUT(Vy);
  INPUT(Vz);
  OUTPUT(Vy);
  OUTPUT(Vz);
//<\USER_DEFINED>

//<EXTERNAL>
  real* vy = Vy->field_cpu;
  real* vz = Vz->field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x   = Nx;
  int size_y   = Ny+2*NGHY;
  int size_z   = Nz+2*NGHZ;
  int ncpuy    = Gridd.NJ;
  int ncpuz    = Gridd.NK;
  int jcpu     = J;
  int kcpu     = K;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
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

#if YDIMMINCLOSED
	if ((jcpu == 0) && (j == NGHY))
	  vy[l] = 0.0;
#endif

#if YDIMMAXCLOSED
	if ((jcpu == ncpuy-1) && (j == size_y-NGHY-1))
	  vy[l] = 0.0;
#endif

#if ZDIMMINCLOSED
	if ((kcpu == 0) && (k == NGHZ)) {
	  vz[l] = 0.0;
	}
#endif

#if ZDIMMAXCLOSED
	if ((kcpu == ncpuz-1) && (k == size_z-NGHZ-1))
	  vz[l] = 0.0;
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
