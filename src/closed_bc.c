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
  real dx      = Dx;
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

#ifdef Z
  for (k=0; k<size_z; k++) {
#endif
#ifdef Y
    for (j=0; j<size_y; j++) {
#endif
#ifdef X
      for (i=0; i<size_x; i++) {
#endif
//<#>

#ifdef YMINCLOSED	
	if ((jcpu == 0) && (j == NGHY))
	  vy[l] = 0.0;
#endif

#ifdef YMAXCLOSED
	if ((jcpu == ncpuy-1) && (j == size_y-NGHY-1))
	  vy[l] = 0.0;
#endif

#ifdef ZMINCLOSED	
	if ((kcpu == 0) && (k == NGHZ)) {
	  vz[l] = 0.0;
	}
#endif

#ifdef ZMAXCLOSED
	if ((kcpu == ncpuz-1) && (k == size_z-NGHZ-1))
	  vz[l] = 0.0;
#endif

//<\#>
#ifdef X
      }
#endif
#ifdef Y
    }
#endif
#ifdef Z
  }
#endif
//<\MAIN_LOOP>
}
