//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void SetupHook1_cpu() {  // Empty function. May be used as a template for custom function in setup directory.

//<USER_DEFINED>
//<\USER_DEFINED>

//<EXTERNAL>
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
//<\EXTERNAL>

//<INTERNAL>
  int __attribute__((unused))i; //Variables reserved
  int __attribute__((unused))j; //for the topology
  int __attribute__((unused))k; //of the kernels
  int ll;
  (void)ll;
//<\INTERNAL>

//<MAIN_LOOP>

  i = j = k = 0;

#if ZDIM
  for(k=1; k<size_z; k++) {
#endif
#if YDIM
    for(j=1; j<size_y; j++) {
#endif
#if XDIM
      for(i=0; i<size_x; i++) {
#endif
//<#>
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
