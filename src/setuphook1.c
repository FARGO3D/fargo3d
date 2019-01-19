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

#ifdef Z
  for(k=1; k<size_z; k++) {
#endif
#ifdef Y
    for(j=1; j<size_y; j++) {
#endif
#ifdef X
      for(i=0; i<size_x; i++) {
#endif
//<#>
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
