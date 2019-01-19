//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void ChangeFrame_cpu(int sign, Field *V, Field2D *Vm) {
  
//<USER_DEFINED>
  INPUT(V);
  INPUT2D(Vm);
  OUTPUT(V);
//<\USER_DEFINED>
  
//<EXTERNAL>
  real* v  = V->field_cpu;
  real* vm = Vm->field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
  int pitch2d = Pitch2D;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
//<\INTERNAL>

//<MAIN_LOOP>

  i = j = k = 0;

#ifdef Z
  for (k = 0; k < size_z; k++) {
#endif
#ifdef Y
    for (j = 0; j < size_y; j++) {
#endif
#ifdef X
      for (i = 0; i < size_x; i++) {
#endif
//<#>
	v[l] += (real)sign*vm[l2D];
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
