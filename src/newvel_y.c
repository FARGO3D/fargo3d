//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void NewVelocity_y_cpu () {

//<USER_DEFINED>
  INPUT(Density);
#ifdef Y
  INPUT(Mmy);
  INPUT(Mpy);
  OUTPUT(Vy);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* rho = Density->field_cpu;
#ifdef Y
  real* vy  = Vy -> field_cpu;
  real* mmy = Mmy->field_cpu;
  real* mpy = Mpy->field_cpu;
#endif
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
//<\EXTERNAL>

//<INTERNAL>
  int i; //Variables reserved
  int j; //for the topology
  int k; //of the kernels
  int ll;
  int llym;
//<\INTERNAL>

//<MAIN_LOOP>

  i = j = k = 0;

#ifdef Z
  for (k=1; k<size_z; k++) {
#endif
#ifdef Y
    for (j=1; j<size_y; j++) {
#endif
#ifdef X
      for (i=XIM; i<size_x; i++) {
#endif
//<#>
#ifdef Y
	ll = l;
	llym = lym;
	vy[ll] = (mmy[ll]+mpy[llym])/(rho[ll]+rho[llym]);
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
