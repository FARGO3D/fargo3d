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
#if YDIM
  INPUT(Mmy);
  INPUT(Mpy);
  OUTPUT(Vy);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* rho = Density->field_cpu;
#if YDIM
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

//<CONSTANT>
// real Syk(Nz+2*NGHZ);
// real InvVj(Ny+2*NGHY);
// real Sxi(Nx+2*NGHX);
//<\CONSTANT>


//<MAIN_LOOP>

  i = j = k = 0;

#if ZDIM
  for (k=1; k<size_z; k++) {
#endif
#if YDIM
    for (j=1; j<size_y; j++) {
#endif
#if XDIM
      for (i=XIM; i<size_x; i++) {
#endif
//<#>
#if YDIM
	ll = l;
	llym = lym;
	vy[ll] = (mmy[ll]*Vol(i,j,k)+mpy[llym]*Vol(i,j-1,k))/(rho[ll]*Vol(i,j,k)+rho[llym]*Vol(i,j-1,k));
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
