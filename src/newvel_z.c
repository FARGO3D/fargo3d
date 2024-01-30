//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void NewVelocity_z_cpu () {

//<USER_DEFINED>
  INPUT(Density);
#if ZDIM
  INPUT(Mmz);
  INPUT(Mpz);
  OUTPUT(Vz);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* rho = Density->field_cpu;
#if ZDIM
  real* vz  = Vz -> field_cpu;
  real* mmz = Mmz->field_cpu;
  real* mpz = Mpz->field_cpu;
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
  int llzm;
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+1);
// real ymin(Ny+2*NGHY+1);
// real Syk(Nz+2*NGHZ);
// real InvVj(Ny+2*NGHY);
// real Sxi(Nx);
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
#if ZDIM
	ll = l;
	llzm = lzm;
	vz[ll] = (mmz[ll]*Vol(i,j,k)+mpz[llzm]*Vol(i,j,k-1))/(rho[ll]*Vol(i,j,k)+rho[llzm]*Vol(i,j,k-1));
#if SPHERICAL
	vz[ll] /= ymed(j);
#endif
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
