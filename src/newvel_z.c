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
#ifdef Z
  INPUT(Mmz);
  INPUT(Mpz);
  OUTPUT(Vz);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* rho = Density->field_cpu;
#ifdef Z
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
// real ymin(Ny+2*NGHY+1);
//<\CONSTANT>

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
#ifdef Z
	ll = l;
	llzm = lzm;
	vz[ll] = (mmz[ll]+mpz[llzm])/(rho[ll]+rho[llzm]);
#ifdef SPHERICAL
	vz[ll] /= ymed(j);
#endif
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
