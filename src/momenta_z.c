//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void momenta_z_cpu() {

//<USER_DEFINED>
  INPUT(Density);
#ifdef Z
  INPUT(Vz_temp);
  OUTPUT(Mmz);
  OUTPUT(Mpz);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* rho = Density->field_cpu;
#ifdef Z
  real* vz = Vz_temp->field_cpu;
  real* mmz = Mmz->field_cpu;
  real* mpz = Mpz->field_cpu;
#endif
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ-1;
//<\EXTERNAL>

//<INTERNAL>
  int i; //Reserved variables 
  int j; //for the topology
  int k; //of the kernels
  int ll;
  int llzp;
//<\INTERNAL>

//<CONSTANT>
// real ymin(Ny+2*NGHY+1);
//<\CONSTANT>

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
#ifdef Z

	ll = l;
	llzp = lzp;

	mpz[ll] = vz[llzp] * rho[ll];
	mmz[ll] = vz[ll] * rho[ll];
#ifdef SPHERICAL
	mpz[ll] *= ymed(j);
	mmz[ll] *= ymed(j);
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
