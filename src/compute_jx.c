//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void ComputeJx_cpu() {

//<USER_DEFINED>
  INPUT(By);
  INPUT(Bz);
  OUTPUT(Jx);
//<\USER_DEFINED>

//<EXTERNAL>
  real* by = By->field_cpu;
  real* bz = Bz->field_cpu;
  real* jx = Jx->field_cpu;
  real dx = Dx;
  real mu0 = MU0;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx;
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ-1;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
  real fact1;
  real fact2;
//<\INTERNAL>

//<CONSTANT>
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
//<\CONSTANT>

//<MAIN_LOOP>
  for (k=1; k<size_z; k++) {
    for (j=1; j<size_y; j++) {
      for (i=0; i<size_x; i++) {
//<#>
	ll = l;
#ifdef CARTESIAN
	fact1 = 1.0/(ymed(j)-ymed(j-1));
	fact2 = 1.0/(zmed(k)-zmed(k-1));
	jx[ll] = ((bz[ll]-bz[lym])*fact1-(by[ll]-by[lzm])*fact2)/mu0; // rot(B)_x
#endif
#ifdef CYLINDRICAL
	fact1 = 1.0/(ymed(j)-ymed(j-1));
	fact2 = 1.0/(zmed(k)-zmed(k-1));
	jx[ll] = ((by[ll]-by[lzm])*fact2-(bz[ll]-bz[lym])*fact1)/mu0;  //rot(B)_phi
#endif
#ifdef SPHERICAL
	fact1 = 1.0/(ymin(j)*(ymed(j)-ymed(j-1)));
	fact2 = 1.0/(ymin(j)*(zmed(k)-zmed(k-1)));
	jx[ll] = ((ymed(j)*bz[ll]-ymed(j-1)*bz[lym])*fact1-(by[ll]-by[lzm])*fact2)/mu0; //rot(B)_phi
#endif
//<\#>
      }
    }
  }
//<\MAIN_LOOP>
}
