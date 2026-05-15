//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void ComputeJy_cpu() {

//<USER_DEFINED>
  INPUT(Bx);
  INPUT(Bz);
  OUTPUT(Jy);
//<\USER_DEFINED>

//<EXTERNAL>
  real* bx = Bx->field_cpu;
  real* bz = Bz->field_cpu;
  real* jy = Jy->field_cpu;
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
  real fact;
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
	fact1 = 1.0/(zmed(k)-zmed(k-1));
	fact2 = 1.0/dx;
	jy[ll] = ((bx[ll]-bx[lzm])*fact1-(bz[ll]-bz[lxm])*fact2)/mu0;
#endif
#ifdef CYLINDRICAL
	fact1 = 1.0/(ymed(j)*dx);
	fact2 = 1.0/(zmed(k)-zmed(k-1));
	jy[ll] = ((bz[ll]-bz[lxm])*fact1-(bx[ll]-bx[lzm])*fact2)/mu0;
#endif
#ifdef SPHERICAL
	fact  = ymed(j)*sin(zmin(k));
	fact1 = 1.0/(fact*dx);
	fact2 = 1.0/(fact*(zmed(k)-zmed(k-1)));
	jy[ll] = ((sin(zmed(k))*bx[ll]-sin(zmed(k-1))*bx[lzm])*fact2-(bz[ll]-bz[lxm])*fact1)/mu0;
#endif
//<\#>
      }
    }
  }
//<\MAIN_LOOP>
}
