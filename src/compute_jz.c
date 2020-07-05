//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void ComputeJz_cpu() {

//<USER_DEFINED>
  INPUT(Bx);
  INPUT(By);
  OUTPUT(Jz);
//<\USER_DEFINED>

//<EXTERNAL>
  real* bx = Bx->field_cpu;
  real* by = By->field_cpu;
  real* jz = Jz->field_cpu;
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
	fact1 = 1.0/dx;
	fact2 = 1.0/(ymed(j)-ymed(j-1));
	jz[ll] = ((by[ll]-by[lxm])*fact1-(bx[ll]-bx[lym])*fact2)/mu0;
#endif
#ifdef CYLINDRICAL
	fact1 = 1.0/(ymin(j)*dx);
	fact2 = 1.0/(ymin(j)*(ymed(j)-ymed(j-1)));
	jz[ll] = ((ymed(j)*bx[ll]-ymed(j-1)*bx[lym])*fact2-(by[ll]-by[lxm])*fact1)/mu0;
#endif
#ifdef SPHERICAL
	fact1 = 1.0/(ymin(j)*sin(zmed(k))*dx);
	fact2 = 1.0/(ymin(j)*(ymed(j)-ymed(j-1)));
	jz[ll] = ((by[ll]-by[lxm])*fact1-(ymed(j)*bx[ll]-ymed(j-1)*bx[lym])*fact2)/mu0;
#endif
//<\#>
      }
    }
  }
//<\MAIN_LOOP>
}
