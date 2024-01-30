//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void momenta_x_cpu() {

//<USER_DEFINED>
  INPUT(Density);
#if XDIM
  INPUT(Vx_temp);
  OUTPUT(Mmx);
  OUTPUT(Mpx);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* rho = Density->field_cpu;
#if XDIM
  real* vx = Vx_temp->field_cpu;
  real* mmx = Mmx->field_cpu;
  real* mpx = Mpx->field_cpu;
#endif
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = XIP;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
//<\EXTERNAL>

//<INTERNAL>
  int i; //Reserved variables
  int j; //for the topology
  int k; //of the kernels
  int ll;
  int llxp;
  real rcyl;
  (void) rcyl;
//<\INTERNAL>

//<CONSTANT>
// real zmin(Nz+2*NGHZ+1);
// real ymin(Ny+2*NGHY+1);
// real OMEGAFRAME(1);
//<\CONSTANT>

//<MAIN_LOOP>
  i = j = k = 0;

#if ZDIM
  for (k=0; k<size_z; k++) {
#endif
#if YDIM
    for (j=0; j<size_y; j++) {
#endif
#if XDIM
      for (i=0; i<size_x; i++) {
#endif
//<#>
#if XDIM
	ll = l;
	llxp = lxp;
#if CARTESIAN
	mmx[ll] = vx[ll] * rho[ll];
	mpx[ll] = vx[llxp] * rho[ll];
#if SHEARINGBOX
	mmx[ll] += 2.0*OMEGAFRAME*ymed(j)*rho[ll];
	mpx[ll] += 2.0*OMEGAFRAME*ymed(j)*rho[ll];
#endif
#endif

#if CYLINDRICAL
	mmx[ll] = (vx[ll] + ymed(j)*OMEGAFRAME) * ymed(j) * rho[ll];
	mpx[ll] = (vx[llxp] + ymed(j)*OMEGAFRAME) * ymed(j) * rho[ll];
#endif

#if SPHERICAL
	rcyl = ymed(j) * sin(zmed(k));
	mmx[ll] = (vx[ll] + rcyl * OMEGAFRAME) * rcyl * rho[ll];
	mpx[ll] = (vx[llxp] + rcyl * OMEGAFRAME) * rcyl * rho[ll];
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
