//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void NewVelocity_x_cpu () {

//<USER_DEFINED>
  INPUT(Density);
#ifdef X
  INPUT(Mmx);
  INPUT(Mpx);
  OUTPUT(Vx);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* rho = Density->field_cpu;
#ifdef X
  real* vx  = Vx->field_cpu;
  real* mmx = Mmx->field_cpu;
  real* mpx = Mpx->field_cpu;
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
  int llxm;
#ifdef SPHERICAL
  real rcyl;
#endif
//<\INTERNAL>

//<CONSTANT>
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real OMEGAFRAME(1);
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
#ifdef X
	ll = l;
	llxm = lxm;

#ifdef CARTESIAN
	vx[ll] = (mmx[ll]+mpx[llxm])/(rho[ll]+rho[llxm]);
#ifdef SHEARINGBOX
	vx[ll] -= 2.0*OMEGAFRAME*ymed(j);
#endif
#endif
#ifdef CYLINDRICAL
	vx[ll] = (mmx[ll]+mpx[llxm])/((rho[ll]+rho[llxm])*ymed(j))-OMEGAFRAME*ymed(j);
#endif
#ifdef SPHERICAL
	rcyl = ymed(j) * sin(zmed(k));
	vx[ll] = (mmx[ll]+mpx[llxm])/((rho[ll]+rho[llxm])*rcyl)-OMEGAFRAME*rcyl;
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
