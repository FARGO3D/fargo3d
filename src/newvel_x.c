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
#if XDIM
  INPUT(Mmx);
  INPUT(Mpx);
  OUTPUT(Vx);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* rho = Density->field_cpu;
#if XDIM
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
#if SPHERICAL
  real rcyl;
#endif
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real OMEGAFRAME(1);
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
#if XDIM
	ll = l;
	llxm = lxm;

#if CARTESIAN
	vx[ll] = (mmx[ll]*Vol(i,j,k) + mpx[llxm]*Vol(ixm,j,k) )/(rho[ll]*Vol(i,j,k)+rho[llxm]*Vol(ixm,j,k));
#if SHEARINGBOX
	vx[ll] -= 2.0*OMEGAFRAME*ymed(j);
#endif
#endif
#if CYLINDRICAL
	vx[ll] = (mmx[ll]*Vol(i,j,k) + mpx[llxm]*Vol(ixm,j,k) )/((rho[ll]*Vol(i,j,k)+rho[llxm]*Vol(ixm,j,k))*ymed(j))-OMEGAFRAME*ymed(j);
#endif
#if SPHERICAL
	rcyl = ymed(j) * sin(zmed(k));
	vx[ll] = (mmx[ll]*Vol(i,j,k) + mpx[llxm]*Vol(ixm,j,k) )/((rho[ll]*Vol(i,j,k)+rho[llxm]*Vol(ixm,j,k))*rcyl)-OMEGAFRAME*rcyl;
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
