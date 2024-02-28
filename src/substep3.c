//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void SubStep3_cpu (real dt) {

//<USER_DEFINED>
  INPUT(Energy);
#if XDIM
  INPUT(Vx_temp);
#endif
#if YDIM
  INPUT(Vy_temp);
#endif
#if ZDIM
  INPUT(Vz_temp);
#endif
  OUTPUT(Energy);
//<\USER_DEFINED>

//<EXTERNAL>
  real* e   = Energy->field_cpu;
#if XDIM
  real* vx  = Vx_temp->field_cpu;
#endif
#if YDIM
  real* vy  = Vy_temp->field_cpu;
#endif
#if ZDIM
  real* vz  = Vz_temp->field_cpu;
#endif
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = XIP;
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ-1;
//<\EXTERNAL>

//<INTERNAL>
  int i; //Variables reserved
  int j; //for the topology
  int k; //of the kernels
  int ll;
#if XDIM
  int llxp;
#endif
#if YDIM
  int llyp;
#endif
#if ZDIM
  int llzp;
#endif
  real term;
  real div_v;
//<\INTERNAL>

//<CONSTANT>
// real GAMMA(1);
// real Sxi(Nx);
// real Sxj(Ny+2*NGHY);
// real Syj(Ny+2*NGHY);
// real Szj(Ny+2*NGHY);
// real Sxk(Nz+2*NGHZ);
// real Syk(Nz+2*NGHZ);
// real Szk(Nz+2*NGHZ);
// real InvVj(Ny+2*NGHY);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;

#if ZDIM
  for(k=0; k<size_z; k++) {
#endif
#if YDIM
    for(j=0; j<size_y; j++) {
#endif
#if XDIM
      for(i=0; i<size_x; i++) {
#endif
//<#>

	ll = l;
#if XDIM
	llxp = lxp;
#endif
#if YDIM
	llyp = lyp;
#endif
#if ZDIM
	llzp = lzp;
#endif
	div_v = 0.0;
#if XDIM
	div_v += (vx[llxp]-vx[ll])*SurfX(j,k);
#endif
#if YDIM
	div_v += (vy[llyp]*SurfY(i,j+1,k)-vy[ll]*SurfY(i,j,k));
#endif
#if ZDIM
	div_v += (vz[llzp]*SurfZ(i,j,k+1)-vz[ll]*SurfZ(i,j,k));
#endif
	term = 0.5 * dt * (GAMMA - 1.) * div_v * InvVol(i,j,k);
	e[ll] *= (1.0-term)/(1.0+term);
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
