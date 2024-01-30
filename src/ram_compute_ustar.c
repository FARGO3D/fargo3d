//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void RamComputeUstar_cpu(real dt) {

//<USER_DEFINED>
  INPUT2D(VxMed);
  OUTPUT(UStarmin);
  OUTPUT(PhiStarmin);
//<\USER_DEFINED>

//<EXTERNAL>
  real* vxmed      = VxMed->field_cpu;
  real* ustarmin   = UStarmin->field_cpu;
  real* phistarmin = PhiStarmin->field_cpu;
  int pitch   = Pitch_cpu;
  int stride  = Stride_cpu;
  int size_x  = Nx+2*NGHX;
  int size_y  = Ny+2*NGHY;
  int size_z  = Nz+2*NGHZ;
  int pitch2d = Pitch2D;
  real xmc0   = Xmc0;
  real xmc1   = Xmc1;
  real xmc2   = Xmc2;
  real xmc3   = Xmc3;
  real xmc4   = Xmc4;
  real xma    = XMA;
  real xmb    = XMB;
  real xmc    = XMC;
  real x_mesh_I = X_mesh_I;
  real _xmin = XMIN;
  real _xmax = XMAX;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
  int ll2D;
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+2*NGHX+1);
// real ymin(Ny+2*NGHY+1);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;

#if ZDIM
  for (k = 0; k < size_z; k++) {
#endif
#if YDIM
    for (j = 0; j < size_y; j++) {
#endif
#if XDIM
      for (i = 0; i < size_x; i++) {
#endif
//<#>
	ll = l;
	ll2D = l2D;

#if CARTESIAN
	phistarmin[ll] = xmin(i) - vxmed[ll2D]*dt;
#else
	phistarmin[ll] = xmin(i) - vxmed[ll2D]*dt/ymed(j);
#endif

	// Periodicity
	while(phistarmin[ll] < _xmin) {
	  phistarmin[ll] += (_xmax-_xmin);
	}
	while(phistarmin[ll] > _xmax) {
	  phistarmin[ll] -= (_xmax-_xmin);
	}

	ustarmin[ll] = UX(phistarmin[ll]);

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
