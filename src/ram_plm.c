//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void RamSlopes_cpu(Field *Q){

//<USER_DEFINED>
  INPUT(Q);
  OUTPUT(Slope);
//<\USER_DEFINED>

//<EXTERNAL>
  real* slope = Slope->field_cpu;
  real* q = Q->field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = XIP;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
  int llxm;
  int llxp;
  real dqm;
  real dqp;
  real dxmed;
  real dxmedp;
//<\INTERNAL>

//<CONSTANT>
  // real Sxi(Nx+2*NGHX);
  // real xmin(Nx+2*NGHX+1);
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
      for (i=XIM; i<size_x; i++) {
#endif
//<#>
	ll = l;
	llxm = lxm;
	llxp = lxp;

	dxmed  = 0.5*( Sxi(i) + Sxi(ixm) );
	dxmedp = 0.5*( Sxi(ixp) + Sxi(i) );

	dqm = (q[ll]-q[llxm])/dxmed;
	dqp = (q[llxp]-q[ll])/dxmedp;
	if(dqp*dqm<=0.0)  slope[ll] = 0.0;
#if (!DONOR)
	else  slope[ll] = (2.*dqp*dqm)/(dqm+dqp);
#else
	else  slope[ll] = 0.0;
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
