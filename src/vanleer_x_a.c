//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void VanLeerX_a_cpu(Field *Q){

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
  real rawm;
  real rawp;
  real s_harm;
  real s_max;
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+2*NGHX+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real InvDiffXmed(Nx+2*NGHX);
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
      for (i=XIM; i<size_x; i++) {	
#endif
//<#>
	ll = l;
	llxm = lxm;
	llxp = lxp;
	
	rawm = q[ll]-q[llxm];
	rawp = q[llxp]-q[ll];
	dqm = rawm*Inv_zone_size_xmed(i,j,k);
	dqp = rawp*Inv_zone_size_xmed(ixp,j,k);

	if(dqp*dqm<=0.0)  slope[ll] = 0.0;
#ifndef DONOR
	else {
	  s_harm = (2.*dqp*dqm) /(dqm+dqp);
	  s_max = 2.0*min2(fabs(rawm),fabs(rawp))/zone_size_x(i,j,k);
	  slope[ll] = (s_harm > 0.0) ? min2(s_harm, s_max) : max2(s_harm, -s_max);
	}
#else
	else  slope[ll] = 0.0;
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
