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
  real dx = Dx;
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
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
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
	
	dqm = (q[ll]-q[llxm]);
	dqp = (q[llxp]-q[ll]);
	if(dqp*dqm<=0.0)  slope[ll] = 0.0;
#ifndef DONOR
	else  slope[ll] = (2.*dqp*dqm) /
		((dqm+dqp)*(zone_size_x(j,k)));
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
