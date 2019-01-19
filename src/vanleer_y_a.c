//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void VanLeerY_a_cpu(Field *Q){
  
//<USER_DEFINED>
  INPUT(Q);
  OUTPUT(Slope);
//<\USER_DEFINED>

//<EXTERNAL>
  real* q = Q->field_cpu;
  real* slope = Slope->field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  real dqm;
  real dqp;
  int ll;
  int llym;
  int llyp;
//<\INTERNAL>

//<CONSTANT>
// real ymin(Ny+2*NGHY+1);
//<\CONSTANT>


//<MAIN_LOOP>

  i = j = k = 0;

#ifdef Z
  for (k=0; k<size_z; k++) {
#endif
#ifdef Y
    for (j=1; j<size_y; j++) {
#endif
#ifdef X
      for (i=0; i<size_x; i++) {
#endif
//<#>
	ll = l;
	llym = lym;
	llyp = lyp;

	dqm = (q[ll]-q[llym])/zone_size_y(j,k);
	dqp = (q[llyp]-q[ll])/zone_size_y(j+1,k);
	if(dqp*dqm<=0) slope[ll] = 0;
#ifndef DONOR
	else  slope[ll] = 2.*dqp*dqm/(dqm+dqp);
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
