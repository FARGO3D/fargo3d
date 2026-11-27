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
  real rawm;
  real rawp;
  real s_harm;
  real s_max;
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

	rawm = q[ll]-q[llym];
	rawp = q[llyp]-q[ll];
	dqm = rawm/(ymed(j)-ymed(j-1));
	dqp = rawp/(ymed(j+1)-ymed(j));
	if(dqp*dqm<=0.0) slope[ll] = 0.0;
#ifndef DONOR
	else {
	  s_harm = 2.*dqp*dqm/(dqm+dqp);
	  s_max = 2.0*min2(fabs(rawm),fabs(rawp))/zone_size_y(j,k);
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
