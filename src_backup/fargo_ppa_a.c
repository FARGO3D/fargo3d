//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void VanLeerX_PPA_a_cpu(Field *Q){
  
//<USER_DEFINED>
  INPUT(Q);
  OUTPUT(Slope);
#ifdef PPA_STEEPENER
  OUTPUT(LapPPA);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* slope = Slope->field_cpu;
#ifdef PPA_STEEPENER
  real* lapla = LapPPA->field_cpu;
#endif
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
  real dqm;
  real dqp;
  real work;
//<\INTERNAL>
  
//<MAIN_LOOP>

  i = j = k = 0;

#ifdef Z
  for (k=0; k<size_z; k++) {
#endif
#ifdef Y
    for (j=0; j<size_y; j++) {
#endif
      for (i=XIM; i<size_x; i++) {
//<#>
	dqm = (q[l]-q[lxm]);
	dqp = (q[lxp]-q[l]);
	if(dqp*dqm<=0.0)  slope[l] = 0.0;
	else { // Monotonized centered slope limited
	  slope[l] = 0.5*(q[lxp]-q[lxm]);
	  work = fabs(slope[l]);
	  if (2.0*fabs(dqm) < work) work = 2.0*fabs(dqm);
	  if (2.0*fabs(dqp) < work) work = 2.0*fabs(dqp);
	  if (slope[l] < 0) slope[l] = -work;
	  else slope[l] = work;
	}
#ifdef PPA_STEEPENER
	lapla[l] = (q[lxp]+q[lxm]-2.0*q[l])*.1666666666666666666666;
#endif
//<\#>
      }
#ifdef Y
    }
#endif
#ifdef Z
  }
#endif
//<\MAIN_LOOP>
}
