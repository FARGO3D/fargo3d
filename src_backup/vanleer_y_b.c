//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void VanLeerY_b_cpu(real dt, Field *Q, Field *Qs){

//<USER_DEFINED>
  INPUT(Q);
  INPUT(Slope);
  INPUT(Vy_temp);
  OUTPUT(Qs);
//<\USER_DEFINED>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
  int llym;
//<\INTERNAL>
  
//<EXTERNAL>
  real* q = Q->field_cpu;
  real* qs = Qs->field_cpu;
  real* slope = Slope->field_cpu;
  real* vy = Vy_temp -> field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ;
//<\EXTERNAL>

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
    	if(vy[ll]>0.)
	  qs[ll] = q[llym] + 0.5 * (zone_size_y(j-1,k)
				  -vy[ll]*dt)*slope[llym];
	else
	  qs[ll] = q[ll] - 0.5 * (zone_size_y(j,k)
	  			  +vy[ll]*dt)*slope[ll];
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
