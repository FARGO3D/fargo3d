//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void VanLeerZ_b_cpu(real dt, Field *Q, Field *Qs){

//<USER_DEFINED>
  INPUT(Q);
  INPUT(Slope);
  INPUT(Vz_temp);
  OUTPUT(Qs);
//<\USER_DEFINED>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
  int llzm;
//<\INTERNAL>
  
//<EXTERNAL>
  real* q = Q->field_cpu;
  real* qs = Qs->field_cpu;
  real* slope = Slope->field_cpu;
  real* vz = Vz_temp -> field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ-1;
//<\EXTERNAL>

//<CONSTANT>
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
//<\CONSTANT>


//<MAIN_LOOP>

  i = j = k = 0;

#ifdef Z
  for (k=1; k<size_z; k++) {
#endif
#ifdef Y
    for (j=0; j<size_y; j++) {
#endif
#ifdef X
      for (i=0; i<size_x; i++) {
#endif
//<#>
	ll = l;
	llzm = lzm;
    	if(vz[l]>0.)
	  qs[ll] = q[llzm] + 0.5 * (zone_size_z(j,k-1)
				  -vz[ll]*dt)*slope[llzm];
	else
	  qs[ll] = q[ll] - 0.5 * (zone_size_z(j,k)
	  			  +vz[ll]*dt)*slope[ll];
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
