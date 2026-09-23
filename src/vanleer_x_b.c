//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void VanLeerX_b_cpu(real dt, Field *Q, Field *Qs, Field *Vx_t){

//<USER_DEFINED>
  INPUT(Slope);
  INPUT(Q);
  INPUT(Vx_t);
  OUTPUT(Qs);
//<\USER_DEFINED>

//<EXTERNAL>
  real* q = Q->field_cpu;
  real* qs = Qs->field_cpu;
  real* vx = Vx_t->field_cpu;
  real* slope = Slope->field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
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

	if(vx[ll]>0.0)
	  qs[ll] = q[llxm] + 0.5*(zone_size_x(j,k)-vx[ll]*dt)*slope[llxm];
	else
	  qs[ll] = q[ll] - 0.5*(zone_size_x(j,k)+vx[ll]*dt)*slope[ll];
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
