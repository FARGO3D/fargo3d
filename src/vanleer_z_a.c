//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void VanLeerZ_a_cpu(Field *Q){

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
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ-1;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  real dqm;
  real dqp;
  int ll;
  int llzm;
  int llzp;
//<\INTERNAL>

//<CONSTANT>
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
//<\CONSTANT>


//<MAIN_LOOP>

  i = j = k = 0;

#if ZDIM
  for (k=1; k<size_z; k++) {
#endif
#if YDIM
    for (j=0; j<size_y; j++) {
#endif
#if XDIM
      for (i=0; i<size_x; i++) {
#endif
//<#>
	ll = l;
	llzm = lzm;
	llzp = lzp;

	dqm = (q[ll]-q[llzm])/zone_size_z(j,k);
	dqp = (q[llzp]-q[ll])/zone_size_z(j,k+1);
	if(dqp*dqm<=0) slope[ll] = 0;
#if (!DONOR)
	else  slope[ll] = 2.*dqp*dqm/(dqm+dqp);
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
