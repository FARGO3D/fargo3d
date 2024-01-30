//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void ComputeVmed(Field *V) {
  int j,k;
  int ll2D;
  
  FARGO_SAFE(ComputeVweight(V, Qs));

  reduction_SUM(Qs, 0, Ny+2*NGHY, 0, Nz+2*NGHZ);

  INPUT2D(Reduction2D);
  OUTPUT2D(VxMed);

  k = j = 0;

#if ZDIM
  for (k = 0; k < Nz+2*NGHZ; k++) {
#endif
#if YDIM
    for (j = 0; j < Ny+2*NGHY; j++) {
#endif
      ll2D = l2D;
      VxMed->field_cpu[ll2D] = Reduction2D->field_cpu[ll2D]/(XMAX-XMIN);
#if YDIM
    }
#endif
#if ZDIM
  }
#endif


}


void ComputeVweight_cpu(Field *V, Field *Q) {
  
//<USER_DEFINED>
  INPUT(V);
  OUTPUT(Q);
//<\USER_DEFINED>

//<EXTERNAL>
  real* v     = V->field_cpu;
  real* q     = Q->field_cpu;
  int pitch   = Pitch_cpu;
  int stride  = Stride_cpu;
  int size_x  = Nx+2*NGHX;
  int size_y  = Ny+2*NGHY;
  int size_z  = Nz+2*NGHZ;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+2*NGHX+1);
//<\CONSTANT>

//<MAIN_LOOP>
  
  i = j = k = 0;

#if ZDIM
  for (k = 0; k < size_z; k++) {
#endif 
#if YDIM
    for (j = 0; j < size_y; j++) {
#endif
#if XDIM
      for (i = 0; i < size_x; i++) {
#endif
//<#>
	ll = l;
	q[ll]  = v[ll]*(xmin(i+1)-xmin(i));
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
