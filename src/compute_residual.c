//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void ComputeResidual_cpu(real dt) {

//<USER_DEFINED>
  INPUT(Vx_temp);
  INPUT2D(VxMed);
  OUTPUT(Vx);
  OUTPUT(Vx_temp);
  OUTPUT2DINT(Nshift);
//<\USER_DEFINED>

//<EXTERNAL>
  real* vx    = Vx_temp->field_cpu;
  real* vxr   = Vx->field_cpu;
  real* vxmed = VxMed->field_cpu;
  int* nshift = Nshift->field_cpu;
  int pitch   = Pitch_cpu;
  int stride  = Stride_cpu;
  int size_x  = Nx+2*NGHX;
  int size_y  = Ny+2*NGHY;
  int size_z  = Nz+2*NGHZ;
  real dx     = Dx;
  int pitch2d = Pitch2D;
  int pitch2d_int = Pitch_Int_gpu;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  real ntilde;
  real nround;
  int ll;
  int ll2D;
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+2*NGHX+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;

#ifdef Z
  for (k = 0; k < size_z; k++) {
#endif 
#ifdef Y
    for (j = 0; j < size_y; j++) {
#endif
#ifdef X
      for (i = 0; i < size_x; i++) {
#endif
//<#>
	ll = l;
	ll2D = l2D;
	ntilde = vxmed[ll2D]*dt/zone_size_x(j,k);
	nround = floor(ntilde+0.5);
	if(i == 0) 
	  nshift[l2D_int] = (int)nround;
	vxr[ll] = vx[ll]-vxmed[ll2D];
	vx[ll] = (ntilde-nround)*zone_size_x(j,k)/dt;
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
