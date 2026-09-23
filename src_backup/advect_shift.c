//<FLAGS>
//#define __GPU
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void AdvectSHIFT_cpu (Field *F, FieldInt2D *NS) {

//<USER_DEFINED>
  DRAFT(Pressure);
  INPUT2DINT(NS);
  INPUT(F);
  OUTPUT(F);
//<\USER_DEFINED>

//<EXTERNAL>
  real* f     = F->field_cpu;
  real* aux   = Pressure->field_cpu;
  int* nshift = NS->field_cpu;
  int pitch   = Pitch_cpu;
  int stride  = Stride_cpu;
  int size_x  = Nx+2*NGHX;
  int size_y  = Ny+2*NGHY;
  int size_z  = Nz+2*NGHZ;
  int nx = Nx;
  int pitch2d = Pitch2D;
  int pitch2d_int = Pitch_Int_gpu;
//<\EXTERNAL>

//<INTERNAL>
  int i; //Variables reserved
  int j; //for the topology
  int k; //of the kernels
  int itarget;
  int ltarget;
  int ll;
//<\INTERNAL>

//<MAIN_LOOP>

  i = j = k = 0;

  for (k = 0; k < size_z; k++) {
    for (j = 0; j < size_y; j++) {
      for (i = 0; i < size_x; i++) {
//<#>
	ll = l;
	itarget = i-nshift[l2D_int];
	while (itarget <  NGHX)  itarget += nx;
	while (itarget >= nx+NGHX) itarget -= nx;
	ltarget = ll-i+itarget;
	aux[ll] = f[ltarget];
//<\#>
      }
    }
  }
//<\MAIN_LOOP>

//<LAST_BLOCK>
#ifdef __GPU
  Dev2Dev3D(F,Pressure);
#else
  memcpy(f, aux, sizeof(real)*size_x*size_y*size_z);
#endif
//<\LAST_BLOCK>
}
