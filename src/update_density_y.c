//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void UpdateDensityY_cpu(real dt, Field *Q) {

//<USER_DEFINED>
  INPUT(Q);
  INPUT(Vy_temp);
  INPUT(DensStar);
  OUTPUT(Q);
//<\USER_DEFINED>

//<EXTERNAL>
  real* qb = Q->field_cpu;
  real* vy = Vy_temp->field_cpu;
  real* rho_s = DensStar->field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ;
//<\EXTERNAL>

//<INTERNAL>
  int i; //Variables reserved
  int j; //for the topology
  int k; //of the kernels
  int ll;
  int llyp;
//<\INTERNAL>

//<CONSTANT>
// real Sxi(Nx);
// real Sxj(Ny+2*NGHY);
// real Syj(Ny+2*NGHY);
// real Szj(Ny+2*NGHY);
// real Sxk(Nz+2*NGHZ);
// real Syk(Nz+2*NGHZ);
// real Szk(Nz+2*NGHZ);
// real InvVj(Ny+2*NGHY);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;

#if ZDIM
  for (k=0; k<size_z; k++) {
#endif
#if YDIM
    for (j=0; j<size_y; j++) {
#endif
#if XDIM
      for (i=0; i<size_x; i++) {
#endif
//<#>
	ll = l;
	llyp = lyp;

	qb[ll] += (vy[ll]*rho_s[ll]*SurfY(i,j,k) -	\
		  vy[llyp]*rho_s[llyp]		\
		  *SurfY(i,j+1,k)) * dt * InvVol(i,j,k);
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
