//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void HallEffect_emfy_cpu(){

//<USER_DEFINED>
  INPUT(Bx);
  INPUT(By);
  INPUT(Bz);
  INPUT(Jx);
  INPUT(Jz);
  INPUT(EtaHall);
  OUTPUT(EmfyH);
//<\USER_DEFINED>

//<EXTERNAL>
  real* bx = Bx->field_cpu;
  real* by = By->field_cpu;
  real* bz = Bz->field_cpu;
  real* jx = Jx->field_cpu;
  real* jz = Jz->field_cpu;
  real* emf = EmfyH->field_cpu;
  real* eta = EtaHall->field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx;
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ-1;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
  real j1;
  real j2;
  real b1;
  real b2;
  real b3;
  real bmod;
  real etac;
  real eps = 1.0e-30;
//<\INTERNAL>


//<MAIN_LOOP>
  for (k=1; k<size_z; k++) {
    for (j=1; j<size_y; j++) {
      for (i=0; i<size_x; i++) {
//<#>
	ll = l;
	b2 = 0.5*(bx[ll]+bx[lzm]);
	b1 = 0.5*(bz[ll]+bz[lxm]);
	j2 = 0.25*(jx[ll] + jx[lxm] + jx[lyp] + jx[lxm+pitch]);
	j1 = 0.25*(jz[ll] + jz[lzm] + jz[lyp] + jz[lyp-stride]);

	b3 = 0.125*(by[ll] + by[lxm] + by[lzm] + by[lxm-stride] + by[lyp] + by[lxm+pitch] + by[lyp-stride] + by[lxm+pitch-stride]);

	bmod = sqrt(b1*b1 + b2*b2 + b3*b3 + eps);

	etac = 0.25*( eta[ll] + eta[lzm] + eta[lxm] + eta[lxm-stride]);

#if CYLINDRICAL
	emf[ll] =  etac*(j1*b2 - j2*b1)/bmod;
#else
	emf[ll] = -etac*(j1*b2 - j2*b1)/bmod;
#endif

//<\#>
      }
    }
  }
//<\MAIN_LOOP>

}
