//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void AmbipolarDiffusion_emfx_cpu(){

//<USER_DEFINED>
  INPUT(Bx);
  INPUT(By);
  INPUT(Bz);
  INPUT(Jx);
  INPUT(Jy);
  INPUT(Jz);
  INPUT(EtaAD);
  INPUT(Emfx);
  OUTPUT(Emfx);
//<\USER_DEFINED>

//<EXTERNAL>
  real* bx = Bx->field_cpu;
  real* by = By->field_cpu;
  real* bz = Bz->field_cpu;
  real* jx = Jx->field_cpu;
  real* jy = Jy->field_cpu;
  real* jz = Jz->field_cpu;
  real* eta = EtaAD->field_cpu;
  real* emf = Emfx->field_cpu;
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
  real j3;
  real b1;
  real b2;
  real b3;
  real bmod;
  real etac;
  real eps = 1e-30;
//<\INTERNAL>

//<MAIN_LOOP>
  for (k=1; k<size_z; k++) {
    for (j=1; j<size_y; j++) {
      for (i=0; i<size_x; i++) {
//<#>
	ll = l;
	j2 = 0.25*(jy[ll] + jy[lym] + jy[lxp] + jy[lxp-pitch]);
	j1 = jx[ll];
	j3 = 0.25*(jz[ll] + jz[lzm] + jz[lxp] + jz[lxp-stride]);

	b1 = 0.125*(bx[ll] + bx[lxp] + bx[lzm] + bx[lxp-stride] + bx[lym] + bx[lxp-pitch] + bx[lym-stride] + bx[lxp-pitch-stride]);
	b2 = 0.5*(by[ll] + by[lzm]);
	b3 = 0.5*(bz[ll] + bz[lym]);

	bmod = b1*b1 + b2*b2 + b3*b3 + eps;
	etac = 0.25*(eta[ll]+eta[lym]+eta[lzm]+eta[lym-stride]);
	
#if (defined(CARTESIAN) || defined(SPHERICAL) )
	emf[ll] += etac*(j1 - (b1*j1+b2*j2+b3*j3)*b1/bmod);
#endif
#ifdef CYLINDRICAL
	emf[ll] -= etac*(j1 - (b1*j1+b2*j2+b3*j3)*b1/bmod);
#endif
//<\#>
      }
    }
  }
//<\MAIN_LOOP>

}
