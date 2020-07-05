//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void AmbipolarDiffusion_emfz_cpu(){

//<USER_DEFINED>
  INPUT(Bx);
  INPUT(By);
  INPUT(Bz);
  INPUT(Jx);
  INPUT(Jy);
  INPUT(Jz);
  INPUT(EtaAD);
  INPUT(Emfz);
  OUTPUT(Emfz);
//<\USER_DEFINED>

//<EXTERNAL>
  real* bx = Bx->field_cpu;
  real* by = By->field_cpu;
  real* bz = Bz->field_cpu;
  real* jx = Jx->field_cpu;
  real* jy = Jy->field_cpu;
  real* jz = Jz->field_cpu;
  real* eta = EtaAD->field_cpu;
  real* emf = Emfz->field_cpu;
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
	j1 = 0.25*(jx[ll] + jx[lzp] + jx[lxm] + jx[lxm+stride]);
	j2 = 0.25*(jy[ll] + jy[lzp] + jy[lym] + jy[lzp - pitch]);
	j3 = jz[ll];
	b1 = 0.5*(bx[ll] + bx[lym]);
	b2 = 0.5*(by[ll] + by[lxm]);
	b3 = 0.125*(bz[ll] + bz[lym] + bz[lxm] + bz[lxm-pitch] + bz[lzp] + bz[lym+stride] + bz[lxm+stride] + bz[lxm-pitch+stride]);
	bmod = b1*b1 + b2*b2 + b3*b3 + eps;

	etac = 0.25*( eta[ll] + eta[lxm] + eta[lym] + eta[lxm-pitch] );

#if ( defined(CARTESIAN) || defined(SPHERICAL) )    
       	emf[ll] += etac*(j3 - (b1*j1+b2*j2+b3*j3)*b3/bmod);
#endif
#ifdef CYLINDRICAL
	emf[ll] -= etac*(j3 - (b1*j1+b2*j2+b3*j3)*b3/bmod);
#endif

//<\#>
      }
    }
  }
//<\MAIN_LOOP>
}
