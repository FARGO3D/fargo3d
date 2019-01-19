//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void visctensor_cart_cpu(){

//<USER_DEFINED>
  INPUT(Density);
#ifdef X
  INPUT(Vx);
  OUTPUT(Mmx);
  OUTPUT(Mpx);
#endif
#ifdef Y
  INPUT(Vy);
  OUTPUT(Mmy);
  OUTPUT(Mpy);
#endif
#ifdef Z
  INPUT(Vz);
  OUTPUT(Mmz);
  OUTPUT(Mpz);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* rho = Density->field_cpu;
#ifdef X
  real* vx = Vx->field_cpu;
#endif
#ifdef Y
  real* vy = Vy->field_cpu;
#endif
#ifdef Z
  real* vz = Vz->field_cpu;
#endif
#ifdef X
  real* tauxx = Mmx->field_cpu;
#endif
#ifdef Y
  real* tauyy = Mmy->field_cpu;
#endif
#ifdef Z
  real* tauzz = Mmz->field_cpu;
#endif
#if defined(X) && defined(Z)
  real* tauxz = Mpx->field_cpu;
#endif
#if defined(Y) && defined(X)
  real* tauyx = Mpy->field_cpu;
#endif
#if defined(Z) && defined(Y)
  real* tauzy = Mpz->field_cpu;
#endif
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = XIP; 
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ-1;
  real dx = Dx;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  real div_v;
//<\INTERNAL>

//<CONSTANT>
// real NU(1);
// real Sxj(Ny+2*NGHY);
// real Syj(Ny+2*NGHY);
// real Szj(Ny+2*NGHY);
// real Sxk(Nz+2*NGHZ);
// real Syk(Nz+2*NGHZ);
// real Szk(Nz+2*NGHZ);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real InvVj(Ny+2*NGHY);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;

#ifdef Z
  for(k=1; k<size_z; k++) {
#endif
#ifdef Y
    for(j=1; j<size_y; j++) {
#endif
#ifdef X
      for(i=XIM; i<size_x; i++) {
#endif
//<#>
//Evaluate centered divergence.
	div_v = 0.0;
#ifdef X
	div_v += (vx[lxp]-vx[l])*SurfX(j,k);
#endif
#ifdef Y
	div_v += (vy[lyp]*SurfY(j+1,k)-vy[l]*SurfY(j,k));
#endif
#ifdef Z
	div_v += (vz[lzp]*SurfZ(j,k+1)-vz[l]*SurfZ(j,k));
#endif
	div_v *= 2.0/3.0*InvVol(j,k);

#ifdef X
	tauxx[l] = NU*rho[l]*(2.0*(vx[lxp]-vx[l])/dx - div_v);
#endif
#ifdef Y
	tauyy[l] = NU*rho[l]*(2.0*(vy[lyp]-vy[l])/(ymin(j+1)-ymin(j)) - div_v);
#endif
#ifdef Z
	tauzz[l] = NU*rho[l]*(2.0*(vz[lzp]-vz[l])/(zmin(k+1)-zmin(k)) - div_v);
#endif

#if defined(X) && defined(Z)
	tauxz[l] = NU*.25*(rho[l]+rho[lzm]+rho[lxm]+rho[lxm-stride])*((vx[l]-vx[lzm])/(zmed(k)-zmed(k-1)) + (vz[l]-vz[lxm])/dx); //centered on lower, left "radial" edge in y
#endif

#if defined(Y) && defined(X)
	tauyx[l] = NU*.25*(rho[l]+rho[lxm]+rho[lym]+rho[lxm-pitch])*((vy[l]-vy[lxm])/dx + (vx[l]-vx[lym])/(ymed(j)-ymed(j-1))); //centered on left, inner vertical edge in z
#endif

#if defined(Z) && defined(Y)
	tauzy[l] = NU*.25*(rho[l]+rho[lym]+rho[lzm]+rho[lym-stride])*((vz[l]-vz[lym])/(ymed(j)-ymed(j-1)) + (vy[l]-vy[lzm])/(zmed(k)-zmed(k-1))); //centered on lower, inner edge in x ("azimuthal")
#endif
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
