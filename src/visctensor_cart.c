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
#if XDIM
#if COLLISIONPREDICTOR
  INPUT(Vx_half);
#else
  INPUT(Vx);
#endif
  OUTPUT(Mmx);
  OUTPUT(Mpx);
#endif
#if YDIM
#if COLLISIONPREDICTOR
  INPUT(Vy_half);
#else
  INPUT(Vy);
#endif
  OUTPUT(Mmy);
  OUTPUT(Mpy);
#endif
#if ZDIM
#if COLLISIONPREDICTOR
  INPUT(Vz_half);
#else
  INPUT(Vz);
#endif
  OUTPUT(Mmz);
  OUTPUT(Mpz);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* rho = Density->field_cpu;
#if XDIM
#if COLLISIONPREDICTOR
  real* vx = Vx_half->field_cpu;
#else
  real* vx = Vx->field_cpu;
#endif
#endif
#if YDIM
#if COLLISIONPREDICTOR
  real* vy = Vy_half->field_cpu;
#else
  real* vy = Vy->field_cpu;
#endif
#endif
#if ZDIM
#if COLLISIONPREDICTOR
  real* vz = Vz_half->field_cpu;
#else
  real* vz = Vz->field_cpu;
#endif
#endif
#if XDIM
  real* tauxx = Mmx->field_cpu;
#endif
#if YDIM
  real* tauyy = Mmy->field_cpu;
#endif
#if ZDIM
  real* tauzz = Mmz->field_cpu;
#endif
#if (XDIM && ZDIM)
  real* tauxz = Mpx->field_cpu;
#endif
#if (YDIM && XDIM)
  real* tauyx = Mpy->field_cpu;
#endif
#if (ZDIM && YDIM)
  real* tauzy = Mpz->field_cpu;
#endif
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = XIP;
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ-1;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  real div_v;
//<\INTERNAL>

//<CONSTANT>
// real NU(1);
// real Sxi(Nx);
// real Sxj(Ny+2*NGHY);
// real Syj(Ny+2*NGHY);
// real Szj(Ny+2*NGHY);
// real Sxk(Nz+2*NGHZ);
// real Syk(Nz+2*NGHZ);
// real Szk(Nz+2*NGHZ);
// real xmin(Nx+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real InvVj(Ny+2*NGHY);
// real InvDiffXmed(Nx);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;

#if ZDIM
  for(k=1; k<size_z; k++) {
#endif
#if YDIM
    for(j=1; j<size_y; j++) {
#endif
#if XDIM
      for(i=XIM; i<size_x; i++) {
#endif
//<#>
//Evaluate centered divergence.
	div_v = 0.0;
#if XDIM
	div_v += (vx[lxp]-vx[l])*SurfX(j,k);
#endif
#if YDIM
	div_v += (vy[lyp]*SurfY(i,j+1,k)-vy[l]*SurfY(i,j,k));
#endif
#if ZDIM
	div_v += (vz[lzp]*SurfZ(i,j,k+1)-vz[l]*SurfZ(i,j,k));
#endif
	div_v *= 2.0/3.0*InvVol(i,j,k);

#if XDIM
	tauxx[l] = NU*rho[l]*(2.0*(vx[lxp]-vx[l])/(xmin(i+1)-xmin(i)) - div_v);
#endif
#if YDIM
	tauyy[l] = NU*rho[l]*(2.0*(vy[lyp]-vy[l])/(ymin(j+1)-ymin(j)) - div_v);
#endif
#if ZDIM
	tauzz[l] = NU*rho[l]*(2.0*(vz[lzp]-vz[l])/(zmin(k+1)-zmin(k)) - div_v);
#endif

#if (XDIM && ZDIM)
	tauxz[l] = NU*.25*(rho[l]+rho[lzm]+rho[lxm]+rho[lxm-stride])*((vx[l]-vx[lzm])/(zmed(k)-zmed(k-1)) + (vz[l]-vz[lxm])*Inv_zone_size_xmed(i,j,k)); //centered on lower, left "radial" edge in y
#endif

#if (YDIM && XDIM)
	tauyx[l] = NU*.25*(rho[l]+rho[lxm]+rho[lym]+rho[lxm-pitch])*((vy[l]-vy[lxm])*Inv_zone_size_xmed(i,j,k) + (vx[l]-vx[lym])/(ymed(j)-ymed(j-1))); //centered on left, inner vertical edge in z
#endif

#if (ZDIM && YDIM)
	tauzy[l] = NU*.25*(rho[l]+rho[lym]+rho[lzm]+rho[lym-stride])*((vz[l]-vz[lym])/(ymed(j)-ymed(j-1)) + (vy[l]-vy[lzm])/(zmed(k)-zmed(k-1))); //centered on lower, inner edge in x ("azimuthal")
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
