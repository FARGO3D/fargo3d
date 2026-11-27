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
#ifdef COLLISIONPREDICTOR
  INPUT(Vx_half);
#else
  INPUT(Vx);
#endif
  OUTPUT(Mmx);
  OUTPUT(Mpx);
#endif
#ifdef Y
#ifdef COLLISIONPREDICTOR
  INPUT(Vy_half);
#else
  INPUT(Vy);
#endif
  OUTPUT(Mmy);
  OUTPUT(Mpy);
#endif
#ifdef Z
#ifdef COLLISIONPREDICTOR
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
#ifdef X
#ifdef COLLISIONPREDICTOR
  real* vx = Vx_half->field_cpu;
#else
  real* vx = Vx->field_cpu;
#endif
#endif
#ifdef Y
#ifdef COLLISIONPREDICTOR
  real* vy = Vy_half->field_cpu;
#else
  real* vy = Vy->field_cpu;
#endif
#endif
#ifdef Z
#ifdef COLLISIONPREDICTOR
  real* vz = Vz_half->field_cpu;
#else
  real* vz = Vz->field_cpu;
#endif
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
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  real div_v;
#ifdef X
  real dx_c;
  real dx_m;
  real wx_c;
  real wx_m;
#endif
#ifdef Y
  real dy_c;
  real dy_m;
  real wy_c;
  real wy_m;
#endif
#ifdef Z
  real dz_c;
  real dz_m;
  real wz_c;
  real wz_m;
#endif
#if defined(X) && defined(Z)
  real rho_xz;
#endif
#if defined(Y) && defined(X)
  real rho_xy;
#endif
#if defined(Z) && defined(Y)
  real rho_yz;
#endif
//<\INTERNAL>

//<CONSTANT>
// real NU(1);
// real Sxi(Nx+2*NGHX);
// real Sxj(Ny+2*NGHY);
// real Syj(Ny+2*NGHY);
// real Szj(Ny+2*NGHY);
// real Sxk(Nz+2*NGHZ);
// real Syk(Nz+2*NGHZ);
// real Szk(Nz+2*NGHZ);
// real xmin(Nx+2*NGHX+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real InvVj(Ny+2*NGHY);
// real InvDiffXmed(Nx+2*NGHX);
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
	div_v += (vy[lyp]*SurfY(i,j+1,k)-vy[l]*SurfY(i,j,k));
#endif
#ifdef Z
	div_v += (vz[lzp]*SurfZ(i,j,k+1)-vz[l]*SurfZ(i,j,k));
#endif
	div_v *= 2.0/3.0*InvVol(i,j,k);

#ifdef X
	tauxx[l] = NU*rho[l]*(2.0*(vx[lxp]-vx[l])/(xmin(i+1)-xmin(i)) - div_v);
#endif
#ifdef Y
	tauyy[l] = NU*rho[l]*(2.0*(vy[lyp]-vy[l])/(ymin(j+1)-ymin(j)) - div_v);
#endif
#ifdef Z
	tauzz[l] = NU*rho[l]*(2.0*(vz[lzp]-vz[l])/(zmin(k+1)-zmin(k)) - div_v);
#endif

#ifdef X
	dx_c = Sxi(i);
	dx_m = Sxi(ixm);
	wx_c = dx_m/(dx_c + dx_m);
	wx_m = dx_c/(dx_c + dx_m);
#endif
#ifdef Y
	dy_c = ymin(j+1) - ymin(j);
	dy_m = ymin(j) - ymin(j-1);
	wy_c = dy_m/(dy_c + dy_m);
	wy_m = dy_c/(dy_c + dy_m);
#endif
#ifdef Z
	dz_c = zmin(k+1) - zmin(k);
	dz_m = zmin(k) - zmin(k-1);
	wz_c = dz_m/(dz_c + dz_m);
	wz_m = dz_c/(dz_c + dz_m);
#endif

#if defined(X) && defined(Z)
	rho_xz = wx_c*(wz_c*rho[l] + wz_m*rho[lzm]) + wx_m*(wz_c*rho[lxm] + wz_m*rho[lxm-stride]);
	tauxz[l] = NU*rho_xz*((vx[l]-vx[lzm])/(zmed(k)-zmed(k-1)) + (vz[l]-vz[lxm])*Inv_zone_size_xmed(i,j,k)); //centered on lower, left "radial" edge in y
#endif

#if defined(Y) && defined(X)
	rho_xy = wx_c*(wy_c*rho[l] + wy_m*rho[lym]) + wx_m*(wy_c*rho[lxm] + wy_m*rho[lxm-pitch]);
	tauyx[l] = NU*rho_xy*((vy[l]-vy[lxm])*Inv_zone_size_xmed(i,j,k) + (vx[l]-vx[lym])/(ymed(j)-ymed(j-1))); //centered on left, inner vertical edge in z
#endif

#if defined(Z) && defined(Y)
	rho_yz = wy_c*(wz_c*rho[l] + wz_m*rho[lzm]) + wy_m*(wz_c*rho[lym] + wz_m*rho[lym-stride]);
	tauzy[l] = NU*rho_yz*((vz[l]-vz[lym])/(ymed(j)-ymed(j-1)) + (vy[l]-vy[lzm])/(zmed(k)-zmed(k-1))); //centered on lower, inner edge in x ("azimuthal")
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
