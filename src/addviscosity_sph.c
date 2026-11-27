//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void addviscosity_sph_cpu(real dt) {

//<USER_DEFINED>
  INPUT(Density);
#ifdef X
  INPUT(Vx_temp);
  INPUT(Mmx);
  INPUT(Mpx);
  OUTPUT(Vx_temp);
#endif
#ifdef Y
  INPUT(Vy_temp);
  INPUT(Mmy);
  INPUT(Mpy);
  OUTPUT(Vy_temp);
#endif
#ifdef Z
  INPUT(Vz_temp);
  INPUT(Mmz);
  INPUT(Mpz);
  OUTPUT(Vz_temp);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* rho = Density->field_cpu;
#ifdef X
  real* vx = Vx_temp->field_cpu;
#endif
#ifdef Y
  real* vy = Vy_temp->field_cpu;
#endif
#ifdef Z
  real* vz = Vz_temp->field_cpu;
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
  int size_y = Ny+2*NGHY-2;
  int size_z = Nz+2*NGHZ-2;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
#ifdef X
  real dx_c;
  real dx_m;
  real inv_rhox_cons;
#endif
#ifdef Y
  real dy_c;
  real dy_m;
  real wy_c;
  real wy_m;
  real inv_rhoy_cons;
#endif
#ifdef Z
  real dz_c;
  real dz_m;
  real wz_c;
  real wz_m;
  real inv_rhoz_cons;
#endif
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+2*NGHX+1);
// real InvDiffXmed(Nx+2*NGHX);
// real Sxi(Nx+2*NGHX);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
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

#ifdef X
	dx_c = Sxi(i);
	dx_m = Sxi(ixm);
	inv_rhox_cons = (dx_c + dx_m)/(rho[l]*dx_c + rho[lxm]*dx_m);
	vx[l] += (tauxx[l]-tauxx[lxm])*Inv_zone_size_xmed(i,j,k)*inv_rhox_cons*dt;
#if defined(Y) && defined(X)
	vx[l] += (ymin(j+1)*ymin(j+1)*tauyx[lyp]-ymin(j)*ymin(j)*tauyx[l])/((ymin(j+1)-ymin(j))*ymed(j)*ymed(j))*inv_rhox_cons*dt;
	vx[l] += 0.5*(tauyx[lyp]+tauyx[l])/ymed(j)*inv_rhox_cons*dt;
#endif
#if defined(X) && defined(Z)
	vx[l] += (tauxz[lzp]-tauxz[l])/zone_size_z(j,k)*inv_rhox_cons*dt;
	vx[l] += (tauxz[lzp]+tauxz[l])*cos(zmed(k))/(sin(zmed(k))*ymed(j))*inv_rhox_cons*dt;
#endif
#endif

#ifdef Y
	dy_c = ymin(j+1) - ymin(j);
	dy_m = ymin(j) - ymin(j-1);
	inv_rhoy_cons = (dy_c + dy_m)/(rho[l]*dy_c + rho[lym]*dy_m);
	wy_c = dy_m / (dy_c + dy_m);
	wy_m = dy_c / (dy_c + dy_m);
	vy[l] += (ymed(j)*ymed(j)*tauyy[l]-ymed(j-1)*ymed(j-1)*tauyy[lym])/((ymed(j)-ymed(j-1))*ymin(j)*ymin(j))*inv_rhoy_cons*dt;
#if defined(Y) && defined(X)
	vy[l] += (tauyx[lxp]-tauyx[l])/((xmin(i+1)-xmin(i))*ymin(j)*sin(zmed(k)))*inv_rhoy_cons*dt;
#endif
#if defined(X)
	vy[l] -= (wy_c*tauxx[l]+wy_m*tauxx[lym])/ymin(j)*inv_rhoy_cons*dt;
#endif
#if defined(Z)
	vy[l] -= (wy_c*tauzz[l]+wy_m*tauzz[lym])/ymin(j)*inv_rhoy_cons*dt;
#endif
#if defined(Z) && defined(Y)
	vy[l] += (tauzy[lzp]*sin(zmin(k+1))-tauzy[l]*sin(zmin(k)))/((zmin(k+1)-zmin(k))*ymin(j)*sin(zmed(k)))*inv_rhoy_cons*dt;
#endif
#endif

#ifdef Z
	dz_c = zmin(k+1) - zmin(k);
	dz_m = zmin(k) - zmin(k-1);
	inv_rhoz_cons = (dz_c + dz_m)/(rho[l]*dz_c + rho[lzm]*dz_m);
	wz_c = dz_m / (dz_c + dz_m);
	wz_m = dz_c / (dz_c + dz_m);
	vz[l] += (tauzz[l]*sin(zmed(k))-tauzz[lzm]*sin(zmed(k-1)))/((zmed(k)-zmed(k-1))*ymed(j)*sin(zmin(k)))*inv_rhoz_cons*dt;
#if defined(Z) && defined(X)
	vz[l] += (tauxz[lxp]-tauxz[l])/edge_size_x_middley_lowz(i,j,k)*inv_rhoz_cons*dt;
#endif
#if defined(Z) && defined(Y)
	vz[l] += (ymin(j+1)*ymin(j+1)*ymin(j+1)*tauzy[lyp]-ymin(j)*ymin(j)*ymin(j)*tauzy[l])/((ymin(j+1)-ymin(j))*ymed(j)*ymed(j)*ymed(j))*inv_rhoz_cons*dt;
#endif
#if defined(X)
	vz[l] -= (wz_c*tauxx[l]+wz_m*tauxx[lzm])*cos(zmin(k))/(ymed(j)*sin(zmin(k)))*inv_rhoz_cons*dt;
#endif
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
