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
  real dx = Dx;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
//<\INTERNAL>

//<CONSTANT>
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
	vx[l] += 2.0*(tauxx[l]-tauxx[lxm])/(zone_size_x(j,k)*(rho[l]+rho[lxm]))*dt;
#if defined(Y) && defined(X)
	vx[l] += 2.0*(ymin(j+1)*ymin(j+1)*tauyx[lyp]-ymin(j)*ymin(j)*tauyx[l])/((ymin(j+1)-ymin(j))*ymed(j)*ymed(j)*(rho[lxm]+rho[l]))*dt;
	vx[l] += (tauyx[lyp]+tauyx[l])/((rho[lxm]+rho[l])*ymed(j))*dt;
#endif
#if defined(X) && defined(Z)
	vx[l] += 2.0*(tauxz[lzp]-tauxz[l])/(zone_size_z(j,k)*(rho[lxm]+rho[l]))*dt;
	vx[l] += 2.0*(tauxz[lzp]+tauxz[l])*cos(zmed(k))/(sin(zmed(k))*ymed(j)*(rho[lxm]+rho[l]))*dt;
#endif
#endif

#ifdef Y
	vy[l] += 2.0*(ymed(j)*ymed(j)*tauyy[l]-ymed(j-1)*ymed(j-1)*tauyy[lym])/((ymed(j)-ymed(j-1))*(rho[l]+rho[lym])*ymin(j)*ymin(j))*dt;
#if defined(Y) && defined(X)
	vy[l] += 2.0*(tauyx[lxp]-tauyx[l])/(dx*ymin(j)*sin(zmed(k))*(rho[l]+rho[lym]))*dt;
#endif
#if defined(X)
	vy[l] -= (tauxx[l]+tauxx[lym])/(ymin(j)*(rho[l]+rho[lym]))*dt;
#endif
#if defined(Z)
	vy[l] -= (tauzz[l]+tauzz[lym])/(ymin(j)*(rho[l]+rho[lym]))*dt;
#endif
#if defined(Z) && defined(Y)
	vy[l] += 2.0*(tauzy[lzp]*sin(zmin(k+1))-tauzy[l]*sin(zmin(k)))/((zmin(k+1)-zmin(k))*ymin(j)*sin(zmed(k))*(rho[l]+rho[lym]))*dt;
#endif
#endif

#ifdef Z
	vz[l] += 2.0*(tauzz[l]*sin(zmed(k))-tauzz[lzm]*sin(zmed(k-1)))/((zmed(k)-zmed(k-1))*ymed(j)*sin(zmin(k))*(rho[l]+rho[lzm]))*dt;
#if defined(Z) && defined(X)
	vz[l] += 2.0*(tauxz[lxp]-tauxz[l])/(zone_size_x(j,k)*(rho[l]+rho[lzm]))*dt;
#endif
#if defined(Z) && defined(Y)
	vz[l] += 2.0*(ymin(j+1)*ymin(j+1)*ymin(j+1)*tauzy[lyp]-ymin(j)*ymin(j)*ymin(j)*tauzy[l])/((ymin(j+1)-ymin(j))*ymed(j)*ymed(j)*ymed(j)*(rho[l]+rho[lzm]))*dt;
#endif
#if defined(X)
	vz[l] -= (tauxx[l]+tauxx[lzm])*cos(zmin(k))/(ymed(j)*sin(zmin(k))*(rho[l]+rho[lzm]))*dt;
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
