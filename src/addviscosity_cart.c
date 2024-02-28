//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void addviscosity_cart_cpu(real dt) {

//<USER_DEFINED>
  INPUT(Density);
#if XDIM
  INPUT(Vx_temp);
  INPUT(Mmx);
  INPUT(Mpx);
  OUTPUT(Vx_temp);
#endif
#if YDIM
  INPUT(Vy_temp);
  INPUT(Mmy);
  INPUT(Mpy);
  OUTPUT(Vy_temp);
#endif
#if ZDIM
  INPUT(Vz_temp);
  INPUT(Mmz);
  INPUT(Mpz);
  OUTPUT(Vz_temp);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* rho = Density->field_cpu;
#if XDIM
  real* vx = Vx_temp->field_cpu;
#endif
#if YDIM
  real* vy = Vy_temp->field_cpu;
#endif
#if ZDIM
  real* vz = Vz_temp->field_cpu;
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
  int size_y = Ny+2*NGHY-2;
  int size_z = Nz+2*NGHZ-2;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+1);
// real InvDiffXmed(Nx+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
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

#if XDIM
  vx[l] += 2.0*(tauxx[l]-tauxx[lxm])/((rho[l]+rho[lxm]))*dt*Inv_zone_size_xmed(i,j,k);
#if (YDIM && XDIM)
	vx[l] += 2.0*(tauyx[lyp]-tauyx[l])/((ymin(j+1)-ymin(j))*(rho[lxm]+rho[l]))*dt;
#endif
#if (XDIM && ZDIM)
	vx[l] += 2.0*(tauxz[lzp]-tauxz[l])/((zmin(k+1)-zmin(k))*(rho[lxm]+rho[l]))*dt;
#endif
#endif

#if YDIM
	vy[l] += 2.0*(tauyy[l]-tauyy[lym])/((ymed(j)-ymed(j-1))*(rho[l]+rho[lym]))*dt;
#if (YDIM && XDIM)
  vy[l] += 2.0*(tauyx[lxp]-tauyx[l])/((xmin(i+1)-xmin(i))*(rho[l]+rho[lym]))*dt;
#endif
#if (ZDIM && YDIM)
	vy[l] += 2.0*(tauzy[lzp]-tauzy[l])/((zmin(k+1)-zmin(k))*(rho[l]+rho[lym]))*dt;
#endif
#endif

#if ZDIM
	vz[l] += 2.0*(tauzz[l]-tauzz[lzm])/((zmed(k)-zmed(k-1))*(rho[l]+rho[lzm]))*dt;
#if (ZDIM && XDIM)
  vz[l] += 2.0*(tauxz[lxp]-tauxz[l])/((xmin(i+1)-xmin(i))*(rho[l]+rho[lzm]))*dt;
#endif
#if (ZDIM && YDIM)
	vz[l] += 2.0*(tauzy[lyp]-tauzy[l])/((ymin(j+1)-ymin(j))*(rho[l]+rho[lzm]))*dt;
#endif
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
