//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void SubStep2_b_cpu (real dt) {

//<USER_DEFINED>
  INPUT(Density);
#ifdef X
#ifdef COLLISIONPREDICTOR
  INPUT(Vx_half);
#else
  INPUT(Vx);
#endif
  INPUT(Vx_temp);
  INPUT(Mpx);
  OUTPUT(Vx_temp);
#endif
#ifdef Y
#ifdef COLLISIONPREDICTOR
  INPUT(Vy_half);
#else
  INPUT(Vy);
#endif
  INPUT(Vy_temp);
  INPUT(Mpy);
  OUTPUT(Vy_temp);
#endif
#ifdef Z
#ifdef COLLISIONPREDICTOR
  INPUT(Vz_half);
#else
  INPUT(Vz);
#endif
  INPUT(Vz_temp);
  INPUT(Mpz);
  OUTPUT(Vz_temp);
#endif
#ifdef ADIABATIC
  INPUT(Energy);
  OUTPUT(Energy);
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
  real* vx_temp = Vx_temp->field_cpu;
  real* pres_x = Mpx->field_cpu;
#endif
#ifdef Y
#ifdef COLLISIONPREDICTOR
  real* vy = Vy_half->field_cpu;
#else
  real* vy = Vy->field_cpu;
#endif
  real* vy_temp = Vy_temp->field_cpu;
  real* pres_y = Mpy->field_cpu;
#endif
#ifdef Z
#ifdef COLLISIONPREDICTOR
  real* vz = Vz_half->field_cpu;
#else
  real* vz = Vz->field_cpu;
#endif
  real* vz_temp = Vz_temp->field_cpu;
  real* pres_z = Mpz->field_cpu;
#endif
#ifdef ADIABATIC
  real* e = Energy->field_cpu;
#endif  
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = XIP; 
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ-1;
  int fluidtype = Fluidtype;
//<\EXTERNAL>
  
//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
#ifdef X
  int llxm;
  int llxp;
  real dxmed;
  real dxrho1;
  real dxrho2;
#endif
#ifdef Y
  int llym;
  int llyp;
  real dyrho1;
  real dyrho2;
#endif
#ifdef Z
  int llzm;
  int llzp;
  real dzmed;
  real dzrho1;
  real dzrho2;
#endif
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real Sxi(Nx);
// real InvDiffXmed(Nx);
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
	ll = l;
#ifdef X
	llxm = lxm;
	llxp = lxp;
#endif
#ifdef Y
	llym = lym;
	llyp = lyp;
#endif
#ifdef Z
	llzp = lzp;
	llzm = lzm;
#endif
#ifdef X

	dxrho1 = Sxi(i);
	dxrho2 = Sxi(ixm);
	dxmed  = 0.5*( Sxi(i) + Sxi(ixm) );

	vx_temp[ll] += - 2.0*dt*dxmed*(pres_x[ll]-pres_x[llxm])/(rho[ll]*dxrho1+rho[llxm]*dxrho2)*Inv_zone_size_xmed(i,j,k);

#ifdef ADIABATIC
	e[ll] += -dt*(pres_x[ll]*(vx[llxp]-vx[ll])/zone_size_x(i,j,k));
#endif
#endif


#ifdef Y

	dyrho1 = ymin(j+1)-ymin(j);
	dyrho2 = ymin(j)-ymin(j-1);

	//ymed(j)-ymed(j-1) cancels out (see substep1_y.c)
	vy_temp[ll] += - 2.0*(pres_y[ll]-pres_y[llym])/(rho[ll]*dyrho1+rho[llym]*dyrho2)*dt;


#ifdef ADIABATIC
	e[ll] += -dt*(pres_y[ll]*(vy[llyp]-vy[ll])/zone_size_y(j,k));
#endif
#endif


#ifdef Z
	dzmed  = zmed(k)-zmed(k-1);
	dzrho1 = zmin(k+1)-zmin(k);
	dzrho2 = zmin(k)-zmin(k-1);


#ifdef SPHERICAL
	vz_temp[ll] += -2.0*dzmed*(pres_z[ll]-pres_z[llzm])/(rho[ll]*dzrho1+rho[llzm]*dzrho2)*dt/( ymed(j)*(zmed(k)-zmed(k-1)));
	
#else
	vz_temp[ll] += -2.0*(pres_z[ll]-pres_z[llzm])/(rho[ll]*dzrho1+rho[llzm]*dzrho2)*dt;
#endif

#ifdef ADIABATIC
	e[ll] += -dt*(pres_z[ll]*(vz[llzp]-vz[ll])/zone_size_z(j,k));
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
