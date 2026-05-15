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
  real dx = Dx;
//<\EXTERNAL>
  
//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
#ifdef X
  int llxm;
  int llxp;
#endif
#ifdef Y
  int llym;
  int llyp;
#endif
#ifdef Z
  int llzm;
  int llzp;
#endif
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+1);
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
	vx_temp[ll] += - 2.0*(pres_x[ll]-pres_x[llxm])/(rho[ll]+rho[llxm])* \
	  dt/zone_size_x(j,k);//Should be distance from center to center
#ifdef ADIABATIC
	e[ll] += -dt*(pres_x[ll]*(vx[llxp]-vx[ll])/zone_size_x(j,k));
#endif
#endif
#ifdef Y
	vy_temp[ll] += - 2.0*(pres_y[ll]-pres_y[llym])/(rho[ll]+rho[llym])*	\
	  dt/zone_size_y(j,k);// instead of zone_size_(x,y,z)
#ifdef ADIABATIC
	e[ll] += -dt*(pres_y[ll]*(vy[llyp]-vy[ll])/zone_size_y(j,k));
#endif
#endif
#ifdef Z
	vz_temp[ll] += -2.0*(pres_z[ll]-pres_z[llzm])/(rho[ll]+rho[llzm])*	\
	  dt/zone_size_z(j,k);// which is the distance from edge to edge
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
