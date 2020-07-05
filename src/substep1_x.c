//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void SubStep1_x_cpu (real dt) {


//<USER_DEFINED>
  INPUT(Pressure);
  INPUT(Density);
  INPUT(Pot);
  INPUT(Vx);
#ifdef MHD
#if defined (CYLINDRICAL) || defined (SPHERICAL)
  INPUT(Bx);
#endif
  INPUT(By);
  INPUT(Bz);
#endif  
  OUTPUT(Vx_temp);
//<\USER_DEFINED>

//<EXTERNAL>
  real* p   = Pressure->field_cpu;
  real* pot = Pot->field_cpu;
  real* rho = Density->field_cpu;
#ifdef X
  real* vx      = Vx->field_cpu;
  real* vx_temp = Vx_temp->field_cpu;
#endif
#ifdef MHD
  real* bx = Bx->field_cpu;
  real* by = By->field_cpu;
  real* bz = Bz->field_cpu;
#endif
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = XIP; 
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ-1;
  real dx = Dx;
  int fluidtype = Fluidtype;
//<\EXTERNAL>
  
//<INTERNAL>
  int i; //Variables reserved
  int j; //for the topology
  int k; //of the kernels
  int ll;
  real dtOVERrhom;
#ifdef X
  int llxm;
#endif
#ifdef Y
  int llyp;
#endif
#ifdef Z
  int llzp;
#endif
#ifdef MHD
  real db1;
  real db2;
  real bmeanm;
  real bmean;
#if defined (CYLINDRICAL)|| defined (SPHERICAL)
  real brmean;
#endif
#ifdef SPHERICAL
  real btmean;
#endif
#endif
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
	ll = l;
	llxm = lxm;
#ifdef Y
	llyp = lyp;
#endif
#ifdef Z
	llzp = lzp;
#endif
	dtOVERrhom=2.0*dt/(rho[ll]+rho[llxm]);
	vx_temp[ll] = vx[ll] - 
	  dtOVERrhom*(p[ll]-p[llxm])/zone_size_x(j,k);
#ifdef POTENTIAL
	vx_temp[ll] -= (pot[ll]-pot[llxm])*dt/zone_size_x(j,k);
#endif

#ifdef MHD
	if(fluidtype == GAS) {

#ifndef PASSIVEMHD
	  
	  bmean  = 0.5*(by[ll] + by[llyp]);
	  bmeanm = 0.5*(by[llxm] + by[llxm+pitch]);
	  
	  db1 = (bmean*bmean-bmeanm*bmeanm);
	  
#if defined(CYLINDRICAL) || defined(SPHERICAL)
	  brmean = .5*(bmean+bmeanm);
	  vx_temp[ll] += dtOVERrhom*brmean*bx[ll]/(MU0*ymed(j));
#endif
	  
	  bmean  = 0.5*(bz[ll] + bz[llzp]);
	  bmeanm = 0.5*(bz[llxm] + bz[llxm+stride]);
	  
	  db2 = (bmean*bmean-bmeanm*bmeanm);
	  
	  vx_temp[ll] -= .5*dtOVERrhom*(db1 + db2)/(MU0*zone_size_x(j,k));
	  
#ifdef SPHERICAL
	btmean = .5*(bmean+bmeanm);
	vx_temp[ll] += dtOVERrhom*btmean*cos(zmed(k))*bx[ll]/(MU0*ymed(j)*sin(zmed(k)));
#endif
      
#endif
      }
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
