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
#if MHD
#if (CYLINDRICAL || SPHERICAL)
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
#if XDIM
  real* vx      = Vx->field_cpu;
  real* vx_temp = Vx_temp->field_cpu;
#endif
#if MHD
  real* bx = Bx->field_cpu;
  real* by = By->field_cpu;
  real* bz = Bz->field_cpu;
#endif
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = XIP;
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ-1;
  int fluidtype = Fluidtype;
//<\EXTERNAL>

//<INTERNAL>
  int i; //Variables reserved
  int j; //for the topology
  int k; //of the kernels
  int ll;
  real dtOVERrhom;
  real dxmed;
#if XDIM
  int llxm;
#endif
#if YDIM
  int llyp;
#endif
#if ZDIM
  int llzp;
#endif
#if MHD
  real db1;
  real db2;
  real bmeanm;
  real bmean;
#if defined (CYLINDRICAL)|| defined (SPHERICAL)
  real brmean;
#endif
#if SPHERICAL
  real btmean;
#endif
#endif
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+2*NGHX+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real InvDiffXmed(Nx+2*NGHX);
// real Sxi(Nx+2*NGHX);
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
	ll = l;
	llxm = lxm;
#if YDIM
	llyp = lyp;
#endif
#if ZDIM
	llzp = lzp;
#endif

	dxmed = 0.5*( Sxi(i) + Sxi(ixm) );
	dtOVERrhom = 2.*dt/( rho[ll]*Sxi(i) + rho[llxm]*Sxi(ixm) )*dxmed;

	vx_temp[ll] = vx[ll];
	if(fluidtype != DUST) vx_temp[ll] -=  dtOVERrhom*(p[ll]-p[llxm])*Inv_zone_size_xmed(i,j,k);

#if POTENTIAL
	vx_temp[ll] -= (pot[ll]-pot[llxm])*dt*Inv_zone_size_xmed(i,j,k);
#endif

#if MHD
	if(fluidtype == GAS) {

#if (!PASSIVEMHD)

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

	  vx_temp[ll] -= .5*dtOVERrhom*(db1 + db2)*Inv_zone_size_xmed(i,j,k)/MU0;

#if SPHERICAL
	btmean = .5*(bmean+bmeanm);
	vx_temp[ll] += dtOVERrhom*btmean*cos(zmed(k))*bx[ll]/(MU0*ymed(j)*sin(zmed(k)));
#endif

#endif
      }
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
