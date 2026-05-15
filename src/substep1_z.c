//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void SubStep1_z_cpu (real dt) {

//<USER_DEFINED>
  INPUT(Pressure);
  INPUT(Density);
  INPUT(Pot);
#ifdef X
  INPUT(Vx);
#ifdef COLLISIONPREDICTOR
  INPUT(Vx_half);
#endif
#endif
#ifdef Z
  INPUT(Vz);
#endif
#ifdef MHD
  INPUT(Bx);
  INPUT(By);
#ifdef SPHERICAL
  INPUT(Bz);
#endif
#endif  
  OUTPUT(Vz_temp);
//<\USER_DEFINED>

//<EXTERNAL>
  real* p   = Pressure->field_cpu;
  real* pot = Pot->field_cpu;
  real* rho = Density->field_cpu;
#ifdef X
  real* vx      = Vx->field_cpu;
#ifdef COLLISIONPREDICTOR
  real* vx_half = Vx_half->field_cpu;
#else
  real* vx_half = Vx->field_cpu;
#endif
  real* vx_temp = Vx_temp->field_cpu;
#endif
#ifdef Z
  real* vz      = Vz->field_cpu;
  real* vz_temp = Vz_temp->field_cpu;
#endif
#ifdef MHD
  real* bx = Bx->field_cpu;
  real* by = By->field_cpu;
#ifdef SPHERICAL
  real* bz = Bz->field_cpu;
#endif
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
#ifdef X
  int llxp;
#endif
#ifdef Y
  int llyp;
#endif
#ifdef Z
  int llzm;
#endif
#ifdef MHD
  real db1;
  real db2;
  real bmeanm;
  real bmean;
#endif
  real vphi;
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real OMEGAFRAME(1);
// real VERTICALDAMPING(1);
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
#ifdef Z
	ll = l;
#ifdef X
	llxp = lxp;
#endif //END X
#ifdef Y
	llyp = lyp;
#endif //END Y
	llzm = lzm;

	dtOVERrhom = 2.0*dt/(rho[ll]+rho[llzm]);

#ifdef CARTESIAN
 	vz_temp[ll] = vz[ll] - 
	  dtOVERrhom*(p[ll]-p[llzm])/(zmed(k)-zmed(k-1));
	vz_temp[ll] /= (1.+VERTICALDAMPING*dt);

#ifdef POTENTIAL
	vz_temp[ll] -= (pot[ll]-pot[llzm]) * dt / (zmed(k)-zmed(k-1));
#endif //END POTENTIAL

#ifdef MHD
	if(fluidtype==GAS) {
	  bmean  = 0.5*(bx[ll] + bx[llxp]);
	  bmeanm = 0.5*(bx[llzm] + bx[llxp-stride]);
	  db1 = (bmean*bmean-bmeanm*bmeanm);
	  
	  bmean = 0.5*(by[ll] + by[llyp]);
	  bmeanm  = 0.5*(by[llzm] + by[llzm+pitch]);
	  
	  db2 = (bmean*bmean-bmeanm*bmeanm);
	  
	  vz_temp[ll] -= .5*dtOVERrhom*(db1 + db2)/(MU0*(zmed(k)-zmed(k-1)));
	}
#endif //END MHD
#endif //END CARTESIAN

#ifdef CYLINDRICAL
 	vz_temp[ll] = vz[ll] -
	  dtOVERrhom*(p[ll]-p[llzm])/(zmed(k)-zmed(k-1));
	vz_temp[ll] /= (1.+VERTICALDAMPING*dt);

#ifdef POTENTIAL
	vz_temp[ll] -= (pot[ll]-pot[llzm]) * dt / (zmed(k)-zmed(k-1));
#endif //END POTENTIAL
	
#ifdef MHD
	if(fluidtype==GAS) {
#ifndef PASSIVEMHD
	  bmean  = 0.5*(bx[ll] + bx[llxp]);
	  bmeanm = 0.5*(bx[llzm] + bx[llxp-stride]);
	  db1 = (bmean*bmean-bmeanm*bmeanm);
	  
	  bmean = 0.5*(by[ll] + by[llyp]);
	  bmeanm  = 0.5*(by[llzm] + by[llzm+pitch]);
	  
	  db2 = (bmean*bmean-bmeanm*bmeanm);
	  
	  vz_temp[l] -= (db1 + db2)/(rho[l]+rho[lzm])*dt/((zmed(k)-zmed(k-1))*MU0);
#endif //END PASSIVEMHD
	}
#endif //END MHD
#endif //END CYLINDRICAL
	  
#ifdef SPHERICAL
	 vphi = .25*(vx_half[ll] + vx_half[llxp] + vx_half[llzm] + vx_half[llxp-stride]);
	vphi += ymed(j)*sin(zmin(k))*OMEGAFRAME;
 	vz_temp[ll] = vz[ll] - 
	  dtOVERrhom*(p[ll]-p[llzm])/(ymed(j)*(zmed(k)-zmed(k-1)));
	vz_temp[ll] += vphi*vphi*cos(zmin(k))/(sin(zmin(k))*ymed(j))*dt;
	vz_temp[ll] /= (1.+VERTICALDAMPING*dt);

#ifdef POTENTIAL
	vz_temp[ll] -= (pot[ll]-pot[llzm]) * dt / (ymed(j)*(zmed(k)-zmed(k-1)));
#endif //END POTENTIAL

#ifdef MHD
	if(fluidtype==GAS) {
#ifndef PASSIVEMHD
	  bmean  = 0.5*(bx[ll] + bx[llxp]);
	  bmeanm = 0.5*(bx[llzm] + bx[llxp-stride]);
	  db1 = (bmean*bmean-bmeanm*bmeanm);
	  
	  vz_temp[ll] -= (dtOVERrhom*.25*(bmean+bmeanm)*(bmean+bmeanm)	\
			  *cos(zmin(k))/sin(zmin(k))/(MU0*ymed(j)));
	  /* Above: geometric source term -B_\phi^2*cot(theta)/(mu0*rho*r) */
	
	  bmean = 0.5*(by[ll] + by[llyp]);
	  bmeanm  = 0.5*(by[llzm] + by[llzm+pitch]);
	  
	  db2 = (bmean*bmean-bmeanm*bmeanm);
	  
	  vz_temp[ll] += dtOVERrhom*bz[ll]*.5*(bmean+bmeanm)/(MU0*ymed(j));
	  /* Above: geometric source term +B_r * B_\theta/(MU0*rho*r) */

	  vz_temp[ll] -= .5*dtOVERrhom*(db1 + db2)/(ymed(j)*(zmed(k)-zmed(k-1))*MU0);
#endif //END !PASSIVEMHD
	}
#endif //END MHD
#endif //END SPHERICAL
#endif //END Z
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
