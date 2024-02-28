//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void SubStep1_y_cpu (real dt) {

//<USER_DEFINED>
  INPUT(Pressure);
  INPUT(Density);
  INPUT(Pot);
#if XDIM
  INPUT(Vx);
#if COLLISIONPREDICTOR
  INPUT(Vx_half);
#endif
#endif
#if YDIM
  INPUT(Vy);
#if COLLISIONPREDICTOR
  INPUT(Vy_half);
#endif
  OUTPUT(Vy_temp);
#endif
#if ZDIM
#if COLLISIONPREDICTOR
  INPUT(Vz_half);
#endif
  INPUT(Vz);
#endif
#if MHD
  INPUT(Bx);
  INPUT(Bz);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* p   = Pressure->field_cpu;
  real* pot = Pot->field_cpu;
  real* rho = Density->field_cpu;
#if XDIM
  real* vx      = Vx->field_cpu;
#if COLLISIONPREDICTOR
  real* vx_half = Vx_half->field_cpu;
#else
  real* vx_half = Vx->field_cpu;
#endif
  real* vx_temp = Vx_temp->field_cpu;
#endif
#if YDIM
  real* vy      = Vy->field_cpu;
#if COLLISIONPREDICTOR
  real* vy_half = Vy_half->field_cpu;
#else
  real* vy_half = Vy->field_cpu;
#endif
  real* vy_temp = Vy_temp->field_cpu;
#endif
#if ZDIM
  real* vz      = Vz->field_cpu;
#if COLLISIONPREDICTOR
  real* vz_half = Vz_half->field_cpu;
#else
  real* vz_half = Vz->field_cpu;
#endif
  real* vz_temp = Vz_temp->field_cpu;
#endif
#if MHD
  real* bx = Bx->field_cpu;
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
#if XDIM
  int llxp;
#endif
#if YDIM
  int llym;
#endif
#if ZDIM
  int llzp;
#endif
#if MHD
  real db1;
  real db2;
  real bmeanm;
  real bmean;
#endif
#if (!CARTESIAN)
  real vphi;
#endif
#if SHEARINGBOX
  real vm;
#endif
#if SPHERICAL
  real vzz;
#endif
//<\INTERNAL>


//<CONSTANT>
// real xmin(Nx+2*NGHX+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real OMEGAFRAME(1);
// real OORTA(1);
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
#if YDIM
	ll = l;
#if XDIM
	llxp = lxp;
#endif //ENDIF X
	llym = lym;
#if ZDIM
	llzp = lzp;
#endif //ENDIF Z
	dtOVERrhom = 2.0*dt/(rho[ll]*(ymin(j+1)-ymin(j))+rho[llym]*(ymin(j)-ymin(j-1)))*(ymed(j)-ymed(j-1));

	vy_temp[ll] = vy[ll];
	if(fluidtype != DUST) vy_temp[ll]-=  dtOVERrhom*(p[ll]-p[llym])/(ymed(j)-ymed(j-1));

#if CARTESIAN
#if SHEARINGBOX

	vm = 0.25*(vx_half[ll]+vx_half[llxp]+vx_half[llym]+vx_half[llxp-pitch]);
	vy_temp[l] += dt*(2.0*OMEGAFRAME*vm + 2.0*SHEARPARAM*OMEGAFRAME*OMEGAFRAME*ymin(j));

#if DRAGFORCE
	if (Fluidtype == GAS) {
	  vy_temp[ll] += 2*ASPECTRATIO*ASPECTRATIO*OMEGAFRAME*R0*dt;
	}
#endif

#endif //ENDIF SHEARINGBOX

#if MHD
	if(fluidtype==GAS) {
#if (!PASSIVEMHD)
	  bmean  = 0.5*(bx[ll] + bx[llxp]);
	  bmeanm = 0.5*(bx[llym] + bx[llxp-pitch]);

	  db1 = (bmean*bmean-bmeanm*bmeanm);

	  bmean  = 0.5*(bz[ll] + bz[llzp]);
	  bmeanm = 0.5*(bz[llym] + bz[llym+stride]);

	  db2 = (bmean*bmean-bmeanm*bmeanm); //grad(b^2/2)

	  vy_temp[ll] -= .5*dtOVERrhom*(db1 + db2)/((ymed(j)-ymed(j-1))*MU0);

#endif //ENDIF !PASSIVEMHD
	}
#endif //ENDIF MHD
#endif //ENDIF CARTESIAN

#if CYLINDRICAL
	vphi = .25*(vx_half[ll] + vx_half[llxp] + vx_half[llym] + vx_half[llxp-pitch]);
	vphi += ymin(j)*OMEGAFRAME;
	vy_temp[ll] += vphi*vphi/ymin(j)*dt;

#if MHD
	if(fluidtype==GAS) {
#if (!PASSIVEMHD)
	bmean  = 0.5*(bx[ll] + bx[llxp]);
	bmeanm = 0.5*(bx[llym] + bx[llxp-pitch]);

	vy_temp[ll] -= dtOVERrhom*.25*(bmean+bmeanm)*(bmean+bmeanm)/(MU0*ymin(j));

	db1 = (bmean*bmean-bmeanm*bmeanm);

	bmean  = 0.5*(bz[ll] + bz[llzp]);
	bmeanm = 0.5*(bz[llym] + bz[llym+stride]);

	db2 = (bmean*bmean-bmeanm*bmeanm); //-grad((bphi^2+bz^2)/2)

	vy_temp[ll] -= .5*dtOVERrhom*(db1 + db2)/((ymed(j)-ymed(j-1))*MU0);

#endif //END !PASSIVEMHD
	}
#endif //END MHD
#endif //END CYLINDRICAL

#if SPHERICAL
	vphi =  .25*(vx_half[ll] + vx_half[llxp] + vx_half[llym] + vx_half[llxp-pitch]);
	vphi += ymin(j)*sin(zmed(k))*OMEGAFRAME;
	vzz = .25*(vz_half[ll] + vz_half[llzp]  + vz_half[llym] + vz_half[llzp-pitch]);
	vy_temp[ll] += (vphi*vphi + vzz*vzz)/ymin(j)*dt;
#if MHD
	if(fluidtype==GAS) {
#if (!PASSIVEMHD)
	  bmean  = 0.5*(bx[ll] + bx[llxp]);
	  bmeanm = 0.5*(bx[llym] + bx[llxp-pitch]);

	  vy_temp[ll] -= dtOVERrhom*.25*(bmean+bmeanm)*(bmean+bmeanm)/(MU0*ymin(j));//Source term -B_phi^2/(mu_0\rho r)

	  db1 = (bmean*bmean-bmeanm*bmeanm);

	  bmean  = 0.5*(bz[ll] + bz[llzp]);
	  bmeanm = 0.5*(bz[llym] + bz[llym+stride]);

	  vy_temp[ll] -= dtOVERrhom*.25*(bmean+bmeanm)*(bmean+bmeanm)/(MU0*ymin(j));//Source term -B_\theta^2/(mu_0\rho r)

	  db2 = (bmean*bmean-bmeanm*bmeanm); //-grad((bphi^2+btheta^2)/2)

	  vy_temp[ll] -= .5*dtOVERrhom*(db1 + db2)/((ymed(j)-ymed(j-1))*MU0);

#endif //END !PASSIVEMHD
	}
#endif //END MHD
#endif //END SPHERICAL

#if POTENTIAL
	  vy_temp[ll] -= (pot[ll]-pot[llym])*dt/(ymed(j)-ymed(j-1));
#endif //ENDIF POTENTIAL
#endif //ENDIF Y


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
