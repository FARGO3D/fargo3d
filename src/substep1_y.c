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
#ifdef X
  INPUT(Vx);
#ifdef COLLISIONPREDICTOR
  INPUT(Vx_half);
#endif
#endif
#ifdef Y
  INPUT(Vy);
#ifdef COLLISIONPREDICTOR
  INPUT(Vy_half);
#endif
  OUTPUT(Vy_temp);
#endif
#ifdef Z
#ifdef COLLISIONPREDICTOR
  INPUT(Vz_half);
#endif
  INPUT(Vz);
#endif
#ifdef MHD
  INPUT(Bx);
  INPUT(Bz);
#endif
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
#ifdef Y
  real* vy      = Vy->field_cpu;
#ifdef COLLISIONPREDICTOR
  real* vy_half = Vy_half->field_cpu;
#else
  real* vy_half = Vy->field_cpu;
#endif
  real* vy_temp = Vy_temp->field_cpu;
#endif
#ifdef Z
  real* vz      = Vz->field_cpu;
#ifdef COLLISIONPREDICTOR
  real* vz_half = Vz_half->field_cpu;
#else
  real* vz_half = Vz->field_cpu;
#endif
  real* vz_temp = Vz_temp->field_cpu;
#endif
#ifdef MHD
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
#ifdef X
  int llxp;
#endif
#ifdef Y
  int llym;
  real dy_c;
  real dy_m;
  real wy_c;
  real wy_m;
#endif
#ifdef Z
  int llzp;
#endif
#ifdef MHD
  real db1;
  real db2;
  real bmeanm;
  real bmean;
  real bface;
#endif
#ifndef CARTESIAN
  real vphi;
  real vphi_c;
  real vphi_m;
#ifdef KEPLERIAN_MESH_WEIGHTS
  real s_c;
  real s_m;
  real s_e;
  real wk_c;
  real wk_m;
#endif
#endif
#ifdef POTENTIAL
#if !defined(NODEFAULTSTAR) && (defined(SPHERICAL) || (defined(CYLINDRICAL) && !defined(Z)))
  real pot_star_c;
  real pot_star_m;
#elif defined(KEPLERIAN_MESH_WEIGHTS) && !defined(NODEFAULTSTAR) && defined(CYLINDRICAL) && defined(Z)
  real pot_star_c;
  real pot_star_m;
  real r_star;
#endif
#endif
#ifdef SHEARINGBOX
  real vm;
#endif
#ifdef SPHERICAL
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
#ifdef Y
	ll = l;
#ifdef X
	llxp = lxp;
#endif //ENDIF X
	llym = lym;
#ifdef Z
	llzp = lzp;
#endif //ENDIF Z
	dtOVERrhom = 2.0*dt/(rho[ll]*(ymin(j+1)-ymin(j))+rho[llym]*(ymin(j)-ymin(j-1)))*(ymed(j)-ymed(j-1));
	dy_c = ymin(j+1) - ymin(j);
	dy_m = ymin(j) - ymin(j-1);
	wy_c = dy_m / (dy_c + dy_m);
	wy_m = dy_c / (dy_c + dy_m);

	vy_temp[ll] = vy[ll];
	if(fluidtype != DUST) vy_temp[ll]-=  dtOVERrhom*(p[ll]-p[llym])/(ymed(j)-ymed(j-1));
	
#ifdef CARTESIAN
#ifdef SHEARINGBOX

	vm = wy_c*0.5*(vx_half[ll]+vx_half[llxp]) + wy_m*0.5*(vx_half[llym]+vx_half[llxp-pitch]);	
	vy_temp[l] += dt*(2.0*OMEGAFRAME*vm + 2.0*SHEARPARAM*OMEGAFRAME*OMEGAFRAME*ymin(j));

#ifdef DRAGFORCE
	if (Fluidtype == GAS) {
	  vy_temp[ll] += 2*ASPECTRATIO*ASPECTRATIO*OMEGAFRAME*R0*dt;
	}
#endif
	
#endif //ENDIF SHEARINGBOX

#ifdef MHD
	if(fluidtype==GAS) {
#ifndef PASSIVEMHD
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
	  
#ifdef CYLINDRICAL
#ifdef KEPLERIAN_MESH_WEIGHTS
	vphi_c = 0.5*(vx_half[ll] + vx_half[llxp]) + ymed(j)*OMEGAFRAME;
	vphi_m = 0.5*(vx_half[llym] + vx_half[llxp-pitch]) + ymed(j-1)*OMEGAFRAME;
	s_c = 1.0/sqrt(ymed(j));
	s_m = 1.0/sqrt(ymed(j-1));
	s_e = 1.0/sqrt(ymin(j));
	wk_c = (s_e - s_m)/(s_c - s_m);
	wk_m = 1.0 - wk_c;
	vphi = wk_c*vphi_c + wk_m*vphi_m;
#else
	vphi_c = 0.5*(vx_half[ll] + vx_half[llxp]);
	vphi_m = 0.5*(vx_half[llym] + vx_half[llxp-pitch]);
	vphi = wy_c*vphi_c + wy_m*vphi_m + ymin(j)*OMEGAFRAME;
#endif
	vy_temp[ll] += vphi*vphi/ymin(j)*dt;
	
#ifdef MHD
	if(fluidtype==GAS) {
#ifndef PASSIVEMHD
	bmean  = 0.5*(bx[ll] + bx[llxp]);
	bmeanm = 0.5*(bx[llym] + bx[llxp-pitch]);
	bface = wy_c*bmean + wy_m*bmeanm;
	
	vy_temp[ll] -= dtOVERrhom*bface*bface/(MU0*ymin(j));
	
	db1 = (bmean*bmean-bmeanm*bmeanm);
	
	bmean  = 0.5*(bz[ll] + bz[llzp]);
	bmeanm = 0.5*(bz[llym] + bz[llym+stride]);
	
	db2 = (bmean*bmean-bmeanm*bmeanm); //-grad((bphi^2+bz^2)/2)
	
	vy_temp[ll] -= .5*dtOVERrhom*(db1 + db2)/((ymed(j)-ymed(j-1))*MU0);
	
#endif //END !PASSIVEMHD
	}
#endif //END MHD
#endif //END CYLINDRICAL

#ifdef SPHERICAL
#ifdef KEPLERIAN_MESH_WEIGHTS
	vphi_c = 0.5*(vx_half[ll] + vx_half[llxp]) + ymed(j)*sin(zmed(k))*OMEGAFRAME;
	vphi_m = 0.5*(vx_half[llym] + vx_half[llxp-pitch]) + ymed(j-1)*sin(zmed(k))*OMEGAFRAME;
	s_c = 1.0/sqrt(ymed(j));
	s_m = 1.0/sqrt(ymed(j-1));
	s_e = 1.0/sqrt(ymin(j));
	wk_c = (s_e - s_m)/(s_c - s_m);
	wk_m = 1.0 - wk_c;
	vphi = wk_c*vphi_c + wk_m*vphi_m;
#else
	vphi_c = 0.5*(vx_half[ll] + vx_half[llxp]);
	vphi_m = 0.5*(vx_half[llym] + vx_half[llxp-pitch]);
	vphi = wy_c*vphi_c + wy_m*vphi_m + ymin(j)*sin(zmed(k))*OMEGAFRAME;
#endif
	vzz = wy_c*0.5*(vz_half[ll] + vz_half[llzp]) + wy_m*0.5*(vz_half[llym] + vz_half[llzp-pitch]);
	vy_temp[ll] += (vphi*vphi + vzz*vzz)/ymin(j)*dt;
#ifdef MHD
	if(fluidtype==GAS) {
#ifndef PASSIVEMHD
	  bmean  = 0.5*(bx[ll] + bx[llxp]);
	  bmeanm = 0.5*(bx[llym] + bx[llxp-pitch]);
	  bface = wy_c*bmean + wy_m*bmeanm;
	  
	  vy_temp[ll] -= dtOVERrhom*bface*bface/(MU0*ymin(j));//Source term -B_phi^2/(mu_0\rho r)
	  
	  db1 = (bmean*bmean-bmeanm*bmeanm);
	  
	  bmean  = 0.5*(bz[ll] + bz[llzp]);
	  bmeanm = 0.5*(bz[llym] + bz[llym+stride]);
	  bface = wy_c*bmean + wy_m*bmeanm;
	  
	  vy_temp[ll] -= dtOVERrhom*bface*bface/(MU0*ymin(j));//Source term -B_\theta^2/(mu_0\rho r)
	  
	  db2 = (bmean*bmean-bmeanm*bmeanm); //-grad((bphi^2+btheta^2)/2)
	  
	  vy_temp[ll] -= .5*dtOVERrhom*(db1 + db2)/((ymed(j)-ymed(j-1))*MU0);
	  
#endif //END !PASSIVEMHD
	}
#endif //END MHD
#endif //END SPHERICAL

#ifdef POTENTIAL
#if !defined(NODEFAULTSTAR) && (defined(SPHERICAL) || (defined(CYLINDRICAL) && !defined(Z)))
	  pot_star_c = -G*MSTAR/ymed(j);
	  pot_star_m = -G*MSTAR/ymed(j-1);
	  vy_temp[ll] -= (((pot[ll]-pot_star_c)-(pot[llym]-pot_star_m))/(ymed(j)-ymed(j-1)) + G*MSTAR/(ymin(j)*ymin(j)))*dt;
#elif defined(KEPLERIAN_MESH_WEIGHTS) && !defined(NODEFAULTSTAR) && defined(CYLINDRICAL) && defined(Z)
	  pot_star_c = -G*MSTAR/sqrt(ymed(j)*ymed(j)+ZC*ZC);
	  pot_star_m = -G*MSTAR/sqrt(ymed(j-1)*ymed(j-1)+ZC*ZC);
	  r_star = sqrt(ymin(j)*ymin(j)+ZC*ZC);
	  vy_temp[ll] -= (((pot[ll]-pot_star_c)-(pot[llym]-pot_star_m))/(ymed(j)-ymed(j-1)) + G*MSTAR*ymin(j)/(r_star*r_star*r_star))*dt;
#else
	  vy_temp[ll] -= (pot[ll]-pot[llym])*dt/(ymed(j)-ymed(j-1));
#endif
#endif //ENDIF POTENTIAL
#endif //ENDIF Y


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
