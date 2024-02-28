//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void cfl_cpu() {

//<USER_DEFINED>
  INPUT(Energy);
  INPUT(Density);
  OUTPUT(DensStar);
#if XDIM
  INPUT(Vx);
  INPUT2D(VxMed);
#endif
#if YDIM
  INPUT(Vy);
#endif
#if ZDIM
  INPUT(Vz);
#endif
#if MHD
  INPUT(Bx);
  INPUT(By);
  INPUT(Bz);
#endif
#if HALLEFFECT
  INPUT(EtaHall);
#endif
#if AMBIPOLARDIFFUSION
  INPUT(EtaAD);
#endif
#if OHMICDIFFUSION
  INPUT(EtaOhm);
#endif


//<\USER_DEFINED>

//<EXTERNAL>
  real* cs  = Energy->field_cpu;
  real* rho = Density->field_cpu;
  real* dtime = DensStar->field_cpu;
#if XDIM
  real* vx = Vx->field_cpu;
  real* vxmed = VxMed->field_cpu;
#endif
#if YDIM
  real* vy = Vy->field_cpu;
#endif
#if ZDIM
  real* vz = Vz->field_cpu;
#endif
#if MHD
  real* bx = Bx->field_cpu;
  real* by = By->field_cpu;
  real* bz = Bz->field_cpu;
#endif
#if MHD
#if OHMICDIFFUSION
  real* etao = EtaOhm->field_cpu;
#endif
#if HALLEFFECT
  real* etahall = EtaHall->field_cpu;
#endif
#if AMBIPOLARDIFFUSION
  real* etaad   = EtaAD->field_cpu;
#endif
#endif
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+NGHX;
  int size_y = Ny+NGHY;
  int size_z = Nz+NGHZ;
  int pitch2d = Pitch2D;
  int fluidtype = Fluidtype;
//<\EXTERNAL>

//<INTERNAL>
  real __attribute__((unused))dtmin = 1e30;
  int i;
  int j;
  int k;
  int ll;
  int llxp;
  int llyp;
  int llzp;
  real cfl1_a=0.0;
  real cfl1_b=0.0;
  real cfl1_c=0.0;
  real cfl1=0.0;
  real cfl2=0.0;
  real cfl3=0.0;
  real cfl4=0.0;
  real cfl5_a=0.0;
  real cfl5_b=0.0;
  real cfl5_c=0.0;
  real cfl5=0.0;
  real cfl6=0.0;
  real cfl7_a=0.0;
  real cfl7_b=0.0;
  real cfl7_c=0.0;
  real cfl7=0.0;
  real cfl8=0.0;
  real cfl9=0.0;
  real cfl10=0.0;
  real b;
  real vxx, vxxp;
  real soundspeed;
  real soundspeed2;
  real viscosity;
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+2*NGHX+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real GAMMA(1);
// real CFL(1);
// real ALPHA(1);
// real NU(1);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;

#if ZDIM
  for (k=NGHZ; k<size_z; k++) {
#endif
#if YDIM
    for (j=NGHY; j<size_y; j++) {
#endif
#if XDIM
      for (i=NGHX; i<size_x; i++) {
#endif
//<#>
	ll = l;
	llxp = lxp;
	llyp = lyp;
	llzp = lzp;
#if XDIM
#if STANDARD
	vxx = vx[ll];
	vxxp= vx[llxp];
#else
	vxx = vx[ll] - vxmed[l2D];
	vxxp= vx[llxp] - vxmed[l2D];
#endif
#endif

#if ISOTHERMAL
	soundspeed2 = cs[ll]*cs[ll];
#endif


#if ALPHAVISCOSITY
#if ISOTHERMAL
	viscosity = ALPHA*cs[l]*cs[l]*sqrt(ymed(j)*ymed(j)*ymed(j)/(G*MSTAR));
#else //ADIABATIC
	viscosity = ALPHA*GAMMA*(GAMMA-1.0)*cs[l]/rho[l]*sqrt(ymed(j)*ymed(j)*ymed(j)/(G*MSTAR)); //cs means internal energy.
#endif //END ISOTHERMAL
#else  //NU VISCOSITY
	viscosity = NU;
#endif //END ALPHAVISCOSITY

#if ADIABATIC
	soundspeed2 = GAMMA*(GAMMA-1)*cs[ll]/rho[ll];
#endif

#if POLYTROPIC
	soundspeed2 = GAMMA*cs[ll]*pow(rho[ll],GAMMA-1.0); //In polytropic setups we use cs[] to store the entropy
#endif

#if MHD
	if (fluidtype == GAS) {
	  soundspeed2 += ((bx[ll]*bx[ll]+by[ll]*by[ll]+bz[ll]*bz[ll])/(MU0*rho[ll]));
	}
#endif

	soundspeed = sqrt(soundspeed2);

#if XDIM
	cfl1_a = soundspeed/zone_size_x(i,j,k);
#endif
#if YDIM
	cfl1_b = soundspeed/zone_size_y(j,k);
#endif
#if ZDIM
	cfl1_c = soundspeed/zone_size_z(j,k);
#endif
	cfl1 = max3(cfl1_a, cfl1_b, cfl1_c);


#if XDIM
	cfl2 = (max2(fabs(vxx),fabs(vxxp)))/zone_size_x(i,j,k);
#endif
#if YDIM
	cfl3 = (max2(fabs(vy[ll]),fabs(vy[llyp])))/zone_size_y(j,k);
#endif
#if ZDIM
	cfl4 = (max2(fabs(vz[ll]),fabs(vz[llzp])))/zone_size_z(j,k);
#endif

#if (!NOSUBSTEP2)
#if XDIM
	cfl5_a = fabs(vx[llxp]-vx[ll])/zone_size_x(i,j,k);
#endif
#if YDIM
	cfl5_b = fabs(vy[llyp]-vy[ll])/zone_size_y(j,k);
#endif
#if ZDIM
	cfl5_c = fabs(vz[llzp]-vz[ll])/zone_size_z(j,k);
#endif
	cfl5 = max3(cfl5_a, cfl5_b, cfl5_c)*4.0*CVNR;
#endif

#if STRONG_SHOCK
	cfl6 = cfl5/CVNR*CVNL;
#endif

#if XDIM
	cfl7_a = 1.0/zone_size_x(i,j,k);
#endif
#if YDIM
	cfl7_b = 1.0/zone_size_y(j,k);
#endif
#if ZDIM
	cfl7_c = 1.0/zone_size_z(j,k);
#endif
	cfl7 = 4.0*viscosity*pow(max3(cfl7_a,cfl7_b,cfl7_c),2);

#if MHD
#if OHMICDIFFUSION
	cfl8 = 4.0*etao[ll]*pow(max3(cfl7_a,cfl7_b,cfl7_c),2);
#endif
#if HALLEFFECT
	cfl9 = 6.0*fabs(etahall[ll])*pow(max3(cfl7_a,cfl7_b,cfl7_c),2);
#endif
#if AMBIPOLARDIFFUSION
	cfl10 = 4.0*etaad[ll]*pow(max3(cfl7_a,cfl7_b,cfl7_c),2);
#endif
#endif

	dtime[ll] = CFL/sqrt(cfl1*cfl1 + cfl2*cfl2 +
			     cfl3*cfl3 + cfl4*cfl4 +
			     cfl5*cfl5 + cfl6*cfl6 +
			     cfl7*cfl7 + cfl8*cfl8 +
			     cfl9*cfl9 + cfl10*cfl10 );

//<\#>
#if XDIM
      }
#endif
#if YDIM
    }
#endif
#if ZDIM
  }
//<\MAIN_LOOP>
#endif
//<LAST_BLOCK>
  cfl_b();
//<\LAST_BLOCK>

}
