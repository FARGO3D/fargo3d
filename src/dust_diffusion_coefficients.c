//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void DustDiffusion_Coefficients_cpu() {

//<USER_DEFINED>
#ifdef ALPHAVISCOSITY
  INPUT(Energy);
#ifdef ADIABATIC
  INPUT(Density);
#endif
#endif
  OUTPUT(Sdiffyczc);
  OUTPUT(Sdiffyfzc);
#ifdef Z
  OUTPUT(Sdiffyczf);
  OUTPUT(Sdiffyfzf);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* sdiff_yfzc = Sdiffyfzc->field_cpu;
  real* sdiff_yczc = Sdiffyczc->field_cpu;
#ifdef Z
  real* sdiff_yczf = Sdiffyczf->field_cpu;
  real* sdiff_yfzf = Sdiffyfzf->field_cpu;
#endif
#ifdef ALPHAVISCOSITY  
#ifdef ISOTHERMAL
  real* cs = Fluids[0]->Energy->field_cpu;
#endif
#ifdef ADIABATIC
  real* e = Fluids[0]->Energy->field_cpu;
  real* rhog = Fluids[0]->Density->field_cpu;
  real gamma = GAMMA;
#endif
  real alphavisc = ALPHA;
#else
  real nu = NU;
#endif
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
//<\EXTERNAL>
  
//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
  int llym;
  int llzm;
#ifdef ALPHAVISCOSITY
  real r3yczc;
  real r3yfzc;
  real soundspeed2;
  real soundspeedf2;
  real soundspeedfz2;
#ifdef Y
  real dy_c;
  real dy_m;
  real wy_c;
  real wy_m;
#endif
#ifdef Z
  real dz_c;
  real dz_m;
  real wz_c;
  real wz_m;
#endif
#ifdef ISOTHERMAL
  real cs_f;
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
  for (k=0; k<size_z; k++) {
#endif
#ifdef Y
    for (j=0; j<size_y; j++) {
#endif
#ifdef X
      for (i=0; i<size_x; i++ ) {
#endif
//<#>
	ll = l;
#ifdef Y
	llym = lym;
#endif
#ifdef Z
	llzm = lzm;
#endif
#ifdef ALPHAVISCOSITY
#ifdef ISOTHERMAL
	soundspeed2 = cs[ll]*cs[ll];
#ifdef Y
	if(j==0)
	  soundspeedf2 = soundspeed2;
	else {
	  dy_c = ymin(j+1) - ymin(j);
	  dy_m = ymin(j) - ymin(j-1);
	  wy_c = dy_m/(dy_c + dy_m);
	  wy_m = dy_c/(dy_c + dy_m);
	  cs_f = wy_c*cs[ll] + wy_m*cs[llym];
	  soundspeedf2 = cs_f*cs_f;
	}
#else
	soundspeedf2 = soundspeed2;
#endif
#ifdef Z
	if(k==0)
          soundspeedfz2 = soundspeed2;
        else {
          dz_c = zmin(k+1) - zmin(k);
          dz_m = zmin(k) - zmin(k-1);
          wz_c = dz_m/(dz_c + dz_m);
          wz_m = dz_c/(dz_c + dz_m);
          cs_f = wz_c*cs[ll] + wz_m*cs[llzm];
          soundspeedfz2 = cs_f*cs_f;
        }
#endif //Z
#endif
#ifdef ADIABATIC
	soundspeed2 = gamma*(gamma-1.0)*e[ll]/rhog[ll];
#ifdef Y
	if(j==0)
	  soundspeedf2 = soundspeed2;
	else {
	  dy_c = ymin(j+1) - ymin(j);
	  dy_m = ymin(j) - ymin(j-1);
	  wy_c = dy_m/(dy_c + dy_m);
	  wy_m = dy_c/(dy_c + dy_m);
	  soundspeedf2 = gamma*(gamma-1.0)*(wy_c*e[ll]+wy_m*e[llym])/(wy_c*rhog[ll]+wy_m*rhog[llym]);
	}
#else
	soundspeedf2 = soundspeed2;
#endif
#ifdef Z
	if(k==0)
	  soundspeedfz2 = soundspeed2;
	else {
	  dz_c = zmin(k+1) - zmin(k);
	  dz_m = zmin(k) - zmin(k-1);
	  wz_c = dz_m/(dz_c + dz_m);
	  wz_m = dz_c/(dz_c + dz_m);
	  soundspeedfz2 = gamma*(gamma-1.0)*(wz_c*e[ll]+wz_m*e[llzm])/(wz_c*rhog[ll]+wz_m*rhog[llzm]);
	}
#endif //Z
#endif
	r3yczc = ymed(j)*ymed(j)*ymed(j);
	r3yfzc = ymin(j)*ymin(j)*ymin(j);

	sdiff_yczc[ll] = alphavisc*soundspeed2/sqrt(G*MSTAR/r3yczc);
	sdiff_yfzc[ll] = alphavisc*soundspeedf2/sqrt(G*MSTAR/r3yfzc);
#ifdef Z
	sdiff_yczf[ll] = alphavisc*soundspeedfz2/sqrt(G*MSTAR/r3yczc);
	sdiff_yfzf[ll] = alphavisc*soundspeedfz2/sqrt(G*MSTAR/r3yfzc);
#endif //Z
#endif
#ifdef VISCOSITY
	sdiff_yczc[ll] = nu;
	sdiff_yfzc[ll] = nu;
#ifdef Z
	sdiff_yczf[ll] = nu;
	sdiff_yfzf[ll] = nu;
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
