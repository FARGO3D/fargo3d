//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void DustDiffusion_Coefficients_cpu() {

//<USER_DEFINED>
#if ALPHAVISCOSITY
  INPUT(Energy);
#if ADIABATIC
  INPUT(Density);
#endif
#endif
  OUTPUT(Sdiffyczc);
  OUTPUT(Sdiffyfzc);
#if ZDIM
  OUTPUT(Sdiffyczf);
  OUTPUT(Sdiffyfzf);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* sdiff_yfzc = Sdiffyfzc->field_cpu;
  real* sdiff_yczc = Sdiffyczc->field_cpu;
#if ZDIM
  real* sdiff_yczf = Sdiffyczf->field_cpu;
  real* sdiff_yfzf = Sdiffyfzf->field_cpu;
#endif
#if ALPHAVISCOSITY
#if ISOTHERMAL
  real* cs = Fluids[0]->Energy->field_cpu;
#endif
#if ADIABATIC
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
#if ALPHAVISCOSITY
  real r3yczc;
  real r3yfzc;
  real soundspeed2;
  real soundspeedf2;
  real soundspeedfz2;
#endif
//<\INTERNAL>

//<CONSTANT>
// real ymin(Ny+2*NGHY+1);
//<\CONSTANT>

//<MAIN_LOOP>
  i = j = k = 0;
#if ZDIM
  for (k=0; k<size_z; k++) {
#endif
#if YDIM
    for (j=0; j<size_y; j++) {
#endif
#if XDIM
      for (i=0; i<size_x; i++ ) {
#endif
//<#>
	ll = l;
#if YDIM
	llym = lym;
#endif
#if ZDIM
	llzm = lzm;
#endif
#if ALPHAVISCOSITY
#if ISOTHERMAL
	soundspeed2 = cs[ll]*cs[ll];
	if(j==0)
	  soundspeedf2 = soundspeed2;
	else
	  soundspeedf2 = 0.5*(cs[ll]+cs[llym])*0.5*(cs[ll]+cs[llym]);
#if ZDIM
	if(k==0)
          soundspeedfz2 = soundspeed2;
        else
          soundspeedfz2 = 0.5*(cs[ll]+cs[llzm])*0.5*(cs[ll]+cs[llzm]);
#endif //Z
#endif
#if ADIABATIC
	soundspeed2 = gamma*(gamma-1.0)*e[ll]/rhog[ll];
	if(j==0)
	  soundspeedf2 = soundspeed2;
	else
	  soundspeedf2 = gamma*(gamma-1.0)*(e[ll]+e[llym])/(rhog[ll]+rhog[llym]);
#endif
	r3yczc = ymed(j)*ymed(j)*ymed(j);
	r3yfzc = ymin(j)*ymin(j)*ymin(j);

	sdiff_yczc[ll] = alphavisc*soundspeed2/sqrt(G*MSTAR/r3yczc);
	sdiff_yfzc[ll] = alphavisc*soundspeedf2/sqrt(G*MSTAR/r3yfzc);
#if ZDIM
	sdiff_yczf[ll] = alphavisc*soundspeedfz2/sqrt(G*MSTAR/r3yczc);
	sdiff_yfzf[ll] = alphavisc*soundspeedfz2/sqrt(G*MSTAR/r3yfzc);
#endif //Z
#endif
#if VISCOSITY
	sdiff_yczc[ll] = nu;
	sdiff_yfzc[ll] = nu;
#if ZDIM
	sdiff_yczf[ll] = nu;
	sdiff_yfzf[ll] = nu;
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
