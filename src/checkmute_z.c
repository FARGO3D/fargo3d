//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void CheckMuteZ_cpu() {
  /* Throughout the source, a direction Y or Z is said mute if it is
     defined (the preprocessor variable Y or Z exists, so that VY or
     VZ exists, for instance), but the corresponding number of zones
     is 1 (NY=1 or NZ=1). In that special case the ghosts are filled
     manually with the values of the unique active row or column, in
     this function. */

//<USER_DEFINED>
  INPUT(Density);
#if ADIABATIC
  INPUT(Energy);
#endif
#if XDIM
  INPUT(Vx);
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
  OUTPUT(Density);
#if ADIABATIC
  OUTPUT(Energy);
#endif
#if XDIM
  OUTPUT(Vx);
#endif
#if YDIM
  OUTPUT(Vy);
#endif
#if ZDIM
  OUTPUT(Vz);
#endif
#if MHD
  OUTPUT(Bx);
  OUTPUT(By);
  OUTPUT(Bz);
#endif
//<\USER_DEFINED>

//<INTERNAL>
  int i;
  int j;
  int k;
  int l_act;
  int l_up;
  int ll;
//<\INTERNAL>

//<EXTERNAL>
  real* rho = Density->field_cpu;
#if XDIM  
  real* vx = Vx->field_cpu;
#endif
#if YDIM
  real* vy = Vy->field_cpu;
#endif
#if ZDIM
  real* vz = Vz->field_cpu;
#endif
#if ADIABATIC
  real* energy = Energy->field_cpu;
#endif
#if MHD
  real* bx = Bx->field_cpu;
  real* by = By->field_cpu;
  real* bz = Bz->field_cpu;
#endif
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = NGHZ;
//<\EXTERNAL>

//<MAIN_LOOP>

  i = j = k = 0;

#if ZDIM
  for (k = 0; k < size_z; k++) {
#endif
#if YDIM
    for (j = 0; j < size_y; j++) {
#endif
#if XDIM
      for (i = 0; i < size_x; i++) {
#endif
//<#>
	ll = l;
	l_act = i+j*pitch+NGHZ*stride;
	l_up  = i+j*pitch+(k+NGHZ+1)*stride;
	
	rho[ll]    = rho[l_act];
	rho[l_up]  = rho[l_act];
#if ZDIM
	vz[ll]     = vz[l_act];
	vz[l_up]   = vz[l_act];
#endif
#if XDIM
	vx[ll]     = vx[l_act];
	vx[l_up]   = vx[l_act];
#endif	 
#if YDIM
	vy[ll]     = vy[l_act];
	vy[l_up]   = vy[l_act];
#endif	 
#if ADIABATIC
	energy[ll]   = energy[l_act];
	energy[l_up] = energy[l_act];
#endif
#if MHD
	bx[ll]     = bx[l_act];
	bx[l_up]   = bx[l_act];
	by[ll]     = by[l_act];
	by[l_up]   = by[l_act];
	bz[ll]     = bz[l_act];
	bz[l_up]   = bz[l_act];
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
