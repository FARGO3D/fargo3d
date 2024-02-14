//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void SubStep3_cpu (real dt) {

//<USER_DEFINED>
  INPUT(Energy);

#ifdef BETACOOLING
  INPUT2D(Energy0);
  INPUT2D(OmegaOverBeta);
#endif
#ifdef X
  INPUT(Vx_temp);
#endif
#ifdef Y
  INPUT(Vy_temp);
#endif
#ifdef Z
  INPUT(Vz_temp);
#endif
  OUTPUT(Energy);
//<\USER_DEFINED>

//<EXTERNAL>
  real* e   = Energy->field_cpu;
#ifdef BETACOOLING
  real* e0   = Energy0->field_cpu;
  real* OoB = OmegaOverBeta->field_cpu;
#endif
#ifdef X
  real* vx  = Vx_temp->field_cpu;
#endif
#ifdef Y
  real* vy  = Vy_temp->field_cpu;
#endif
#ifdef Z
  real* vz  = Vz_temp->field_cpu;
#endif
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = XIP; 
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ-1;
//<\EXTERNAL>

//<INTERNAL>
  int i; //Variables reserved
  int j; //for the topology
  int k; //of the kernels
  int ll, ll2D;
#ifdef X
  int llxp;
#endif
#ifdef Y
  int llyp;
#endif
#ifdef Z
  int llzp;
#endif
  real term;
  real div_v;
  real temp_p;
//<\INTERNAL>
  
//<CONSTANT>
// real GAMMA(1);
// real Sxi(Nx);
// real Sxj(Ny+2*NGHY);
// real Syj(Ny+2*NGHY);
// real Szj(Ny+2*NGHY);
// real Sxk(Nz+2*NGHZ);
// real Syk(Nz+2*NGHZ);
// real Szk(Nz+2*NGHZ);
// real InvVj(Ny+2*NGHY);
//<\CONSTANT>

//<MAIN_LOOP>
  
  i = j = k = 0;
  
#ifdef Z
  for(k=0; k<size_z; k++) {
#endif
#ifdef Y
    for(j=0; j<size_y; j++) {
#endif
#ifdef X
      for(i=0; i<size_x; i++) {
#endif
//<#>

	ll = l;
  ll2D = l2D;
#ifdef X
	llxp = lxp;
#endif
#ifdef Y
	llyp = lyp;
#endif
#ifdef Z
	llzp = lzp;
#endif
	div_v = 0.0;
#ifdef X
	div_v += (vx[llxp]-vx[ll])*SurfX(j,k);
#endif
#ifdef Y
	div_v += (vy[llyp]*SurfY(i,j+1,k)-vy[ll]*SurfY(i,j,k));
#endif
#ifdef Z
	div_v += (vz[llzp]*SurfZ(i,j,k+1)-vz[ll]*SurfZ(i,j,k));
#endif
	term = 0.5 * dt * (GAMMA - 1.) * div_v * InvVol(i,j,k);
	e[ll] *= (1.0-term)/(1.0+term);

//beta cooling

#ifdef BETACOOLING
e[ll] = (e[ll] + OoB[ll2D] *dt * e0[ll2D])/(1.0 + OoB[ll2D]*dt);
#endif

  //end beta cooling
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