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
  INPUT(Density);
  INPUT2D(Density0);
  INPUT2D(Energy0);
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
  real* dens   = Density->field_cpu;
  real* e      = Energy->field_cpu;
  real* dens0   = Density0->field_cpu;
  real* e0      = Energy0->field_cpu;
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
  int pitch2d = Pitch2D;
  int stride = Stride_cpu;
  int size_x = XIP; 
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ-1;
//<\EXTERNAL>

//<INTERNAL>
  int i; //Variables reserved
  int j; //for the topology
  int k; //of the kernels
  int ll;
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
  real r, R, omega, torb, trelax;
//<\INTERNAL>
  
//<CONSTANT>
// real TCOOL(1);
// real GAMMA(1);
// real Sxj(Ny+2*NGHY);
// real Syj(Ny+2*NGHY);
// real Szj(Ny+2*NGHY);
// real Sxk(Nz+2*NGHZ);
// real Syk(Nz+2*NGHZ);
// real Szk(Nz+2*NGHZ);
// real InvVj(Ny+2*NGHY);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
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
	div_v += (vy[llyp]*SurfY(j+1,k)-vy[ll]*SurfY(j,k));
#endif
#ifdef Z
	div_v += (vz[llzp]*SurfZ(j,k+1)-vz[ll]*SurfZ(j,k));
#endif
        r = ymed(j);
	R = r;

#ifdef Z
        R = r*sin(zmed(k));
#endif
        omega = sqrt(1.0/(R*R*R));
//        torb = 2.*M_PI/omega;
        torb = 1.0/omega; // following Zhu+2015 notation
        trelax = max2(TCOOL*torb,dt);

//	term = 0.5 * dt * (GAMMA - 1.) * div_v * InvVol(j,k); // without thermal relaxation
        term = 0.5 * dt * (GAMMA - 1.) * div_v * InvVol(j,k) + 0.5*dt/trelax; // with thermal relaxation

//	e[ll] *= (1.0-term)/(1.0+term); // without thermal relaxation
        e[ll] = (1.0-term)/(1.0+term)*e[ll] + dens[ll]/dens0[l2D]*e0[l2D]*dt/trelax/(1.0+term); // with thermal relaxation

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
