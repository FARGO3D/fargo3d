//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void mon_momx_cpu () {

//<USER_DEFINED>
  INPUT(Density);
  INPUT(Vx);
  OUTPUT(Slope);
//<\USER_DEFINED>


//<EXTERNAL>
  real* dens = Density->field_cpu;
  real* vx   = Vx->field_cpu;
  real* mom  = Slope->field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = XIP; 
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
  real rcyl;
//<\INTERNAL>

//<CONSTANT>
// real Syk(Nz+2*NGHZ);
// real InvVj(Ny+2*NGHY);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real OMEGAFRAME(1);
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
#ifdef CARTESIAN
	mom[ll] = dens[ll]*.5*(vx[ll]+vx[lxp])*Vol(j,k);
#ifdef SHEARINGBOX //SHEARINGBOX is an option for the CARTESIAN geometry
	mom[ll] += 2.0*OMEGAFRAME*ymed(j)*dens[ll]*Vol(j,k);
#endif
#endif

#ifdef CYLINDRICAL
	mom[ll] = (.5*(vx[ll]+vx[lxp])+ymed(j)*OMEGAFRAME)*ymed(j)*dens[ll]*Vol(j,k);
#endif

#ifdef SPHERICAL
	rcyl = ymed(j) * sin(zmed(k));
	mom[ll] = (.5*(vx[ll]+vx[lxp]) + rcyl*OMEGAFRAME)*rcyl*dens[ll]*Vol(j,k);
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
