#include "fargo3d.h"

void cfl_b(){
  real step;
  int i,j,k;

  Min[FluidIndex] = reduction_full_MIN(DensStar, NGHY, Ny+NGHY, NGHZ, Nz+NGHZ);

#if XDIM
    real* vxmed = VxMed->field_cpu;
    real shearlimit;

#if (!STANDARD)
    INPUT2D (VxMed);
    i = j = k = 0;
#if ZDIM
    for (k=NGHZ; k<Nz+NGHZ; k++) {
#endif
#if YDIM
      for (j=NGHY; j<Ny+NGHY; j++) {
#endif
#if CARTESIAN
	shearlimit = CFL*Mindx / fabs(vxmed[l2D]-vxmed[l2D+1]);
#endif
#if (CYLINDRICAL || SPHERICAL)
	shearlimit = CFL*Mindx / fabs(vxmed[l2D]/Ymed(j)-vxmed[l2D+1]/Ymed(j+1));
#endif
	if (shearlimit < Min[FluidIndex]) {
	  Min[FluidIndex] = shearlimit;
	}
#if YDIM
      }
#endif
#if ZDIM
    }
#endif
#endif
#endif


}
