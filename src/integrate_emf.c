//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

/* Orbital advection (FARGO) for MHD */

/* We exclusively evaluate the electric field in the y and z
direction, as there is no electric field along the direction of motion */

void EMF_Upstream_Integrate_cpu (real dt) {

//<USER_DEFINED>
  INPUT (Emfx);
  INPUT (Emfy);
  INPUT (Emfz);
  INPUT (By);
  INPUT (Bz);
  INPUT2D (Vxhyr);
  INPUT2D (Vxhzr);
  INPUT2D (Vxhzr);
  INPUT2DINT (Nxhy);
  INPUT2DINT (Nxhz);
  OUTPUT (Emfx);
  OUTPUT (Emfy);
  OUTPUT (Emfz);
//<\USER_DEFINED>

//<INTERNAL>
  int i;
  int j;
  int k;
  int m;
  int l_plus_m;
  int ll;
  int ll2D;
  int ll2D_int;
//<\INTERNAL>

//<EXTERNAL>
  real* emfx = Emfx->field_cpu;
  real* emfy = Emfy->field_cpu;
  real* emfz = Emfz->field_cpu;
  real* by = By->field_cpu;
  real* bz = Bz->field_cpu;
  real* vxhyr = Vxhyr->field_cpu;
  real* vxhzr = Vxhzr->field_cpu;
  int* nxhy = Nxhy->field_cpu;
  int* nxhz = Nxhz->field_cpu;
  int pitch  = Pitch_cpu;
  int pitch2d_int = Pitch_Int_gpu;
  int pitch2d = Pitch2D;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
  real dx = Dx;
  int nx = Nx; 
//<\EXTERNAL>

//<CONSTANT>
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
//<\CONSTANT>

//<MAIN_LOOP>
  for (k = 1; k < size_z; k++) {
    for (j = 1; j < size_y; j++) {
      for (i = 0; i < size_x; i++) {
//<#>
	ll = l;
	ll2D = l2D;
	ll2D_int = l2D_int;
	emfx[ll] = 0.0;
	emfz[ll] *= -vxhyr[ll2D]*dt;
	emfy[ll] *= vxhzr[ll2D]*dt;
	if (nxhy[ll2D_int] > 0) {
	  for (m=0; m < nxhy[ll2D_int]; m++) {
	    l_plus_m = ll+m;
	    if (i+m >= nx+NGHX) l_plus_m -= nx;
	    if (i+m <  NGHX)    l_plus_m += nx;
	    emfz[ll] += -by[l_plus_m]*edge_size_x_middlez_lowy(j,k);
	  }
	} else {
	  for (m=-1; m >= nxhy[ll2D_int]; m--) {
	    l_plus_m = ll+m;
	    if (i+m >= nx+NGHX) l_plus_m -= nx;
	    if (i+m <  NGHX)    l_plus_m += nx;
	    emfz[ll] += by[l_plus_m]*edge_size_x_middlez_lowy(j,k);
	  }
	}
	if (nxhz[ll2D_int] > 0) {
	  for (m=0; m < nxhz[ll2D_int]; m++) {
	    l_plus_m = ll+m;
	    if (i+m >= nx+NGHX) l_plus_m -= nx;
	    if (i+m <  NGHX)    l_plus_m += nx;
	    emfy[ll] += bz[l_plus_m]*edge_size_x_middley_lowz(j,k);
	  }
	} else {
	  for (m=-1; m >= nxhz[ll2D_int]; m--) {
	    l_plus_m = ll+m;
	    if (i+m >= nx+NGHX) l_plus_m -= nx;
	    if (i+m <  NGHX)    l_plus_m += nx;
	    emfy[ll] += -bz[l_plus_m]*edge_size_x_middley_lowz(j,k);
	  }
	}
//<\#>
      }
    }
  }
//<\MAIN_LOOP>
}
