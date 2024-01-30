//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void VanLeerX_PPA_d_2d_cpu(real dt, Field *Q, Field *Qs, Field2D *Vx_t){

//<USER_DEFINED>
  INPUT(Q);
  INPUT(QL);
  INPUT(QR);
  INPUT2D(Vx_t);
  OUTPUT(Qs);
//<\USER_DEFINED>

//<EXTERNAL>
  real* vx = Vx_t->field_cpu;
  real* q  = Q->field_cpu   ;
  real* qs = Qs->field_cpu  ;
  real* qL = QL->field_cpu  ;
  real* qR = QR->field_cpu  ;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
  int pitch2d = Pitch2D;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
  int l2d;
  int llxm;
  real ksi;
//<\INTERNAL>

//Parsed as copytosymbol, from a _d variable allocated on the gpu by users.
//<CONSTANT>
// real xmin(Nx+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;

#if ZDIM
  for (k=1; k<size_z; k++) {
#endif
#if YDIM
    for (j=1; j<size_y; j++) {
#endif
      for (i=XIM; i<size_x; i++) {
//<#>
	l2d = l2D; // Must go there instead of being out of the loop on 'i'
	ll = l;
	llxm = lxm;
	if (vx[l2d] > 0.0) {
	  ksi = vx[l2d]*dt/zone_size_x(i,j,k);
	  qs[ll] = qR[llxm]+ksi*(q[llxm]-qR[llxm]);
	  qs[ll]+= ksi*(1.0-ksi)*(2.0*q[llxm]-qR[llxm]-qL[llxm]);
	} else {
	  ksi = -vx[l2d]*dt/zone_size_x(i,j,k);
	  qs[ll] = qL[ll]+ksi*(q[ll]-qL[ll]);
	  qs[ll]+= ksi*(1.0-ksi)*(2.0*q[ll]-qR[ll]-qL[ll]);
	}
//<\#>
      }
#if YDIM
    }
#endif
#if ZDIM
  }
#endif
//<\MAIN_LOOP>
}
