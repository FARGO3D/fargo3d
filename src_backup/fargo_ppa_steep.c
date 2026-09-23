//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
// The values below are those defined by Colella and Woodward 1984,
// JCP, 54, 174 In the present implementation it is recommended to use
// simple PPA advection (this is the default) and not to activate the
// steepening solution implemented in this file (this would be
// obtained by adding: FARGOP_OPT += -DPPA_STEEPENER in the .opt
// file). Indeed the default 3 values below have been found to yield
// spurious small scale vortices in some high resolution setups.
#define ETA1 20.0
#define ETA2 0.05
#define EPS  0.01

//<\INCLUDES>

void VanLeerX_PPA_steep_cpu(Field *Q){

//<USER_DEFINED>
  INPUT(Q);
  INPUT(Slope);
  INPUT(LapPPA);
  OUTPUT(QL);
  OUTPUT(QR);
//<\USER_DEFINED>

//<EXTERNAL>
  real* slope = Slope->field_cpu;
  real* lapla = LapPPA->field_cpu;
  real* q = Q->field_cpu;
  real* qL = QL->field_cpu;
  real* qR = QR->field_cpu;
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
  int llxp;
  int llxm;
  real eta;
  real etatilde;
  real aL;
  real aR;
//<\INTERNAL>

//<MAIN_LOOP>

  i = j = k = 0;

#ifdef Z
  for (k=0; k<size_z; k++) {
#endif
#ifdef Y
    for (j=0; j<size_y; j++) {
#endif
      for (i=XIM; i<size_x; i++)	{// Now we compute q_j+1/2
//<#>
	ll = l;
	llxp = lxp;
	llxm = lxm;

	if (lapla[llxp]*lapla[llxm] < 0.0) {
	  if ((fabs(q[llxp]-q[llxm]) > EPS*fabs(q[llxp])) &&		\
	      (fabs(q[llxp]-q[llxm]) > EPS*fabs(q[llxm]))) {  /* Do we have a discontinuity ? */
	    etatilde = - (lapla[llxp]-lapla[llxm])/(q[llxp]-q[llxm]);
	    eta = ETA1*(etatilde-ETA2);
	    if (eta > 1.0) eta=1.0;
	    if (eta < 0.0) eta=0.0;
	    aL = q[llxm]+.5*slope[llxm];
	    aR = q[llxp]-.5*slope[llxp];
	    qL[ll] = qL[ll]*(1.0-eta) + aL*eta;
	    qR[ll] = qR[ll]*(1.0-eta) + aR*eta;
	  }
	}
	  
//<\#>
      }
#ifdef Y
    }
#endif
#ifdef Z
  }
#endif
//<\MAIN_LOOP>
}
