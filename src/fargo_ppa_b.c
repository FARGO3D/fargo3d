//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void VanLeerX_PPA_b_cpu(Field *Q){

//<USER_DEFINED>
  INPUT(Q);
  INPUT(Slope);
  OUTPUT(QL);
  OUTPUT(QR);
//<\USER_DEFINED>

//<EXTERNAL>
  real* slope = Slope->field_cpu;
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
  real temp;
//<\INTERNAL>

//<MAIN_LOOP>

  i = j = k = 0;

#ifdef Z
  for (k=0; k<size_z; k++) {
#endif
#ifdef Y
    for (j=0; j<size_y; j++) {
#endif
      for (i=0; i<size_x; i++)	{// Now we compute q_j+1/2
//<#>
	ll = l;
	llxp = lxp;
	
	temp = q[ll]+0.5*(q[llxp]-q[ll])-1.0/6.0*(slope[llxp]-slope[ll]);
	qR[ll] = temp;
	qL[llxp] = temp;
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
