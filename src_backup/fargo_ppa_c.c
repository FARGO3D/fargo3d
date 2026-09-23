//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void VanLeerX_PPA_c_cpu(Field *Q){

//<USER_DEFINED>
  INPUT(Q);
  INPUT(QR);
  INPUT(QL);
  OUTPUT(QR);
  OUTPUT(QL);
//<\USER_DEFINED>

//<EXTERNAL>
  real* q  = Q->field_cpu;
  real* qL = QL->field_cpu;
  real* qR = QR->field_cpu;
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
  real diff;
  real cord;
//<\INTERNAL>

//<MAIN_LOOP>

  i = j = k = 0;

#ifdef Z
  for (k=0; k<size_z; k++) {
#endif
#ifdef Y
    for (j=0; j<size_y; j++) {
#endif
      for (i=0; i<size_x; i++) {
//<#>
	ll = l;

	if ((qR[ll]-q[ll])*(q[ll]-qL[ll]) < 0.0) {
	  qL[ll] = q[ll];
	  qR[ll] = q[ll];
	}
	diff = qR[ll] - qL[ll];
	cord = q[ll] - 0.5*(qL[ll]+qR[ll]);
	if (6.0*diff*cord > diff*diff)  /* do not simplify by diff !!! */
	  qL[ll] = 3.0*q[ll]-2.0*qR[ll];
	if (-diff*diff > 6.0*diff*cord) /* do not simplify by diff !!! */
	  qR[ll] = 3.0*q[ll]-2.0*qL[ll];
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
