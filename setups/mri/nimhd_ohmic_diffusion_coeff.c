//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void OhmicDiffusion_coeff_cpu() {

//<USER_DEFINED>
  OUTPUT(EtaOhm);
  static int already_filled = FALSE;
  if (already_filled == TRUE) return;
//<\USER_DEFINED>

//<EXTERNAL>
  real* eta  = EtaOhm->field_cpu;
  real etao  = OHMICDIFFUSIONCOEFF;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
  real rsup  = YMAX;
  real rinf  = YMIN;
  real width = (YMAX-YMIN)/7.;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
//<\INTERNAL>

//<CONSTANT>
// real ymin(Ny+2*NGHY+1);
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
	if (ymed(j) > rsup-width) {
	  eta[l] = etao*(ymed(j)-(rsup-width))/width;
	}
	if (ymed(j) < rinf+width) {
	  eta[l] = etao*((rinf+width)-ymed(j))/width/5.0;
	}
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

//<LAST_BLOCK>
  already_filled = TRUE;
//<\LAST_BLOCK>
}
