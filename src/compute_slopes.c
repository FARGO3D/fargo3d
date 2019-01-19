//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>



void ComputeSlopes_cpu(int idx, int idy, int idz, Field *f, Field *s) {

  /*
    Function that computes the slopes that will be used in the MOC.
    The indices idx, idy and idz represent the direction in which the
    slope is calculated. f is the input field, s is the output
    slope. Generally in the MHD we need the slopes perpendicular to
    the direction of the field (ComputeSlope (vx, 0,0,1) and
    ComputeSlope (vx, 0,1,0)). The slope has same centering as the
    input field. Van Leer's slopes are calculated in this routine.
   */
  
//<USER_DEFINED>
  INPUT(f);
  OUTPUT(s);
//<\USER_DEFINED>

//<EXTERNAL>
  real* field = f->field_cpu;
  real* slope = s->field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = XIP;
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ-1;
  real dx = Dx;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int lp;
  int lm;
  int ll;
  real dfp;
  real dfm;
  real delta;
//<\INTERNAL>

//<CONSTANT>
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
//<\CONSTANT>

//<MAIN_LOOP>

  for (k=1; k<size_z; k++) {
    for (j=1; j<size_y; j++) {
      for (i=XIM; i<size_x; i++) {
//<#>
	ll = l;

	delta = (zone_size_x(j,k)*idx +		
		 zone_size_y(j,k)*idy +       
		 zone_size_z(j,k)*idz);	
	
	lp = lxp*idx + lyp*idy + lzp*idz;
	lm = lxm*idx + lym*idy + lzm*idz;
	
	dfp = field[lp]-field[ll];
	dfm = field[ll]-field[lm];
	if(dfp*dfm<=0.0)
	  slope[ll] = 0.0;
	else
	  slope[ll] =2.0*dfp*dfm/((dfp+dfm)*delta);
//<\#>
      }
    }
  }
//<\MAIN_LOOP>
}
