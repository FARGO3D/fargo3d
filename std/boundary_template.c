//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
#define LEFT  0
#define RIGHT 1
#define DOWN  2
#define UP    3
//<\INCLUDES>

void boundary_%side () {

//<USER_DEFINED>
  %ifields;
  %ofields;
//<\USER_DEFINED>

//<INTERNAL>
  int __attribute__((unused))i;
  int __attribute__((unused))j;
  int __attribute__((unused))k;
  int __attribute__((unused))jact;
  int __attribute__((unused))jgh;
  int __attribute__((unused))kact;
  int __attribute__((unused))kgh;
  %internal;
//<\INTERNAL>

//<EXTERNAL>
  %pointerfield;
  int size_x = Nx+2*NGHX;
  int size_y = %size_y;
  int size_z = %size_z;
  int nx = Nx;
  int ny = Ny;
  int nz = Nz;
  int nghy = NGHY;
  int nghz = NGHZ;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  real dx = Dx;
  %global;
//<\EXTERNAL>

//<CONSTANT>
// real xmin(Nx+2*NGHX+1);
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
	%boundaries;
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
