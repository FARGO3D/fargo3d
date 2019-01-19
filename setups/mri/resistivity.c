#include "fargo3d.h"

real Resistivity (real y, real z) {
  //The user can freely redefine this function
  //By defining his own function, in a file named
  //resistivity.c, in his own setup directory,
  //which will be built instead of this file.
  real rsup, width, rinf, eta=0.0;
#ifdef MHD
  width = (YMAX-YMIN)/7.;
  rsup = YMAX-width;
  rinf = YMIN+width;
  if (y > rsup) {
    eta = ETA*(y-rsup)/width;
  }
  if (y < rinf) {
    eta = ETA*(rinf-y)/width/5.0;
  }
#endif
  return eta;
}

