#include "fargo3d.h"
void boundaries() {
  if (!PERIODICZ) {
#ifdef Z
    if(Gridd.bc_down)
      boundary_zmin();
    if(Gridd.bc_up)
      boundary_zmax();
#endif
  }
  if (!PERIODICY) {
#ifdef Y
    if(Gridd.bc_left)
      boundary_ymin();
    if(Gridd.bc_right)
      boundary_ymax();
#endif
  }
#ifdef GHOSTSX 
  Fill_GhostsX();
#endif
}

