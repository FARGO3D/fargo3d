#include "fargo3d.h"
void boundaries() {
  if (!PERIODICZ) {
#ifdef Z
    if(Gridd.bc_down)
      boundary_zmin[FluidIndex]();
    if(Gridd.bc_up)
      boundary_zmax[FluidIndex]();
#endif
  }
  if (!PERIODICY) {
#ifdef Y
    if(Gridd.bc_left)
      boundary_ymin[FluidIndex]();
    if(Gridd.bc_right)
      boundary_ymax[FluidIndex]();
#endif
  }
#ifdef GHOSTSX 
  Fill_GhostsX();
#endif
}

