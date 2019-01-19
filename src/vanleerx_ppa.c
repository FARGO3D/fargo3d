#include "fargo3d.h"

void VanLeerX_PPA(Field *Q, Field *Qs, Field *Vx_t, real dt) {
  FARGO_SAFE(VanLeerX_PPA_a(Q));
  FARGO_SAFE(VanLeerX_PPA_b(Q));
#ifdef PPA_STEEPENER
  FARGO_SAFE(VanLeerX_PPA_steep (Q));
#endif
  FARGO_SAFE(VanLeerX_PPA_c(Q));
  FARGO_SAFE(VanLeerX_PPA_d(dt,Q,Qs,Vx_t));
  
}

void VanLeerX_PPA_2D(Field *Q, Field *Qs, Field2D *Vx_t, real dt) {

  FARGO_SAFE(VanLeerX_PPA_a(Q));
  FARGO_SAFE(VanLeerX_PPA_b(Q));
#ifdef PPA_STEEPENER
  FARGO_SAFE(VanLeerX_PPA_steep(Q));
#endif
  FARGO_SAFE(VanLeerX_PPA_c(Q));
  FARGO_SAFE(VanLeerX_PPA_d_2d(dt,Q,Qs,Vx_t));
  
}
