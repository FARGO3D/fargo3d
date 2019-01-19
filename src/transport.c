#include "fargo3d.h"

static void (*__VanLeerX) (Field *, Field *, Field *, real);

void VanLeerX(Field *Density, Field *DensStar, Field *Vx_t, real dt) {
  FARGO_SAFE(VanLeerX_a(Density));
  FARGO_SAFE(VanLeerX_b(dt, Density, DensStar, Vx_t));
}


void TransportX(Field *Q, Field *Qs, Field *Vx_t, real dt) { 
  if (Q != Density){
     FARGO_SAFE(DivideByRho(Q));
     __VanLeerX(DivRho, Qs, Vx_t, dt);
     FARGO_SAFE(UpdateX (dt, Q, Qs, Vx_t));
  }
  else{
     FARGO_SAFE(UpdateDensityX (dt, Q, Vx_t));
  }
}
void TransportY(Field *Q, Field *Qs, real dt) {
  if (Q != Density){
    FARGO_SAFE(DivideByRho(Q));
    FARGO_SAFE(VanLeerY_a(DivRho));
    FARGO_SAFE(VanLeerY_b(dt, DivRho, Qs));
    FARGO_SAFE(UpdateY (dt, Q, Qs));
  }
  else
    FARGO_SAFE(UpdateDensityY (dt, Q));
}

void TransportZ(Field *Q, Field *Qs, real dt) {
   if (Q != Density){
    FARGO_SAFE(DivideByRho(Q));
    FARGO_SAFE(VanLeerZ_a(DivRho));
    FARGO_SAFE(VanLeerZ_b(dt, DivRho, Qs));
    FARGO_SAFE(UpdateZ (dt, Q, Qs));
  }
  else
    FARGO_SAFE(UpdateDensityZ (dt, Q));
}

void X_advection (Field *Vx_t, real dt) {
#ifdef X
  __VanLeerX(Density, DensStar, Vx_t, dt);
  TransportX(Mpx, Qs, Vx_t, dt);
  TransportX(Mmx, Qs, Vx_t, dt);
#endif
#ifdef Y
  TransportX(Mpy, Qs, Vx_t, dt);
  TransportX(Mmy, Qs, Vx_t, dt);
#endif
#ifdef Z
  TransportX(Mpz, Qs, Vx_t, dt);
  TransportX(Mmz, Qs, Vx_t, dt);
#endif
#ifdef ADIABATIC
  TransportX(Energy, Qs, Vx_t, dt);
#endif
  TransportX(Density, Qs, Vx_t, dt);
}

void transport(real dt){

#ifdef X
  FARGO_SAFE(momenta_x());
#endif
#ifdef Y
  FARGO_SAFE(momenta_y());
#endif
#ifdef Z
  FARGO_SAFE(momenta_z());
#endif

#ifdef Z
  FARGO_SAFE(VanLeerZ_a(Density));
  FARGO_SAFE(VanLeerZ_b(dt, Density, DensStar));
#ifdef X
  TransportZ(Mpx, Qs, dt);
  TransportZ(Mmx, Qs, dt);
#endif
#ifdef Y
  TransportZ(Mpy, Qs, dt);
  TransportZ(Mmy, Qs, dt);
#endif
#ifdef Z
  TransportZ(Mpz, Qs, dt);
  TransportZ(Mmz, Qs, dt);
#endif
#ifdef ADIABATIC
  TransportZ(Energy, Qs, dt);
#endif
  TransportZ(Density, Qs, dt);
#endif

#ifdef Y
  FARGO_SAFE(VanLeerY_a(Density));
  FARGO_SAFE(VanLeerY_b(dt, Density, DensStar));

#ifdef X  
  TransportY(Mpx, Qs, dt);
  TransportY(Mmx, Qs, dt);
#endif
#ifdef Y
  TransportY(Mpy, Qs, dt);
  TransportY(Mmy, Qs, dt);
#endif
#ifdef Z
  TransportY(Mpz, Qs, dt);
  TransportY(Mmz, Qs, dt);
#endif
#ifdef ADIABATIC
  TransportY(Energy, Qs, dt);
#endif
  TransportY(Density, Qs, dt);
#endif

#ifdef X
#ifdef STANDARD
  __VanLeerX = VanLeerX;
  X_advection (Vx_temp, dt);
#else // FARGO algorithm below
  FARGO_SAFE(ComputeResidual(dt));
  __VanLeerX = VanLeerX;
  X_advection (Vx, dt); // Vx => variable residual
  //__VanLeerX= VanLeerX;
  __VanLeerX= VanLeerX_PPA;
  X_advection (Vx_temp, dt); // Vx_temp => fixed residual @ given r. This one only is done with PPA
  __VanLeerX = VanLeerX;
  AdvectSHIFT(Mpx, Nshift);
  AdvectSHIFT(Mmx, Nshift);
#ifdef Y
  AdvectSHIFT(Mpy, Nshift);
  AdvectSHIFT(Mmy, Nshift);
#endif
#ifdef Z
  AdvectSHIFT(Mpz, Nshift);
  AdvectSHIFT(Mmz, Nshift);
#endif
#ifdef ADIABATIC
  AdvectSHIFT(Energy, Nshift);
#endif
  AdvectSHIFT(Density, Nshift);
#endif
#endif
  
  FARGO_SAFE(NewVelocity_x());
  FARGO_SAFE(NewVelocity_y());
  FARGO_SAFE(NewVelocity_z());
}
