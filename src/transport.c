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

void XadvectRAM(Field* F, real dt){
  RamSlopes(F);
  if (toupper(*SPACING) == 'L') AdvectRAMlin(dt,F);
  else AdvectRAM(dt,F);
}

void X_advection (Field *Vx_t, real dt) {
#if XDIM
  __VanLeerX(Density, DensStar, Vx_t, dt);
  TransportX(Mpx, Qs, Vx_t, dt);
  TransportX(Mmx, Qs, Vx_t, dt);
#endif
#if YDIM
  TransportX(Mpy, Qs, Vx_t, dt);
  TransportX(Mmy, Qs, Vx_t, dt);
#endif
#if ZDIM
  TransportX(Mpz, Qs, Vx_t, dt);
  TransportX(Mmz, Qs, Vx_t, dt);
#endif
#if ADIABATIC
  TransportX(Energy, Qs, Vx_t, dt);
#endif
  TransportX(Density, Qs, Vx_t, dt);
}

void transport(real dt){

#if XDIM
  FARGO_SAFE(momenta_x());
#endif
#if YDIM
  FARGO_SAFE(momenta_y());
#endif
#if ZDIM
  FARGO_SAFE(momenta_z());
#endif

#if ZDIM
  if(NZ>1){
    FARGO_SAFE(VanLeerZ_a(Density));
    FARGO_SAFE(VanLeerZ_b(dt, Density, DensStar));
#if XDIM
    TransportZ(Mpx, Qs, dt);
    TransportZ(Mmx, Qs, dt);
#endif
#if YDIM
    TransportZ(Mpy, Qs, dt);
    TransportZ(Mmy, Qs, dt);
#endif
#if ZDIM
    TransportZ(Mpz, Qs, dt);
    TransportZ(Mmz, Qs, dt);
#endif
#if ADIABATIC
    TransportZ(Energy, Qs, dt);
#endif
    TransportZ(Density, Qs, dt);
  }
#endif


#if YDIM
  if(NY>1){
    FARGO_SAFE(VanLeerY_a(Density));
    FARGO_SAFE(VanLeerY_b(dt, Density, DensStar));

#if XDIM
    TransportY(Mpx, Qs, dt);
    TransportY(Mmx, Qs, dt);
#endif
#if YDIM
    TransportY(Mpy, Qs, dt);
    TransportY(Mmy, Qs, dt);
#endif
#if ZDIM
    TransportY(Mpz, Qs, dt);
    TransportY(Mmz, Qs, dt);
#endif
#if ADIABATIC
    TransportY(Energy, Qs, dt);
#endif
    TransportY(Density, Qs, dt);
  }
#endif

#if XDIM
  if(NX>1){
#if STANDARD
    __VanLeerX = VanLeerX;
    X_advection (Vx_temp, dt);
#else // FARGO and RAM algorithm below

    FARGO_SAFE(ComputeResidual(dt));
    __VanLeerX = VanLeerX;
    X_advection (Vx, dt); // Vx => variable residual

#if (!RAM)
    //__VanLeerX= VanLeerX;
    __VanLeerX= VanLeerX_PPA;
    X_advection (Vx_temp, dt); // Vx_temp => fixed residual @ given r. This one only is done with PPA
    __VanLeerX = VanLeerX;
    AdvectSHIFT(Mpx, Nshift);
    AdvectSHIFT(Mmx, Nshift);
#if YDIM
    AdvectSHIFT(Mpy, Nshift);
    AdvectSHIFT(Mmy, Nshift);
#endif
#if ZDIM
    AdvectSHIFT(Mpz, Nshift);
    AdvectSHIFT(Mmz, Nshift);
#endif
#if ADIABATIC
    AdvectSHIFT(Energy, Nshift);
#endif
    AdvectSHIFT(Density, Nshift);

#else //RAM algorithm below
    FARGO_SAFE(RamComputeUstar(dt));

    XadvectRAM(Mpx,dt);
    XadvectRAM(Mmx,dt);

#if YDIM
    XadvectRAM(Mpy,dt);
    XadvectRAM(Mmy,dt);
#endif
#if ZDIM
    XadvectRAM(Mpz, dt);
    XadvectRAM(Mmz, dt);
#endif
#if ADIABATIC
    XadvectRAM(Energy, dt);
#endif
    XadvectRAM(Density,dt);

#endif //RAM
#endif //no STD
  }
#endif //X

#if XDIM
  FARGO_SAFE(NewVelocity_x());
#endif
#if YDIM
  FARGO_SAFE(NewVelocity_y());
#endif
#if ZDIM
  FARGO_SAFE(NewVelocity_z());
#endif
}
