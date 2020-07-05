//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void HallEffect(real dt){

  // We need to reset the EMFs before updating them.
  FARGO_SAFE(Reset_field(EmfxH));
  FARGO_SAFE(Reset_field(EmfyH));
  FARGO_SAFE(Reset_field(EmfzH));

  // We use as working arrays BxH, ByH, BzH
  FARGO_SAFE(copy_field(BxH,Bx));
  FARGO_SAFE(copy_field(ByH,By));
  FARGO_SAFE(copy_field(BzH,Bz));

  FARGO_SAFE(ComputeJx());
  FARGO_SAFE(ComputeJy());
  FARGO_SAFE(HallEffect_emfz());

  FARGO_SAFE(HallEffect_UpdateB(dt,1,0,0));
  FARGO_SAFE(HallEffect_UpdateB(dt,0,1,0));

  FARGO_SAFE(ComputeJy());
  FARGO_SAFE(ComputeJz());
  FARGO_SAFE(HallEffect_emfx());

  FARGO_SAFE(HallEffect_UpdateB(dt,0,2,0));
  FARGO_SAFE(HallEffect_UpdateB(dt,0,0,1));
  
  FARGO_SAFE(ComputeJx());
  FARGO_SAFE(ComputeJz());
  FARGO_SAFE(HallEffect_emfy());

  FARGO_SAFE(HallEffect_UpdateEmfs());

  // Once we have the EMFs, we recover the original B
  FARGO_SAFE(copy_field(Bx,BxH));
  FARGO_SAFE(copy_field(By,ByH));
  FARGO_SAFE(copy_field(Bz,BzH));
    
}
