#include "fargo3d.h"

void AmbipolarDiffusion() {

  FARGO_SAFE(ComputeJx());
  FARGO_SAFE(ComputeJy());
  FARGO_SAFE(ComputeJz());
  FARGO_SAFE(AmbipolarDiffusion_emfx());
  FARGO_SAFE(AmbipolarDiffusion_emfy());
  FARGO_SAFE(AmbipolarDiffusion_emfz());

}
