#include "fargo3d.h"

void DustDiffusion_Main(real dt) {

  int init = 0;

  if (init == 0) {
    if (Fluids[0]->Fluidtype != GAS) {
      mastererr("ERROR -  Dust diffusion module assumes that Fluids[0] is of type GAS.\n");
      mastererr("ERROR -  You can fix this by defining the Fluid with:\n");
      mastererr("Fluids[0] = CreateFluid(<label>,GAS);");
      prs_exit (1);      
    }
    
    init = 1;
  }

  // In principle, Diffusion_Coefficients() does not need to be called every time step
  // for temporary constant viscosity.
  FARGO_SAFE(DustDiffusion_Coefficients());
  
  MULTIFLUID(
	     if(Fluidtype == DUST) {
	       FARGO_SAFE(DustDiffusion_Core(dt));        // Updated density is stored in Pressure field.
	       FARGO_SAFE(copy_field(Density,Pressure));  // We update dust densities from the Pressure field.
	       FARGO_SAFE(FillGhosts(DENS));  // We update dust densities from the Pressure field.
	     });
}
