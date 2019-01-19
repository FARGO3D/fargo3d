#include "fargo3d.h"

void init_stockholm() {
  
  static boolean init = TRUE;

  boolean error_density = TRUE;
  boolean error_vx = TRUE;
  boolean error_vy = TRUE;
  boolean error_vz = TRUE;
  boolean error_energy = TRUE;
  
  if(init) {

  INPUT(Density);
  OUTPUT2D(Density0);
#ifdef ADIABATIC
  INPUT(Energy);
  OUTPUT2D(Energy0);
#endif
#ifdef X
  INPUT(Vx);
  OUTPUT2D(Vx0);
#endif
#ifdef Y
  INPUT(Vy);
  OUTPUT2D(Vy0);
#endif
#ifdef Z
  INPUT(Vz);
  OUTPUT2D(Vz0);
#endif

  int i,j,k;

  i = j = k = 0;
  
#ifdef X
  real* vx  = Vx->field_cpu;
  real* vx0 = Vx0->field_cpu;
#endif
#ifdef Y
  real* vy  = Vy->field_cpu;
  real* vy0 = Vy0->field_cpu;
#endif
#ifdef Z
  real* vz  = Vz->field_cpu;
  real* vz0 = Vz0->field_cpu;
#endif
#ifdef ADIABATIC
  real* e    = Energy->field_cpu;
  real* e0   = Energy0->field_cpu;
#endif
  real* rho  = Density->field_cpu;
  real* rho0 = Density0->field_cpu;

  if ((Restart == YES) || (Restart_Full == YES)) {
    error_density = Read2D(Density0, "density0_2d.dat", OUTPUTDIR, GHOSTINC);
    error_vx = Read2D(Vx0, "vx0_2d.dat", OUTPUTDIR, GHOSTINC);
    error_vy = Read2D(Vy0, "vy0_2d.dat", OUTPUTDIR, GHOSTINC);
    error_vz = Read2D(Vz0, "vz0_2d.dat", OUTPUTDIR, GHOSTINC);
#ifdef ADIABATIC
    error_energy = Read2D(Energy0, "e0_2d.dat", OUTPUTDIR, GHOSTINC);
#endif
  }
  
#ifdef Z
    for (k=0; k<Nz+2*NGHZ; k++) {
#endif
#ifdef Y
      for (j=0; j<Ny+2*NGHY; j++) {
#endif
#ifdef ADIABATIC
	if (error_energy)
	  e0[l2D]   = e[l];
#endif
#ifdef X
	if (error_vx)
	  vx0[l2D]  = vx[l];
#endif
#ifdef Y
	if (error_vy)
	  vy0[l2D]  = vy[l];
#endif
#ifdef Z
	if (error_vz)
	  vz0[l2D]  = vz[l];
#endif
	if (error_density)
	  rho0[l2D] = rho[l];
#ifdef Y
      }
#endif
#ifdef Z
    }
#endif
    Write2D(Density0, "density0_2d.dat", OUTPUTDIR, GHOSTINC);
#ifdef X
    Write2D(Vx0,  "vx0_2d.dat", OUTPUTDIR, GHOSTINC);
#endif
#ifdef Y
    Write2D(Vy0,  "vy0_2d.dat", OUTPUTDIR, GHOSTINC);
#endif
#ifdef Z
    Write2D(Vz0,  "vz0_2d.dat", OUTPUTDIR, GHOSTINC);
#endif
#ifdef ADIABATIC
    Write2D(Energy0,   "e0_2d.dat", OUTPUTDIR, GHOSTINC);
#endif
    init = FALSE;
  }
}
