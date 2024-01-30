#include "fargo3d.h"

void _init_stockholm() {

  INPUT(Density);
  OUTPUT2D(Density0);
#if ADIABATIC
  INPUT(Energy);
  OUTPUT2D(Energy0);
#endif
#if XDIM
  INPUT(Vx);
  OUTPUT2D(Vx0);
#endif
#if YDIM
  INPUT(Vy);
  OUTPUT2D(Vy0);
#endif
#if ZDIM
  INPUT(Vz);
  OUTPUT2D(Vz0);
#endif

  boolean error_density = TRUE;
  boolean error_vx      = TRUE;
  boolean error_vy      = TRUE;
  boolean error_vz      = TRUE;
  boolean error_energy  = TRUE;

  char outputname[MAXLINELENGTH];

  if ((Restart == YES) || (Restart_Full == YES)) {
    sprintf(outputname,"%s0_2d.dat",Density->name);
    error_density = Read2D(Density0, outputname, OUTPUTDIR, GHOSTINC);
#if XDIM
    sprintf(outputname,"%s0_2d.dat",Vx->name);
    error_vx = Read2D(Vx0, outputname, OUTPUTDIR, GHOSTINC);
#endif
#if YDIM
    sprintf(outputname,"%s0_2d.dat",Vy->name);
    error_vy = Read2D(Vy0, outputname, OUTPUTDIR, GHOSTINC);
#endif
#if ZDIM
    sprintf(outputname,"%s0_2d.dat",Vz->name);
    error_vz = Read2D(Vz0, outputname, OUTPUTDIR, GHOSTINC);
#endif
#if ADIABATIC
    sprintf(outputname,"%s0_2d.dat",Energy->name);
    error_energy = Read2D(Energy0, outputname, OUTPUTDIR, GHOSTINC);
#endif
  }

  int i,j,k;
  
  i = j = k = 0;
  
#if XDIM
  real* vx  = Vx->field_cpu;
  real* vx0 = Vx0->field_cpu;
#endif
#if YDIM
  real* vy  = Vy->field_cpu;
  real* vy0 = Vy0->field_cpu;
#endif
#if ZDIM
  real* vz  = Vz->field_cpu;
  real* vz0 = Vz0->field_cpu;
#endif
#if ADIABATIC
  real* e    = Energy->field_cpu;
  real* e0   = Energy0->field_cpu;
#endif
  real* rho  = Density->field_cpu;
  real* rho0 = Density0->field_cpu;
  
#if ZDIM
  for (k=0; k<Nz+2*NGHZ; k++) {
#endif
#if YDIM
    for (j=0; j<Ny+2*NGHY; j++) {
#endif
#if ADIABATIC
      if (error_energy)
	e0[l2D]   = e[l];
#endif
#if XDIM
      if (error_vx)
	vx0[l2D]  = vx[l];
#endif
#if YDIM
      if (error_vy)
	vy0[l2D]  = vy[l];
#endif
#if ZDIM
      if (error_vz)
	vz0[l2D]  = vz[l];
#endif
      if (error_density)
	rho0[l2D] = rho[l];
#if YDIM
    }
#endif
#if ZDIM
  }
#endif

  sprintf(outputname,"%s0_2d.dat",Density->name);
  Write2D(Density0, outputname, OUTPUTDIR, GHOSTINC);
#if XDIM
  sprintf(outputname,"%s0_2d.dat",Vx->name);
  Write2D(Vx0, outputname, OUTPUTDIR, GHOSTINC);
#endif
#if YDIM
  sprintf(outputname,"%s0_2d.dat",Vy->name);
  Write2D(Vy0, outputname, OUTPUTDIR, GHOSTINC);
#endif
#if ZDIM
  sprintf(outputname,"%s0_2d.dat",Vz->name);
  Write2D(Vz0, outputname, OUTPUTDIR, GHOSTINC);
#endif
#if ADIABATIC
  sprintf(outputname,"%s0_2d.dat",Energy->name);
  Write2D(Energy0, outputname, OUTPUTDIR, GHOSTINC);
#endif

}

void init_stockholm() {

  static boolean init = TRUE;

  if (init) MULTIFLUID(_init_stockholm());
  
  init = FALSE;

}
