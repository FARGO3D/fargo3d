//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void compute_potential(real dt) {

  real OmegaNew, domega;
  
  if (Corotating) GetPsysInfo (MARK); //WARNING!!!
  
#ifdef GPU
  //Copying all the useful planetary data
  DevMemcpyH2D(Sys->x_gpu, Sys->x_cpu, sizeof(real)*(Sys->nb+1));
  DevMemcpyH2D(Sys->y_gpu, Sys->y_cpu, sizeof(real)*(Sys->nb+1));
  DevMemcpyH2D(Sys->z_gpu, Sys->z_cpu, sizeof(real)*(Sys->nb+1));
  DevMemcpyH2D(Sys->mass_gpu, Sys->mass_cpu, sizeof(real)*(Sys->nb+1));
#endif
  
  DiskOnPrimaryAcceleration = ComputeAccel(0.0, 0.0, 0.0, 0.0, 0.0);
  FARGO_SAFE(ComputeIndirectTerm());
  FARGO_SAFE(Potential()); // Gravitational potential from star and planet(s)
  FARGO_SAFE(AdvanceSystemFromDisk(dt));
  
  FARGO_SAFE(AdvanceSystemRK5(dt));
  
  if (Corotating) {
    OmegaNew = GetPsysInfo(GET)/dt;
    domega = OmegaNew-OMEGAFRAME;
    FARGO_SAFE(CorrectVtheta(domega));
    OMEGAFRAME = OmegaNew;
  }
  RotatePsys(OMEGAFRAME*dt);
}

void Potential_cpu() {
  
//<USER_DEFINED>
  OUTPUT(Pot);
//<\USER_DEFINED>

//<EXTERNAL>
  real* pot  = Pot->field_cpu;
  real* xplanet = Sys->x_cpu;
  real* yplanet = Sys->y_cpu;
  real* zplanet = Sys->z_cpu;
  real* mplanet = Sys->mass_cpu;
  int nb        = Sys->nb;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
  int indirect_term = INDIRECTTERM;
  int indirectx = IndirectTerm.x;
  int indirecty = IndirectTerm.y;
  int indirectz = IndirectTerm.z;
//<\EXTERNAL>
  
//<INTERNAL>
  int i;
  int j;
  int k;
  int n;
  real smoothing;
  real dist;
  real rroche;
  real planetdistance;
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real ASPECTRATIO(1);
// real ROCHESMOOTHING(1);
// real FLARINGINDEX(1);
// real THICKNESSSMOOTHING(1);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;
#ifdef Z
  for (k=0; k<size_z; k++) {
#endif
#ifdef Y
    for (j=0; j<size_y; j++) {
#endif
#ifdef X
      for (i=0; i<size_x; i++) {
#endif
//<#>
	pot[l] =  -G*MSTAR/ymed(j); //Potential from star

	for(n=0; n<nb; n++) {

	  planetdistance = sqrt(xplanet[n]*xplanet[n]+
				yplanet[n]*yplanet[n]);
	  rroche = planetdistance*pow((1.0/3.0*mplanet[n]/MSTAR),1.0/3.0);

	  if (ROCHESMOOTHING != 0)
	    smoothing = rroche*ROCHESMOOTHING;
	  else
	    smoothing = ASPECTRATIO*
	      pow(planetdistance/R0,FLARINGINDEX)*
	      planetdistance*THICKNESSSMOOTHING;

	  smoothing*=smoothing;

	  dist = ((XC-xplanet[n])*(XC-xplanet[n])+
		  (YC-yplanet[n])*(YC-yplanet[n]));
	  if (indirect_term == YES) {
	    pot[l] += G*mplanet[n]*(XC*xplanet[n]+YC*yplanet[n])/(planetdistance*
								  planetdistance*
								  planetdistance); //IT Due to planets
	    pot[l] -= indirectx*XC + indirecty*YC; /* IT Due to gas */
	  }
	  pot[l] += -G*mplanet[n]/sqrt(dist+smoothing); //Potential from planets
	}
//<\#>
#ifdef X
      }
#endif
#ifdef Y
    }
#endif
#ifdef Z
  }
#endif
//<\MAIN_LOOP>
}
