#include "fargo3d.h"

/* This function only runs on the CPU in its present state. */

real ComputeMean(Field *F) {
  int i,j,k;
  real total = 0.;
  real volume = 0., dvol;
  real grandvolume=0., grandtotal = 0.;

  real *f;
  
  INPUT (F);

  f = F->field_cpu;

  i = j = k = 0;
#ifdef Z
  for (k=NGHZ;k<Nz+NGHZ;k++) {
#endif
#ifdef Y
    for (j=NGHY;j<Ny+NGHY;j++) {
#endif
#ifdef X
      for (i=NGHX;i<Nx+NGHX;i++) {
#endif
	dvol = Vol(j,k);
	total += f[l]*dvol;
	volume += dvol;
#ifdef X
      }
#endif
#ifdef Y
    }
#endif
#ifdef Z
  }
#endif
#ifdef FLOAT
  MPI_Allreduce(&total, &grandtotal, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&volume, &grandvolume, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
  MPI_Allreduce(&total, &grandtotal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&volume, &grandvolume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  return grandtotal/grandvolume;
}
