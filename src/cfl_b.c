#include "fargo3d.h"

void cfl_b(){
  real step;
  
  int i,j,k;

  real min = reduction_full_MIN(DensStar, NGHY, Ny+NGHY, NGHZ, Nz+NGHZ);
  step = min;

#ifdef X
  real* vxmed = VxMed->field_cpu;
  real shearlimit;

#ifndef STANDARD
  INPUT2D (VxMed);
  i = j = k = 0;
#ifdef Z
  for (k = NGHZ; k < Nz+NGHZ; k++) {
#endif
#ifdef Y
    for (j = NGHY; j < Ny+NGHY; j++) {
#endif
#ifdef CARTESIAN
      shearlimit = CFL*Dx              /	\
	fabs(vxmed[l2D]-vxmed[l2D+1]);
#endif
#if defined(CYLINDRICAL) || defined(SPHERICAL)
      shearlimit = CFL*Dx              /			\
	fabs(vxmed[l2D]/Ymed(j)-vxmed[l2D+1]/Ymed(j+1));
#endif
      if (shearlimit < step)
      	step = shearlimit;
#ifdef Y
    }
#endif
#ifdef Z
  }
#endif
#endif
#endif
#ifdef FLOAT
  MPI_Allreduce(&step, &step_time, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
#else
  MPI_Allreduce(&step, &step_time, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
  if(step_time <= SMALLTIME) {
    masterprint("Error!!!--> Null dt\n");
    prs_exit(1);
  }
}
