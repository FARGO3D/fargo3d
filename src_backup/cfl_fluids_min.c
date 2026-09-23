#include "fargo3d.h"

void CflFluidsMin() {
  int i;
  real step = 1e30;
  real min;

  for (i=0;i<NFLUIDS;i++) {
    if (step > Min[i])
      step = Min[i];
  }
  
#ifdef FLOAT
  MPI_Allreduce(&step, &StepTime, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
#else
  MPI_Allreduce(&step, &StepTime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
  if(StepTime <= SMALLTIME) {
    masterprint("Error!!!--> Null dt\n");
    prs_exit(1);
  }
}
