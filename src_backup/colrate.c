//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void ColRate(real colrate, int i, int j, int feedback) {
  
  Alpha[i+j*NFLUIDS] = colrate;
  
  if(feedback == YES)
    Alpha[j+i*NFLUIDS] = colrate;
  else
    Alpha[j+i*NFLUIDS] = 0.0;

#ifdef GPU
  DevMemcpyH2D(Alpha_d,Alpha,sizeof(real)*NFLUIDS*NFLUIDS);
#endif

}
