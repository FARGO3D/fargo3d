
#include "fargo3d.h"

void rescale () {
SIGMA0 *= MSTAR/(R0*R0);
MASSTAPER *= sqrt(R0*R0*R0/G/MSTAR);
OMEGAFRAME *= sqrt(G*MSTAR/(R0*R0*R0));
DT *= sqrt(R0*R0*R0/G/MSTAR);
}
