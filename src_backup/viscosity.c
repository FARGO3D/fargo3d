#include "fargo3d.h"


void viscosity(real dt){
#ifdef CARTESIAN
  FARGO_SAFE(visctensor_cart());
  FARGO_SAFE(addviscosity_cart(dt));
#endif
#ifdef CYLINDRICAL
  FARGO_SAFE(visctensor_cyl());
  FARGO_SAFE(addviscosity_cyl(dt));
#endif
#ifdef SPHERICAL
  FARGO_SAFE(visctensor_sph());
  FARGO_SAFE(addviscosity_sph(dt));
#endif
}
