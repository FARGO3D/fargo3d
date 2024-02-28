#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "fargo3d.h"

real x_mesh_I;
real y_mesh_I;

real fx(real x) {
  return x/x_mesh_I;
}

real gx_p(real x) {
  return 0.5*(2.0+XMC)*x/x_mesh_I - XMC*(XMB-XMA)/(2.0*M_PI*x_mesh_I)*sin(M_PI*(XMA+x)/(XMA-XMB));
}

real gx_m(real x) {
  return 0.5*(2.0+XMC)*x/x_mesh_I - XMC*(XMB-XMA)/(2.0*M_PI*x_mesh_I)*sin(M_PI*(x-XMA)/(XMA-XMB));
}

real hx(real x) {
  return (1.0+XMC)*fx(x);
}

real ux(real x) { //Integral of \psi

  // 
  if( XMA>XMB ) prs_error("Error! XMA>XMB\n");
  if( XMB>fabs(XMAX) || XMB>fabs(XMIN) ) prs_error("Error! XMB outside domain \n");

  x_mesh_I = (XMAX-XMIN) + XMC*(XMA + XMB);  
  if ( x <= -XMB              ) return fx   (x) - fx   (XMIN)             ;
  if ( x >  -XMB && x <= -XMA ) return gx_p (x) - gx_p (-XMB) + ux(-XMB) ; 
  if ( x >  -XMA && x <=  XMA ) return hx   (x) - hx   (-XMA) + ux(-XMA) ;
  if ( x >   XMA && x <=  XMB ) return gx_m (x) - gx_m ( XMA) + ux( XMA) ;
  if ( x >   XMB              ) return fx   (x) - fx   ( XMB) + ux( XMB) ;
  return 0.0; //default return
}

void compute_ux_constants(){
  Xmc0 = - fx   (XMIN)             ;
  Xmc1 = - gx_p (-XMB) + ux(-XMB) ; 
  Xmc2 = - hx   (-XMA) + ux(-XMA) ;
  Xmc3 = - gx_m ( XMA) + ux( XMA) ;
  Xmc4 = - fx   ( XMB) + ux( XMB) ;
  X_mesh_I  = (XMAX-XMIN) + XMC*(XMA + XMB);
}

real psi_x(real x) { //Mesh density function
  x_mesh_I = (XMAX-XMIN) + XMC*(XMA + XMB);  
  if      (fabs(x) <= XMA)                    return (1+XMC)/x_mesh_I;
  else if (fabs(x)  > XMA && fabs(x) < XMB)   return (1+XMC*pow(cos(M_PI*(fabs(x)-XMA)/(2*(XMB-XMA))),2))/x_mesh_I;
  else                                        return  1/x_mesh_I;
}

real fy(real y) {
  return log(y)/y_mesh_I;
}

real gy_p(real y) {
  return log(y)/y_mesh_I + 0.5*YMC*y/y_mesh_I - YMC*(YMB-YMA)/(2.0*M_PI*y_mesh_I)*sin(M_PI*(YMA+fabs(y-YMY0))/(YMA-YMB));
}

real gy_m(real y) {
  return log(y)/y_mesh_I + 0.5*YMC*y/y_mesh_I - YMC*(YMB-YMA)/(2.0*M_PI*y_mesh_I)*sin(M_PI*(YMA-fabs(y-YMY0))/(YMA-YMB));
}

real hy(real y) {
  return (log(y) + YMC*y)/y_mesh_I;
}

real uy(real y) { //Integral of \psi

  if( YMA>YMB ) prs_error("Error! YMA>YMB\n");
  if( (YMB+YMY0)> YMAX || (YMY0-YMB)< YMIN ) prs_error("Error! YMY0-YMB or YMY0+YMB outside domain \n");

  y_mesh_I = log(YMAX/YMIN) + YMC*(YMA + YMB); 
  
  if ( y <= YMY0-YMB                    ) return fy   (y) - fy   (YMIN)                     ;
  if ( y >  YMY0-YMB && y <= YMY0-YMA   ) return gy_m (y) - gy_m (YMY0-YMB) + uy(YMY0-YMB) ; 
  if ( y >  YMY0-YMA && y <= YMY0+YMA   ) return hy   (y) - hy   (YMY0-YMA) + uy(YMY0-YMA) ;
  if ( y >  YMY0+YMA && y <= YMY0+YMB   ) return gy_p (y) - gy_p (YMY0+YMA) + uy(YMY0+YMA) ;
  if ( y >  YMY0+YMB                    ) return fy   (y) - fy   (YMY0+YMB) + uy(YMY0+YMB) ;  
  return 0.0; //default return
}

real psi_y(real y) { //Mesh density function
  y_mesh_I = log(YMAX/YMIN) + YMC*(YMA + YMB);
  if      (fabs(y-YMY0) <= YMA)                   return (1/y+YMC)/y_mesh_I;
  else if (fabs(y-YMY0) > YMA && fabs(y) < YMB)   return (1/y+YMC*pow(cos(M_PI*(fabs(y-YMY0)-YMA)/(2*(YMB-YMA))),2))/y_mesh_I;
  else                                            return 1/y/y_mesh_I;
}


real bisect(real a, real b, int N, real (*u)(real), int option) {
  int i;
  if (a>b) prs_error("Error! a>b\n");
  
  real delta = 1e-16;
  int nmax = (int)((log(b-a)-log(delta))/log(2)) - 1;

  real du = 1.0/(real)(N-1); 
  real rhs, fa, fb, fc, c;

  if (option == 0) rhs = u(a) + du;
  if (option == 1) rhs = u(b) - du; // Used to fill ghost zones

  for (i=0; i<nmax; i++) {
    c = 0.5*(a + b);

    fa = u(a) - rhs;
    fb = u(b) - rhs;
    fc = u(c) - rhs;
 
    if (fabs(fc) <= 1e-16) return c;

    if (fa*fc < 0.0)
	    b = c;
    else if (fb*fc < 0.0)
      a = c;
    else {
      masterprint("Error! No root in the interval [%16.16lf,%16.16lf]\n",a,b);
      prs_exit(1); 
    }
  }
  return c;  
}
