/** \file RungeKunta.c

A fifth-order Runge-Kunta integrator for the N-body problem. This
N-body problem consists of the central star and N-1 planets.

*/

#include "fargo3d.h"

static real k1[MAX1D], k2[MAX1D], k3[MAX1D], k4[MAX1D], k5[MAX1D], k6[MAX1D];
static real Dist[MAX1D];
static real q0[MAX1D], q1[MAX1D], PlanetMasses[MAX1D];

void DerivMotionRK5(real *q_init, real *masses, \
		    real *deriv, int n, real dt, \
		    boolean *feelothers) {

  real *x,*y,*z, *vx, *vy, *vz, dist;
  real *derivx, *derivy, *derivz, *derivvx, *derivvy, *derivvz;
  real coef;
  int i, j;

  x = q_init;
  y = q_init+n;
  z = q_init+2*n;
  vx = q_init+3*n;
  vy = q_init+4*n;
  vz = q_init+5*n;
  derivx = deriv;
  derivy = deriv+n;
  derivz = deriv+2*n;
  derivvx = deriv+3*n;
  derivvy = deriv+4*n;
  derivvz = deriv+5*n;
  
  for (i = 0; i < n; i++)
    Dist[i] = sqrt(x[i]*x[i]+y[i]*y[i]+z[i]*z[i]);
  
  for (i = 0; i < n; i++) {
    derivx[i] = vx[i];
    derivy[i] = vy[i];
    derivz[i] = vz[i];
#ifdef NODEFAULTSTAR
    coef = 0.0;
#else
    coef = -G*MSTAR/Dist[i]/Dist[i]/Dist[i];
#endif
    derivvx[i] = coef*x[i];
    derivvy[i] = coef*y[i];
    derivvz[i] = coef*z[i];
    for (j = 0; j < n; j++) {
#ifndef NODEFAULTSTAR
      if (INDIRECTTERM) {
	coef = G*masses[j]/Dist[j]/Dist[j]/Dist[j];
	derivvx[i] -= coef*x[j];
	derivvy[i] -= coef*y[j];
	derivvz[i] -= coef*z[j];
      }
#endif
      if ((j != i) && (feelothers[i] == YES)) {
	dist = (x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j])+\
	  (z[i]-z[j])*(z[i]-z[j]);
	dist = sqrt(dist);
	coef = G*masses[j]/dist/dist/dist;
	if (ThereIsACentralBinary &&\
	    ((i == BinaryStar1) || (i == BinaryStar2)) &&\
	    (j != BinaryStar1) && (j != BinaryStar2))
	  /* Binary stars do not feel the planet(s) */
	  coef = 0.0;
	derivvx[i] += coef*(x[j]-x[i]);
	derivvy[i] += coef*(y[j]-y[i]);
	derivvz[i] += coef*(z[j]-z[i]);
      }
    }
  }

  for (i = 0; i < 6*n; i++)
    deriv[i] *= dt;  
}

void TranslatePlanetRK5(real *qold, real c1, real c2, real c3,
			real c4, real c5, real *qnew, int n) {
  int i;
  for (i = 0; i < 6*n; i++)
    qnew[i] = qold[i]+c1*k1[i]+c2*k2[i]+c3*k3[i]+c4*k4[i]+c5*k5[i];
}

void RungeKutta(real *q0, real dt, real *masses, real *q1,
		int n, boolean *feelothers) {
  int i;
  real timestep;
  timestep = dt;

  DerivMotionRK5 (q0, masses, k1, n, timestep, feelothers);
  TranslatePlanetRK5 (q0, 0.2, 0.0, 0.0, 0.0, 0.0, q1, n);

  DerivMotionRK5 (q1, masses, k2, n, timestep, feelothers);
  TranslatePlanetRK5 (q0, 0.075, 0.225, 0.0, 0.0, 0.0, q1, n);

  DerivMotionRK5 (q1, masses, k3, n, timestep, feelothers);
  TranslatePlanetRK5 (q0, 0.3, -0.9, 1.2, 0.0, 0.0, q1, n);

  DerivMotionRK5 (q1, masses, k4, n, timestep, feelothers);
  TranslatePlanetRK5 (q0, -11.0/54.0, 2.5, -70.0/27.0, 35.0/27.0, 0.0, q1, n);

  DerivMotionRK5 (q1, masses, k5, n, timestep, feelothers);
  TranslatePlanetRK5 (q0, 1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0, q1, n);

  DerivMotionRK5 (q1, masses, k6, n, timestep, feelothers);

  for (i = 0; i < 6*n; i++)
    q1[i] = (q0[i]+
	     37.0/378.0*k1[i]  + 
	     250.0/621.0*k3[i] + 
	     125.0/594.0*k4[i] + 
	     512.0/1771.0*k6[i]);

}

void AdvanceSystemRK5 (real dt) {

  int i, n;
  boolean *feelothers;
  real theta, rdot, r, new_r, omega, x, y;
  real dtheta, vx, vy, denom;
  real xc, yc, zc;
  
  n = Sys->nb; 

  for (i = 0; i < n; i++) { 
    q0[i]     = Sys->x[i];
    q0[i+n]   = Sys->y[i];
    q0[i+2*n] = Sys->z[i];

    q0[i+3*n] = Sys->vx[i];
    q0[i+4*n] = Sys->vy[i];
    q0[i+5*n] = Sys->vz[i];
    
    PlanetMasses[i] = Sys->mass[i];
  }

  feelothers = Sys->FeelOthers;
  RungeKutta (q0, dt, PlanetMasses, q1, n, feelothers);

  for (i = 1-(PhysicalTime >= RELEASEDATE); i < Sys->nb; i++) {
    Sys->x[i]  = q1[i];
    Sys->y[i]  = q1[i+n];
    Sys->z[i]  = q1[i+2*n];
    Sys->vx[i] = q1[i+3*n];
    Sys->vy[i] = q1[i+4*n];
    Sys->vz[i] = q1[i+5*n];
  }
  if (PhysicalTime < RELEASEDATE) { //We hereafter assume the planet to be in the plane.
    x = Sys->x[0];
    y = Sys->y[0];
    r = sqrt(x*x+y*y);
    theta = atan2(y,x);
    rdot = (RELEASERADIUS-r)/(RELEASEDATE-PhysicalTime);
    omega = sqrt((1.+Sys->mass[0])/r/r/r);
    new_r = r + rdot*dt;
    denom = r-new_r;
    if (denom != 0.0) {
      dtheta = 2.*dt*r*omega/denom*(sqrt(r/new_r)-1.);
    } else {
      dtheta = omega*dt;
    }
    vx = rdot;
    vy = new_r*sqrt((1.+Sys->mass[0])/new_r/new_r/new_r);
    Sys->x[0] = new_r*cos(dtheta+theta);
    Sys->y[0] = new_r*sin(dtheta+theta);
    Sys->z[0] = 0.0;
    Sys->vx[0]= vx*cos(dtheta+theta)-vy*sin(dtheta+theta); 
    Sys->vy[0]= vx*sin(dtheta+theta)+vy*cos(dtheta+theta); 
    Sys->vz[0] = 0.0;
  }

  /* Final residual correction for the binary barycenter (in order
     to avoid a long term spurious drift) */
  if (ThereIsACentralBinary) {
    xc = (PlanetMasses[BinaryStar1]*(Sys->x[BinaryStar1]) + PlanetMasses[BinaryStar2]*(Sys->x[BinaryStar2]))/	\
      (PlanetMasses[BinaryStar1]+PlanetMasses[BinaryStar2]);
    yc = (PlanetMasses[BinaryStar1]*(Sys->y[BinaryStar1]) + PlanetMasses[BinaryStar2]*(Sys->y[BinaryStar2]))/	\
      (PlanetMasses[BinaryStar1]+PlanetMasses[BinaryStar2]);
    zc = (PlanetMasses[BinaryStar1]*(Sys->z[BinaryStar1]) + PlanetMasses[BinaryStar2]*(Sys->z[BinaryStar2]))/	\
      (PlanetMasses[BinaryStar1]+PlanetMasses[BinaryStar2]);
    for (i = 0; i < 3; i++) {
      Sys->x[i] -= xc;
      Sys->y[i] -= yc;
      Sys->z[i] -= zc;
    }
  }
}
