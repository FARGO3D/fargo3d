#include "fargo3d.h"

void ComputeIndirectTerm () {
#ifndef NODEFAULTSTAR
  IndirectTerm.x = -DiskOnPrimaryAcceleration.x;
  IndirectTerm.y = -DiskOnPrimaryAcceleration.y;
  IndirectTerm.z = -DiskOnPrimaryAcceleration.z;
  if (!INDIRECTTERM) {
    IndirectTerm.x = 0.0;
    IndirectTerm.y = 0.0;
    IndirectTerm.z = 0.0;
  }
#else
  IndirectTerm.x = 0.0;
  IndirectTerm.y = 0.0;
  IndirectTerm.z = 0.0;
#endif
}

Force ComputeForce(real x, real y, real z,
		   real rsmoothing, real mass) {
  
  Force Force;

  /* The trick below, which uses VxMed as a 2D temporary array,
     amounts to subtracting the azimuthally averaged density prior to
     the torque evaluation. This has no impact on the torque, but has
     on the angular speed of the planet and is required for a proper
     location of resonances in a non self-gravitating disk. See
     Baruteau & Masset 2008, ApJ, 678, 483 (arXiv:0801.4413) for
     details. */
#ifdef BM08
  ComputeVmed (Total_Density);
  ChangeFrame (-1, Total_Density, VxMed);
#endif
  /* The density is now the perturbed density */
  FARGO_SAFE(_ComputeForce(x, y, z, rsmoothing, mass)); /* Function/Kernel Launcher. */
  /* We restore the total density below by adding back the azimuthal
     average */
#ifdef BM08
  ChangeFrame (+1, Total_Density, VxMed);
#endif

  
#ifdef FLOAT
  MPI_Allreduce (&localforce, &globalforce, 12, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
  MPI_Allreduce (&localforce, &globalforce, 12, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  
  Force.fx_inner    = globalforce[0];
  Force.fy_inner    = globalforce[1];
  Force.fz_inner    = globalforce[2];
  Force.fx_ex_inner = globalforce[3];
  Force.fy_ex_inner = globalforce[4];
  Force.fz_ex_inner = globalforce[5];
  Force.fx_outer    = globalforce[6];
  Force.fy_outer    = globalforce[7];
  Force.fz_outer    = globalforce[8];
  Force.fx_ex_outer = globalforce[9];
  Force.fy_ex_outer = globalforce[10];
  Force.fz_ex_outer = globalforce[11];

  return Force;

}

Point ComputeAccel(real x, real y, real z,
		   real rsmoothing, real mass) {
  Point acceleration;
  Force force;
  force = ComputeForce (x, y, z, rsmoothing, mass);
  if (EXCLUDEHILL) {
    acceleration.x = force.fx_ex_inner+force.fx_ex_outer;
    acceleration.y = force.fy_ex_inner+force.fy_ex_outer;
    acceleration.z = force.fz_ex_inner+force.fz_ex_outer;
  } 
  else {
    acceleration.x = force.fx_inner+force.fx_outer;
    acceleration.y = force.fy_inner+force.fy_outer;
    acceleration.z = force.fz_inner+force.fz_outer;
  }
  return acceleration;
}

void AdvanceSystemFromDisk(real dt) {
  int NbPlanets, k;
  Point gamma;
  real x, y, z;
  real r, m, smoothing;
  NbPlanets = Sys->nb;
  for (k = 0; k < NbPlanets; k++) {
    if (Sys->FeelDisk[k] == YES) {
      m = Sys->mass[k];
      x = Sys->x[k];
      y = Sys->y[k];
      z = Sys->z[k];
      r = sqrt(x*x + y*y + z*z);
      if (ROCHESMOOTHING != 0)
	smoothing = r*pow(m/3./MSTAR,1./3.)*ROCHESMOOTHING;
      else
	smoothing = ASPECTRATIO*pow(r/R0,FLARINGINDEX)*r*THICKNESSSMOOTHING;
      gamma = ComputeAccel (x, y, z, smoothing, m);
      Sys->vx[k] += dt * gamma.x;
      Sys->vy[k] += dt * gamma.y;
      Sys->vz[k] += dt * gamma.z;
#ifdef GASINDIRECTTERM
      Sys->vx[k] += dt * IndirectTerm.x;
      Sys->vy[k] += dt * IndirectTerm.y;
      Sys->vz[k] += dt * IndirectTerm.z;
#endif
    }
  }
}

OrbitalElements SV2OE (StateVector v, real m) {
  real x,y,z,vx,vy,vz;
  real Ax, Ay, Az, h, h2, inc, e;
  real d, hx, hy, hz, a, E, M, V;
  real hhor, per, an;//Ascending node
  OrbitalElements o;
  x = v.x;
  y = v.y;
  z = v.z;
  vx = v.vx;
  vy = v.vy;
  vz = v.vz;

  d = sqrt(x*x+y*y+z*z);
  
  hx   = y*vz - z*vy;
  hy   = z*vx - x*vz;
  hz   = x*vy - y*vx;
  hhor = sqrt(hx*hx + hy*hy);

  h2  = hx*hx + hy*hy + hz*hz;
  h   = sqrt(h2);
  o.i = inc = asin(hhor/h);

  Ax = vy*hz-vz*hy - G*m*x/d; // v x h - ri/abs(r);
  Ay = vz*hx-vx*hz - G*m*y/d;
  Az = vx*hy-vy*hx - G*m*z/d;

  o.e = e = sqrt(Ax*Ax+Ay*Ay+Az*Az)/(G*m); //Laplace-Runge-Lenz vector
  o.a = a = h*h/(G*m*(1.-e*e));

  //Eccentric anomaly
  if (e != 0.0) {
    E = acos((1.0-d/a)/e); //E evaluated as such is between 0 and PI
  } else {
    E = 0.0;
  }
  if (x*vx+y*vy+z*vz < 0) E= -E; //Planet goes toward central object,
  //hence on its way from aphelion to perihelion (E < 0)

  if (isnan(E)) {
    if (d < a) 
      E = 0.0;
    else
      E = M_PI;
  }

  o.M = M = E-e*sin(E);
  o.E = E;

  //V: true anomaly
  if (e > 1.e-14) {
    V = acos ((a*(1.0-e*e)/d-1.0)/e);
  } else {
    V = 0.0;
  }
  if (E < 0.0) V = -V;

  o.ta = V;
  
  if (fabs(o.i) > 1e-5) {
    an = atan2(hy,hx)+M_PI*.5; //Independently of sign of (hz)
    if (an > 2.0*M_PI) an -= 2.0*M_PI;
  } else {
    an = 0.0;//Line of nodes not determined ==> defaults to x axis
  }

  o.an = an;

  // Argument of periapsis
  per = acos((Ax*cos(an)+Ay*sin(an))/sqrt(Ax*Ax+Ay*Ay+Az*Az));
  if ((-hz*sin(an)*Ax+hz*cos(an)*Ay+(hx*sin(an)-hy*cos(an))*Az) < 0.0)
    per = 2.0*M_PI-per;
  o.per = per;
  if (Ax*Ax+Ay*Ay > 0.0)
    o.Perihelion_Phi = atan2(Ay,Ax);
  else
    o.Perihelion_Phi = atan2(y,x);
  return o;
}

void FindOrbitalElements (v,m,n)
     StateVector v;
     real m;
     int n;
{
  FILE *output;
  char name[256];
  OrbitalElements o;
  if (CPU_Rank) return;
  sprintf (name, "%sorbit%d.dat", OUTPUTDIR, n);
  output = fopen_prs (name, "a");
  o = SV2OE (v,m);
 
  fprintf (output, "%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g", \
	   PhysicalTime, o.e, o.a, o.M, o.ta, o.per, XAxisRotationAngle);
  fprintf (output, "\t%.12g\t%.12g\t%.12g\n", o.i, o.an, o.Perihelion_Phi);
  fclose (output);
}

void SolveOrbits (sys)
     PlanetarySystem *sys;
{
  int i, n;
  StateVector v;
  n = sys->nb;
  for (i = 0; i < n; i++) {
    v.x = sys->x[i];
    v.y = sys->y[i];
    v.z = sys->z[i];
    v.vx = sys->vx[i];
    v.vy = sys->vy[i];
    v.vz = sys->vz[i];
    FindOrbitalElements (v,MSTAR+sys->mass[i],i);
  }
} 
