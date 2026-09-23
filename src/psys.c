#include "fargo3d.h"

int FindNumberOfPlanets(char *filename) {
  FILE* input;
  char s[512];
  int counter=0;

  input = fopen(filename, "r");
  if(input == NULL) {
    fprintf(stderr, "%s cannot be opened\n", filename);
    prs_exit(1);
  }
  while(fgets(s, 510, input)!=NULL) {
    if(isalpha(s[0])) counter++;
  }
  fclose(input);
  return counter;
}

PlanetarySystem *AllocPlanetSystem(int nb) {
  char command[512];
  real *mass, *x, *y, *z, *vx, *vy, *vz, *acc;
  boolean *feeldisk, *feelothers;
  int i;
  PlanetarySystem *sys;
  int temp;
  sys  = (PlanetarySystem *)malloc (sizeof(PlanetarySystem));
  if (sys == NULL) {
    fprintf (stderr, "Not enough memory to alloc PlanetarySystem.\n");
    prs_exit (1);
  }
  x    = (real*)malloc(sizeof(real)*(nb+1));
  y    = (real*)malloc(sizeof(real)*(nb+1));
  z    = (real*)malloc(sizeof(real)*(nb+1));
  vx   = (real*)malloc(sizeof(real)*(nb+1));
  vy   = (real*)malloc(sizeof(real)*(nb+1));
  vz   = (real*)malloc(sizeof(real)*(nb+1));
  mass = (real*)malloc(sizeof(real)*(nb+1));
  acc  = (real*)malloc(sizeof(real)*(nb+1));
  if ((x == NULL) || (y == NULL) || (z == NULL)	      \
      || (vx == NULL) || (vy == NULL) || (vz == NULL) \
      || (acc == NULL) || (mass == NULL)) {
    fprintf (stderr, "Not enough memory to alloc components of planetary system.\n");
    prs_exit (1);
  }
  feeldisk   = (boolean*)malloc(sizeof(char)*(nb+1));
  feelothers = (boolean*)malloc(sizeof(char)*(nb+1));
  if ((feeldisk == NULL) || (feelothers == NULL)) {
    fprintf (stderr, "Not enough memory for boolean allocation in PlanetarySystem.\n");
    prs_exit (1);
  }
  sys->x = x;
  sys->y = y;
  sys->z = z;
  sys->vx= vx;
  sys->vy= vy;
  sys->vz= vz;
  sys->acc=acc;
  sys->mass = mass;
  sys->FeelDisk = feeldisk;
  sys->FeelOthers = feelothers;
  for (i = 0; i < nb; i++) {
    x[i] = y[i] = z[i] = vx[i] = vy[i] = vz[i] = mass[i] = acc[i] = 0.0;
    feeldisk[i] = feelothers[i] = YES;
  }
  for (i = 0; i < nb; i++) {
    /* Creates orbit[i].dat if it does not exist */
    sprintf (command, "touch %s/orbit%d.dat", OUTPUTDIR, i);
    temp = system (command);
  }

  sys->x = x;
  sys->y = y;
  sys->z = z;

  sys->x_cpu = sys->x; //Alias
  sys->y_cpu = sys->y; //Alias
  sys->z_cpu = sys->z; //Alias
  sys->mass_cpu = sys->mass; //Alias

  sys->vx= vx;
  sys->vy= vy;
  sys->vz= vz;

  sys->acc=acc;
  sys->mass = mass;
  sys->FeelDisk = feeldisk;
  sys->FeelOthers = feelothers;

#ifdef GPU
  int status;
  status = DevMalloc(&(sys->x_gpu),(sizeof(real)*(nb+1)));
  status = DevMalloc(&(sys->y_gpu),(sizeof(real)*(nb+1)));
  status = DevMalloc(&(sys->z_gpu),(sizeof(real)*(nb+1)));
  status = DevMalloc(&(sys->mass_gpu),(sizeof(real)*(nb+1)));
//  status = DevMemcpyH2D(sys->x_gpu, sys->x_cpu, sizeof(real)*(nb+1));
//  status = DevMemcpyH2D(sys->y_gpu, sys->y_cpu, sizeof(real)*(nb+1));
//  status = DevMemcpyH2D(sys->z_gpu, sys->z_cpu, sizeof(real)*(nb+1));
//  status = DevMemcpyH2D(sys->mass_gpu, sys->mass_cpu, sizeof(real)*(nb+1));
#endif

  return sys;
}

void FreePlanetary () {
  free (Sys->x);
  free (Sys->vx);
  free (Sys->y);
  free (Sys->vy);
  free (Sys->mass);
  free (Sys->acc);
  free (Sys->FeelOthers);
  free (Sys->FeelDisk);
  free (Sys);
}

real ComputeInnerMass(real r) {
  int i,j,k;
  real mass=0.0;
  real *rho;
  real innermass;
  rho = Density->field_cpu;
  for (k=NGHZ; k<Nz+NGHZ; k++) {
    for (j=NGHY; j<Ny+NGHY; j++) {
      for (i=NGHX; i<Nx+NGHX; i++) {
	if(Ymed(j)<r) {
	  mass+=rho[l]*Vol(j,k);
	}
      }	
    }
  }
#ifdef FLOAT
  MPI_Allreduce (&mass, &innermass, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
  MPI_Allreduce (&mass, &innermass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  // printf("InnerMass=%lf\n", innermass);
  return innermass;
}

PlanetarySystem *InitPlanetarySystem (char *filename) {
  FILE *input;
  char s[512], nm[512], test1[512], test2[512], *s1;
  PlanetarySystem *sys;
  int i=0, j, nb, nbstars=0, i_star1=-1, i_star2=-1;
  real xp,yp,zp,vxp,vyp,vzp,mp,M1,M2,r1,r2,v1,v2;
  real mass, dist, accret;
  boolean feeldis, feelothers;
  real newmass;
  real summass=0.0;
  real e_bin, a_bin, period_bin;
  int status;
  
  if (ThereArePlanets == NO) {
    sys = AllocPlanetSystem (1);
    sys->nb = 0;
    sys->x = sys->vx = sys->y = sys->vy = NULL;
    return sys;
  }
  
  nb = FindNumberOfPlanets (filename);
  if (CPU_Master) {
    if(nb > 1) printf  ("%d planets found.\n", nb);
    else printf  ("%d planet found.\n", nb);
  }
  sys = AllocPlanetSystem (nb);
  input = fopen (filename, "r"); // Its existence has been checked already
  sys->nb = nb;
  while (fgets(s, 510, input) != NULL) {
    sscanf(s, "%s ", nm);
    if (isalpha(s[0])) {
      s1 = s + strlen(nm);
#ifdef FLOAT
      sscanf(s1 + strspn(s1, "\t :=>_"), "%f %f %f %s %s", &dist, &mass, &accret, test1, test2);
#else
      sscanf(s1 + strspn(s1, "\t :=>_"), "%lf %lf %lf %s %s", &dist, &mass, &accret, test1, test2);
#endif
      if ((SEMIMAJORAXIS > 0.0) && (i == 0)) // SemiMajorAxis can be
					     // used to overwrite the
					     // first planet's initial
					     // orbital radius.
	dist = SEMIMAJORAXIS;
      if (ORBITALRADIUS > 1e-30)
	dist *= ORBITALRADIUS;
#ifdef RESCALE
      dist *= R0;
      accret *= sqrt(G*MSTAR/(R0*R0*R0));
      mass *= MSTAR;
#endif
      sys->mass[i] = mass;
      if (PLANETMASS > 1e-18)
	sys->mass[0] = PLANETMASS;
      feeldis = feelothers = YES;
      if (tolower(*test1) == 'n') feeldis = NO;
      if (tolower(*test2) == 'n') feelothers = NO;
      sys->x[i] = (real)dist*(1.0+ECCENTRICITY); // Planets are initialized at apoastron
      sys->y[i] = 0.0;
      sys->z[i] = 0.0;
      sys->vy[i] = (real)sqrt(G*(MSTAR+mass)/dist)*	\
	sqrt( (1.0-ECCENTRICITY)/(1.0+ECCENTRICITY))*	\
	cos(INCLINATION);
      sys->vx[i] = -0.0000000000*sys->vy[i];
      sys->vz[i] = sys->vy[i]*sin(INCLINATION)/	\
	cos(INCLINATION);
      sys->acc[i] = accret;
      sys->FeelDisk[i] = feeldis;
      sys->FeelOthers[i] = feelothers;
      if (sys->mass[i] > THRESHOLD_STELLAR_MASS)
	nbstars++;
      i++;
    }
  }
  if (nbstars > 0)
    if (CPU_Master)
      printf ("%d appear to be stellar mass objects.\n", nbstars);
  /* We check whether some "planets" are actually stars */
  for (i = 0; i < nb; i++) {
    if (sys->mass[i] > THRESHOLD_STELLAR_MASS) {/* Our arbitrary threshold for a star */
      summass += sys->mass[i];
    }
  }
  if ((summass > 0.0) && (fabs(summass-MSTAR)/MSTAR > 0.01)) {
    mastererr ("WARNING --- It seems that some 'planet(s)' in file %s\n", PLANETCONFIG);
    mastererr ("WARNING --- is (are) actually star(s). In that case, it is a good idea that\n");
    mastererr ("WARNING --- the preprocessor variable MSTAR be the sum of their mass.\n");
    mastererr ("WARNING --- Currently this is not the case.\n");
    mastererr ("WARNING --- You can edit the value of MSTAR in the file src/fondam.h.\n");
    mastererr ("WARNING --- In this file, you want to edit the flavor of MSTAR which\n");
    mastererr ("WARNING --- corresponds to the unit system you are presently working with.\n");
    mastererr ("WARNING --- The current unit system is: ");
#if !(defined(MKS) || defined (CGS))
    mastererr ("scale free (code units).\n");
#endif
#ifdef CGS
    mastererr ("cgs\n");
#endif
#ifdef MKS
    mastererr ("MKS\n");
#endif
    mastererr ("WARNING --- So we suggest you edit src/fondam.h to have:\n");
    mastererr ("\n    #define MSTAR_");
#if !(defined(MKS) || defined (CGS))
    mastererr ("SF");
#endif
#ifdef CGS
    mastererr ("CGS");
#endif
#ifdef MKS
    mastererr ("MKS");
#endif
    mastererr (" %g\n\n", summass);
    mastererr ("WARNING --- This will ensure that:\n");
    mastererr ("WARNING ---   * The planets are initialized on nearly circular orbits.\n");
    mastererr ("WARNING ---   * Roche radii, wherever needed, are correctly estimated.\n");
    mastererr ("WARNING ---   * Wave-killing BCs have the correct strength.\n\n");
  }
#ifndef NODEFAULTSTAR
  if (summass > 0.0) {
    mastererr("WARNING --- Some of your planets appear to be stars.\n");
    mastererr("WARNING --- Their total mass is %g.\n", summass);
    mastererr("WARNING --- Yet you have also by default a star at the mesh center.\n");
    mastererr("WARNING --- Its mass is %g, ie %5.2f %% of the other stars mass.\n",MSTAR, MSTAR/summass*100);
    mastererr("WARNING --- If this is what you intended, you can safely ignore this message.\n");
    mastererr("WARNING --- Otherwise, you must recompile the code with the flag NODEFAULTSTAR\n");
    mastererr("WARNING --- turned on. To do so, find your .opt file (in the setups folder),\n");
    mastererr("WARNING --- and add to it the following line:\n");
    mastererr("\nFARGO_OPT += -DNODEFAULTSTAR\n\n");
  }
#endif

#ifdef NODEFAULTSTAR
  /* We now properly re-initialize the stars */
  if (nbstars > 2) {
    mastererr ("ERROR --- at the present time you cannot have more than two stars in your .cfg file.\n");
    prs_exit (1);
  }
  xp = yp = zp = vxp = vyp = vzp = 0.0;
  for (i = 0; i < nb; i++) {
    if (sys->mass[i] < THRESHOLD_STELLAR_MASS) {
      mp = sys->mass[i];
      xp += mp*sys->x[i];
      yp += mp*sys->y[i];
      zp += mp*sys->z[i];
      vxp += mp*sys->vx[i];
      vyp += mp*sys->vy[i];
      vzp += mp*sys->vz[i];
    } else
      if (i_star1 < 0)
	i_star1 = i;
      else
	i_star2 = i;
  }
  if (nbstars == 1) {		/* We impose that the mesh be centered
				   on the center of mass of the
				   system */
    mp = sys->mass[i_star1];
    sys->x[i_star1] = -xp/mp;
    sys->y[i_star1] = -yp/mp;
    sys->z[i_star1] = -zp/mp;
    sys->vx[i_star1] = -vxp/mp;
    sys->vy[i_star1] = -vyp/mp;
    sys->vz[i_star1] = -vzp/mp;
  }
  if (nbstars == 2) {
    ThereIsACentralBinary = YES;
    BinaryStar1 = i_star1;
    BinaryStar2 = i_star2;
    e_bin = sys->x[i_star2];
    period_bin = sys->x[i_star1];
    M1 = sys->mass[i_star1];
    M2 = sys->mass[i_star2];
    a_bin = pow((period_bin*sqrt(G*(M1+M2)))/2.0/M_PI,2./3.);
    r1 = a_bin*M2/(M1+M2)*(1.+e_bin);
    r2 = M1*r1/M2;
    v1 = 2.0*M_PI*a_bin*M2/(M1+M2)/period_bin*sqrt((1.0-e_bin)/(1.0+e_bin));
    v2 = M1*v1/M2;
    /* We impose below that the mesh be centered on the center of mass
       of the binary */
    sys->x[i_star1] = r1;
    sys->x[i_star2] = -r2;
    sys->vy[i_star1] = v1;
    sys->vy[i_star2] = -v2;
    sys->y[i_star1] = 0.0;
    sys->z[i_star1] = 0.0;
    sys->vx[i_star1] = 0.0;
    sys->vz[i_star1] = 0.0;
  }
#endif
  return sys;
}

void ListPlanets () {
  int nb;
  int i;
  nb = Sys->nb;
  if (!CPU_Master) return;
  for (i = 0; i < nb; i++) {
    printf ("Planet number %d\n", i);
    printf ("---------------\n");
    printf ("x = %.10f\ty = %.10f\tz = %.10f\n", \
	    Sys->x[i],Sys->y[i],Sys->z[i]);
    printf ("vx = %.10f\tvy = %.10f\tvz = %.10f\n", \
	    Sys->vx[i],Sys->vy[i],Sys->vz[i]);
    if (Sys->acc[i] == 0.0)
      printf ("Non-accreting.\n");
    else
      printf ("accretion time = %.10f\n", 1.0/(Sys->acc[i]));
    if (Sys->FeelDisk[i] == YES) {
      printf ("Feels the disk potential\n");
    } else {
      printf ("Doesn't feel the disk potential\n");
    }
    if (Sys->FeelOthers[i] == YES) {
      printf ("Feels the other planets potential\n");
    } else {
      printf ("Doesn't feel the other planets potential\n");
    }
    printf ("\n");
  }
}

real GetPsysInfo (boolean action) {
  real d1, d2, cross;
  real x, y, z;
  real vx, vy, vz;
  real m;
  real xc, yc;
  real omega;
  OrbitalElements o;
  StateVector v;
  static real X_planet, Y_planet;

  v.x = xc = x = Sys->x[0];
  v.y = yc = y = Sys->y[0];
  v.z = z = Sys->z[0];
  v.vx = vx = Sys->vx[0];
  v.vy = vy = Sys->vy[0];
  v.vz = vz = Sys->vz[0];

  m = Sys->mass[0]+MSTAR;

  o = SV2OE (v,m);

  if (GuidingCenter == YES) {
    xc = o.a*cos(o.M+o.Perihelion_Phi)*cos(o.i);
    yc = o.a*sin(o.M+o.Perihelion_Phi)*cos(o.i);
  }

  if (o.e < 1e-8) {
    xc = x;
    yc = y;
  }

  switch (action) {
  case MARK: 
    X_planet = xc;
    Y_planet = yc;
    return 0.;
  case GET:
    x = xc;
    y = yc;
    d2 = sqrt(x*x+y*y);
    d1 = sqrt(X_planet*X_planet+Y_planet*Y_planet);
    cross = X_planet*y-x*Y_planet;
    X_planet = x;
    Y_planet = y;
    return asin(cross/(d1*d2)); // azimuth change
  case FREQUENCY:
    if (GuidingCenter == YES)
      return sqrt(G*m/pow(o.a,3.0));
    else
      return (x*vy-y*vx)/(x*x+y*y); // True in 3D as well: frame rotates about z axis
  }
  return 0.0;
}

void RotatePsys (real angle) {
  //rotate -angle
  int nb;
  int i;
  real sint, cost, xt, yt;
  nb = Sys->nb;
  sint = sin(angle);
  cost = cos(angle);
  XAxisRotationAngle += angle;
  if (XAxisRotationAngle >= 2.0*M_PI)
    XAxisRotationAngle -= 2.0*M_PI;
  if (XAxisRotationAngle < 0.0)
    XAxisRotationAngle += 2.0*M_PI;
  for (i = 0; i < nb; i++) {
    xt = Sys->x[i];
    yt = Sys->y[i];
    Sys->x[i] = xt*cost+yt*sint;
    Sys->y[i] = -xt*sint+yt*cost;
    xt = Sys->vx[i];
    yt = Sys->vy[i];
    Sys->vx[i] = xt*cost+yt*sint;
    Sys->vy[i] = -xt*sint+yt*cost;
  }
}
