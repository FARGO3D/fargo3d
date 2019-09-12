#include "fargo3d.h"

void prs_exit(int numb) {
  MPI_Finalize();
  exit(numb);
}

void prs_error(char *string){
  fprintf(stderr, "%s\n", string);
  prs_exit(1);
}

int PrimitiveVariables () {
  int var=DENS;
#ifdef ADIABATIC
  var |= ENERGY;
#endif
#ifdef X
  var |= VX;
#endif
#ifdef Y
  var |= VY;
#endif
#ifdef Z
  var |= VZ;  
#endif

#ifdef MHD
  if(Fluidtype==GAS){
    var |= BX|BY|BZ;
    var |= EMFX|EMFY|EMFZ;
  }
#endif

  return var;
}

inline real Swap(real f) {
  union {
    real value;
#ifndef FLOAT
    char b[8];
#else
    char b[4];
#endif
  } value1, value2;

  value1.value = f;
#ifdef FLOAT
  value2.b[0] = value1.b[3];
  value2.b[1] = value1.b[2];
  value2.b[2] = value1.b[1];
  value2.b[3] = value1.b[0];
#else
  value2.b[0] = value1.b[7];
  value2.b[1] = value1.b[6];
  value2.b[2] = value1.b[5];
  value2.b[3] = value1.b[4];
  value2.b[4] = value1.b[3];
  value2.b[5] = value1.b[2];
  value2.b[6] = value1.b[1];
  value2.b[7] = value1.b[0];
#endif
  return value2.value;
}

void Check_CUDA_Blocks_Consistency () {
  boolean problem = NO;
#ifdef GPU
  #ifndef X
  if (BLOCK_X > 1) {
    mastererr ("ERROR  ---  ERROR   ---- ERROR\n");
    mastererr ("ERROR  ---  ERROR   ---- ERROR\n");
    mastererr ("ERROR  ---  ERROR   ---- ERROR\n");
    mastererr ("You have CUDA blocks that are larger than 1 in X,\n");
    mastererr ("but the dimension X does not exist in your setup. Fix this !!\n");
    problem = YES;
  }
  #endif
  #ifndef Y
  if (BLOCK_Y > 1) {
    mastererr ("ERROR  ---  ERROR   ---- ERROR\n");
    mastererr ("ERROR  ---  ERROR   ---- ERROR\n");
    mastererr ("ERROR  ---  ERROR   ---- ERROR\n");
    mastererr ("You have CUDA blocks that are larger than 1 in Y,\n");
    mastererr ("but the dimension Y does not exist in your setup. Fix this !!\n");
    problem = YES;
  }
  #endif
  #ifndef Z
  if (BLOCK_Z > 1) {
    mastererr ("ERROR  ---  ERROR   ---- ERROR\n");
    mastererr ("ERROR  ---  ERROR   ---- ERROR\n");
    mastererr ("ERROR  ---  ERROR   ---- ERROR\n");
    mastererr ("You have CUDA blocks that are larger than 1 in Z,\n");
    mastererr ("but the dimension Z does not exist in your setup. Fix this !!\n");
    problem = YES;
  }
  #endif
  if (problem == YES) {
    mastererr ("The code would produce undetermined  results in this case.\n");
    mastererr ("You must edit the file setups/%s/%s.opt\n", xstr(SETUPNAME), xstr(SETUPNAME));
    mastererr ("then rebuild the code\n");
    prs_exit (1);
  }
#endif
}

void MakeDir (char *string) {
  int foo=0;
  char command[MAX1D];
  DIR *dir;
  int temp;
  /* Each processor tries to create the directory, sequentially */
  /* Silent if directory exists */
  /* If all processes see the same partition, only the first process
     will create the directory. Alternatively, they will create as
     many directories as necessary. For instance, if we have say 4 PEs per node
     and each node sees its own scratchdir, nbprocesses/4 
     mkdir() commands will be issued */
  if (CPU_Rank) MPI_Recv (&foo, 1, MPI_INT, CPU_Rank-1, 53, MPI_COMM_WORLD, &fargostat);
  dir = opendir (string);
  if (dir) {
    closedir (dir);
  } else {
    fprintf (stdout, "Process %d created the directory %s\n", CPU_Rank, string);
    sprintf (command, "mkdir -p %s", string);
    temp = system (command);
  }
  if (CPU_Rank < CPU_Number-1) MPI_Send (&foo, 1, MPI_INT, CPU_Rank+1, 53, MPI_COMM_WORLD);
}

FILE *fopen_prs (char *string, char *mode) {
  FILE *f;
  char dir[MAXLINELENGTH];
  char *p;
  f = fopen (string, mode);
  if (f == NULL) {
    /* This should be redundant with the call to MakeDir () at the
       beginning, from main.c; this is not a problem however */
    strncpy (dir, string, MAXLINELENGTH-1);
    if ((p = strrchr (dir, '/')) != NULL) {
      *p = 0;
      MakeDir (dir);
    }
    f = fopen (string, "w");	/* "w" instead of mode: at this stage we know the file does not exist */
    if (f == NULL) {
      printf ("Process %d could not open %s\n", CPU_Rank, string);
      printf ("Tried to create %s\n", dir);
      fprintf (stdout, "Still could not open %s.\n", string);
      fprintf (stdout, "You should check that the permissions are correctly set.\n");
      fprintf (stdout, "Run aborted\n");
      prs_exit (1);
    }
  }
  return f;
}

FILE *master_fopen (char *filename, char *mode) {
  FILE *f=NULL;
  boolean Abort = NO;
  if (CPU_Rank == 0) {
    f = fopen (filename, mode);
    if (f == NULL) {
      mastererr ("Could not open file %s\n", filename);
      Abort = YES;
    }
  }
  if (Abort == YES)
    prs_exit (EXIT_FAILURE);
  return f;
}

void masterfprintf(FILE *f, const char *template, ...) {
  va_list ap;
  if (!CPU_Master) return;
  va_start(ap, template);
  vfprintf(f, template, ap);
  va_end(ap);
}

void masterprint(const char *template, ...) {
  va_list ap;
  if (!CPU_Master) return;
  va_start(ap, template);
  vfprintf(stdout, template, ap);
  va_end(ap);
}

void mastererr(const char *template, ...) {
  va_list ap;
  if (!CPU_Master) return;
  va_start(ap, template);
  vfprintf(stderr, template, ap);
  va_end(ap);
}

void InitSpace() {
  real dy, dz;
  real x0;
  int  i,j,k;
  
  FILE *domain;
  char domain_out[512];
  real ymin, zmin, xmin;
  int jmin, jmax, kmin, kmax;
  real temp1;
  boolean already_x=NO;
  boolean already_y=NO;
  boolean already_z=NO;
  int temp, relay;
  int init = 0;

  if (*SPACING=='F') { //Fixed spacing

    masterprint("Warning: zone spacing will be taken from the files domain_i.dat.\n");
    sprintf(domain_out, "%s%s", OUTPUTDIR, "domain_x.dat");
    domain = fopen(domain_out, "r");
    if(domain != NULL) {
      masterprint("Warning: x spacing taken from domain_x.dat file!\n");
      init = 0;
      for (i=0; i<NX+1; i++) {
#ifdef FLOAT
	temp = fscanf(domain, "%f\n", &temp1);
#else
	temp = fscanf(domain, "%lf\n", &temp1);
#endif
	Xmin(i) = temp1;
      }
      Dx = (XMAX-XMIN)/NX;
      already_x = YES;
    }
    fclose(domain);
    
    sprintf(domain_out, "%s%s", OUTPUTDIR, "domain_y.dat");
    domain = fopen(domain_out, "r");
    if(domain != NULL){
      masterprint("Warning: y spacing taken from domain_y.dat file!\n");
      for (j=0; j<NY+2*NGHY+1; j++) {
#ifdef FLOAT
	temp = fscanf(domain, "%f\n", &temp1);
#else
	temp = fscanf(domain, "%lf\n", &temp1);
#endif
	if((j>=(Y0) && j<(Y0+Ny+2*NGHY+1))) {
	  Ymin(j-Y0) = temp1;
	}
      }
      already_y = YES;
    }
    
    fclose(domain);
    sprintf(domain_out, "%s%s", OUTPUTDIR, "domain_z.dat");
    domain = fopen(domain_out, "r");
    if(domain != NULL) {
      masterprint("Warning: z spacing taken from domain_z.dat file!\n");
      for (k=0; k<NZ+2*NGHZ+1; k++) {
#ifdef FLOAT
	temp = fscanf(domain, "%f\n", &temp1);
#else
	temp = fscanf(domain, "%lf\n", &temp1);
#endif
	if((k>=(Z0) && k<(Z0+Nz+2*NGHZ+1))) {
	  Zmin(k-Z0) = temp1;
	}
      }
      already_z = YES;
    }
    fclose(domain);
  }

  else {

    Dx = (XMAX-XMIN)/NX;
    for (i = 0; i<Nx+2*NGHX+1; i++) {
      Xmin(i) = XMIN + Dx*(i-NGHX);
    }
    
#ifdef Y
    dy = (YMAX-YMIN)/NY;
#else
    dy = 0;
#endif
#ifdef Z
    dz = (ZMAX-ZMIN)/NZ;
#else
    dz = 0;
#endif
    
  if (((toupper(*SPACING)) == 'L') && ((toupper(*(SPACING+1))) == 'O')) { //Logarithmic
      masterprint("Warning: The Y spacing is logarithmic.\n");
      dy = (log(YMAX)-log(YMIN))/NY;
      for (j = 0; j<Ny+2*NGHY+1; j++) {
	Ymin(j) = exp(log(YMIN) + dy*(j+Y0-NGHY));
      }
    }
    else {  //Linear
      masterprint("Warning: The Y spacing is linear (default).\n");
      for (j = 0; j<Ny+2*NGHY+1; j++) {
	Ymin(j) = YMIN + dy*(j+Y0-NGHY);
      }
    }
    for (k = 0; k<Nz+2*NGHZ+1; k++) {
#ifdef Z
      Zmin(k) = ZMIN + dz*(k+Z0-NGHZ);
#else
      Zmin(k) = 0.0;
#endif
    }
  }

  for (i = 0; i<Nx+2*NGHX; i++) {
    Xmed(i) = 0.5*(Xmin(i+1)+Xmin(i));
  }
  for (j = 0; j<Ny+2*NGHY; j++) {
    Ymed(j) = 0.5*(Ymin(j+1)+Ymin(j));
  }
  for (k = 0; k<Nz+2*NGHZ; k++) {
    Zmed(k) = 0.5*(Zmin(k+1)+Zmin(k));
  }

  for (i = 1; i<Nx+2*NGHX; i++) {
    InvDiffXmed(i) = 1./(Xmed(i)-Xmed(i-1));
  }
  if (Nx+2*NGHX>1) InvDiffXmed(0) = InvDiffXmed(1);
  else InvDiffXmed(0) = 0.0;

  for (j = 1; j<Ny+2*NGHY; j++) {
    InvDiffYmed(j) = 1./(Ymed(j)-Ymed(j-1));
  }

  if (Ny+2*NGHY>1) InvDiffYmed(0) = InvDiffYmed(1);
  else InvDiffYmed(0) = 0.0;
  
#ifdef Z
  for (k = 1; k<Nz+2*NGHZ; k++) {
    InvDiffZmed(k) = 1./(Zmed(k)-Zmed(k-1));
  }

  if(Nz+2*NGHZ>1){
    InvDiffZmed(1) = 0.0;
    InvDiffZmed(0) = InvDiffZmed(1);
  }
  else{
    InvDiffZmed(0) = 0.0;
  }
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  
  if (!already_x) {
    if(CPU_Master) {
      sprintf(domain_out, "%s%s", OUTPUTDIR, "domain_x.dat");
      domain = fopen(domain_out, "w");
      for (i=0; i<Nx+2*NGHX+1; i++) {
	fprintf(domain, "%#.18lf\n",Xmin(i));
	fflush(domain);
      }
    }
  }
  
  if (!already_y) {
    if (CPU_Rank > 0) { // Force sequential read
      MPI_Recv (&relay, 1, MPI_INT, CPU_Rank-1, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    sprintf(domain_out, "%s%s", OUTPUTDIR, "domain_y.dat");
    if(CPU_Master)  {
      domain = fopen(domain_out, "w");
      jmin = 0;
      jmax = Ny+NGHY+1; 
    }
    else {
      if (CPU_Rank < Ncpu_x) {
	domain = fopen(domain_out, "a");
	jmin = NGHY+1;
	jmax = Ny+NGHY+1;
      }
    }
    if(CPU_Rank == Ncpu_x-1)
      jmax = Ny+2*NGHY+1;
    if (CPU_Rank < Ncpu_x) {
      for (j=jmin; j<jmax; j++) {
	fprintf(domain, "%#.18lf\n",Ymin(j));
      }
      fclose(domain);
    }
    if (CPU_Rank < CPU_Number-1) {  // Force sequential read
      MPI_Send (&relay, 1, MPI_INT, CPU_Rank+1, 42, MPI_COMM_WORLD);
    }
  }

  MPI_Barrier (MPI_COMM_WORLD);
  
  if (!already_z) {
    if (CPU_Rank > 0) { // Force sequential read
      MPI_Recv (&relay, 1, MPI_INT, CPU_Rank-1, 43, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    sprintf(domain_out, "%s%s", OUTPUTDIR, "domain_z.dat");
    if(CPU_Master)  {
      domain = fopen(domain_out, "w");
      jmin = 0;
      jmax = Nz+NGHZ+1;
    } 
    else {
      if (J == 0) {
	domain = fopen(domain_out, "a");
	jmin = NGHZ+1;
	jmax = Nz+NGHZ+1;
      }
    }
    if ((K == Ncpu_y-1) && (J == 0))
      jmax = Nz+2*NGHZ+1;
    if (J == 0) {
      for (j=jmin; j<jmax; j++) {
	fprintf(domain, "%#.18lf\n",Zmin(j));
      }
      fclose(domain);
    }
    if (CPU_Rank < CPU_Number-1) {  // Force sequential read
      MPI_Send (&relay, 1, MPI_INT, CPU_Rank+1, 43, MPI_COMM_WORLD);
    }
    
    MPI_Barrier (MPI_COMM_WORLD);
  }
}


void InitSurfaces() {

  int j,k;
  
#ifdef CARTESIAN
    for (j = 0; j<Ny+2*NGHY; j++) {
#ifdef Y
      Sxj(j) = Ymin(j+1)-Ymin(j);
#else
      Sxj(j) = 1.0;
#endif
      Syj(j) = 1.0;
#if (defined(X) && defined(Y))
      Szj(j) = Dx*(Ymin(j+1)-Ymin(j));
#elif defined(X)
      Szj(j) = Dx;
#elif defined(Y)
      Szj(j) = (Ymin(j+1)-Ymin(j));
#else
      Szj(j) = 1.0;
#endif
      InvVj(j)  = 1./(Sxj(j));
    }    
    for (k = 0; k<Nz+2*NGHZ; k++) {
#ifdef Z
      Sxk(k) = Zmin(k+1)-Zmin(k);
#else
      Sxk(k) = 1.0;
#endif
#if (defined(X) && defined(Z))
      Syk(k) = Dx*(Zmin(k+1)-Zmin(k));
#elif defined(X)
      Syk(k) = Dx;
#elif defined(Z)
      Syk(k) = (Zmin(k+1)-Zmin(k));
#else 
      Syk(k) = 1.0;
#endif
      Szk(k) = 1.0;
    }
#endif
#ifdef CYLINDRICAL
#if !defined(Y)
    masterprint("Error! In simulations w/ cylindrical geometry, Y must be activated!\n");
    exit(0);
#endif
    for (j = 0; j<Ny+2*NGHY; j++) {
      Sxj(j) = Ymin(j+1)-Ymin(j);
#if defined(X)
      Syj(j) = Ymin(j)*Dx;
      Szj(j) = 0.5*(Ymin(j+1)*Ymin(j+1) -
		    Ymin(j)*Ymin(j))*Dx;
#else
      Syj(j) = Ymin(j);
      Szj(j) = 0.5*(Ymin(j+1)*Ymin(j+1) -
		    Ymin(j)*Ymin(j));
#endif
      InvVj(j) = 1.0/Szj(j);
    }
    for (k = 0; k<Nz+2*NGHZ; k++) {
#ifdef Z
      Sxk(k) = Zmin(k+1)-Zmin(k);
#else
      Sxk(k) = 1.0;
#endif
      Syk(k) = Sxk(k);
      Szk(k) = 1.0;
    }
#endif
#ifdef SPHERICAL
#if !defined(Y)
    masterprint("Error! In simulations w/ spherical geometry, Y must be activated!\n");
    exit(0);
#endif
    for (j = 0; j<Ny+2*NGHY; j++) {
      Sxj(j) = 0.5*(Ymin(j+1)*Ymin(j+1)
		    -Ymin(j)*Ymin(j));
#ifdef X
      Syj(j) = Ymin(j)*Ymin(j)*Dx;
      Szj(j) = 0.5*(Ymin(j+1)*Ymin(j+1)
		    -Ymin(j)*Ymin(j))*Dx;
      InvVj(j) = 3./((Ymin(j+1)*Ymin(j+1)*Ymin(j+1) - 
		      Ymin(j)*Ymin(j)*Ymin(j))*Dx);
#else
      Syj(j) = Ymin(j)*Ymin(j);
      Szj(j) = 0.5*(Ymin(j+1)*Ymin(j+1)
		    -Ymin(j)*Ymin(j)); 
      InvVj(j) = 3./((Ymin(j+1)*Ymin(j+1)*Ymin(j+1) - 
		      Ymin(j)*Ymin(j)*Ymin(j)));
#endif
    }
    for (k = 0; k<Nz+2*NGHZ; k++) {
#ifdef Z
      Sxk(k) = Zmin(k+1)-Zmin(k);
      Syk(k) = cos(Zmin(k))-cos(Zmin(k+1));
      Szk(k) = sin(Zmin(k));
#else
      Sxk(k) = 1.0;
      Syk(k) = 1.0;
      Szk(k) = 1.0; 
#endif
    }
#endif
#ifdef DEBUG
    masterprint("SURFACES OK\n");
#endif
}

void SelectFluid(int n) {
  //Function for selecting the current fluid
  Fluidtype = Fluids[n]->Fluidtype;
  Density = Fluids[n]->Density;
  Energy = Fluids[n]->Energy;
  VxMed = Fluids[n]->VxMed;
#ifdef X
  Vx = Fluids[n]->Vx;
  Vx_temp = Fluids[n]->Vx_temp;
  Vx_half = Fluids[n]->Vx_half;
#endif
#ifdef Y
  Vy = Fluids[n]->Vy;
  Vy_temp = Fluids[n]->Vy_temp;
  Vy_half = Fluids[n]->Vy_half;
#endif
#ifdef Z
  Vz = Fluids[n]->Vz;
  Vz_temp = Fluids[n]->Vz_temp;
  Vz_half = Fluids[n]->Vz_half;
#endif
#ifdef STOCKHOLM
  Density0 = Fluids[n]->Density0;
  Energy0 = Fluids[n]->Energy0;
  Vx0 = Fluids[n]->Vx0;
  Vy0 = Fluids[n]->Vy0;
  Vz0 = Fluids[n]->Vz0;
#endif
}

void CreateFields() {

  Reduction2D = CreateField2D ("Reduction2D", YZ);

#if (defined(X) || defined(MHD))
  
  Mpx              = CreateField   ("Moment_Plus_X" , 0, 1,0,0);
  Mmx              = CreateField   ("Moment_Minus_X", 0, 1,0,0);
  
  Vxhy             = CreateField2D ("Vxhy"    , YZ);
  Vxhyr            = CreateField2D ("Vxhyr"   , YZ);
  Vxhz             = CreateField2D ("Vxhz"    , YZ);
  Vxhzr            = CreateField2D ("Vxhzr"   , YZ);
  Eta_profile_xi   = CreateField2D ("Eta_xi"  , YZ);
  Eta_profile_xizi = CreateField2D ("Eta_xizi", YZ);
  Eta_profile_zi   = CreateField2D ("Eta_zi"  , YZ);

  Nshift = CreateFieldInt2D ("Nshift");
  Nxhy   = CreateFieldInt2D ("Nxhy");
  Nxhz   = CreateFieldInt2D ("Nxhz");
#endif

#if (defined(Y) || defined(MHD))
  Mpy     = CreateField("Moment_Plus_Y" , 0,0,1,0);
  Mmy     = CreateField("Moment_Minus_Y", 0,0,1,0);
#endif
  
#if (defined(Z) || defined(MHD))
  Mpz     = CreateField("Moment_Plus_Z" , 0,0,0,1);
  Mmz     = CreateField("Moment_Minus_Z", 0,0,0,1);
#endif

  Pot     = CreateField("potential", 0,0,0,0);
  Slope   = CreateField("Slope"    , 0,0,0,0);
  DivRho  = CreateField("DivRho"   , 0,0,0,0);  // This field cannot
						// be aliased wherever
						// reductions are
						// needed
  
  DensStar      = CreateField("DensStar"     , 0,0,0,0);
  Qs            = CreateField("Qs"           , 0,0,0,0);
  Pressure      = CreateField("Pressure"     , 0,0,0,0);
  Total_Density = CreateField("Total_Density", 0,0,0,0);
  
  QL      = CREATEFIELDALIAS("QLeft", Pressure, 0);
  QR      = CreateField("QRight", 0,0,0,0);
  
#ifdef PPA_STEEPENER
  LapPPA  = CreateField("LapPPA", 0,0,0,0);
#endif

#ifdef DUSTDIFFUSION
  Sdiffyczc = CREATEFIELDALIAS("Sdiffyczc",Mpx,0);
  Sdiffyfzc = CREATEFIELDALIAS("Sdiffyfzc",Mmx,0);
#ifdef Z
  Sdiffyczf = CREATEFIELDALIAS("Sdiffyczf",Mmy,0);
  Sdiffyfzf = CREATEFIELDALIAS("Sdiffyfzf",Mpy,0);
#endif
#endif
  
#ifdef MHD
  Bx      = CreateField("bx", BX,1,0,0);
  By      = CreateField("by", BY,0,1,0);
  Bz      = CreateField("bz", BZ,0,0,1);

  B1_star = CREATEFIELDALIAS("B1_star" , Mpy      , 0);
  B2_star = CREATEFIELDALIAS("B2_star" , Mmz      , 0);
  V1_star = CREATEFIELDALIAS("V1_star" , Pressure , 0);
  V2_star = CREATEFIELDALIAS("V2_star" , Mmx      , 0);
  Slope_b1= CREATEFIELDALIAS("Slope_b1", Mpx      , 0);
  Slope_v1= CREATEFIELDALIAS("Slope_v1", Mmy      , 0);
  Slope_b2= CREATEFIELDALIAS("Slope_b2", DensStar , 0);
  Slope_v2= CREATEFIELDALIAS("Slope_v2", Qs       , 0);

  Emfx    = CREATEFIELDALIAS("Emfx", Mpz   , EMFX);
  Emfy    = CREATEFIELDALIAS("Emfy", Slope , EMFY);
  Emfz    = CREATEFIELDALIAS("Emfz", DivRho, EMFZ); // Legal ?? we cannot alise DivRho it seems...

  //Claim ownership of storage area
  *(Emfx->owner) = Emfx;
  *(Emfy->owner) = Emfy;
  *(Emfz->owner) = Emfz;
  
  Divergence = CreateField("divb", 0, 0,0,0);  

#endif

}

real ComputeMass() {
  int i,j,k;
  real mass = 0;
  real totalmass;

  real *rho;
  
  INPUT (Density);

  rho = Density->field_cpu;

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
	mass += rho[l]*Vol(j,k);
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
  MPI_Allreduce(&mass, &totalmass, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  masterprint("TotalMass = %3.10f \n", totalmass );
#else
  MPI_Allreduce(&mass, &totalmass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  masterprint("TotalMass = %3.10lf \n", totalmass );
#endif
  return totalmass;
}

void SaveState () {
  Field *current;
  real *backup;
  int size;
  current = ListOfGrids;
  size = sizeof(real)*(Nx+2*NGHX)*(Ny+2*NGHY)*(Nz+2*NGHZ);
  while (current != NULL) {
    if (current->backup == NULL) {
      // No space has been reserved yet for the backup. We take care of that below
      backup = (real *)malloc(size);
      if (backup == NULL) {
	prs_error ("Insufficient memory for check point creation\n");
      }
      current->backup = backup;
    }
    if (*(current->owner) == current) {
      INPUT (current);
      memcpy (current->backup, current->field_cpu, size);
    }
    current = current->next;
  }
  masterprint("\n\n******\nCheck point created\n******\n\n");
}

void SaveStateSecondary () {
  Field *current;
  real *backup;
  int size;
  current = ListOfGrids;
  size = sizeof(real)*(Nx+2*NGHX)*(Ny+2*NGHY)*(Nz+2*NGHZ);
  while (current != NULL) {
    if (current->secondary_backup == NULL) {
      // No space has been reserved yet for the secondary backup. We take care of that below
      backup = (real *)malloc(size);
      if (backup == NULL) {
	prs_error ("Insufficient memory for check point creation\n");
      }
      current->secondary_backup = backup;
    }
    if (*(current->owner) == current) {
      INPUT (current);
      memcpy (current->secondary_backup, current->field_cpu, size);
    }
    current = current->next;
  }
  masterprint("\n\n******\nSecondary Check point created\n******\n\n");
}

void RestoreState () {
  Field *current;
  int size;
  current = ListOfGrids;
  while (current != NULL) {
    if (current->backup == NULL) {
      prs_error ("Cannot restore state: no check point ever created\n");
    }
    size = sizeof(real)*(Nx+2*NGHX)*(Ny+2*NGHY)*(Nz+2*NGHZ);
    if (*(current->owner) == current) {
      memcpy (current->field_cpu, current->backup, size);
      OUTPUT (current);
    }
    current = current->next;
  }
  masterprint("\n\n******\nCheck point restored\n*******\n\n");
}

int RestartSimulation(int n) {

  int begin;
  masterprint("Restarting simulation...\n");

#ifndef MPIIO
  if (VTK)
    __Restart = RestartVTK;
  else
    __Restart = RestartDat;
  
  if (Dat2vtk) {
    Merge = YES;
    __Restart = RestartDat;
  }
  if (Vtk2dat) {
    Merge = YES;
    __Restart = RestartVTK;
  }
  __Restart(Density, n);
#ifdef X
  __Restart(Vx, n);
#endif
#ifdef Y
  __Restart(Vy, n);
#endif
#ifdef Z
  __Restart(Vz, n);
#endif
  __Restart(Energy, n);
#ifdef MHD
  __Restart(Bx, n);
  __Restart(By, n);
  __Restart(Bz, n);
#endif
#endif
  
#ifdef MPIIO
  MPI_Offset offset;
  offset = 0; //We start at the begining of the file
  
  //Density and Energy are mandatory for a restart
  offset = ParallelIO(Density, n, MPI_MODE_RDONLY, offset,FALSE);
  offset = ParallelIO(Energy, n, MPI_MODE_RDONLY, offset,FALSE);
#ifdef X
  //Vx is also mandatory ifdef X
  offset = ParallelIO(Vx, n, MPI_MODE_RDONLY, offset,FALSE);
#endif
#ifdef Y
  //Idem
  offset = ParallelIO(Vy, n, MPI_MODE_RDONLY, offset,FALSE);
#endif
#ifdef Z
  //Idem
  offset = ParallelIO(Vz, n, MPI_MODE_RDONLY, offset,FALSE);
#endif
#ifdef MHD //MHD is 3D.
  if(Fluidtype == GAS){
    offset = ParallelIO(Bx, n, MPI_MODE_RDONLY, offset,FALSE);
    offset = ParallelIO(By, n, MPI_MODE_RDONLY, offset,FALSE);
    offset = ParallelIO(Bz, n, MPI_MODE_RDONLY, offset,FALSE);    
  }
  //We don't need the divergency for a restart
  //offset = ParallelIO(Divergence, n, MPI_MODE_RDONLY, offset, FALSE);
#endif
#endif
  
  begin = n*NINTERM;
  if (PostRestart)
    PostRestartHook ();
  return begin;
}

void RestartVTK(Field *f, int n) {
  int i,j,k,m;
  char filename[200];
  char *name;
  FILE *fi;
  char line[MAXLINELENGTH*max3(NX,NY,NZ)];
  real temp1;
  int temp;
  int origin;
  long curpos;
  int relay;

  static int count = 0;

  name = f->name;

  if (Restart) {
    sprintf(filename, "%s%s%d_%d.vtk", OUTPUTDIR, name, n, CPU_Rank);
    fi = fopen(filename, "r");
    if(fi == NULL) {
      masterprint("Error reading %s\n", filename);
      exit(1);
    }
    
    masterprint("Reading %s\n", filename);
    
    while(1) {
      temp = fscanf(fi, "%s\n", line);
      if (strcmp(line,"LOOKUP_TABLE") == 0){
	temp = fscanf(fi, "%s\n", line);
	break;
      }
    }
    
    i = j = k = 0;

#ifndef SPHERICAL
    for (k=NGHZ; k<Nz+NGHZ; k++) {
      for (i=NGHX; i<Nx+NGHX; i++) {
	for (j=NGHY; j<Ny+NGHY; j++) {
	  temp = fread(&temp1, sizeof(real), 1, fi);
	  f->field_cpu[l] = Swap(temp1);
        }
      }
    }
#else
    for (i=NGHX; i<Nx+NGHX; i++) {
      for (k=NGHZ; k<Nz+NGHZ; k++) {
	for (j=NGHY; j<Ny+NGHY; j++) {
	  temp = fread(&temp1, sizeof(real), 1, fi);
	  f->field_cpu[l] = Swap(temp1);
	}
      }
    }
#endif
  }

  if (Restart_Full) {

    sprintf(filename, "%s%s%d.vtk", OUTPUTDIR, name, n);
    fi = fopen(filename, "r");
    if(fi == NULL) {
      masterprint("Error reading %s\n", filename);
      exit(1);
    }
    
    masterprint("Reading %s\n", filename);
    
    while(1) {
      temp = fscanf(fi, "%s\n", line);
      if (strcmp(line,"LOOKUP_TABLE") == 0){
	temp = fscanf(fi, "%s\n", line);
	curpos = ftell(fi);
	break;
      }
    }
    
    i = j = k = 0;

    origin = Y0+Z0*NX*NY;

#ifndef SPHERICAL
    for (k=NGHZ; k<Nz+NGHZ; k++) {
      for (i=NGHX;i<Nx+NGHX;i++) {
	fseek(fi, curpos+(origin+(i-NGHX)*NY+(k-NGHZ)*NX*NY)*sizeof(real), SEEK_SET);
	for (j=NGHY;j<Ny+NGHY;j++) {
	  temp = fread(&temp1, sizeof(real), 1, fi);
	  f->field_cpu[l] = Swap(temp1);
	}
      }
    }
#else
    for (i=NGHX;i<Nx+NGHX;i++) {
      for (k=NGHZ; k<Nz+NGHZ; k++) {
	fseek(fi, curpos+(origin+(i-NGHX)*NY+(k-NGHZ)*NX*NY)*sizeof(real), SEEK_SET);
	for (j=NGHY;j<Ny+NGHY;j++) {
	  temp = fread(&temp1, sizeof(real), 1, fi);
	  f->field_cpu[l] = Swap(temp1);
	}
      }
    }
#endif

    fclose(fi);
  }
}


void RestartDat(Field *field, int n) {
  int i,j,k;
  real *f;
  char *name;
  char filename[200];
  FILE *fi;
  int origin;

  int temp;

  f = field->field_cpu;
  name = field->name;

  if(Restart == YES) {
    sprintf(filename, "%s%s%d_%d.dat", OUTPUTDIR, name, n, CPU_Rank);
    fi = fopen(filename, "r");
    if(fi == NULL) {
      masterprint("Error reading %s\n", filename);
      exit(1);
    }
    masterprint("Reading %s\n", filename);
    
    for (k=NGHZ; k<Nz+NGHZ; k++) {
      for (j=NGHY; j<Ny+NGHY; j++) {
	temp = fread(f+j*(Nx+2*NGHX)+k*Stride+NGHX, sizeof(real), Nx, fi);
      }
    }
    masterprint("%s OK\n", filename);
    fclose(fi);
    if(Restart_Full == YES) {
      masterprint("Only one restart option must be enabled.\n");
      MPI_Finalize();
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  if(Restart_Full == YES) {
    sprintf(filename, "%s%s%d.dat", OUTPUTDIR, name, n);
    fi = fopen(filename, "r");
    if(fi == NULL) {
      masterprint("Error reading %s\n", filename);
      exit(1);
    }
    masterprint("Reading %s\n", filename);
    
    origin = (z0cell)*NX*NY + (y0cell)*NX; //z0cell and y0cell are global variables.
    for (k=NGHZ; k<Nz+NGHZ; k++) {
      fseek(fi, (origin+(k-NGHZ)*NX*NY)*sizeof(real), SEEK_SET); // critical part
      for (j=NGHY; j < Ny+NGHY; j++)
	temp = fread(f+k*Stride+j*(Nx+2*NGHX)+NGHX, sizeof(real), Nx, fi);
    }
    masterprint("%s OK\n", filename);
    fclose(fi);
    if(Restart == YES) {
      masterprint("Only one restart option must be enabled.\n");
      MPI_Finalize();
    }
  }
}
