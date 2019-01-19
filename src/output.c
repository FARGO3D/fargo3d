#include "fargo3d.h"

void SelectWriteMethod() {

  __WriteField = WriteFieldGhost;
  
#if !defined(WRITEGHOSTS)
  masterprint ("I do not output the ghost values\n");
  __WriteField = WriteField;
  if (Merge) {
    __WriteField = WriteMerging;
  }
#endif
  
  if (VTK) {
    __WriteField = WriteVTK;
    if (Merge) {
      __WriteField = WriteVTKMerging;
    }
  }
  
  if (Dat2vtk) {
    __WriteField = WriteVTKMerging;
  }

  if (Vtk2dat) {
    __WriteField = WriteMerging;
  }
  
}

void EmptyPlanetSystemFiles () {
  FILE *output;
  char name[256];
  int i, n;
  if (Sys == NULL) return;
  n = Sys->nb;
  if (!CPU_Master) return;
  for (i = 0; i < n; i++) {
    sprintf (name, "%splanet%d.dat", OUTPUTDIR, i);
    output = fopen_prs(name, "w"); /* This empties the file */
    fclose (output);
  }
}  

void WritePlanetFile (int TimeStep, int n, boolean big) {
  FILE *output;
  char name[256];

  if (Sys == NULL) return;

  if (!CPU_Master) return;
  //  printf ("Updating 'planet%d.dat'...", n);
  //fflush (stdout);

  if (big == YES)
    sprintf (name, "%sbigplanet%d.dat", OUTPUTDIR, n);
  else
    sprintf (name, "%splanet%d.dat", OUTPUTDIR, n);
  output = fopen_prs (name, "a");

  fprintf (output, "%d\t%#.18g\t%#.18g\t%#.18g\t%#.18g\t%#.18g\t%#.18g\t%#.18g\t%#.18g\t%#.18g\n",
	   TimeStep,
	   Xplanet,
	   Yplanet,
	   Zplanet,
	   VXplanet,
	   VYplanet,
	   VZplanet,
	   MplanetVirtual,
	   PhysicalTime,
	   OMEGAFRAME);
  fclose (output);
  //printf ("done\n");
  fflush (stdout);
}

real GetfromPlanetFile (TimeStep, column, n)
int TimeStep, column, n;
{
  FILE *input;
  char name[256];
  char testline[256];
  int time;
  char *pt;
  double value;
  sprintf (name, "%splanet%d.dat", OUTPUTDIR, n);
  input = fopen (name, "r");
  if (input == NULL) {
    mastererr ("Can't read 'planet%d.dat' file. Aborting restart.\n",n);
    prs_exit (1);
  }
  if (column < 2) {
    mastererr ("Invalid column number in 'planet%d.dat'. Aborting restart.\n",n);
    prs_exit (1);
  }
  do {
    pt = fgets (testline, 255, input);
    sscanf (testline, "%d", &time);
  } while ((time != TimeStep) && (pt != NULL));
  if (pt == NULL) {
    mastererr ("Can't read entry %d in 'planet%d.dat' file. Aborting restart.\n", TimeStep,n);
    prs_exit (1);
  }
  fclose (input);
  pt = testline;
  while (column > 1) {
    pt += strspn(pt, "eE0123456789-+.");
    pt += strspn(pt, "\t :=>_");
    column--;
  }
  sscanf (pt, "%lf", &value);
  return (real)value;
}

void RestartPlanetarySystem (timestep, sys)
PlanetarySystem *sys;
int timestep;
{
  int k;
  for (k = 0; k < sys->nb; k++) {
    sys->x[k] = GetfromPlanetFile (timestep, 2, k);
    sys->y[k] = GetfromPlanetFile (timestep, 3, k);
    sys->z[k] = GetfromPlanetFile (timestep, 4, k);
    sys->vx[k] = GetfromPlanetFile (timestep, 5, k);
    sys->vy[k] = GetfromPlanetFile (timestep, 6, k);
    sys->vz[k] = GetfromPlanetFile (timestep, 7, k);
    sys->mass[k] = GetfromPlanetFile (timestep, 8, k);
  }
}

void WriteTorqueAndWork(int TimeStep, int n) {

  FILE *output;
  char name[256];

  Force force;

  real x,y,z;
  real vx,vy,vz;
  real m,r,smoothing;

  if (Sys == NULL) return;

  m = Sys->mass[n];
  x = Sys->x[n];
  y = Sys->y[n];
  z = Sys->z[n];
  r = sqrt(x*x + y*y + z*z);
  if (ROCHESMOOTHING != 0)
    smoothing = r*pow(m/3./MSTAR,1./3.)*ROCHESMOOTHING;
  else
    smoothing = ASPECTRATIO*pow(r/R0,FLARINGINDEX)*r*THICKNESSSMOOTHING;

  force = ComputeForce(x,y,z,smoothing,m);
  //  printf ("Updating 'planet%d.dat'...", n);
  //fflush (stdout);

  sprintf (name, "%stqwk%d.dat", OUTPUTDIR, n);

  x = Xplanet;
  y = Yplanet;
  z = Zplanet;
  vx = VXplanet;
  vy = VYplanet;
  vz = VZplanet;
  m = MplanetVirtual;
  r = sqrt(x*x + y*y + z*z);

  if (ROCHESMOOTHING != 0)
    smoothing = r*pow(m/3./MSTAR,1./3.)*ROCHESMOOTHING;
  else
    smoothing = ASPECTRATIO*pow(r/R0,FLARINGINDEX)*r*THICKNESSSMOOTHING;
  
  if (!CPU_Master) return;
  output = fopen_prs (name, "a");
  fprintf (output, "%d\t%.18g\t%.18g\t%.18g\t%.18g\t%.18g\t%.18g\t%.18g\t%.18g\t%.18g\n",
	   TimeStep,
	   x*force.fy_inner-y*force.fx_inner,
	   x*force.fy_outer-y*force.fx_outer,
	   x*force.fy_ex_inner-y*force.fx_ex_inner,
	   x*force.fy_ex_outer-y*force.fx_ex_outer,
	   vx*force.fx_inner+vy*force.fy_inner,
	   vx*force.fx_outer+vy*force.fy_outer,
	   vx*force.fx_ex_inner+vy*force.fy_ex_inner,
	   vx*force.fx_ex_outer+vy*force.fy_ex_outer,
	   PhysicalTime);
  fclose (output);
  //  printf ("done\n");
  fflush (stdout);
}


void WritePlanetSystemFile (int t, boolean big) {
  int i, n;
  if (Sys == NULL) return;
  n = Sys->nb;
  for (i = 0; i < n; i++) {
    Xplanet  = Sys->x[i];
    Yplanet  = Sys->y[i];
    Zplanet  = Sys->z[i];
    VXplanet = Sys->vx[i];
    VYplanet = Sys->vy[i];
    VZplanet = Sys->vz[i];
    MplanetVirtual = Sys->mass[i];
    //    WriteTorqueAndWork(t, i);
    WritePlanetFile (t, i, big);
  }
}

void WriteDim () {
  char filename[200];
  char command[MAXLINELENGTH];
  FILE *dims;
  int temp;

  if(CPU_Rank==0) {
    sprintf(filename, "%sdimensions.dat", OUTPUTDIR);
    dims = fopen(filename, "w");
    fprintf(dims,"%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
	    "#XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX",	\
	    "NX", "NY", "NZ", "CPUS_Y", "CPUS_Z", "NGHY", "NGHZ", "NGHX");		
    fprintf(dims,"%f\t%f\t%f\t%f\t%f\t%f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",	\
    	    XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX, NX, NY, NZ,	\
    	    Ncpu_x, Ncpu_y, NGHY, NGHZ, NGHX);
    fclose(dims);
#ifdef LEGACY
    sprintf(filename, "%sdims.dat", OUTPUTDIR);
    dims = fopen(filename, "w");
    fprintf(dims,"%d\t%d\t%d\t%d\t%f\t%d\t%d\t%d\n",	\
    	    0, 0, 0, 0, YMAX, NTOT/NINTERM, NY, NX);
    fclose(dims);
    sprintf (command, "cp %sdomain_y.dat %stemprad", OUTPUTDIR, OUTPUTDIR);
    temp = system (command);
    sprintf (command, "tail -n %d %stemprad | head -n %d > %sused_rad.dat",NY+1+NGHY,\
	     OUTPUTDIR, NY+1, OUTPUTDIR);
    temp = system (command);
    sprintf (command, "rm -f %stemprad", OUTPUTDIR);
    temp = system (command);
#endif
  }
}

void WriteField2D(Field2D *f, int n) {
  char filename[200];
  FILE *fo;
  INPUT2D (f);
  sprintf(filename, "%s%s%d_%d.dat", OUTPUTDIR, f->name, n, CPU_Rank);
  fo = fopen(filename,"w");
  fwrite(f->field_cpu, sizeof(real), (Ny+2*NGHY)*(Nz+2*NGHZ), fo);
  fclose(fo);
}

void WriteFieldInt2D(FieldInt2D *f, int n) {
  char filename[200];
  FILE *fo;
  INPUT2DINT (f);
  sprintf(filename, "%s%s%d_%d.dat", OUTPUTDIR, f->name, n, CPU_Rank);
  fo = fopen(filename,"w");
  fwrite(f->field_cpu, sizeof(int), (Ny+2*NGHY)*(Nz+2*NGHZ), fo);
  fclose(fo);
}

void WriteField(Field *f, int n) {
  int i,j,k;
  char filename[200];
  FILE *fo;
  INPUT (f);
  sprintf(filename, "%s%s%d_%d.dat", OUTPUTDIR, f->name, n, CPU_Rank);
  fo = fopen(filename,"w");
  for (k=NGHZ; k<Nz+NGHZ; k++) { //Write grid without ghost cells
    for (j=NGHY; j<Ny+NGHY; j++) {
      fwrite(f->field_cpu+j*(Nx+2*NGHX)+k*Stride+NGHX, sizeof(real), Nx, fo);
    }
  }
  fclose(fo);
}

void WriteFieldGhost(Field *f, int n) { // Diagnostic function
  int i,j,k;
  char filename[200];
  FILE *fo;
  INPUT (f);
  sprintf(filename, "%s%s%d_%d.dat", OUTPUTDIR, f->name, n, CPU_Rank);
  fo = fopen(filename,"w");
  for (k=0; k<Nz+2*NGHZ; k++) { //Write grid with ghost cells
    for (j=0; j<Ny+2*NGHY; j++) {
      fwrite(f->field_cpu+j*(Nx+2*NGHX)+k*Stride, sizeof(real), Nx+2*NGHX, fo);
    }
  }
  fclose(fo);
}

void WriteMerging(Field *f, int n) {
  INPUT(f);

  FILE *fo;
  int i,j,k,m,jj;
  char outname[MAXLINELENGTH];
  int next, previous;
  int relay;

  sprintf(outname, "%s%s%d.dat", OUTPUTDIR, f->name, n);

  if (CPU_Rank > 0) { // Force sequential read
    MPI_Recv (&relay, 1, MPI_INT, CPU_Rank-1, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (CPU_Master){ //An inefficient way to delete a file...
    fo = fopen(outname, "w");
    fclose(fo);
    fo = fopen(outname, "a+");
  }
  else 
    fo = fopen(outname, "a+");

  if (CPU_Rank < CPU_Number-1) {  // Force sequential read
    MPI_Send (&relay, 1, MPI_INT, CPU_Rank+1, 42, MPI_COMM_WORLD);
  }


  for (k=0; k<NZ; k++) {
    for (j = 0; j<Ncpu_x; j++) {
      if ((J==j) && (k>=Z0) && (k<(Z0+Nz))) {
	for (jj = NGHY; jj < Ny+NGHY; jj++)
	  fwrite(f->field_cpu+(k-Z0+NGHZ)*Stride+jj*(Nx+2*NGHX)+NGHX, sizeof(real)*Nx, 1, fo);
      }
      fflush(fo);
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
  fclose(fo);
}

void WriteBinFile(int n1, int n2, int n3,	\
		  real *var1, char *filename) {
  int i,j,k;
  int ntemp; 
  FILE *F;
  F = fopen(filename,"w"); 
  if(F == NULL) prs_exit(1);
  
  static float *var;
  static boolean init=TRUE;
  
  if(init) {
    var = (float*)malloc(sizeof(float)*n1*n2*n3);
    init = FALSE;
  }

  for(i=0; i<n1*n2*n3; i++) {
    var[i] = (float)var1[i];
  }

  ntemp = 12;
  fwrite(&ntemp,4,1,F);
  fwrite(&n1,4,1,F);
  fwrite(&n2,4,1,F);
  fwrite(&n3,4,1,F);
  fwrite(&ntemp,4,1,F);
  
  ntemp = n1*n2*n3*sizeof(float);
  fwrite(&ntemp,4,1,F); fwrite(var,sizeof(float)*n1*n2*n3,1,F); fwrite(&ntemp,4,1,F);
  fclose(F);
}

void DumpAllFields (int number) {
  Field *current;
  current = ListOfGrids;
  printf ("Dumping at #%d\t", number);
  while (current != NULL) {
    if (*(current->owner) == current) {
      if (!CPU_Rank)
	printf ("%s ", current->name);
      __WriteField (current, number);
    }
    current = current->next;
  }
  printf ("\n");
}

void WriteOutputsAndDisplay(int type) {

 /* If type=ALL, all fields are dumped (This is the old fashion
     style). If type=SPECIFIC, this routine only dumps specific
     fields, given by the .par variables WRITE+FIELD. By default all
     WRITE parameters are NO. */ 
  
  boolean writedensity;
  boolean writeenergy;
  boolean writedivergence;
  boolean writebx;
  boolean writeby;
  boolean writebz;
  boolean writevx;
  boolean writevy;
  boolean writevz;
  boolean writeenergyrad;
  boolean writetau;

  char outputdir[MAXLINELENGTH];
  static int init = 0;

  Summary (TimeStep);
  
  if (type == ALL){ //We store the .par variables' value for a while.
    writedensity = WRITEDENSITY;
    writeenergy = WRITEENERGY;
    writebx = WRITEBX;
    writeby = WRITEBY;
    writebz = WRITEBZ;
    writevx = WRITEVX;
    writevy = WRITEVY;
    writevz = WRITEVZ;
    writeenergyrad = WRITEENERGYRAD;
    writetau = WRITETAU;
    WRITEDENSITY = YES;
    WRITEENERGY = YES;
    WRITEBX = YES;
    WRITEBY = YES;
    WRITEBZ = YES;
    WRITEVX = YES;
    WRITEVY = YES;
    WRITEVZ = YES;
    WRITEENERGYRAD = YES;
    WRITETAU = YES;
  }
  else {
    sprintf(outputdir,"%s",OUTPUTDIR);
    sprintf(OUTPUTDIR,"%ssnaps/",outputdir);
    if (init == 0) {
      MakeDir(OUTPUTDIR);
      init = 1;
    }
  }

  if (WRITEDENSITY)
    __WriteField(Density, TimeStep);
  if (WRITEENERGY)
    __WriteField(Energy, TimeStep);
#ifdef MHD //MHD is 3D.
  if (WRITEDIVERGENCE)
    __WriteField(Divergence,TimeStep);
  if (WRITEBX)
    __WriteField(Bx, TimeStep);
  if (WRITEBY)
    __WriteField(By, TimeStep);
  if (WRITEBZ)
    __WriteField(Bz, TimeStep);
#endif
#ifdef X
  if (WRITEVX)
    __WriteField(Vx, TimeStep);
#endif
#ifdef Y
  if (WRITEVY)
    __WriteField(Vy, TimeStep);
#endif
#ifdef Z
  if (WRITEVZ)
    __WriteField(Vz, TimeStep);
#endif
  
if (type == ALL){ //We recover the .par variables' value
    WRITEDENSITY = writedensity;
    WRITEENERGY = writeenergy;
    WRITEBX = writebx;
    WRITEBY = writeby;
    WRITEBZ = writebz;
    WRITEVX = writevx;
    WRITEVY = writevy;
    WRITEVZ = writevz;
  }

  if(Vtk2dat)
    prs_exit(EXIT_SUCCESS);
  if(Dat2vtk)
    prs_exit(EXIT_SUCCESS);

#if ((defined(X) && defined(Y) && !defined(Z)) || \
     (defined(X) && defined(Z) && !defined(Y)) || \
     (defined(Y) && defined(Z) && !defined(X)))
#ifdef MATPLOTLIB
  if (Merge) {
    if (CPU_Master) {
      plot2d(FIELD, TimeStep, Merge);
    }
  }
  else {
    plot2d(FIELD, TimeStep, Merge);
  }
#endif
#endif

#if (defined(X) && defined(Y) && defined(Z))
#ifdef MATPLOTLIB
  if (Merge) {
    if (CPU_Master) {
      plot3d(FIELD, TimeStep, Merge);
    }
  }
  else {
    plot3d(FIELD, TimeStep, Merge);
  }
#endif
#endif
  
#if ((defined(X) & !(defined(Y) || defined(Z))) ||  \
     (defined(Y) & !(defined(X) || defined(Z)))  || \
     (defined(Z) & !(defined(X) || defined(Y))))
#ifdef MATPLOTLIB
  plot1d(FIELD, TimeStep, Merge);
#endif
#endif

 if (type != ALL)
    sprintf(OUTPUTDIR,"%s",outputdir);

  if (OnlyInit) 
    prs_exit(EXIT_SUCCESS);
}
