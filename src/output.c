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
  vx = Sys->vx[n];
  vy = Sys->vy[n];
  vz = Sys->vz[n];
  r = sqrt(x*x + y*y + z*z);
  
  if (ROCHESMOOTHING != 0)
    smoothing = r*pow(m/3./MSTAR,1./3.)*ROCHESMOOTHING;
  else
    smoothing = ASPECTRATIO*pow(r/R0,FLARINGINDEX)*r*THICKNESSSMOOTHING;

  force = ComputeForce(x,y,z,smoothing,m);

  sprintf (name, "%stqwk%d.dat", OUTPUTDIR, n);

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
    WriteTorqueAndWork(t, i);
    WritePlanetFile (t, i, big);
  }
}

void WriteDim () {
  char filename[200];
  char command[MAXLINELENGTH];
  FILE *dims;
  int temp;

  if(CPU_Rank==0) {
#ifdef DEBUG
    sprintf(filename, "%sdimensions.dat", OUTPUTDIR);
    dims = fopen(filename, "w");
    fprintf(dims,"%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
	    "#XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX",	\
	    "NX", "NY", "NZ", "CPUS_Y", "CPUS_Z", "NGHY", "NGHZ", "NGHX");		
    fprintf(dims,"%f\t%f\t%f\t%f\t%f\t%f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",	\
    	    XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX, NX, NY, NZ,	\
    	    Ncpu_x, Ncpu_y, NGHY, NGHZ, NGHX);
    fclose(dims);
#endif
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

void Write_offset(int file_offset, char* fieldname, char* fluidname){

  FILE *fp;
  char filename[MAXNAMELENGTH];

  if(CPU_Master){
    sprintf(filename, "%s/output%s.dat", OUTPUTDIR, fluidname);
    fp = fopen(filename,"a");
    fprintf(fp, "%d\t%s\n", file_offset, fieldname);
    fclose(fp);
  }
  
}

#ifdef MPIIO
MPI_Offset ParallelIO(Field *field, int n, int mode, MPI_Offset file_offset, int writeoffset) {

  INPUT(field);
  
  char filename[MAXNAMELENGTH];

  //subarray of memory (fields)
  MPI_Datatype mpi_memtype;
  int mem_global_size[3]; 
  int mem_local_size[3];
  int mem_start[3];
  //subarray of file (fields)
  MPI_Datatype mpi_filetype;
  int file_global_size[3];
  int file_local_size[3];
  int file_start[3];
  //---------------------

  MPI_Info mpi_info;
  MPI_File mpi_file;
  MPI_Status status;

  sprintf(filename, "%s%s_%d.mpiio", OUTPUTDIR, Fluids[FluidIndex]->name, n);

  //Setting memory space per rank
  mem_global_size[0] = Nz+2*NGHZ;
  mem_global_size[1] = Ny+2*NGHY;
  mem_global_size[2] = Nx;

  
#if !defined(WRITEGHOSTS)
  mem_local_size[0] = Nz;
  mem_local_size[1] = Ny;
  mem_local_size[2] = Nx;

  mem_start[0] = NGHZ;
  mem_start[1] = NGHY;
  mem_start[2] = NGHX;
#else
  if (CPU_Number>1) {
    mastererr("This version doesn't write ghosts in the parallel version.\n");
    exit(1);
  }
  mem_local_size[0] = Nz+2*NGHZ;
  mem_local_size[1] = Ny+2*NGHY;
  mem_local_size[2] = Nx+2*NGHX;

  mem_start[0] = 0;
  mem_start[1] = 0;
  mem_start[2] = 0;
#endif
  
  MPI_Type_create_subarray(3, mem_global_size, mem_local_size, mem_start,  
			   MPI_ORDER_C, MPI_DOUBLE, &mpi_memtype);
  MPI_Type_commit(&mpi_memtype);

  //Setting file space

#if !defined(WRITEGHOSTS)
  file_global_size[0] = NZ;
  file_global_size[1] = NY;
  file_global_size[2] = NX;

  file_local_size[0] = Nz;
  file_local_size[1] = Ny;
  file_local_size[2] = Nx;
#else
  file_global_size[0] = NZ+2*NGHZ;
  file_global_size[1] = NY+2*NGHY;
  file_global_size[2] = NX+2*NGHX;

  file_local_size[0] = Nz+2*NGHZ;
  file_local_size[1] = Ny+2*NGHY;
  file_local_size[2] = Nx+2*NGHX;
#endif
  file_start[0] = Z0;
  file_start[1] = Y0;
  file_start[2] = 0;

  MPI_Type_create_subarray(3, file_global_size, file_local_size, file_start,  
			   MPI_ORDER_C, MPI_DOUBLE, &mpi_filetype);
  MPI_Type_commit(&mpi_filetype);

  
  //Writing.....
  MPI_File_open(MPI_COMM_WORLD, filename, mode, MPI_INFO_NULL, &mpi_file);
  
  //We write the only at the begining of the file
  if (file_offset == 0) {
    if (mode & MPI_MODE_WRONLY) {
      if (CPU_Master)
	MPI_File_write_at(mpi_file, 0, Xmin, NX+1, MPI_DOUBLE, &status);
    }
    else {
      MPI_File_read_at(mpi_file, 0, Xmin, NX+1, MPI_DOUBLE, &status);
    }    

    file_offset += NX+1;
    
    if (mode & MPI_MODE_WRONLY) {
      if (Z0 == 0) {
	if (J == (Ncpu_x - 1))
	  MPI_File_write_at(mpi_file, (file_offset+Y0)*sizeof(real),
			    Ymin+NGHY, Ny+1, MPI_DOUBLE, &status);
	else
	  MPI_File_write_at(mpi_file, (file_offset+Y0)*sizeof(real),
			    Ymin+NGHY, Ny, MPI_DOUBLE, &status);
 
      }

      file_offset += NY+1;

      if (Y0 == 0) {
	if (K == (Ncpu_y - 1))
	  MPI_File_write_at(mpi_file, (file_offset+Z0)*sizeof(real),
			    Zmin+NGHZ, Nz+1, MPI_DOUBLE, &status);
	else
	  MPI_File_write_at(mpi_file, (file_offset+Z0)*sizeof(real),
			    Zmin+NGHZ, Nz, MPI_DOUBLE, &status);
      }
      
      file_offset += (NZ+1);
          
    }
    else { //If LOADFIELDS
      if (J == (Ncpu_x - 1))
	MPI_File_read_at(mpi_file, (file_offset+Y0)*sizeof(real),
			 Ymin+NGHY, Ny+1, MPI_DOUBLE, &status);
      else
	MPI_File_read_at(mpi_file, (file_offset+Y0)*sizeof(real),
			 Ymin+NGHY, Ny, MPI_DOUBLE, &status);

    file_offset += NY+1;
    
    if (K == (Ncpu_y - 1))
      MPI_File_read_at(mpi_file, (file_offset+Z0)*sizeof(real),
		       Zmin+NGHZ, Nz+1, MPI_DOUBLE, &status);
    else
      MPI_File_read_at(mpi_file, (file_offset+Z0)*sizeof(real),
		       Zmin+NGHZ, Nz, MPI_DOUBLE, &status);

    file_offset += (NZ+1);
    
    }
  }

  //We append more fields on the same file

  MPI_File_set_view(mpi_file, file_offset*sizeof(real), MPI_DOUBLE, mpi_filetype,
		    "native", MPI_INFO_NULL);

  if (mode & MPI_MODE_WRONLY)
    MPI_File_write_all(mpi_file, field->field_cpu, 1,
		       mpi_memtype, &status);
  else
    MPI_File_read_all(mpi_file, field->field_cpu, 1,
		      mpi_memtype, &status);

  if(writeoffset == TRUE ) Write_offset(file_offset, field->name, Fluids[FluidIndex]->name);  
#if !defined(WRITEGHOSTS)
  file_offset += NX*NY*NZ;
#else
  file_offset += (NX+2*NGHX)*(NY+2*NGHY)*(NZ+2*NGHZ);
#endif
  
  MPI_File_close(&mpi_file);
  return file_offset;

}
#endif

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

void WriteOutputs(int type) {

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
  static int writeoffset = TRUE;
  static int counter = 0;
  MPI_Offset offset;

  FILE *fp;
  char filename[MAXNAMELENGTH];

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
  // we truncate the output.dat file
  if(CPU_Master && (writeoffset == TRUE)) {
    sprintf(filename, "%s/output%s.dat", OUTPUTDIR, Fluids[FluidIndex]->name);
    fp = fopen(filename,"w");
    fclose(fp);
  }
  
  /// MPIIO ouput version
#ifdef MPIIO
  offset = 0; //We start at the begining of the file  
 
  if (WRITEDENSITY)
    offset = ParallelIO(Density, TimeStep, MPI_MODE_WRONLY|MPI_MODE_CREATE, offset,writeoffset);
  if (WRITEENERGY)
    offset = ParallelIO(Energy, TimeStep, MPI_MODE_WRONLY|MPI_MODE_CREATE, offset,writeoffset);
#ifdef X
  if (WRITEVX)
    offset = ParallelIO(Vx, TimeStep, MPI_MODE_WRONLY|MPI_MODE_CREATE, offset,writeoffset);
#endif
#ifdef Y
  if (WRITEVY)
    offset = ParallelIO(Vy, TimeStep, MPI_MODE_WRONLY|MPI_MODE_CREATE, offset,writeoffset);
#endif
#ifdef Z
  if (WRITEVZ)
    offset = ParallelIO(Vz, TimeStep, MPI_MODE_WRONLY|MPI_MODE_CREATE, offset,writeoffset);
#endif
#ifdef MHD //MHD is 3D.
  if(Fluidtype == GAS){
    if (WRITEBX)
      offset = ParallelIO(Bx, TimeStep, MPI_MODE_WRONLY|MPI_MODE_CREATE, offset,writeoffset);
    if (WRITEBY)
      offset = ParallelIO(By, TimeStep, MPI_MODE_WRONLY|MPI_MODE_CREATE, offset,writeoffset);
    if (WRITEBZ)
      offset = ParallelIO(Bz, TimeStep, MPI_MODE_WRONLY|MPI_MODE_CREATE, offset,writeoffset);
    if (WRITEDIVERGENCE)
      offset = ParallelIO(Divergence, TimeStep, MPI_MODE_WRONLY|MPI_MODE_CREATE, offset,writeoffset);
  }
#endif

  if (counter < 2*NFLUIDS) {
    if (NSNAP == 0) {
      if (counter == NFLUIDS-1 || Restart == YES || Restart_Full == YES) {
	writeoffset = FALSE;
      }
    }
    else {
      if (counter == 2*NFLUIDS-1 || Restart == YES || Restart_Full == YES) {
	writeoffset = FALSE;
      }
    }
    counter += 1;
  }
#endif

  /// Standard ouput version
#ifndef MPIIO
  if (WRITEDENSITY)
    __WriteField(Density, TimeStep);
  if (WRITEENERGY)
    __WriteField(Energy, TimeStep);
#ifdef MHD //MHD is 3D.
  if(Fluidtype == GAS){
    if (WRITEDIVERGENCE)
      __WriteField(Divergence,TimeStep);
    if (WRITEBX)
      __WriteField(Bx, TimeStep);
    if (WRITEBY)
      __WriteField(By, TimeStep);
    if (WRITEBZ)
      __WriteField(Bz, TimeStep);
  }
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

 if (type != ALL)
    sprintf(OUTPUTDIR,"%s",outputdir);

  if (OnlyInit) 
    prs_exit(EXIT_SUCCESS);
}
