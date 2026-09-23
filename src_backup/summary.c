#include <fargo3d.h>

extern Param Var_Set[];
extern int Id_Var;
char dummy_end='\0';

void SelectArchFileName () {
  int i=0;
  char Fullname[1024];
  if (!CPU_Master)
    return;
  sprintf (Fullname, "%s/sources%d.tar.bz2", OUTPUTDIR, i);
  sprintf (ArchFile, "sources%d.tar.bz2", i);
  while ( access(Fullname, F_OK ) != -1 ) {
    i++;
    sprintf (Fullname, "%s/sources%d.tar.bz2", OUTPUTDIR, i);
    sprintf (ArchFile, "sources%d.tar.bz2", i);
  }
}

void StoreFileToChar (char **string, char *filename) {
  FILE *input;
  long size;
  if (CPU_Master) {
    input = fopen (filename, "r");
    if (input == NULL) {
      mastererr ("Cannot open %s in StoreFileToChar.\n", filename);
      prs_exit (1);
    }
    fseek (input, 0, SEEK_END);
    size = ftell(input);
    fseek (input, 0, SEEK_SET);
    *string = (char *)malloc(size+1);
    if (fread (*string, 1, size, input) != size) {
      mastererr ("Could not read correctly %s\n", filename);
      prs_exit (1);
    }
    fclose (input);
    (*string)[size]=0;
  }
}

char *ExtractFromExecutable (boolean tostring, char *filename, int position) {
  FILE *input;
  FILE *output;
  char execname[1024];
  char cwd[1024];
  char string[1024];
  char *longstring;
  int length, count=0, foo;
  long lfoo;
  /* We search the full path of the executable */
  /* We assume it is not in the PATH variable (there is no simple,
     portable way to find the full path in this case) */
  if (CPU_Rank > 0)
    return &dummy_end;
  lfoo = (long)getcwd(CurrentWorkingDirectory, 1023);
  sprintf (execname, "%s/%s", CurrentWorkingDirectory, FirstCommand);
  input = fopen (execname, "r");
  if (input == NULL) {
    mastererr ("Cannot open executable file to retrieve information appended.\n");
    //We test without CurrentWorking directory because SLURM put as
    //first command the full path to the executable.
    sprintf (execname, "%s",FirstCommand);
    input = fopen (execname, "r");
    if (input == NULL) {
      mastererr ("Cannot open executable file to retrieve information appended.\n");
      prs_exit (1);
    } 
  }
  fseek (input, 0, SEEK_END);
  do {
    if (fseek (input, -12, SEEK_CUR) < 0) {
      mastererr ("Fseek problem in 'ExtractFromExecutable' (1)\n");
      prs_exit (1);
    }
    foo = fscanf(input, "%12s", string);
    length = atoi(string);
    if (fseek (input, -12-length, SEEK_CUR) < 0) {
      mastererr ("Fseek problem in 'ExtractFromExecutable (2)'\n");
      prs_exit (1);
    }
    count++;
  } while (count < position);
  longstring = (char *)malloc(length+1);
  if (longstring == NULL) {
    mastererr ("Not enough memory in 'ExtractFromExecutable'\n");
    prs_exit (1);
  }
  foo = fread (longstring, length, 1, input);
  fclose(input);
  longstring[length]=0;
  if (tostring == YES)
    return longstring;
  else {
    sprintf (cwd, "%s/%s", OUTPUTDIR, filename);
    output = fopen (cwd, "w");
    if (output == NULL) {
      mastererr ("Could not open file %s in 'ExtractFromExecutable'\n", cwd);
      prs_exit (1);
    }
    fwrite (longstring, 1, length, output);
    free (longstring);
    fclose (output);
  }
  return NULL;
}

void GetHostsList () {
  char strdevice[1024];
  char hostname[1024];
  static boolean SpaceReservedForHostnames=NO;
  int device;
  gethostname(hostname,1023);
#ifdef GPU
  strcat (hostname, " (device ");
  cudaGetDevice (&device);
  sprintf (strdevice, "%d", device);
  strcat (hostname, strdevice);
  strcat (hostname, ")");
#endif
  if (SpaceReservedForHostnames == NO) {
    if (CPU_Master) {
      HostsList = (char *)malloc(1024*CPU_Number);
      if (HostsList == NULL) {
	mastererr ("Out of memory in 'summary.c::GetHostsList()'\n");
	prs_exit (1);
      }
    }
    SpaceReservedForHostnames = YES;
  }
  MPI_Gather (hostname, 1024, MPI_CHAR, HostsList, 1024, MPI_CHAR, 0, MPI_COMM_WORLD);
}

void Summary (int nout) {
  FILE *sum;
  static boolean FirstTime = YES;
  char filename[1024];
  char hostname[1024];
  char sep[]="==============================\n";
  static char *hosts;
  time_t t;
  struct tm tm;
  int i, type, n;
  if (FirstTime == YES) {
    StoreFileToChar (&InputFile, ParameterFile);
    if (ThereArePlanets)
      StoreFileToChar (&PlanetaryFile, PLANETCONFIG);
    FirstTime = NO;
  }
  if (CPU_Master) {
    sprintf (filename, "%s/summary%d.dat", OUTPUTDIR, nout);
    sum = fopen_prs (filename, "w");
    fprintf (sum, "%sSUMMARY:\n%s",sep,sep);
    fprintf (sum, "SETUP '%s' of %s\n", xstr(SETUPNAME), VersionString);
#ifdef CARTESIAN
    fprintf (sum, "Cartesian ");
#endif
#ifdef CYLINDRICAL
    fprintf (sum, "Cylindrical ");
#endif
#ifdef SPHERICAL
    fprintf (sum, "Spherical ");
#endif
    fprintf (sum, "mesh of size %d x %d x %d (%d cells in total)\n", NX, NY, NZ, NX*NY*NZ);
    fprintf (sum, "%d outputs scheduled\n", NTOT/NINTERM);
#ifdef LONGSUMMARY
    fprintf (sum, "Source file archive: %s\n", ArchFile);
#else
    fprintf (sum, "LONGSUMMARY sticky flag not activated: no source file archive\n");
#endif
    if (ThereArePlanets)
      fprintf (sum, "%d planet%s\n", Sys->nb, (Sys->nb > 1 ? "s" : "")); 
    fprintf (sum, "\n%sCOMPILATION OPTION SECTION:\n%s",sep,sep);
    fprintf (sum, "%s\n", xstr(OPTIONS));
    fprintf (sum, "Ghost layer sizes: NGHX=%d\tNGHY=%d\tNGHZ=%d\n",\
	     NGHX, NGHY, NGHZ);
#ifdef LONGSUMMARY
    fprintf (sum, "\n%sSTICKY FLAGS SECTION:\n%s",sep,sep);
    fprintf (sum, "%s", StickyOptions);
#endif
    fprintf (sum, "\n%sRUNTIME GENERAL SECTION:\n%s",sep,sep);
    fprintf (sum, "Current Working Directory is %s\n", CurrentWorkingDirectory);
    fprintf (sum, "Command line: %s\n", CommandLine);
    fprintf (sum, "Parameter file: %s\n", ParameterFile);
    fprintf (sum, "Run on %d process%s\nHosts:\n", CPU_Number,\
	     (CPU_Number > 1 ? "es" : ""));
    for (i = 0; i < CPU_Number; i++) {
      fprintf (sum, "   Rank %d on %s\n", i, HostsList+i*1024);
    }
    fprintf (sum, "\n%sOUTPUT SPECIFIC SECTION:\n%s",sep,sep);
    fprintf (sum, "OUTPUT %d at simulation time %g ", nout, PhysicalTime);
    t = time(NULL);
    tm = *localtime(&t);
    fprintf(sum, "(%d-%d-%d %02d:%02d:%02d)\n", tm.tm_year + 1900,\
	    tm.tm_mon + 1, tm.tm_mday, tm.tm_hour,\
	    tm.tm_min, tm.tm_sec);
    fprintf (sum, "\n%sPREPROCESSOR MACROS SECTION:\n%s",sep,sep);
    DUMP_PPVAR (R0);
    DUMP_PPVAR (R_MU);
    DUMP_PPVAR (MU0);
    DUMP_PPVAR (MSTAR);
    DUMP_PPVAR (G);
    DUMP_PPVAR (STEFANK);
    DUMP_PPVAR (C0);
    DUMP_PPVAR (TCMB);
    DUMP_PPVAR (THRESHOLD_STELLAR_MASS);
    fprintf (sum, "\n%sPARAMETERS SECTION:\n%s",sep,sep);
    for (i = 0; i < Id_Var; i++) {
      type = Var_Set[i].type;
      // Now, we print all variables in an adequate format.
      if (type == REAL)
	fprintf(sum, "   %s\t%.15g\n", Var_Set[i].name, \
		*((real*)Var_Set[i].variable));
      if (type == INT)
	fprintf(sum, "   %s\t%d\n", Var_Set[i].name,	\
		*((int*) Var_Set[i].variable));
      if (type == BOOL)
	fprintf(sum, "   %s\t%d\n", Var_Set[i].name,	\
		*((boolean*) Var_Set[i].variable));
      if (type == STRING)
	fprintf(sum, "   %s\t%s\n", Var_Set[i].name,	\
		Var_Set[i].variable);
    }
    fprintf (sum, "*** Input file: %s\n#-----------\n%s\n#-----------\n",
	     ParameterFile, InputFile);
#ifdef LONGSUMMARY
    fprintf (sum, "\n%sBOUNDARY CONDITIONS SECTION:\n%s",sep,sep);
    fprintf (sum, "%s", BoundaryFile);
#endif
    if (ThereArePlanets) {
      fprintf (sum, "\n%sPLANETARY SYSTEM SECTION:\n%s",sep,sep);
      fprintf (sum, "#### (X,Y,Z,VX,VY,VZ,mass)\n");
      n = Sys->nb;
      for (i = 0; i < n; i++) {
	fprintf (sum, "#### Planet %d out of %d\n", i, n);
	fprintf (sum, "%.15g\t%.15g\t%.15g\t%.15g\t%.15g\t%.15g\t%.15g\n", \
		 Sys->x[i], Sys->y[i], Sys->z[i], Sys->vx[i], Sys->vy[i], \
		 Sys->vz[i], Sys->mass[i]);
      }
      fprintf (sum, "*** Planetary system config file: %s\n#-----------\n%s\n#-----------\n",
	       PLANETCONFIG, PlanetaryFile);
    }
    fclose (sum);
  }
}
