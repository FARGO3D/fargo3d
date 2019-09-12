#include "fargo3d.h"

extern HashParam CommandLineParams[];
extern long nbparamCL;

Param Var_Set[MAXVARIABLES];
int Id_Var = 0;

void DumpToFargo3drc (int argc, char *argv[]) {
  char outputdir[MAXLINELENGTH];
  char *home;
  char rcfile[MAXLINELENGTH];
  FILE *rc;
  time_t tloc;
  int i;
  char *status;

  if (!CPU_Rank) {
    home = getenv ("HOME");
    strcpy(rcfile, home);
    strcat(rcfile, "/.fargo3drc");
    mkdir (rcfile, 0755);
    strcat (rcfile, "/lastout");
    rc = fopen (rcfile, "a");
    if (rc) { 			/* Silent if cannot create file */
      time (&tloc);
      fprintf (rc, "%s", ctime(&tloc));
      if (*OUTPUTDIR == '/') {
	fprintf (rc, "%s\n", OUTPUTDIR);
      } else {
	status = getcwd (outputdir, MAXLINELENGTH);
	strcat (outputdir, "/");
	strcat (outputdir, OUTPUTDIR);
	fprintf (rc, "%s\n", outputdir);
      }
      fclose (rc);
    }
    strcpy(rcfile, home);
    strcat(rcfile, "/.fargo3drc");
    mkdir (rcfile, 0755);
    strcat (rcfile, "/history");
    rc = fopen (rcfile, "a");
    time (&tloc);
    if (rc) fprintf (rc, "-----------\n%s", ctime(&tloc)); 
    for (i = 0; i < argc; i++) {
      if (rc) { 	 /* Silent if cannot create file */
	fprintf (rc, "%s ", argv[i]);
      }
    }
    if (rc) { 			/* Silent if cannot create file */
      fprintf (rc, "(%d process%s in total) on host%s:\n", CPU_Number,
	       (CPU_Number > 1 ? "es" : ""),
	       (CPU_Number > 1 ? "s" : ""));
      for (i = 0; i < CPU_Number; i++)
	fprintf (rc, "%s\n", HostsList+i*1024);
      if (*OUTPUTDIR == '/') {
	fprintf (rc, "%s\n", OUTPUTDIR);
      } else {
	status = getcwd (outputdir, MAXLINELENGTH);
	strcat (outputdir, "/");
	strcat (outputdir, OUTPUTDIR);
	fprintf (rc, "%s\n", outputdir);
      }
      fclose (rc);
    }
  }
}


void init_var(char *name, char *variable, int type, int need, char *value) {
  real real_value;
  int  int_value;
  boolean bool_value;
  real temp;

#ifdef FLOAT
  sscanf(value, "%f", &temp);  // Save default value of variable
#else
  sscanf(value, "%lf", &temp);  // Save default value of variable
#endif
  if (type == REAL) real_value = (real)temp;
  if (type == INT ) int_value  = (int) temp;
  if (type == BOOL) bool_value = (boolean) temp;
  strcpy(Var_Set[Id_Var].name, name); // Save name into a struct Var_Set
  Var_Set[Id_Var].variable = variable;  // Point to memory address of variable
  Var_Set[Id_Var].type = type;
  Var_Set[Id_Var].need = need;
  Var_Set[Id_Var].read = NO;
  // Set to default values. These come from var.c
  if (type == INT)  *((int*) variable) = int_value; 
  if (type == REAL) *((real*)variable) = real_value;
  if (type == BOOL) *((boolean*)variable) = bool_value;
  if (type == STRING) strcpy(variable, value);
  // Move to the next variable
  Id_Var++;
}

void ReadVarFile(char *filename) {
  
  FILE *input;
  
  real real_value;
  int  int_value, type;
  boolean bool_value = FALSE;
 
  real temp;

  char separator[20] = "\t :=>"; //Separators between a parameter name and its value
  char s[MAXLINELENGTH], name[MAXNAMELENGTH], strval[MAXNAMELENGTH];
  char testbool;
  char *s1;

  int  *int_var;
  real *real_var;
  boolean *bool_var;

  int found;

  int i, success;

  input = fopen(filename, "r");

  if (input == NULL) {
    mastererr("Unable to read '%s'. Program stopped.\n", filename);
    prs_exit(1);
  }
  ReadRedefined ();
  mastererr("Reading parameters file %s\n", filename);
  while (fgets(s, MAXLINELENGTH-1, input) != NULL ) {
    success = sscanf(s, "%s", name); // read finish in a white-space.
    if(name[0]!='#' && success == 1) {
      s1 = s + (int)strlen(name); // pointer shift (Here is the data)
      // cast to int because type of strlen is size_t.
#ifdef FLOAT
      sscanf(s1 + strspn(s1, separator),"%f", &temp); //single precision floating point value
#else
      sscanf(s1 + strspn(s1, separator),"%lf", &temp); //double precision floating point value
#endif
      sscanf(s1 + strspn(s1, separator),"%s", strval); //string value
      real_value = (real)temp;
      int_value  = (int)temp;
      testbool   = toupper(strval[0]); // Convert to upper case
      if (testbool == 'Y')  bool_value = TRUE;  //Yes, or
      else bool_value = FALSE;
      // name to upper case
      for (i = 0; i<strlen(name); i++) name[i] = (char)toupper(name[i]);
      if (strcmp(name, "INCLUDE") == 0) {
	ReadVarFile(strval); // Recursive call. You can include a parameter file within another one.
      }
      else {
	found = NO;
	for (i = 0; i<Id_Var; i++) {
	  if (strcmp(name, Var_Set[i].name) == 0) {
	    if (Var_Set[i].read == YES) {
	      mastererr("Warning : %s is defined more than once.\n", name);
	    }
	    if (Var_Set[i].read == REDEFINED) {
	      mastererr("Warning : %s is redefined on the command line.\n", name);
	    }
	    if (Var_Set[i].need == IRRELEVANT) {
	      mastererr("Warning : variable %s is irrelevant and nonetheless defined.\n", name);
	    }
	    found = YES;
	    if (Var_Set[i].read == NO) {
	      Var_Set[i].read = YES;
	      // Now we point to the correct variable
	      real_var = (real*)(Var_Set[i].variable);
	      int_var  = (int*) (Var_Set[i].variable);
	      bool_var = (boolean*)(Var_Set[i].variable);
	      // Now we reassign the value of the variable, if needed
	      if (Var_Set[i].type == REAL)   *real_var = real_value;
	      if (Var_Set[i].type == INT )   *int_var  = int_value;
	      if (Var_Set[i].type == BOOL)   *bool_var = bool_value;
	      if (Var_Set[i].type == STRING) strcpy(Var_Set[i].variable, strval);
	    }
	  }
	}
      }
      if ((found == NO) && strcmp(name, "INCLUDE")!=0 && strcmp(name, "END")!=0)
      mastererr("Warning: variable %s defined but does not exist in code.\n", name);
    }
  }
  found = NO;
  for (i = 0; i<Id_Var; i++) {
    if ((Var_Set[i].read == NO) && (Var_Set[i].need == YES)) {
      if (found == NO) {
	mastererr("Fatal error : undefined mandatory variable(s):\n");
	found = YES;
      }
      mastererr("%s\n", Var_Set[i].name);
    }
    if (found == YES)
      prs_exit(1);
  }
  found = NO;
  if(strcmp(name, "END")==0) {
    for (i=0; i<Id_Var; i++) {
      if (Var_Set[i].read == NO) {
	if (found == NO) {
	  mastererr("Secondary variables omitted:\n");
	  found = YES;
	}
	if ((type = Var_Set[i].type) == REAL)
	  mastererr("%s;\t Default Value : %.5g\n", \
		    Var_Set[i].name, *((real *) Var_Set[i].variable));
	if (type == INT)
	  mastererr("%s;\t Default Value : %d\n", \
		    Var_Set[i].name, *((int *) Var_Set[i].variable));
	if (type == STRING)
	  mastererr("%s;\t Default Value : %s\n", \
		    Var_Set[i].name, Var_Set[i].variable);
      }
    }
  }
  fclose(input);
  var_assign();
}

void var_assign(){
  char SetUpName[MAXLINELENGTH];
  sprintf (SetUpName, xstr(SETUPNAME));
  if ((strcmp(SetUpName, "SETUPNAME") != 0) && (strcmp(SETUP, "Undefined") !=0)) {
    if (strcmp(SetUpName,SETUP) != 0) {
      mastererr ("\n\n******\n\nERROR\n\n");
      mastererr ("The parameter file is meant to run exclusively with\n");
      mastererr ("the '%s' setup, but the code has been build with\n", SETUP);
      mastererr ("the '%s' setup. I must exit.\n", SetUpName);
      mastererr ("You can fix that either by choosing another parameter file,\n");
      mastererr ("or, if you wish to use the one you have specified,\n");
      mastererr ("you should rebuild the code by issuing the command\n");
      mastererr ("'make SETUP=%s'\n", SETUP);
      prs_exit(1);
    }
  }

  if ((*FRAME == 'C') || (*FRAME == 'c')) Corotating = YES;
  if ((*FRAME == 'G') || (*FRAME == 'g')) {
    Corotating = YES;
    GuidingCenter = YES;
  }

#if defined(CARTESIAN)
  sprintf(COORDINATES,"%s","cartesian");
#elif defined(CYLINDRICAL)
  sprintf(COORDINATES,"%s","cylindrical");
#elif defined(SPHERICAL)
  sprintf(COORDINATES,"%s","spherical");
#endif

  if ((THICKNESSSMOOTHING != 0.0) && (ROCHESMOOTHING != 0.0)) {
    mastererr ("You cannot use at the same time\n");
    mastererr ("`ThicknessSmoothing' and `RocheSmoothing'.\n");
    mastererr ("Edit the parameter file so as to remove\n");
    mastererr ("one of these variables and run again.\n");
    prs_exit (1);
  }

#ifndef VISCOSITY
  if (NU != 0.0) {
    mastererr ("ERROR - You have defined a non-vanishing value for\n");
    mastererr ("the kinematic viscosity NU, but the code is built\n");
    mastererr ("without the viscosity module. Edit your setup file\n");
    mastererr ("and add the line:\n");
    mastererr ("\nFARGO_OPT += -DVISCOSITY\n\n");
    mastererr ("and rebuild the code.\n");
    prs_exit (1);
  }
#endif
  
#ifndef ALPHAVISCOSITY
  if (ALPHA != 0.0) {
    mastererr ("ERROR - You have defined a non-vanishing value for\n");
    mastererr ("the disk's alpha viscosity, but the code is built\n");
    mastererr ("without the viscosity module. Edit your setup file\n");
    mastererr ("and add the line:\n");
    mastererr ("\nFARGO_OPT += -DALPHAVISCOSITY\n\n");
    mastererr ("and rebuild the code.\n");
    prs_exit (1);
  }
#endif

#if (defined(VISCOSITY) && defined(ALPHAVISCOSITY))
  mastererr ("ERROR - You cannot activate at the same time\n");
  mastererr ("VISCOSITY and ALPHAVISCOSITY. Fix, rebuild and rerun.\n");
  prs_exit (1);
#endif

#if (defined(COLLISIONPREDICTOR) && !defined(DRAGFORCE))
  mastererr ("ERROR - You cannot activate the COLLISIONPREDICTOR without the DRAGFORCE\n");
  prs_exit (1);
#endif


#if defined(DUSTDIFFUSION) && defined(ALPHAVISCOSITY) && !defined(Y)
  mastererr("ERROR - Direction Y (-DY in the .opt file) must be activated\n");
  mastererr("\tfor the dust diffusion module with Alpha Viscosity.\n");
  prs_exit (1);
#endif



  if (NGHX > NX) {
    mastererr ("\n\n\nERROR\n\nThe buffer zones in X are wider than the active mesh\n");
    mastererr ("This is not permitted.\n");
    mastererr ("Either increase NX (it should be at least %d)\n", NGHX);
    mastererr ("or rebuild the code with the option GHOSTSX=0 (noghostsx)\n");
    prs_exit (1);
  }

  /* Add a trailing slash to OUTPUTDIR if needed */
  if (*(OUTPUTDIR+strlen(OUTPUTDIR)-1) != '/')
    strcat (OUTPUTDIR, "/");

#ifdef FLOAT
  sprintf(REALTYPE,"%s","float32");
#else
  sprintf(REALTYPE,"%s","float64");
#endif


#ifdef RESCALE
  YMAX *= R0;
  YMIN *= R0;
#if defined(CYLINDRICAL) || defined(CARTESIAN)
  ZMAX *= R0;
  ZMIN *= R0;
#endif
#ifdef CARTESIAN
  XMIN *= R0;
  XMAX *= R0;
#endif
  rescale();
#endif
}

void ListVariables (char *filename) {
  int i, type;
  FILE *stream;
  char fullname[MAXLINELENGTH];
  sprintf (fullname, "%s/%s", OUTPUTDIR, filename);
  if (CPU_Rank == 0) {
    stream = fopen_prs (fullname,"w");
    for (i = 0; i < Id_Var; i++) {
      type = Var_Set[i].type;
      // Now, we print all variables in an adequate format.
      if (type == REAL)
	fprintf(stream, "%s\t%.15g\n", Var_Set[i].name, \
		*((real*)Var_Set[i].variable));
      if (type == INT)
      fprintf(stream, "%s\t%d\n", Var_Set[i].name, \
	      *((int*) Var_Set[i].variable));
      if (type == BOOL)
	fprintf(stream, "%s\t%d\n", Var_Set[i].name,	\
		*((boolean*) Var_Set[i].variable));
      if (type == STRING)
	fprintf(stream, "%s\t%s\n", Var_Set[i].name,	\
		Var_Set[i].variable);
    }
    fclose(stream);
  }
}

void ListVariablesIDL (char *filename)
{
  int i, type;
  FILE *stream;
  char fullname[MAXLINELENGTH];
  sprintf (fullname, "%s/%s", OUTPUTDIR, filename);
  if (CPU_Rank == 0) {
    stream = fopen_prs (fullname,"w");
    fprintf(stream, "input_par = { $\n");
    for (i = 0; i < Id_Var; i++) {
      type = Var_Set[i].type;
      if (type == REAL)
	fprintf(stream, "%s:%.15g", Var_Set[i].name, *((real *) Var_Set[i].variable));
      if (type == INT)
	fprintf(stream, "%s:%d", Var_Set[i].name, *((int *) Var_Set[i].variable));
      if (type == BOOL)
	fprintf(stream, "%s:%d", Var_Set[i].name, *((boolean *) Var_Set[i].variable));
      if (type == STRING)
	fprintf(stream, "%s:'%s'", Var_Set[i].name, Var_Set[i].variable);
      if (i != Id_Var-1) fprintf(stream, ",$\n");
    }
    fprintf(stream, "}\n");
    fclose (stream);
  }
}

void ReadRedefined () {
  int i, j, found = NO;
  int            *ptri;
  real           *ptrr;
  boolean        *ptrb;
  for (j = 0; j < nbparamCL; j++) {
    for (i = 0; i < Id_Var; i++) {
      if (strcmp (Var_Set[i].name, CommandLineParams[j].name) == 0) {
	found = YES;
	if (Var_Set[i].read == REDEFINED) {
	  mastererr ("Parameter %s on command line is specified twice\n", \
		    CommandLineParams[j].name);
	  exit (EXIT_FAILURE);
	}
	Var_Set[i].read = REDEFINED;
	ptri = (int *) (Var_Set[i].variable);
	ptrr = (real *) (Var_Set[i].variable);
	ptrb = (boolean *) (Var_Set[i].variable);
	if (Var_Set[i].type == INT) {
	  *ptri = CommandLineParams[j].intvalue;
	} else if (Var_Set[i].type == REAL) {
	  *ptrr = CommandLineParams[j].floatvalue;
	    } else if (Var_Set[i].type == BOOL) {
	  *ptrb = CommandLineParams[j].boolvalue;
	} else if (Var_Set[i].type == STRING) {
	      strcpy (Var_Set[i].variable, CommandLineParams[j].stringvalue);
	}
      }
    }
    if (found == NO) {
      mastererr ("Parameter %s on command line is unknown\n", CommandLineParams[j].name);
      exit (EXIT_FAILURE);
    }
  }
}
