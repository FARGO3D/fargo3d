#include "fargo3d.h"

void Write2D (Field2D *f, char *filename, char *dir, int kind) {
  int jmin, jmax, kmin, kmax, j, k, jg, kg;
  real profile[MAX1D], gprofile[MAX1D], value;
  char name[MAXLINELENGTH];
  FILE *out;

  INPUT2D (f);

  sprintf (name, "%s/%s", dir, filename);
  out = fopen_prs (name, "w");
  jmin = (kind == GHOSTINC ? 0 : NGHY);
  jmax = (kind == GHOSTINC ? NY+2*NGHY : NY+NGHY);
  kmin = (kind == GHOSTINC ? 0 : NGHZ);
  kmax = (kind == GHOSTINC ? NZ+2*NGHZ : NZ+NGHZ);
  for (kg = kmin; kg < kmax; kg++) {
    for (jg = 0; jg < NY+2*NGHY; jg++)
      profile[jg] = gprofile[jg] = 0.0;
    for (jg = jmin; jg < jmax; jg++) {
      j = jg-y0cell;
      k = kg-z0cell;
      if ((j >= 0) && (j < Ny+2*NGHY) && (k >= 0) && (k < Nz+2*NGHZ))
	value = f->field_cpu[l2D];
      if ((j >= NGHY) && (j < NGHY+Ny) && (k >= NGHZ) && (k < NGHZ+Nz))
	profile[jg] = value;
      // In the tests below we deal with the different kinds of ghosts if we have to dump them
      if (kind == GHOSTINC) {
	if ((j < NGHY) && (Gridd.J == 0) && (k >= NGHZ) && (k < NGHZ+Nz)) //Left edge
	  profile[jg] = value;
	if ((j >= NGHY+Ny) && (Gridd.J == Gridd.NJ-1) && (k >= NGHZ) && (k < NGHZ+Nz)) //Right edge
	  profile[jg] = value;
	if ((k < NGHZ) && (Gridd.K == 0) && (j >= NGHY) && (j < NGHY+Ny)) //Bottom edge
	  profile[jg] = value;
	if ((k >= NGHZ+Nz) && (Gridd.K == Gridd.NK-1) && (j >= NGHY) && (j < NGHY+Ny)) //Top edge
	  profile[jg] = value;
	if ((k < NGHZ) && (Gridd.K == 0) && (j < NGHY) && (Gridd.J == 0)) //Bottom-left corner
	  profile[jg] = value;
	if ((k >= NGHZ+Nz) && (Gridd.K == Gridd.NK-1) && (j < NGHY) && (Gridd.J == 0)) //Top-left corner
	  profile[jg] = value;
	if ((k < NGHZ) && (Gridd.K == 0) && (j >= Ny+NGHY) && (Gridd.J == Gridd.NJ-1)) //Bottom-right corner
	  profile[jg] = value;
	if ((k >= NGHZ+Nz) && (Gridd.K == Gridd.NK-1) && (j >= Ny+NGHY) && (Gridd.J == Gridd.NJ-1)) //Top-right corner
	  profile[jg] = value;
      }
    }
#ifndef FLOAT
    MPI_Reduce (profile, gprofile, NY+2*NGHY, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
    MPI_Reduce (profile, gprofile, NY+2*NGHY, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
    if (CPU_Rank == 0) {
      if (kind == NOGHOSTINC)
	fwrite (gprofile+NGHY, sizeof(real), NY, out);
      if (kind == GHOSTINC)
	fwrite (gprofile, sizeof(real), NY+2*NGHY, out);
    }
  }
  fclose (out);
}

boolean Read2D (Field2D *f, char *filename, char *dir, int kind) {
  int jmin, jmax, kmin, kmax, j, k, jg, kg;
  real profile[MAX1D], gprofile[MAX1D], value;
  char name[MAXLINELENGTH];
  FILE *in;
  boolean error_occured=FALSE;
  int relay, filesize, sz;

  OUTPUT2D (f);

  filesize = (2*NGHZ+NZ)*(2*NGHY+NY)*sizeof(real);
  if (kind == NOGHOSTINC)
    filesize = NZ*NY*sizeof(real);
  sprintf (name, "%s/%s", dir, filename);
  if (CPU_Rank > 0) { // Force sequential read
    MPI_Recv (&relay, 1, MPI_INT, CPU_Rank-1, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  in = fopen (name, "r");
  if (in == NULL) {
    error_occured = TRUE;
  } else {
    fseek (in, 0, SEEK_END);
    sz = ftell(in);
    fseek (in, 0, SEEK_SET);
    // We check that the file has the right size in order to avoid reading an older file
    if (sz != filesize) error_occured = TRUE; 
    jmin = (kind == GHOSTINC ? 0 : NGHY);
    jmax = (kind == GHOSTINC ? NY+2*NGHY : NY+NGHY);
    kmin = (kind == GHOSTINC ? 0 : NGHZ);
    kmax = (kind == GHOSTINC ? NZ+2*NGHZ : NZ+NGHZ);
    for (kg = kmin; kg < kmax; kg++) {
      if (kind == NOGHOSTINC)
	if (fread (gprofile+NGHY, sizeof(real), NY, in) == 0) error_occured = TRUE;
      if (kind == GHOSTINC)
	if (fread (gprofile, sizeof(real), NY+2*NGHY, in) == 0) error_occured = TRUE;
      for (jg = jmin; jg < jmax; jg++) {
	j = jg-y0cell;
	k = kg-z0cell;
	if ((j >= 0) && (j < Ny+2*NGHY) && (k >= 0) && (k < Nz+2*NGHZ) && (kind == GHOSTINC))
	  f->field_cpu[l2D] = gprofile[jg];
	if ((j >= NGHY) && (j < NGHY+Ny) && (k >= NGHZ) && (k < NGHZ+Nz) && (kind == NOGHOSTINC))
	  f->field_cpu[l2D] = gprofile[jg];
      }
    }
    fclose (in);
  }
  if (CPU_Rank < CPU_Number-1) {  // Force sequential read
    MPI_Send (&relay, 1, MPI_INT, CPU_Rank+1, 42, MPI_COMM_WORLD);
  }

  if (error_occured == FALSE) {
    masterprint ("File %s was read successfully\n", filename);
  }
  if ((error_occured == TRUE) && ((Restart == TRUE) || (Restart_Full == TRUE))) {
    printf ("Problem reading %s, CPU %d\n", filename, CPU_Rank);
  }
  /* The MPI_Barrier below MUST NOT be commented. Several files are
     read upon restart (density, energy, vx, etc.), and the relay sent
     by CPU 0 in order to read one of them could be caught by CPU 1 when
     trying to read another one, which may result in race
     condition. Another solution could be to have an MPI tag that is a
     hash of the filename, but using the barrier is much simpler. */
  MPI_Barrier (MPI_COMM_WORLD);
  return error_occured;
}
