#include "fargo3d.h"

void RestartStretch (Field *field, int n) {
  int i,j,k;
  real *f;
  char *name;
  char filename[200];
  FILE *fi;
  int origin;
  size_t sz, exp_sz, size_ratio;
  int temp, Nxp;

  OUTPUT(field);

  f = field->field_cpu;
  name = field->name;

  sprintf(filename, "%s%s%d.dat", OUTPUTDIR, name, n);
  fi = fopen(filename, "r");
  if(fi == NULL) {
    masterprint("Error reading %s\n", filename);
    exit(1);
  }
  
  fseek (fi, 0L, SEEK_END);
  sz = ftell(fi);  // File size
  fseek (fi, 0L, SEEK_SET);
  exp_sz = NX*NY*NZ*sizeof(real); // Expected file size
  if (exp_sz % sz != 0) {
    fprintf (stderr,"Mesh size is not a multiple of size of mesh to strech. Aborting.\n");
    MPI_Finalize ();
    exit (EXIT_FAILURE);
  }
  size_ratio = exp_sz / sz;
  Nxp = Nx / size_ratio;

  masterprint("Expanding %s a factor %dx in X\n", filename, (int)size_ratio);


  origin = (z0cell)*Nxp*NY + (y0cell)*Nxp; //z0cell and y0cell are global variables.
  for (k = NGHZ; k < Nz+NGHZ; k++) {
    for (j = NGHY; j < Ny+NGHY; j++) {
      fseek(fi, (origin+(k-NGHZ)*Nxp*NY+(j-NGHY)*Nxp)*sizeof(real), SEEK_SET); 
      temp = fread(f+j*(Nx+2*NGHX)+k*Stride+NGHX, sizeof(real), Nxp, fi);
      for (i = NX+NGHX-1; i>=0; i--) //backward sweep as we overwrite data
	f[l] = f[(i-NGHX)/size_ratio + j*(Nx+2*NGHX) + k*Stride+NGHX];
    }
  }
  masterprint("%s OK\n", filename);
  fclose(fi);
}

void StretchOutput (int n) {
  RestartStretch(Density, n);
#ifdef X	
  RestartStretch(Vx, n);
#endif
#ifdef Y	
  RestartStretch(Vy, n);
#endif
#ifdef Z	
  RestartStretch(Vz, n);
#endif
  RestartStretch(Energy, n);
#ifdef MHD
  RestartStretch(Bx, n);
  RestartStretch(By, n);
  RestartStretch(Bz, n);
#endif
}
