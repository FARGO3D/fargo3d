#include "fargo3d.h"

void write_vtk_header(FILE *ofile, Field *field, int n) {
  fprintf(ofile, "# vtk DataFile Version 2.0\n");
#if FLOAT
  fprintf(ofile, "Output %d - Field: %s - Physical time %f\n",
	  n, field->name, PhysicalTime);
#else
  fprintf(ofile, "Output %d - Field: %s - Physical time %lf\n",
	  n, field->name, PhysicalTime);
#endif
  fprintf(ofile, "BINARY\n");
  fprintf(ofile, "DATASET RECTILINEAR_GRID\n");
#if (!SPHERICAL)
  fprintf(ofile, "DIMENSIONS %d %d %d\n", Ny, Nx, Nz);
#else
  fprintf(ofile, "DIMENSIONS %d %d %d\n", Ny, Nz, Nx);
#endif
}

void write_vtk_coordinates(FILE *ofile, Field *field) {
  /*There is a difference between the coordinates defined
    in FARGO3D and the coordinates used by Visit in Cylindrical &
    Spherical case. Be careful with the meaning of X,Y,Z in the
    VTK File.*/
  int i;
  real temp;

#if FLOAT
  fprintf(ofile, "X_COORDINATES %d FLOAT\n", Ny);
#else
  fprintf(ofile, "X_COORDINATES %d DOUBLE\n", Ny);
#endif

  for (i=NGHY;i<Ny+NGHY;i++) {
    temp = Swap(field->y[i]);
    fwrite(&temp, sizeof(real), 1, ofile);
  }
  fprintf(ofile, "\n");

#if (!SPHERICAL)
#if FLOAT
  fprintf(ofile, "Y_COORDINATES %d FLOAT\n", Nx);
#else
  fprintf(ofile, "Y_COORDINATES %d DOUBLE\n", Nx);
#endif
  for (i=0;i<Nx;i++) {
    temp = Swap(field->x[i]);
    fwrite(&temp, sizeof(real), 1, ofile);
  }
  fprintf(ofile, "\n");

#if FLOAT
  fprintf(ofile, "Z_COORDINATES %d FLOAT\n", Nz);
#else
  fprintf(ofile, "Z_COORDINATES %d DOUBLE\n", Nz);
#endif
  for (i=NGHZ;i<Nz+NGHZ;i++) {
    temp = Swap(field->z[i]);
    fwrite(&temp, sizeof(real), 1, ofile);
  }
  fprintf(ofile, "\n");
#else
#if FLOAT
  fprintf(ofile, "Y_COORDINATES %d FLOAT\n", Nz);
#else
  fprintf(ofile, "Y_COORDINATES %d DOUBLE\n", Nz);
#endif
  for (i=NGHZ;i<Nz+NGHZ;i++) {
    temp = Swap(field->z[i]);
    fwrite(&temp, sizeof(real), 1, ofile);
  }
  fprintf(ofile, "\n");

#if FLOAT
  fprintf(ofile, "Z_COORDINATES %d FLOAT\n", Nx);
#else
  fprintf(ofile, "Z_COORDINATES %d DOUBLE\n", Nx);
#endif
  for (i=0;i<Nx;i++) {
    temp = Swap(field->x[i]);
    fwrite(&temp, sizeof(real), 1, ofile);
  }
  fprintf(ofile, "\n");
#endif
  fprintf(ofile, "POINT_DATA %d\n", Nx*Ny*Nz);
}

void write_vtk_scalar(FILE *ofile, Field *f) {
  int i, j, k;
  real temp;
#if FLOAT
  fprintf(ofile, "SCALARS %s FLOAT\n", f->name);
#else
  fprintf(ofile, "SCALARS %s DOUBLE\n", f->name);
#endif
  fprintf(ofile, "LOOKUP_TABLE default\n");
  if (strcmp(f->name,FIELD) == 0)
    VtkPosition = ftell(ofile);

  i = j = k = 0;

#if (!SPHERICAL)
#if ZDIM
  for (k=NGHZ; k<Nz+NGHZ; k++) {
#endif
#if XDIM
    for (i=0; i<Nx; i++) {
#endif
#if YDIM
      for (j=NGHY; j<Ny+NGHY; j++) {
#endif
#else
#if XDIM
  for (i=0; i<Nx; i++) {
#endif
#if ZDIM
    for (k=NGHZ; k<Nz+NGHZ; k++) {
#endif
#if YDIM
      for (j=NGHY; j<Ny+NGHY; j++) {
#endif
#endif
	temp = Swap(f->field_cpu[l]);
	fwrite(&temp, sizeof(real), 1, ofile);
#if YDIM
      }
#endif
#if XDIM
    }
#endif
#if ZDIM
  }
#endif
}

void WriteVTK(Field *f, int n) {
  int i,j,k;
  char filename[MAXLINELENGTH];
  FILE *ofile;
  INPUT (f);
  sprintf(filename, "%s%s%d_%d.vtk", OUTPUTDIR, f->name, n, CPU_Rank);
  ofile = fopen(filename,"w");

  write_vtk_header(ofile, f, n);
  write_vtk_coordinates(ofile, f);
  write_vtk_scalar(ofile, f);

  fclose(ofile);
}
