#include <fargo3d.h>

void WriteVTKMerging(Field *f, int n) {
  int i,j,k,m,p;
  char outname[MAXLINELENGTH];
  real temp;
  FILE *ofile;
  static boolean init = YES;

  INPUT (f);
  sprintf(outname, "%s%s%d.vtk", OUTPUTDIR, f->name, n);

  if (CPU_Master) {
    ofile = fopen(outname, "w");
    fclose(ofile);
    ofile = fopen(outname, "a+");
  }
  else {
    ofile = fopen(outname,"a");
  }

  if (CPU_Master) {
    fprintf(ofile, "# vtk DataFile Version 2.0\n");
#ifdef FLOAT
    fprintf(ofile, "Output %d - Field: %s - Physical time %f\n",
	    n, f->name, PhysicalTime);  
#else
    fprintf(ofile, "Output %d - Field: %s - Physical time %lf\n",
	    n, f->name, PhysicalTime);
#endif
    fprintf(ofile, "BINARY\n");
    fprintf(ofile, "DATASET RECTILINEAR_GRID\n");
#ifndef SPHERICAL
    fprintf(ofile, "DIMENSIONS %d %d %d\n", NY, NX, NZ);
#else
    fprintf(ofile, "DIMENSIONS %d %d %d\n", NY, NZ, NX);
#endif
#ifdef FLOAT
    fprintf(ofile, "X_COORDINATES %d FLOAT\n", NY);
#else
    fprintf(ofile, "X_COORDINATES %d DOUBLE\n", NY);
#endif
  }

  for (i=0;i<NY;i++) {
    for (j = 0; j<Ncpu_x; j++) {
      if ((J==j) && (K==0) && (i>=Y0) && (i<(Y0+Ny))) {
	temp = Swap(f->y[i+NGHY-Y0]);
	fwrite(&temp, sizeof(real), 1, ofile);
      }
    }
    fflush(ofile);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  if (CPU_Master)
    fprintf(ofile, "\n");
  
#ifndef SPHERICAL
  if (CPU_Master) {
#ifdef FLOAT
    fprintf(ofile, "Y_COORDINATES %d FLOAT\n", NX);
#else
    fprintf(ofile, "Y_COORDINATES %d DOUBLE\n", NX);
#endif  

    for (i=0;i<NX;i++) {
      temp = Swap(f->x[i]);
      fwrite(&temp, sizeof(real), 1, ofile);
    }
    fprintf(ofile, "\n");
  }
  fflush(ofile);
  MPI_Barrier(MPI_COMM_WORLD);
  
  if (CPU_Master)
#ifdef FLOAT
    fprintf(ofile, "Z_COORDINATES %d FLOAT\n", NZ);
#else
    fprintf(ofile, "Z_COORDINATES %d DOUBLE\n", NZ);
#endif  

  MPI_Barrier(MPI_COMM_WORLD);
  
  for (i=0;i<NZ;i++) {
    for (j = 0; j<Ncpu_y; j++) {
      if ((K==j) && (J==0) && (i>=Z0) && (i<(Z0+Nz))) {
	temp = Swap(f->z[i+NGHZ-Z0]);
	fwrite(&temp, sizeof(real), 1, ofile);
      }
    }
    fflush(ofile);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  if (CPU_Master)
    fprintf(ofile, "\n");

#else
  
  if (CPU_Master) {
#ifdef FLOAT
    fprintf(ofile, "Y_COORDINATES %d FLOAT\n", NZ);
#else
    fprintf(ofile, "Y_COORDINATES %d DOUBLE\n", NZ);
#endif
  }

  MPI_Barrier(MPI_COMM_WORLD);
  
  for (i=0;i<NZ;i++) {
    for (j = 0; j<Ncpu_y; j++) {
      if ((K==j) && (J==0) && (i>=Z0) && (i<(Z0+Nz))) {
	temp = Swap(f->z[i+NGHZ-Z0]);
	fwrite(&temp, sizeof(real), 1, ofile);
      }
    }
    fflush(ofile);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  if (CPU_Master)
    fprintf(ofile, "\n");
  
  
  if (CPU_Master) {
#ifdef FLOAT
    fprintf(ofile, "Z_COORDINATES %d FLOAT\n", NX);
#else
    fprintf(ofile, "Z_COORDINATES %d DOUBLE\n", NX);
#endif

    for (i=0;i<NX;i++) {
      temp = Swap(f->x[i]);
      fwrite(&temp, sizeof(real), 1, ofile);
    }
    fprintf(ofile, "\n");
  }
  fflush(ofile);
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  
  fflush(stdout);

  if (CPU_Master) {
    fprintf(ofile, "POINT_DATA %d\n", NX*NY*NZ);
    
#ifdef FLOAT
    fprintf(ofile, "SCALARS %s FLOAT\n", f->name);
#else
    fprintf(ofile, "SCALARS %s DOUBLE\n", f->name);
#endif
    fprintf(ofile, "LOOKUP_TABLE default\n");
    if (strcmp(f->name,FIELD) == 0)
      VtkPosition = ftell(ofile);
  }

  fflush(ofile);
  MPI_Barrier(MPI_COMM_WORLD);

  i = j = k = 0;
  
#ifndef SPHERICAL
  for (k=0; k<NZ; k++) {
    for (i=0; i<NX; i++) {
      for (j = 0; j<Ncpu_x; j++) {
	if ((J==j) && (k>=Z0) && (k<(Z0+Nz))) {
	  for (m=NGHY;m<Ny+NGHY;m++) {
	    temp = Swap(f->field_cpu[i+m*Nx+(k-Z0)*Stride+NGHZ*Stride]);
	    fwrite(&temp, sizeof(real), 1, ofile);
	  }
	}
	fflush(ofile);
	MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
#else
  for (i=0; i<NX; i++) {
    for (k=0; k<NZ; k++) {
      for (j = 0; j<Ncpu_x; j++) {
	if ((J==j) && (k>=Z0) && (k<(Z0+Nz))) {
	  for (m=NGHY;m<Ny+NGHY;m++) {
	    temp = Swap(f->field_cpu[m*Nx+(k-Z0)*Stride+NGHZ*Stride]);
	    fwrite(&temp, sizeof(real), 1, ofile);
	  }
	}
	fflush(ofile);
	MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
#endif
  MPI_Barrier(MPI_COMM_WORLD);
  fclose(ofile);
}
