#include "fargo3d.h"

int CheckNansField (Field *f) {
  int i,j,k;
  for(k=0; k<Nz+2*NGHZ; k++) {
    for(j=0; j<Ny+2*NGHY; j++) {
      for(i=0; i<Nx+2*NGHX; i++) {
	if (isnan(f->field_cpu[l])) {
	  return l;
	}
      }
    }
  }
  return -1;
}

int CheckAxiSym (Field *f) {
  int i=NGHX-1,j,k;
  for(k=NGHZ; k<Nz+NGHZ; k++) {
    for(j=NGHY; j<Ny+NGHY; j++) {
      if (fabs(f->field_cpu[l] - f->field_cpu[l+1]) > 1e-12)
	return l;
    }
  }
  return -1;
}

void AxiSym (Field *f) {
  int i=NGHX-1,j,k;
  for(k=NGHZ; k<Nz+NGHZ; k++) {
    for(j=NGHY; j<Ny+NGHY; j++) {
      f->field_cpu[l] = f->field_cpu[l+1];
    }
  }
}

void CheckNans (char *string){
  static int count=0;
  int i;
  Field *g;
  g = ListOfGrids;
  while (g != NULL) {
    if ((i = CheckAxiSym (g)) >= 0) {
      if (g == DensStar) {
	printf ("DensStar being examined after call to %s\n", string);
      }
      printf ("Found non axisym in grid %s at location (i=%d, j=%d, k=%d)\n",	\
	      g->name, i%(Nx+2*NGHX), (i/(Nx+2*NGHX))%(Ny+2*NGHY), i/((Nx+2*NGHX)*(Ny+2*NGHY)));
      printf ("position: after call to %s\n", string);
      exit(1);
      if (g == DensStar) {
	printf ("DensStar being examined after call to %s\n", string);
      }
    }
    g = g->next;
  }
}
