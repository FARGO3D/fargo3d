#include "fargo3d.h"

boolean CompareField (Field *f) { // Compare a field to its secondary backup
  int size, i;
  int diff = 0;
  real *f1, *f2;
  //  if(f->name[0]!='S') //Problem with Slope field when reduction is analized
  if (*(f->owner) == NULL) {
    if (CPU_Rank == 0) {
      fprintf (stderr, "Skipping comparison of field %s used as a temporary work array\n",f->name);
      fprintf (stderr, "in file %s (as declared at line %d)\n",f->file_origin,f->line_origin);
    }
    return FALSE;
  }
  if (*(f->owner) != f) return FALSE;
  INPUT (f);
  f1 = f->field_cpu;
  f2 = f->secondary_backup;
  size = (Nx+2*NGHX)*(Ny+2*NGHY)*(Nz+2*NGHZ);
  for (i=0; i < size; i++) {
    if (f1[i] != f2[i]) {
      diff++;
      //      printf ("%d\tf1=%g\tf2=%g\n",i,f1[i],f2[i]);
    }
  }
  if (diff > 0) return TRUE;
  return FALSE;
}

void CompareAllFields () {
  Field *current;
  int size;
  current = ListOfGrids;
  size = (Nx+2*NGHX)*(Ny+2*NGHY)*(Nz+2*NGHZ);
  if (!CPU_Rank) printf ("List of fields that differ:\n");
  while (current != NULL) {
    if (CompareField (current)) {
      if (!CPU_Rank) printf ("Fields %s differ:\n", current->name);
      GiveStats (current->name, current->field_cpu, current->secondary_backup, size);
    }
    current = current->next;
  }
}

void GiveStats (char *name, real *f1, real *f2, int size) {
  int i;
  real min1=1e40, min2=1e40, max1=-1e40, max2=-1e4;
  real mindiff=1e40, maxdiff=-1e40, minratio=1e40, maxratio=-1e40;
  real maxabs;
  for (i = 0; i < size; i++) {
    if (f1[i] < min1) min1 = f1[i];
    if (f1[i] > max1) max1 = f1[i];

    if (f2[i] < min2) min2 = f2[i];
    if (f2[i] > max2) max2 = f2[i];

    if (f2[i]-f1[i] < mindiff) mindiff = f2[i]-f1[i];
    if (f2[i]-f1[i] > maxdiff) maxdiff = f2[i]-f1[i];
    
    if (f2[i]/f1[i] < minratio) minratio = f2[i]/f1[i];    
    if (f2[i]/f1[i] > maxratio) maxratio = f2[i]/f1[i];
    
  }
  maxabs = max1;
  if (fabs(min1) > max1) maxabs = fabs(min1);
  printf ("Minimum of %s on GPU: %.17g\n", name, min1);
  printf ("Minimum of %s on CPU: %.17g\n", name, min2);
  printf ("Maximum of %s on GPU: %.17g\n", name, max1);
  printf ("Maximum of %s on CPU: %.17g\n", name, max2);
  printf ("Minimum for %s of GPU/CPU-1: %g\n", name, minratio-1.);
  printf ("Maximum for %s of GPU/CPU-1: %g\n", name, maxratio-1.);
  printf ("Minimum for %s of GPU-CPU: %g\n", name, mindiff);
  printf ("Maximum for %s of GPU-CPU: %g\n", name, maxdiff);
  printf ("(Minimum for %s of GPU-CPU)/max(abs(CPU)): %g\n", name, mindiff/maxabs);
  printf ("(Maximum for %s of GPU-CPU)/max(abs(CPU)): %g\n", name, maxdiff/maxabs);
  printf ("**********\n\n");
}
