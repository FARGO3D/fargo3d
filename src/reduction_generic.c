// This is the CPU version of the reduction of an n-dim array into an (n-1)-dim array
// The final reduction of this intermediate array is always performed on the CPU
// by the function in  "reduction_full_generic.c"

void name_reduction_cpu(macro) (Field *F, int ymin, int ymax, int zmin, int zmax) {
  int i,j,k;
  real *reduc2d;
  real *f;
  reduc2d = Reduction2D->field_cpu;
  f = F->field_cpu;

  INPUT (F);
  OUTPUT2D (Reduction2D);

  for (k = zmin; k < zmax; k++) {
    for (j = ymin; j < ymax; j++) {
      reduc2d[l2D] = INIT_REDUCTION (macro);
      for (i = NGHX; i < (Nx+NGHX); i++) {
	reduc2d[l2D] = macro(reduc2d[l2D], f[l]);
      }
    }
  }
}
