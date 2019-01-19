real name_full_reduction (macro) (Field *F, int ymin, int ymax, int zmin, int zmax) {
  int j,k;
  real *reduc2d;
  real result;
  reduc2d = Reduction2D->field_cpu;

  name_reduction(macro) (F, ymin, ymax, zmin, zmax);

  INPUT2D (Reduction2D);
  
  result = INIT_REDUCTION(macro);
  for (k = zmin; k < zmax; k++) {
    for (j = ymin; j < ymax; j++) {
      result = macro(reduc2d[l2D], result);
    }
  }
  return result;
}
