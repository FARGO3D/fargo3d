#include "fargo3d.h"

Field *CreateFieldAlias(char *name, Field *clone, int type) {
  Field *field;
  //real *array;
  char *string;
  int i,j,k;

  field = (Field *) malloc(sizeof(Field));
  if (field == NULL)
    prs_error("Insufficient memory for Field cloning");
  string = (char *) malloc(sizeof(char) * 80);
  if (string == NULL)
    prs_error("Insufficient memory for Field creation-step3");
  sprintf(string, "%s", name);
  field->field_cpu = clone->field_cpu; //Cloning fields
  field->backup = NULL;
  field->secondary_backup = NULL;
  field->owner = (Field **)(field->field_cpu+(Ny+2*NGHY)*(Nx+2*NGHX)*(Nz+2*NGHZ));
  field->name = string;
  field->next = ListOfGrids;     //Linkedlist
  ListOfGrids = field;
#if GPU
  field->field_gpu = clone->field_gpu;
  field->gpu_pp = clone->gpu_pp;
  field->cpu_pp = clone->cpu_pp;
#endif

  field->fresh_cpu     =  YES;
  for (i = 0; i < 4; i++) {
    field->fresh_inside_contour_cpu[i] = YES;
    field->fresh_outside_contour_cpu[i] = YES;
  }
  field->fresh_gpu     =  NO;
  for (i = 0; i < 4; i++) {
    field->fresh_inside_contour_gpu[i] = NO;
    field->fresh_outside_contour_gpu[i] = NO;
  }

  field->type = type;
  masterprint("Grids %s and %s share their storage\n", clone->name, name);
  return field;
}

Fluid *CreateFluid(char *name, int fluidtype) {
  Fluid *f;
  char fieldname[80];

  char *fluidname = (char*) malloc(sizeof(char)*MAXNAMELENGTH);
  fluidname = (char *) malloc(sizeof(char) * 80);
  sprintf(fluidname, "%s", name);

  f = (Fluid *) malloc(sizeof(Fluid));
  f->name = fluidname;
  f->Fluidtype = fluidtype;

  sprintf(fieldname,"%s%s",name,"dens");
  f->Density = CreateField(fieldname, DENS, 0,0,0);

  sprintf(fieldname,"%s%s",name,"energy");
  f->Energy  = CreateField(fieldname, ENERGY, 0,0,0);
  f->VxMed   = CreateField2D ("VxMed", YZ);

#if XDIM
  sprintf(fieldname,"%s%s",name,"vx");
  f->Vx      = CreateField(fieldname, VX, 1,0,0);
  f->Vx_temp = CreateField("Vx_temp", VXTEMP, 1,0,0);
#if COLLISIONPREDICTOR
  f->Vx_half = CreateField("Vx_half", VXTEMP, 1,0,0);
#endif
#endif
#if YDIM
  sprintf(fieldname,"%s%s",name,"vy");
  f->Vy      = CreateField(fieldname, VY, 0,1,0);
  f->Vy_temp = CreateField("Vy_temp", VYTEMP,0,1,0);
#if COLLISIONPREDICTOR
  f->Vy_half = CreateField("Vy_half", VYTEMP, 1,0,0);
#endif
#endif
#if ZDIM
  sprintf(fieldname,"%s%s",name,"vz");
  f->Vz      = CreateField(fieldname, VZ, 0,0,1);
  f->Vz_temp = CreateField("Vz_temp", VZTEMP,0,0,1);
#if COLLISIONPREDICTOR
  f->Vz_half = CreateField("Vz_half", VZTEMP, 1,0,0);
#endif
#endif

#if STOCKHOLM
  f->Density0 = CreateField2D ("rho0", YZ);
  f->Energy0   = CreateField2D ("e0", YZ);
  f->Vx0  = CreateField2D ("vx0", YZ);
  f->Vy0  = CreateField2D ("vy0", YZ);
  f->Vz0  = CreateField2D ("vz0", YZ);
#endif

  return f;
}

Field *CreateField(char *name, int type, boolean sx, boolean sy, boolean sz) {
  /*sx = YES ==> Field is staggered in X. Useful for determining the
    domain of each field.*/

  Field *field;
  real *array;
  void *arr_gpu;
  char *string;
  int i,j,k;
  size_t pitch;

  field = (Field *) malloc(sizeof(Field));
  if (field == NULL)
    prs_error("Insufficient memory for Field creation-step1.");

#if (!GPU)
  array = (real *) malloc(sizeof(real)*(Ny+2*NGHY)*(Nx+2*NGHX)*(Nz+2*NGHZ)+sizeof(Field*));
#else
#ifndef PINNED
  array = (real *) malloc(sizeof(real)*(Ny+2*NGHY)*(Nx+2*NGHX)*(Nz+2*NGHZ)+sizeof(Field*));
#else
  cudaMallocHost((void**)&array,sizeof(real)*(Ny+2*NGHY)*(Nx+2*NGHX)*(Nz+2*NGHZ)+sizeof(Field*));
#endif
#endif

  if (array == NULL)
    prs_error("Insufficient memory for Field creation-step2.");
  string = (char *) malloc(sizeof(char) * 80);
  if (string == NULL)
    prs_error("Insufficient memory for Field creation-step3.");
  sprintf(string, "%s", name);
  field->field_cpu = array;
  field->backup = NULL;
  field->secondary_backup = NULL;
  field->name = string;
  field->owner = (Field **)(array+(Ny+2*NGHY)*(Nx+2*NGHX)*(Nz+2*NGHZ));
  *(field->owner) = field;
  field->line_origin = __LINE__;
  strncpy (field->file_origin, __FILE__, MAXLINELENGTH-1);

  field->next = ListOfGrids;     //Linkedlist
  ListOfGrids = field;

  i = j = k = 0;

#if ZDIM
  for (k = 0; k<Nz+2*NGHZ; k++) {
#endif
#if YDIM
    for (j = 0; j<Ny+2*NGHY; j++) {
#endif
#if XDIM
      for (i = 0; i<Nx+2*NGHX; i++) {
#endif
	array[l] = 0.0;
#if XDIM
      }
#endif
#if YDIM
    }
#endif
#if ZDIM
  }
#endif
  masterprint("Field %s has been created\n", name);
  //Now on the GPU
#if GPU
  if (Nx+2*NGHX > 1 ) {
    #ifndef NOPITCH
      cudaMallocPitch (&arr_gpu, &pitch, (Nx+2*NGHX)*sizeof(real), (Ny+2*NGHY)*(Nz+2*NGHZ));
    #else
      cudaMalloc (&arr_gpu, (Nx+2*NGHX)*(Ny+2*NGHY)*(Nz+2*NGHZ)*sizeof(real));
      pitch = (Nx+2*NGHX)*sizeof(real);
    #endif
    field->gpu_pp = make_cudaPitchedPtr (arr_gpu, pitch, (Nx+2*NGHX), Ny+2*NGHY);
  } else {
    #ifndef NOPITCH
      cudaMallocPitch (&arr_gpu, &pitch, (Ny+2*NGHY)*sizeof(real), Nz+2*NGHZ);
    #else
      cudaMalloc (&arr_gpu, (Ny+2*NGHY)*(Nz+2*NGHZ)*sizeof(real));
      pitch = (Ny+2*NGHY)*sizeof(real);
    #endif
  }
  check_errors ("CreateField");
  field->cpu_pp = make_cudaPitchedPtr (array, (Nx+2*NGHX)*sizeof(real), (Nx+2*NGHX), Ny+2*NGHY);
  masterprint("Field %s has been created on the GPU\n", name);

  field->fresh_cpu     =  YES;
  for (i = 0; i < 4; i++) {
    field->fresh_inside_contour_cpu[i] = YES;
    field->fresh_outside_contour_cpu[i] = YES;
  }
  field->fresh_gpu     =  NO;
  for (i = 0; i < 4; i++) {
    field->fresh_inside_contour_gpu[i] = NO;
    field->fresh_outside_contour_gpu[i] = NO;
  }

  field->field_gpu     =  (real *)arr_gpu;

  //OVERWRITING A LOT OF TIMES THE SAME VARIABLES
  Pitch_gpu            =  pitch/sizeof(real);
  Stride_gpu           =  Pitch_gpu*(Ny+2*NGHY);
  if ((Nx+2*NGHX) == 1) {
    Pitch_gpu = 1;
    Stride_gpu = pitch/sizeof(real);
  }
#if DEBUG
  masterprint("------>>>Pitch of %s = %d\n",field->name,pitch);
#endif
  Host2Dev3D(field); // Do NOT remove this
#endif
  Pitch_cpu            =  (Nx+2*NGHX);
  Stride_cpu           =  Pitch_cpu*(Ny+2*NGHY);

  field->type = type;

  if (sx)
    field->x = Xmin;
  else
    field->x = Xmed;
  if (sy)
    field->y = Ymin;
  else
    field->y = Ymed;
  if (sz)
    field->z = Zmin;
  else
    field->z = Zmed;

  return field;
}

Field2D *CreateField2D(char *name, int dim) {
  Field2D *field;
  real *array;
  void *arr_gpu;
  char *string;
  int i,j,k;
  size_t pitch;
  int size1, size2;

  if (dim == YZ) {
    size1 = Ny+2*NGHY;
    size2 = Nz+2*NGHZ;
  }

  if (dim == ZX) {
    size1 = Nx+2*NGHX;
    size2 = Nz+2*NGHZ;
  }

  if (dim == XY) {
    size1 = Nx+2*NGHX;
    size2 = Ny+2*NGHY;
  }

  field = (Field2D *) malloc(sizeof(Field));
  if (field == NULL)
    prs_error("Insufficient memory for Field2D creation-step1.");

#if (!GPU)
  array = (real *) malloc(sizeof(real)*size1*size2);
#else
#ifndef PINNED
  array = (real *) malloc(sizeof(real)*size1*size2);
#else
  cudaMallocHost((void**)&array,sizeof(real)*size1*size2);
#endif
#endif


  if (array == NULL)
    prs_error("Insufficient memory for Field2D creation-step2.");
  string = (char *) malloc(sizeof(char) * 80);
  if (string == NULL)
    prs_error("Insufficient memory for Field2D creation-step3.");
  sprintf(string, "%s", name);
  field->field_cpu = array;
  field->name = string;

  i = j = k = 0;

  for (i = 0; i < size1*size2; i++)
    array[i] = 0.0;

  masterprint("Field2D %s has been created\n", name);
  //Now on the GPU
#if GPU
  if(cudaMallocPitch(&arr_gpu, &pitch, size1*sizeof(real), size2) == cudaSuccess){
    masterprint("Field %s has created on the GPU\n", name);
    masterprint("Pitch = %d bytes (%d elements)\n", (int)pitch, (int)(pitch/sizeof(real)));
    field->field_gpu = (real*)arr_gpu;
    field->pitch = pitch/sizeof(real); //number of elements
  }
  else{
    masterprint("There was an error allocating %s on the GPU.\n", field->name);
    check_errors ("CreateField2D");
    MPI_Finalize();
    exit(1);
  }
  field->fresh_gpu     =  NO;
  if (dim == YZ) // Backward compatibility (old 2D arrays were only YZ).
    Pitch2D = pitch/sizeof(real);
  //If the array is not in YZ, we store its pitch in a new field of the 2D structure.
#endif
  field->fresh_cpu     =  YES;
  field->kind = dim;
  return field;
}

FieldInt2D *CreateFieldInt2D(char *name) {
  FieldInt2D *field;
  int *array;
  void *arr_gpu;
  char *string;
  int i,j,k;
  size_t pitch;

  field = (FieldInt2D *) malloc(sizeof(Field));
  if (field == NULL)
    prs_error("Insufficient memory for FieldInt2D creation-step1.");

#if (!GPU)
  array = (int *) malloc(sizeof(int)*(Ny+2*NGHY)*(Nz+2*NGHZ));
#else
#ifndef PINNED
  array = (int *) malloc(sizeof(int)*(Ny+2*NGHY)*(Nz+2*NGHZ));
#else
  cudaMallocHost((void**)&array,sizeof(int)*(Ny+2*NGHY)*(Nz+2*NGHZ));
#endif
#endif

  if (array == NULL)
    prs_error("Insufficient memory for FieldInt2D creation-step2.");
  string = (char *) malloc(sizeof(char) * 80);
  if (string == NULL)
    prs_error("Insufficient memory for FieldInt2D creation-step3.");
  sprintf(string, "%s", name);
  field->field_cpu = array;
  field->backup = NULL;
  field->secondary_backup = NULL;
  field->name = string;

  i = j = k = 0;

#if ZDIM
  for (k = 0; k<Nz+2*NGHZ; k++) {
#endif
#if YDIM
    for (j = 0; j<Ny+2*NGHY; j++) {
#endif
	array[l2D] = 0;
#if YDIM
    }
#endif
#if ZDIM
  }
#endif
  masterprint("Field2D %s has been created\n", name);
  //Now on the GPU
#if GPU
  cudaMallocPitch (&arr_gpu, &pitch, (Ny+2*NGHY)*sizeof(int), Nz+2*NGHZ);
  check_errors ("CreateFieldInt2D");
  masterprint("Integer field %s has been created on the GPU\n", name);
  field->field_gpu =  (int*)arr_gpu;
#endif
  Pitch_Int_gpu = pitch/sizeof(int);
  field->fresh_cpu     =  YES;
  return field;
}
