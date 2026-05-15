#include "fargo3d.h"

void send2cpu() {
#ifdef GPU
  Field* g;
  g = ListOfGrids;
  printf("\nCopying Fields--------------------------------------\n");
  while(g != NULL){
    if(Dev2Host3D(g)){
      printf("Error in send2cpu() with the field %s\n.", g->name);
      exit(-1);
    }
    else {
      printf("Field %s has been copied (Dev2Host)\n", g->name);
    }
    g = g->next;
  }
  printf("\n--------------------------------------------------\n\n");
#endif
}

void send2gpu() {
#ifdef GPU
  Field* g;
  g = ListOfGrids;
  printf("\nCopying Fields--------------------------------------\n");
  while(g != NULL){
    if(Host2Dev3D(g)){
      printf("Error in send2gpu() with the field %s\n.", g->name);
      exit(-1);
    }
    else {
      printf("Field %s has been copied (Host2Dev)\n", g->name);
    }
    g = g->next;
  }
  printf("\n--------------------------------------------------\n\n");
#endif
}

void Input2D_CPU(Field2D *field, int line, const char *string){
  int status;
#ifdef GPU
  if(!field->fresh_cpu) {
    status = Dev2Host2D(field);
    //    printf("%d\n",status);
    //    printf("Copying %s from Dev to Host\n",field->name);
    if(status!=0) {
      printf("Error %d in Dev2Host2D()! Field: %s\n",status,field->name);
      printf("called from line %d of file %s\n", line, string);
      exit(EXIT_FAILURE);
    }
  }
//  else {
//    printf("Field %s is up to date on Host.\n",field->name);
//  }
  field->fresh_cpu = YES;
#endif
  return;
}

void Input2D_GPU(Field2D *field, int line, const char *string){
  int status;
#ifdef GPU
  if(!field->fresh_gpu) {
    status = Host2Dev2D(field);
    //    printf("%d\n",status);
    //    printf("Copying %s from Dev to Host\n",field->name);
    if(status!=0) {
      printf("Error %d in Host2Dev2D()! Field: %s\n",status,field->name);
      printf("called from line %d of file %s\n", line, string);
      exit(EXIT_FAILURE);
    }
  }
//  else {
//    printf("Field %s is up to date on Host.\n",field->name);
//  }
  field->fresh_gpu = YES;
#endif
  return;
}

void Input2DInt_CPU(FieldInt2D *field, int line, const char *string){
  int status;
#ifdef GPU
  if(!field->fresh_cpu) {
    status = Dev2Host2DInt(field);
    //    printf("%d\n",status);
    //    printf("Copying %s from Dev to Host\n",field->name);
    if(status!=0) {
      printf("Error %d in Dev2Host2DInt()! Field: %s\n",status,field->name);
      printf("called from line %d of file %s\n", line, string);
      exit(EXIT_FAILURE);
    }
  }
//  else {
//    printf("Field %s is up to date on Host.\n",field->name);
//  }
  field->fresh_cpu = YES;
#endif
  return;
}

void Input2DInt_GPU(FieldInt2D *field, int line, const char *string){
  int status;
#ifdef GPU
  if(!field->fresh_gpu) {
    status = Host2Dev2DInt(field);
    //    printf("%d\n",status);
    //    printf("Copying %s from Dev to Host\n",field->name);
    if(status!=0) {
      printf("Error %d in Host2Dev2DInt()! Field: %s\n",status,field->name);
      printf("called from line %d of file %s\n", line, string);
      exit(EXIT_FAILURE);
    }
  }
//  else {
//    printf("Field %s is up to date on Host.\n",field->name);
//  }
  field->fresh_gpu = YES;
#endif
  return;
}

void Input_CPU(Field *field, int line, const char *string){
  int status, i;
  boolean problem = NO, take_action = NO;
  if (*(field->owner) == NULL) {
    printf ("Error ! You pretend to use as an input\n");
    printf ("the field %s, at line %d of file %s.\n", field->name, line, string);
    printf ("However, this storage has been altered and used as a temporary field\n");
    printf ("At line %d in file %s\n", field->line_origin, field->file_origin);
    prs_exit (EXIT_FAILURE);
  }
  if (*(field->owner) != field) {
    printf ("Error ! You pretend to use as input a shared storage (see CreateFieldAlias()) for\n");
    printf ("the field %s, at line %d of file %s.\n", field->name, line, string);
    printf ("However, its storage area is presently used by %s\n", (*(field->owner))->name);
    printf ("Since the line %d of file %s\n", (*(field->owner))->line_origin, (*(field->owner))->file_origin);
    prs_exit (EXIT_FAILURE);
  }
#ifdef GPU
  if (!field->fresh_cpu) take_action = YES;
  for (i = 0; i < 4; i++) {
    if (field->fresh_inside_contour_cpu[i] == NO) take_action = YES;
    if (field->fresh_outside_contour_cpu[i] == NO) take_action = YES;
  }
  if(take_action) {
    if (field->fresh_gpu == NO) problem = YES;
    for (i = 0; i < 4; i++) {
      if (field->fresh_inside_contour_gpu[i] == NO) problem = YES;
      if (field->fresh_outside_contour_gpu[i] == NO) problem = YES;
    }
    if (problem) {
      printf ("Problem on CPU %d: you want to transfer the field %s\n", CPU_Rank, field->name);
      printf ("From device to host in file %s at line %d, but the\n", string, line);
      printf ("field is not fresh everywhere. Here are the values of\n");
      printf ("'fresh' on the mesh, on the inside contour, and outside contour\n");
      INSPECT_INT (field->fresh_gpu);
      for (i = 0; i < 4; i++)
	INSPECT_INT(field->fresh_inside_contour_gpu[i]);
      for (i = 0; i < 4; i++)
	INSPECT_INT(field->fresh_outside_contour_gpu[i]);
    }
    status = Dev2Host3D(field);
    FullArrayComms++;
    //    printf("%d\n",status);
    //    printf("D==>H: %s\n",field->name);
    if(status!=0) {
      printf("Error %d in Dev2Host3D()! Field: %s\n",status,field->name);
      printf("called from line %d of file %s\n", line, string);
      exit(0);
    }
  }
//  else {
//    printf("Field %s is up to date on Host.\n",field->name);
//  }
  field->fresh_cpu = YES;
  for (i = 0; i < 4; i++) {
    field->fresh_inside_contour_cpu[i] = YES;
    field->fresh_outside_contour_cpu[i] = YES;
  }
#endif
  return;
}

void Input_GPU(Field *field, int line, const char *string){
  int i, status;
  boolean problem=NO, take_action=NO;
  if (*(field->owner) == NULL) {
    printf ("Error ! You pretend to use as an input\n");
    printf ("the field %s, at line %d of file %s.\n", field->name, line, string);
    printf ("However, its storage area has been altered and used as a temporary field\n");
    printf ("At line %d in file %s\n", field->line_origin, field->file_origin);
    prs_exit (EXIT_FAILURE);
  }
  if (*(field->owner) != field) {
    printf ("Error ! You pretend to use as input a shared storage (see CreateFieldAlias()) for\n");
    printf ("the field %s, at line %d of file %s.\n", field->name, line, string);
    printf ("However, its storage area is presently used by %s\n", (*(field->owner))->name);
    printf ("Since the line %d of file %s\n", (*(field->owner))->line_origin, (*(field->owner))->file_origin);
    prs_exit (EXIT_FAILURE);
  }
#ifdef GPU
  if (!field->fresh_gpu) take_action = YES;
  for (i = 0; i < 4; i++) {
    if (field->fresh_inside_contour_gpu[i] == NO) take_action = YES;
    if (field->fresh_outside_contour_gpu[i] == NO) take_action = YES;
  }
  if(take_action) {
    if (field->fresh_cpu == NO) problem = YES;
    for (i = 0; i < 4; i++) {
      if (field->fresh_inside_contour_cpu[i] == NO) problem = YES;
      if (field->fresh_outside_contour_cpu[i] == NO) problem = YES;
    }
    if (problem) {
      printf ("Problem on CPU %d: you want to transfer the field %s\n", CPU_Rank, field->name);
      printf ("From host to device in file %s at line %d, but the\n", string, line);
      printf ("field is not fresh everywhere. Here are the values of\n");
      printf ("'fresh' on the mesh, on the inside contour, and outside contour\n");
      INSPECT_INT (field->fresh_cpu);
      for (i = 0; i < 4; i++)
	INSPECT_INT(field->fresh_inside_contour_cpu[i]);
      for (i = 0; i < 4; i++)
	INSPECT_INT(field->fresh_outside_contour_cpu[i]);
    }
    status = Host2Dev3D(field);
    FullArrayComms++;
    //    printf("H==>D: %s\n",field->name);
    if(status!=0) {
      printf("Error %d in Host2Dev3D()! Field: %s\n",status,field->name);
      printf("called from line %d of file %s\n", line, string);
      exit(0);
    }
  }
//  else {
//    printf("Field %s is up to date on Device.\n",field->name);
//  }
  field->fresh_gpu = YES;
  for (i = 0; i < 4; i++) {
    field->fresh_inside_contour_gpu[i] = YES;
    field->fresh_outside_contour_gpu[i] = YES;
  }
#endif
  return;
}

void Output_CPU(Field *field, int line, const char *string){
  int i;
  field->line_origin = line;
  strncpy(field->file_origin, string, MAXLINELENGTH-1);
  field->fresh_cpu = YES;
  field->fresh_gpu = NO;
  for (i = 0; i < 4; i++) {
    field->fresh_inside_contour_cpu[i] = YES;
    field->fresh_outside_contour_cpu[i] = YES;
    field->fresh_inside_contour_gpu[i] = NO;
    field->fresh_outside_contour_gpu[i] = NO;
  }
  *(field->owner) = field;
}

void Output_GPU(Field *field, int line, const char *string){
  int i;
  field->line_origin = line;
  strncpy(field->file_origin, string, MAXLINELENGTH-1);
  field->fresh_cpu = NO;
  field->fresh_gpu = YES;
  for (i = 0; i < 4; i++) {
    field->fresh_inside_contour_cpu[i] = NO;
    field->fresh_outside_contour_cpu[i] = NO;
    field->fresh_inside_contour_gpu[i] = YES;
    field->fresh_outside_contour_gpu[i] = YES;
  }
  *(field->owner) = field;
}

void Output2D_CPU(Field2D *field, int line, const char *string){
  field->fresh_cpu = YES;
  field->fresh_gpu = NO;
}

void Output2D_GPU(Field2D *field, int line, const char *string){
  field->fresh_gpu = YES;
  field->fresh_cpu = NO;
}

void Output2DInt_CPU(FieldInt2D *field, int line, const char *string){
  field->fresh_cpu = YES;
  field->fresh_gpu = NO;
}

void Output2DInt_GPU(FieldInt2D *field, int line, const char *string){
  field->fresh_gpu = YES;
  field->fresh_cpu = NO;
}

void Draft (Field *field, int line, const char *string) {
  *(field->owner) = NULL;
  strncpy (field->file_origin, string, MAXLINELENGTH-1);
  field->line_origin = line;
}

void WhereIsField(Field *field) {
  int id=0;

  if(field->fresh_cpu){
    printf("Field %s is fresh on the CPU %d\n",field->name,id);
    id=1;
  }
  if(field->fresh_gpu)
    printf("Field %s is fresh on the GPU %d\n",field->name,id);
}

void WhereIsFieldInt2D(FieldInt2D *field) {
  int id=0;

  if(field->fresh_cpu){
    printf("Field %s is fresh on the CPU %d\n",field->name,id);
    id=1;
  }
  if(field->fresh_gpu)
    printf("Field %s is fresh on the GPU %d\n",field->name,id);
}

void WhoOwns (Field *field) {
  printf ("Field %s is currently storing %s\n", field->name, (*(field->owner))->name);
}

void SynchronizeHD () {
#ifdef GPU
  Field *current;
  current = ListOfGrids;
  while (current != NULL) {
    if (*(current->owner) == current) {
      if ((current->fresh_cpu == YES) && (current->fresh_gpu == NO)) {
	Host2Dev3D (current);
	//      printf ("Sending %s to device\n", current->name);
	current->fresh_gpu = YES;
      }
      if ((current->fresh_cpu == NO) && (current->fresh_gpu == YES)) {
	Dev2Host3D (current);
	//printf ("Sending %s to host\n", current->name);
	current->fresh_cpu = YES;
      }
    }
    current = current->next;
  }
#endif
}

void WhereIsWho () {
#ifdef GPU
  Field *current;
  char loc[3];
  current = ListOfGrids;
  loc[1]=0;
  while (current != NULL) {
    if (*(current->owner) == current) {
      if ((current->fresh_cpu == YES) && (current->fresh_gpu == YES))
	loc[0] = 'B';
      if ((current->fresh_cpu == NO) && (current->fresh_gpu == YES))
	loc[0] = 'G';
      if ((current->fresh_cpu == YES) && (current->fresh_gpu == NO))
	loc[0] = 'C';
      if ((current->fresh_cpu == NO) && (current->fresh_gpu == NO))
	loc[0] = '?';
      printf ("%-20s%s\n",current->name, loc);
    }
    current = current->next;
  }
#endif
}
