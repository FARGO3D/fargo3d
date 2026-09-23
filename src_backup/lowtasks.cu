#define __NOPROTO
#define __GPU 

#include "fargo3d.h"

extern "C" void check_errors(const char *s) {
  cudaDeviceSynchronize();
  cudaError_t  cudaError = cudaGetLastError();
  const char *error; 
  error = cudaGetErrorString(cudaError);
  if(cudaError!=0) {
    printf("%s in %s\n",error, s);
    exit(EXIT_FAILURE);
  }
}

extern "C" int DevMalloc(void **v,size_t size) {
  int status;
  status = cudaMalloc(v,size);
  if(status != 0) {
    printf("\n----------------------------------------------------");
    printf("\nError allocating memory with DevMalloc. Error %d.\n", \
	   status);
    printf("----------------------------------------------------\n\n");
    exit(1);
  }
  else {
    return status;
  }
}

extern "C" int DevMemcpyD2H(void *dst, void *src, size_t size) {
  int status;
  status = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  if(status != 0) {
    printf("\n-------------------------------------------------");
    printf("\nError copying data with DevMemcpyD2H. Error %d.\n",	\
	   status);
    printf("--------------------------------------------------\n\n");
    exit(1);
  }
  else {
    return status;
  }
}

extern "C" int DevMemcpyH2D(void *dst, void *src, size_t size) {
  int status;
  status = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
  if(status != 0) {
    printf("\n-------------------------------------------------");
    printf("\nError copying data with DevMemcpyH2D. Error %d.\n",	\
	   status);
    printf("--------------------------------------------------\n\n");
    exit(1);
  }
  else {
    return status;
  }
}

extern "C" int DevMemcpyH2H(void *dst, void *src, size_t size) {
  int status;
  status = cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
  if(status != 0) {
    printf("\n-------------------------------------------------");
    printf("\nError copying data with DevMemcpyH2H. Error %d.\n",	\
	   status);
    printf("--------------------------------------------------\n\n");
    exit(1);
  }
  else {
    return status;
  }
}

extern "C" int DevMemcpyD2D(void *dst, void *src, size_t size) {
  int status;
  status = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
  if(status != 0) {
    printf("\n-------------------------------------------------");
    printf("\nError copying data with DevMemcpyD2D. Error %d.\n",	\
	   status);
    printf("--------------------------------------------------\n\n");
    exit(1);
  }
  else {
    return status;
  }
}

extern "C" int Host2Dev3D(Field *f) {
  if (Nx +2*NGHX== 1)
    return cudaMemcpy2D (f->field_gpu, Stride_gpu*sizeof(real), f->field_cpu, (Ny+2*NGHY)*sizeof(real),\
			 (Ny+2*NGHY)*sizeof(real),Nz+2*NGHZ,cudaMemcpyHostToDevice);
  else
    return cudaMemcpy2D (f->field_gpu, Pitch_gpu*sizeof(real), f->field_cpu, (Nx+2*NGHX)*sizeof(real), \
			 (Nx+2*NGHX)*sizeof(real),(Ny+2*NGHY)*(Nz+2*NGHZ),cudaMemcpyHostToDevice);
}

extern "C" int Dev2Dev3D(Field *fdst, Field *fsrc) {
  if (Nx +2*NGHX== 1)
    return cudaMemcpy2D (fdst->field_gpu, Stride_gpu*sizeof(real), fsrc->field_gpu, Stride_gpu*sizeof(real),\
			 (Ny+2*NGHY)*sizeof(real),Nz+2*NGHZ,cudaMemcpyDeviceToDevice);
  else
    return cudaMemcpy2D (fdst->field_gpu, Pitch_gpu*sizeof(real), fsrc->field_gpu, Pitch_gpu*sizeof(real), \
			 (Nx+2*NGHX)*sizeof(real),(Ny+2*NGHY)*(Nz+2*NGHZ),cudaMemcpyDeviceToDevice);
}

extern "C" int Dev2Host3D(Field *f) {
  if (Nx+2*NGHX == 1)
    return cudaMemcpy2D (f->field_cpu, (Ny+2*NGHY)*sizeof(real), f->field_gpu, Stride_gpu*sizeof(real),\
			 (Ny+2*NGHY)*sizeof(real),Nz+2*NGHZ,cudaMemcpyDeviceToHost);
  else
    return cudaMemcpy2D (f->field_cpu, (Nx+2*NGHX)*sizeof(real), f->field_gpu, Pitch_gpu*sizeof(real), \
			 (Nx+2*NGHX)*sizeof(real),(Ny+2*NGHY)*(Nz+2*NGHZ),cudaMemcpyDeviceToHost);
}

extern "C" int Host2Dev2D (Field2D *F) {
  if (F->kind == YZ)
    return (int)cudaMemcpy2D (F->field_gpu, Pitch2D*sizeof(real), F->field_cpu, (Ny+2*NGHY)*sizeof(real), \
			      (Ny+2*NGHY)*sizeof(real),Nz+2*NGHZ,cudaMemcpyHostToDevice);
  if (F->kind == ZX)
    return (int)cudaMemcpy2D (F->field_gpu, (F->pitch)*sizeof(real), F->field_cpu, (Nx+2*NGHX)*sizeof(real), \
			      (Nx+2*NGHX)*sizeof(real), Nz+2*NGHZ, cudaMemcpyHostToDevice);
  return 1; //To silence a warning
}

extern "C" int Dev2Host2D (Field2D *F) {
  if (F->kind == YZ)
    return (int)cudaMemcpy2D ((F->field_cpu), ((Ny+2*NGHY)*sizeof(real)), (F->field_gpu), (Pitch2D*sizeof(real)), \
			      ((Ny+2*NGHY)*sizeof(real)),(Nz+2*NGHZ),cudaMemcpyDeviceToHost);
  if (F->kind == ZX)
    return (int)cudaMemcpy2D (F->field_cpu, (Nx+2*NGHX)*sizeof(real), F->field_gpu, (F->pitch)*sizeof(real), \
			      (Nx+2*NGHX)*sizeof(real), Nz+2*NGHZ, cudaMemcpyDeviceToHost);
  return 1; //To silence a warning
}

extern "C" int Host2Dev2DInt (FieldInt2D *F) {
  return (int)cudaMemcpy2D (F->field_gpu, Pitch_Int_gpu*sizeof(int), F->field_cpu, (Ny+2*NGHY)*sizeof(int),\
			    (Ny+2*NGHY)*sizeof(int),Nz+2*NGHZ,cudaMemcpyHostToDevice);
}

extern "C" int Dev2Host2DInt (FieldInt2D *F) {
  return (int)cudaMemcpy2D (F->field_cpu, (Ny+2*NGHY)*sizeof(int), F->field_gpu, Pitch_Int_gpu*sizeof(int),\
			    (Ny+2*NGHY)*sizeof(int),Nz+2*NGHZ,cudaMemcpyDeviceToHost);
}
