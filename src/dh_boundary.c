#include "fargo3d.h"

#define LEFT 0
#define RIGHT 1
#define DOWN 2
#define UP 3
#define INSIDE 0
#define OUTSIDE 1

static boolean BuffersReady = NO;
static int StartAddress_CPU[4][2];
static int StartAddress_GPU[4][2];
static int Stride_j_CPU;
static int Stride_k_CPU;
static int Stride_j_GPU;
static int Stride_k_GPU;
static int Nj[4][2];
static int Nk[4][2];

#ifdef GPU
static struct cudaMemcpy3DParms Inner[4] = {0,0,0,0};
static struct cudaMemcpy3DParms Outer[4] = {0,0,0,0};
#endif 

void Prepare_DH_buffers () {
#ifdef GPU
  Inner[LEFT].srcPos = Inner[LEFT].dstPos = make_cudaPos (0, NGHY, 0);
  Outer[LEFT].extent = Inner[LEFT].extent = \
    make_cudaExtent ((Nx+2*NGHX)*sizeof(real), NGHY, Nz+2*NGHZ);
  Outer[LEFT].dstPos = Outer[LEFT].srcPos = make_cudaPos (0, 0, 0);

  ////////////////

  Inner[RIGHT].srcPos = Inner[RIGHT].dstPos = make_cudaPos (0, Ny, 0);
  Outer[RIGHT].extent = make_cudaExtent ((Nx+2*NGHX)*sizeof(real), NGHY, Nz+2*NGHZ);
  Inner[RIGHT].extent = make_cudaExtent ((Nx+2*NGHX)*sizeof(real), NGHY, Nz+2*NGHZ);
  Outer[RIGHT].dstPos = Outer[RIGHT].srcPos = make_cudaPos (0, Ny+NGHY, 0);

  ////////////////

  Inner[DOWN].srcPos = Inner[DOWN].dstPos = make_cudaPos (0, 0, NGHZ);
  Outer[DOWN].extent = Inner[DOWN].extent = \
    make_cudaExtent ((Nx+2*NGHX)*sizeof(real), Ny+2*NGHY, NGHZ);
  Outer[DOWN].dstPos = Outer[DOWN].srcPos = make_cudaPos (0, 0, 0);

  ////////////////

  Inner[UP].srcPos = Inner[UP].dstPos = make_cudaPos (0, 0, Nz);
  Inner[UP].extent = make_cudaExtent ((Nx+2*NGHX)*sizeof(real), Ny+2*NGHY, NGHZ);
  Outer[UP].extent = make_cudaExtent ((Nx+2*NGHX)*sizeof(real), Ny+2*NGHY, NGHZ);
  Outer[UP].dstPos = Outer[UP].srcPos = make_cudaPos (0, 0, Nz+NGHZ);

  ////////////////

  Stride_j_CPU = (Nx+2*NGHX);
  Stride_k_CPU = (Nx+2*NGHX)*(Ny+2*NGHY);
  Stride_j_GPU = Pitch_gpu;
  Stride_k_GPU = Stride_gpu;
  //////////////
  StartAddress_CPU[LEFT][OUTSIDE] = 0;
  StartAddress_GPU[LEFT][OUTSIDE] = 0;
  Nj[LEFT][OUTSIDE] = NGHY+1;
  Nk[LEFT][OUTSIDE] = Nz+2*NGHZ;

  StartAddress_CPU[LEFT][INSIDE] = NGHY*(Nx+2*NGHX);
  StartAddress_GPU[LEFT][INSIDE] = NGHY*Pitch_gpu;
  Nj[LEFT][INSIDE] = NGHY+1;
  Nk[LEFT][INSIDE] = Nz+2*NGHZ;
  
  StartAddress_CPU[RIGHT][OUTSIDE] = (Ny+NGHY)*(Nx+2*NGHX);
  StartAddress_GPU[RIGHT][OUTSIDE] = (Ny+NGHY)*Pitch_gpu;
  Nj[RIGHT][OUTSIDE] = NGHY;
  Nk[RIGHT][OUTSIDE] = Nz+2*NGHZ;

  StartAddress_CPU[RIGHT][INSIDE] = Ny*(Nx+2*NGHX);
  StartAddress_GPU[RIGHT][INSIDE] = Ny*Pitch_gpu;
  Nj[RIGHT][INSIDE] = NGHY+1;
  Nk[RIGHT][INSIDE] = Nz+2*NGHZ;
  
  StartAddress_CPU[DOWN][OUTSIDE] = 0;
  StartAddress_GPU[DOWN][OUTSIDE] = 0;
  Nj[DOWN][OUTSIDE] = Ny+2*NGHY;
  Nk[DOWN][OUTSIDE] = NGHZ+1;
  
  StartAddress_CPU[DOWN][INSIDE] = NGHZ*(Ny+2*NGHY)*(Nx+2*NGHX);
  StartAddress_GPU[DOWN][INSIDE] = NGHZ*Stride_gpu;
  Nj[DOWN][INSIDE] = Ny+2*NGHY;
  Nk[DOWN][INSIDE] = NGHZ+1;
  
  StartAddress_CPU[UP][OUTSIDE] = (Nz+NGHZ)*(Ny+2*NGHY)*(Nx+2*NGHX);
  StartAddress_GPU[UP][OUTSIDE] = (Nz+NGHZ)*Stride_gpu;
  Nj[UP][OUTSIDE] = Ny+2*NGHY;
  Nk[UP][OUTSIDE] = NGHZ;
  
  StartAddress_CPU[UP][INSIDE] = Nz*(Ny+2*NGHY)*(Nx+2*NGHX);
  StartAddress_GPU[UP][INSIDE] = Nz*Stride_gpu;
  Nj[UP][INSIDE] = Ny+2*NGHY;
  Nk[UP][INSIDE] = NGHZ+1;

  BuffersReady = YES;
#endif
}


void Input_Contour_Inside (Field *f, int side) { // Active zones go from D to H
#ifdef GPU
  if (BuffersReady == NO) Prepare_DH_buffers();
  if (side>3) return;
  if (f->fresh_inside_contour_cpu[side] == YES) return;
  //  printf ("Field %s sends boundary %d from D to H\n", f->name, side);
  if ((Nx+2*NGHX) == 1) {
    cudaMemcpy2D(f->field_cpu+StartAddress_CPU[side][INSIDE],
		 (Ny+2*NGHY)*sizeof(real),
		 f->field_gpu+StartAddress_GPU[side][INSIDE],
		 Stride_k_GPU*sizeof(real),
		 Nj[side][INSIDE]*sizeof(real),Nk[side][INSIDE],
		 cudaMemcpyDeviceToHost);
  } else {
    Inner[side].kind = cudaMemcpyDeviceToHost;
    Inner[side].srcPtr = f->gpu_pp;
    Inner[side].dstPtr = f->cpu_pp;
    cudaMemcpy3D (&(Inner[side]));
    check_errors ("memcpy3d");
  }  
  ContourComms++;
  f->fresh_inside_contour_cpu[side] = YES;
#endif
}

void Output_Contour_Outside (Field *f, int side) { // Ghost zones go from H to D
#ifdef GPU
  if (BuffersReady == NO) Prepare_DH_buffers();
  if (side>3) return;
  if (f->fresh_outside_contour_gpu[side] == YES) return;
  //  printf ("Field %s sends boundary %d from H to D\n", f->name, side);
  if ((Nx+2*NGHX) == 1) {
    cudaMemcpy2D(f->field_gpu+StartAddress_GPU[side][OUTSIDE],
		 Stride_k_GPU*sizeof(real),
		 f->field_cpu+StartAddress_CPU[side][OUTSIDE],
		 (Ny+2*NGHY)*sizeof(real),
		 Nj[side][OUTSIDE]*sizeof(real),Nk[side][OUTSIDE],
		 cudaMemcpyHostToDevice);
  } else {
    Outer[side].kind = cudaMemcpyHostToDevice;
    Outer[side].srcPtr = f->cpu_pp;
    Outer[side].dstPtr = f->gpu_pp;
    cudaMemcpy3D (&(Outer[side]));
    check_errors ("memcpy3d_outer");
  }
  ContourComms++;
  f->fresh_outside_contour_gpu[side] = YES;
#endif
}
