// This is the GPU version of the reduction of an n-dim array into an (n-1)-dim array
// The final reduction of this intermediate array is always performed on the CPU
// by the function in  "reduction_full_generic.c"

// The user should never have to interfere with this kernel.

#define __GPU
#include "fargo3d.h"

//Below: custom size of blocks. DO NOT CHANGE THIS !
#define LOCAL_BLOCK_X  256 /// DO NOT CHANGE - IMPORTANT
#define LOCAL_BLOCK_Y  1   /// DO NOT CHANGE - IMPORTANT
#define LOCAL_BLOCK_Z  1   /// DO NOT CHANGE - IMPORTANT
#define WARPSIZE  32


__global__ void kernel_reduction(macro) (real *array, real *array_int, int pitch, int stride,\
					 int pitchb, int pitchj, int pitchk, int ymin, int ymax,\
					 int Nx, int offset, int ngh) {
  
  __shared__ real sdata[LOCAL_BLOCK_X*LOCAL_BLOCK_Y/WARPSIZE];
  real wdata, shuffle;
  unsigned int s;
  unsigned int id  = threadIdx.x;
  unsigned int i   = threadIdx.x+ngh;
  unsigned int j   = blockIdx.y;
  unsigned int k   = blockIdx.z;  // This kernel is written for blockDim.z = 1, necessarily.
  // This avoids to have to test the value of k and compare with  zmin and zmax.
  unsigned int ll;
  
  unsigned int lane  = i%WARPSIZE;
  unsigned int warp  = i/WARPSIZE;   
  unsigned int warps = LOCAL_BLOCK_X/WARPSIZE;

  ll = j*pitch+k*stride;
  wdata = INIT_REDUCTION(macro);

  for (;i < Nx+ngh;i+=blockDim.x)wdata = macro (wdata,array[i+ll]);
    
  __syncthreads ();

  int lo, hi;
  for (s = WARPSIZE/2; s > 0; s >>= 1){
    
#ifndef FLOAT
    asm volatile("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"d"(wdata));
    // Shuffle the two 32b registers.
    lo = __shfl_down(lo,s,32);
    hi = __shfl_down(hi,s,32);
    // Recreate the 64b number.
    asm volatile("mov.b64 %0,{%1,%2};":"=d"(shuffle):"r"(lo),"r"(hi));	
#else
    shuffle = __shfl_down(wdata,s,WARPSIZE);
#endif
    wdata = macro (wdata,shuffle);
  }

  if (lane==0)sdata[warp]=wdata;

  __syncthreads();        
    
  wdata = (id < warps) ? sdata[id] : 0;
  
  if (warp==0){
    for (s = warps/2; s > 0; s >>= 1){
#ifndef FLOAT
    asm volatile("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"d"(wdata));
    // Shuffle the two 32b registers.
    lo = __shfl_down(lo,s,32);
    hi = __shfl_down(hi,s,32);
    // Recreate the 64b number.
    asm volatile("mov.b64 %0,{%1,%2};":"=d"(shuffle):"r"(lo),"r"(hi));	
#else
    shuffle = __shfl_down(wdata,s,WARPSIZE);
#endif
    wdata = macro (wdata,shuffle);
     }
  }
  
  if (id == 0)array_int[pitchb*blockIdx.x+j*pitchj+k*pitchk+offset] = wdata;
  
}

extern "C" void name_reduction_gpu(macro) (Field *F, int ymin, int ymax, int zmin, int zmax) {
  
  INPUT (F);
  OUTPUT2D (Reduction2D);
  
  dim3 block (LOCAL_BLOCK_X, LOCAL_BLOCK_Y, LOCAL_BLOCK_Z);
  dim3 grid (1,
	     ((Ny+2*NGHY)+block.y-1)/block.y,
	     ((Nz+2*NGHZ)+block.z-1)/block.z);

  kernel_reduction(macro) <<< grid, block >>> (F->field_gpu, Reduction2D->field_gpu, Pitch_gpu, Stride_gpu, \
					       0, 1, Pitch2D, ymin, ymax, Nx, 0, NGHX);
#define str(s) #s
#define xstr(s) str(s)
 
  check_errors(xstr(name_reduction_gpu(macro)));

  // The result is ultimately sent to the CPU, where the final 2D (or
  // 1D) sum will be performed (the gain resulting from sending this
  // to the GPU would be negligible... usually Nx is much larger than
  // the GPU/CPU speed up...)
}
