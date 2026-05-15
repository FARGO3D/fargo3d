// This is the GPU version of the reduction of an n-dim array into an (n-1)-dim array
// The final reduction of this intermediate array is always performed on the CPU
// by the function in  "reduction_full_generic.c"

// The user should never have to interfere with this kernel.

#define __GPU
#include "fargo3d.h"

//Below: custom size of blocks. DO NOT CHANGE THIS !
#define LOCAL_BLOCK_X  64  /// DO NOT CHANGE - IMPORTANT
#define LOCAL_BLOCK_Y  4
#define LOCAL_BLOCK_Z  1  /// DO NOT CHANGE - IMPORTANT

__global__ void kernel_reduction(macro) (real *array, real *array_int, int pitch, int stride,\
					 int pitchb, int pitchj, int pitchk, int ymin, int ymax,\
					 int Nx, int offset, int ngh) {
  __shared__ double sdata[LOCAL_BLOCK_X*LOCAL_BLOCK_Y];
  unsigned int tid = threadIdx.x;
  unsigned int yt  = threadIdx.y * blockDim.x;
  unsigned int i   = ngh+threadIdx.x + blockIdx.x * blockDim.x * 2;
  unsigned int j   = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int k   = blockIdx.z;  // This kernel is written for blockDim.z = 1, necessarily.
  // This avoids to have to test the value of k and compare with  zmin and zmax.
  unsigned int ll;
  
  unsigned int ytid = yt+tid;

  if ((j >= ymin) && (j < ymax)) {
    ll = i+j*pitch+k*stride;
    sdata[ytid] = INIT_REDUCTION(macro);
    if (i < Nx+ngh)
      sdata[ytid] = array[ll];
    if (i+blockDim.x < Nx+ngh)
      sdata[ytid] = macro (sdata[ytid], array[ll+blockDim.x]);
    
    __syncthreads ();
    
    if (tid < 32) {
      volatile double *smem = sdata;
      smem[ytid] = macro (smem[ytid],smem[ytid+32]);
      smem[ytid] = macro (smem[ytid],smem[ytid+16]);
      smem[ytid] = macro (smem[ytid],smem[ytid+8]);
      smem[ytid] = macro (smem[ytid],smem[ytid+4]);
      smem[ytid] = macro (smem[ytid],smem[ytid+2]);
      smem[ytid] = macro (smem[ytid],smem[ytid+1]);
    }
    // We introduce below a series of pitches and strides, plus an
    // offset, in order to store the result of the block. Either the
    // result is stored in a 3D array, in which case we use adequate
    // values for pitchb (=1), pitchj (=Pitch_GPU) and pitchk
    // (=Stride_GPU). The result can be stored either at i=0, 1,
    // etc... up to the number of blocks, or further out (at "offset"
    // bytes from i=0). This last possibility is useful when the array
    // contains already partial sums, and we want to iterate the
    // kernel to sum these sums. In order not to overwrite the data
    // near i=0, we use the "free" space further out in 'i', in the
    // same array. Upon completion of the kernel, the results are
    // shifted back to i=0 by "kernel_shift_slice", which is the next
    // kernel in this file. Finally, when all the partial sums fit in
    // 2*LOCAL_BLOCK_X (factor of 2 because we already perform a sum
    // from the coalesced reads of global memory into the shared
    // memory), we write the final sums (over i) in the 2D array
    // Reduction2D. We therefore use for pitchb the value 0, for
    // pitchj the value 1 and for pitchk the value Pitch2D_GPU. No
    // offset can obviously be used in this case.
    if (tid == 0)
      array_int[pitchb*blockIdx.x+j*pitchj+k*pitchk+offset] = sdata[yt];
  }
}

__global__ void kernel_shift(macro)  (real *field, int pitch, int stride, int ymin, \
				     int ymax, int zmin, int zmax, int offset) {
  unsigned int i   = blockIdx.x;
  unsigned int j   = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int k   = threadIdx.z + blockIdx.z * blockDim.z;
  unsigned int ll;
  
  if ((j >= ymin) && (j < ymax) && (k >= zmin) && (k < zmax)) {
    ll = i+j*pitch+k*stride;
    field[ll] = field[ll+offset]; // See long comment above for the reason of this kernel
  }
}

extern "C" void name_reduction_gpu(macro) (Field *F, int ymin, int ymax, int zmin, int zmax) {
  Field *Work;
  Work = DivRho;
  DRAFT (Work);
  INPUT (F);
  OUTPUT2D (Reduction2D);
  int nx,nnx;

  dim3 block (LOCAL_BLOCK_X, LOCAL_BLOCK_Y, LOCAL_BLOCK_Z);
  dim3 grid ((Nx+block.x*2-1)/(block.x*2),
	     ((Ny+2*NGHY)+block.y-1)/block.y,
	     ((Nz+2*NGHZ)+block.z-1)/block.z);
  dim3 block_shift, grid_shift;

  nx = Nx;

  kernel_reduction(macro) <<< grid, block >>> (F->field_gpu, Work->field_gpu, Pitch_gpu, Stride_gpu, \
					       1, Pitch_gpu, Stride_gpu, ymin, ymax, nx, 0, NGHX);
#define str(s) #s
#define xstr(s) str(s)
 
  check_errors(xstr(name_reduction_gpu(macro)));

  nx = (nx+2*block.x-1)/(2*block.x);
  grid.x = (nx+block.x*2-1)/(2*block.x);

  while ((nnx=grid.x) > 1) { // Recursive part
    // This happens when Nx > (LOCAL_BLOCK_X*2)^2 = 16384...

    kernel_reduction(macro) <<< grid, block >>> (Work->field_gpu, Work->field_gpu, Pitch_gpu, \
						 Stride_gpu, 1, Pitch_gpu, Stride_gpu,\
						 ymin, ymax, nx, nx, 0);
    check_errors(xstr(name_reduction_gpu(macro)));
    block_shift.x = 1;
    block_shift.y = BLOCK_Y;
    block_shift.z = BLOCK_Z;
    grid_shift.x = nnx;
    grid_shift.y = ((Ny+2*NGHY)+block_shift.y-1)/block_shift.y;
    grid_shift.z = ((Nz+2*NGHZ)+block_shift.z-1)/block_shift.z;

    kernel_shift(macro) <<<grid_shift, block_shift >>> (Work->field_gpu, Pitch_gpu, Stride_gpu, \
						   ymin, ymax, zmin, zmax, nx);

    nx = (nx+2*block.x-1)/(2*block.x);
    grid.x = (nx+block.x*2-1)/(2*block.x);
    //End of recursive part
  }

  kernel_reduction(macro) <<< grid, block >>> (Work->field_gpu, Reduction2D->field_gpu, Pitch_gpu,\
					       Stride_gpu, 0, 1, Pitch2D, ymin, ymax, nx, 0, 0);

  check_errors(xstr(name_reduction_gpu(macro)));
  // The result is ultimately sent to the CPU, where the final 2D (or
  // 1D) sum will be performed (the gain resulting from sending this
  // to the GPU would be negligible... usually Nx is much larger than
  // the GPU/CPU speed up...)
}
