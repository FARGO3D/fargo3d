#include "fargo3d.h"

void copy_field_cpu(Field *Dst, Field *Src) {

  real *dst = Dst->field_cpu;
  real *src = Src->field_cpu;
  
  INPUT(Src);
  OUTPUT(Dst);
  memcpy (dst, src, sizeof(real)*(Nx+2*NGHX)*(Ny+2*NGHY)*(Nz+2*NGHZ));

}

void copy_field_gpu(Field *Dst, Field *Src) {

#ifdef GPU
  int width, height, pitch;

  real *dst = Dst->field_gpu;
  real *src = Src->field_gpu;

  Input_GPU(Src, __LINE__, __FILE__);
  Output_GPU(Dst, __LINE__, __FILE__);
  
  if (Nx+2*NGHX == 1) { //Actually we need something like check mute in x direction also
    pitch = Stride_gpu * sizeof(real);
    width = (Ny+2*NGHY)*sizeof(real);
    height = Nz+2*NGHZ;
  } else {
    pitch = Pitch_gpu * sizeof(real);
    width = (Nx+2*NGHX)*sizeof(real);
    height = (Ny+2*NGHY)*(Nz+2*NGHZ);
  }

  cudaMemcpy2D (dst, pitch, src, pitch, width, height ,cudaMemcpyDeviceToDevice);
#endif

}
