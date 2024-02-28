#include "fargo3d.h"

void copy_velocities_cpu(int option) {
#if XDIM
  real *vx = Vx->field_cpu;
  real *vx_temp = Vx_temp->field_cpu;
#endif
#if YDIM
  real *vy = Vy->field_cpu;
  real *vy_temp = Vy_temp->field_cpu;
#endif
#if ZDIM
  real *vz = Vz->field_cpu;
  real *vz_temp = Vz_temp->field_cpu;
#endif
  if (option == VTEMP2V) {
#if XDIM
    INPUT (Vx_temp);
    OUTPUT (Vx);
    memcpy (vx, vx_temp, sizeof(real)*(Nx+2*NGHX)*(Ny+2*NGHY)*(Nz+2*NGHZ));
#endif
#if YDIM
    INPUT (Vy_temp);
    OUTPUT (Vy);
    memcpy (vy, vy_temp, sizeof(real)*(Nx+2*NGHX)*(Ny+2*NGHY)*(Nz+2*NGHZ));
#endif
#if ZDIM
    INPUT (Vz_temp);
    OUTPUT (Vz);
    memcpy (vz, vz_temp, sizeof(real)*(Nx+2*NGHX)*(Ny+2*NGHY)*(Nz+2*NGHZ));
#endif
  } else {
#if XDIM
    INPUT (Vx);
    OUTPUT (Vx_temp);
    memcpy (vx_temp, vx, sizeof(real)*(Nx+2*NGHX)*(Ny+2*NGHY)*(Nz+2*NGHZ));
#endif
#if YDIM
    INPUT (Vy);
    OUTPUT (Vy_temp);
    memcpy (vy_temp, vy, sizeof(real)*(Nx+2*NGHX)*(Ny+2*NGHY)*(Nz+2*NGHZ));
#endif
#if ZDIM
    INPUT (Vz);
    OUTPUT (Vz_temp);
    memcpy (vz_temp, vz, sizeof(real)*(Nx+2*NGHX)*(Ny+2*NGHY)*(Nz+2*NGHZ));
#endif
  }
}

void copy_velocities_gpu(int option) {
#if GPU
  int width, height, pitch;
#if XDIM
  real *vx = Vx->field_gpu;
  real *vx_temp = Vx_temp->field_gpu;
#endif
#if YDIM
  real *vy = Vy->field_gpu;
  real *vy_temp = Vy_temp->field_gpu;
#endif
#if ZDIM
  real *vz = Vz->field_gpu;
  real *vz_temp = Vz_temp->field_gpu;
#endif
  if (option == VTEMP2V) {
#if XDIM
    Input_GPU(Vx_temp, __LINE__, __FILE__);
    Output_GPU(Vx, __LINE__, __FILE__);
#endif
#if YDIM
    Input_GPU(Vy_temp, __LINE__, __FILE__);
    Output_GPU(Vy, __LINE__, __FILE__);
#endif
#if ZDIM
    Input_GPU(Vz_temp, __LINE__, __FILE__);
    Output_GPU(Vz, __LINE__, __FILE__);
#endif
  } else {
#if XDIM
    Input_GPU(Vx, __LINE__, __FILE__);
    Output_GPU(Vx_temp, __LINE__, __FILE__);
#endif
#if YDIM
    Input_GPU(Vy, __LINE__, __FILE__);
    Output_GPU(Vy_temp, __LINE__, __FILE__);
#endif
#if ZDIM
    Input_GPU(Vz, __LINE__, __FILE__);
    Output_GPU(Vz_temp, __LINE__, __FILE__);
#endif
  }
  if (Nx+2*NGHX == 1) { //Actually we need something like check mute in x direction also
    pitch = Stride_gpu * sizeof(real);
    width = (Ny+2*NGHY)*sizeof(real);
    height = Nz+2*NGHZ;
  } else {
    pitch = Pitch_gpu * sizeof(real);
    width = (Nx+2*NGHX)*sizeof(real);
    height = (Ny+2*NGHY)*(Nz+2*NGHZ);
  }
  if (option == 1) {
#if XDIM
    cudaMemcpy2D (vx, pitch, vx_temp, pitch, width, height ,cudaMemcpyDeviceToDevice);
#endif
#if YDIM
    cudaMemcpy2D (vy, pitch, vy_temp, pitch, width, height ,cudaMemcpyDeviceToDevice);
#endif
#if ZDIM
    cudaMemcpy2D (vz, pitch, vz_temp, pitch, width, height ,cudaMemcpyDeviceToDevice);
#endif
  } else {
#if XDIM
    cudaMemcpy2D (vx_temp, pitch, vx, pitch, width, height ,cudaMemcpyDeviceToDevice);
#endif
#if YDIM
    cudaMemcpy2D (vy_temp, pitch, vy, pitch, width, height ,cudaMemcpyDeviceToDevice);
#endif
#if ZDIM
    cudaMemcpy2D (vz_temp, pitch, vz, pitch, width, height ,cudaMemcpyDeviceToDevice);
#endif
  }
#endif
}
