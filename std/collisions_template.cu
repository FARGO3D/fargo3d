#define __GPU
#define __NOPROTO

#include "fargo3d.h"

#define ymin(i) ymin_s[(i)]
CONSTANT(real, ymin_s, 3846);

__global__
void _collisions_kernel(real dt, int id1, int id2, int id3,
    int option, int pitch, int stride, int size_x, int size_y, int size_z,
    real *alpha, {fluids_defs}) {{

  int i;
  int j;
  int k;
  int ll;
  int o;
  int p;
  int q;
  int ir;
  int ir2;
  int ir_max;
  int ic;
  real max_value;
  real factor;
  real big;
  real temp;
  real sum;
  int idm;
  real b[NFLUIDS];
  real m[NFLUIDS*NFLUIDS];
  real omega;
  real rho_p;
  real rho_o;
  real rho_q;

  {fluids_assign}

#if XDIM
  i = threadIdx.x + blockIdx.x * blockDim.x;
#else
  i = 0;
#endif
#if YDIM
  j = threadIdx.y + blockIdx.y * blockDim.y;
#else
  j = 0;
#endif
#if ZDIM
  k = threadIdx.z + blockIdx.z * blockDim.z;
#else
  k = 0;
#endif

#if ZDIM
  if (k>=1 && k<size_z) {{
#endif
#if YDIM
    if (j>=1 && j<size_y) {{
#endif
#if XDIM
      if (i<size_x) {{
#endif

#include "collision_kernel.h"
#include "gauss.h"

        for (o=0; o<NFLUIDS; o++) {{
          velocities_output[o][l] = b[o];
        }}

#if XDIM
      }}
#endif
#if YDIM
    }}
#endif
#if ZDIM
  }}
#endif
}}

extern "C" void _collisions_gpu(real dt,
    int id1, int id2, int id3, int option) {{

  real *rho[NFLUIDS];
  real *velocities_input[NFLUIDS];
  real *velocities_output[NFLUIDS];

  int ii;

  for (ii=0; ii<NFLUIDS; ii++) {{

    INPUT(Fluids[ii]->Density);
    rho[ii]  = Fluids[ii]->Density->field_gpu;

    //Collisions along X
#if XDIM
    if (id1 == 1) {{
      if (option == 1) {{
        INPUT(Fluids[ii]->Vx_temp);
        OUTPUT(Fluids[ii]->Vx_temp);
        velocities_input[ii] = Fluids[ii]->Vx_temp->field_gpu;
        velocities_output[ii] = Fluids[ii]->Vx_temp->field_gpu;
      }}
      if (option == 0) {{
        INPUT(Fluids[ii]->Vx);
        OUTPUT(Fluids[ii]->Vx_half);
        velocities_input[ii] = Fluids[ii]->Vx->field_gpu;
        velocities_output[ii] = Fluids[ii]->Vx_half->field_gpu;
      }}
    }}
#endif

    //Collisions along Y
#if YDIM
    if (id2 == 1) {{
      if (option == 1) {{
        INPUT(Fluids[ii]->Vy_temp);
        OUTPUT(Fluids[ii]->Vy_temp);
        velocities_input[ii] = Fluids[ii]->Vy_temp->field_gpu;
        velocities_output[ii] = Fluids[ii]->Vy_temp->field_gpu;
      }}
      if (option == 0) {{
        INPUT(Fluids[ii]->Vy);
        OUTPUT(Fluids[ii]->Vy_half);
        velocities_input[ii] = Fluids[ii]->Vy->field_gpu;
        velocities_output[ii] = Fluids[ii]->Vy_half->field_gpu;
      }}
    }}
#endif

    //Collisions along Z
#if ZDIM
    if (id3 == 1) {{
      if (option == 1) {{
        INPUT(Fluids[ii]->Vz_temp);
        OUTPUT(Fluids[ii]->Vz_temp);
        velocities_input[ii] = Fluids[ii]->Vz_temp->field_gpu;
        velocities_output[ii] = Fluids[ii]->Vz_temp->field_gpu;
      }}
      if (option == 0) {{
        INPUT(Fluids[ii]->Vz);
        OUTPUT(Fluids[ii]->Vz_half);
        velocities_input[ii] = Fluids[ii]->Vz->field_gpu;
        velocities_output[ii] = Fluids[ii]->Vz_half->field_gpu;
      }}
    }}
#endif
  }}

  dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
  dim3 grid((Nx+2*NGHX+block.x-1)/block.x,
            ((Ny+2*NGHY)+block.y-1)/block.y,
            ((Nz+2*NGHZ)+block.z-1)/block.z);

#ifdef BIGMEM
#define ymin_d &Ymin_d
#endif

  CUDAMEMCPY(ymin_s, ymin_d, sizeof(real)*(Ny+2*NGHY+1),
    0, cudaMemcpyDeviceToDevice);

  cudaFuncSetCacheConfig(_collisions_kernel, cudaFuncCachePreferL1);

  _collisions_kernel<<<grid, block>>>(dt, id1, id2, id3, option,
    Pitch_gpu, Stride_gpu, XIP, Ny+2*NGHY, Nz+2*NGHZ, Alpha_d,
    {fluids_inputs});

  check_errors("_collisions_kernel");
}}
