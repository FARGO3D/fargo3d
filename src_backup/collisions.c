//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void _collisions_cpu(real dt, int id1, int id2, int id3, int option) {
  
//<USER_DEFINED>

  real *rho[NFLUIDS];
  real *velocities_input[NFLUIDS];
  real *velocities_output[NFLUIDS];

  int ii;

  for (ii=0; ii<NFLUIDS; ii++) {

    INPUT(Fluids[ii]->Density);
    rho[ii]  = Fluids[ii]->Density->field_cpu;
    
    //Collisions along X
    #ifdef X
    if (id1 == 1) {
      if (option == 1) {
	INPUT(Fluids[ii]->Vx_temp);
	OUTPUT(Fluids[ii]->Vx_temp);
	velocities_input[ii] = Fluids[ii]->Vx_temp->field_cpu;
	velocities_output[ii] = Fluids[ii]->Vx_temp->field_cpu;
      }
      if (option == 0) {
	INPUT(Fluids[ii]->Vx);
	OUTPUT(Fluids[ii]->Vx_half);
	velocities_input[ii] = Fluids[ii]->Vx->field_cpu;
	velocities_output[ii] = Fluids[ii]->Vx_half->field_cpu;
      }
    }
    #endif
    
    //Collisions along Y
    #ifdef Y
    if (id2 == 1) {
      if (option == 1) {
	INPUT(Fluids[ii]->Vy_temp);
	OUTPUT(Fluids[ii]->Vy_temp);
	velocities_input[ii] = Fluids[ii]->Vy_temp->field_cpu;
	velocities_output[ii] = Fluids[ii]->Vy_temp->field_cpu;
      }
      if (option == 0) {
	INPUT(Fluids[ii]->Vy);
	OUTPUT(Fluids[ii]->Vy_half);
	velocities_input[ii] = Fluids[ii]->Vy->field_cpu;
	velocities_output[ii] = Fluids[ii]->Vy_half->field_cpu;
      }
    }
    #endif
    
    //Collisions along Z
    #ifdef Z
    if (id3 == 1) {
      if (option == 1) {
	INPUT(Fluids[ii]->Vz_temp);
	OUTPUT(Fluids[ii]->Vz_temp);
	velocities_input[ii] = Fluids[ii]->Vz_temp->field_cpu;
	velocities_output[ii] = Fluids[ii]->Vz_temp->field_cpu;
      }
      if (option == 0) {
	INPUT(Fluids[ii]->Vz);
	OUTPUT(Fluids[ii]->Vz_half);
	velocities_input[ii] = Fluids[ii]->Vz->field_cpu;
	velocities_output[ii] = Fluids[ii]->Vz_half->field_cpu;
      }
    }
    #endif
  }
//<\USER_DEFINED>

//<EXTERNAL>
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = XIP; 
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
  real* alpha = Alpha;
//<\EXTERNAL>

//<INTERNAL>
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
//<\INTERNAL>

//<CONSTANT>
// real Alpha(NFLUIDS*NFLUIDS);
//<\CONSTANT>

  
//<MAIN_LOOP>

  i = j = k = 0;

#ifdef Z
  for(k=1; k<size_z; k++) {
#endif
#ifdef Y
    for(j=1; j<size_y; j++) {
#endif
#ifdef X
      for(i=XIM; i<size_x; i++) {
#endif
//<#>
	
#include  "collision_kernel.h"
#include  "gauss.h"
	
	for (o=0; o<NFLUIDS; o++) {
	  velocities_output[o][l] = b[o];
	}

//<\#>
#ifdef X
      }
#endif
#ifdef Y
    }
#endif
#ifdef Z
  }
#endif
//<\MAIN_LOOP>
}

void Collisions(real dt, int option) {

  //Input and output velocities are the same Fields
  if (option == 1) {
#ifdef X
    //Collisions along the X direction
    FARGO_SAFE(_collisions(dt,1,0,0,option));
#endif
#ifdef Y
    //Collisions along the Y direction
    FARGO_SAFE(_collisions(dt,0,1,0,option));
#endif
#ifdef Z
    //Collisions along the Z direction
    FARGO_SAFE(_collisions(dt,0,0,1,option));
#endif
  }
  
  //Input and output velocities are not the same Fields
  if (option == 0) {
#ifdef X
    //Collisions along the X direction
    FARGO_SAFE(_collisions(dt,1,0,0,option));
#endif
#ifdef Y
    //Collisions along the Y direction
    FARGO_SAFE(_collisions(dt,0,1,0,option));
#endif
#ifdef Z
    //Collisions along the Z direction
    FARGO_SAFE(_collisions(dt,0,0,1,option));
#endif
  }
}
