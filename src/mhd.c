#include "fargo3d.h"


void ComputeMHD(real dt) {
  
  // EMF in x
  FARGO_SAFE(ComputeSlopes(0,0,1,Vy,Slope_v1));
  FARGO_SAFE(ComputeSlopes(0,0,1,By,Slope_b1));
  FARGO_SAFE(ComputeSlopes(0,1,0,Vz,Slope_v2));
  FARGO_SAFE(ComputeSlopes(0,1,0,Bz,Slope_b2));

  FARGO_SAFE(ComputeStar(dt, 0,1,0, 0,0,1, 1, B1_star,V1_star,Slope_b2,Slope_v2,Slope_b1,Slope_v1));
  FARGO_SAFE(ComputeStar(dt, 0,0,1, 0,1,0, 1, B2_star,V2_star,Slope_b1,Slope_v1,Slope_b2,Slope_v2));

  FARGO_SAFE(ComputeEmf(dt, 1, 0, 0, B1_star, V1_star, B2_star, V2_star));

  // EMF in y
  FARGO_SAFE(ComputeSlopes(1,0,0,Vz,Slope_v1));
  FARGO_SAFE(ComputeSlopes(1,0,0,Bz,Slope_b1));
  FARGO_SAFE(ComputeSlopes(0,0,1,Vx,Slope_v2));
  FARGO_SAFE(ComputeSlopes(0,0,1,Bx,Slope_b2));

  FARGO_SAFE(ComputeStar(dt, 0,0,1, 1,0,0, 1, B1_star,V1_star,Slope_b2,Slope_v2,Slope_b1,Slope_v1));
  FARGO_SAFE(ComputeStar(dt, 1,0,0, 0,0,1, 1, B2_star,V2_star,Slope_b1,Slope_v1,Slope_b2,Slope_v2));

  FARGO_SAFE(ComputeEmf(dt, 0, 1, 0, B1_star, V1_star, B2_star, V2_star));

// EMF in z
  FARGO_SAFE(ComputeSlopes(0,1,0,Vx,Slope_v1));
  FARGO_SAFE(ComputeSlopes(0,1,0,Bx,Slope_b1));
  FARGO_SAFE(ComputeSlopes(1,0,0,Vy,Slope_v2));
  FARGO_SAFE(ComputeSlopes(1,0,0,By,Slope_b2));

  FARGO_SAFE(ComputeStar(dt, 1,0,0, 0,1,0, 1, B1_star,V1_star,Slope_b2,Slope_v2,Slope_b1,Slope_v1));
  FARGO_SAFE(ComputeStar(dt, 0,1,0, 1,0,0, 1, B2_star,V2_star,Slope_b1,Slope_v1,Slope_b2,Slope_v2));

  FARGO_SAFE(ComputeEmf(dt, 0, 0, 1, B1_star, V1_star, B2_star, V2_star));

  //---------------------ADD RESISTIVE TERMS TO EMFs

  FARGO_SAFE(Resist (1,0,0));
  FARGO_SAFE(Resist (0,1,0));
  FARGO_SAFE(Resist (0,0,1));
  
  //-------------------------------------------------------

#ifndef PASSIVEMHD
  FARGO_SAFE(ComputeSlopes(0,1,0,Bx,Slope_b1));
  FARGO_SAFE(ComputeSlopes(0,1,0,Vx,Slope_v1));
  FARGO_SAFE(ComputeSlopes(1,0,0,By,Slope_b2));
  FARGO_SAFE(ComputeSlopes(1,0,0,Vy,Slope_v2));
  FARGO_SAFE(ComputeStar(dt, 0,1,0, 1,0,0, 0, B1_star,V1_star,Slope_b1,Slope_v1,Slope_b2,Slope_v2));
  FARGO_SAFE(ComputeSlopes(0,0,1,Bx,Slope_b1));
  FARGO_SAFE(ComputeSlopes(0,0,1,Vx,Slope_v1));
  FARGO_SAFE(ComputeSlopes(1,0,0,Bz,Slope_b2));
  FARGO_SAFE(ComputeSlopes(1,0,0,Vz,Slope_v2));
  FARGO_SAFE(ComputeStar(dt, 0,0,1, 1,0,0, 0, B2_star,V2_star,Slope_b1,Slope_v1,Slope_b2,Slope_v2));
  FARGO_SAFE(LorentzForce(dt, B1_star, B2_star, 1, 0, 0));
  FARGO_SAFE(ComputeSlopes(0,0,1,By,Slope_b1));
  FARGO_SAFE(ComputeSlopes(0,0,1,Vy,Slope_v1));
  FARGO_SAFE(ComputeSlopes(0,1,0,Bz,Slope_b2));
  FARGO_SAFE(ComputeSlopes(0,1,0,Vz,Slope_v2));
  FARGO_SAFE(ComputeStar(dt, 0,0,1, 0,1,0, 0, B1_star,V1_star,Slope_b1,Slope_v1,Slope_b2,Slope_v2));
  FARGO_SAFE(ComputeSlopes(1,0,0,By,Slope_b1));
  FARGO_SAFE(ComputeSlopes(1,0,0,Vy,Slope_v1));
  FARGO_SAFE(ComputeSlopes(0,1,0,Bx,Slope_b2));
  FARGO_SAFE(ComputeSlopes(0,1,0,Vx,Slope_v2));
  FARGO_SAFE(ComputeStar(dt, 1,0,0, 0,1,0, 0, B2_star,V2_star,Slope_b1,Slope_v1,Slope_b2,Slope_v2));
  FARGO_SAFE(LorentzForce(dt, B1_star, B2_star, 0, 1, 0));
  FARGO_SAFE(ComputeSlopes(1,0,0,Bz,Slope_b1));
  FARGO_SAFE(ComputeSlopes(1,0,0,Vz,Slope_v1));
  FARGO_SAFE(ComputeSlopes(0,0,1,Bx,Slope_b2));
  FARGO_SAFE(ComputeSlopes(0,0,1,Vx,Slope_v2));
  FARGO_SAFE(ComputeStar(dt, 1,0,0, 0,0,1, 0, B1_star,V1_star,Slope_b1,Slope_v1,Slope_b2,Slope_v2));
  FARGO_SAFE(ComputeSlopes(0,1,0,Bz,Slope_b1));
  FARGO_SAFE(ComputeSlopes(0,1,0,Vz,Slope_v1));
  FARGO_SAFE(ComputeSlopes(0,0,1,By,Slope_b2));
  FARGO_SAFE(ComputeSlopes(0,0,1,Vy,Slope_v2));
  FARGO_SAFE(ComputeStar(dt, 0,1,0, 0,0,1, 0, B2_star,V2_star,Slope_b1,Slope_v1,Slope_b2,Slope_v2));
  FARGO_SAFE(LorentzForce(dt, B1_star, B2_star, 0, 0, 1));
#endif

}


void ComputeDivergence(Field *CompX, Field *CompY, Field *CompZ){
  int i,j,k;
  real *bx,*by,*bz,*d;
  real *sxj;
  real *sxk;
  real *syj;
  real *syk;
  real *szj;
  real *szk;

  INPUT (CompX);
  INPUT (CompY);
  INPUT (CompZ);
  OUTPUT (Divergence);

  bx = CompX->field_cpu;
  by = CompY->field_cpu;
  bz = CompZ->field_cpu;
  d = Divergence->field_cpu;

  sxj = Sxj;
  sxk = Sxk;
  syj = Syj;
  syk = Syk;
  szj = Szj;
  szk = Szk;

  i = j = k = 0;  
  for (k=0; k<Nz+2*NGHZ-1; k++) {
    for (j=0; j<Ny+2*NGHY-1; j++) {
      for (i=0; i<Nx; i++) {
	d[l] = (bx[lxp]-bx[l])*SurfX(j,k);
	d[l]+= by[lyp]*SurfY(j+1,k);
	d[l]-= by[l]  *SurfY(j,k);
	d[l]+= bz[lzp]*SurfZ(j,k+1);
	d[l]-= bz[l]  *SurfZ(j,k);
      }
    }
  }
}
