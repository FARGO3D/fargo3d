//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void LorentzForce(real dt, Field *Bs1, Field *Bs2, int idx, int idy, int idz) {

  //Be careful with the order of bs1 and bs2!!

  int idx1, idy1, idz1;
  int idx2, idy2, idz2;
  int stride1;
  int stride2;
  
  Field* B1;
  Field* B2;
  Field* V;

  if(idx == 1) {
    V = Vx;
    B1 = By;
    B2 = Bz;
    stride1 = (Nx+2*NGHX);
    stride2 = (Nx+2*NGHX)*(Ny+2*NGHY); 
#ifdef GPU
    if ( _LorentzForce ==  _LorentzForce_gpu) {
      stride1 = Pitch_gpu;
      stride2 = Stride_gpu;
    }
#endif
    idx1 = 0; idy1 = 1; idz1 = 0;
    idx2 = 0; idy2 = 0; idz2 = 1;
  }
  
  if(idy == 1) {
    V = Vy;
    B1 = Bz;
    B2 = Bx;
    stride1 = (Nx+2*NGHX)*(Ny+2*NGHY);
    stride2 = 1;    
#ifdef GPU
    if ( _LorentzForce ==  _LorentzForce_gpu) {
      stride1 = Stride_gpu;
      stride2 = 1;
    }
#endif
    idx1 = 0; idy1 = 0; idz1 = 1;
    idx2 = 1; idy2 = 0; idz2 = 0;
  }
  
  if(idz == 1) {
    V = Vz;
    B1 = Bx;
    B2 = By;
    stride1 = 1;
    stride2 = Nx+2*NGHX;
#ifdef GPU
    if ( _LorentzForce ==  _LorentzForce_gpu) {
      stride1 = 1;
      stride2 = Pitch_gpu;
    }
#endif
    idx1 = 1; idy1 = 0; idz1 = 0;
    idx2 = 0; idy2 = 1; idz2 = 0;
  }

  FARGO_SAFE (_LorentzForce(dt, idx, idy, idz, idx1, idy1, idz1, idx2, idy2, idz2, stride1, stride2, B1, B2,V,Bs1,Bs2));

}

void _LorentzForce_cpu(real dt, int idx, int idy, int idz, int idx1, int idy1, int idz1, int idx2, int idy2, int idz2, int stride1, int stride2, Field* B1, Field* B2,Field*V, Field *Bs1,Field *Bs2) {

//<USER_DEFINED>
  INPUT (Density);
  INPUT(B1);
  INPUT(B2);
  INPUT(V);
  INPUT(Bs1);
  INPUT(Bs2);
  OUTPUT(V);
//<\USER_DEFINED>

//<EXTERNAL>
  real* b1 = B1->field_cpu;
  real* b2 = B2->field_cpu;
  real* rho = Density->field_cpu;
  real* v = V->field_cpu;
  real* bs1 = Bs1->field_cpu;
  real* bs2 = Bs2->field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = XIP;
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ-1;
  real dx = Dx;
  real nx = Nx;
//<\EXTERNAL>
    

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
  int lm;
  int lmperp1plus;
  int lmperp2plus;
  int lp1;
  int lp2;
  real b1_mean;
  real b2_mean;
  real b1_mean1;
  real b2_mean1;
  real b1_mean2;
  real b2_mean2;
  real delta1;
  real delta2;
  real d1bs1;
  real d2bs2;
//<\INTERNAL>

//<CONSTANT>
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real Sxj(Ny+2*NGHY);
// real Syj(Ny+2*NGHY);
// real Szj(Ny+2*NGHY);
// real Sxk(Nz+2*NGHZ);
// real Syk(Nz+2*NGHZ);
// real Szk(Nz+2*NGHZ);
//<\CONSTANT>


//<MAIN_LOOP>

  for (k=1; k<size_z; k++) {
    for (j=1; j<size_y; j++) {
      for (i=XIM; i<size_x; i++) {

//<#>
	ll = l;
	lm = lxm*idx+lym*idy+lzm*idz;

	lp1    = lxp*idx1 + lyp*idy1 + lzp*idz1;
	lp2    = lxp*idx2 + lyp*idy2 + lzp*idz2;
	lmperp1plus = lm+stride1;
	lmperp2plus = lm+stride2;

	delta1 = (zone_size_x(j,k)*idx1	+ \
		  zone_size_y(j,k)*idy1 + \
		  zone_size_z(j,k)*idz1);
	
	delta2 = (zone_size_x(j,k)*idx2 + \
		  zone_size_y(j,k)*idy2 + \
		  zone_size_z(j,k)*idz2);
	
	/* The test below MUST be "if (idz == 1)" and NOT "if (stride1
	   == 1)". What is intended is to flush the index to the mesh
	   in X **if the direction 1 is X**. Now we could have stride1
	   being 1 and yet the direction 1 not being X (if Nx =
	   1...) */

#ifndef GHOSTSX
	if (idz == 1) {
	  if (i == nx-1) {
	    lmperp1plus -= nx;
	  }
	}
	/* Similar considerations apply below */
	if (idy == 1) {
	  if (i == nx-1) {
	    lmperp2plus -= nx;
	  }
	}
#endif
	
	b1_mean1 = 0.5*(b1[lp1]+b1[lmperp1plus]);
	b1_mean2 = 0.5*(b1[ll]+b1[lm]);
	b1_mean  = 0.5*(b1_mean1+b1_mean2);
	b2_mean1 = 0.5*(b2[lp2]+b2[lmperp2plus]);
	b2_mean2 = 0.5*(b2[ll]+b2[lm]);
	b2_mean  = 0.5*(b2_mean1+b2_mean2);

	d1bs1 = (bs1[lp1]-bs1[ll])/delta1;
	d2bs2 = (bs2[lp2]-bs2[ll])/delta2;
	
	v[ll] += 2.0*dt/(rho[ll]+rho[lm])*(b1_mean*d1bs1 + b2_mean*d2bs2)/MU0;
//<\#>
      }
    }
  }
//<\MAIN_LOOP>
}
