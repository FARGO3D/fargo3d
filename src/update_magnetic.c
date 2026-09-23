//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void UpdateMagneticField(real dt,int idx, int idy, int idz) { 
  
  int idx1,idy1,idz1,idx2,idy2,idz2; 
  Field* Emf1;
  Field* Emf2;
  Field* B;

  if(idx == 1) { 
    B = Bx; 
    Emf1 = Emfy; 
    Emf2 = Emfz; 
    idx1 = 0; idy1 = 1; idz1 = 0; 
    idx2 = 0; idy2 = 0; idz2 = 1; 
  } 
  if(idy == 1) { 
    B = By; 
    Emf1 = Emfz; 
    Emf2 = Emfx; 
    idx1 = 0; idy1 = 0; idz1 = 1; 
    idx2 = 1; idy2 = 0; idz2 = 0; 
  } 
  if(idz == 1) { 
    B = Bz; 
    Emf1 = Emfx; 
    Emf2 = Emfy; 
    idx1 = 1; idy1 = 0; idz1 = 0; 
    idx2 = 0; idy2 = 1; idz2 = 0; 
  } 
  
  FARGO_SAFE(_UpdateMagneticField(dt, idx, idy, idz,idx1,idy1,idz1,
				    idx2,idy2,idz2,B,Emf1,Emf2));

}


void _UpdateMagneticField_cpu(real dt,int idx,int idy,int idz,int idx1,int idy1,int idz1, int idx2,int idy2,int idz2, Field* B,Field* Emf1,Field* Emf2) {

//<USER_DEFINED>
  INPUT(B);
  INPUT(Emf1);
  INPUT(Emf2);
  OUTPUT(B);
//<\USER_DEFINED>

//<EXTERNAL>
  real* b    = B->field_cpu; 
  real* emf1 = Emf1->field_cpu; 
  real* emf2 = Emf2->field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = XIP; 
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ-1;
  real dx = Dx;
//<\EXTERNAL>
  
//<INTERNAL>
  int i;
  int j;
  int k;
  int lp1;
  int lp2; 
  real dp1p,dp1m,dp2p,dp2m; 
  real surf; 
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

  i = j = k = 0;

  for (k=1; k<size_z; k++) { 
    for (j=1; j<size_y; j++) { 
      for (i=XIM; i<size_x; i++) {

//<#>
	lp1 = lxp*idx1 + lyp*idy1 + lzp*idz1; 
	lp2 = lxp*idx2 + lyp*idy2 + lzp*idz2; 
	surf =SurfX(j,k)*idx; 
	surf+=SurfY(j,k)*idy; 
	surf+=SurfZ(j,k)*idz; 
	dp1m = (edge_size_x(j,k)*idx2 +
 		edge_size_y(j,k)*idy2 +
		edge_size_z(j,k)*idz2); 
	
	dp1p = (edge_size_x(j+idy1,k+idz1)*idx2 + 
		edge_size_y(j+idy1,k+idz1)*idy2 + 
		edge_size_z(j+idy1,k+idz1)*idz2); 
	
	dp2m = (edge_size_x(j,k)*idx1 + 
		edge_size_y(j,k)*idy1 + 
		edge_size_z(j,k)*idz1); 
	
	dp2p = (edge_size_x(j+idy2,k+idz2)*idx1 + 
		edge_size_y(j+idy2,k+idz2)*idy1 + 
		edge_size_z(j+idy2,k+idz2)*idz1); 
	
	b[l] += dt/surf * (emf1[lp2]*dp2p - emf1[l]*dp2m
			   - emf2[lp1]*dp1p + emf2[l]*dp1m); 	
//<\#>
      } 
    } 
  } 
//<\MAIN_LOOP>
} 
