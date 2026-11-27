//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>


void ComputeEmf(real dt, int idx, int idy, int idz, 
		Field *Bs1, Field *Vs1, Field *Bs2, Field *Vs2) {

  int idx1,idy1,idz1;
  int idx2,idy2,idz2;

  Field* B1;
  Field* B2;
  Field* V1;
  Field* V2;
  Field* Emf;
    
  if(idx == 1) {
    Emf = Emfx;
    B2 = By;
    B1 = Bz;
    V2 = Vy;
    V1 = Vz;
    idx1 = 0; idy1 = 0; idz1 = 1;
    idx2 = 0; idy2 = 1; idz2 = 0;
  }
  if(idy == 1) {
    Emf = Emfy;
    B2 = Bz;
    B1 = Bx;
    V2 = Vz;
    V1 = Vx;
    idx1 = 1; idy1 = 0; idz1 = 0;
    idx2 = 0; idy2 = 0; idz2 = 1;
  }
  if(idz == 1) {
    Emf = Emfz;
    B2 = Bx;
    B1 = By;
    V2 = Vx;
    V1 = Vy;
    idx1 = 0; idy1 = 1; idz1 = 0;
    idx2 = 1; idy2 = 0; idz2 = 0;
  }

  FARGO_SAFE(_ComputeEmf(dt, idx1, idy1, idz1, idx2, idy2, idz2,Bs1,Vs1,Bs2,Vs2,B1,B2,V1,V2,Emf));
}


void _ComputeEmf_cpu(real dt, int idx1, int idy1, int idz1, int idx2, int idy2, int idz2, Field *Bs1, Field *Vs1, Field *Bs2, Field *Vs2, Field* B1, Field*B2, Field* V1, Field* V2, Field* Emf) {

//<USER_DEFINED>

  /*
    Function that computes the EMF. It needs as input the star fields
    (bs1,vs1,bs2,vs2). The integers idx,idy,idz represent the direction
    in which the EMF is computed.
  */

  INPUT(B1);
  INPUT(B2);
  INPUT(V1);
  INPUT(V2);
  INPUT(Bs1);
  INPUT(Bs2);
  INPUT(Vs1);
  INPUT(Vs2);
  INPUT(Slope_v1);
  INPUT(Slope_v2);
  INPUT(Slope_b1);
  INPUT(Slope_b2);
  OUTPUT(Emf);
//<\USER_DEFINED>

//<EXTERNAL>
  real* slope_v1 = Slope_v1->field_cpu;
  real* slope_b1 = Slope_b1->field_cpu;
  real* slope_v2 = Slope_v2->field_cpu;
  real* slope_b2 = Slope_b2->field_cpu;
  real* b1 = B1->field_cpu;
  real* v1 = V1->field_cpu;
  real* b2 = B2->field_cpu;
  real* v2 = V2->field_cpu;
  real* bs1 = Bs1->field_cpu;
  real* bs2 = Bs2->field_cpu;
  real* vs1 = Vs1->field_cpu;
  real* vs2 = Vs2->field_cpu;
  real* emf = Emf->field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int l1;
  int l2;
  int ll;
  real b1_mean;
  real b2_mean;
  real v1_mean;
  real v2_mean;
  real delta1;
  real delta2;
  real delta1_m;
  real delta2_m;
  real dq1_c;
  real dq1_m;
  real dq2_c;
  real dq2_m;
  real w1_c;
  real w1_m;
  real w2_c;
  real w2_m;
  real v1_mean_old;
  real v2_mean_old;
//<\INTERNAL>
  
//<CONSTANT>
// real xmin(Nx+2*NGHX+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real Sxi(Nx+2*NGHX);
//<\CONSTANT>

//<MAIN_LOOP>

  for (k=1; k<size_z; k++) {
    for (j=1; j<size_y; j++) {
      for (i=XIM; i<size_x; i++) {

//<#>
	ll = l;
	l1 = lxm*idx1+lym*idy1+lzm*idz1;
	l2 = lxm*idx2+lym*idy2+lzm*idz2;
	
	delta1 = (zone_size_x(i,j,k)*idx1 +
		  zone_size_y(j,k)*idy1 +
		  zone_size_z(j,k)*idz1);
	
	delta2 = (zone_size_x(i,j,k)*idx2 +
		  zone_size_y(j,k)*idy2 +
		  zone_size_z(j,k)*idz2);

	delta1_m = (zone_size_x(idx1 ? ixm : i,j-idy1,k-idz1)*idx1 +
		    zone_size_y(j-idy1,k-idz1)*idy1 +
		    zone_size_z(j-idy1,k-idz1)*idz1);

	delta2_m = (zone_size_x(idx2 ? ixm : i,j-idy2,k-idz2)*idx2 +
		    zone_size_y(j-idy2,k-idz2)*idy2 +
		    zone_size_z(j-idy2,k-idz2)*idz2);

	dq1_c = Sxi(i)*idx1 + (ymin(j+1)-ymin(j))*idy1 + (zmin(k+1)-zmin(k))*idz1;
	dq1_m = Sxi(ixm)*idx1 + (ymin(j)-ymin(j-1))*idy1 + (zmin(k)-zmin(k-1))*idz1;
	w1_c = dq1_m / (dq1_c + dq1_m);
	w1_m = dq1_c / (dq1_c + dq1_m);

	dq2_c = Sxi(i)*idx2 + (ymin(j+1)-ymin(j))*idy2 + (zmin(k+1)-zmin(k))*idz2;
	dq2_m = Sxi(ixm)*idx2 + (ymin(j)-ymin(j-1))*idy2 + (zmin(k)-zmin(k-1))*idz2;
	w2_c = dq2_m / (dq2_c + dq2_m);
	w2_m = dq2_c / (dq2_c + dq2_m);
	
	v1_mean_old = w2_c*v1[ll] + w2_m*v1[l2];
	v2_mean_old = w1_c*v2[ll] + w1_m*v2[l1];
	
	if(v2_mean_old<0.0){
	  v1_mean = v1[ll]-.5*slope_v2[ll]*(delta2+v2_mean_old*dt);
	  b1_mean = b1[ll]-.5*slope_b2[ll]*(delta2+v2_mean_old*dt);
	}
	else{
	  v1_mean = v1[l2]+.5*slope_v2[l2]*(delta2_m-v2_mean_old*dt);
	  b1_mean = b1[l2]+.5*slope_b2[l2]*(delta2_m-v2_mean_old*dt);
	}
#ifdef STRICTSYM
	if (fabs(v2_mean_old) < SMALLVEL) {
	  v1_mean = w2_c*v1[ll] + w2_m*v1[l2];
	  b1_mean = w2_c*b1[ll] + w2_m*b1[l2];
	}
#endif
	if(v1_mean_old<0.0){
	  v2_mean = v2[ll]-.5*slope_v1[ll]*(delta1+v1_mean_old*dt);
	  b2_mean = b2[ll]-.5*slope_b1[ll]*(delta1+v1_mean_old*dt);
	}
	else{
	  v2_mean = v2[l1]+.5*slope_v1[l1]*(delta1_m-v1_mean_old*dt);
	  b2_mean = b2[l1]+.5*slope_b1[l1]*(delta1_m-v1_mean_old*dt);
	}
#ifdef STRICTSYM
	if (fabs(v1_mean_old) < SMALLVEL) {
	  v2_mean = w1_c*v2[ll] + w1_m*v2[l1];
	  b2_mean = w1_c*b2[ll] + w1_m*b2[l1];
	}
#endif
	emf[ll] = 0.5*(vs1[ll]*b2_mean + bs2[ll]*v1_mean -
		       vs2[ll]*b1_mean - bs1[ll]*v2_mean);
//<\#>
      }
    }
  }
//<\MAIN_LOOP>
}
