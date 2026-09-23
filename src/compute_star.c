//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void ComputeStar(real dt, int idx1, int idy1, int idz1,			 
		 int idx2, int idy2, int idz2, int index,
		 Field* Bs, Field* Vs, Field* Slope_b,	
		 Field* Slope_v, Field* Slope_bvl, Field* Slope_vvl) {
  
  /* This function is an intermediate stage to the wrapper function _ComputeStar */
  
  Field* B1;
  Field* B2;
  Field* V1;
  Field* V2;
  int stride1;
  int stride2;

  if(idx1 == 1) {
    B1  = Bx;
    V1  = Vx;
    stride1 = 1;
  }
  if(idy1 == 1) {
    B1  = By;
    V1  = Vy;
#ifdef GPU
    if (_ComputeStar == _ComputeStar_gpu)
      stride1 = Pitch_gpu;    
    else
#endif
      stride1 = Nx+2*NGHX;    
  }
  if(idz1 == 1) {
    B1  = Bz;
    V1  = Vz;
#ifdef GPU
    if (_ComputeStar == _ComputeStar_gpu)
      stride1 = Stride_gpu;
    else
#endif
      stride1 = (Nx+2*NGHX)*(Ny+2*NGHY);
  }
  /* Now index of field */
  if(idx2 == 1) {
    B2  = Bx;
    V2  = Vx;
    stride2 = 1;
  }
  if(idy2 == 1) {
    B2  = By;
    V2  = Vy;
#ifdef GPU
    if (_ComputeStar == _ComputeStar_gpu)
      stride2 = Pitch_gpu;
    else
#endif
      stride2 = Nx+2*NGHX;
  }
  if(idz2 == 1) { //------------------------------
    B2  = Bz;
    V2  = Vz;
#ifdef GPU
    if (_ComputeStar == _ComputeStar_gpu)
      stride2 = Stride_gpu;
    else
#endif
      stride2 = (Nx+2*NGHX)*(Ny+2*NGHY);
  } 
  FARGO_SAFE(_ComputeStar(dt, idx1,idy1,idz1,idx2,idy2,idz2,index,stride1,stride2,B1,B2,V1,V2,Bs,Vs,Slope_b,Slope_v,Slope_bvl,Slope_vvl));
}


void _ComputeStar_cpu(real dt, int idx1, int idy1, int idz1, int idx2, int idy2, int idz2, int index, int stride1, int stride2, Field* B1, Field* B2, Field* V1, Field* V2, Field* Bs, Field* Vs, Field* Slope_b, Field* Slope_v, Field* Slope_bvl, Field* Slope_vvl) {
  
//<USER_DEFINED>

  /* This function computes the star fields for the Method of
     Characteristics (magnetic field and velocity). The slopes of each
     field are needed as input.  Index=0 implies a calculation in a
     frame at rest, whereas Index=1 implies a calculation in a
     comobile frame (with velocity v1). The integers idx1,idy1 and
     idz1 are for the direction of propagation of the characteristics,
     and the integers idx2,idy2 and idz2 are for the direction of the
     field of which we want the star value
  */
  
  INPUT(B1);
  INPUT(B2);
  INPUT(V1);
  INPUT(V2);
  INPUT(Density);
  INPUT(Slope_b);
  INPUT(Slope_v);
  INPUT(Slope_bvl);
  INPUT(Slope_vvl);

  OUTPUT(Bs);
  OUTPUT(Vs);

//<\USER_DEFINED>

//<EXTERNAL>
  real* rho = Density->field_cpu;
  real* b1  = B1->field_cpu;
  real* b2  = B2->field_cpu;
  real* v1  = V1->field_cpu;
  real* v2  = V2->field_cpu;
  real* bs  = Bs->field_cpu;
  real* vs  = Vs->field_cpu;
  real* slope_b   = Slope_b->field_cpu;	
  real* slope_v   = Slope_v->field_cpu;
  real* slope_bvl = Slope_bvl->field_cpu;
  real* slope_vvl = Slope_vvl->field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = XIP;
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ-1;
  real dx = Dx;
  int nx = Nx+2*NGHX;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  real cp;
  real cm;
  real rho_mean;
  real rho_mean1;
  real rho_mean2;
  real bp;
  real bm;
  real vp;
  real vm;
  real delta1;
  real delta2;
  int lpropm;
  int lperpm;
  int lmperpm;
  int ll;
  real v1_mean;
  real v2_mean;
  real b1_mean;
  real temp;
  //  real b2_mean;
//<\INTERNAL>

  /* Propagation index first */

//<CONSTANT>
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
//<\CONSTANT>

//<MAIN_LOOP>

  for (k=2; k<size_z; k++) {
    for (j=2; j<size_y; j++) {
      for (i=XIM; i<size_x; i++) {
//<#>

	ll = l;

	lpropm  = lxm*idx1 + lym*idy1 + lzm*idz1; //Propagation index minus
	lperpm  = lxm*idx2 + lym*idy2 + lzm*idz2; //Minus along vectors	
	lmperpm = lpropm - stride2;  //direction! (l plus perpendicular minus)

#ifndef GHOSTSX
	if(idx2 == 1) {
	  if(i == 0)
	    lmperpm = lpropm + nx -1; //passes permutation test
	}
#endif

	rho_mean1 = 0.5*(rho[lperpm]+rho[lmperpm]);
	rho_mean2 = 0.5*(rho[ll]+rho[lpropm]);
	rho_mean  = 0.5*(rho_mean1+rho_mean2);

	delta1 = (zone_size_x(j,k)*idx1 +
		  zone_size_y(j,k)*idy1 +
		  zone_size_z(j,k)*idz1);

	delta2 = (zone_size_x(j,k)*idx2 +
		  zone_size_y(j,k)*idy2 +
		  zone_size_z(j,k)*idz2);
	
	v2_mean = 0.5*(v2[ll] + v2[lpropm]);
	
	if(v2_mean>0.0){	/* van Leer upwind estimate below */
	  temp = (delta2-v2_mean*dt);
	  v1_mean = v1[lperpm]+.5*slope_vvl[lperpm]*temp;
	  b1_mean = b1[lperpm]+.5*slope_bvl[lperpm]*temp;
	}
	else{
	  temp = (delta2+v2_mean*dt);
	  v1_mean = v1[ll]-0.5*slope_vvl[ll]*temp;
	  b1_mean = b1[ll]-0.5*slope_bvl[ll]*temp;
	}
#ifdef STRICTSYM
	if (fabs(v2_mean)*dt/delta2 < 1e-9) {
	  v1_mean = 0.5*(v1[ll]+v1[lperpm]);
	  b1_mean = 0.5*(b1[ll]+b1[lperpm]);
	}
#endif

	temp = b1_mean/sqrt(MU0*rho_mean);
	cp = v1_mean*(real)index + temp;
	cm = v1_mean*(real)index - temp;
	
	
	if(cp>0.0) {
	  temp = 0.5*(delta1-cp*dt);
	  bp = b2[lpropm] + temp*slope_b[lpropm];
	  vp = v2[lpropm] + temp*slope_v[lpropm];
	}
	else {
	  temp = 0.5*(delta1+cp*dt);
	  bp = b2[ll] - temp*slope_b[ll];
	  vp = v2[ll] - temp*slope_v[ll];
	}
#ifdef STRICTSYM
	if (fabs(cp)*dt/delta2 < 1e-9) {
	  bp = .5*(b2[lpropm]+b2[ll]);
	  vp = .5*(v2[lpropm]+v2[ll]);
	}
#endif
	if(cm>0.0) {
	  temp = 0.5*(delta1-cm*dt);
	  bm = b2[lpropm] + temp*slope_b[lpropm];
	  vm = v2[lpropm] + temp*slope_v[lpropm];
	}
	else {
	  temp = 0.5*(delta1+cm*dt);
	  bm = b2[ll] - temp*slope_b[ll];
	  vm = v2[ll] - temp*slope_v[ll];
	}
#ifdef STRICTSYM
	if (fabs(cm)*dt/delta2 < 1e-9) {
	  bm = .5*(b2[lpropm]+b2[ll]);
	  vm = .5*(v2[lpropm]+v2[ll]);
	}
#endif	
	temp = sqrt(MU0*rho_mean);
	bs[ll] = 0.5 * ((bp+bm) - (vp-vm)*temp);
	vs[ll] = 0.5 * ((vp+vm) - (bp-bm)/temp);
//<\#>
      }
    }
  }
//<\MAIN_LOOP>
}
