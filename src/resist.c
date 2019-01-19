//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void Resist (int idx, int idy, int idz) {

  int idx1, idy1, idz1;
  int idx2, idy2, idz2;

  Field *B1;
  Field *B2;
  Field *Emf;
  Field2D *Eta;
  
  int i,j,k;

  if (idx == 1) {
    idx1 = 0; idy1 = 1; idz1 = 0;
    idx2 = 0; idy2 = 0; idz2 = 1;
    B1 = By;
    B2 = Bz;
    Emf = Emfx;
    Eta = Eta_profile_zi;
  }

  if (idy == 1) {
    idx1 = 0; idy1 = 0; idz1 = 1;
    idx2 = 1; idy2 = 0; idz2 = 0;
    B1 = Bz;
    B2 = Bx;
    Emf = Emfy;
    Eta = Eta_profile_xizi;
  }

  if (idz == 1) {
    idx1 = 1; idy1 = 0; idz1 = 0;
    idx2 = 0; idy2 = 1; idz2 = 0;
    B1 = Bx;
    B2 = By;
    Emf = Emfz;
    Eta = Eta_profile_xi;
  }
  FARGO_SAFE(_Resist(idx, idy, idz, idx1, idy1, idz1, idx2, idy2, idz2, B1, B2, Emf, Eta));
}

void _Resist_cpu (int idx, int idy, int idz, int idx1, int idy1, int idz1, int idx2, int idy2, int idz2, Field *B1, Field *B2, Field *Emf, Field2D *Eta) {

//<USER_DEFINED>
  INPUT(B1);
  INPUT(B2);
  INPUT(Emf);
  INPUT2D(Eta);
  OUTPUT(Emf);
//<\USER_DEFINED>

//<EXTERNAL>
  real* b1 = B1->field_cpu;
  real* b2 = B2->field_cpu;
  real* emf = Emf->field_cpu;
  real* eta = Eta->field_cpu;
  real dx = Dx;
  int pitch  = Pitch_cpu;
  int pitch2d = Pitch2D;
  int stride = Stride_cpu;
  int size_x = XIP; 
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ-1;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int l1m;
  int l2m;
  int ll;
  real diff1;
  real diff2;
//<\INTERNAL>

//<CONSTANT>
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
//<\CONSTANT>

//<MAIN_LOOP>
  for (k=1; k<size_z; k++) {
    for (j=1; j<size_y; j++) {
      for (i=XIM; i<size_x; i++) {
//<#>
	ll = l;
	l1m = lxm*idx1+lym*idy1+lzm*idz1;
	l2m = lxm*idx2+lym*idy2+lzm*idz2;
	/* Ideally it should not be the zone size but the distance
	   between zone centers. The different is very minute here and
	   does not matter */
	diff1 = zone_size_x(j,k)*idx1+zone_size_y(j,k)*idy1+zone_size_z(j,k)*idz1;
	diff2 = zone_size_x(j,k)*idx2+zone_size_y(j,k)*idy2+zone_size_z(j,k)*idz2;
	emf[ll] += eta[l2D]*((b2[ll]-b2[l1m])/diff1-(b1[ll]-b1[l2m])/diff2);
#ifdef CYLINDRICAL
	if (idz1+idz2 == 0) // Considering vertical component of Emf
	  emf[ll] -= eta[l2D]*(b1[ll]+b1[l2m])*.5/ymin(j); 
//Note that (phi,r,z) has a left-handed orientation.
#endif
#ifdef SPHERICAL
	if (idy1+idy2 == 0) // Considering radial component of Emf
	  emf[ll] += eta[l2D]*(b2[ll]+b2[l1m])*.5/ymed(j)*cos(zmin(k))/sin(zmin(k));
	if (idz1+idz2 == 0) // Considering colatitude component of Emf
	  emf[ll] -= eta[l2D]*(b1[ll]+b1[l2m])*.5/ymin(j);
	if (idx1+idx2 == 0) // Considering azimuthal component of Emf
	  emf[ll] += eta[l2D]*(b2[ll]+b2[l1m])*.5/ymin(j);
#endif
//<\#>
      }
    }
  }
//<\MAIN_LOOP>
}
