#include "fargo3d.h"

void Compute_Staggered_2D_fields (real dt)  {
  int i=0,j,k;
  real *vxhy, *vxhyr, *vxhz, *vxhzr, *vxmed;
  real dy_c, dy_m, dz_c, dz_m;
  int *nxhy, *nxhz;
  vxhy  = Vxhy->field_cpu;
  vxhz  = Vxhz->field_cpu;
  vxhyr = Vxhyr->field_cpu;
  vxhzr = Vxhzr->field_cpu;
  nxhy  = Nxhy->field_cpu;
  nxhz  = Nxhz->field_cpu;
  vxmed = VxMed->field_cpu;

  INPUT2D  (VxMed);
  OUTPUT2D (Vxhy);
  OUTPUT2D (Vxhz);
  OUTPUT2D (Vxhyr);
  OUTPUT2D (Vxhzr);
  OUTPUT2DINT (Nxhy);
  OUTPUT2DINT (Nxhz);
  
  for (k = 0; k < Nz+2*NGHZ; k++) {
    for (j = 1; j < Ny+2*NGHY; j++) {
      dy_c = Ymin(j+1) - Ymin(j);
      dy_m = Ymin(j) - Ymin(j-1);
      vxhy[l2D] = (dy_m*vxmed[l2D] + dy_c*vxmed[l2D-1])/(dy_c + dy_m);
      nxhy[l2D] = (int)(floor(vxhy[l2D]*dt/edge_size_x_middlez_lowy(i,j,k)+.5));
      vxhyr[l2D] = vxhy[l2D] - (real)nxhy[l2D]*edge_size_x_middlez_lowy(i,j,k)/dt;
    }
  }
  for (k = 1; k < Nz+2*NGHZ; k++) {
    for (j = 0; j < Ny+2*NGHY; j++) {
      dz_c = Zmin(k+1) - Zmin(k);
      dz_m = Zmin(k) - Zmin(k-1);
      vxhz[l2D] = (dz_m*vxmed[l2D] + dz_c*vxmed[l2D-Ny-2*NGHY])/(dz_c + dz_m);
      nxhz[l2D] = (int)(floor(vxhz[l2D]*dt/edge_size_x_middley_lowz(i,j,k)+.5));
      vxhzr[l2D] = vxhz[l2D] - (real)nxhz[l2D]*edge_size_x_middley_lowz(i,j,k)/dt;
    }
  }
}

void MHD_fargo (real dt) {
  Compute_Staggered_2D_fields (dt);

  FARGO_SAFE(VanLeerX_PPA_2D(By, Emfz, Vxhyr, dt));
  FARGO_SAFE(VanLeerX_PPA_2D(Bz, Emfy, Vxhzr, dt));

  FARGO_SAFE(EMF_Upstream_Integrate(dt));

  FARGO_SAFE(AdvectSHIFT (Emfz, Nxhy));
  FARGO_SAFE(AdvectSHIFT (Emfy, Nxhz));

  FARGO_SAFE(UpdateMagneticField (1.0, 1,0,0));
  FARGO_SAFE(UpdateMagneticField (1.0, 0,1,0));
  FARGO_SAFE(UpdateMagneticField (1.0, 0,0,1));

}
