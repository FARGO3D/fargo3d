#include "fargo3d.h"

void Compute_Staggered_2D_fields (real dt)  {
  int j,k;
  real *vxhy, *vxhyr, *vxhz, *vxhzr, *vxmed;
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
      vxhy[l2D] = .5*(vxmed[l2D-1]+vxmed[l2D]);
      nxhy[l2D] = (int)(floor(vxhy[l2D]*dt/edge_size_x_middlez_lowy(j,k)+.5));
      vxhyr[l2D] = vxhy[l2D] - (real)nxhy[l2D]*edge_size_x_middlez_lowy(j,k)/dt;
    }
  }
  for (k = 1; k < Nz+2*NGHZ; k++) {
    for (j = 0; j < Ny+2*NGHY; j++) {
      vxhz[l2D] = .5*(vxmed[l2D-Ny-2*NGHY]+vxmed[l2D]);
      nxhz[l2D] = (int)(floor(vxhz[l2D]*dt/edge_size_x_middley_lowz(j,k)+.5));
      vxhzr[l2D] = vxhz[l2D] - (real)nxhz[l2D]*edge_size_x_middley_lowz(j,k)/dt;
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
