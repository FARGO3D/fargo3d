#include "fargo3d.h"

//WARNING, ALL THESE VARIABLES MUST BE DEFINED IN DEFINE.H WITHOUT CAPITAL

void LightGlobalDev(){
#ifdef GPU
  DevMemcpyH2D(Xmin_d,Xmin,sizeof(real)*(Nx+2*NGHX+1));
  DevMemcpyH2D(Ymin_d,Ymin,sizeof(real)*(Ny+2*NGHY+1));
  DevMemcpyH2D(Zmin_d,Zmin,sizeof(real)*(Nz+2*NGHZ+1));

  DevMemcpyH2D(Sxj_d,Sxj,sizeof(real)*(Ny+2*NGHY));
  DevMemcpyH2D(Syj_d,Syj,sizeof(real)*(Ny+2*NGHY));
  DevMemcpyH2D(Szj_d,Szj,sizeof(real)*(Ny+2*NGHY));

  DevMemcpyH2D(Sxk_d,Sxk,sizeof(real)*(Nz+2*NGHZ));
  DevMemcpyH2D(Syk_d,Syk,sizeof(real)*(Nz+2*NGHZ));
  DevMemcpyH2D(Szk_d,Szk,sizeof(real)*(Nz+2*NGHZ));

  DevMemcpyH2D(InvVj_d,InvVj,sizeof(real)*(Ny+2*NGHY));
#endif
}
