//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void mon_power_cpu () {

  //<USER_DEFINED>
  INPUT(Density);
  OUTPUT(Slope); 

  // 使用当前全局变量 Xplanet, Yplanet, Zplanet (由 monitor.c 在循环中更新)
  real rplanet = sqrt(Xplanet*Xplanet + Yplanet*Yplanet + Zplanet*Zplanet);
  real rsmoothing = THICKNESSSMOOTHING * ASPECTRATIO * pow(rplanet/R0, FLARINGINDEX) * rplanet;
  real rsm2 = rsmoothing * rsmoothing;
  //<\USER_DEFINED>

  //<EXTERNAL>
  real* dens = Density->field_cpu;
  real* interm = Slope->field_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
  //<\EXTERNAL>

  //<INTERNAL>
  int i, j, k, ll;
  real dx, dy, dz = 0.0;
  real dist2, InvDist3, cellmass;
  real fxi, fyi, fzi = 0.0;
  //<\INTERNAL>

  //<MAIN_LOOP>
  i = j = k = 0;
#ifdef Z
  for (k=0; k<size_z; k++) {
#endif
#ifdef Y
    for (j=0; j<size_y; j++) {
#endif
#ifdef X
      for (i=0; i<size_x; i++ ) {
#endif
        ll = l; 
        cellmass = Vol(i,j,k) * dens[ll];

        // 坐标计算（与 mon_torque.c 一致）
#ifdef CARTESIAN
        dx = xmed(i) - Xplanet;
        dy = ymed(j) - Yplanet;
#ifdef Z
        dz = zmed(k) - Zplanet;
#endif
#endif
#ifdef CYLINDRICAL
        dx = ymed(j) * cos(xmed(i)) - Xplanet;
        dy = ymed(j) * sin(xmed(i)) - Yplanet;
#ifdef Z
        dz = zmed(k) - Zplanet;
#endif
#endif
#ifdef SPHERICAL
        dx = ymed(j) * cos(xmed(i)) * sin(zmed(k)) - Xplanet;
        dy = ymed(j) * sin(xmed(i)) * sin(zmed(k)) - Yplanet;
#ifdef Z
        dz = ymed(j) * cos(zmed(k)) - Zplanet;
#endif
#endif

        dist2 = dx*dx + dy*dy + dz*dz + rsm2;
        InvDist3 = G * cellmass / (dist2 * sqrt(dist2));

        fxi = dx * InvDist3;
        fyi = dy * InvDist3;
#ifdef Z
        fzi = dz * InvDist3;
#endif

        // 能量变化率：力与行星速度的点积
        // VXplanet, VYplanet, VZplanet 也会由 monitor.c 自动更新为当前行星的速度
        interm[ll] = fxi * VXplanet + fyi * VYplanet + fzi * VZplanet;

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
