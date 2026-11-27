//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void DustDiffusion_Core_cpu(real dt) {
  
//<USER_DEFINED>
  INPUT(Sdiffyczc);
  INPUT(Sdiffyfzc);
#ifdef Z
  INPUT(Sdiffyczf);
  INPUT(Sdiffyfzf);
#endif
  INPUT(Density);
  INPUT(Fluids[0]->Density);
  OUTPUT(Pressure);// we use the pressure field for temporal storage
//<\USER_DEFINED>

  //Arrays Mmx, Mpx, Mmy and Mpy were filled with the dust diffusion coefficients in DustDiffusion_Coefficients()
  
//<EXTERNAL>
  real* sdiff_yfzc = Sdiffyfzc->field_cpu;
  real* sdiff_yczc = Sdiffyczc->field_cpu;
#ifdef Z
  real* sdiff_yczf = Sdiffyczf->field_cpu;
  real* sdiff_yfzf = Sdiffyfzf->field_cpu;
#endif
  real* rhod  = Density->field_cpu;
  real* rhog  = Fluids[0]->Density->field_cpu;
  real* temp  = Pressure->field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = XIP;
  int size_y = Ny+2*NGHY-1;
  int size_z = Nz+2*NGHZ-1;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
  real c;
  real update;
#ifdef X
  real d1;
  real d2;
  real cxp;
  real cxm;
  int llxm;
  int llxp;
  real dxmedm;
  real dxmedp;
  real dxmin;
  real dx_c;
  real dx_p;
  real dx_m;
  real wx_p_r;
  real wx_p_l;
  real wx_m_r;
  real wx_m_l;
#endif
#ifdef Y
  real cyp;
  real cym;
  real d3;
  real d4;
  int llyp;
  int llym;
  real dy_c;
  real dy_p;
  real dy_m;
  real wy_p_r;
  real wy_p_l;
  real wy_m_r;
  real wy_m_l;
#endif
#ifdef Z
  real czp;
  real czm;
  real d5;
  real d6;
  int llzp;
  int llzm;
  real dz_c;
  real dz_p;
  real dz_m;
  real wz_p_r;
  real wz_p_l;
  real wz_m_r;
  real wz_m_l;
#endif
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+2*NGHX+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real InvDiffXmed(Nx+2*NGHX);
// real Sxi(Nx+2*NGHX);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;

#ifdef Z
  for (k=1; k<size_z; k++) {
#endif
#ifdef Y
    for (j=1; j<size_y; j++) {
#endif
#ifdef X
      for (i=XIM; i<size_x; i++ ) {
#endif
//<#>
	ll = l;
#ifdef X
	llxm = lxm;
	llxp = lxp;
#endif
#ifdef Y
	llyp = lyp;
	llym = lym;
#endif
#ifdef Z
	llzp = lzp;
	llzm = lzm;
#endif

	update = 0.0;
	c    = rhod[ll]/(rhod[ll] + rhog[ll]); //Cell centered
	
	// DUST DIFFUSION ALONG X-DIRECTION
#ifdef X
	dxmin  = Sxi(i);
	dxmedp = 1.0/InvDiffXmed(ixp);
	dxmedm = 1.0/InvDiffXmed(i);
	dx_c = Sxi(i);
	dx_p = Sxi(ixp);
	dx_m = Sxi(ixm);
	wx_p_r = dx_c/(dx_p + dx_c);
	wx_p_l = dx_p/(dx_p + dx_c);
	wx_m_r = dx_m/(dx_c + dx_m);
	wx_m_l = dx_c/(dx_c + dx_m);
        d1   = (wx_p_r*(rhod[llxp]+rhog[llxp]) + wx_p_l*(rhod[ll]+rhog[ll]))*(wx_p_r*sdiff_yczc[llxp] + wx_p_l*sdiff_yczc[ll]); //face centered in X
        d2   = (wx_m_r*(rhod[ll]+rhog[ll]) + wx_m_l*(rhod[llxm]+rhog[llxm]))*(wx_m_r*sdiff_yczc[ll] + wx_m_l*sdiff_yczc[llxm]); //face centered in X
	cxp  = rhod[llxp]/(rhod[llxp] + rhog[llxp]);                                                   //Cell centered
        cxm  = rhod[llxm]/(rhod[llxm] + rhog[llxm]);                                                   //Cell centered
	
#ifdef CARTESIAN
	update += 1.0/(dxmin)*(d1*(cxp-c)/(dxmedp) - (d2*(c-cxm))/(dxmedm));
#endif
	
#ifdef CYLINDRICAL
        update += 1.0/ymed(j)/ymed(j)/(dxmin)*(d1*(cxp-c)/(dxmedp) - (d2*(c-cxm))/(dxmedm));
#endif
	
#ifdef SPHERICAL
        update += 1.0/ymed(j)/ymed(j)/sin(zmed(k))/sin(zmed(k))/(dxmin)*(d1*(cxp-c)/(dxmedp) - (d2*(c-cxm))/(dxmedm));
#endif
#endif //X
	
	// DUST DIFFUSION ALONG Y-DIRECTION
#ifdef Y
	dy_c = ymin(j+1) - ymin(j);
	dy_p = ymin(j+2) - ymin(j+1);
	dy_m = ymin(j) - ymin(j-1);
	wy_p_r = dy_c/(dy_p + dy_c);
	wy_p_l = dy_p/(dy_p + dy_c);
	wy_m_r = dy_m/(dy_c + dy_m);
	wy_m_l = dy_c/(dy_c + dy_m);
	d3   = (wy_p_r*(rhod[llyp]+rhog[llyp]) + wy_p_l*(rhod[ll]+rhog[ll]))*sdiff_yfzc[llyp];//face centered in Y
	d4   = (wy_m_r*(rhod[ll]+rhog[ll]) + wy_m_l*(rhod[llym]+rhog[llym]))*sdiff_yfzc[ll];  //face centered in Y
	cyp  = rhod[llyp]/(rhod[llyp] + rhog[llyp]);                                //Cell centered
	cym  = rhod[llym]/(rhod[llym] + rhog[llym]);                                //Cell centered
	
#ifdef CARTESIAN
	update += 1.0/(ymin(j+1)-ymin(j))*(d3*(cyp-c)/(ymed(j+1)-ymed(j)) - (d4*(c-cym))/(ymed(j)-ymed(j-1)));
#endif
	
#ifdef CYLINDRICAL
        update += 1.0/ymed(j)/(ymin(j+1)-ymin(j))*(ymin(j+1)*d3*(cyp-c)/(ymed(j+1)-ymed(j)) -
						   (ymin(j)*d4*(c-cym))/(ymed(j)-ymed(j-1)));
#endif
	
#ifdef SPHERICAL
	update += 1.0/ymed(j)/ymed(j)/(ymin(j+1)-ymin(j))*(ymin(j+1)*ymin(j+1)*d3*(cyp-c)/(ymed(j+1)-ymed(j)) -
							   (ymin(j)*ymin(j)*d4*(c-cym))/(ymed(j)-ymed(j-1)));
#endif
#endif //Y

        // DUST DIFFUSION ALONG Z-DIRECTION
#ifdef Z
	dz_c = zmin(k+1) - zmin(k);
	dz_p = zmin(k+2) - zmin(k+1);
	dz_m = zmin(k) - zmin(k-1);
	wz_p_r = dz_c/(dz_p + dz_c);
	wz_p_l = dz_p/(dz_p + dz_c);
	wz_m_r = dz_m/(dz_c + dz_m);
	wz_m_l = dz_c/(dz_c + dz_m);
	d5   = (wz_p_r*(rhod[llzp]+rhog[llzp]) + wz_p_l*(rhod[ll]+rhog[ll]))*sdiff_yczf[llzp]; // face centered in Z
	d6   = (wz_m_r*(rhod[ll]+rhog[ll]) + wz_m_l*(rhod[llzm]+rhog[llzm]))*sdiff_yczf[ll];   // face centered in Z
	czp  = rhod[llzp]/(rhod[llzp] + rhog[llzp]);                                 // Cell centered
	czm  = rhod[llzm]/(rhod[llzm] + rhog[llzm]);                                 // Cell centered
#ifdef CARTESIAN
	update += 1.0/(zmin(k+1)-zmin(k))*(d5*(czp-c)/(zmed(k+1)-zmed(k)) - (d6*(c-czm))/(zmed(k)-zmed(k-1)));
#endif
#ifdef CYLINDRICAL
        update += 1.0/(zmin(k+1)-zmin(k))*(d5*(czp-c)/(zmed(k+1)-zmed(k)) - (d6*(c-czm))/(zmed(k)-zmed(k-1)));
#endif
#ifdef SPHERICAL
	update += 1.0/ymed(j)/ymed(j)/sin(zmed(k))/(zmin(k+1)-zmin(k))*(sin(zmin(k+1))*d5*(czp-c)/(zmed(k+1)-zmed(k)) -
									(sin(zmin(k))*d6*(c-czm))/(zmed(k)-zmed(k-1)));
#endif
#endif // Z
      	temp[ll] = rhod[ll] + dt*update; // Density update
//<\#>
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
