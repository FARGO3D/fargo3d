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
  OUTPUT(Pressure);// we use the pressure field for temporal storage.
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
#ifdef __GPU
  real* alpha = Alpha_d;
#else
  real* alpha = Alpha;
#endif
  real dx    = Dx;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
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
#endif
#ifdef Y
  real cyp;
  real cym;
  real d3;
  real d4;
  int llyp;
  int llym;
#endif
#ifdef Z
  real czp;
  real czm;
  real d5;
  real d6;
  int llzp;
  int llzm;
#endif
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
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
      for (i=0; i<size_x; i++ ) {
#endif
//<#>
#ifdef X
	ll = l;
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
        d1   = 0.25*(rhod[ll] + rhog[ll] + rhod[llxp] + rhog[llxp])*(sdiff_yczc[llxp]+sdiff_yczc[ll]); //face centered in X
        d2   = 0.25*(rhod[ll] + rhog[ll] + rhod[llxm] + rhog[llxm])*(sdiff_yczc[llxm]+sdiff_yczc[ll]); //face centered in X
	cxp  = rhod[llxp]/(rhod[llxp] + rhog[llxp]);                                                   //Cell centered
        cxm  = rhod[llxm]/(rhod[llxm] + rhog[llxm]);                                                   //Cell centered
	
#ifdef CARTESIAN
	update += 1.0/(dx)*(d1*(cxp-c)/(dx) - (d2*(c-cxm))/(dx));
#endif
	
#ifdef CYLINDRICAL
        update += 1.0/ymed(j)/ymed(j)/(dx)*(d1*(cxp-c)/(dx) - (d2*(c-cxm))/(dx));
#endif
	
#ifdef SPHERICAL
        update += 1.0/ymed(j)/ymed(j)/sin(zmed(k))/sin(zmed(k))/(dx)*(d1*(cxp-c)/(dx) - (d2*(c-cxm))/(dx));
#endif
#endif //X
	
	// DUST DIFFUSION ALONG Y-DIRECTION
#ifdef Y
	d3   = 0.5*(rhod[ll] + rhog[ll] + rhod[llyp] + rhog[llyp])*sdiff_yfzc[llyp];//face centered in Y
	d4   = 0.5*(rhod[ll] + rhog[ll] + rhod[llym] + rhog[llym])*sdiff_yfzc[ll];  //face centered in Y
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
	d5   = 0.5*(rhod[ll] + rhog[ll] + rhod[llzp] + rhog[llzp])*sdiff_yczf[llzp]; // face centered in Z
	d6   = 0.5*(rhod[ll] + rhog[ll] + rhod[llzm] + rhog[llzm])*sdiff_yczf[ll];   // face centered in Z
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
