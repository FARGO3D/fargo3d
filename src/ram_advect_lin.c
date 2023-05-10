//<FLAGS>
//#define __GPU
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void AdvectRAMlin_cpu(real dt, Field *F) {

//<USER_DEFINED>
  INPUT(Slope);
  INPUT(F);
  INPUT(PhiStarmin);
  DRAFT(Pressure);
//<\USER_DEFINED>


//<EXTERNAL>
  real* f        = F->field_cpu;
  real* slopes   = Slope->field_cpu;
  real* phistarmin = PhiStarmin->field_cpu;
  real* aux   = Pressure->field_cpu;
  int pitch  = Pitch_cpu;
  int pitch2d  = Pitch2D;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
  real _xmax = XMAX;
  real _xmin = XMIN;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
  int llxp;
  int m;
  int id_star;
  int id_starp;
  int l_star;
  int l_starp;
  int umin_total;
  int idm;
  real phistarmin_last;
  real deltax;
  real deltax0;
  real deltax1;
  real deltax2;
  real deltax3;
  real deltax4;
//<\INTERNAL>


//<CONSTANT>
//  real xmin(Nx+2*NGHX+1);
//<\CONSTANT>


  
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
//<#>
      ll = l;
      llxp = lxp; 
      

      id_star  = (int) ( (phistarmin[ll]-_xmin)/(_xmax-_xmin)*size_x);   //p(i)
      id_starp = (int) ( (phistarmin[llxp]-_xmin)/(_xmax-_xmin)*size_x); //p(i+1)
      

      l_star   = id_star  + j*size_x + k*stride;
      l_starp  = id_starp + j*size_x + k*stride;
      
      umin_total  = (id_starp-id_star);
      while(umin_total < 0) umin_total += size_x;

      //idm = (int) ( umin_total/(umin_total+1e-17) );


      if(umin_total == 0){
	      deltax0  = ( phistarmin[lxp]-phistarmin[ll]);
	      deltax1  = ( phistarmin[lxp]-xmed(id_star) );
	      deltax2 = 0.0;
	      deltax3 = 0.0;
	      deltax4 = 0.0;
      }
      else{
        deltax0 = ( xmin(id_star+1)-phistarmin[ll] );
	      deltax1 = ( xmin(id_star+1)-xmed(id_star)  );
	      deltax2 = ( phistarmin[lxp]-xmin(id_starp) );
	      deltax3 = ( phistarmin[lxp]-xmed(id_starp) );
	      deltax4 = ( xmin(id_starp) -xmed(id_starp) );
      }

      deltax  = phistarmin[ll]-xmed(id_star);

      aux[ll]  = f[l_star]*deltax0; 
      aux[ll] += 0.5*slopes[l_star]*(deltax1*deltax1-deltax*deltax);

      aux[ll] += f[l_starp]*deltax2; 
      aux[ll] += 0.5*slopes[l_starp]*(deltax3*deltax3-deltax4*deltax4);

      for(m=id_star+1; m<id_starp; m++) aux[ll] += f[m+j*size_x+k*stride]*(xmin(m+1)-xmin(m));

      aux[ll] /= (xmin(i+1)-xmin(i));
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
//<LAST_BLOCK>
#ifdef __GPU
  Dev2Dev3D(F,Pressure);
#else
  memcpy(f, aux, sizeof(real)*size_x*size_y*size_z);
#endif
//<\LAST_BLOCK>
}
