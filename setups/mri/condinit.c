#include "fargo3d.h"

void Init() {
  int i,j,k;
  int gj,gk;
  real *v1;
  real *v2;
  real *v3;
  real *b1, *b2, *b3;
  real *e;
  real *rho, *pot;
  real h;
  real v1_tmp, v2_tmp, v3_tmp, b1_tmp, b2_tmp, b3_tmp, e_tmp, rho_tmp, pot_tmp;

  real omega,bphi,cutoff,buffer,cs;
  real r, r3, denslocal;
  
  rho = Density->field_cpu;
  e   = Energy->field_cpu;
  v1  = Vx->field_cpu;
  v2  = Vy->field_cpu;
  v3  = Vz->field_cpu;
  b1  = Bx->field_cpu;
  b2  = By->field_cpu;
  b3  = Bz->field_cpu;
  pot = Pot->field_cpu;

#if defined(CARTESIAN) || defined(SPHERICAL)
  mastererr("MRI presently defined only for unstratified cylindrical setups\n");
  prs_exit (1);
#endif
#ifndef MHD
  mastererr ("Cannot run an MRI setup if MHD is not defined...\n");
  prs_exit (1);
#endif
  
  buffer = (YMAX-YMIN)/14.;
  
  srand48 (ArrayNb); //Same seed for all processes

  /* The following initialization loop is a bit tricky. Instead of the
     standard procedure that consists in each PE initializing its own
     patch (only), here each PE performs a global loop over the whole
     mesh (hence the indices gj, gk). The aim is that each of them
     follows the exact same sequence of random numbers, so that the
     noise initialized in the density field be independent on the
     number of processors (which allows to test that the code output
     is independent of the number of processors). The values are
     stored in _tmp variables, and used only if they fall in the
     current PE.*/

  for (gk=0; gk<NZ+2*NGHZ; gk++) {
    for (gj=0; gj<NY+2*NGHY; gj++) {
      j = gj-y0cell;
      k = gk-z0cell;
      if ((j >= 0) && (j < Ny+2*NGHY))
	r = Ymed(j);
      else
	r = Ymed(0);
      h = ASPECTRATIO*r;
      r3 = r*r*r;
      omega = sqrt(G*MSTAR/(r3));
      denslocal = SIGMA0/(ZMAX-ZMIN)*pow(r/R0,-SIGMASLOPE);
      bphi = sqrt(MU0)*h*omega*sqrt(2.*denslocal)/sqrt(BETA);
      cs = ASPECTRATIO*omega*r;
      cutoff = 1.0;
      if (r < YMIN+buffer)
	cutoff = pow((r-YMIN)/buffer,0.5);
      if (r > YMAX-buffer)
	cutoff = pow((YMAX-r)/buffer,0.5);
      for (i=NGHX; i<Nx+NGHX; i++) {
	v2_tmp = (drand48()-.5)*2.*0.01*NOISE*cs;
	v3_tmp = (drand48()-.5)*2.*0.01*NOISE*cs;
	v1_tmp = r*omega;
	rho_tmp = denslocal;
#ifdef ISOTHERMAL
	e_tmp = cs;
#else
	e_tmp = rho_tmp*h*h*v1_tmp*v1_tmp/(r*r)/(GAMMA-1.0);
#endif
	v1_tmp *= sqrt(1.-ASPECTRATIO*ASPECTRATIO);
	v1_tmp -= OMEGAFRAME*r;
	b2_tmp = b3_tmp = 0.0;

	//no cutoff near radial boundaries
	cutoff = 1.0;
	
	b1_tmp = bphi*cutoff;
	pot_tmp = -G*MSTAR/r; //Cylindrical, unstratified potential (r is cylindrical radius)
	if ((j >= 0) && (j < Ny+2*NGHY) && (k >= 0) && (k < Nz+2*NGHZ)) {
	  rho[l] = rho_tmp;
	  e[l]   = e_tmp;
	  v1[l]  = v1_tmp;
	  v2[l]  = v2_tmp;
	  v3[l]  = v3_tmp;
	  b1[l]  = b1_tmp;
	  b2[l]  = b2_tmp;
	  b3[l]  = b3_tmp;
	  pot[l] = pot_tmp;
	}
      }
    }
  }
}

void CondInit() {
   Fluids[0] = CreateFluid("gas",GAS);
   SelectFluid(0);
   Init();
}
