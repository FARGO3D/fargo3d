#include "fargo3d.h"

void Init() {
  int i,j,k;
  real Q1,Q2;

  real *v1;
  real *v2;
  real *b1;
  real *b2;
  real *rho;
  real *cs;

  OUTPUT(Density);
  OUTPUT(Energy);
  OUTPUT(Vx);
  OUTPUT(Vy);
  OUTPUT(Vz);
  OUTPUT(Bx);
  OUTPUT(By);
  OUTPUT(Bz);

  /* Dimensional scaling of the fields */
  real b, bm, vm, rhom, em;
  vm = sqrt(G*MSTAR/R0);
  bm = MSTAR*sqrt(G*MU0)/(R0*R0);
  rhom = MSTAR/(R0*R0*R0);
  em = G*MSTAR*MSTAR/(R0*R0*R0*R0);
  /* This scaling is used in the "testdim" test */

  if(Nz == 1) {
    b1 = Bx->field_cpu;
    b2 = By->field_cpu; 
    v1 = Vx->field_cpu;
    v2 = Vy->field_cpu;
    printf("OTVORTEX XY SETUP\n");
  }
  if(Ny == 1) {
    b1 = Bx->field_cpu;
    b2 = Bz->field_cpu; 
    v1 = Vx->field_cpu;
    v2 = Vz->field_cpu;
    printf("OTVORTEX XZ SETUP\n");
  }
  if(Nx == 1) {
    b1 = By->field_cpu;
    b2 = Bz->field_cpu; 
    v1 = Vy->field_cpu;
    v2 = Vz->field_cpu;
    printf("OTVORTEX YZ SETUP\n");
  }

  rho = Density->field_cpu;
  cs  = Energy->field_cpu;
  
  i = j = k = 0;
  
#ifdef Z
  for (k=0; k<Nz+2*NGHZ; k++) {
#endif
#ifdef Y
    for (j=0; j<Ny+2*NGHY; j++) {
#endif
#ifdef X
      for (i=0; i<Nx; i++) {
#endif
	if(Nz == 1) {
	  Q1 = (Xmed(i)-XMIN)/(XMAX-XMIN);
	  Q2 = (Ymed(j)-YMIN)/(YMAX-YMIN);
	}
	if(Ny == 1) {
	  Q1 = (Xmed(i)-XMIN)/(XMAX-XMIN);
	  Q2 = (Zmed(k)-ZMIN)/(ZMAX-ZMIN);
	}
	if(Nx == 1) {
	  Q1 = (Ymed(j)-YMIN)/(YMAX-YMIN);
	  Q2 = (Zmed(k)-ZMIN)/(ZMAX-ZMIN);
	}

	b = 1.0/sqrt(4.0*M_PI)*bm;
	v1[l] = -sin(2.0*M_PI*Q2)*vm;
	if (Q2 > 0.5)	v1[l] = sin(2.0*M_PI*(1.-Q2))*vm;
	v2[l] = sin(2.0*M_PI*Q1)*vm;
	if (Q1 > 0.5)	v2[l] = -sin(2.0*M_PI*(1.-Q1))*vm;
	rho[l] = 25.0/(36.0*M_PI)*rhom;
	b1[l] = b*v1[l]/vm;
	b2[l] = b*sin(4.0*M_PI*Q1);
	if (Q1 > 0.5) b2[l] = -b*sin(4.0*M_PI*(1.-Q1));
	cs[l]=5.0/(12.0*M_PI*(GAMMA-1.0))*em;
	if (NOISE > 1e-10) {
	  v1[l] += NOISE*cs[l]*(drand48()-.5);
	  v2[l] += NOISE*cs[l]*(drand48()-.5);
	  b1[l] += NOISE*b*(drand48()-.5);
	  b2[l] += NOISE*b*(drand48()-.5);
	  rho[l] *= (1.+NOISE*(drand48()-.5));
	}
#ifdef X
      }
#endif
#ifdef Y
    }
#endif
#ifdef Z
  }
#endif
}

void CondInit() {
   Fluids[0] = CreateFluid("gas",GAS);
   SelectFluid(0);
   Init();
}
