#include "fargo3d.h"

void InitDensity() {

  int i,j,k;
  real *rho;

  rho = Density->field_cpu;
    
  real xi = SIGMASLOPE+1.+FLARINGINDEX;
  real beta = 1.-2*FLARINGINDEX; // beta = -q

  for (k=0; k<Nz+2*NGHZ; k++) {
    for (j=0; j<Ny+2*NGHY; j++) {
      real r = Ymed(j);
      real R = r*sin(Zmed(k));
      real z = Zmed(k);
      real h = ASPECTRATIO*R;
      real r3 = r*r*r;
      real R3 = R*R*R;
      real omega = sqrt(G*MSTAR/(R3));
      real H = ASPECTRATIO*pow(R/R0,FLARINGINDEX)*R;
      real cs2 = ASPECTRATIO*ASPECTRATIO*pow(R/R0,2.*FLARINGINDEX)*G*MSTAR/R; // isothermal sound speed
      for (i=NGHX; i<Nx+NGHX; i++) {
#ifdef CYLINDRICAL
        rho[l] = SIGMA0*pow(r/R0,-xi);
#else
        //rho[l] = SIGMA0*pow(R/R0,-xi)*exp(G*MSTAR*(1./r-1./R)/cs2);
        rho[l] = SIGMA0*pow(R/R0,-SIGMASLOPE)*exp(G*MSTAR*(1./r-1./R)/cs2);
#endif
      }
    }
  }
}

void InitDensityStrat() {

  int i,j,k,kk;
  real cszm;
  real *rho;
  real selfConsistent = SELFCONSISTENT;
  real zq;

  rho = Density->field_cpu;
    
  for (k=0; k<Nz+2*NGHZ; k++) {
    for (j=0; j<Ny+2*NGHY; j++) {
      real r = Ymed(j);
      real R = r*sin(Zmed(k));
      real z = Zmed(k);
      real zz = r*cos(Zmed(k));
      real r3 = r*r*r;
      real R3 = R*R*R;
      real omega = sqrt(G*MSTAR/(R3));
      real vk = sqrt(G*MSTAR/R);
      real csmid = ASPECTRATIO * pow(R/R0, FLARINGINDEX) * vk;
      real csatm = ASPECTRATIO2 * pow(R/R0, FLARINGINDEX2) * vk;
//      real cs0 = pow(pow(csmid,8.)+0.5*pow(csatm,8.)*(1.+tanh(-TEMP_ALPHA)),0.125);
      real cs0 = csmid;
//      real H = cs0/omega;
      real H = ASPECTRATIO*pow(R/R0,FLARINGINDEX)*R;
//      real sigmar = SIGMA0*pow(R/(RC/R0),-SIGMASLOPE)*exp(-pow(R/(RC/R0),2.-SIGMASLOPE)); // 2d surface density at R
      real sigmar = SIGMA0*pow(R/R0,-SIGMASLOPE); // 2d surface density at R

      for (i=NGHX; i<Nx+NGHX; i++) {
	if (i == NGHX){
// calibrate rho_mid
          real nheight = 8. ;
          int ngrid = (int)(nheight*1024.);
          real z1dm[ngrid+1];
          real z1d[ngrid];
          real csz[ngrid];
          real rhoz[ngrid];
          real rho_mid = 1.0;
//          real zq = TEMP_Z0*pow(R/R0,TEMP_BETA);
//          real zq = 3.0*H;
          if (selfConsistent == 1){
//	    real zq = 38.220155 * ASPECTRATIO * ASPECTRATIO * pow(R/R0, 1.059203) * R;
	    real zq = HEIGHT/100 * pow(R/R0, Z0SLOPE);
          } else {
            real zq = 3.0*H;
          }
	  real integ;
	  real sigma_int = 0.0;
	  real dz1d, zm;

          for (kk = 0; kk<ngrid+1; kk++){
	    z1dm[kk] = nheight*H*((real)(kk))/((real)ngrid);
	  }

	  integ = 0.0;
          for (kk = 0; kk<ngrid; kk++){
	    z1d[kk] = 0.5*(z1dm[kk]+z1dm[kk+1]);
	    dz1d = z1dm[kk+1]-z1dm[kk];
//	    csz[kk] = pow(csmid,8.) + 0.5*pow(csatm,8.)*(1.+tanh((z1d[kk]-TEMP_ALPHA*zq)/zq));
//            csz[kk] = pow(csz[kk],0.125);

            if (z1d[kk] < zq){
              csz[kk] = csatm*csatm + (csmid*csmid-csatm*csatm)*pow(cos(M_PI*z1d[kk]/2./zq),2.);
              csz[kk] = sqrt(csz[kk]);
            }else{
              csz[kk] = csatm*csatm;
              csz[kk] = sqrt(csz[kk]);
            }

	    if (kk == 0){
	      zm = 0.5*(0.0+z1d[kk]);
//	      cszm = pow(csmid,8.) + 0.5*pow(csatm,8.)*(1.+tanh((zm-TEMP_ALPHA*zq)/zq));
//              cszm = pow(cszm,0.125);
              if (zm < zq){
                cszm = csatm*csatm + (csmid*csmid-csatm*csatm)*pow(cos(M_PI*zm/2./zq),2.);
                cszm = sqrt(cszm);
              }else{
                cszm = csatm*csatm;
                cszm = sqrt(cszm);
              }
	      integ = zm/pow(R*R+zm*zm,1.5)/cszm/cszm*(z1d[kk]-0.0);
	      rhoz[kk] = rho_mid*cs0*cs0/csz[kk]/csz[kk];
	      rhoz[kk] *= exp(-integ);
	    }else{
	      zm = 0.5*(z1d[kk-1]+z1d[kk]);
//              cszm = pow(csmid,8.) + 0.5*pow(csatm,8.)*(1.+tanh((zm-TEMP_ALPHA*zq)/zq));
//              cszm = pow(cszm,0.125);
              if (zm < zq){
                cszm = csatm*csatm + (csmid*csmid-csatm*csatm)*pow(cos(M_PI*zm/2./zq),2.);
                cszm = sqrt(cszm);
              }else{
                cszm = csatm*csatm;
                cszm = sqrt(cszm);
              }
	      integ += zm/pow(R*R+zm*zm,1.5)/cszm/cszm*(z1d[kk]-z1d[kk-1]);
	      rhoz[kk] = rho_mid*cs0*cs0/csz[kk]/csz[kk];
	      rhoz[kk] *= exp(-integ); 
	    }
	      sigma_int += rhoz[kk]*dz1d;
	  }
	  rho_mid = sigmar/sigma_int;
//#ifndef HALFDISK
//          rho_mid *= 0.5;
//#endif
	  // calculate 3d density
          real nheight2 = fabs(zz)/H ; // we need fabs in case zz < 0.
          int ngrid2 = (int)(nheight2*1024.);
          real z1dm2[ngrid2+1];
          real z1d2[ngrid2];
          real csz2[ngrid2];

          for (kk = 0; kk<ngrid2+1; kk++){
            z1dm2[kk] = nheight2*H*((real)(kk))/((real)ngrid2);
          }

	  integ = 0.0;
          for (kk = 0; kk<ngrid2; kk++){
            z1d2[kk] = 0.5*(z1dm2[kk]+z1dm2[kk+1]);
            dz1d = z1dm2[kk+1]-z1dm2[kk];
//            csz2[kk] = pow(csmid,8.) + 0.5*pow(csatm,8.)*(1.+tanh((z1d2[kk]-TEMP_ALPHA*zq)/zq));
//            csz2[kk] = pow(csz2[kk],0.125);
//            cszm = pow(csmid,8.) + 0.5*pow(csatm,8.)*(1.+tanh((z1dm2[kk+1]-TEMP_ALPHA*zq)/zq));
//            cszm = pow(cszm,0.125);
            if (z1d2[kk] < zq){
              csz2[kk] = csatm*csatm + (csmid*csmid-csatm*csatm)*pow(cos(M_PI*z1d2[kk]/2./zq),2.);
              csz2[kk] = sqrt(csz2[kk]);
            }else{
              csz2[kk] = csatm*csatm;
              csz2[kk] = sqrt(csz2[kk]);
            }
            if (z1dm2[kk+1] < zq){
              cszm = csatm*csatm + (csmid*csmid-csatm*csatm)*pow(cos(M_PI*z1dm2[kk+1]/2./zq),2.);
              cszm = sqrt(cszm);
            }else{
              cszm = csatm*csatm;
              cszm = sqrt(cszm);
            }
            integ += z1d2[kk]/pow(R*R+z1d2[kk]*z1d2[kk],1.5)/csz2[kk]/csz2[kk]*(z1dm2[kk+1]-z1dm2[kk]);
          }
          rho[l] = rho_mid*cs0*cs0/cszm/cszm;
          rho[l] *= exp(-integ); 
	}else{
	  rho[l] = rho[lxm];
	}
      }
    }
  }
}


void InitSoundSpeed() {

  int i,j,k;
  real *field;
  real dr, dz;
  real r, R, z, H, r0, rho_o, t, omega, vk;
  FILE *fo;
  real *d;
  real *e;

  field = Energy->field_cpu;
  d = Density->field_cpu;

  for (k=0; k<Nz+2*NGHZ; k++) {
    for (j=0; j<Ny+2*NGHY; j++) {
      for (i=NGHX; i<Nx+NGHX; i++) {
	r = Ymed(j);
        R = r*sin(Zmed(k));
        H = ASPECTRATIO*pow(R/R0,FLARINGINDEX)*R;
	vk = sqrt(G*MSTAR/R);
	field[l] = ASPECTRATIO * pow(R/R0, FLARINGINDEX) * vk;
#ifdef ADIABATIC
	field[l] = field[l]*field[l]*d[l]/(GAMMA-1.0);
#endif
      }
    }
  }    
}

void InitSoundSpeedStrat() {

  int i,j,k;
  real *field;
  real dr, dz;
  real r, R, z, H, r0, rho_o, t, omega, vk;
  real csmid, csatm;
  FILE *fo;
  real *d;
  real *e;
  real selfConsistent = SELFCONSISTENT;
  real zq;

  field = Energy->field_cpu;
  d = Density->field_cpu;

  for (k=0; k<Nz+2*NGHZ; k++) {
    for (j=0; j<Ny+2*NGHY; j++) {
      for (i=NGHX; i<Nx+NGHX; i++) {
        r = Ymed(j);
        R = r*sin(Zmed(k));
        z = r*cos(Zmed(k));
        vk = sqrt(G*MSTAR/R);
        csmid = ASPECTRATIO * pow(R/R0, FLARINGINDEX) * vk;
        csatm = ASPECTRATIO2 * pow(R/R0, FLARINGINDEX2) * vk;
        H = ASPECTRATIO*pow(R/R0,FLARINGINDEX)*R;
        if (selfConsistent == 1){
//          real zq = 38.220155 * ASPECTRATIO * ASPECTRATIO * pow(R/R0, 1.059203) * R;
          real zq = HEIGHT/100 * pow(R/R0, Z0SLOPE);
        } else {
          real zq = 3.0*H;
        }
//	real zq = TEMP_Z0*pow(R/R0,TEMP_BETA);
//        field[l] = pow(csmid,8.) + 0.5*pow(csatm,8.)*(1.+tanh((fabs(z)-TEMP_ALPHA*zq)/zq));
//	field[l] = pow(field[l],0.125);
        if (fabs(z) < zq){
          field[l] = csatm*csatm + (csmid*csmid-csatm*csatm)*pow(cos(M_PI*z/2./zq),2.);
          field[l] = sqrt(field[l]);
        }else{
          field[l] = csatm*csatm;
          field[l] = sqrt(field[l]);
        }
#ifdef ADIABATIC
        field[l] = field[l]*field[l]*d[l]/(GAMMA-1.0);
#endif
      }
    }
  }
}


void InitVazim() {

  int i,j,k;
  real *field;
  real dr, dz;
  real r, R, z, H, r0, rho_o, t;
  real rho;
  FILE *fo;
  real vt, omega;
  real *v1;
  real *v2;
  real *v3;
  real *cs;

  v1 = Vx->field_cpu;
  v2 = Vy->field_cpu;
  v3 = Vz->field_cpu;
  cs = Energy->field_cpu;

  for (k=0; k<Nz+2*NGHZ; k++) {
    for (i=NGHX; i<Nx+NGHX; i++) {
      for (j=0; j<Ny+2*NGHY; j++) {
	r = Ymed(j);
        R = r*sin(Zmed(k));
	omega = sqrt(G*MSTAR/R/R/R);
        real xi = SIGMASLOPE+1.+FLARINGINDEX;
        real beta = 1.-2*FLARINGINDEX; // beta = -q
        real H = ASPECTRATIO*pow(R/R0,FLARINGINDEX)*R;
        real cs2 = ASPECTRATIO*ASPECTRATIO*pow(R/R0,2.*FLARINGINDEX)*G*MSTAR/R;

	v1[l] = omega*R;
	v1[l] *= sqrt((1.-beta)+(-SIGMASLOPE-beta)*H*H/R/R + beta*R/r);
	v1[l] -= OMEGAFRAME*R;

        v2[l] = v3[l] = 0.0;
        //v3[l] = 1.0e-6*sqrt(cs2)*(drand48()-.5);
      }
    }
  }    
}

void InitVazimStrat() {

  int i,j,k;
  int gj,gk;
  int pitch  = Pitch_cpu;
  real *field;
  real dr, dz;
  real r, R, z, H, r0, rho_o, t;
  FILE *fo;
  real vt, omega, omega_k;
  real *rho;
  real *v1;
  real *v2;
  real *v3;
  real *cs;

  rho = Density->field_cpu;
  v1 = Vx->field_cpu;
  v2 = Vy->field_cpu;
  v3 = Vz->field_cpu;
  cs = Energy->field_cpu;
  

  for (k=0; k<Nz+2*NGHZ; k++) {
    for (i=NGHX; i<Nx+NGHX; i++) {
      for (j=0; j<Ny+2*NGHY; j++) {
	r = Ymed(j);
        R = r*sin(Zmed(k));
	real st = sin(Zmed(k));
	omega_k = sqrt(G*MSTAR/R/R/R);
        real xi = SIGMASLOPE+1.+FLARINGINDEX;
        real beta = 1.-2*FLARINGINDEX; // beta = -q
        real cs2 = ASPECTRATIO*ASPECTRATIO*pow(R/R0,2.*FLARINGINDEX)*G*MSTAR/R;

	omega = omega_k*omega_k*st; // bae+2021

	if ((j > 0) && (j < Ny+2*NGHY-1)){
#ifdef ISOTHERMAL
	  omega += (rho[lyp]*cs[lyp]*cs[lyp]-rho[lym]*cs[lym]*cs[lym])/Ymed(j)/(Ymed(j+1)-Ymed(j-1))/rho[l]/st/st; // bae+2021
#endif
#ifdef ADIABATIC
	  omega += (GAMMA-1.)*(cs[lyp]-cs[lym])/Ymed(j)/(Ymed(j+1)-Ymed(j-1))/rho[l]/st/st;
#endif
	}
	omega = sqrt(omega);

	v1[l] = omega*R;
	v1[l] -= OMEGAFRAME*R;

        v2[l] = v3[l] = 0.0;
        v3[l] = 1.0e-6*sqrt(cs2)*(drand48()-.5);
      }
    }
  }    
}

void CondInit() {
  Fluids[0] = CreateFluid("gas",GAS);
  SelectFluid(0);

  if(STRAT){
    InitDensityStrat();
    InitSoundSpeedStrat();
    InitVazimStrat();
  }else{
    InitDensity();
    InitSoundSpeed();
    //InitVazim();
    InitVazimStrat();
  }
}

