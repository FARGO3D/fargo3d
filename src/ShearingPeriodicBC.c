#include "fargo3d.h"

static real SB_resi[2];
static int  SB_nint[2];
static real SlideVel;

// Below is a set of routines used for the shearing-periodic
// conditions. It shifts the outer and inner "radial" ghost zones,
// prior to a periodic type communication.

/* This routine is in Beta version and should not be used without
   intensive prior testing */
/* These boundary conditions only run on the CPU at the present time,
   which somehow impacts performance */

void ShearBC (int var) {
  FARGO_SAFE(ShearingPeriodicCondition()); // Determine integer and
  //fractional shifts at inner and outer boundaries
  if (var & DENS)
    FARGO_SAFE(SB_slide (Density));
  if (var & VX)
    FARGO_SAFE(SB_slide (Vx));
  if (var & VY)
    FARGO_SAFE(SB_slide (Vy));
  if (var & VZ)
    FARGO_SAFE(SB_slide (Vz));
  if (var & VXTEMP)
    FARGO_SAFE(SB_slide (Vx_temp));
  if (var & VYTEMP)
    FARGO_SAFE(SB_slide (Vy_temp));
  if (var & VZTEMP)
    FARGO_SAFE(SB_slide (Vz_temp));
#ifdef ADIABATIC
  if (var & ENERGY)
    FARGO_SAFE(SB_slide (Energy));
#endif
#ifdef MHD
  if (var & EMFX)
    FARGO_SAFE(SB_slide (Emfx));
  if (var & EMFY)
    FARGO_SAFE(SB_slide (Emfy));
  if (var & EMFZ)
    FARGO_SAFE(SB_slide (Emfz));
  if (var & BX)
    FARGO_SAFE(SB_slide (Bx));
  if (var & BY)
    FARGO_SAFE(SB_slide (By));
  if (var & BZ)
    FARGO_SAFE(SB_slide (Bz));
#endif
}

void SB_slide (Field *F) {
  FARGO_SAFE(SlideResShearingBoundary (F));
  FARGO_SAFE(SlideIntShearingBoundary (F));
}

void ShearingPeriodicCondition () {
  real shift_total, nfrac;
  int nint;
  SlideVel = 2.*(YMAX-YMIN)*OORTA;
  shift_total = SlideVel*PhysicalTime;
  nint = (int)(shift_total/(XMAX-XMIN));
  shift_total = shift_total-nint*(XMAX-XMIN);
  while (shift_total < 0.0)  shift_total += XMAX-XMIN;
  while (shift_total >= XMAX-XMIN) shift_total -= XMAX-XMIN;
  nfrac = ((real)Nx*shift_total/(XMAX-XMIN));
  SB_nint[1] = (int)(nfrac+.5); // OUTER BOUNDARY
  SB_resi[1] = nfrac - SB_nint[1];
  shift_total = -shift_total;
  while (shift_total < 0.0)  shift_total += XMAX-XMIN;
  nfrac = ((real)Nx*shift_total/(XMAX-XMIN));
  SB_nint[0] = (int)(nfrac+.5); // INNER BOUNDARY
  SB_resi[0] = nfrac - SB_nint[0];
}

void SlideIntShearingBoundary (Field *F) {
  int i,j,k,side,ii,exception_by=0;
  real *f;
  static real buffer[MAX1D];
  f = F->field_cpu;

  if (F->type == BY)
    exception_by = 1;

  for (side = 0; side < 2; side++) {
    if (((side == 0) && (J == 0)) || ((side == 1) && (J == Ncpu_x-1))) {
      for (k = 0; k < Nz+2*NGHZ; k++) {
	for (j = side*(Ny+NGHY)+side*exception_by; j < side*(Ny+NGHY)+NGHY; j++) {
	  for (i = 0; i < Nx; i++) {
	    ii = i+SB_nint[side];
	    while (ii >= Nx) ii -= Nx;
	    while (ii < 0) ii += Nx;
	    buffer[ii] = f[l];
	  }
	  for (i = 0; i < Nx; i++)
	    f[l] = buffer[i];
	}
      }
    }
  }
}

void SlideResShearingBoundary (Field *Q) {
  
  int i,j,k,side;
  real dqm, dqp, work, diff, cord, ksi;
  
  real *q;
  static real qs[MAX1D], slope[MAX1D], qL[MAX1D], qR[MAX1D], qH[MAX1D];
  boolean IsVx=NO;
  int exception_by=0;
  
  q  = Q->field_cpu;
  if (((Q->type == VX) || (Q->type == VXTEMP)) && (VxIsResidual == NO))
    IsVx = YES;

  if (Q->type == BY)
    exception_by = 1;

  i = j = k = 0;
  for (side = 0; side < 2; side++) {
    if (((side == 0) && (J == 0)) || ((side == 1) && (J == Ncpu_x-1))) {
      for (k=0; k<Nz+2*NGHZ; k++) {
	for (j=(Ny+NGHY)*side+side*exception_by; j<NGHY+(Ny+NGHY)*side; j++) { //Inner or outer radial ghost
	  for (i=0; i<Nx; i++) {	
	    dqm = (q[l]-q[lxm]);
	    dqp = (q[lxp]-q[l]);
	    if(dqp*dqm<=0.0)  slope[i] = 0.0;
	    else { // Monotonized centered slope limited
	      slope[i] = 0.5*(q[lxp]-q[lxm]);
	      work = fabs(slope[i]);
	      if (2.0*fabs(dqm) < work) work = 2.0*fabs(dqm);
	      if (2.0*fabs(dqp) < work) work = 2.0*fabs(dqp);
	      if (slope[i] < 0) slope[i] = -work;
	      else slope[i] = work;
	    }
	  }
	  
	  for (i=0; i<Nx; i++)	// Now we compute q_j+1/2
	    qH[i] = q[l]+0.5*(q[lxp]-q[l])-1.0/6.0*(slope[ixp]-slope[i]);
	  
	  for (i=0; i<Nx; i++) {  // Now we compute qRight & qLeft
	    qR[i] = qH[i];
	    qL[i] = qH[ixm];
	  }
	  
	  /* Monotonicity constraints */
	  /* -> Modify qRight & qLeft */
	  
	  for (i=0; i<Nx; i++) {
	    if ((qR[i]-q[l])*(q[l]-qL[i]) < 0.0) {
	      qL[i] = q[l];
	      qR[i] = q[l];
	    }
	    diff = qR[i] - qL[i];
	    cord = q[l] - 0.5*(qL[i]+qR[i]);
	    if (6.0*diff*cord > diff*diff)  /* don't simplify by diff !!! */
	      qL[i] = 3.0*q[l]-2.0*qR[i];
	    if (-diff*diff > 6.0*diff*cord) /* don't simplify by diff !!! */
	      qR[i] = 3.0*q[l]-2.0*qL[i];
	  }
	  
	  /* Now we've got qRight & qLeft */
	  /* Switch back to Stone & Norman paper */
	  /* for notations */
	  
	  for (i=0; i<Nx; i++) {
	    if (SB_resi[side] > 0.0) {
	      ksi = SB_resi[side];
	      qs[i] = qR[ixm]+ksi*(q[lxm]-qR[ixm]);
	      qs[i]+= ksi*(1.0-ksi)*(2.0*q[lxm]-qR[ixm]-qL[ixm]);
	    } else {
	      ksi = -SB_resi[side];
	      qs[i] = qL[i]+ksi*(q[l]-qL[i]);
	      qs[i]+= ksi*(1.0-ksi)*(2.0*q[l]-qR[i]-qL[i]);
	    }
	  }
	  
	  for (i = 0; i<Nx; i++) { // Final update
	    q[l] += (qs[i]-qs[ixp])*SB_resi[side];
	  }
	  if (IsVx == YES) {
	    for (i = 0; i < Nx; i++) {
	      q[l] += (real)(2*side-1)*2.0*OORTA*(YMAX-YMIN);
	    }
	  }
	}
      }
    }
  }
}
