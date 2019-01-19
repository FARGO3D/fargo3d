#include "fargo3d.h"

TimeProcess t_Comm;
TimeProcess t_Hydro;
TimeProcess t_Mhd;
TimeProcess t_sub1;
TimeProcess t_sub1_x;
TimeProcess t_sub1_y;
TimeProcess t_sub1_z;

void FillGhosts (int var) {

  InitSpecificTime (&t_Comm, "MPI Communications");
  FARGO_SAFE(comm (var));
  GiveSpecificTime (t_Comm);
  FARGO_SAFE(boundaries()); // Always after a comm.

#if defined(Y)
  if (NY == 1)    /* Y dimension is mute */
    CheckMuteY();
#endif
#if defined(Z)
  if (NZ == 1)    /* Z dimension is mute */
    CheckMuteZ();
#endif

}

static boolean Resistivity_Profiles_Filled = NO;

void Fill_Resistivity_Profiles () {

  OUTPUT2D(Eta_profile_xi);
  OUTPUT2D(Eta_profile_xizi);
  OUTPUT2D(Eta_profile_zi);

  int j,k;
  if (Resistivity_Profiles_Filled) return;
  real* eta_profile_xi = Eta_profile_xi->field_cpu;
  real* eta_profile_xizi = Eta_profile_xizi->field_cpu;
  real* eta_profile_zi = Eta_profile_zi->field_cpu;

  for (k=0; k<Nz+2*NGHZ; k++) {
    for (j=0; j<Ny+2*NGHY; j++) {
      eta_profile_xi[l2D] = Resistivity (Ymin(j),Zmed(k));
      eta_profile_xizi[l2D] = Resistivity (Ymin(j),Zmin(k));
      eta_profile_zi[l2D] = Resistivity (Ymed(j),Zmin(k));
    }
  }
  Resistivity_Profiles_Filled = YES;
}


void AlgoGas () {
  
  real dtemp=0.0;
  real dt=1.0;  
  int var=0;

  while(dtemp<DT) { // DT LOOP    

    SetupHook1 (); //Setup specific hook. Defaults to empty function.

#ifdef MHD
  if (Resistivity_Profiles_Filled == NO) {
    FARGO_SAFE(Fill_Resistivity_Profiles ());
  }
#endif

#ifdef ADIABATIC
    FARGO_SAFE(ComputePressureFieldAd());
#endif
    
#ifdef ISOTHERMAL
    FARGO_SAFE(ComputePressureFieldIso());
#endif

#ifdef POLYTROPIC
    FARGO_SAFE(ComputePressureFieldPoly());
#endif
    
    /// AT THIS STAGE Vx IS THE INITIAL TOTAL VELOCITY IN X
    
#ifdef X
#ifndef STANDARD
    FARGO_SAFE(ComputeVmed(Vx)); // FARGO algorithm
#endif
#endif

    /// NOW THE 2D MESH VxMed CONTAINS THE AZIMUTHAL AVERAGE OF Vx in X

    InitSpecificTime (&t_Hydro, "Eulerian Hydro (no transport) algorithms");

    /// REGARDLESS OF WHETHER WE USE FARGO, Vx IS ALWAYS THE TOTAL VELOCITY IN X
    FARGO_SAFE(cfl());
    dt = step_time; //cfl works with the 'step_time' global variable.
    /// BEFORE AND AFTER THE CALL TO CFL.
    dtemp+=dt;
    if(dtemp>DT)  dt = DT - (dtemp-dt); // updating dt


#ifdef POTENTIAL
    FARGO_SAFE(compute_potential(dt));
#endif

#if ((defined(SHEARINGSHEET2D) || defined(SHEARINGBOX3D)) && !defined(SHEARINGBC))
    FARGO_SAFE(NonReflectingBC(Vy));
#endif
    
#ifdef X
    FARGO_SAFE(SubStep1_x(dt));
#endif
#ifdef Y
    FARGO_SAFE(SubStep1_y(dt));
#endif
#ifdef Z
    FARGO_SAFE(SubStep1_z(dt));
#endif

#if (defined(VISCOSITY) || defined(ALPHAVISCOSITY))
    viscosity(dt);
#endif

#ifndef NOSUBSTEP2
    FARGO_SAFE(SubStep2_a(dt));
    FARGO_SAFE(SubStep2_b(dt));
    /// NOW: Vx INITIAL X VELOCITY, Vx_temp UPDATED X VELOCITY FROM SOURCE TERMS + ARTIFICIAL VISCOSITY
#endif
#ifdef ADIABATIC
    FARGO_SAFE(SubStep3(dt));
#endif


    GiveSpecificTime (t_Hydro);
    
#ifdef MHD //----------------------------------------------------------------
    InitSpecificTime (&t_Mhd, "MHD algorithms");
    // THIS COPIES Vx_temp INTO Vx
    FARGO_SAFE(copy_velocities(VTEMP2V));
#ifndef STANDARD // WE USE THE FARGO ALGORITHM
    FARGO_SAFE(ComputeVmed(Vx));
    FARGO_SAFE(ChangeFrame(-1, Vx, VxMed)); //Vx becomes the residual velocity
    VxIsResidual = YES;
#endif

    ComputeMHD(dt);
    
#ifndef STANDARD
    FARGO_SAFE(ChangeFrame(+1, Vx, VxMed)); //Vx becomes the total, updated velocity
    VxIsResidual = NO;
#endif
    FARGO_SAFE(copy_velocities(V2VTEMP));
    // THIS COPIES Vx INTO Vx_temp
    GiveSpecificTime (t_Mhd);
#endif  //END MHD------------------------------------------------------------
        
    InitSpecificTime (&t_Hydro, "Transport algorithms");

    // V_temp IS USED IN TRANSPORT

#if ((defined(SHEARINGSHEET2D) || defined(SHEARINGBOX3D)) && !defined(SHEARINGBC))
    FARGO_SAFE(NonReflectingBC (Vy_temp));
#endif

    FARGO_SAFE(copy_velocities(VTEMP2V));
    FARGO_SAFE(FillGhosts(PrimitiveVariables()));
    FARGO_SAFE(copy_velocities(V2VTEMP));

    
#ifdef MHD
    FARGO_SAFE(UpdateMagneticField(dt,1,0,0));
    FARGO_SAFE(UpdateMagneticField(dt,0,1,0));
    FARGO_SAFE(UpdateMagneticField(dt,0,0,1));
#endif

#if defined (MHD) && (!defined(STANDARD))
    FARGO_SAFE(MHD_fargo (dt)); // Perform additional field update with uniform velocity
#endif

#ifdef X
#ifndef STANDARD
    FARGO_SAFE(ComputeVmed(Vx_temp)); 
#endif
#endif

    transport(dt);

    GiveSpecificTime (t_Hydro);

    if(CPU_Master) {
      if (FullArrayComms)
	printf("%s", "!");
      else {
	if (ContourComms)
	  printf("%s", ":");
	else
	  printf("%s", ".");
      }
#ifndef NOFLUSH
      fflush(stdout);
#endif
    }
    if (ForwardOneStep == YES) prs_exit(EXIT_SUCCESS);
    PhysicalTime+=dt;
    FullArrayComms = 0;
    ContourComms = 0;
#ifdef MHD
    // EMFs claim ownership of their storage area
    *(Emfx->owner) = Emfx;
    *(Emfy->owner) = Emfy;
    *(Emfz->owner) = Emfz;
#endif


#ifdef STOCKHOLM
    FARGO_SAFE(StockholmBoundary(dt));
#endif

    FARGO_SAFE(FillGhosts (PrimitiveVariables()));

  }

  dtemp = 0.0;
  if(CPU_Master) printf("%s", "\n");

}
