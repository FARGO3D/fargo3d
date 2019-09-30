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
void Sources(real dt) {
     
  SetupHook1 (); //Setup specific hook. Defaults to empty function.
  
  //Equations of state-----------------------------------------------------------
#ifdef ADIABATIC
  FARGO_SAFE(ComputePressureFieldAd());
#endif
#ifdef ISOTHERMAL
  FARGO_SAFE(ComputePressureFieldIso());
#endif
#ifdef POLYTROPIC
  FARGO_SAFE(ComputePressureFieldPoly());
#endif
  //-----------------------------------------------------------------------------
    
  InitSpecificTime (&t_Hydro, "Eulerian Hydro (no transport) algorithms");
  
  // REGARDLESS OF WHETHER WE USE FARGO, Vx IS ALWAYS THE TOTAL VELOCITY IN X
  
#ifdef POTENTIAL
  FARGO_SAFE(compute_potential(dt));
  if (Corotating) {
    FARGO_SAFE(CorrectVtheta(Domega));
  }
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
  if (Fluidtype == GAS) viscosity(dt);
#endif
  
#ifndef NOSUBSTEP2
  FARGO_SAFE(SubStep2_a(dt));
  FARGO_SAFE(SubStep2_b(dt));
#endif

  // NOW: Vx INITIAL X VELOCITY, Vx_temp UPDATED X VELOCITY FROM SOURCE TERMS + ARTIFICIAL VISCOSITY

#ifdef ADIABATIC
  FARGO_SAFE(SubStep3(dt));
#endif
    
  GiveSpecificTime (t_Hydro);
  
#ifdef MHD //-------------------------------------------------------------------
  if(Fluidtype == GAS){
    InitSpecificTime (&t_Mhd, "MHD algorithms");
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
#endif //STANDARD
    FARGO_SAFE(copy_velocities(V2VTEMP));
    // THIS COPIES Vx INTO Vx_temp
    GiveSpecificTime (t_Mhd);
  }
#endif //END MHD----------------------------------------------------------------

  InitSpecificTime (&t_Hydro, "Transport algorithms");

#if ((defined(SHEARINGSHEET2D) || defined(SHEARINGBOX3D)) && !defined(SHEARINGBC))
  FARGO_SAFE(NonReflectingBC (Vy_temp));
#endif
  
  FARGO_SAFE(copy_velocities(VTEMP2V));
  FARGO_SAFE(FillGhosts(PrimitiveVariables()));
  FARGO_SAFE(copy_velocities(V2VTEMP));

#ifdef MHD //-------------------------------------------------------------------
  if(Fluidtype == GAS){ //We do MHD only for the gaseous component
    
    FARGO_SAFE(UpdateMagneticField(dt,1,0,0));
    FARGO_SAFE(UpdateMagneticField(dt,0,1,0));
    FARGO_SAFE(UpdateMagneticField(dt,0,0,1));

#if !defined(STANDARD)
    FARGO_SAFE(MHD_fargo (dt)); // Perform additional field update with uniform velocity
#endif

  } 
#endif //END MHD ---------------------------------------------------------------
}

void Transport(real dt) {

  //NOTE: V_temp IS USED IN TRANSPORT

#ifdef X
#ifndef STANDARD
  FARGO_SAFE(ComputeVmed(Vx_temp)); 
#endif
#endif

  transport(dt);
  
  GiveSpecificTime (t_Hydro);
  
  if (ForwardOneStep == YES) prs_exit(EXIT_SUCCESS);
  
#ifdef MHD
  if(Fluidtype == GAS) {   // We do MHD only for the gaseous component
   *(Emfx->owner) = Emfx;  // EMFs claim ownership of their storage area
   *(Emfy->owner) = Emfy;
   *(Emfz->owner) = Emfz;
 }
#endif

}
