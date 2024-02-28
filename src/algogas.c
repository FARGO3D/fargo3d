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

#if YDIM
  if (NY == 1)    /* Y dimension is mute */
    CheckMuteY();
#endif
#if ZDIM
  if (NZ == 1)    /* Z dimension is mute */
    CheckMuteZ();
#endif

}


void Sources(real dt) {

  SetupHook1 (); //Setup specific hook. Defaults to empty function.

  //Equations of state-----------------------------------------------------------
#if ADIABATIC
  FARGO_SAFE(ComputePressureFieldAd());
#endif
#if ISOTHERMAL
  FARGO_SAFE(ComputePressureFieldIso());
#endif
#if POLYTROPIC
  FARGO_SAFE(ComputePressureFieldPoly());
#endif
  //-----------------------------------------------------------------------------

  InitSpecificTime (&t_Hydro, "Eulerian Hydro (no transport) algorithms");

  // REGARDLESS OF WHETHER WE USE FARGO, Vx IS ALWAYS THE TOTAL VELOCITY IN X

#if POTENTIAL
  FARGO_SAFE(compute_potential(dt));
  if (Corotating) {
    FARGO_SAFE(CorrectVtheta(Domega));
  }
#endif

#if ((SHEARINGSHEET2D || SHEARINGBOX3D) && (!SHEARINGBC))
  FARGO_SAFE(NonReflectingBC(Vy));
#endif

#if XDIM
  FARGO_SAFE(SubStep1_x(dt));
#endif
#if YDIM
  FARGO_SAFE(SubStep1_y(dt));
#endif
#if ZDIM
  FARGO_SAFE(SubStep1_z(dt));
#endif

#if (VISCOSITY || ALPHAVISCOSITY)
  if (Fluidtype == GAS) viscosity(dt);
#endif

#if (!NOSUBSTEP2)
  FARGO_SAFE(SubStep2_a(dt));
  FARGO_SAFE(SubStep2_b(dt));
#endif

  // NOW: Vx INITIAL X VELOCITY, Vx_temp UPDATED X VELOCITY FROM SOURCE TERMS + ARTIFICIAL VISCOSITY

#if ADIABATIC
 if(Fluidtype == GAS) FARGO_SAFE(SubStep3(dt));
#endif

  GiveSpecificTime (t_Hydro);

#if MHD //-------------------------------------------------------------------
  if(Fluidtype == GAS){
    InitSpecificTime (&t_Mhd, "MHD algorithms");
    FARGO_SAFE(copy_velocities(VTEMP2V));
#if (!STANDARD) // WE USE THE FARGO ALGORITHM
    FARGO_SAFE(ComputeVmed(Vx));
    FARGO_SAFE(ChangeFrame(-1, Vx, VxMed)); //Vx becomes the residual velocity
    VxIsResidual = YES;
#endif

    ComputeMHD(dt);

#if (!STANDARD)
    FARGO_SAFE(ChangeFrame(+1, Vx, VxMed)); //Vx becomes the total, updated velocity
    VxIsResidual = NO;
#endif //STANDARD
    FARGO_SAFE(copy_velocities(V2VTEMP));
    // THIS COPIES Vx INTO Vx_temp
    GiveSpecificTime (t_Mhd);
  }
#endif //END MHD----------------------------------------------------------------

  InitSpecificTime (&t_Hydro, "Transport algorithms");

#if ((SHEARINGSHEET2D || SHEARINGBOX3D) && (!SHEARINGBC))
  FARGO_SAFE(NonReflectingBC (Vy_temp));
#endif

  FARGO_SAFE(copy_velocities(VTEMP2V));
  FARGO_SAFE(FillGhosts(PrimitiveVariables()));
  FARGO_SAFE(copy_velocities(V2VTEMP));

#if MHD //-------------------------------------------------------------------
  if(Fluidtype == GAS){ //We do MHD only for the gaseous component

    FARGO_SAFE(UpdateMagneticField(dt,1,0,0));
    FARGO_SAFE(UpdateMagneticField(dt,0,1,0));
    FARGO_SAFE(UpdateMagneticField(dt,0,0,1));

#if (!STANDARD)
    FARGO_SAFE(MHD_fargo (dt)); // Perform additional field update with uniform velocity
#endif

  }
#endif //END MHD ---------------------------------------------------------------
}

void Transport(real dt) {

  //NOTE: V_temp IS USED IN TRANSPORT

#if XDIM
#if (!STANDARD)
  FARGO_SAFE(ComputeVmed(Vx_temp));
#endif
#endif

  transport(dt);

  GiveSpecificTime (t_Hydro);

  if (ForwardOneStep == YES) prs_exit(EXIT_SUCCESS);

#if MHD
  if(Fluidtype == GAS) {   // We do MHD only for the gaseous component
   *(Emfx->owner) = Emfx;  // EMFs claim ownership of their storage area
   *(Emfy->owner) = Emfy;
   *(Emfz->owner) = Emfz;
 }
#endif

}
