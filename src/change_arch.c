#include "fargo3d.h"

void ChangeArch() {
  FILE *func_arch;
  char separator[20] = "\t :=>";

  char s[MAXLINELENGTH];
  char name[MAXNAMELENGTH];
  char strval[MAXNAMELENGTH];
  char *s1;
  int success;
  int i;

  func_arch = fopen(FUNCARCHFILE, "r");
  
  if(func_arch == NULL) {
    printf("Error!! %s cannot be opened.\n", FUNCARCHFILE);
  }

  //Function pointers assignment (Default values ==> _cpu)
  //----------------------------------------------------
  ComputePressureFieldIso = ComputePressureFieldIso_cpu;
  ComputePressureFieldAd = ComputePressureFieldAd_cpu;
  ComputePressureFieldPoly = ComputePressureFieldPoly_cpu;
  SubStep1_x  = SubStep1_x_cpu;
  SubStep1_y  = SubStep1_y_cpu;
  SubStep1_z  = SubStep1_z_cpu;
  SubStep2_a  = SubStep2_a_cpu;
  SubStep2_b  = SubStep2_b_cpu;
  SubStep3    = SubStep3_cpu;
  DivideByRho = DivideByRho_cpu;
  VanLeerX_a  = VanLeerX_a_cpu;
  VanLeerX_b  = VanLeerX_b_cpu;
  VanLeerY_a  = VanLeerY_a_cpu;
  VanLeerY_b  = VanLeerY_b_cpu;
  VanLeerZ_a  = VanLeerZ_a_cpu;
  VanLeerZ_b  = VanLeerZ_b_cpu;
  momenta_x   = momenta_x_cpu;
  momenta_y   = momenta_y_cpu;
  momenta_z   = momenta_z_cpu;
  UpdateX     = UpdateX_cpu;
  UpdateY     = UpdateY_cpu;
  UpdateZ     = UpdateZ_cpu;
  UpdateDensityX = UpdateDensityX_cpu;
  UpdateDensityY = UpdateDensityY_cpu;
  UpdateDensityZ = UpdateDensityZ_cpu;
  NewVelocity_x = NewVelocity_x_cpu;
  NewVelocity_y = NewVelocity_y_cpu;
  NewVelocity_z = NewVelocity_z_cpu;
  AdvectSHIFT   = AdvectSHIFT_cpu;
  reduction_SUM = reduction_SUM_cpu;
  reduction_MIN = reduction_MIN_cpu;
  ComputeResidual = ComputeResidual_cpu;
  ChangeFrame = ChangeFrame_cpu;
  Potential   = Potential_cpu;
  CorrectVtheta = CorrectVtheta_cpu;
  cfl = cfl_cpu;
  copy_velocities = copy_velocities_cpu;
  _ComputeForce = _ComputeForce_cpu;
  StockholmBoundary = StockholmBoundary_cpu;
  
  visctensor_cart   = visctensor_cart_cpu;
  addviscosity_cart = addviscosity_cart_cpu;
  visctensor_cyl    = visctensor_cyl_cpu;
  addviscosity_cyl  = addviscosity_cyl_cpu;
  visctensor_sph    = visctensor_sph_cpu;
  addviscosity_sph  = addviscosity_sph_cpu;

  #include <../scripts/bound_cpu.code>

  Fill_GhostsX =  Fill_GhostsX_cpu;

  mon_dens = mon_dens_cpu;
  mon_momx = mon_momx_cpu;
  mon_momy = mon_momy_cpu;
  mon_momz = mon_momz_cpu;
  mon_torq = mon_torq_cpu;
  mon_reynolds = mon_reynolds_cpu;
  mon_maxwell  = mon_maxwell_cpu;
  mon_bxflux   = mon_bxflux_cpu;

  comm = comm_cpu;

  CheckMuteY = CheckMuteY_cpu;
  CheckMuteZ = CheckMuteZ_cpu;

  SetupHook1 = SetupHook1_cpu;

  //DUST DIFFUSION---------------------------------------
  DustDiffusion_Core         = DustDiffusion_Core_cpu;
  DustDiffusion_Coefficients = DustDiffusion_Coefficients_cpu;
  //-----------------------------------------------------

  copy_field = copy_field_cpu;

  //MHD------------------------------------------------
  ComputeSlopes = ComputeSlopes_cpu;
  _ComputeStar = _ComputeStar_cpu;
  _ComputeEmf  = _ComputeEmf_cpu;
  _UpdateMagneticField  = _UpdateMagneticField_cpu;
  _LorentzForce = _LorentzForce_cpu;
  _Resist = _Resist_cpu;
  EMF_Upstream_Integrate = EMF_Upstream_Integrate_cpu;
  //----------------------------------------------------
  _collisions = _collisions_cpu;
  ComputeTotalDensity = ComputeTotalDensity_cpu;
  Floor = Floor_cpu; 
  Reset_field = Reset_field_cpu; 
  //-----------------------------------------------------


  VanLeerX_PPA_a    = VanLeerX_PPA_a_cpu;
  VanLeerX_PPA_b    = VanLeerX_PPA_b_cpu;
  VanLeerX_PPA_steep= VanLeerX_PPA_steep_cpu;
  VanLeerX_PPA_c    = VanLeerX_PPA_c_cpu;
  VanLeerX_PPA_d    = VanLeerX_PPA_d_cpu;
  VanLeerX_PPA_d_2d = VanLeerX_PPA_d_2d_cpu;

  while (fgets(s, MAXLINELENGTH-1, func_arch) != NULL) {
    success = sscanf(s, "%s", name);
    if(name[0]!='#' && success == 1){
      s1 = s + (int)strlen(name);
      sscanf(s1 + strspn(s1,separator), "%s", strval);
      for (i = 0; i<strlen(name); i++) {
	name[i] = (char)tolower(name[i]);
      }
      for (i = 0; i<strlen(strval); i++){
	strval[i] = (char)tolower(strval[i]);
      }
      
#ifdef GPU
      if (EverythingOnCPU == YES) {
	fclose (func_arch);
	return;
      }
      if (strcmp(name, "computepressurefieldiso") == 0) {
	if(strval[0] == 'g') {
	  ComputePressureFieldIso = ComputePressureFieldIso_gpu;
	  printf("CompPressFieldIso runs on the GPU\n");
	}
      }
      if (strcmp(name, "computepressurefieldad") == 0) {
	if(strval[0] == 'g') {
	  ComputePressureFieldAd = ComputePressureFieldAd_gpu;
	  printf("CompPressFieldAd runs on the GPU\n");
	}
      }
      if (strcmp(name, "computepressurefieldpoly") == 0) {
	if(strval[0] == 'g') {
	  ComputePressureFieldPoly = ComputePressureFieldPoly_gpu;
	  printf("CompPressFieldPoly runs on the GPU\n");
	}
      }
      if (strcmp(name, "substep1") == 0) {
	if(strval[0] == 'g') {
	  SubStep1_x = SubStep1_x_gpu;
	  SubStep1_y = SubStep1_y_gpu; 
	  SubStep1_z = SubStep1_z_gpu;
	  printf("Substep1 runs on the GPU\n");
	}
      }
      if (strcmp(name, "substep2") == 0) {
	if(strval[0] == 'g'){
	  SubStep2_a = SubStep2_a_gpu;
	  SubStep2_b = SubStep2_b_gpu;
	  printf("Substep2 runs on the GPU\n");
	}
      }
      if (strcmp(name, "substep3") == 0) {
	if(strval[0] == 'g'){
	  SubStep3 = SubStep3_gpu;
	  printf("Substep3 runs on the GPU\n");
	}
      }
      if (strcmp(name, "dividebyrho") == 0) {
	if(strval[0] == 'g'){
	  DivideByRho = DivideByRho_gpu;
	  printf("DivideByRho runs on the GPU\n");
	}
      }
      if (strcmp(name, "resetfield") == 0) {
	if(strval[0] == 'g'){
	  Reset_field = Reset_field_gpu;
	  printf("resetfield on the GPU\n");
	}
      }

      if (strcmp(name, "vanleer") == 0) {
	if(strval[0] == 'g'){
	  VanLeerX_a = VanLeerX_a_gpu;
	  VanLeerX_b = VanLeerX_b_gpu;
	  VanLeerY_a = VanLeerY_a_gpu;
	  VanLeerY_b = VanLeerY_b_gpu;
	  VanLeerZ_a = VanLeerZ_a_gpu;
	  VanLeerZ_b = VanLeerZ_b_gpu;
	  printf("Vanleer runs on the GPU\n");
	}
      }
      if (strcmp(name, "momenta") == 0) {
	if(strval[0] == 'g'){
	  momenta_x = momenta_x_gpu;
	  momenta_y = momenta_y_gpu;
	  momenta_z = momenta_z_gpu;
	  printf("momenta runs on the GPU\n");
	}
      }
      if (strcmp(name, "update") == 0) {
	if(strval[0] == 'g'){
	  UpdateX = UpdateX_gpu;
	  UpdateY = UpdateY_gpu;
	  UpdateZ = UpdateZ_gpu;
	  printf("update runs on the GPU\n");
	}
      }
      if (strcmp(name, "updatedensity") == 0) {
	if(strval[0] == 'g'){
	  UpdateDensityX = UpdateDensityX_gpu;
	  UpdateDensityY = UpdateDensityY_gpu;
	  UpdateDensityZ = UpdateDensityZ_gpu;
	  printf("updatedensity runs on the GPU\n");
	}
      }
      if (strcmp(name, "newvelocity") == 0) {
	if(strval[0] == 'g'){
	  NewVelocity_x = NewVelocity_x_gpu;
	  NewVelocity_y = NewVelocity_y_gpu;
	  NewVelocity_z = NewVelocity_z_gpu;
	  printf("NewVelocity runs on the GPU\n");
	}
      }
      if (strcmp(name, "copyvelocities") == 0) {
	if(strval[0] == 'g'){
	  copy_velocities = copy_velocities_gpu;
	  printf("Copy velocities runs on the GPU\n");
	}
      }
      if (strcmp(name, "copyfield") == 0) {
	if(strval[0] == 'g'){
	  copy_field = copy_field_gpu;
	  printf("Copy field runs on the GPU\n");
	}
      }
      if (strcmp(name, "reduction") == 0) {
	if(strval[0] == 'g'){
	  reduction_SUM = reduction_SUM_gpu;
	  reduction_MIN = reduction_MIN_gpu;
	  printf("Reduction runs on the GPU\n");
	}
      }
      if (strcmp(name, "advectshift") == 0) {
	if(strval[0] == 'g'){
	  AdvectSHIFT = AdvectSHIFT_gpu;
	  printf("AdvectShift runs on the GPU\n");
	}
      }
      if (strcmp(name, "computeresidual") == 0) {
	if(strval[0] == 'g'){
	  ComputeResidual = ComputeResidual_gpu;
	  printf("ComputeResidual runs on the GPU\n");
	}
      }
      if (strcmp(name, "changeframe") == 0) {
	if(strval[0] == 'g'){
	  ChangeFrame = ChangeFrame_gpu;
	  printf("ChangeFrame runs on the GPU\n");
	}
      }
      if (strcmp(name, "potential") == 0) {
	if(strval[0] == 'g'){
	  Potential = Potential_gpu;
	  printf("Potential runs on the GPU\n");
	}
      }
      if (strcmp(name, "correctvtheta") == 0) {
	if(strval[0] == 'g'){
	  CorrectVtheta = CorrectVtheta_gpu;
	  printf("CorrectVtheta runs on the GPU\n");
	}
      }
      if (strcmp(name, "cfl") == 0) {
	if(strval[0] == 'g'){
	  cfl = cfl_gpu;
	  printf("cfl runs on the GPU\n");
	}
      }
      if (strcmp(name, "computeforce") == 0) {
	if(strval[0] == 'g'){
	  _ComputeForce = _ComputeForce_gpu;
	  printf("ComputeForce runs on the GPU\n");
	}
      }
      if (strcmp(name, "computeslopes") == 0) {
	if(strval[0] == 'g'){
	  ComputeSlopes = ComputeSlopes_gpu;
	  printf("ComputeSlopes runs on the GPU\n");
	}
      }
      if (strcmp(name, "computestar") == 0) {
	if(strval[0] == 'g'){
	  _ComputeStar = _ComputeStar_gpu;
	  printf("ComputeStar runs on the GPU\n");
	}
      }
      if (strcmp(name, "computeemf") == 0) {
	if(strval[0] == 'g'){
	  _ComputeEmf  = _ComputeEmf_gpu;
	  printf("ComputeEmf runs on the GPU\n");
	}
      }
      if (strcmp(name, "updatemagneticfield") == 0) {
	if(strval[0] == 'g'){
	  _UpdateMagneticField  = _UpdateMagneticField_gpu;
	  printf("UpdateMagneticField runs on the GPU\n");
	}
      }
      if (strcmp(name, "lorentzforce") == 0) {
	if(strval[0] == 'g'){
	  _LorentzForce = _LorentzForce_gpu;
	  printf("LorentzForce runs on the GPU\n");
	}
      }
      if (strcmp(name, "vanleerppa") == 0) {
	if(strval[0] == 'g'){
	  VanLeerX_PPA_a    = VanLeerX_PPA_a_gpu;
	  VanLeerX_PPA_b    = VanLeerX_PPA_b_gpu;
	  VanLeerX_PPA_steep= VanLeerX_PPA_steep_gpu;
	  VanLeerX_PPA_c    = VanLeerX_PPA_c_gpu;
	  VanLeerX_PPA_d    = VanLeerX_PPA_d_gpu;
	  VanLeerX_PPA_d_2d = VanLeerX_PPA_d_2d_gpu;
	  printf("VanLeerPPA runs on the GPU\n");
	}
      }
      if (strcmp(name, "resistivity") == 0) {
	if(strval[0] == 'g'){
	  _Resist = _Resist_gpu;
	  printf("Resistivity runs on the GPU\n");
	}
      }
      if (strcmp(name, "fargomhd") == 0) {
	if(strval[0] == 'g'){
	  EMF_Upstream_Integrate = EMF_Upstream_Integrate_gpu;
	  printf("FargoMHD runs on the GPU\n");
	}
      }
      if (strcmp(name, "stockholmboundary") == 0) {
	if(strval[0] == 'g'){
	  StockholmBoundary = StockholmBoundary_gpu;
	  printf("Stockholm Boundary runs on the GPU\n");
	}
      }
      if (strcmp(name, "viscoustensor") == 0) {
	if(strval[0] == 'g'){
	  visctensor_cart = visctensor_cart_gpu;
	  visctensor_cyl = visctensor_cyl_gpu;
	  visctensor_sph = visctensor_sph_gpu;
	  printf("Viscous tensor is computed on the GPU\n");
	}
      }
      if (strcmp(name, "addviscosity") == 0) {
	if(strval[0] == 'g'){
	  addviscosity_cart = addviscosity_cart_gpu;
	  addviscosity_cyl = addviscosity_cyl_gpu;
	  addviscosity_sph = addviscosity_sph_gpu;
	  printf("addviscosity runs on the GPU\n");
	}
      }
      if (strcmp(name, "monitor") == 0) {
	if(strval[0] == 'g'){
	  mon_dens = mon_dens_gpu;
	  mon_momx = mon_momx_gpu;
	  mon_momy = mon_momy_gpu;
	  mon_momz = mon_momz_gpu;
	  mon_torq = mon_torq_gpu;
	  mon_reynolds = mon_reynolds_gpu;
	  mon_maxwell  = mon_maxwell_gpu;
	  mon_bxflux   = mon_bxflux_gpu;
	  printf("Monitoring runs on the GPU\n");
	}
      }
      if (strcmp(name, "dustdiffusion") == 0) {
	if(strval[0] == 'g'){
	  DustDiffusion_Core         = DustDiffusion_Core_gpu;
	  DustDiffusion_Coefficients = DustDiffusion_Coefficients_gpu;
	  printf("Dust diffusion runs on the GPU\n");
	}
      }
      if (strcmp(name, "communications") == 0) {
	if(strval[0] == 'g'){
#ifdef MPICUDA
	  comm = comm_gpu;
	  printf("Communications are done directy on GPU\n");
#else
	  printf("Warning: your version of MPI does not allow direct GPU-GPU communications.\n");
	  printf("Communications are done through the hosts (CPUs), and may not be very efficient.\n");
#endif
	}
      }
      
      if (strcmp(name, "boundaries") == 0) {
	if(strval[0] == 'g'){
	  #include <../scripts/bound_gpu.code>
	  printf("boundaries runs on the GPU\n");
	}
      }

      if (strcmp(name, "fillghostsx") == 0) {
	if(strval[0] == 'g'){
          Fill_GhostsX =  Fill_GhostsX_gpu;
	  printf("Fill_GhostsX runs on the GPU\n");
	}
      }

      if (strcmp(name, "checkmute") == 0) {
	if(strval[0] == 'g'){
	  CheckMuteY = CheckMuteY_gpu;
	  CheckMuteZ = CheckMuteZ_gpu;
	  printf("CheckMute runs on the GPU\n");
	}
      }


      if (strcmp(name, "setuphook") == 0) {
	if(strval[0] == 'g'){
	  SetupHook1 = SetupHook1_gpu;
	  printf("SETUP hook runs on the GPU\n");
	}
      }

      if (strcmp(name, "collisions") == 0) {
	if(strval[0] == 'g'){
	  _collisions = _collisions_gpu;
	  printf("collisions runs on the GPU\n");
	}
      }

if (strcmp(name, "computetotaldensity") == 0) {
	if(strval[0] == 'g'){
	  ComputeTotalDensity = ComputeTotalDensity_gpu;
	  printf("ComputeTotalDensity runs on the GPU\n");
	}
      }
      if (strcmp(name, "resetfield") == 0) {
	if(strval[0] == 'g'){
	  Reset_field = Reset_field_gpu;
	  printf("Reset_field runs on the GPU\n");
	}
      }

      if (strcmp(name, "floor") == 0) {
	if(strval[0] == 'g'){
	  Floor = Floor_gpu;
	  printf("Floor runs on the GPU\n");
	}
      }
#endif
    }
  }
  fclose(func_arch);
}
