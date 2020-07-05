#ifdef __GPU
#define ex extern "C" 
#else
#define ex extern
#endif

//matplotlib.c prototypes

void pyrun(const char *, ...);
void finalize_python(void);
int check_simtype(void);
void initialize_python(void);
void plot1d(char*, int, int);
void plot2d(char*, int, int);
void plot3d(char*, int, int);

//param.c Prototypes

ex void Explode(void);

ex void init_var(char*, char*, int, int, char*);
ex void ReadVarFile(char*);
ex void var_assign(void);
ex void ListVariables(char*);
ex void ListVariablesIDL(char*);

ex void rescale (void);

ex void thermal_diff_a (void);
ex void thermal_diff_b (real);
ex void energy_release (real);
ex void energy_release_b (real, real, int, int, int, int, int);

ex void ParseRedefinedOptions (char *);
ex void OptionError (void);
ex void ReadRedefined (void);
ex void DumpToFargo3drc (int, char **);

ex void SolveOrbits (PlanetarySystem *);
ex void FindOrbitalElements (StateVector,real,int);
ex OrbitalElements SV2OE (StateVector, real);


//var.c Prototypes
ex void InitVariables(void);
ex void Init(void);

//LowTask.c Prototypes
ex int PrimitiveVariables (void);
ex void prs_exit(int);
ex void prs_error(char*);
ex void MakeDir(char*);
ex void EarlyDeviceSelection (void) ;
ex FILE *fopen_prs (char*,char*);
ex FILE *master_fopen (char*, char*);
ex void masterprint(const char*, ...);
ex void masterfprintf(FILE*,const char*, ...);
ex void mastererr(const char*, ...);
ex Field *CreateField(char*, int, boolean, boolean, boolean);
ex Field *CreateFieldAlias(char*, Field*, int);
ex Field2D *CreateField2D(char*,int);
ex FieldInt2D *CreateFieldInt2D(char*);
ex void SelectDevice(int);
ex void ChangeArch(void);

ex void StretchOutput (int);
ex void RestartStretch (Field*,int);

ex real Swap(real f);

ex void RestartVTK(Field*,int);

ex void InitMonitoring3D (int);


ex Point ComputeAccel(real,real,real,real,real);

ex void CreateBuffers(void);
ex void InitSpace(void);
ex void InitSurfaces(void);
ex void CreateFields(void);
ex real ComputeMass(void);
ex void SaveState(void);
ex void SaveStateSecondary(void);
ex void RestoreState(void);
ex void Check_CUDA_Blocks_Consistency (void);

ex void copy_velocities_cpu (int);
ex void copy_velocities_gpu (int);

ex real ComputeMagneticFlux_X (void);
ex real ComputeMagneticFlux_Y (void);
ex real ComputeMagneticFlux_Z (void);

//mpi_dummy.c Prototypes are present in mpi_dummy.h

//resetfields prototypes
ex void Reset_field_cpu(Field*);

//RungeKutta.c Prototypes
ex void DerivMotionRK5(real*, real*, real*, int, real, boolean*);
ex void TranslatePlanetRK5 (real*, real, real, real, real, real, real*, int);
ex void RungeKutta (real*, real, real*, real*, int, boolean*);
ex void AdvanceSystemRK5(real);

//algogas.c Prototypes
ex void FillGhosts (int);
ex void Sources(real);
ex void Transport(real);
ex void SetupHook1_cpu (void);

//boundary.c Prototypes
#include "../scripts/bound_proto.code"

ex void Fill_GhostsX_cpu(void);
ex void Fill_GhostsX_erad_cpu(void);
ex void boundaries(void);
ex void NonReflectingBC (Field *);
ex void CheckMuteY_cpu(void);
ex void CheckMuteZ_cpu(void);


//cfl.c Prototypes
ex void cfl_cpu(void);
ex void cfl_b(void);

//checknans.c
ex int CheckNansField(Field*);
ex int CheckInfiniteField (Field*);
ex void CheckNans(char*);
ex int CheckAxiSym (Field *);
ex void AxiSym (Field *f);

//ComPresAd.c prototypes;
ex void ComputePressureFieldAd_cpu(void);

//ComPresIso.c prototypes;
ex void ComputePressureFieldIso_cpu(void);

//ComPresPoly.c prototypes;
ex void ComputePressureFieldPoly_cpu(void);

//comm.c Prototypes
ex void ResetBuffers(void);
ex void MakeCommunicator (int, int, int, int, int, int, int, int, int, int, int, int);
ex void FillBuffers(Field*, Buffer*, int, int, \
			int, int, char*);
ex void FillBuffersFromField(Field*);
ex void FillFieldFromBuffer(Field*);
ex void SendRecv(int);
ex void comm_cpu(int);
ex void comm_gpu(int);

ex void (_collisions_gpu)(real,int,int,int,int);

//Monitoring Prototypes
ex void mon_dens_cpu(void);
ex void mon_momx_cpu(void);
ex void mon_momy_cpu(void);
ex void mon_momz_cpu(void);
ex void mon_torq_cpu(void);
ex void mon_reynolds_cpu(void);
ex void mon_maxwell_cpu(void);
ex void mon_bxflux_cpu(void);

//Monitoring management Prototypes
ex void MonitorFunction (int, int, char*, int);
ex void DoMonitoring (int, int, char*);
ex void InitFunctionMonitoring (int, void (*f)(void), char *, int, char *, int);
ex int Index(int);
ex void Write1DFile (char *, real *, real *, int);
ex void InitMonitoring(void);
ex void MonitorGlobal (int);
ex void Monitor (int);

ex void Write2D (Field2D *, char *, char *, int);
ex boolean Read2D (Field2D *, char *, char *, int);

//disk3d.c Prototypes
ex void Disk3d(void);

//divrho.c Prototypes
ex void DivideByRho_cpu(Field*);

//usage.c Prototypes
ex void PrintUsage (char *execname);

//Psys.c Prototypes
ex int FindNumberOfPlanets (char *filename);
ex PlanetarySystem *AllocPlanetSystem (int nb);
ex void FreePlanetary(void);
ex void ListPlanets(void);
ex real GetPsysInfo(boolean action);
ex void RotatePsys(real angle);
ex PlanetarySystem *InitPlanetarySystem (char *filename);

//split.c prototypes
ex void buildprime(int*);
ex void primefactors (int, int*, int*);
ex void repartition (int*, int, int*);
ex void split(Grid*);

//initfield.c Prototypes
ex int  RestartSimulation(int);
ex void RestartDat(Field*, int);

ex void compute_potential(real);
//planet2d.c Prototypes
ex void InitDensPlanet(void);
ex void InitSoundSpeedPlanet(void);
ex void InitVazimPlanet(void);
ex void planet2d(void);

ex void compute_potential(real);
//planets.c Prototypes
ex void Potential_cpu(void);
ex Force ComputeForce(real, real, real, real, real);
ex void  _ComputeForce_cpu(real, real, real, real, real);
ex void AdvanceSystemFromDisk(real);
ex void CorrectVtheta_cpu(real);
ex void ComputeIndirectTerm (void);

//psys.c Prototypes
ex int FindNumberOfPlanets(char *);
ex PlanetarySystem *AllocPlanetSystem(int);
ex void FreePlanetary(void);
ex real ComputeInnerMass(real);

//stockholm.c Prototypes
ex void init_stockholm(void);
ex void StockholmBoundary_cpu(real);

//substep1.c Prototypes
ex void SubStep1_cpu(real);
ex void SubStep1_x_cpu(real);
ex void SubStep1_y_cpu(real);
ex void SubStep1_z_cpu(real);


//substep2.c Prototypes
ex void SubStep2_a_cpu(real);
ex void SubStep2_b_cpu(real);
ex void SubStep2_cpu(real);

//substep3.c Prototypes
ex void SubStep3_cpu(real);

//transport Prototypes
ex void VanLeerX(Field*, Field*, Field*, real);
ex void TransportX(Field*, Field*, Field*, real); 
ex void TransportY(Field*, Field*, real); 
ex void TransportZ(Field*, Field*, real); 
ex void X_advection (Field*, real);
ex void transport(real);

//vanleer.c Prototypes

ex void VanLeerX_a_cpu(Field *);
ex void VanLeerX_b_cpu(real, Field *, Field *, Field *);

ex void VanLeerY_a_cpu(Field *);
ex void VanLeerY_b_cpu(real, Field *, Field *);

ex void VanLeerZ_a_cpu(Field *);
ex void VanLeerZ_b_cpu(real, Field *, Field *);

//momenta.c Prototypes
ex void momenta_x_cpu(void);
ex void momenta_y_cpu(void);
ex void momenta_z_cpu(void);

//output.c Prototypes
ex void SelectWriteMethod(void);
ex void WriteTorqueAndWork(int, int);
ex void EmptyPlanetSystemFiles(void);
ex void WritePlanetFile (int, int, boolean);
ex void WritePlanetSystemFile(int, boolean);
ex real GetfromPlanetFile (int, int, int);
ex void RestartPlanetarySystem (int, PlanetarySystem *);
ex void Summary (int);
ex char *ExtractFromExecutable (boolean, char*, int);
ex void SelectArchFileName (void);
ex void StoreFileToChar (char**, char*);
ex void GetHostsList (void);
ex void SelectWrite(void);
ex void Write_offset(int, char*, char*);
ex MPI_Offset ParallelIO(Field *, int, int , MPI_Offset, int);

ex void WriteDim(void);

ex void WriteField(Field*, int);
ex void WriteField_vector(Field*, int);
ex void WriteMerging(Field*, int);
ex void WriteMerging1(Field*, int);

ex void WriteFieldInt2D(FieldInt2D*, int);
ex void WriteField2D(Field2D*, int);
ex void WriteFieldGhost(Field*, int);
ex void WriteBinFile(int, int, int, real*, char*);
ex void WriteOutputs(int);
ex void Display(void);
ex void DumpAllFields (int);

ex void WriteVTK(Field *, int);
ex void WriteVTKMerging(Field *, int);

ex void write_vtk_header(FILE*, Field*, int);
ex void write_vtk_coordinates(FILE*, Field*);
ex void write_vtk_scalar(FILE*, Field*);

//update.c
ex void UpdateX_cpu(real, Field*, Field*, Field*);
ex void UpdateY_cpu(real, Field*, Field*);
ex void UpdateZ_cpu(real, Field*, Field*);
ex void UpdateDensityX_cpu(real, Field*, Field*);
ex void UpdateDensityY_cpu(real, Field*);
ex void UpdateDensityZ_cpu(real, Field*);
//newvel Prototypes
ex void NewVelocity_x_cpu (void);
ex void NewVelocity_y_cpu (void);
ex void NewVelocity_z_cpu (void);

ex void visctensor_cart_cpu(void);
ex void addviscosity_cart_cpu(real);
ex void visctensor_cyl_cpu(void);
ex void addviscosity_cyl_cpu(real);
ex void visctensor_sph_cpu(void);
ex void addviscosity_sph_cpu(real);

ex void viscosity(real);

//Var.c Prototypes
ex void InitVariables(void);

//fargo.c Prototypes
ex void ComputeVmed (Field*);
ex void ComputeResidual_cpu(real);
ex void AdvectSHIFT_cpu(Field*, FieldInt2D*);
ex void ChangeFrame_cpu(int, Field*, Field2D*);

ex void copy_field_cpu(Field*,Field*);

//Dust Diffusion module Prototypes
ex void DustDiffusion_Main(real);
ex void DustDiffusion_Core_cpu(real);
ex void DustDiffusion_Coefficients_cpu();

//mhd.c Prototypes
ex void ComputeSlopes_cpu(int, int, int, Field *, Field *);
ex void _ComputeStar_cpu(real, int, int, int, int,  int, int, int,
			 int, int, Field*,Field*,Field*,Field*,
			 Field*,Field*,Field*,Field*,Field*,Field*);
ex void ComputeStar (real, int, int, int, int, int, int, int,
		     Field*, Field*,Field*,Field*,Field*,Field*);

ex void ComputeEmf (real,int, int, int,Field*, Field*, Field*, Field*);
void _ComputeEmf_cpu(real,int,int,int,int,int,int,
		     Field*,Field*,Field*,Field*,
		     Field*,Field*, Field*, Field*,Field*);

ex void LorentzForce (real, Field*, Field*, int, int, int);
ex void _LorentzForce_cpu(real, int, int, int, int, int, int, int, int, int, int, int, Field*, Field*,Field*, Field*,Field*);


ex void UpdateMagneticField (real, int, int, int);
ex void _UpdateMagneticField_cpu(real,int,int,int,int,int,int,int,int,int,
			      Field*,Field*,Field*);
  

ex void ComputeMHD (real);
ex void ComputeDivergence (Field *, Field *, Field *);
ex void Fill_Resistivity_Profiles (void);
ex real Resistivity (real, real);
ex void Resist (int, int, int);
ex void _Resist_cpu (int, int, int, int, int , int, int, int, int, Field*, Field*, Field*, Field2D*);

//AD and Hall prototypes
ex void ComputeJx_cpu();
ex void ComputeJy_cpu();
ex void ComputeJz_cpu();
//Hall effect prototypes
ex void HallEffect(real);
ex void HallEffect_emfx_cpu(void);
ex void HallEffect_emfy_cpu(void);
ex void HallEffect_emfz_cpu(void);
ex void HallEffect_UpdateB(real, int, int, int);
ex void _HallEffect_UpdateB_cpu(real,int,int,int,int,int,int,int,int,int,int,int,Field*,Field*,Field*);
ex void HallEffect_UpdateEmfs_cpu();
ex void HallEffect_coeff_cpu();
//Ambipolar diffusion prototypes
ex void AmbipolarDiffusion(void);
ex void AmbipolarDiffusion_emfx_cpu(void);
ex void AmbipolarDiffusion_emfy_cpu(void);
ex void AmbipolarDiffusion_emfz_cpu(void);
ex void AmbipolarDiffusion_coeff_cpu(void);
//Ohmic diffusion prototypes
ex void OhmicDiffusion(void);
ex void OhmicDiffusion_emf(int,int,int);
ex void _OhmicDiffusion_emf_cpu (int,int,int,int,int,int,int,int, int,Field*, Field*, Field*);

ex void OhmicDiffusion_coeff_cpu(void);

//timeinfo.c Prototypes
ex void GiveTimeInfo (int);
ex void InitSpecificTime (TimeProcess *, char *);
ex real GiveSpecificTime (TimeProcess);

//defout.c Prototypes
ex void ReadDefaultOut (void);
ex void SubsDef (char *, char *);

//fargo_mhd.c Prototypes
ex void MHD_fargo (real);
ex void Compute_Staggered_2D_fields (real);
ex void EMF_Upstream_Integrate_cpu (real);

//fargo_ppa.c Prototypes
ex void VanLeerX_PPA_2D(Field *, Field *, Field2D *, real);
ex void VanLeerX_PPA(Field *, Field *, Field *, real);
ex void VanLeerX_PPA_a_cpu(Field *);
ex void VanLeerX_PPA_b_cpu(Field *);
ex void VanLeerX_PPA_steep_cpu(Field *);
ex void VanLeerX_PPA_c_cpu(Field *);
ex void VanLeerX_PPA_d_cpu(real, Field *, Field *, Field *);
ex void VanLeerX_PPA_d_2d_cpu(real, Field *, Field *, Field2D *);

ex void ShearBC (int);
ex void SB_slide (Field *);
ex void ShearingPeriodicCondition (void);
ex void SlideIntShearingBoundary (Field *);
ex void SlideResShearingBoundary (Field *);

//compfields.c Prototypes
ex void GiveStats (char *, real *, real *, int);
ex void CompareAllFields (void);
ex boolean CompareField (Field *);

ex void CondInit(void);
ex void PostRestartHook(void);


//CUDA PROTOTYPES-----------------------------------------------------

//fresh.c Prototypes
ex void send2cpu(void);
ex void send2gpu(void);
ex void Input_CPU(Field *, int, const char *);
ex void Input_GPU(Field *, int, const char *);
ex void Input2D_CPU(Field2D *, int, const char *);
ex void Input2D_GPU(Field2D *, int, const char *);
ex void Input2DInt_CPU(FieldInt2D *, int, const char *);
ex void Input2DInt_GPU(FieldInt2D *, int, const char *);
ex void Output_GPU(Field *, int, const char *);
ex void Output_CPU(Field *, int, const char *);
ex void Output2D_GPU(Field2D *, int, const char *);
ex void Output2D_CPU(Field2D *, int, const char *);
ex void Output2DInt_GPU(FieldInt2D *, int, const char *);
ex void Output2DInt_CPU(FieldInt2D *, int, const char *);
ex void Draft(Field *, int, const char *);
ex void check_errors(const char*);
ex void WhereIsField(Field*);
ex void WhereIsFieldInt2D(FieldInt2D*);
ex void WhoOwns (Field *);
ex void SynchronizeHD (void);
ex void WhereIsWho (void);

ex void LightGlobalDev(void);
ex void viscousdrift(void);
ex void CurrentSheetDiffusion(void);

//reduction Prototypes
ex real reduction_full_SUM (Field *, int, int, int, int);
ex void reduction_SUM_cpu (Field *, int, int, int, int);
ex real reduction_full_MIN (Field *, int, int, int, int);
ex void reduction_MIN_cpu (Field *, int, int, int, int);

ex void Prepare_DH_buffers (void);
ex void Input_Contour_Inside (Field *, int);
ex void Output_Contour_Outside (Field *, int);
ex void Input_Contour_Outside (Field *);

//comm_device prototypes
ex void MakeCommunicatorGPU (int, int, int, int, int, int, int, int, int, int, int, int);
ex void ResetBuffersGPU(void);

//resetfields prototypes
ex void Reset_field_gpu(Field*);

ex void reduction_SUM_gpu (Field *, int, int, int, int);
ex void reduction_MIN_gpu (Field *, int, int, int, int);

ex void ComputePressureFieldIso_gpu(void);
ex void ComputePressureFieldAd_gpu(void);
ex void ComputePressureFieldPoly_gpu(void);

ex void SubStep1_x_gpu(real);
ex void SubStep1_y_gpu(real);
ex void SubStep1_z_gpu(real);
ex void SubStep2_a_gpu(real);
ex void SubStep2_b_gpu(real);
ex void SubStep3_gpu(real);
ex void DivideByRho_gpu(Field*);

ex void mon_dens_gpu(void);
ex void mon_momx_gpu(void);
ex void mon_momy_gpu(void);
ex void mon_momz_gpu(void);
ex void mon_torq_gpu(void);
ex void mon_reynolds_gpu(void);
ex void mon_maxwell_gpu(void);
ex void mon_bxflux_gpu(void);

ex void VanLeerX_a_gpu(Field*);
ex void VanLeerX_b_gpu(real,Field*,Field*,Field*);
ex void VanLeerY_a_gpu(Field*);
ex void VanLeerY_b_gpu(real,Field*,Field*);
ex void VanLeerZ_a_gpu(Field*);
ex void VanLeerZ_b_gpu(real,Field*,Field*);

ex void momenta_x_gpu(void);
ex void momenta_y_gpu(void);
ex void momenta_z_gpu(void);

ex void UpdateX_gpu(real, Field*, Field*, Field*);
ex void UpdateY_gpu(real, Field*, Field*);
ex void UpdateZ_gpu(real, Field*, Field*);
ex void UpdateDensityX_gpu(real, Field*, Field*);
ex void UpdateDensityY_gpu(real, Field*);
ex void UpdateDensityZ_gpu(real, Field*);

ex void NewVelocity_x_gpu (void);
ex void NewVelocity_y_gpu (void);
ex void NewVelocity_z_gpu (void);

ex void AdvectSHIFT_gpu(Field*, FieldInt2D*);
ex void ComputeResidual_gpu(real);
ex void ChangeFrame_gpu(int, Field*, Field2D*);

ex void Potential_gpu(void);
ex void CorrectVtheta_gpu(real);
ex void cfl_gpu(void);

ex void  _ComputeForce_gpu(real, real, real, real, real);

ex void Fill_GhostsX_gpu(void);

ex void CheckMuteY_gpu(void);
ex void CheckMuteZ_gpu(void);
ex void SetupHook1_gpu (void);

ex void copy_field_gpu(Field*,Field*);

//DIFFUSION-----------------------------------------------
ex void DustDiffusion_Core_gpu(real);
ex void DustDiffusion_Coefficients_gpu();

//MHD-----------------------------------------------------

ex void ComputeSlopes_gpu(int, int, int, Field *, Field *);
ex void _ComputeStar_gpu(real, int, int, int, int,  int, int, int,
			 int, int, Field*,Field*,Field*,Field*,
			 Field*,Field*,Field*,Field*,Field*,Field*);
ex void _ComputeEmf_gpu(real,int,int,int,int,int,int,
		     Field*,Field*,Field*,Field*,
		     Field*,Field*, Field*, Field*,Field*);
ex void _UpdateMagneticField_gpu(real,int,int,int,int,int,int,int,int,int,
			      Field*,Field*,Field*);
ex void _LorentzForce_gpu(real, int, int, int, int, int, int, int, int, int, int, int, Field*, Field*,Field*, Field*,Field*);


ex void VanLeerX_PPA_a_gpu(Field *);
ex void VanLeerX_PPA_b_gpu(Field *);
ex void VanLeerX_PPA_steep_gpu(Field *);
ex void VanLeerX_PPA_c_gpu(Field *);
ex void VanLeerX_PPA_d_gpu(real, Field *, Field *, Field *);
ex void VanLeerX_PPA_d_2d_gpu(real, Field *, Field *, Field2D *);
ex void _Resist_gpu (int, int, int, int, int , int, int, int, int, Field*, Field*, Field*, Field2D*);
ex void EMF_Upstream_Integrate_gpu (real);

//AD and Hall prototypes
ex void ComputeJx_gpu();
ex void ComputeJy_gpu();
ex void ComputeJz_gpu();
//Hall effect prototypes
ex void HallEffect_emfx_gpu(void);
ex void HallEffect_emfy_gpu(void);
ex void HallEffect_emfz_gpu(void);
ex void _HallEffect_UpdateB_gpu(real,int,int,int,int,int,int,int,int,int,int,int,Field*,Field*,Field*);
ex void HallEffect_UpdateEmfs_gpu();
ex void HallEffect_coeff_gpu();
//Ambipolar diffusion prototypes
ex void AmbipolarDiffusion_emfx_gpu(void);
ex void AmbipolarDiffusion_emfy_gpu(void);
ex void AmbipolarDiffusion_emfz_gpu(void);
ex void AmbipolarDiffusion_coeff_gpu(void);
//Ohmic diffusion prototypes
ex void _OhmicDiffusion_emf_gpu (int,int,int,int,int,int,int,int,int,Field*,Field*,Field*);
ex void OhmicDiffusion_coeff_gpu(void);

ex void StockholmBoundary_gpu(real);

ex void visctensor_cart_gpu(void);
ex void addviscosity_cart_gpu(real);
ex void visctensor_cyl_gpu(void);
ex void addviscosity_cyl_gpu(real);
ex void visctensor_sph_gpu(void);
ex void addviscosity_sph_gpu(real);

ex void ComputeTotalDensity_gpu(void);
ex void Floor_gpu(void);

#ifndef __NOPROTO

ex int DevMalloc(void *, size_t);
ex int DevMemcpyD2H(void *, void *,size_t);
ex int DevMemcpyD2D(void *, void *,size_t);
ex int DevMemcpyH2D(void *, void *,size_t);
ex int DevMemcpyH2H(void *, void *,size_t);
ex int Host2Dev(Field *f);
ex int Dev2Host(Field *f);
ex int Host2Dev2D (Field2D*);
ex int Dev2Host2D (Field2D*);
ex int Host2Dev3D (Field*);
ex int Dev2Host3D (Field*);
ex int Dev2Dev3D(Field *, Field *);
ex int Host2Dev2DInt (FieldInt2D *);
ex int Dev2Host2DInt (FieldInt2D *);

ex void explore(real *, int);

// Multifluid prototypes
ex Fluid *CreateFluid(char*,int);
ex void CflFluidsMin(void);
ex void SelectFluid(int);
ex void ColRate(real,int,int,int);
ex void Collisions(real,int);

ex void (_collisions_cpu)(real,int,int,int,int);

ex void ComputeTotalDensity_cpu(void);
ex void Floor_cpu(void);



#endif

