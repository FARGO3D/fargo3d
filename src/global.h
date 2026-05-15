//MPI global variables

int CPU_Rank;
int CPU_Number;
boolean CPU_Master = YES;

//Global variables


boolean Resistivity_Profiles_Filled = NO;
boolean VxIsResidual = NO;
boolean LogGrid = NO;
boolean GuidingCenter = NO;
boolean Corotating = NO;
boolean Restart = NO;
boolean Restart_Full = NO;
boolean Stockholm = NO;
boolean Merge = YES;
boolean Merge_All = NO;
boolean MonitorIntegral;
boolean TimeInfo = NO;
boolean EverythingOnCPU = NO;
boolean ForwardOneStep = NO;
boolean Vtk2dat = NO;
boolean Dat2vtk = NO;
boolean PostRestart = NO;
boolean OnlyInit = NO;
boolean EarlyOutputRename = NO;
boolean RedefineOptions = NO;
boolean DeviceFileSpecified = NO;
boolean StretchOldOutput = NO;
boolean ThereArePlanets = NO;
boolean ThereIsACentralBinary = NO;
real    PhysicalTimeInitial;
real    PhysicalTime = 0;
real    XAxisRotationAngle = 0.0;
char    NewOutputdir[1024];
char    VersionString[1024];
char    StickyOptions[1024];
char    BoundaryFile[4096];
char    CommandLine[1024];
char    FirstCommand[1024];
char    ArchFile[1024];
char    ParameterFile[1024];
char    DefaultOut[1024];
char    DeviceFile[1024];
char    CurrentWorkingDirectory[1024];
char    *InputFile;
char    *PlanetaryFile;
char    *HostsList;
int     TimeStep = 0;
int     FullArrayComms = 0;
int     ContourComms = 0;
int     DeviceManualSelection = -1;
int     ArrayNb = 0;
int     StretchNumber = 0;
int     BinaryStar1 = 0;
int     BinaryStar2 = 0;

real Domega;

real InnerBorder;
real OuterBorder;

TimeProcess t_speedup_cpu;
TimeProcess t_speedup_gpu;
int t_speedup_count;
real time_speedup_cpu;
real time_speedup_gpu;

PlanetarySystem *Sys;
Point DiskOnPrimaryAcceleration;
Point IndirectTerm;

real StepTime;

real localforce[12];
real globalforce[12];

//FIELDS VARIABLES

Grid Gridd;

Field *Vx;
Field *Vy;
Field *Vz;

Field *Vx_temp;
Field *Vy_temp;
Field *Vz_temp;

Field *Vx_half;
Field *Vy_half;
Field *Vz_half;

Field *Slope;

Field *Mpx;
Field *Mpy;
Field *Mpz;
Field *Mmx;
Field *Mmy;
Field *Mmz;

Field *Pot;

Field *DivRho;
Field *DensStar;
Field *Qs;

Field *Density;
Field *Energy;
Field *Pressure;

Field *Total_Density;

Field *QL;
Field *QR;
Field *LapPPA;

Field *Sdiffyczc;
Field *Sdiffyfzc;
Field *Sdiffyczf;
Field *Sdiffyfzf;


// Below: fields specific to FARGO algorithms
Field2D *VxMed;
Field2D *Vxhy;
Field2D *Vxhyr;
Field2D *Vxhz;
Field2D *Vxhzr;
Field2D *Reduction2D;

FieldInt2D *Nxhy;
FieldInt2D *Nxhz;
FieldInt2D *Nshift;

//MHD FIELDS
//#ifdef MHD
Field *Bx;
Field *By;
Field *Bz;

Field *B1_star;
Field *B2_star;

Field *V1_star;
Field *V2_star;

Field *Slope_b1;
Field *Slope_v1;
Field *Slope_b2;
Field *Slope_v2;

Field *Emfx; 
Field *Emfy;
Field *Emfz;

Field *EmfxH;
Field *EmfyH;
Field *EmfzH;
Field *BxH;
Field *ByH;
Field *BzH;
Field *Jx;
Field *Jy;
Field *Jz;
Field *EtaOhm;
Field *EtaHall;
Field *EtaAD;

Field *Divergence;
//#endif

Field2D *Density0;
Field2D *Vx0;
Field2D *Vy0;
Field2D *Vz0;
Field2D *Energy0;

//Communications variables

int Ncpu_x; // Numbers of cpus in x-axis;
int Ncpu_y; // Idem in y-axis;

Buffer Bfd;   //|
Buffer Bfu;   //|
Buffer Bfl;   //|
Buffer Bfr;   //|----> interface buffers for density field; (borders)
Buffer Bfcdl; //|                view comm.c
Buffer Bfcdr; //|
Buffer Bfcul; //|
Buffer Bfcur; //|

//Useful numbers

//CPU GLOBAL LIGHT ARRAYS

real Dx;
real *Xmin;
real *Ymin;
real *Zmin;
real *Xmed;
real *Ymed;
real *Zmed;
real *InvDiffXmed;
real *InvDiffYmed;
real *InvDiffZmed;
real *Sxj;
real *Sxk;
real *Syj;
real *Syk;
real *Szj;
real *Szk;
real *InvVj;
real shift_buffer[MAX1D];

//GPU GLOBAL LIGHT ARRAYS
real *Alpha;
real *Alpha_d;
real *Dx_d;
real *Xmin_d;
real *Ymin_d;
real *Zmin_d;
real *Sxj_d;
real *Sxk_d;
real *Syj_d;
real *Syk_d;
real *Szj_d;
real *Szk_d;
real *InvVj_d;
real shift_buffer_d[MAX1D];

//Grid variables

int Nx;
int Ny;
int Nz;
int J;
int K;
int Y0;
int Z0;
int Stride_cpu;
int Stride_gpu;
int Pitch_cpu;
int Pitch_gpu;
int Pitch_Int_gpu;
int Pitch2D;
int Stride;
int ix;
int iy;
int ycells;
int zcells;
int y0cell;
int z0cell;

//For checknan
Field *ListOfGrids = NULL;

//Boundary variables

int Bounl;
int Bounr;
int Bounu;
int Bound;

//psys variables.
real Xplanet;
real Yplanet;
real Zplanet;
real VXplanet;
real VYplanet;
real VZplanet;
real MplanetVirtual;

MPI_Status fargostat;

real OMEGAFRAME0;

int Fscan;

long VtkPosition = 0; 

//Multifluid variables
int Timestepcount = 0;
int Fluidtype;
int FluidIndex;
real Min[NFLUIDS];
Fluid *Fluids[NFLUIDS];

//Pointers to functions
//WARNING!!! FUNCTIONS' ARGUMENTS MUST NOT CONTAIN BLANK SPACES
void (*ComputePressureFieldIso)();
void (*ComputePressureFieldAd)();
void (*ComputePressureFieldPoly)();
void (*SubStep1_x)(real);
void (*SubStep1_y)(real);
void (*SubStep1_z)(real);
void (*SubStep2_a)(real);
void (*SubStep2_b)(real);
void (*SubStep3)(real);
void (*DivideByRho)(Field*);
void (*VanLeerX_a)(Field*);
void (*VanLeerX_b)(real,Field*,Field*,Field*);
void (*VanLeerY_a)(Field*);
void (*VanLeerY_b)(real,Field*,Field*);
void (*VanLeerZ_a)(Field*);
void (*VanLeerZ_b)(real,Field*,Field*);
void (*momenta_x)();
void (*momenta_y)();
void (*momenta_z)();
void (*reduction_SUM)(Field*,int,int,int,int); 
void (*reduction_MIN)(Field*,int,int,int,int); 
void (*UpdateX)(real,Field*,Field*,Field*);
void (*UpdateY)(real,Field*,Field*);
void (*UpdateZ)(real,Field*,Field*);
void (*UpdateDensityX)(real,Field*,Field*);
void (*UpdateDensityY)(real,Field*);
void (*UpdateDensityZ)(real,Field*);
void (*NewVelocity_x)();
void (*NewVelocity_y)();
void (*NewVelocity_z)();
void (*AdvectSHIFT)(Field*,FieldInt2D*);
void (*ComputeResidual)(real);
void (*ChangeFrame)(int,Field*,Field2D*);
void (*Potential)();
void (*CorrectVtheta)(real);
void (*cfl)(void);
void (*_ComputeForce)(real,real,real,real,real);
void (*copy_velocities)(int);
void (*VanLeerX_PPA_a)(Field*);
void (*VanLeerX_PPA_b)(Field*);
void (*VanLeerX_PPA_steep)(Field*);
void (*VanLeerX_PPA_c)(Field*);
void (*VanLeerX_PPA_d)(real,Field*,Field*,Field*);
void (*VanLeerX_PPA_d_2d)(real,Field*,Field*,Field2D*);
void (*mon_dens)();
void (*mon_momx)();
void (*mon_momy)();
void (*mon_momz)();
void (*mon_torq)();
void (*mon_reynolds)();
void (*mon_maxwell)();
void (*mon_bxflux)();
void (*comm)();
void (*Reset_field)(Field*);
void (*ComputeTotalDensity)();
void (*copy_field)(Field*,Field*);
//DUST DIFFUSION
void (*DustDiffusion_Core)(real);
void (*DustDiffusion_Coefficients)();
//MHD..........................................
void (*ComputeSlopes)(int,int,int,Field*,Field*);
void (*_ComputeStar)(real,int,int,int,int,int,int,int,int,int,Field*,Field*,Field*,Field*,Field*,Field*,Field*,Field*,Field*,Field*);
void (*_ComputeEmf)(real,int,int,int,int,int,int,Field*,Field*,Field*,Field*,Field*,Field*,Field*,Field*,Field*);
void (*_UpdateMagneticField)(real,int,int,int,int,int,int,int,int,int,Field*,Field*,Field*);
void (*_LorentzForce)(real,int,int,int,int,int,int,int,int,int,int,int,Field*,Field*,Field*,Field*,Field*);
void (*_Resist)(int,int,int,int,int,int,int,int,int,Field*,Field*,Field*,Field2D*);
void (*EMF_Upstream_Integrate)(real);

void (*ComputeJx)();
void (*ComputeJy)();
void (*ComputeJz)();

void (*_OhmicDiffusion_emf)(int,int,int,int,int,int,int,int,int,Field*,Field*,Field*);
void (*OhmicDiffusion_coeff)();
void (*HallEffect_emfx)();
void (*HallEffect_emfy)();
void (*HallEffect_emfz)();
void (*_HallEffect_UpdateB)(real,int,int,int,int,int,int,int,int,int,int,int,Field*,Field*,Field*);
void (*HallEffect_UpdateEmfs)();
void (*HallEffect_coeff)();
void (*AmbipolarDiffusion_emfx)();
void (*AmbipolarDiffusion_emfy)();
void (*AmbipolarDiffusion_emfz)();
void (*AmbipolarDiffusion_coeff)();
//.............................................

void (*StockholmBoundary)(real);
void (*visctensor_cart)();
void (*addviscosity_cart)(real);
void (*visctensor_cyl)();
void (*addviscosity_cyl)(real);
void (*visctensor_sph)();
void (*addviscosity_sph)(real);
void (*Fill_GhostsX)();
void (*CheckMuteY)();
void (*CheckMuteZ)();

void (*SetupHook1)();

void (*_collisions)(real,int,int,int,int);
void (*Floor)();

void (*__WriteField)();
void (*__Restart)(Field*,int);

void (*boundary_ymin[NFLUIDS])();
void (*boundary_ymax[NFLUIDS])();
void (*boundary_zmin[NFLUIDS])();
void (*boundary_zmax[NFLUIDS])();
