//Max-Min define
#define MAXVARIABLES  500
#define MAXLINELENGTH 500
#define MAXNAMELENGTH 80
#define WIDESCREEN 80
#define MAX1D 16384
#define MAXPRIME 5000
#define MAX_FIELDS 10
#define CVNR 1.41
#define CVNL 0.05

#define GAS 1
#define DUST 2

#define Input 0
#define Output 1

#define XY 0
#define YZ 1
#define ZX 2

//Types of variables
#define INT     0
#define REAL    1
#define STRING  2
#define BOOL    3

#define ALL      1
#define SPECIFIC 0

//Boolean definitions
#define TRUE  1
#define FALSE 0

//Answer definitions
#define YES 1
#define NO 0
#define IRRELEVANT 2
#define REDEFINED 3

//Method for planet tracking

#define GET 0
#define MARK 1
#define FREQUENCY 2

// Ghost cells

#ifdef X
#ifdef GHOSTSX
#define NGHX 12
#else
#define NGHX 0
#endif
#endif

#ifdef Y
#define NGHY 3
#endif

#ifdef Z
#define NGHZ 3
#endif

#ifndef NGHX
#define NGHX 0
#endif
#ifndef NGHY
#define NGHY 0
#endif
#ifndef NGHZ
#define NGHZ 0
#endif

#ifdef GHOSTSX
#define XIP (Nx+2*NGHX-1) //X Interval Plus (we need to address right neighbor in loop)
#define XIM (1)           //X Interval Minus (we need to address left neighbor in loop)
#else
#define XIP (Nx)
#define XIM (0)
#endif

//comm options definitions
#define DENS 1L
#define VY   2L
#define VX   4L
#define VZ   8L
#define ENERGY 16L
#define BX     32L
#define BY     64L
#define BZ     128L
#define EMFX 256L
#define EMFY 512L
#define EMFZ 1024L
#define VXTEMP 2048L
#define VYTEMP 4096L
#define VZTEMP 8192L

/////////////////////////////////////////////////////
//Definitions relative to Fine Grain Monitoring 
#define MONITORSCALAR (MONITOR_SCALAR+0)
#define MONITOR2D     (MONITOR_2D+0)
#define MONITORY      (MONITOR_Y+0)
#define MONITORY_RAW  (MONITOR_Y_RAW+0)
#define MONITORZ      (MONITOR_Z+0)
#define MONITORZ_RAW  (MONITOR_Z_RAW+0)

#define NOGHOSTINC 0
#define GHOSTINC 1

#define TOTAL 0
#define AVG_MASS 1
#define AVG_VOL 2

#define INDEP_PLANET 0
#define DEP_PLANET 1


//monitoring options definitions
#define MASS       1
#define MAGBETA    2 //Do not use BETA, which is a global variable (input parameter)
#define BXFLUX     4
#define FLUX_Y     8
//#define ENERGY 16 Already defined above
#define FLUX_Z     32
#define MOM_X      64
#define MOM_Y      128
#define MOM_Z      256
#define TORQ       512
#define REYNOLDS   1024
#define MAXWELL    2048

////////////////////////////////////////////////////////////////

#define VTEMP2V    1
#define V2VTEMP    2

#define MERGE 1

//Definitions used for reductions on CPU and GPU
#ifdef __GPU
#define MIN(a,b)  min((a),(b))
#define MAX(a,b)  max((a),(b))
#else
#define MIN(a,b) ((a) < (b) ? (a) : (b));
#define MAX(a,b) ((a) > (b) ? (a) : (b));
#endif
#define max2(a,b) ((a) > (b) ? (a) : (b))
#define max3(a,b,c) (((a)>(b)?(a):(b)) > (c) ? \
		     ((a)>(b)?(a):(b)) :       \
		     (c))
#define SUM(a,b) ((a)+(b))
#define MINSTART 1e20
#define MAXSTART -1e20
#define SUMSTART 0.0
#define function_name_cpu(macro) reduction_ ## macro ## _cpu
#define name_reduction_cpu(name) function_name_cpu(name)
#define function_name_gpu(macro) reduction_ ## macro ## _gpu
#define name_reduction_gpu(name) function_name_gpu(name)
#define function_name(macro) reduction_ ## macro
#define name_reduction(name) function_name(name)
#define function_shift_kernel(macro) kernel_shift_ ## macro
#define kernel_shift(name) function_shift_kernel(name)
#define function_kernel(macro) kernel_ ## macro
#define kernel_reduction(name) function_kernel(name)
#define function_name_full(macro) reduction_full_ ## macro
#define name_full_reduction(name) function_name_full(name)
#define RESET_REDUCTION(name) name ## START
#define INIT_REDUCTION(name) RESET_REDUCTION(name)
// two nested commands to force double expansion above
// Use INIT_REDUCTION in the reduction routines

#ifdef CARTESIAN
#define XC xmed(i)
#define YC ymed(j)
#define ZC zmed(k)
#endif
#ifdef CYLINDRICAL
#define XC ymed(j)*cos(xmed(i))
#define YC ymed(j)*sin(xmed(i))
#define ZC zmed(k)
#endif
#ifdef SPHERICAL
#define XC ymed(j)*cos(xmed(i))*sin(zmed(k))
#define YC ymed(j)*sin(xmed(i))*sin(zmed(k))
#define ZC ymed(j)*cos(zmed(k))
#endif

//Useful definitions for GPU Compat
#ifndef __GPU

#define xmin(i) Xmin[(i)]
#define ymin(i) Ymin[(i)]
#define zmin(i) Zmin[(i)]
#define xmed(i) Xmed[(i)]
#define ymed(i) Ymed[(i)]
#define zmed(i) Zmed[(i)]

#define alpha(i) Alpha[(i)]

#else // #ifdef __GPU

#ifndef BIGMEM
#define xmin_d Xmin_d
#define ymin_d Ymin_d
#define zmin_d Zmin_d
#endif

#define xmed(i) (0.5*(xmin_s[(i+1)]+xmin_s[(i)]))
#define ymed(i) (0.5*(ymin_s[(i+1)]+ymin_s[(i)]))
#define zmed(i) (0.5*(zmin_s[(i+1)]+zmin_s[(i)]))
#endif

#define invDiffXmed(i) invDiffXmed[(i)]
#define invDiffYmed(i) invDiffYmed[(i)]
#define invDiffZmed(i) invDiffZmed[(i)]

#define Xmin(i) Xmin[(i)]
#define Ymin(i) Ymin[(i)]
#define Zmin(i) Zmin[(i)]
#define Xmed(i) Xmed[(i)]
#define Ymed(i) Ymed[(i)]
#define Zmed(i) Zmed[(i)]

#define alpha(i) Alpha[(i)]


#define InvDiffXmed(i) InvDiffXmed[(i)]
#define InvDiffYmed(i) InvDiffYmed[(i)]
#define InvDiffZmed(i) InvDiffZmed[(i)]


#ifndef __GPU

#define InvVol(j,k) (InvVj(j)/Syk(k))
#define Vol(j,k) Syk(k)/InvVj(j)

#define InvVj(i) InvVj[(i)]
#define invVj(i) invVj[(i)]

#define SurfX(j,k) Sxj(j)*Sxk(k)
#define SurfY(j,k) Syj(j)*Syk(k)
#define SurfZ(j,k) Szj(j)*Szk(k)

#else //#ifdef __GPU

#define InvVol(j,k) (InvVj_s[(j)]/Syk_s[(k)])
#define Vol(j,k) Syk_s[(k)]/InvVj_s[(j)]

//#define InvVj(i) InvVj_s[(i)]

#define SurfX(j,k) Sxj_s[(j)]*Sxk_s[(k)]
#define SurfY(j,k) Syj_s[(j)]*Syk_s[(k)]
#define SurfZ(j,k) Szj_s[(j)]*Szk_s[(k)]

#endif

#ifndef __GPU
#ifdef CARTESIAN
#define zone_size_x(j,k) Dx
#define zone_size_y(j,k) (Ymin(j+1)-Ymin(j))
#define zone_size_z(j,k) (Zmin(k+1)-Zmin(k))
#endif

#ifdef CYLINDRICAL
#define zone_size_x(j,k) (Dx*Ymed(j))
#define zone_size_y(j,k) (Ymin(j+1)-Ymin(j))
#define zone_size_z(j,k) (Zmin(k+1)-Zmin(k))
#endif

#ifdef SPHERICAL
#define zone_size_x(j,k) (Dx*Ymed(j)*sin(Zmed(k)))
#define zone_size_y(j,k) (Ymin(j+1)-Ymin(j))
#define zone_size_z(j,k) ((Zmin(k+1)-Zmin(k))*Ymed(j))
#endif

#else // ifdef GPU

#ifdef CARTESIAN
#define zone_size_x(j,k) (dx)
#define zone_size_y(j,k) (ymin(j+1)-ymin(j))
#define zone_size_z(j,k) (zmin(k+1)-zmin(k))
#endif

#ifdef CYLINDRICAL
#define zone_size_x(j,k) (dx*ymed(j))
#define zone_size_y(j,k) (ymin(j+1)-ymin(j))
#define zone_size_z(j,k) (zmin(k+1)-zmin(k))
#endif

#ifdef SPHERICAL
#define zone_size_x(j,k) (dx*ymed(j)*sin(zmed(k)))
#define zone_size_y(j,k) (ymin(j+1)-ymin(j))
#define zone_size_z(j,k) ((zmin(k+1)-zmin(k))*ymed(j))
#endif

#endif

#ifndef __GPU

#ifdef CARTESIAN
#define edge_size_x(j,k) Dx
#define edge_size_x_middlez_lowy(j,k) Dx //Used by FARGO-MHD
#define edge_size_x_middley_lowz(j,k) Dx //Used by FARGO-MHD
#define edge_size_y(j,k) (Ymin(j+1)-Ymin(j))
#define edge_size_z(j,k) (Zmin(k+1)-Zmin(k))
#endif

#ifdef CYLINDRICAL
#define edge_size_x(j,k) (Dx*Ymin(j))
#define edge_size_x_middlez_lowy(j,k) (Dx*Ymin(j))
#define edge_size_x_middley_lowz(j,k) (Dx*Ymed(j))
#define edge_size_y(j,k) (Ymin(j+1)-Ymin(j))
#define edge_size_z(j,k) (Zmin(k+1)-Zmin(k))
#endif

#ifdef SPHERICAL
#define edge_size_x(j,k) (Dx*Ymin(j)*sin(Zmin(k)))
#define edge_size_x_middlez_lowy(j,k) (Dx*Ymin(j)*sin(Zmed(k)))
#define edge_size_x_middley_lowz(j,k) (Dx*Ymed(j)*sin(Zmin(k)))
#define edge_size_y(j,k) (Ymin(j+1)-Ymin(j))
#define edge_size_z(j,k) ((Zmin(k+1)-Zmin(k))*Ymin(j))
#endif

#else //ifdef __GPU

#ifdef CARTESIAN
#define edge_size_x(j,k) dx
#define edge_size_x_middlez_lowy(j,k) dx //Used by FARGO-MHD
#define edge_size_x_middley_lowz(j,k) dx //Used by FARGO-MHD
#define edge_size_y(j,k) (ymin(j+1)-ymin(j))
#define edge_size_z(j,k) (zmin(k+1)-zmin(k))
#endif

#ifdef CYLINDRICAL
#define edge_size_x(j,k) (dx*ymin(j))
#define edge_size_x_middlez_lowy(j,k) (dx*ymin(j))
#define edge_size_x_middley_lowz(j,k) (dx*ymed(j))
#define edge_size_y(j,k) (ymin(j+1)-ymin(j))
#define edge_size_z(j,k) (zmin(k+1)-zmin(k))
#endif

#ifdef SPHERICAL
#define edge_size_x(j,k) (dx*ymin(j)*sin(zmin(k)))
#define edge_size_x_middlez_lowy(j,k) (dx*ymin(j)*sin(zmed(k)))
#define edge_size_x_middley_lowz(j,k) (dx*ymed(j)*sin(zmin(k)))
#define edge_size_y(j,k) (ymin(j+1)-ymin(j))
#define edge_size_z(j,k) ((zmin(k+1)-zmin(k))*ymin(j))
#endif

#endif


#ifndef __GPU

#define Sxj(i) Sxj[(i)]
#define Sxk(i) Sxk[(i)]
#define Syj(i) Syj[(i)]
#define Syk(i) Syk[(i)]
#define Szj(i) Szj[(i)]
#define Szk(i) Szk[(i)]

//#define sxj(i) sxj[(i)]
//#define sxk(i) sxk[(i)]
//#define syj(i) syj[(i)]
//#define syk(i) syk[(i)]
//#define szj(i) szj[(i)]
//#define szk(i) szk[(i)]

#endif

//Index define
#ifndef __GPU

#ifdef GHOSTSX              // <==== MAIN ifdef
#define lxp ((l)+1)
#define lxm ((l)-1)

#define ixm ((i)+1)
#define ixp ((i)-1)

#ifdef Y
#define lyp ((l)+Nx+2*NGHX)
#define lym ((l)-Nx-2*NGHX)
#else
#define lyp ((l))
#define lym ((l))
#endif
#else                           // <====== MAIN else
#define lxp (((i)<(Nx-1)) ? ((l)+1) : ((l)-(Nx-1)))
#define lxm (((i)>0) ? ((l)-1) : ((l)+(Nx-1)))

#define ixm ((i)>0 ? ((i)-1) : Nx-1)
#define ixp ((i)<Nx-1 ? ((i)+1) : 0)

#ifdef Y
#define lyp ((l)+Nx)
#define lym ((l)-Nx)
#else
#define lyp ((l))
#define lym ((l))
#endif
#endif    // <============ MAIN endif

#ifdef Z
#define lzp (l+Stride)
#define lzm (l-Stride)
#else
#define lzp (l)
#define lzm (l)
#endif

#ifdef GHOSTSX
#define l   ((i)+(j)*(Nx+2*NGHX)+((k)*Stride))
#else
#define l   ((i)+(j)*(Nx)+((k)*Stride))
#endif
#define l2D ((j)+((k)*(Ny+2*NGHY)))
#define l2D_int ((j)+((k)*(Ny+2*NGHY)))

#else //defined __GPU

#ifndef GHOSTSX
#define lxp (((i)<size_x-1) ? ((l)+1) : ((l)-(size_x-1)))
#define lxm (((i)>0) ? ((l)-1) : ((l)+(size_x-1)))
#define ixm ((i)>0 ? ((i)-1) : size_x-1)
#define ixp (((i)<size_x-1) ? ((i)+1) : 0)

#else

#define lxp ((l)+1)
#define lxm ((l)-1)
#define ixp ((i)+1)
#define ixm ((i)-1)

#endif

#ifdef Y
#define lyp ((l)+pitch)
#define lym ((l)-pitch)
#else
#define lyp ((l))
#define lym ((l))
#endif

#ifdef Z
#define lzp ((l)+stride)
#define lzm ((l)-stride)
#else
#define lzp (l)
#define lzm (l)
#endif

#define l   ((i)+((j)*pitch)+((k)*stride))
#define l2D ((j)+((k)*pitch2d))
#define l2D_int ((j)+((k)*pitch2d_int))

#endif

//CHECK NANS (redefine FARGO_SAFE by uncommenting the following lines if needed)

//#define FARGO_SAFE( call) {			\
//  printf ("Executing %s\n", #call); \
//  call;					\
//  CheckNans (#call);				\
//  SynchronizeHD ();\
//  }

#define FARGO_SAFE( call) call;

// #define FARGO_SAFE( call) {			\
//printf ("*** Executing %s\n", #call);		\
// call;						\
// DumpAllFields (nbdump--);\
//}


/* The macro defined below (FARGO_DEBUG) is used exclusively to check
   tGPU kernels against their CPU counterparts. It can be used to
   check only one kernel at a time, since it returns to the shell upon
   completion. It must be used as follows: wrap the invocation of a
   function (such as substep1_x(dt)) within FARGO_DEBUG. The macro
   internally needs to insert either _cpu or _gpu at the end of the
   function. As the C preprocessor cannot manipulate strings, we must
   help it by inserting a comma after the name of the function. Here
   is an example:

//Standard invocation:           Substep1_x (dt);
//Alternate standard invocation: FARGO_SAFE(Substep1_x (dt));
//Debugger invocation:           FARGO_DEBUG(Substep1_x,(dt));
Note the extra comma before the argument list.
*/

/* In order for this macro to perform its job, you must build the code
for GPU (make GPU=1). You can choose the setup that you
want. Obviously the setup must correspond to the function that you
want to test (e.g. if you want to test substep3, do not choose an
isothermal setup). Note also that this debugging macro returns upon
the first call of the function to be debugged. If your initial
conditions are too symmetric or regular, you may miss a bug and have
false-negatives. It is therefore advised that you introduce some noise
in the initial conditions of the setup that is used to run the code in
this debugging mode. There is a variable NOISE that defaults to 0. 
*/

/* The fields produced by the CPU function are all written in files
with number 999. Those produced by the GPU kernel are all written in
files with number 998. Fields untouched by the function/kernel should
yield undistinguishable output files. If the statistical information
given by the code looks suspicious, you may examine the files written
with IDL or python, for instance, in order to assess the existence of
a bug and obtain hints about its origin. */

/* This macro uses the possibility of creating "check points" for the
   whole set of Fields. Each field can have a backup and a secondary
   backup. If the function "SaveState()" or "SaveStateSecondary()" is
   never invoked, the backup arrays are never allocated. Using
   FARGO_DEBUG is not memory friendly, as it triplicates the amount of
   host memory used, but this should not be a problem as the debugging
   should be undertaken with setups of moderate size.  */

#define FARGO_DEBUG( function, arguments) {		\
  SaveState ();                                         \
  printf ("Executing %s_cpu%s\n",#function,#arguments);	\
  function ##_cpu arguments;                            \
  DumpAllFields (999);                                  \
  SaveStateSecondary ();                                \
  RestoreState ();                                      \
  printf ("Executing %s_gpu%s\n",#function,#arguments);	\
  function ##_gpu arguments;                            \
  DumpAllFields (998);                                  \
  CompareAllFields ();                                  \
  prs_exit (0);                                         \
}

#define FARGO_SPEEDUP( function, arguments) {           \
  SynchronizeHD ();                                     \
  SaveState ();                                         \
  InitSpecificTime (&t_speedup_cpu, "");                \
  for (t_speedup_count=0; t_speedup_count < 200; t_speedup_count++) {              \
     function ##_cpu arguments;                         \
  }                                                     \
  time_speedup_cpu = GiveSpecificTime (t_speedup_cpu);             \
  RestoreState ();                                      \
  SynchronizeHD ();                                     \
  InitSpecificTime (&t_speedup_gpu, "");                \
  for (t_speedup_count=0; t_speedup_count < 2000; t_speedup_count++) {              \
     function ##_gpu arguments;                         \
  }                                                     \
  time_speedup_gpu = GiveSpecificTime (t_speedup_gpu);             \
  printf ("GPU/CPU speedup in %s: %g\n", #function, time_speedup_cpu/time_speedup_gpu*10.0);\
  printf ("CPU time : %g ms\n", 1e3*time_speedup_cpu/200.0);\
  printf ("GPU time : %g ms\n", 1e3*time_speedup_gpu/2000.0);\
}

//psys define
#define GET             0
#define MARK            1
#define FREQUENCY       2

#ifdef __GPU
#define INPUT( field) Input_GPU(field, __LINE__, __FILE__);
#define OUTPUT( field) Output_GPU(field, __LINE__, __FILE__);
#define INPUT2D( field) Input2D_GPU(field, __LINE__, __FILE__);
#define OUTPUT2D( field) Output2D_GPU(field, __LINE__, __FILE__);
#define INPUT2DINT( field) Input2DInt_GPU(field, __LINE__, __FILE__);
#define OUTPUT2DINT( field) Output2DInt_GPU(field, __LINE__, __FILE__);
#else
#define INPUT( field) Input_CPU(field, __LINE__, __FILE__);
#define OUTPUT( field) Output_CPU(field, __LINE__, __FILE__);
#define INPUT2D( field) Input2D_CPU(field, __LINE__, __FILE__);
#define OUTPUT2D( field) Output2D_CPU(field, __LINE__, __FILE__);
#define INPUT2DINT( field) Input2DInt_CPU(field, __LINE__, __FILE__);
#define OUTPUT2DINT( field) Output2DInt_CPU(field, __LINE__, __FILE__);
#endif

#define DRAFT( field) Draft(field, __LINE__, __FILE__);

#ifdef BIGMEM
#define CUDAMEMCPY(a,b,c,d,e)  cudaMemcpyToSymbol(a, b, sizeof(real*), d, cudaMemcpyHostToDevice);
#define CONSTANT(a,b,c) __device__ __constant__ a *b;
#else
#define CUDAMEMCPY(a,b,c,d,e)  cudaMemcpyToSymbol(a, b, c, d, e);
#define CONSTANT(a,b,c) __device__ __constant__ a b[c];
#endif

#ifdef FLOAT
#define INSPECT_REAL( var) {printf ("%s = %f\n", #var, var);}
#else
#define INSPECT_REAL( var) {printf ("%s = %.18g\n", #var, var);}
#endif

#define INSPECT_INT( var) {printf ("%s = %d\n", #var, var);}

#ifdef FLOAT
#define pow powf
#define sin sinf
#define cos cosf
#define sqrt sqrtf
#endif

#define xstr(s) str(s)
#define str(s) #s

#define DUMP_PPVAR( var) {fprintf (sum, "%s = %s = %g\n", #var, xstr(var), (double)var);} // To be used in summary.c only
#define MULTIFLUID( call)						\
  for (FluidIndex=0;FluidIndex<NFLUIDS;FluidIndex++) {			\
    SelectFluid(FluidIndex);						\
    call;}

#define index(i,j) j+i*NFLUIDS

#define SMALLVEL (1e-9*sqrt(G*MSTAR/R0))
#define SMALLTIME (1e-10*sqrt(R0*R0*R0/(G*MSTAR)))

#define CREATEFIELDALIAS(a,b,c) CreateFieldAlias (a,b,c)
//#define CREATEFIELDALIAS(a,b,c) CreateField (a,c,0,0,0)
