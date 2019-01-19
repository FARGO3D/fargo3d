struct param{
  char name[80];
  char *variable;
  int type;
  int need;
  int read;
};


struct hashparam {
  char name[MAXLINELENGTH];
  char stringvalue[MAXLINELENGTH];
  real floatvalue;
  long intvalue;
  long boolvalue;
};



/** Contains all the information about a planetary system at a given
    instant in time. */
struct planetary_system {
  int nb;			/**< Number of planets */
  real *mass;			/**< Masses of the planets */
  real *mass_cpu;			/**< Masses of the planets */
  real *mass_gpu;			/**< Masses of the planets */
  real *x;			/**< x-coordinate of the planets */
  real *x_cpu;			/**< x-coordinate of the planets */
  real *x_gpu;			/**< x-coordinate of the planets */
  real *y;			/**< y-coordinate of the planets */
  real *y_cpu;			/**< y-coordinate of the planets */
  real *y_gpu;			/**< y-coordinate of the planets */
  real *z;			/**< z-coordinate of the planets */
  real *z_cpu;			/**< z-coordinate of the planets */
  real *z_gpu;			/**< z-coordinate of the planets */
  real *vx;			/**< x-coordinate of the planets'velocities */
  real *vy;		        /**< y-coordinate of the planets'velocities */
  real *vz;		        /**< z-coordinate of the planets'velocities */
  real *acc;			/**< The planets' accretion times^-1 */
  char **name;  		/**< The planets' names */
  boolean *FeelDisk;		/**< For each planet tells if it feels the disk (ie migrates) */
  boolean *FeelOthers;		/**< For each planet tells if it feels
				   the other planets' gravity */
};

struct grid {   // Store all relevant information for grid
  int nx; //Nsec --> j
  int ny; //Nrad --> i
  int nz; //Ncol --> k
  int J;  //|--> Global index in X
  int K;  //| --> Global index in Y
  int stride; // init in CreateField();
  int bc_up;
  int bc_down;
  int bc_left;
  int bc_right;
  int NJ; // --> Size of array of processors, in Y (same for everyone)
  int NK; // --> Size of array of processors, in Z (same for everyone)
};

struct zmeanprop {
  real PreviousDate;
  real ResetDate;
  boolean NeverReset;
};

struct field { //Multiple fields on code (density, vx, vy,...)
  char *name;
  real *field_cpu;
  real *backup; // used for creation of check points in debugging CPU vs GPU
  real *secondary_backup; // same thing
  real *field_gpu;
  int x_cpus;
  int y_cpus;
  int type;
  boolean fresh_cpu;
  boolean fresh_gpu;
  boolean fresh_inside_contour_cpu[4];
  boolean fresh_inside_contour_gpu[4];
  boolean fresh_outside_contour_cpu[4];
  boolean fresh_outside_contour_gpu[4];
  struct field *next; // Linkedlist
  struct field **owner; //used for aliases
#ifdef GPU
  struct cudaPitchedPtr gpu_pp;
  struct cudaPitchedPtr cpu_pp;
#endif
  char file_origin[MAXLINELENGTH];
  int line_origin;
  real *x;
  real *y;
  real *z;
  double *zmean;
  struct zmeanprop zp;
};

struct fluid {
  char *name;
  int Fluidtype;
  struct field2D *VxMed;
  struct field *Density;
  struct field *Energy;
  struct field *Vx;
  struct field *Vx_temp;
  struct field *Vy;
  struct field *Vy_temp;
  struct field *Vz;
  struct field *Vz_temp;
  struct field *Vx_half;
  struct field *Vy_half;
  struct field *Vz_half;
#ifdef STOCKHOLM
  struct field2D *Density0;
  struct field2D *Energy0;
  struct field2D *Vx0;
  struct field2D *Vy0;
  struct field2D *Vz0;
#endif
};

struct field2D { //Multiple 2D fields on code (azimuthal averages, etc.)
  char *name;
  real *field_cpu;
  real *backup; // used for creation of check points in debugging CPU vs GPU
  real *secondary_backup;
  real *field_gpu;
  int x_cpus;
  int y_cpus;
  size_t pitch;
  int kind;
  boolean fresh_cpu;
  boolean fresh_gpu;
  struct field2D *next; // Linkedlist
};

struct fieldint2D { //Multiple 2D fields on code (azimuthal averages, etc.)
  char *name;
  int *field_cpu;
  int *backup;
  int *secondary_backup;
  int *field_gpu;
  int x_cpus;
  int y_cpus;
  boolean fresh_cpu;
  boolean fresh_gpu;
  struct fieldint2D *next; // Linkedlist
};

struct buffer {
  real *buffer;
  int index;
};

struct force {			
  real fx_inner;		/**< x-component of the force arising from the inner disk, without Hill sphere avoidance  */
  real fy_inner;		/**< y-component of the force arising from the inner disk, without Hill sphere avoidance  */
  real fz_inner;
  real fx_ex_inner;    /**< x-component of the force arising from the inner disk, with Hill sphere avoidance  */	
  real fy_ex_inner;    /**< y-component of the force arising from the inner disk, with Hill sphere avoidance  */        
  real fz_ex_inner;
  real fx_outer;         /**< x-component of the force arising from the outer disk, without Hill sphere avoidance */
  real fy_outer;	        /**< y-component of the force arising from the outer disk, without Hill sphere avoidance */
  real fz_outer;
  real fx_ex_outer;	/**< x-component of the force arising from the outer disk, with Hill sphere avoidance  */	
  real fy_ex_outer;	/**< x-component of the force arising from the outer disk, with Hill sphere avoidance  */        
  real fz_ex_outer;
};


struct point {
  real x;
  real y;
  real z;
};

/** This structure is used for monitoring CPU time usage. It is used
    only if -t is specified on the command line. */
struct timeprocess {
  char name[80];
  clock_t clicks;
};

struct orbital_elements {
  real a;
  real e;
  real i;
  real an;
  real per;
  real ta;
  real E; //Additional useful anomalies
  real M;
  real Perihelion_Phi;
};

struct state_vector {
  real x;
  real y;
  real z;
  real vx;
  real vy;
  real vz;
};
