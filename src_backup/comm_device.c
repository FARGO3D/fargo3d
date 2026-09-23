#include "fargo3d.h"
#define LEFT  0
#define RIGHT 1
#define DOWN  2
#define UP    3
#define UPRIGHT 4
#define UPLEFT  5
#define DOWNLEFT 6
#define DOWNRIGHT 7

#define MAX_FIELDS_PER_COMM 12

struct gpucommunicator {
  int src; 			/* rank of source and destination */
  int dst;
  int direction;
  struct gpucommunicator *next;	/* We use a chained list */
#ifdef GPU
  struct cudaPitchedPtr buffer;    /* Will be allocated according to
				      size only on the relevant
				      processes, of course */
  struct cudaMemcpy3DParms OnSrc;
  struct cudaMemcpy3DParms OnDst;
#endif
  int size;			/* Size of buffer in YZ plane */
  int nvarmax;			/* Maximal number of variables handled
				   simultaneously */
  void *buffer2d;
  size_t pitch;
  int offset_src;
  int offset_dst;
  int dwbytes;
  int dz;
  int parity;
};

typedef struct gpucommunicator GpuCommunicator;

static GpuCommunicator *ListStart = NULL;
static int NbCom=0;

void MakeCommunicatorGPU (int src, int dest, int direction,			\
			  int yminsrc, int zminsrc, int ymaxsrc, int zmaxsrc, \
			  int ymindst, int zmindst, int ymaxdst, int zmaxdst, int parity)
{
#ifdef GPU
  GpuCommunicator *comm;
  int nvar;
  size_t pitch;
  struct cudaPitchedPtr buffer;
  struct cudaExtent extent;

  nvar = MAX_FIELDS_PER_COMM;

  if (((ymaxdst-ymindst) != (ymaxsrc-yminsrc)) || ((zmaxdst-zmindst) != (zmaxsrc-zminsrc))) {
    prs_error ("Internal error: source/dest size mismatch error in MakeCommunicator.\n");
    exit (EXIT_FAILURE); //Do not use prs_exit() here to avoid dead-lock
  }

  comm = malloc (sizeof(GpuCommunicator));
  comm->src = src;
  comm->dst = dest;
  comm->direction = direction;
  if (Nx+2*NGHX > 1) { //We use 3D functions of the CUDA library (see also dh_boundary.c)
    /* Note : min included, max excluded */
    comm->OnSrc.srcArray = NULL;
    comm->OnSrc.dstArray = NULL;
    comm->OnDst.srcArray = NULL;
    comm->OnDst.dstArray = NULL;
    comm->OnSrc.srcPos = make_cudaPos (0, yminsrc, zminsrc);
    comm->OnDst.dstPos = make_cudaPos (0, ymindst, zmindst);
    // OnSrc.dstPos and OnDst.srcPos defined "on the fly" in comm_gpu ();
    comm->OnSrc.extent = make_cudaExtent ((Nx+2*NGHX)*sizeof(real), ymaxsrc-yminsrc, zmaxsrc-zminsrc);
    comm->OnDst.extent = make_cudaExtent ((Nx+2*NGHX)*sizeof(real), ymaxsrc-yminsrc, zmaxsrc-zminsrc);
    comm->OnSrc.kind = cudaMemcpyDeviceToDevice;
    comm->OnDst.kind = cudaMemcpyDeviceToDevice;
    //  comm->OnSrc.srcPtr defined "on the fly" in comm_gpu ();
    //  comm->OnDst.dstPtr defined "on the fly" in comm_gpu ();
    if ((src == CPU_Rank) || (dest == CPU_Rank)) {
      extent = make_cudaExtent ((Nx+2*NGHX)*sizeof(real),ymaxsrc-yminsrc,(zmaxsrc-zminsrc)*nvar);
      cudaMalloc3D (&buffer, extent);
      check_errors ("allocating comm buffer on GPU");
    }
    comm->OnSrc.dstPtr = buffer;
    comm->OnDst.srcPtr = buffer;
    comm->size = (ymaxsrc-yminsrc)*(zmaxsrc-zminsrc);
  } else {// Case Nx == 1: we use 2D routines such as cudaMallocPitch. Here the pitch is in Y
    cudaMallocPitch (&(buffer.ptr), &pitch,\
		     sizeof(real)*(ymaxsrc-yminsrc), (zmaxsrc-zminsrc)*nvar);
    comm->pitch = pitch;
    comm->offset_src = yminsrc+zminsrc*Stride_gpu;
    comm->offset_dst = ymindst+zmindst*Stride_gpu;
    /* The following two lines are a trick to be able to use the same
       MPI_Isend (or MPI_Irecv) instruction for the cases Nx > 1 and
       Nx==1. The two variables below are only used in these
       MPI_Isend/MPI_Irecv invocations */
    buffer.pitch = 1;
    comm->size = pitch*(zmaxsrc-zminsrc);
  }
  comm->dwbytes = (ymaxsrc-yminsrc)*sizeof(real);
  comm->dz = zmaxsrc-zminsrc;
  comm->buffer = buffer;
  comm->next = ListStart;	/* insert into chained list */
  ListStart = comm;
  comm->nvarmax = nvar;
  comm->parity = parity;
  NbCom++;
#endif
}

void ResetBuffersGPU() {
#ifdef GPU
  int j, k, rank, rankdest;
  int extraz = 0, extray = 0;
#ifdef Z
  extraz = 1;
#endif
#ifdef Y
  extray = 1;
#endif
  for (j = 0; j < Ncpu_x; j++) { /* We scan all CPUs */
    for (k = 0; k < Ncpu_y; k++) {
      rank = j+k*Ncpu_x;
      
      /* ****************************************************** */
      /* ******************  EDGES  *************************** */
      /* ****************************************************** */
      
      /* Do we have a neighbor on our right side ? */
      if ((j < Ncpu_x-1) || PERIODICY) {
	MakeCommunicatorGPU (rank,					\
			     (j < Ncpu_x-1 ? rank+1 : rank - Ncpu_x+1),	\
			     RIGHT,					\
			     Ny, NGHZ, Ny+NGHY,NGHZ+Nz+extraz,		\
			     0, NGHZ, NGHY, NGHZ+Nz+extraz,		\
			     j%2);
      }
      /* Do we have a neighbor on our left side ? */
      if ((j > 0) || PERIODICY) {
	MakeCommunicatorGPU (rank,					\
			     (j > 0 ? rank-1 : rank + Ncpu_x-1),	\
			     LEFT,					\
			     NGHY, NGHZ, 2*NGHY,NGHZ+Nz+extraz,		\
			     Ny+NGHY, NGHZ, Ny+2*NGHY, NGHZ+Nz+extraz,	\
			     j%2);
      }
      /* Do we have a neighbor on top of us ? */
      if ((k < Ncpu_y-1) || PERIODICZ) {
	MakeCommunicatorGPU (rank,					\
			     (k < Ncpu_y-1 ? rank+Ncpu_x : rank + Ncpu_x-(Ncpu_x*Ncpu_y)), \
			     UP,					\
			     NGHY, Nz, NGHY+Ny+extray, NGHZ+Nz,		\
			     NGHY, 0, NGHY+Ny+extray, NGHZ,		\
			     k%2);
      }
      /* Do we have a neighbor below us ? */
      if ((k > 0) || PERIODICZ) {
	MakeCommunicatorGPU (rank,					\
			     (k > 0 ? rank-Ncpu_x : rank - Ncpu_x+(Ncpu_x*Ncpu_y)), \
			     DOWN,					\
			     NGHY, NGHZ, NGHY+Ny+extray, 2*NGHZ,	\
			     NGHY, Nz+NGHZ, NGHY+Ny+extray, Nz+2*NGHZ,	\
			     k%2);
      }
      
      /* ****************************************************** */
      /* ******************  CORNERS  ************************* */
      /* ****************************************************** */

      /* Do we have a right-top neighbor ? */
      if ((j < Ncpu_x-1 + PERIODICY) && (k < Ncpu_y-1 + PERIODICZ))
	/* The sums in the above test is a trick to include a fast
	   test for periodicity. Note that a given test is always true
	   when its associated dimension is periodic (provided YES is
	   defined to 1, and NO defined to 0, which is the case in
	   define.h) */
	{
	  rankdest = rank+1+Ncpu_x;
	  if (j == Ncpu_x-1) rankdest -= Ncpu_x;
	  if (k == Ncpu_y-1) rankdest -= Ncpu_x*Ncpu_y;
	  MakeCommunicatorGPU (rank,			\
			       rankdest,		\
			       UPRIGHT,			\
			       Ny, Nz, NGHY+Ny, NGHZ+Nz,	\
			       0, 0, NGHY, NGHZ,		\
			       j%2);
	}
      
      /* Do we have a left-top neighbor ? */
      if ((j + PERIODICY > 0) && (k < Ncpu_y-1 + PERIODICZ))
	{
	  rankdest = rank-1+Ncpu_x;
	  if (j == 0) rankdest += Ncpu_x;
	  if (k == Ncpu_y-1) rankdest -= Ncpu_x*Ncpu_y;
	  MakeCommunicatorGPU (rank,			\
			       rankdest,		\
			       UPLEFT,			\
			       NGHY, Nz, 2*NGHY, NGHZ+Nz,	\
			       Ny+NGHY, 0, Ny+2*NGHY, NGHZ,	\
			       j%2);
	}
      
      /* Do we have a left-bottom neighbor ? */
      if ((j + PERIODICY > 0) && (k + PERIODICZ > 0))
	{
	  rankdest = rank-1-Ncpu_x;
	  if (j == 0) rankdest += Ncpu_x;
	  if (k == 0) rankdest += Ncpu_x*Ncpu_y;
	  MakeCommunicatorGPU (rank,					\
			       rankdest,				\
			       DOWNLEFT,				\
			       NGHY, NGHZ, 2*NGHY, 2*NGHZ,		\
			       Ny+NGHY, Nz+NGHZ, Ny+2*NGHY, Nz+2*NGHZ,	\
			       j%2);
	}
      
      /* Do we have a right-bottom neighbor ? */
      if ((j < Ncpu_x-1 + PERIODICY) && (k + PERIODICZ > 0))
	{
	  rankdest = rank+1-Ncpu_x;
	  if (j == Ncpu_x-1) rankdest -= Ncpu_x;
	  if (k == 0) rankdest += Ncpu_x*Ncpu_y;
	  MakeCommunicatorGPU (rank,			\
			       rankdest,		\
			       DOWNRIGHT,		\
			       Ny, NGHZ, Ny+NGHY, 2*NGHZ,	\
			       0, Nz+NGHZ, NGHY, Nz+2*NGHZ,	\
			       j%2);
	}
    }
  }
#endif
}      

void comm_gpu (int options) {
#ifdef GPU
  static boolean comm_init = NO;
  MPI_Request reqs[8], reqr[8];		/* At most 8 requests per PE */
  Field *f[MAX_FIELDS_PER_COMM];
  real *field[MAX_FIELDS_PER_COMM];
  int special[MAX_FIELDS_PER_COMM];
  int   nvar=0, i, n, nbreqs=0, nbreqr=0, parity, direction;
  int skip_line;
  GpuCommunicator *comm;
  if (comm_init == NO) {
    ResetBuffersGPU ();
    printf ("Found %d GPU-communicators\n", NbCom);
    comm_init = YES;
  }
  for (i=0; i < MAX_FIELDS_PER_COMM; i++)
    special[i] = 0;
  if (options & DENS)
    f[nvar++] = Density;
  if (options & ENERGY)
    f[nvar++] = Energy;
#ifdef X
  if (options & VX)
    f[nvar++] = Vx;
  if (options & VXTEMP)
    f[nvar++] = Vx_temp;
#endif
#ifdef Y
  if (options & VY)
    f[nvar++] = Vy;
  if (options & VYTEMP)
    f[nvar++] = Vy_temp;
#endif
#ifdef Z
  if (options & VZ)
    f[nvar++] = Vz;
  if (options & VZTEMP)
    f[nvar++] = Vz_temp;
#endif
#ifdef MHD
  if (options & BX)
    f[nvar++] = Bx;
  if (options & BY) {
#ifdef SHEARINGBC
    if (J == Ncpu_x-1) {
      special[nvar] = 1;
    }
#endif
    f[nvar++] = By;
  }
  if (options & BZ)
    f[nvar++] = Bz;
  if (options & EMFX)
    f[nvar++] = Emfx;
  if (options & EMFY)
    f[nvar++] = Emfy;
  if (options & EMFZ)
    f[nvar++] = Emfz;
#endif
  if (nvar == 0) return;
  if (nvar > MAX_FIELDS_PER_COMM) {
    mastererr ("Too many fields sent in one communication\n");
    mastererr ("Rebuild after increasing MAX_FIELDS_PER_COMM\n");
    mastererr ("In %s\n", __FILE__);
    prs_exit (EXIT_FAILURE);
  }

  for (i = 0; i < nvar; i++) {
    Input_GPU (f[i], __LINE__, __FILE__);
    Output_GPU (f[i], __LINE__, __FILE__);
  }
  
  for (parity = 0; parity < 2; parity++) {
    for (direction = 0; direction < 8; direction++) {
      comm = ListStart;
      while (comm != NULL ) {
	if ((comm->parity == parity) && (comm->direction == direction)) {
	  if (comm->src == CPU_Rank) {
	    for (i = 0; i < nvar; i++) {
	      if (Nx+2*NGHX > 1) { // We pile up "bricks" of fields in Z
		comm->OnSrc.dstPos = make_cudaPos (0, 0, i*comm->dz);
		comm->OnSrc.srcPtr = f[i]->gpu_pp;
		cudaMemcpy3D (&(comm->OnSrc));
	      } else {
		field[i] = f[i]->field_gpu;
		cudaMemcpy2D (comm->buffer.ptr+i*comm->pitch*comm->dz, comm->pitch,\
			      field[i]+comm->offset_src, Stride_gpu*sizeof(real),\
			      comm->dwbytes, comm->dz, cudaMemcpyDeviceToDevice);
	      }
	      check_errors ("mpi src send to buffer");
	    }
	    if (comm->src != comm->dst) {
#ifdef FLOAT
	      MPI_Isend (comm->buffer.ptr, comm->size*nvar*comm->buffer.pitch/sizeof(real), \
			 MPI_FLOAT, comm->dst, comm->direction,		\
			 MPI_COMM_WORLD, reqs+nbreqs++);
#else
	      MPI_Isend (comm->buffer.ptr, comm->size*nvar*comm->buffer.pitch/sizeof(real), \
			 MPI_DOUBLE, comm->dst, comm->direction,	\
			 MPI_COMM_WORLD, reqs+nbreqs++);
#endif
	    }
	  }
	  if (comm->dst == CPU_Rank) {
	    if (comm->dst != comm->src) {
#ifdef FLOAT
	      MPI_Irecv (comm->buffer.ptr, comm->size*nvar*comm->buffer.pitch/sizeof(real),\
			 MPI_FLOAT, comm->src, comm->direction,		\
			 MPI_COMM_WORLD, reqr+nbreqr);
#else
	      MPI_Irecv (comm->buffer.ptr, comm->size*nvar*comm->buffer.pitch/sizeof(real), \
			 MPI_DOUBLE, comm->src, comm->direction,	\
			 MPI_COMM_WORLD, reqr+nbreqr);
#endif
	      MPI_Wait (reqr+nbreqr++, MPI_STATUS_IGNORE);
	      /* This WAIT instruction must be here, as we have to
		 wait for the data to be ready to send it to the
		 device. We therefore split send and receive requests. */
	    }
	    for (i = 0; i < nvar; i++) {
	      if (Nx+2*NGHX > 1) {
		comm->OnDst.srcPos = make_cudaPos (0, 0, i*comm->dz);
		comm->OnDst.dstPtr = f[i]->gpu_pp;
		cudaMemcpy3D (&(comm->OnDst));
	      } else {
		field[i] = f[i]->field_gpu;
		cudaMemcpy2D (field[i]+comm->offset_dst, Stride_gpu*sizeof(real), \
			      comm->buffer.ptr+i*comm->dz*comm->pitch, comm->pitch, \
			      comm->dwbytes, comm->dz, cudaMemcpyDeviceToDevice);
	      }
	      check_errors ("mpi dst receive from buffer");

	      //	      skip_line = 0; UNIMPLEMENTED ON GPU - Used for ShearingBox BCs
	      //	      if ((comm->direction == LEFT) && (special[i] == 1)) skip_line=1;
	    }
	  }
	}
	comm = comm->next;
      }
    }
  }

  for (n = 0; n < nbreqs; n++)
    MPI_Wait (reqs+n, MPI_STATUS_IGNORE);
  MPI_Barrier (MPI_COMM_WORLD);
#ifdef SHEARINGBC
  FARGO_SAFE(ShearBC (options));
#endif  
#endif
}
