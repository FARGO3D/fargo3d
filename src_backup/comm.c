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

struct communicator {
  int src; 			/* rank of source and destination */
  int dst;
  struct communicator *next;	/* We use a chained (or linked) list */
  real *buffer; 		/* Will be allocated according to size */
  /* only on the relevant processes, of course */
  int direction;
  int yminsrc;			/* location on source */
  int zminsrc;
  int ymaxsrc;
  int zmaxsrc;
  int ymindst;			/* location on destination */
  int zmindst;
  int ymaxdst;
  int zmaxdst;
  int size;			/* Size of buffer in YZ plane */
  int nvarmax;			/* Maximal number of variables handled
				   simultaneously */
  int stride;
  int parity;
};

typedef struct communicator Communicator;

static Communicator *ListStart = NULL;
static int NbCom=0;

void MakeCommunicator (int src, int dest, int direction,			\
		       int yminsrc, int zminsrc, int ymaxsrc, int zmaxsrc, \
		       int ymindst, int zmindst, int ymaxdst, int zmaxdst, int parity)
{
  Communicator *comm;
  int nbytes, nvar;

  nvar = MAX_FIELDS_PER_COMM;

  comm = malloc (sizeof(Communicator));
  comm->src = src;
  comm->dst = dest;
  comm->direction = direction;
  /* Note : min included, max excluded */
  comm->yminsrc = yminsrc;
  comm->zminsrc = zminsrc;
  comm->ymaxsrc = ymaxsrc;
  comm->zmaxsrc = zmaxsrc;
  comm->ymindst = ymindst;
  comm->zmindst = zmindst;
  comm->ymaxdst = ymaxdst;
  comm->zmaxdst = zmaxdst;
  comm->next = ListStart;	/* insert into chained list */
  ListStart = comm;
  comm->stride = ymaxsrc-yminsrc;
  comm->size = (ymaxsrc-yminsrc)*(zmaxsrc-zminsrc);
  comm->nvarmax = nvar;
  comm->parity = parity;
  if ((src == CPU_Rank) || (dest == CPU_Rank)) {
    nbytes = sizeof(real)*(Nx+2*NGHX)*nvar*comm->size;
    comm->buffer = (real *)malloc(nbytes);
  }
  NbCom++;
}

void ResetBuffers() {
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
	MakeCommunicator (rank,						\
			  (j < Ncpu_x-1 ? rank+1 : rank - Ncpu_x+1),	\
			  RIGHT,					\
			  Ny, NGHZ, Ny+NGHY,NGHZ+Nz+extraz,		\
			  0, NGHZ, NGHY, NGHZ+Nz+extraz,		\
			  j%2);
      }
      /* Do we have a neighbor on our left side ? */
      if ((j > 0) || PERIODICY) {
	MakeCommunicator (rank,						\
			  (j > 0 ? rank-1 : rank + Ncpu_x-1),		\
			  LEFT,						\
			  NGHY, NGHZ, 2*NGHY,NGHZ+Nz+extraz,		\
			  Ny+NGHY, NGHZ, Ny+2*NGHY, NGHZ+Nz+extraz,	\
			  j%2);
      }
      /* Do we have a neighbor on top of us ? */
      if ((k < Ncpu_y-1) || PERIODICZ) {
	MakeCommunicator (rank,						\
			  (k < Ncpu_y-1 ? rank+Ncpu_x : rank + Ncpu_x-(Ncpu_x*Ncpu_y)), \
			  UP,						\
			  NGHY, Nz, NGHY+Ny+extray, NGHZ+Nz,		\
			  NGHY, 0, NGHY+Ny+extray, NGHZ,		\
			  k%2);
      }
      /* Do we have a neighbor below us ? */
      if ((k > 0) || PERIODICZ) {
	MakeCommunicator (rank,						\
			  (k > 0 ? rank-Ncpu_x : rank - Ncpu_x+(Ncpu_x*Ncpu_y)), \
			  DOWN,						\
			  NGHY, NGHZ, NGHY+Ny+extray, 2*NGHZ,		\
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
	  MakeCommunicator (rank,			\
			    rankdest,			\
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
	  MakeCommunicator (rank,			\
			    rankdest,			\
			    UPLEFT,			\
			    NGHY, Nz, 2*NGHY, NGHZ+Nz,	\
			    Ny+NGHY, 0, Ny+2*NGHY, NGHZ,\
			    j%2);
	}
      
      /* Do we have a left-bottom neighbor ? */
      if ((j + PERIODICY > 0) && (k + PERIODICZ > 0))
	{
	  rankdest = rank-1-Ncpu_x;
	  if (j == 0) rankdest += Ncpu_x;
	  if (k == 0) rankdest += Ncpu_x*Ncpu_y;
	  MakeCommunicator (rank,					\
			    rankdest,					\
			    DOWNLEFT,					\
			    NGHY, NGHZ, 2*NGHY, 2*NGHZ,			\
			    Ny+NGHY, Nz+NGHZ, Ny+2*NGHY, Nz+2*NGHZ,	\
			    j%2);
	}
      
      /* Do we have a right-bottom neighbor ? */
      if ((j < Ncpu_x-1 + PERIODICY) && (k + PERIODICZ > 0))
	{
	  rankdest = rank+1-Ncpu_x;
	  if (j == Ncpu_x-1) rankdest -= Ncpu_x;
	  if (k == 0) rankdest += Ncpu_x*Ncpu_y;
	  MakeCommunicator (rank,			\
			    rankdest,			\
			    DOWNRIGHT,			\
			    Ny, NGHZ, Ny+NGHY, 2*NGHZ,	\
			    0, Nz+NGHZ, NGHY, Nz+2*NGHZ,\
			    j%2);
	}
    }
  }
}      

void comm_cpu (int options) {
  static boolean comm_init = NO;
  MPI_Request reqs[8], reqr[8];		/* At most 8 requests per PE */
  Field *f[MAX_FIELDS_PER_COMM];
  real *field[MAX_FIELDS_PER_COMM];
  int special[MAX_FIELDS_PER_COMM];
  int   nvar=0, i, j, k, n, offset, nbreqs=0, nbreqr=0, parity, direction;
  int skip_line;
  Communicator *comm;
  if (comm_init == NO) {
    ResetBuffers ();
    masterprint ("Found %d communicators\n", NbCom);
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

  for (direction = 0; direction < 8; direction++) {
    comm = ListStart;
    while (comm != NULL ) {
      if (comm->direction == direction) {
	if (comm->src == CPU_Rank) {
	  for (i = 0; i < nvar; i++) {
	    field[i] = f[i]->field_cpu;
	    Input_Contour_Inside (f[i],direction);
	  }
	}
      }
      comm = comm->next;
    }
  }
  
  for (parity = 0; parity < 2; parity++) {
    for (direction = 0; direction < 8; direction++) {
      comm = ListStart;
      while (comm != NULL ) {
	if ((comm->parity == parity) && (comm->direction == direction)) {
	  if (comm->src == CPU_Rank) {
	    for (i = 0; i < nvar; i++) {
	      field[i] = f[i]->field_cpu;
	      for (j = comm->yminsrc; j < comm->ymaxsrc; j++) {
		for (k = comm->zminsrc; k < comm->zmaxsrc; k++) {
		  offset = (i*(comm->size)+j-comm->yminsrc+(k-comm->zminsrc)*comm->stride)*(Nx+2*NGHX);
		  memcpy (comm->buffer+offset,field[i]+j*(Nx+2*NGHX)+k*Stride, (Nx+2*NGHX)*sizeof(real));
		}
	      }
	    }
	    if (comm->src != comm->dst) {
#ifdef FLOAT
	      MPI_Isend (comm->buffer, comm->size*nvar*(Nx+2*NGHX), MPI_FLOAT, comm->dst, comm->direction, \
			 MPI_COMM_WORLD, reqs+nbreqs++);
#else
	      MPI_Isend (comm->buffer, comm->size*nvar*(Nx+2*NGHX), MPI_DOUBLE, comm->dst, comm->direction, \
			 MPI_COMM_WORLD, reqs+nbreqs++);
#endif
	    }
	  }
	  if (comm->dst == CPU_Rank) {
	    if (comm->dst != comm->src) {
#ifdef FLOAT
	      MPI_Irecv (comm->buffer, comm->size*nvar*(Nx+2*NGHX), MPI_FLOAT, comm->src, comm->direction, \
			 MPI_COMM_WORLD, reqr+nbreqr);
#else
	      MPI_Irecv (comm->buffer, comm->size*nvar*(Nx+2*NGHX), MPI_DOUBLE, comm->src, comm->direction, \
			 MPI_COMM_WORLD, reqr+nbreqr);
#endif
	      MPI_Wait (reqr+nbreqr++, MPI_STATUS_IGNORE);
	      /* This WAIT instruction must be here, as we have to
		 wait for the data to be ready to send it to the
		 device. We therefore split send and receive requests. */
	    }
	    for (i = 0; i < nvar; i++) {
	      field[i] = f[i]->field_cpu;
	      /* The arithmetical trick below gives the direction of
		 reception. For instance, if a send is toward the
		 left, the receive must be for the right and
		 vice-versa. The expression yields the following
		 involutive correspondences: 0 <--> 1 and 2 <--> 3 */
	      if (direction < 4)
		f[i]->fresh_outside_contour_gpu[1-direction+4*(direction/2)] = NO;
	      skip_line = 0;
	      if ((comm->direction == LEFT) && (special[i] == 1)) skip_line=1;
	      for (j = comm->ymindst+skip_line; j < comm->ymaxdst; j++) {
		for (k = comm->zmindst; k < comm->zmaxdst; k++) {
		  offset = (i*(comm->size)+j-comm->ymindst+(k-comm->zmindst)*comm->stride)*(Nx+2*NGHX);
		  memcpy (field[i]+j*(Nx+2*NGHX)+k*Stride, comm->buffer+offset, (Nx+2*NGHX)*sizeof(real));
		}
	      }
	    }
	  }
	}
	comm = comm->next;
      }
    }
  }

  for (direction = 0; direction < 8; direction++) {
    comm = ListStart;
    while (comm != NULL ) {
      if (comm->direction == direction) {
	if (comm->dst == CPU_Rank) {
	  for (i = 0; i < nvar; i++) {
	    field[i] = f[i]->field_cpu;
	    if (direction < 4)
	      Output_Contour_Outside (f[i],1-direction+4*(direction/2));
	  }
	}
      }
      comm = comm->next;
    }
  }
  for (n = 0; n < nbreqs; n++)
    MPI_Wait (reqs+n, MPI_STATUS_IGNORE);
  MPI_Barrier (MPI_COMM_WORLD);
#ifdef SHEARINGBC
  FARGO_SAFE(ShearBC (options));
#endif  
}
