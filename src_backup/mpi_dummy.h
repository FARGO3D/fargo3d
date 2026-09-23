#define MPI_COMM_WORLD 0
#define MPI_IN_PLACE 0
#define MPI_DOUBLE 2
#define MPI_FLOAT 4
#define MPI_CHAR 1
#define MPI_LONG 3
#define MPI_INT 0
#define MPI_MIN 0
#define MPI_MAX 0
#define MPI_SUM 0

#define MPI_STATUS_IGNORE 0

typedef int MPI_Request;
typedef int MPI_Status;
typedef int MPI_Comm;
typedef long MPI_Offset;

void MPI_Comm_rank();
void MPI_Barrier();
void MPI_Comm_size();
void MPI_Scan();
void MPI_Comm_split();
void MPI_Init();
void MPI_Finalize();
void MPI_Bcast();
void MPI_Isend();
void MPI_Irecv();
void MPI_Allreduce();
void MPI_Reduce();
void MPI_Send();
void MPI_Recv();
void MPI_Wait();
void MPI_Gather();
