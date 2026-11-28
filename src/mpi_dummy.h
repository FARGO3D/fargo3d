#define MPI_COMM_WORLD 0
#define MPI_IN_PLACE ((void *)0)
#define MPI_DOUBLE 2
#define MPI_FLOAT 4
#define MPI_CHAR 1
#define MPI_LONG 3
#define MPI_INT 0
#define MPI_MIN 0
#define MPI_MAX 0
#define MPI_SUM 0

#define MPI_STATUS_IGNORE ((MPI_Status *)0)

typedef int MPI_Request;
typedef int MPI_Status;
typedef int MPI_Comm;
typedef long MPI_Offset;

void MPI_Comm_rank(MPI_Comm comm, int *rank);
void MPI_Barrier(MPI_Comm comm);
void MPI_Comm_size(MPI_Comm comm, int *size);
void MPI_Scan(const void *sendbuf, void *recvbuf, int count, int datatype, int op, MPI_Comm comm);
void MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
void MPI_Init(int *argc, char ***argv);
void MPI_Finalize(void);
void MPI_Bcast(void *buffer, int count, int datatype, int root, MPI_Comm comm);
void MPI_Isend(const void *buf, int count, int datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
void MPI_Irecv(void *buf, int count, int datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
void MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, int datatype, int op, MPI_Comm comm);
void MPI_Reduce(const void *sendbuf, void *recvbuf, int count, int datatype, int op, int root, MPI_Comm comm);
void MPI_Send(const void *buf, int count, int datatype, int dest, int tag, MPI_Comm comm);
void MPI_Recv(void *buf, int count, int datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
void MPI_Wait(MPI_Request *request, MPI_Status *status);
void MPI_Gather(const void *sendbuf, int sendcount, int sendtype, void *recvbuf, int recvcount, int recvtype, int root, MPI_Comm comm);
