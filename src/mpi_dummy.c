/** \file mpi_dummy.c

Dummy MPI functions library for sequential built.
It is used instead of the true MPI library in the
case of a sequential built (see makefile).

*/

#include <stdio.h>
#include "mpi_dummy.h"

void MPI_Comm_rank (MPI_Comm a, int *b) {(void)a; *b = 0;} /* Only one process, with rank zero... */

void MPI_Comm_size (MPI_Comm a, int *b) {(void)a; *b = 1;} /* Only one process in the world communicator... */

void MPI_Init (int *argc, char **argv[]) {
  (void)argc;
  fprintf (stderr, "\n       !!!! WARNING !!!!\n\n");
  fprintf (stderr, "This is a sequential built of the %s code\n", *argv[0]);
  fprintf (stderr, "If you planned to run the MPI-parallel version,\n");
  fprintf (stderr, "then you MUST rebuild the executable. In this case,\n");
  fprintf (stderr, " issue:\nmake PARALLEL=1 (or make para)\n");
  fprintf (stderr, "\nAny subsequent invocation of make will build an MPI version.\n");
}

void MPI_Allreduce (const void *ptr, void *ptr2, int count, int type, int foo3, MPI_Comm foo4) {
  int i;
  (void)foo3;
  (void)foo4;
  for (i = 0; i < count; i++) {
    switch (type) {
    case MPI_FLOAT:
      *(((float *)ptr2)+i) = (float)(*(((const float *)ptr)+i));
      break;
    case MPI_DOUBLE:
      *(((double *)ptr2)+i) = (double)(*(((const double *)ptr)+i));
      break;
    case MPI_INT:
      *(((int *)ptr2)+i) = (int)(*(((const int *)ptr)+i));
      break;
    }
  }
}

void MPI_Gather (const void *s, int n, int type, void *r, int m, int type2, int root, MPI_Comm foo) {
  int i;
  (void)m;
  (void)type2;
  (void)root;
  (void)foo;
  for (i = 0; i < n; i++) {
    switch (type) {
    case MPI_FLOAT:
      *(((float *)r)+i) = (float)(*(((const float *)s)+i));
      break;
    case MPI_DOUBLE:
      *(((double *)r)+i) = (double)(*(((const double *)s)+i));
      break;
    case MPI_INT:
      *(((int *)r)+i) = (int)(*(((const int *)s)+i));
      break;
    case MPI_CHAR:
      *(((char *)r)+i) = (char)(*(((const char *)s)+i));
      break;
    }
  }
}

void MPI_Reduce (const void *ptr, void *ptr2, int count, int type, int foo3, int foo4, MPI_Comm foo5) {
  int i;
  (void)foo3;
  (void)foo4;
  (void)foo5;
  for (i = 0; i < count; i++) {
    switch (type) {
    case MPI_FLOAT:
      *(((float *)ptr2)+i) = (float)(*(((const float *)ptr)+i));
      break;
    case MPI_DOUBLE:
      *(((double *)ptr2)+i) = (double)(*(((const double *)ptr)+i));
      break;
    case MPI_INT:
      *(((int *)ptr2)+i) = (int)(*(((const int *)ptr)+i));
      break;
    }
  }
}

//do nothing...
void MPI_Finalize(void){}
void MPI_Bcast(void *buffer, int count, int datatype, int root, MPI_Comm comm){
  (void)buffer; (void)count; (void)datatype; (void)root; (void)comm;
}
void MPI_Isend(const void *buf, int count, int datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request){
  (void)buf; (void)count; (void)datatype; (void)dest; (void)tag; (void)comm; (void)request;
}
void MPI_Irecv(void *buf, int count, int datatype, int source, int tag, MPI_Comm comm, MPI_Request *request){
  (void)buf; (void)count; (void)datatype; (void)source; (void)tag; (void)comm; (void)request;
}
void MPI_Send(const void *buf, int count, int datatype, int dest, int tag, MPI_Comm comm){
  (void)buf; (void)count; (void)datatype; (void)dest; (void)tag; (void)comm;
}
void MPI_Recv(void *buf, int count, int datatype, int source, int tag, MPI_Comm comm, MPI_Status *status){
  (void)buf; (void)count; (void)datatype; (void)source; (void)tag; (void)comm; (void)status;
}
void MPI_Barrier(MPI_Comm comm){
  (void)comm;
}
void MPI_Wait(MPI_Request *request, MPI_Status *status){
  (void)request; (void)status;
}
void MPI_Scan(const void *sendbuf, void *recvbuf, int count, int datatype, int op, MPI_Comm comm){
  (void)sendbuf; (void)recvbuf; (void)count; (void)datatype; (void)op; (void)comm;
} //In place scans require no special action
void MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm){
  (void)comm; (void)color; (void)key; (void)newcomm;
}
