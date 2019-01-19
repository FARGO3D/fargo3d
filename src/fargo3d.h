//System includes
#ifdef GPU
#include <cuda.h>
#include <driver_functions.h> //for some CUDA structs
#include <cuda_runtime_api.h>
#endif

#ifndef __APPLE__
#include <malloc.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <stdarg.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <time.h>

#include "define.h"
#include "types_def.h"
#include "fondam.h"

//#ifndef __NOPROTO //Nvcc is problematic with double references.
#include "prototypes.h"
//#endif

#ifdef PARALLEL
#ifndef __GPU
#include <mpi.h>
#else
#include "mpi_dummy.h"
#endif
#else 
#include "mpi_dummy.h"
#endif

#include "param.h"

#ifndef __LOCAL // ONLY VAR.C HAS __LOCAL
#include "structs.h"
#include "global_ex.h"
#else
#include "structs.h"
#include "global.h"
#endif
