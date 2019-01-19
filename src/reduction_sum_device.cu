#define macro SUM
#ifndef WARPSHUFFLE
#include "reduction_generic_gpu.cu"
#else
#include "reduction_arch35_gpu.cu"
#endif
