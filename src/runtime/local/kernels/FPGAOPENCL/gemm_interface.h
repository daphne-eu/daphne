#ifndef SGEMM_INTERFACE
#define SGEMM_INTERFACE
#include <runtime/local/context/DaphneContext.h>


extern int sgemm(const float *A, const float *B, float *C, const int OUTERMOST_I, const int OUTERMOST_J, const int OUTERMOST_K, DCTX(ctx));

#endif


