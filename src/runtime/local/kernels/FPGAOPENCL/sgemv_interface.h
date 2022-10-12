#ifndef SGEMV_INTERFACE
#define SGEMV_INTERFACE
#include <runtime/local/context/DaphneContext.h>


extern int sgemv(const float *A, const float *B, float *C, const int OUTERMOST_I, const int OUTERMOST_K,DCTX(ctx));

#endif


