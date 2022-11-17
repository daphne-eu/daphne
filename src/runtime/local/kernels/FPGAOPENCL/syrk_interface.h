#ifndef SSYRK_INTERFACE
#define SSYRK_INTERFACE
#include <runtime/local/context/DaphneContext.h>

extern int syrk(const float *A, float *C, const int OUTERMOST_I, const int OUTERMOST_K, DCTX(ctx));

#endif


