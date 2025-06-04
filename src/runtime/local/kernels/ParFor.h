

#include <runtime/local/context/DaphneContext.h>
#include <cstdio>

#ifndef SRC_RUNTIME_LOCAL_KERNELS_PARFOR_H
#define SRC_RUNTIME_LOCAL_KERNELS_PARFOR_H

// ****************************************************************************
// Convenience function
// ****************************************************************************
void parfor(int64_t from, int64_t to, int64_t step, void * func, DCTX(ctx)) {
    auto body = reinterpret_cast<void (*)()>(func);
    for(int64_t i = from; i <= to; i+=step) {
      printf("[parforLoop] Iteration i = %ld\n", i);
      body();
    }
}


#endif // SRC_RUNTIME_LOCAL_KERNELS_PARFOR_H