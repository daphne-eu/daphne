

#include <cstdio>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/Structure.h>

#ifndef SRC_RUNTIME_LOCAL_KERNELS_PARFOR_H
#define SRC_RUNTIME_LOCAL_KERNELS_PARFOR_H

// ****************************************************************************
// Convenience function
// ****************************************************************************
void parfor(int64_t from, int64_t to, int64_t step, Structure **inputs, size_t numInputs, bool *isScalar, void *func,
            DCTX(ctx)) {
    auto body = reinterpret_cast<void (*)(int64_t, Structure **, DCTX(ctx))>(func);
    for (int64_t i = from; i <= to; i += step) {
        printf("[parforLoop] Iteration i = %ld\n", i);
        body(i, inputs, ctx);
    }
}

#endif // SRC_RUNTIME_LOCAL_KERNELS_PARFOR_H
