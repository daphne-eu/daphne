

#include <runtime/local/context/DaphneContext.h>


#ifndef SRC_RUNTIME_LOCAL_KERNELS_PARFOR_H
#define SRC_RUNTIME_LOCAL_KERNELS_PARFOR_H

// ****************************************************************************
// Convenience function
// ****************************************************************************
void parfor(int64_t from, int64_t to, int64_t step, void* args, void *func, DCTX(ctx)) {
    //TODO : 1. propagate (llvm) code block to be executed and parameters of iterations(set or range tasks)
    //TODO : 2. create threads - distinct worker implementation?
    //TODO : 3. return results - meta-programming templates for EACH type? 
     auto body = reinterpret_cast<void (*)(void*)>(func);
     for(int64_t i = from; from < to; i+=step) {
        body(args);
     }
}


#endif // SRC_RUNTIME_LOCAL_KERNELS_PARFOR_H