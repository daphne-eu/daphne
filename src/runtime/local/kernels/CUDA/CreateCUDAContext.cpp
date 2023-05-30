
#include "CreateCUDAContext.h"

namespace CUDA {
    void createCUDAContext(DCTX(ctx)) {
        // ToDo: one context per device
        ctx->cuda_contexts.emplace_back(CUDAContext::createCudaContext(0));
        if(ctx->getUserConfig().log_ptr)
            ctx->getUserConfig().log_ptr->registerLoggers();
    }
}