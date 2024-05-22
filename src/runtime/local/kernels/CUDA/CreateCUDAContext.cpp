
#include "CreateCUDAContext.h"

namespace CUDA {
    void createCUDAContext(DCTX(ctx)) {
        // ToDo: one context per device
        auto dctx = reinterpret_cast<DaphneContext*>(ctx->getUserConfig().dctx_ptr);
        if(dctx && dctx->cuda_contexts.empty()) {
            if (ctx->getUserConfig().log_ptr)
                ctx->getUserConfig().log_ptr->registerLoggers();
            ctx->cuda_contexts.emplace_back(CUDAContext::createCudaContext(0));
        }
    }
}