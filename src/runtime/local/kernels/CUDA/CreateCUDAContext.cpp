
#include "CreateCUDAContext.h"

namespace CUDA {
void createCUDAContext(DCTX(ctx)) {
    // ToDo: one context per device
    if (ctx->getUserConfig().log_ptr)
        ctx->getUserConfig().log_ptr->registerLoggers();
    ctx->cuda_contexts.emplace_back(CUDAContext::createCudaContext(0));
}
} // namespace CUDA