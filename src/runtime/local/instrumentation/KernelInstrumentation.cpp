#include <runtime/local/instrumentation/KernelInstrumentation.h>

void preKernelInstrumentation(int kId, DaphneContext *ctx)
{
    if (ctx->getUserConfig().statistics)
        ctx->startKernelTimer(kId);
}

void postKernelInstrumentation(int kId, DaphneContext *ctx)
{
    if (ctx->getUserConfig().statistics)
        ctx->stopKernelTimer(kId);
}
