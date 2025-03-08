#include <runtime/local/instrumentation/KernelInstrumentation.h>

void preKernelInstrumentation(int kId, DaphneContext *ctx) {
    if (ctx->getUserConfig().enable_statistics)
        ctx->startKernelTimer(kId);
}

void postKernelInstrumentation(int kId, DaphneContext *ctx) {
    if (ctx->getUserConfig().enable_statistics)
        ctx->stopKernelTimer(kId);
}
