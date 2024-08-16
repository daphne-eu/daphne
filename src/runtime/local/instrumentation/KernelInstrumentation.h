#include <runtime/local/context/DaphneContext.h>

#include <string>

/**
 * @brief Executes instrumentation code before a kernel is called.
 * Currently only starts the statistics runtime tracking when --statistics is
 * specified by the user.
 */
void preKernelInstrumentation(int kId, DaphneContext *ctx);

/**
 * @brief Executes instrumentation code after a kernel call returned.
 * Currently only stops the statistics runtime tracking when --statistics is
 * specified by the user.
 */
void postKernelInstrumentation(int kId, DaphneContext *ctx);
