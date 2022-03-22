#include <inttypes.h>

struct DaphneLibResult
{
    void* address;
    int64_t rows;
    int64_t cols;
    int vtc;
};

DaphneLibResult daphne_lib_res;