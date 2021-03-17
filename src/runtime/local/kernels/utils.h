#ifndef LIB_KERNELS_UTILS_H
#define LIB_KERNELS_UTILS_H

#include <cassert>

/**
 * `typeNew` should be a pointer type.
 */
#define dynamic_cast_assert(typeNew, varNew, varOld) \
    typeNew varNew = dynamic_cast<typeNew>(varOld); \
    assert(varNew && "'" #varOld "'does not have the expected type '" #typeNew "'");

#endif //LIB_KERNELS_UTILS_H