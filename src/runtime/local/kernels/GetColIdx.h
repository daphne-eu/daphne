#ifndef SRC_RUNTIME_LOCAL_KERNELS_INNERJOIN_H
#define SRC_RUNTIME_LOCAL_KERNELS_INNERJOIN_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <stdexcept>

#include <cstddef>
#include <cstdint>


// ****************************************************************************
// Convenience function
// ****************************************************************************

size_t getColIdx(
    // input frame
    const Frame * arg,
    // column name
    const char * colName,
    // context
    DCTX(ctx)
) {
    return arg->getColumnIdx(colName);
}
#endif //SRC_RUNTIME_LOCAL_KERNELS_INNERJOIN_H
