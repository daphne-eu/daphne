/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SRC_RUNTIME_LOCAL_KERNELS_WRITE_H
#define SRC_RUNTIME_LOCAL_KERNELS_WRITE_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/FileMetaData.h>
#include <runtime/local/io/WriteCsv.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes>
struct Write {
    static void apply(const DTRes * arg, const char * filename, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes>
void write(const DTRes * arg, const char * filename, DCTX(ctx)) {
    Write<DTRes>::apply(arg, filename, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Write<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> * arg, const char * filename, DCTX(ctx)) {
        
        File * file = openFileForWrite(filename);
        FileMetaData::ifFile(filename, arg->getNumRows(), arg->getNumCols(), 1, ValueTypeUtils::codeFor<VT>);
        writeCsv(arg, file);
        closeFile(file);
    }
};


// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template<>
struct Write<Frame> {
    static void apply(const Frame * arg, const char * filename, DCTX(ctx)) {
        //TODO
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_WRITE_H