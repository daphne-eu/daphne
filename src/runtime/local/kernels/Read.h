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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_READ_H
#define SRC_RUNTIME_LOCAL_KERNELS_READ_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/FileMetaData.h>
#include <runtime/local/io/ReadCsv.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes>
struct Read {
    static void apply(DTRes *& res, const char * filename, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes>
void read(DTRes *& res, const char * filename, DCTX(ctx)) {
    Read<DTRes>::apply(res, filename, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Read<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const char * filename, DCTX(ctx)) {
        FileMetaData fmd = FileMetaData::ofFile(filename);
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(
                    fmd.numRows, fmd.numCols, false
            );
        
        File * file = openFile(filename);
        readCsv(res, file, fmd.numRows, fmd.numCols, ',');
        closeFile(file);
    }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template<>
struct Read<Frame> {
    static void apply(Frame *& res, const char * filename, DCTX(ctx)) {
        FileMetaData fmd = FileMetaData::ofFile(filename);
        
        ValueTypeCode * schema;
        if(fmd.isSingleValueType) {
            schema = new ValueTypeCode[fmd.numCols];
            for(size_t i = 0; i < fmd.numCols; i++)
                schema[i] = fmd.schema[0];
        }
        else
            schema = fmd.schema.data();
        
        std::string * labels;
        if(fmd.labels.empty())
            labels = nullptr;
        else
            labels = fmd.labels.data();
        
        if(res == nullptr)
            res = DataObjectFactory::create<Frame>(
                    fmd.numRows, fmd.numCols, schema, labels, false
            );
        
        File * file = openFile(filename);
        readCsv(res, file, fmd.numRows, fmd.numCols, ',', schema);
        closeFile(file);
        
        if(fmd.isSingleValueType)
            delete[] schema;
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_READ_H