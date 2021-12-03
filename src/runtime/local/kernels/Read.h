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

#include <queue>

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
// CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Read<CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *& res, const char * filename, DCTX(ctx)) {
        FileMetaData fmd = FileMetaData::ofFile(filename);

        assert(fmd.numNonZeros != -1
            && "Currently reading of sparse matrices requires a number of non zeros to be defined");

        if(res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VT>>(
                fmd.numRows, fmd.numCols, fmd.numNonZeros, false
            );

        // TODO/FIXME: file format should be inferred from file extension or specified by user
        // Read file of COO format
        File * file = openFile(filename);
        DenseMatrix<uint64_t> *rowColumnPairs = nullptr;
        readCsv(rowColumnPairs, file, static_cast<size_t>(fmd.numNonZeros), 2, ',');
        closeFile(file);

        // pairs are ordered by first then by second argument (row, then col)
        using RowColPos = std::pair<size_t, size_t>;
        std::priority_queue<RowColPos, std::vector<RowColPos>, std::greater<>> positions;
        for (auto r = 0u; r < rowColumnPairs->getNumRows(); ++r) {
            positions.emplace(rowColumnPairs->get(r, 0), rowColumnPairs->get(r, 1));
        }

        auto *rowOffsets = res->getRowOffsets();
        rowOffsets[0] = 0;
        auto *colIdxs = res->getColIdxs();
        auto *values = res->getValues();
        size_t currValIdx = 0;
        size_t rowIdx = 0;
        while(!positions.empty()) {
            auto pos = positions.top();
            if(pos.first >= res->getNumRows() || pos.second >= res->getNumCols()) {
                throw std::runtime_error("Position [" + std::to_string(pos.first) + ", " + std::to_string(pos.second)
                    + "] is not part of matrix<" + std::to_string(res->getNumRows()) + ", "
                    + std::to_string(res->getNumCols()) + ">");
            }
            while(rowIdx < pos.first) {
                rowOffsets[rowIdx + 1] = currValIdx;
                rowIdx++;
            }
            // TODO: valued COO files?
            values[currValIdx] = 1;
            colIdxs[currValIdx] = pos.second;
            currValIdx++;
            positions.pop();
        }
        while(rowIdx < fmd.numRows) {
            rowOffsets[rowIdx + 1] = currValIdx;
            rowIdx++;
        }
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