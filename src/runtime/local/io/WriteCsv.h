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

#ifndef SRC_RUNTIME_LOCAL_IO_WRITECSV_H
#define SRC_RUNTIME_LOCAL_IO_WRITECSV_H

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>

#include <runtime/local/io/File.h>
#include <runtime/local/io/utils.h>

#include <stdexcept>
#include <type_traits>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTArg>
struct WriteCsv {
    static void apply(const DTArg *arg, File *file) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTArg>
void writeCsv(const DTArg *arg, File *file) {
    WriteCsv<DTArg>::apply(arg, file);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT>
struct WriteCsv<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> *arg, File* file) {
        assert(file != nullptr && "File required");
        const VT * valuesArg = arg->getValues();
        const size_t rowSkip = arg->getRowSkip();
        const size_t argNumCols = arg->getNumCols();

        for (size_t i = 0; i < arg->getNumRows(); ++i)
        {
            for(size_t j = 0; j < argNumCols; ++j)
            {
                fprintf(
                        file->identifier,
                        std::is_floating_point<VT>::value ? "%f" : (std::is_same<VT, long int>::value ? "%ld" : "%d"),
                        valuesArg[i*rowSkip + j]
                );
                if(j < (arg->getNumCols() - 1))
                    fprintf(file->identifier, ",");
                else
                    fprintf(file->identifier, "\n");
            }
        }
   }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <> struct WriteCsv<Frame> {
    static void apply(const Frame * arg, File * file) {

    assert(file != nullptr && "File required");

    for(size_t i = 0; i < arg->getNumRows(); ++i) {
        for(size_t j = 0; j < arg->getNumCols(); ++j) {
            const void* array = arg->getColumnRaw(j);
            ValueTypeCode vtc = arg->getColumnType(j);
            switch(vtc) {
                // Conversion int8->int32 for formating as number as opposed to character.
                case ValueTypeCode::SI8:  fprintf(file->identifier, "%" PRId8, static_cast<int32_t>(reinterpret_cast<const int8_t *>(array)[i])); break;
                case ValueTypeCode::SI32: fprintf(file->identifier, "%" PRId32, reinterpret_cast<const int32_t *>(array)[i]); break;
                case ValueTypeCode::SI64: fprintf(file->identifier, "%" PRId64, reinterpret_cast<const int64_t *>(array)[i]); break;
                // Conversion uint8->uint32 for formating as number as opposed to character.
                case ValueTypeCode::UI8:  fprintf(file->identifier, "%" PRIu8, static_cast<uint32_t>(reinterpret_cast<const uint8_t *>(array)[i])); break;
                case ValueTypeCode::UI32: fprintf(file->identifier, "%" PRIu32, reinterpret_cast<const uint32_t *>(array)[i]); break;
                case ValueTypeCode::UI64: fprintf(file->identifier, "%" PRIu64, reinterpret_cast<const uint64_t *>(array)[i]); break;
                case ValueTypeCode::F32: fprintf(file->identifier, "%f", reinterpret_cast<const float  *>(array)[i]); break;
                case ValueTypeCode::F64: fprintf(file->identifier, "%f", reinterpret_cast<const double *>(array)[i]); break;
                default: throw std::runtime_error("unknown value type code");
            }

            if(j < (arg->getNumCols() - 1))
                fprintf(file->identifier, ",");
            else
                fprintf(file->identifier, "\n");
        }
    }
}

};

// ----------------------------------------------------------------------------
// Matrix
// ----------------------------------------------------------------------------

template <typename VT>
struct WriteCsv<Matrix<VT>> {
    static void apply(const Matrix<VT> *arg, File* file) {
        if (file == nullptr)
            throw std::runtime_error("WriteCsv: File required");

        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        for (size_t r = 0; r < numRows; ++r) {
            for (size_t c = 0; c < numCols; ++c) {
                fprintf(
                        file->identifier,
                        std::is_floating_point<VT>::value ? "%f" : (std::is_same<VT, long int>::value ? "%ld" : "%d"),
                        arg->get(r, c)
                );
                if (c < (numCols - 1))
                    fprintf(file->identifier, ",");
                else
                    fprintf(file->identifier, "\n");
            }
        }
   }
};
  
#endif // SRC_RUNTIME_LOCAL_IO_WRITECSV_H