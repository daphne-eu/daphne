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
#include <inttypes.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/utils.h>

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
        size_t cell = 0;
        for (size_t i = 0; i < arg->getNumRows(); ++i)
        {
            for(size_t j = 0; j < arg->getNumCols(); ++j)
            {
                fprintf(
                        file->identifier,
                        std::is_floating_point<VT>::value ? "%f" : (std::is_same<VT, long int>::value ? "%ld" : "%d"),
                        valuesArg[cell++]
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
  static void apply(const Frame *arg, File* file) {
      
    assert(file != nullptr && "File required");
        for (size_t i = 0; i < arg->getNumRows(); ++i)
        {
            for(size_t j = 0; j < arg->getNumCols(); ++j)
            {
                 auto array = arg->getColumnRaw(j);
                 auto type = arg->getColumnType(j);
                 switch(type) {
                    //Conversion int8->int32 for formating as number as opposed to character
                    case ValueTypeCode::SI8:  fprintf(file->identifier, "%" PRId8, static_cast<int32_t>(reinterpret_cast<const int8_t  *>(array)[i])); break;
                    case ValueTypeCode::SI32: fprintf(file->identifier, "%" PRId32, reinterpret_cast<const int32_t *>(array)[i]); break;
                    case ValueTypeCode::SI64: fprintf(file->identifier, "%" PRId64, reinterpret_cast<const int64_t *>(array)[i]); break;
                    //Conversion uint8->uint32 for formating as number as opposed to character
                    case ValueTypeCode::UI8:  fprintf(file->identifier, "%" PRIu8, static_cast<uint32_t>(reinterpret_cast<const uint8_t  *>(array)[i])); break;
                    case ValueTypeCode::UI32: fprintf(file->identifier, "%" PRIu32, reinterpret_cast<const uint32_t *>(array)[i]); break;
                    case ValueTypeCode::UI64: fprintf(file->identifier,  "%" PRIu64,reinterpret_cast<const uint64_t *>(array)[i]); break;
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
#endif // SRC_RUNTIME_LOCAL_IO_WRITECSV_H