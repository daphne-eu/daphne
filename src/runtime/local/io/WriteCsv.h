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
                if(j < (arg->getNumCols() - 1))
                    fprintf(file->identifier, "%f,", (valuesArg[cell++]));
                else if (j == (arg->getNumCols() - 1))
                    fprintf(file->identifier,"%f\n", (valuesArg[cell++]));
            }
        }
   }
};
  
#endif // SRC_RUNTIME_LOCAL_IO_WRITECSV_H