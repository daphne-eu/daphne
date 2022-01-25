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

template <class DTRes> struct WriteCsv {
  static void apply(const DTRes *arg, File *file, size_t numRows, size_t numCols) = delete;
  static void apply(DTRes *arg, File *file, ValueTypeCode *schema) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes>
void writeCsv(const DTRes *arg, File *file) {
  WriteCsv<DTRes>::apply(arg, file);
}



template <class DTRes>
void writeCsv(DTRes *arg, File *file,  ValueTypeCode *schema) {
  WriteCsv<DTRes>::apply(arg, file, schema);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct WriteCsv<DenseMatrix<VT>> {
  static void apply(const DenseMatrix<VT> *arg, File* file) {
    assert(file != nullptr && "File required");
    assert(numRows > 0 && "numRows must be > 0");
    assert(numCols > 0 && "numCols must be > 0");
    const VT * valuesArg = arg->getValues();
    size_t cell = 0;
    for (size_t i = 0; i < arg->getNumRows(); ++i)
    {
        for(size_t j = 0; j < arg->getNumCols(); ++j)
        {
            if(j < (arg->getNumCols()-1)){
               fprintf(file->identifier, "%f,", (valuesArg[cell++]));
           }
            else if (j == (arg->getNumCols() -1)) fprintf(file->identifier,"%f\n", (valuesArg[cell++]));
        }
    }
   }
};

  // ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <> struct WriteCsv<Frame> {
  static void apply(Frame *arg, struct File *file, ValueTypeCode *schema) {
    //TODO
  }
};
  
#endif // SRC_RUNTIME_LOCAL_IO_WRITECSV_H