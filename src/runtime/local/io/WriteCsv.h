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
  static void apply(const DTRes *res, File *file, size_t numRows, size_t numCols) = delete;

  static void apply(const DTRes *res, File *file, size_t numRows, size_t numCols, ValueTypeCode *schema) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes>
void writeCsv(const DTRes *res, File *file, size_t numRows, size_t numCols) {
  WriteCsv<DTRes>::apply(res, file, numRows, numCols);
}

template <class DTRes>
void writeCsv(const DTRes *res, File *file, size_t numRows, size_t numCols,
              ValueTypeCode *schema) {
  WriteCsv<DTRes>::apply(res, file, numRows, numCols,  schema);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct WriteCsv<DenseMatrix<VT>> {
  static void apply(const DenseMatrix<VT> *res, File* file, size_t numRows,
                    size_t numCols) {
    assert(file != nullptr && "File required");
    assert(numRows > 0 && "numRows must be > 0");
    assert(numCols > 0 && "numCols must be > 0");
    if (res == nullptr) printf("WTF\n");
    const VT * valuesRes = res->getValues();
    size_t cell = 0;
    for (size_t i = 0; i < numRows; ++i)
    {
        for(size_t j = 0; j < numCols; ++j)
        {
            if(j < (numCols-1)){
               fprintf(file->identifier, "%f,", (valuesRes[cell++]));
           }
            else if (j == (numCols -1)) fprintf(file->identifier,"%f\n", (valuesRes[cell++]));
        }
    }
   }
};
    
    template <> struct WriteCsv<Frame>{
      static void apply(const Frame *res, struct File* file, size_t numRows, size_t numCols,
                        ValueTypeCode *schema)
                        {
                          //do stuff
                        }
    };
#endif // SRC_RUNTIME_LOCAL_IO_WRITECSV_H