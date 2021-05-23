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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_READCSV_H
#define SRC_RUNTIME_LOCAL_KERNELS_READCSV_H

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

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

template <class DTRes> struct ReadCsv {
  static void apply(DTRes *&res, char *filename, size_t numRows, size_t numCols,
                    char delim) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes>
void readCsv(DTRes *&res, char *filename, size_t numRows, size_t numCols,
             char delim) {
  ReadCsv<DTRes>::apply(res, filename, numRows, numCols, delim);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

void convert(std::string const &x, double *v) {
  try {
    *v = stod(x);
  } catch (const std::invalid_argument &) {
    *v = std::numeric_limits<double>::quiet_NaN();
  }
}
void convert(std::string const &x, float *v) {
  try {
    *v = stof(x);
  } catch (const std::invalid_argument &) {
    *v = std::numeric_limits<float>::quiet_NaN();
  }
}
void convert(std::string const &x, int8_t *v) { *v = stoi(x); }
void convert(std::string const &x, int32_t *v) { *v = stoi(x); }
void convert(std::string const &x, int64_t *v) { *v = stoi(x); }
void convert(std::string const &x, uint8_t *v) { *v = stoi(x); }
void convert(std::string const &x, uint32_t *v) { *v = stoi(x); }
void convert(std::string const &x, uint64_t *v) { *v = stoi(x); }

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct ReadCsv<DenseMatrix<VT>> {
  static void apply(DenseMatrix<VT> *&res, char *filename, size_t numRows,
                    size_t numCols, char delim) {
    assert(numRows > 0 && "numRows must be > 0");
    assert(numCols > 0 && "numCols must be > 0");

    if (res == nullptr) {
      res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
    }

    std::ifstream file(filename);
    std::string line;
    std::stringstream lineStream;
    std::string cell;

    if (file.is_open()) {
      size_t row = 0, col = 0;
      std::string line;

      while (std::getline(file, line)) {
        lineStream.str(line);

        while (std::getline(lineStream, cell, delim)) {
          VT val;
          convert(cell, &val);
          res->set(row, col, val);
          if (++col >= numCols) {
            break;
          }
        }

        lineStream.clear();
        if (++row >= numRows) {
          break;
        }
        col = 0;
      }

      file.close();
    } else {
      assert(false && "File not found.");
    }
  }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_READCSV_H
