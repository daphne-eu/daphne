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

#ifndef SRC_RUNTIME_LOCAL_IO_READCSV_H
#define SRC_RUNTIME_LOCAL_IO_READCSV_H

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>

#include <runtime/local/io/File.h>

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
  static void apply(DTRes *&res, File *file, size_t numRows, size_t numCols,
                    char delim) = delete;

  static void apply(DTRes *&res, File *file, size_t numRows, size_t numCols,
                    char delim, ValueTypeCode *schema) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes>
void readCsv(DTRes *&res, File *file, size_t numRows, size_t numCols,
             char delim) {
  ReadCsv<DTRes>::apply(res, file, numRows, numCols, delim);
}

template <class DTRes>
void readCsv(DTRes *&res, File *file, size_t numRows, size_t numCols,
             char delim, ValueTypeCode *schema) {
  ReadCsv<DTRes>::apply(res, file, numRows, numCols, delim, schema);
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
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct ReadCsv<DenseMatrix<VT>> {
  static void apply(DenseMatrix<VT> *&res, struct File *file, size_t numRows,
                    size_t numCols, char delim) {
    assert(numRows > 0 && "numRows must be > 0");
    assert(numCols > 0 && "numCols must be > 0");

    if (res == nullptr) {
      res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
    }

    char *line;
    std::stringstream lineStream;
    std::string cell;

    size_t row = 0, col = 0;

    while (1) {
      line = getLine(file);

      if (line == NULL)
        break;

      lineStream.str(std::string(line));

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
  }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <> struct ReadCsv<Frame> {
  static void apply(Frame *&res, struct File *file, size_t numRows,
                    size_t numCols, char delim, ValueTypeCode *schema) {
    assert(numRows > 0 && "numRows must be > 0");
    assert(numCols > 0 && "numCols must be > 0");

    if (res == nullptr) {
      res = DataObjectFactory::create<Frame>(numRows, numCols, schema, nullptr, false);
    }

    char *line;
    std::stringstream lineStream;
    std::string cell;
    size_t row = 0, col = 0;

    while (1) {
      line = getLine(file);
      if (line == NULL)
        break;
      lineStream.str(std::string(line));

      while (std::getline(lineStream, cell, delim)) {
        switch (res->getColumnType(col)) {
        case ValueTypeCode::SI8:
          int8_t val_si8;
          convert(cell, &val_si8);
          res->getColumn<int8_t>(col)->set(row, 0, val_si8);
          break;
        case ValueTypeCode::SI32:
          int32_t val_si32;
          convert(cell, &val_si32);
          res->getColumn<int32_t>(col)->set(row, 0, val_si32);
          break;
        case ValueTypeCode::SI64:
          int64_t val_si64;
          convert(cell, &val_si64);
          res->getColumn<int64_t>(col)->set(row, 0, val_si64);
          break;
        case ValueTypeCode::UI8:
          uint8_t val_ui8;
          convert(cell, &val_ui8);
          res->getColumn<uint8_t>(col)->set(row, 0, val_ui8);
          break;
        case ValueTypeCode::UI32:
          uint32_t val_ui32;
          convert(cell, &val_ui32);
          res->getColumn<uint32_t>(col)->set(row, 0, val_ui32);
          break;
        case ValueTypeCode::UI64:
          uint64_t val_ui64;
          convert(cell, &val_ui64);
          res->getColumn<uint64_t>(col)->set(row, 0, val_ui64);
          break;
        case ValueTypeCode::F32:
          float val_f32;
          convert(cell, &val_f32);
          res->getColumn<float>(col)->set(row, 0, val_f32);
          break;
        case ValueTypeCode::F64:
          double val_f64;
          convert(cell, &val_f64);
          res->getColumn<double>(col)->set(row, 0, val_f64);
          break;
        }

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
  }
};

#endif // SRC_RUNTIME_LOCAL_IO_READCSV_H
