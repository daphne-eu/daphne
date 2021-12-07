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
#include <runtime/local/datastructures/Handle.h>
#include <runtime/local/kernels/DistributedCaller.h>

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

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct ReadCsv<DenseMatrix<VT>> {
  static void apply(DenseMatrix<VT> *&res, struct File *file, size_t numRows,
                    size_t numCols, char delim) {
    assert(file != nullptr && "File required");
    assert(numRows > 0 && "numRows must be > 0");
    assert(numCols > 0 && "numCols must be > 0");

    if (res == nullptr) {
      res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
    }

    char *line;
    size_t cell = 0;
    VT * valuesRes = res->getValues();

    for(size_t r = 0; r < numRows; r++) {
      line = getLine(file);
      // TODO Assuming that the given numRows is available, this should never
      // happen.
//      if (line == NULL)
//        break;

      size_t pos = 0;
      for(size_t c = 0; c < numCols; c++) {
        VT val;
        convertCstr(line + pos, &val);
        
        // TODO This assumes that rowSkip == numCols.
        valuesRes[cell++] = val;
        
        // TODO We could even exploit the fact that the strtoX functions can
        // return a pointer to the first character after the parsed input, then
        // we wouldn't have to search for that ourselves, just would need to
        // check if it is really the delimiter.
        if(c < numCols - 1) {
            while(line[pos] != delim) pos++;
            pos++; // skip delimiter
        }
      }
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
    size_t row = 0, col = 0;

    uint8_t ** rawCols = new uint8_t * [numCols];
    ValueTypeCode * colTypes = new ValueTypeCode[numCols];
    for(size_t i = 0; i < numCols; i++) {
        rawCols[i] = reinterpret_cast<uint8_t *>(res->getColumnRaw(i));
        colTypes[i] = res->getColumnType(i);
    }

    while (1) {
      line = getLine(file);
      if (line == NULL)
        break;

      size_t pos = 0;
      while (1) {
        switch (colTypes[col]) {
        case ValueTypeCode::SI8:
          int8_t val_si8;
          convertCstr(line + pos, &val_si8);
          reinterpret_cast<int8_t *>(rawCols[col])[row] = val_si8;
          break;
        case ValueTypeCode::SI32:
          int32_t val_si32;
          convertCstr(line + pos, &val_si32);
          reinterpret_cast<int32_t *>(rawCols[col])[row] = val_si32;
          break;
        case ValueTypeCode::SI64:
          int64_t val_si64;
          convertCstr(line + pos, &val_si64);
          reinterpret_cast<int64_t *>(rawCols[col])[row] = val_si64;
          break;
        case ValueTypeCode::UI8:
          uint8_t val_ui8;
          convertCstr(line + pos, &val_ui8);
          reinterpret_cast<uint8_t *>(rawCols[col])[row] = val_ui8;
          break;
        case ValueTypeCode::UI32:
          uint32_t val_ui32;
          convertCstr(line + pos, &val_ui32);
          reinterpret_cast<uint32_t *>(rawCols[col])[row] = val_ui32;
          break;
        case ValueTypeCode::UI64:
          uint64_t val_ui64;
          convertCstr(line + pos, &val_ui64);
          reinterpret_cast<uint64_t *>(rawCols[col])[row] = val_ui64;
          break;
        case ValueTypeCode::F32:
          float val_f32;
          convertCstr(line + pos, &val_f32);
          reinterpret_cast<float *>(rawCols[col])[row] = val_f32;
          break;
        case ValueTypeCode::F64:
          double val_f64;
          convertCstr(line + pos, &val_f64);
          reinterpret_cast<double *>(rawCols[col])[row] = val_f64;
          break;
        }

        if (++col >= numCols) {
          break;
        }
        
        // TODO We could even exploit the fact that the strtoX functions can
        // return a pointer to the first character after the parsed input, then
        // we wouldn't have to search for that ourselves, just would need to
        // check if it is really the delimiter.
        while(line[pos] != delim) pos++;
        pos++; // skip delimiter
      }

      if (++row >= numRows) {
        break;
      }
      col = 0;
    }
    
    delete[] rawCols;
    delete[] colTypes;
  }
};


// ----------------------------------------------------------------------------
// Handle
// ----------------------------------------------------------------------------

template <class DT> struct ReadCsv<Handle<DT>> {
  static void apply(Handle<DT> *&res, struct File *file, 
                      size_t numRows, size_t numCols, char delim) {
    assert(file != nullptr && "File required");

    typename Handle<DT>::HandleMap map;
    // DistributedCaller obj for channel creation
    DistributedCaller<void *, void*, void*> caller;
    
    // Find number of Workers

    auto envVar = std::getenv("DISTRIBUTED_WORKERS");
    assert(envVar && "Environment variable has to be set");
    std::string workersStr(envVar);
    std::string delimiter(",");

    size_t pos;
    size_t numWorkers = 0;
    while ((pos = workersStr.find(delimiter)) != std::string::npos) {
        numWorkers++;
        workersStr.erase(0, pos + delimiter.size());
    }
    numWorkers++;
    // If numRows > numWorkers then -> iterations (number of handles) = numWorkers 
    // But if numRows < numWorkers then iterations (number of handles) = numRows
    size_t maxii = std::min(numRows, numWorkers);

    char *line;
    for (size_t r = 0; r < maxii; r++){
      line = getLine(file);
      size_t pos = 0;
      
      // Get address
      std::string workerAddr;
      while (line[pos] != delim)
        workerAddr.push_back(line[pos++]);
      pos++; // skip delimiter
      
      // Get worker filename
      std::string filename;
      while (line[pos] != delim)
        filename.push_back(line[pos++]);
      pos++; // skip delimiter
      
      // DistributedIndex

      // Get row ix
      size_t rowIx;
      convertStr(line + pos, &rowIx);
      // TODO We could even exploit the fact that the strtoX functions can
      // return a pointer to the first character after the parsed input, then
      // we wouldn't have to search for that ourselves, just would need to
      // check if it is really the delimiter.
      while(line[pos] != delim) pos++;
      pos++; // skip delimiter

      // Get col ix
      size_t colIx;
      convertStr(line + pos, &colIx);
      
      while(line[pos] != delim) pos++;
      pos++; // skip delimiter

      // Get numRows
      size_t workerNumRows;
      convertStr(line + pos, &workerNumRows);
      
      while(line[pos] != delim) pos++;
      pos++; // skip delimiter

      // Get numCols
      size_t workerNumCols;
      convertStr(line + pos, &workerNumCols);
      
      while(line[pos] != delim) pos++;
      pos++; // skip delimiter

      // Add to map handle
      DistributedIndex ix(rowIx, colIx);
      distributed::StoredData storedData;
      storedData.set_filename(filename);
      storedData.set_num_rows(workerNumRows);
      storedData.set_num_cols(workerNumCols);
      
      // Channel creation
      auto channel = caller.GetOrCreateChannel(workerAddr);
      DistributedData data(storedData, workerAddr, channel);
      map.insert({ix, data});
    }
    res = new Handle<DT>(map, numRows, numCols);
  }
};

#endif // SRC_RUNTIME_LOCAL_IO_READCSV_H
