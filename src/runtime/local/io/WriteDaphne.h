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

#pragma once

#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <runtime/local/io/utils.h>
#include <runtime/local/io/DaphneFile.h>
#include <runtime/local/io/DaphneSerializer.h>

#include <type_traits>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <limits>
#include <stdlib.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTArg>
struct WriteDaphne {
    static void apply(const DTArg *arg, const char *filename) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTArg>
void writeDaphne(const DTArg *arg, const char *filename) {
    WriteDaphne<DTArg>::apply(arg, filename);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT>
struct WriteDaphne<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> *arg, const char *filename) {
        std::ofstream f;
        f.open(filename, std::ios::out | std::ios::binary);
        // TODO: check f.good()

        auto ser = DaphneSerializerChunks<const DenseMatrix<VT>>(arg, DaphneSerializer<DenseMatrix<VT>>::DEFAULT_SERIALIZATION_BUFFER_SIZE);
        for (auto it = ser.begin(); it != ser.end(); ++it) {
            f.write(it->second->data(), it->first);
        }

        f.close();
        return;
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT>
struct WriteDaphne<CSRMatrix<VT>> {
    static void apply(const CSRMatrix<VT> *arg, const char *filename) {
        std::ofstream f;
        f.open(filename, std::ios::out | std::ios::binary);
        // TODO: check f.good()

        auto ser = DaphneSerializerChunks<const CSRMatrix<VT>>(arg, DaphneSerializer<CSRMatrix<VT>>::DEFAULT_SERIALIZATION_BUFFER_SIZE);
        for (auto it = ser.begin(); it != ser.end(); ++it) {
            f.write(it->second->data(), it->first);
        }

        f.close();
        return;
    }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <>
struct WriteDaphne<Frame> {
    static void apply(const Frame *arg, const char *filename) {

        std::ofstream f;
        f.open(filename, std::ios::out | std::ios::binary);
        // TODO: check f.good()

        // write header
        DF_header h;
        h.version = 1;
        h.dt = (uint8_t)DF_data_t::Frame_t;
        h.nbrows = (uint64_t)arg->getNumRows();
        h.nbcols = (uint64_t)arg->getNumCols();
        f.write((const char *)&h, sizeof(h));

        const ValueTypeCode *schema = arg->getSchema();
        const std::string *labels = arg->getLabels();

        for (uint64_t c = 0; c < h.nbcols; c++) {
            f.write((const char *)&(schema[c]), sizeof(ValueTypeCode));
        }

        for (uint64_t c = 0; c < h.nbcols; c++) {
            uint16_t len = (labels[c]).length();
            f.write((const char *)&len, sizeof(len));
            f.write((const char *)&(labels[c]), len);
        }

        DF_body b;
        b.rx = 0;
        b.cx = 0;
        f.write((char *)&b, sizeof(b));
        // TODO: consider multiple blocks
        //  Assuming a dense block representation
        //  TODO: Consider alternative representations for frames

        void *vals[h.nbcols];
        for (size_t c = 0; c < h.nbcols; c++) {
            vals[c] = const_cast<void *>(arg->getColumnRaw(c));
        }

        for (size_t r = 0; r < h.nbrows; r++) {
            for (size_t c = 0; c < h.nbcols; c++) {
                switch (schema[c]) {
                    case ValueTypeCode::SI8:
                        f.write((char *)&(reinterpret_cast<int8_t *>(vals[c])[r]), sizeof(int8_t));
                        break;
                    case ValueTypeCode::SI32:
                        f.write((char *)&(reinterpret_cast<int32_t *>(vals[c])[r]), sizeof(int32_t));
                        break;
                    case ValueTypeCode::SI64:
                        f.write((char *)&(reinterpret_cast<int64_t *>(vals[c])[r]), sizeof(int64_t));
                        break;
                    case ValueTypeCode::UI8:
                        f.write((char *)&(reinterpret_cast<uint8_t *>(vals[c])[r]), sizeof(uint8_t));
                        break;
                    case ValueTypeCode::UI32:
                        f.write((char *)&(reinterpret_cast<uint32_t *>(vals[c])[r]), sizeof(uint32_t));
                        break;
                    case ValueTypeCode::UI64:
                        f.write((char *)&(reinterpret_cast<uint64_t *>(vals[c])[r]), sizeof(uint64_t));
                        break;
                    case ValueTypeCode::F32:
                        f.write((char *)&(reinterpret_cast<float *>(vals[c])[r]), sizeof(float));
                        break;
                    case ValueTypeCode::F64:
                        f.write((char *)&(reinterpret_cast<double *>(vals[c])[r]), sizeof(double));
                        break;
                    default:
                        throw std::runtime_error("WriteDaphne::apply: unknown value type code");
                }
            }
        }

        f.close();
        return;
    }
};
