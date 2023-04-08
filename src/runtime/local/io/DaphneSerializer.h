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

#include <runtime/local/io/DaphneFile.h>

#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/Frame.h>

#include <cassert>
#include <cstdint>
#include <stdlib.h>
#include <stdexcept>
#include <iterator>
#include <climits>

inline DF_data_t DF_Dtype(const char *buf) { 
    return (DF_data_t)((const DF_header *)buf)->dt;
}
inline DF_data_t DF_Dtype(const std::vector<char>& buf) { return DF_Dtype(buf.data()); };

inline ValueTypeCode DF_Vtype(const char *buf) {
    const ValueTypeCode *vt = (const ValueTypeCode *)((const char *)buf + sizeof(DF_header));
    return *vt;
}
inline ValueTypeCode DF_Vtype(const std::vector<char>& buf) { return DF_Vtype(buf.data()); }

template <class DTArg, bool isFundumental = std::is_fundamental<DTArg>::value>
struct DaphneSerializer { 
    static size_t length(const DTArg *arg);
    static size_t serialize(const DTArg *arg, char *buf, size_t serializeFromByte = 0, size_t chunkSize = 0);
    static Structure *deserialize(const char *buf, DTArg *arg = nullptr, size_t deserializeFromByte = 0, size_t chunkSize = 0);
};

template <typename VT>
struct DaphneSerializer<DenseMatrix<VT>, false> {

    static size_t length(const DenseMatrix<VT> *arg) {
        size_t len = 0;

        /* TODO MPI: What types to use for the size caluclation? */
        len += sizeof(DF_header);
        len += sizeof(ValueTypeCode);
        len += sizeof(DF_body);
        len += sizeof(DF_body_block);
        len += sizeof(ValueTypeCode);
        len += arg->getNumRows() * arg->getNumCols() * sizeof(VT);

        return len;
    };
    
    static size_t serialize(const DenseMatrix<VT> *arg, char *buffer, size_t serializeFromByte = 0, size_t chunkSize = 0) {
        size_t bufferIdx = 0;
        size_t serializationIdx = 0;
        chunkSize = chunkSize != 0 ? chunkSize : DaphneSerializer<DenseMatrix<VT>>::length(arg);

        if (buffer == nullptr){
            throw std::runtime_error("Buffer is nullptr");
        }


        if (serializeFromByte == 0 && chunkSize < 60)
            throw std::runtime_error("Minimum starting chunk size 60 bytes"); // For now..?

        // write header
        DF_header h;
        h.version = 1;
        h.dt = (uint8_t)DF_data_t::DenseMatrix_t;
        h.nbrows = (uint64_t) arg->getNumRows();
        h.nbcols = (uint64_t) arg->getNumCols();
   
        if (serializeFromByte < serializationIdx + sizeof(h)) {
            std::copy(reinterpret_cast<const char*>(&h), reinterpret_cast<const char*>(&h) + sizeof(h), buffer);
            bufferIdx += sizeof(h);
        }
        serializationIdx += sizeof(h);

        // value type
        const ValueTypeCode vt = ValueTypeUtils::codeFor<VT>;

        // Check if we go out of limit
        if (chunkSize < bufferIdx + sizeof(vt))
            return bufferIdx;
            
        // Check if we actually need this    
        if (serializeFromByte < serializationIdx + sizeof(vt)) {
            std::copy(reinterpret_cast<const char*>(&vt), reinterpret_cast<const char*>(&vt) + sizeof(vt), buffer + bufferIdx);
            bufferIdx += sizeof(vt);
        }
        serializationIdx += sizeof(vt);

        // write body
        // single block
        DF_body b;
        b.rx = 0;
        b.cx = 0;

        // Check if we go out of limit
        if (chunkSize < bufferIdx + sizeof(b))
            return bufferIdx;
        if (serializeFromByte < serializationIdx + sizeof(b)) {
            std::copy(reinterpret_cast<const char*>(&b), reinterpret_cast<const char*>(&b) + sizeof(b), buffer + bufferIdx);
            bufferIdx += sizeof(b);
        }
        serializationIdx += sizeof(b);

        // block header
        DF_body_block bb;
        bb.nbrows = (uint32_t) arg->getNumRows();
        bb.nbcols = (uint32_t) arg->getNumCols();
        bb.bt = (uint8_t)DF_body_t::dense;
        
        // Check if we go out of limit
        if (chunkSize < bufferIdx + sizeof(bb))
            return bufferIdx;
        if (serializeFromByte < serializationIdx + sizeof(bb)) {
            std::copy(reinterpret_cast<const char*>(&bb), reinterpret_cast<const char*>(&bb) + sizeof(bb), buffer + bufferIdx);
            bufferIdx += sizeof(bb);
        }
        serializationIdx += sizeof(bb);

        // Check if we go out of limit
        if (chunkSize < bufferIdx + sizeof(vt))
            return bufferIdx;
        // value type
        if (serializeFromByte < serializationIdx + sizeof(vt)) {
            // memcpy((char *)buf + bufferIdx, (const char *) &vt, sizeof(vt));
            std::copy(reinterpret_cast<const char*>(&vt), reinterpret_cast<const char*>(&vt) + sizeof(vt), buffer + bufferIdx);
            bufferIdx += sizeof(vt);
        }
        serializationIdx += sizeof(vt);
        
        // block values
        const VT * valuesArg = arg->getValues();
            
        size_t bytesToCopy = 0;
        size_t valuesSize = arg->getNumRows() * arg->getNumCols() * sizeof(VT);

        if (serializeFromByte < serializationIdx) {
            bytesToCopy = (chunkSize > (serializationIdx - serializeFromByte) + valuesSize) ?
                valuesSize : (chunkSize - bufferIdx);
        } else {
            bytesToCopy = (serializeFromByte + chunkSize > valuesSize) ?
                (valuesSize + serializationIdx - serializeFromByte) : (chunkSize - bufferIdx);
        }

        size_t startOffset = (serializeFromByte > serializationIdx ? serializeFromByte - serializationIdx : 0);
        std::copy(reinterpret_cast<const char*>(valuesArg) + startOffset,
                    reinterpret_cast<const char*>(valuesArg) + startOffset + bytesToCopy,
                    buffer + bufferIdx);
        bufferIdx += bytesToCopy;

        return bufferIdx;
    };
    // Gets the address of a pointer buffer and if it is nullptr,
    // it allocates chunksize memory
    static size_t serialize(const DenseMatrix<VT> *arg, char **buffer, size_t serializeFromByte = 0, size_t chunkSize = 0) {        
        if (*buffer == nullptr) {
            chunkSize = chunkSize != 0 ? chunkSize : DaphneSerializer<DenseMatrix<VT>>::length(arg);
            *buffer = new char[chunkSize];
        }
        return serialize(arg, *buffer, serializeFromByte, chunkSize);
    }
    // Serializes into the vector<char> buffer. If capacity is less than chunksize, it reserves memory.
    static size_t serialize(const DenseMatrix<VT> *arg, std::vector<char> &buffer, size_t serializeFromByte = 0, size_t chunkSize = 0) {
        chunkSize = chunkSize != 0 ? chunkSize : DaphneSerializer<DenseMatrix<VT>>::length(arg);
        if (buffer.capacity() < chunkSize)
            buffer.reserve(chunkSize);
        return serialize(arg, buffer.data(), serializeFromByte, chunkSize);
    }

    static DenseMatrix<VT> *deserialize(const char *buf, size_t chunkSize, DenseMatrix<VT> *matrix = nullptr, size_t deserializeFromByte = 0) {
        if (deserializeFromByte == 0 && chunkSize < 60)
            throw std::runtime_error("Minimum starting chunk size 60 bytes"); // For now..?
        
        size_t bufIdx = 0;
        const size_t BUFFER_HEADER = 45;
        
        if (deserializeFromByte == 0) {
            assert((DF_Dtype(buf) == DF_data_t::DenseMatrix_t) && "DenseMatrix deserialize(): DT mismatch");
            assert((DF_Vtype(buf) == ValueTypeUtils::codeFor<VT>) && "DenseMatrix deserialize(): VT mismatch");
            // FF to the body
            bufIdx += sizeof(DF_header);
            bufIdx += sizeof(ValueTypeCode);
            bufIdx += sizeof(DF_body);
        
            DF_body_block bb;
            std::copy(buf + bufIdx, buf + bufIdx + sizeof(DF_body_block), reinterpret_cast<char*>(&bb));

            bufIdx += sizeof(DF_body_block);
            // empty Matrix
            if (bb.bt == (uint8_t)DF_body_t::empty) {
                return DataObjectFactory::create<DenseMatrix<VT>>(0, 0, false);
            }
            // Dense Matrix
            else if (bb.bt == (uint8_t)DF_body_t::dense) {     
                bufIdx += sizeof(ValueTypeCode);       
                size_t len = bb.nbrows * bb.nbcols * sizeof(VT);
                // Allocate if first chunk
                if (matrix == nullptr){
                    std::shared_ptr<VT[]> data(new VT[len]);
                    matrix = DataObjectFactory::create<DenseMatrix<VT>>((size_t)bb.nbrows,
                            (size_t)bb.nbcols, false);
                }
            } else {
                throw std::runtime_error("unknown body type code");
            }
        }
        auto valuesArg = matrix->getValues();
        
        size_t valuesOffset = deserializeFromByte == 0 ? 0 : deserializeFromByte - BUFFER_HEADER;
        std::copy(buf + bufIdx, buf + chunkSize, reinterpret_cast<char*>(valuesArg) + valuesOffset);
        
        return matrix;
    };        
    static DenseMatrix<VT> *deserialize(const std::vector<char> &buffer, DenseMatrix<VT> *matrix = nullptr, size_t deserializeFromByte = 0) {                        
        return deserialize(buffer.data(), buffer.capacity(), matrix, deserializeFromByte);
    }

};

template <typename VT>
struct DaphneSerializer<CSRMatrix<VT>, false> {
    const CSRMatrix<VT> *matrix;
    CSRMatrix<VT> **matrixPtr;
    size_t chunkSize;
    DaphneSerializer(const CSRMatrix<VT> *matrix, size_t chunkSize = 1024) : matrix(matrix), chunkSize(chunkSize) {
        if (chunkSize < 60)
            throw std::runtime_error("Minimum chunk size 60 bytes"); // For now..?
    };
    DaphneSerializer(CSRMatrix<VT> **matrix, size_t chunkSize = 1024) : matrixPtr(matrix), chunkSize(chunkSize) {
        if (chunkSize < 60)
            throw std::runtime_error("Minimum chunk size 60 bytes"); // For now..?
    };

    static size_t length(const CSRMatrix<VT> *arg) {
        size_t len = 0;

        // header
        len += sizeof(DF_header);
        // VT code
        len += sizeof(ValueTypeCode);
        // body
        len += sizeof(DF_body);
        // body block
        len += sizeof(DF_body_block);
        // VT code for block?
        len += sizeof(ValueTypeCode);
        // num non-zeros for the whole matrix
        len += sizeof(size_t);
        // rowOffsets
        len += ((arg->getNumRows() + 1) * sizeof(size_t));
        // colIdxs
        size_t nzb = 0;
        for (size_t r = 0; r < arg->getNumRows(); r++){
            nzb += arg->getNumNonZeros(r);            
        }
        len += (nzb * sizeof(size_t));
        // non-zero values
        len += (nzb * sizeof(VT));

        return len;
    };

   static size_t serialize(const CSRMatrix<VT> *arg, char *buffer, size_t serializeFromByte = 0, size_t chunkSize = 0) {
        size_t bufferIdx = 0;
        size_t serializationIdx = 0;

        if (buffer == nullptr){
            throw std::runtime_error("buffer is nullptr");
        }

        if (serializeFromByte == 0 && chunkSize < 60)
            throw std::runtime_error("Minimum starting chunk size 60 bytes"); // For now..?

        // write header
        DF_header h;
        h.version = 1;
        h.dt = (uint8_t)DF_data_t::CSRMatrix_t;
        h.nbrows = (uint64_t) arg->getNumRows();
        h.nbcols = (uint64_t) arg->getNumCols();

        if (serializeFromByte < serializationIdx + sizeof(h)) {
            std::copy(reinterpret_cast<const char*>(&h), reinterpret_cast<const char*>(&h) + sizeof(h), buffer);
            bufferIdx += sizeof(h);
        }
        serializationIdx += sizeof(h);

        // value type
        const ValueTypeCode vt = ValueTypeUtils::codeFor<VT>;
        // Check if we actually need this    
        if (serializeFromByte < serializationIdx + sizeof(vt)) {
            std::copy(reinterpret_cast<const char*>(&vt), reinterpret_cast<const char*>(&vt) + sizeof(vt), buffer + bufferIdx);
            bufferIdx += sizeof(vt);
        }
        serializationIdx += sizeof(vt);

        // write body
        // single block
        DF_body b;
        b.rx = 0;
        b.cx = 0;

        if (serializeFromByte < serializationIdx + sizeof(b)) {
            std::copy(reinterpret_cast<const char*>(&b), reinterpret_cast<const char*>(&b) + sizeof(b), buffer + bufferIdx);
            bufferIdx += sizeof(b);
        }
        serializationIdx += sizeof(b);

        // block header
        DF_body_block bb;
        bb.nbrows = (uint32_t) arg->getNumRows();
        bb.nbcols = (uint32_t) arg->getNumCols();
        bb.bt = (uint8_t)DF_body_t::sparse;

        if (serializeFromByte < serializationIdx + sizeof(bb)) {
            std::copy(reinterpret_cast<const char*>(&bb), reinterpret_cast<const char*>(&bb) + sizeof(bb), buffer + bufferIdx);
            bufferIdx += sizeof(bb);
        }
        serializationIdx += sizeof(bb);

        // value type
        if (serializeFromByte < serializationIdx + sizeof(vt)) {
            // memcpy((char *)buf + bufferIdx, (const char *) &vt, sizeof(vt));
            std::copy(reinterpret_cast<const char*>(&vt), reinterpret_cast<const char*>(&vt) + sizeof(vt), buffer + bufferIdx);
            bufferIdx += sizeof(vt);
        }
        serializationIdx += sizeof(vt);

        // num non-zeros
        size_t nzb = 0;
        for (size_t r = 0; r < arg->getNumRows(); r++){
            nzb += arg->getNumNonZeros(r);            
        }
        if (serializeFromByte < serializationIdx + sizeof(nzb)) {
            std::copy(reinterpret_cast<const char*>(&nzb), reinterpret_cast<const char*>(&nzb) + sizeof(nzb), buffer + bufferIdx);
            bufferIdx += sizeof(nzb);
        }
        serializationIdx += sizeof(nzb);

        const size_t * rowOffsets = arg->getRowOffsets();
        const size_t offset_diff = *arg->getRowOffsets();
        std::unique_ptr<size_t> new_rows(new size_t[arg->getNumRows() + 1]);
        for (size_t r = 0; r < arg->getNumRows() + 1; r++){
            auto newVal = *(rowOffsets + r) - offset_diff;                        
            new_rows.get()[r] = newVal;
        }
        if (serializeFromByte < serializationIdx + (arg->getNumRows() + 1) * sizeof(size_t)) {
            size_t startOffset = serializeFromByte > serializationIdx ? serializeFromByte - serializationIdx : 0;
            size_t bytesToCopy = 0;
            size_t arraySize = (arg->getNumRows() + 1) * sizeof(size_t);
            if (serializeFromByte < serializationIdx){
                bytesToCopy = chunkSize - bufferIdx > arraySize ? 
                        arraySize : 
                        chunkSize - bufferIdx;            
            } else {
                bytesToCopy = chunkSize > arraySize + serializationIdx - serializeFromByte ?
                            arraySize + serializationIdx - serializeFromByte:
                            chunkSize - bufferIdx;
            }

            std::copy(reinterpret_cast<const char*>(new_rows.get()) + startOffset, reinterpret_cast<const char*>(new_rows.get()) + startOffset + bytesToCopy, buffer + bufferIdx);
            bufferIdx += bytesToCopy;
        }
        serializationIdx += sizeof(size_t) * (arg->getNumRows() + 1);
        // Check if we go out of limit
        if (chunkSize <= bufferIdx)
            return bufferIdx;


        const size_t * colIdxs = arg->getColIdxs(0);
        if (serializeFromByte < serializationIdx + nzb * sizeof(size_t)){
            size_t startOffset = serializeFromByte > serializationIdx ? serializeFromByte - serializationIdx : 0;
            size_t bytesToCopy = 0;
            size_t arraySize = (nzb * sizeof(size_t));
            if (serializeFromByte < serializationIdx){
                bytesToCopy = chunkSize - bufferIdx > arraySize ? 
                        arraySize : 
                        chunkSize - bufferIdx;            
            } else {
                bytesToCopy = chunkSize > arraySize + serializationIdx - serializeFromByte ?
                            arraySize + serializationIdx - serializeFromByte:
                            chunkSize - bufferIdx;
            }
            std::copy(reinterpret_cast<const char*>(colIdxs) + startOffset, reinterpret_cast<const char*>(colIdxs) + startOffset + bytesToCopy, buffer + bufferIdx);
            bufferIdx += bytesToCopy;
        }
        serializationIdx += sizeof(size_t) * nzb;

        // Check if we go out of limit
        if (chunkSize <= bufferIdx)
            return bufferIdx;

        const VT * vals = arg->getValues(0);
        if (serializeFromByte < serializationIdx + nzb * sizeof(VT)){
            size_t startOffset = serializeFromByte > serializationIdx ? serializeFromByte - serializationIdx : 0;
            size_t bytesToCopy = 0;
            size_t arraySize = nzb * sizeof(VT);
            if (serializeFromByte < serializationIdx){
                bytesToCopy = chunkSize - bufferIdx > arraySize ? 
                        arraySize : 
                        chunkSize - bufferIdx;            
            } else {
                bytesToCopy = chunkSize > arraySize + serializationIdx - serializeFromByte ?
                            arraySize + serializationIdx - serializeFromByte:
                            chunkSize - bufferIdx;
            }
            std::copy(reinterpret_cast<const char*>(vals) + startOffset, reinterpret_cast<const char*>(vals) + startOffset + bytesToCopy, buffer + bufferIdx);
            bufferIdx += bytesToCopy;
        }

        return bufferIdx;
    };
    // Gets the address of a pointer buffer and if it is nullptr,
    // it allocates chunksize memory
    static size_t serialize(const CSRMatrix<VT> *arg, std::vector<char> &buffer, size_t serializeFromByte = 0, size_t chunkSize = 0) {
        chunkSize = chunkSize == 0 ? DaphneSerializer<CSRMatrix<VT>>::length(arg) : chunkSize;
        
        if (buffer.capacity() < chunkSize) // Maybe if is unecessary here..
            buffer.reserve(chunkSize);
        return serialize(arg, buffer.data(), serializeFromByte, chunkSize);
    }
    // Serializes into the vector<char> buffer. If capacity is less than chunksize, it reserves memory.
    static size_t serialize(const CSRMatrix<VT> *arg, char **buffer, size_t serializeFromByte = 0, size_t chunkSize = 0) {
        chunkSize = chunkSize == 0 ? DaphneSerializer<CSRMatrix<VT>>::length(arg) : chunkSize;
        
        if (*buffer == nullptr) // Maybe if is unecessary here..
            *buffer = new char[sizeof(chunkSize)];
        return serialize(arg, *buffer, serializeFromByte, chunkSize);
    }

    static CSRMatrix<VT> *deserialize(const char *buffer, size_t chunkSize, CSRMatrix<VT> * matrix = nullptr, size_t deserializeFromByte = 0) {            
        if (deserializeFromByte == 0 && chunkSize < 60)
            throw std::runtime_error("Minimum starting chunk size 60 bytes"); // For now..?

        size_t bufferIdx = 0;
        const size_t BUFFER_HEADER = 53;

        if (deserializeFromByte == 0) {

            assert((DF_Dtype(buffer) == DF_data_t::CSRMatrix_t) && "CSRMatrix deserialize(): DT mismatch");
            assert((DF_Vtype(buffer) == ValueTypeUtils::codeFor<VT>) && "CSRMatrix deserialize(): VT mismatch");

            // FF to the body
            bufferIdx += sizeof(DF_header);
            bufferIdx += sizeof(ValueTypeCode);
            bufferIdx += sizeof(DF_body);

            const DF_body_block *bb = (const DF_body_block *)(buffer + bufferIdx);
            bufferIdx += sizeof(DF_body_block);        
        
            // empty Matrix
            if (bb->bt == (uint8_t)DF_body_t::empty) {
                return DataObjectFactory::create<CSRMatrix<VT>>(0, 0, 0, false);
            // CSRMatrix
            } else if (bb->bt == (uint8_t)DF_body_t::sparse) {
                bufferIdx += sizeof(ValueTypeCode);

                size_t nzb;
                std::copy(buffer + bufferIdx, buffer + bufferIdx + sizeof(nzb), reinterpret_cast<char*>(&nzb));
                bufferIdx += sizeof(nzb);

                if (matrix == nullptr){
                    matrix = DataObjectFactory::create<CSRMatrix<VT>>(bb->nbrows, bb->nbcols, nzb, true);
                }
            // /* TODO MPI: No COO support for write? */
            // // COO Matrix
            // } else if (bb->bt == (uint8_t)DF_body_t::ultra_sparse) {
            //     bufferIdx += sizeof(ValueTypeCode);

            //     size_t nzb;
            //     std::copy(buffer.begin() + bufferIdx, buffer.begin() + bufferIdx + sizeof(nzb), reinterpret_cast<char*>(&nzb));
            //     bufferIdx += sizeof(nzb);

            //     if (matrix == nullptr)
            //         matrix = DataObjectFactory::create<CSRMatrix<VT>>(bb->nbrows, bb->nbcols, nzb, false);
            // }
            }
        }
        size_t serializationIdx = BUFFER_HEADER;
        
        if (deserializeFromByte < serializationIdx + sizeof(size_t) * (matrix->getNumRows() + 1)) {
            size_t * rowOffsets = matrix->getRowOffsets();
            size_t rowOffsets_offset = deserializeFromByte == 0 ? 0 : deserializeFromByte - serializationIdx;

            size_t bufferLen = 0;
            size_t arraySize = sizeof(size_t) * (matrix->getNumRows() + 1);
            if (deserializeFromByte < serializationIdx) 
                bufferLen = deserializeFromByte + chunkSize > serializationIdx + arraySize ? arraySize : chunkSize - bufferIdx;
            else
                bufferLen = deserializeFromByte + chunkSize > serializationIdx + arraySize ? arraySize + serializationIdx - deserializeFromByte : chunkSize - bufferIdx;

            std::copy(buffer + bufferIdx, buffer + bufferIdx + bufferLen, reinterpret_cast<char*>(rowOffsets) + rowOffsets_offset);            
            bufferIdx += bufferLen;
        }
        serializationIdx += sizeof(size_t) * (matrix->getNumRows() + 1);
        if (chunkSize <= bufferIdx)
            return matrix;

        size_t nzb = matrix->getNumNonZeros();
        if (deserializeFromByte < serializationIdx + sizeof(size_t) * nzb) {
            size_t * colIdxs = matrix->getColIdxs();
            size_t colIdxs_offset = deserializeFromByte < serializationIdx ? 0 : deserializeFromByte - serializationIdx; 

            size_t bufferLen = 0;
            size_t arraySize = sizeof(size_t) * nzb;
            if (deserializeFromByte < serializationIdx) 
                bufferLen = deserializeFromByte + chunkSize > serializationIdx + arraySize ? arraySize : chunkSize - bufferIdx;
            else
                bufferLen = deserializeFromByte + chunkSize > serializationIdx + arraySize ? arraySize + serializationIdx - deserializeFromByte : chunkSize - bufferIdx;

            std::copy(buffer + bufferIdx, buffer + bufferIdx + bufferLen, reinterpret_cast<char*>(colIdxs) + colIdxs_offset);
            bufferIdx += bufferLen;
        }
        serializationIdx += sizeof(size_t) * nzb;
        if (chunkSize <= bufferIdx)
            return matrix;

        if(deserializeFromByte < serializationIdx + sizeof(VT) * nzb) {
            VT * vals = matrix->getValues();
            size_t vals_offset = deserializeFromByte < serializationIdx ? 0 : deserializeFromByte - serializationIdx;
            
            size_t bufferLen = 0;
            size_t arraySize = sizeof(VT) * nzb;
            if (deserializeFromByte < serializationIdx) 
                bufferLen = deserializeFromByte + chunkSize > serializationIdx + arraySize ? arraySize : chunkSize - bufferIdx;
            else
                bufferLen = deserializeFromByte + chunkSize > serializationIdx + arraySize ? arraySize + serializationIdx - deserializeFromByte : chunkSize - bufferIdx;

            std::copy(buffer + bufferIdx, buffer + bufferIdx + bufferLen, reinterpret_cast<char*>(vals) + vals_offset);
        }
        
        return matrix;
        /* TODO */
        // COO Matrix
        /*
        } else if (bb->bt == (uint8_t)DF_body_t::ultra_sparse) {
            ibuf += sizeof(ValueTypeCode);

            size_t nzb;
            memcpy(&nzb, ibuf, sizeof(nzb));
            ibuf += sizeof(nzb);

            auto res = DataObjectFactory::create<CSRMatrix<VT>>(bb->nbrows, bb->nbcols, nzb, false);

            // Single column case
            if (bb->nbcols == 1) {
               for (uint64_t n = 0; n < nzb; n++) {
                    uint32_t i;
                    memcpy(&i, ibuf, sizeof(i));
                    ibuf += sizeof(nzb);

                    VT val;
                    memcpy(&val, ibuf, sizeof(val));
                    ibuf += sizeof(val);

                    res->set(i, 1, val);
                }
            } else {
                // TODO: check numcols is greater than 1
                for (uint64_t n = 0; n < nzb; n++) {
                    uint32_t i;
                    memcpy(&i, ibuf, sizeof(i));
                    ibuf += sizeof(i);

                    uint32_t j;
                    memcpy(&j, ibuf, sizeof(j));
                    ibuf += sizeof(j);

                    VT val;
                    memcpy(&val, ibuf, sizeof(val));
                    ibuf += sizeof(val);

                    res->set(i, j, val);
                }
            }
            return res;
        } else {
            throw std::runtime_error("unknown body type code");
        }
        */
   };
    static CSRMatrix<VT> *deserialize(const std::vector<char> &buffer, CSRMatrix<VT> * matrix = nullptr, size_t deserializeFromByte = 0) {
        return deserialize(buffer.data(), buffer.capacity(), matrix, deserializeFromByte);
    }
};

template <>
struct DaphneSerializer<Frame> {
    static size_t length(const Frame *arg) {
        throw std::runtime_error("not implemented");
    };

    static size_t serialize(const Frame *arg, char *buf, size_t serializeFromByte = 0, size_t chunkSize = 0) {
        throw std::runtime_error("not implemented");
    };
    static size_t serialize(const Frame *arg, std::vector<char> &buf, size_t serializeFromByte = 0, size_t chunkSize = 0) {
        return serialize(arg, buf.data(), serializeFromByte, chunkSize);
    }


   static Frame *deserialize(const void *buf) {
       throw std::runtime_error("not implemented");
   };
};

template<>
struct DaphneSerializer<Structure> {
    const Structure *matrix;
    Structure **matrixPtr;
    size_t chunkSize;
    DaphneSerializer(const Structure *matrix, size_t chunkSize = 1024) : matrix(matrix), chunkSize(chunkSize) {
        if (chunkSize < 60)
            throw std::runtime_error("Minimum chunk size 60 bytes"); // For now..?
    };
    DaphneSerializer(Structure **matrix, size_t chunkSize = 1024) : matrixPtr(matrix), chunkSize(chunkSize) {
        if (chunkSize < 60)
            throw std::runtime_error("Minimum chunk size 60 bytes"); // For now..?
    };

    static size_t length(const Structure *arg) {
        /* DenseMatrix */
        if (auto mat = dynamic_cast<const DenseMatrix<double>*>(arg))
            return DaphneSerializer<DenseMatrix<double>>::length(mat);
        if (auto mat = dynamic_cast<const DenseMatrix<float>*>(arg))
            return DaphneSerializer<DenseMatrix<float>>::length(mat);
        if (auto mat = dynamic_cast<const DenseMatrix<int8_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int8_t>>::length(mat);
        if (auto mat = dynamic_cast<const DenseMatrix<int32_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int32_t>>::length(mat);
        if (auto mat = dynamic_cast<const DenseMatrix<int64_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int64_t>>::length(mat);
        if (auto mat = dynamic_cast<const DenseMatrix<uint8_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint8_t>>::length(mat);
        if (auto mat = dynamic_cast<const DenseMatrix<uint32_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint32_t>>::length(mat);
        if (auto mat = dynamic_cast<const DenseMatrix<uint64_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint64_t>>::length(mat);
        /* CSRMatrix */
        if (auto mat = dynamic_cast<const CSRMatrix<double>*>(arg))
            return DaphneSerializer<CSRMatrix<double>>::length(mat);
        if (auto mat = dynamic_cast<const CSRMatrix<float>*>(arg))
            return DaphneSerializer<CSRMatrix<float>>::length(mat);
        if (auto mat = dynamic_cast<const CSRMatrix<int8_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int8_t>>::length(mat);
        if (auto mat = dynamic_cast<const CSRMatrix<int32_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int32_t>>::length(mat);
        if (auto mat = dynamic_cast<const CSRMatrix<int64_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int64_t>>::length(mat);
        if (auto mat = dynamic_cast<const CSRMatrix<uint8_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint8_t>>::length(mat);
        if (auto mat = dynamic_cast<const CSRMatrix<uint32_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint32_t>>::length(mat);
        if (auto mat = dynamic_cast<const CSRMatrix<uint64_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint64_t>>::length(mat);
        // else   
        throw std::runtime_error("Serialization length: uknown value type");
    };

    static size_t serialize(const Structure *arg, char *buf, size_t serializeFromByte = 0, size_t chunkSize = 0) {
        /* DenseMatrix */
        if (auto mat = dynamic_cast<const DenseMatrix<double>*>(arg))
            return DaphneSerializer<DenseMatrix<double>>::serialize(mat, buf, serializeFromByte, chunkSize);
        if (auto mat = dynamic_cast<const DenseMatrix<float>*>(arg))
            return DaphneSerializer<DenseMatrix<float>>::serialize(mat, buf, serializeFromByte, chunkSize);
        if (auto mat = dynamic_cast<const DenseMatrix<int8_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int8_t>>::serialize(mat, buf, serializeFromByte, chunkSize);
        if (auto mat = dynamic_cast<const DenseMatrix<int32_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int32_t>>::serialize(mat, buf, serializeFromByte, chunkSize);
        if (auto mat = dynamic_cast<const DenseMatrix<int64_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int64_t>>::serialize(mat, buf, serializeFromByte, chunkSize);
        if (auto mat = dynamic_cast<const DenseMatrix<uint8_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint8_t>>::serialize(mat, buf, serializeFromByte, chunkSize);
        if (auto mat = dynamic_cast<const DenseMatrix<uint32_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint32_t>>::serialize(mat, buf, serializeFromByte, chunkSize);
        if (auto mat = dynamic_cast<const DenseMatrix<uint64_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint64_t>>::serialize(mat, buf, serializeFromByte, chunkSize);
            
        /* CSRMatrix */
        if (auto mat = dynamic_cast<const CSRMatrix<double>*>(arg))
            return DaphneSerializer<CSRMatrix<double>>::serialize(mat, buf, serializeFromByte, chunkSize);
        if (auto mat = dynamic_cast<const CSRMatrix<float>*>(arg))
            return DaphneSerializer<CSRMatrix<float>>::serialize(mat, buf, serializeFromByte, chunkSize);
        if (auto mat = dynamic_cast<const CSRMatrix<int8_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int8_t>>::serialize(mat, buf, serializeFromByte, chunkSize);
        if (auto mat = dynamic_cast<const CSRMatrix<int32_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int32_t>>::serialize(mat, buf, serializeFromByte, chunkSize);
        if (auto mat = dynamic_cast<const CSRMatrix<int64_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int64_t>>::serialize(mat, buf, serializeFromByte, chunkSize);
        if (auto mat = dynamic_cast<const CSRMatrix<uint8_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint8_t>>::serialize(mat, buf, serializeFromByte, chunkSize);
        if (auto mat = dynamic_cast<const CSRMatrix<uint32_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint32_t>>::serialize(mat, buf, serializeFromByte, chunkSize);
        if (auto mat = dynamic_cast<const CSRMatrix<uint64_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint64_t>>::serialize(mat, buf, serializeFromByte, chunkSize);
        // else   
        throw std::runtime_error("Serialization serialize: uknown value type");
    };
    // Gets the address of a pointer buffer and if it is nullptr,
    // it allocates chunksize memory
    static size_t serialize(const Structure *arg, std::vector<char> &buffer, size_t serializeFromByte = 0, size_t chunkSize = 0) {
        chunkSize = chunkSize == 0 ? DaphneSerializer<Structure>::length(arg) : chunkSize;
        
        if (buffer.capacity() < chunkSize) // Maybe if is unecessary here..
            buffer.reserve(chunkSize);
        return serialize(arg, buffer.data(), serializeFromByte, chunkSize);
    }
    // Serializes into the vector<char> buffer. If capacity is less than chunksize, it reserves memory.
    static size_t serialize(const Structure *arg, char **buffer, size_t serializeFromByte = 0, size_t chunkSize = 0) {
        chunkSize = chunkSize == 0 ? DaphneSerializer<Structure>::length(arg) : chunkSize;
        
        if (*buffer == nullptr) // Maybe if is unecessary here..
            *buffer = new char[sizeof(chunkSize)];
        return serialize(arg, *buffer, serializeFromByte, chunkSize);
    }


    static Structure *deserialize(const void *buf) {
       throw std::runtime_error("not implemented");
    };
};

/* Partial specialization for fundumental types */
template<typename VT>
struct DaphneSerializer<VT, true> {
    static size_t length(const VT arg) {
        size_t len = 0;
        len += sizeof(DF_header);
        len += sizeof(ValueTypeCode);
        len += sizeof(VT);
        return len;
    };
    static size_t serialize(const VT &arg, char *buf) {
        if (buf == nullptr)
            throw std::runtime_error("buf is nullptr");

        size_t bufferIdx = 0;

        DF_header h;
        h.version = 1;
        h.dt = (uint8_t)DF_data_t::Value_t;
        
        std::copy(reinterpret_cast<char*>(&h), reinterpret_cast<char*>(&h) + sizeof(h), buf + bufferIdx);
        bufferIdx += sizeof(h);

        const ValueTypeCode vt = ValueTypeUtils::codeFor<VT>;
        std::copy(reinterpret_cast<const char*>(&vt), reinterpret_cast<const char*>(&vt) + sizeof(vt), buf + bufferIdx);
        bufferIdx += sizeof(vt);

        std::copy(reinterpret_cast<const char*>(&arg), reinterpret_cast<const char*>(&arg) + sizeof(VT), buf + bufferIdx);
        return length(arg);
    };
    // Gets the address of a pointer buffer and if it is nullptr,
    // it allocates chunksize memory
    static size_t serialize(const VT &arg, std::vector<char> &buf) {
        if (buf.capacity() < length(arg)) // Maybe if is unecessary here..
            buf.reserve(length(arg));
        return serialize(arg, buf.data());
    }
    // Serializes into the vector<char> buffer. If capacity is less than chunksize, it reserves memory.
    static size_t serialize(const VT &arg, char **buffer) {
        if (*buffer == nullptr)
            *buffer = new char(length(arg));
        return serialize(arg, *buffer);
    }
    
    static VT deserialize(const char *buf) {        
        
        size_t bufferIdx = 0;
        bufferIdx += sizeof(DF_header);
        bufferIdx += sizeof(ValueTypeCode);
        VT val;
        std::copy(buf + bufferIdx, buf + bufferIdx + sizeof(val), reinterpret_cast<char*>(&val));

        return val;
    };
    static VT deserialize(const std::vector<char> &buf) {        
        return deserialize(buf.data());
    }
};

inline Structure *DF_deserialize(const char *buf, size_t bufferSize) {
    if (DF_Dtype(buf) == DF_data_t::DenseMatrix_t) {
        switch(DF_Vtype(buf)) {
            case ValueTypeCode::SI8: return DaphneSerializer<DenseMatrix<int8_t>>::deserialize(buf, bufferSize); break;
            case ValueTypeCode::SI32: return DaphneSerializer<DenseMatrix<int32_t>>::deserialize(buf, bufferSize); break;
            case ValueTypeCode::SI64: return DaphneSerializer<DenseMatrix<int64_t>>::deserialize(buf, bufferSize); break;
            case ValueTypeCode::UI8: return DaphneSerializer<DenseMatrix<uint8_t>>::deserialize(buf, bufferSize); break;
            case ValueTypeCode::UI32: return DaphneSerializer<DenseMatrix<uint32_t>>::deserialize(buf, bufferSize); break;
            case ValueTypeCode::UI64: return DaphneSerializer<DenseMatrix<uint64_t>>::deserialize(buf, bufferSize); break;
            case ValueTypeCode::F32: return DaphneSerializer<DenseMatrix<float>>::deserialize(buf, bufferSize); break;
            case ValueTypeCode::F64: return DaphneSerializer<DenseMatrix<double>>::deserialize(buf, bufferSize); break;
            default: throw std::runtime_error("unknown value type code");
        }
    } else if (DF_Dtype(buf) == DF_data_t::CSRMatrix_t) {
        switch(DF_Vtype(buf)) {
            case ValueTypeCode::SI8: return DaphneSerializer<CSRMatrix<int8_t>>::deserialize(buf, bufferSize); break;
            case ValueTypeCode::SI32: return DaphneSerializer<CSRMatrix<int32_t>>::deserialize(buf, bufferSize); break;
            case ValueTypeCode::SI64: return DaphneSerializer<CSRMatrix<int64_t>>::deserialize(buf, bufferSize); break;
            case ValueTypeCode::UI8: return DaphneSerializer<CSRMatrix<uint8_t>>::deserialize(buf, bufferSize); break;
            case ValueTypeCode::UI32: return DaphneSerializer<CSRMatrix<uint32_t>>::deserialize(buf, bufferSize); break;
            case ValueTypeCode::UI64: return DaphneSerializer<CSRMatrix<uint64_t>>::deserialize(buf, bufferSize); break;
            case ValueTypeCode::F32: return DaphneSerializer<CSRMatrix<float>>::deserialize(buf, bufferSize); break;
            case ValueTypeCode::F64: return DaphneSerializer<CSRMatrix<double>>::deserialize(buf, bufferSize); break;
            default: throw std::runtime_error("unknown value type code");
        }
    } else {
        throw std::runtime_error("unknown value type code");
    }
}
inline Structure *DF_deserialize(const std::vector<char> &buf) {
    return DF_deserialize(buf.data(), buf.capacity());
}

/* Iterator, used for serializing in chunks. */
template <class DT>
struct DaphneSerializerChunks
{
    DT *matrix;
    size_t chunkSize;
    DaphneSerializerChunks(DT *matrix, size_t chunkSize = 1024) : matrix(matrix), chunkSize(chunkSize) {
        if (chunkSize < 60)
            throw std::runtime_error("Minimum chunk size 60 bytes"); // For now..?
    };
    struct Iterator {
        using iterator_category = std::input_iterator_tag;
        // using difference_type = // todo

        using buffer = std::vector<char>;
        using value_type = std::pair<size_t, buffer>; // TODO verify this
        using pointer = std::pair<size_t, buffer> *;
        using reference = std::pair<size_t, buffer> &; // TODO verify this
    private:
        const DT *matrix;
        size_t chunkSize;
        value_type serializedData;

    public:
        size_t numberOfBytesSerialized = 0;
        size_t index;

        // Constructors
        Iterator(){};
        Iterator(const DT *matrix, size_t chunkSize) : matrix(matrix), chunkSize(chunkSize), numberOfBytesSerialized(0), index(0)
        {
            if (chunkSize < 60)
                throw std::runtime_error("Minimum chunk size 60 bytes"); // For now..?
            serializedData.second.reserve(chunkSize);

            serializedData.first = DaphneSerializer<DT>::serialize(matrix, serializedData.second, numberOfBytesSerialized, chunkSize);
            numberOfBytesSerialized += serializedData.first;
        };

        reference operator*() { return serializedData; }
        pointer operator->() { return &serializedData; }

        // Prefix increment
        Iterator operator++()
        {
            index++;
            serializedData.first = DaphneSerializer<DT>::serialize(matrix, serializedData.second, numberOfBytesSerialized, chunkSize);
            numberOfBytesSerialized += serializedData.first;
            return *this;
        };
        // Postfix increment
        Iterator operator++(int)
        {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
            ;
        }

        friend bool operator==(const Iterator &a, const Iterator &b) { return a.index == b.index; };
        friend bool operator!=(const Iterator &a, const Iterator &b) { return a.index != b.index; };
    };
    Iterator begin() {
        return Iterator(matrix, chunkSize);
    }

    // Iterator end
    Iterator end() {
        Iterator iter;
        iter.index = DaphneSerializer<DT>::length(matrix) / chunkSize + 1;
        return iter;
    }
};

/* Iterator, used for deserializing in chunks. */
template <class DT>
struct DaphneDeserializerChunks
{
    DT **matrixPtr;
    size_t chunkSize;
    DaphneDeserializerChunks(DT **matrix, size_t chunkSize = 1024) : matrixPtr(matrix), chunkSize(chunkSize)
    {
        if (chunkSize < 60)
            throw std::runtime_error("Minimum chunk size 60 bytes"); // For now..?
    };
    struct Iterator
    {
        using iterator_category = std::output_iterator_tag;
        // using difference_type = // todo

        using buffer = std::vector<char>;
        using value_type = std::pair<size_t, buffer>; // TODO verify this
        using pointer = std::pair<size_t, buffer> *;
        using reference = std::pair<size_t, buffer> &; // TODO verify this
        DT **matrixPtr;

    private:
        size_t chunkSize;
        value_type serializedData;

    public:
        size_t numberOfBytesSerialized = 0;
        size_t index;

        // Constructors
        Iterator(){};
        Iterator(DT **matrix, size_t chunkSize) : matrixPtr(matrix), chunkSize(chunkSize), numberOfBytesSerialized(0), index(0) {
            if (chunkSize < 60)
                throw std::runtime_error("Minimum chunk size 60 bytes"); // For now..?
            serializedData.second.reserve(chunkSize);
        };

        reference operator*() { return serializedData; }
        pointer operator->() { return &serializedData; }

        // Prefix increment
        Iterator operator++() {
            index++;
            *matrixPtr = DaphneSerializer<DT>::deserialize(serializedData.second, *matrixPtr, numberOfBytesSerialized, serializedData.first);
            numberOfBytesSerialized += serializedData.first;
            return *this;
        }
        // Postfix increment
        Iterator operator++(int)
        {
            Iterator tmp = *this;
            ++(*this);
            return tmp;            
        }

        friend bool operator==(const Iterator &a, const Iterator &b) { return a.index == b.index; };
        friend bool operator!=(const Iterator &a, const Iterator &b) { return a.index != b.index; };
    };
    Iterator begin() {
        return Iterator(matrixPtr, chunkSize);
    }
    // Iterator end
    Iterator end() {
        Iterator iter;
        // If we just initialized the iterator is possible that we have no information about the deserialized object
        // but the buffer is also uninitialized. If matrix is uninitialized we simply set Iterator::end() as 1, otherwise
        // we calculate it based on matrix information.
        if (*matrixPtr != nullptr)
            iter.index = DaphneSerializer<DT>::length(*matrixPtr) / chunkSize + 1;
        else
            iter.index = 1;
        return iter;
    }
};