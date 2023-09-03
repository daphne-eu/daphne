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

// ****************************************************************************
// Helper functions
// ****************************************************************************

/**
 * @brief Deserializes a DF_data_t struct.
*/
inline DF_data_t DF_Dtype(const char *buf) { 
    return (DF_data_t)((const DF_header *)buf)->dt;
}
inline DF_data_t DF_Dtype(const std::vector<char>& buf) { return DF_Dtype(buf.data()); };

/**
 * @brief Deserializes the ValueTypeCode.
*/
inline ValueTypeCode DF_Vtype(const char *buf) {
    const ValueTypeCode *vt = (const ValueTypeCode *)((const char *)buf + sizeof(DF_header));
    return *vt;
}
inline ValueTypeCode DF_Vtype(const std::vector<char>& buf) { return DF_Vtype(buf.data()); }


// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
/**
 * @brief Serialize and deserialize Daphne objects.
*/
template <class DTArg, bool isFundumental = std::is_fundamental<DTArg>::value>
struct DaphneSerializer { 
    static size_t length(const DTArg *arg);
    static size_t serialize(const DTArg *arg, char *buf, size_t chunkSize = 0, size_t serializeFromByte = 0);
    static Structure *deserialize(const char *buf, size_t chunkSize = 0, DTArg *arg = nullptr, size_t deserializeFromByte = 0);
};


// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

/**
 * @brief Serialize and deserialize DenseMatrix data types.
 * 
 * Contains static methods for finding the length in bytes, serializing and
 * deserializing DenseMatrix objects.
*/
template <typename VT>
struct DaphneSerializer<DenseMatrix<VT>, false> {
    /**
     * @brief The default serialization chunk size
     */
    static const size_t DEFAULT_SERIALIZATION_BUFFER_SIZE = 1048576;
    // Size of the header
    static const size_t HEADER_BUFFER_SIZE = 45;
    /**
     * @brief Returns the size of the header.
    */
    static size_t headerSize(const DenseMatrix<VT> *arg) { return HEADER_BUFFER_SIZE; }

    /**
     * @brief Calculates the byte length of the object.
     * 
    */
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
    
    /**
     * @brief Creates a header and copies it to the buffer, containing information about the object (dimensions, types, other)
     * 
     * @param arg The object to be serialized.
     * @param buffer A pointer to copy the data.
     * @param bufferIdx (optional) A byte index for the buffer pointer.
    */
    static size_t serializeHeader(const DenseMatrix<VT> *arg, char *buffer, size_t bufferIdx = 0) {
        size_t serializationIdx = 0;

        if (buffer == nullptr)
            throw std::runtime_error("Buffer is nullptr");

        // write header
        DF_header h;
        h.version = 1;
        h.dt = (uint8_t)DF_data_t::DenseMatrix_t;
        h.nbrows = (uint64_t) arg->getNumRows();
        h.nbcols = (uint64_t) arg->getNumCols();
   
        std::copy(reinterpret_cast<const char*>(&h), reinterpret_cast<const char*>(&h) + sizeof(h), buffer);
        bufferIdx += sizeof(h);
        serializationIdx += sizeof(h);

        // value type
        const ValueTypeCode vt = ValueTypeUtils::codeFor<VT>;
            
        std::copy(reinterpret_cast<const char*>(&vt), reinterpret_cast<const char*>(&vt) + sizeof(vt), buffer + bufferIdx);
        bufferIdx += sizeof(vt);        
        serializationIdx += sizeof(vt);

        // write body
        // single block
        DF_body b;
        b.rx = 0;
        b.cx = 0;
        
        std::copy(reinterpret_cast<const char*>(&b), reinterpret_cast<const char*>(&b) + sizeof(b), buffer + bufferIdx);
        bufferIdx += sizeof(b);
        serializationIdx += sizeof(b);

        // block header
        DF_body_block bb;
        bb.nbrows = (uint32_t) arg->getNumRows();
        bb.nbcols = (uint32_t) arg->getNumCols();
        bb.bt = (uint8_t)DF_body_t::dense;
                
        std::copy(reinterpret_cast<const char*>(&bb), reinterpret_cast<const char*>(&bb) + sizeof(bb), buffer + bufferIdx);
        bufferIdx += sizeof(bb);
        serializationIdx += sizeof(bb);
        
        // value type        
        std::copy(reinterpret_cast<const char*>(&vt), reinterpret_cast<const char*>(&vt) + sizeof(vt), buffer + bufferIdx);
        bufferIdx += sizeof(vt);        
        serializationIdx += sizeof(vt);

        return serializationIdx;
    }

    /**
     * @brief Partially serializes an Daphne object into a buffer
     * 
     * @param arg The daphne Matrix
     * @param buffer A pointer to char, the buffer that data will be serialized to.
     * @param chunkSize Optional The size of the buffer (default is DEFAULT_SERIALIZATION_BUFFER_SIZE). Since at least one chunk will contain the header, the minimum chunk size should be HEADER_BUFFER_SIZE bytes (so the header won't be partially serialized).
     * @param serializeFromByte Optional The byte index of the object, at which serialization should begin (default 0). 
    */
    static size_t serialize(const DenseMatrix<VT> *arg, char *buffer, size_t chunkSize = DEFAULT_SERIALIZATION_BUFFER_SIZE, size_t serializeFromByte = 0) {
        size_t bufferIdx = 0;
        size_t serializationIdx = 0;
        chunkSize = chunkSize != 0 ? chunkSize : DaphneSerializer<DenseMatrix<VT>>::length(arg);

        if (buffer == nullptr){
            throw std::runtime_error("Buffer is nullptr");
        }

        // Since at least one chunk will contain the header, the minimum chunk size should be HEADER_BUFFER_SIZE bytes (so the header won't be partially serialized).
        if (serializeFromByte == 0 && chunkSize < HEADER_BUFFER_SIZE)
            throw std::runtime_error("Minimum starting chunk size " + std::to_string(HEADER_BUFFER_SIZE) + " bytes"); // For now..?

        if (serializeFromByte == 0) {
            auto bytesWritten = serializeHeader(arg, buffer);
            bufferIdx += bytesWritten;
        }
        serializationIdx += headerSize(arg);
        
        // block values
        const VT * valuesArg = arg->getValues();
            
        size_t bytesToCopy = 0;
        size_t valuesSize = arg->getNumRows() * arg->getNumCols() * sizeof(VT);

        if (serializeFromByte < serializationIdx) {
            bytesToCopy = (chunkSize > (serializationIdx - serializeFromByte) + valuesSize) ?
                valuesSize : (chunkSize - bufferIdx);
        } else {
            bytesToCopy = (serializeFromByte + chunkSize > valuesSize + serializationIdx) ?
                (valuesSize + serializationIdx - serializeFromByte) : (chunkSize - bufferIdx);
        }

        size_t startOffset = (serializeFromByte > serializationIdx ? serializeFromByte - serializationIdx : 0);
        std::copy(reinterpret_cast<const char*>(valuesArg) + startOffset,
                    reinterpret_cast<const char*>(valuesArg) + startOffset + bytesToCopy,
                    buffer + bufferIdx);
        bufferIdx += bytesToCopy;

        return bufferIdx;
    };
    /**
     * @brief Partially serializes an Daphne object into a buffer. This overloaded function can allocate memory for the buffer if needed.
     * 
     * @param arg The daphne Matrix
     * @param buffer A pointer to pointer to char, the buffer that data will be serialized to.
     * @param chunkSize Optional The size of the buffer (default is 0 - the size needed for the whole object)
     * @param serializeFromByte Optional The byte index of the object, at which serialization should begin (default 0). 
    */
    static size_t serialize(const DenseMatrix<VT> *arg, char **buffer, size_t chunkSize = 0, size_t serializeFromByte = 0) {        
        if (*buffer == nullptr) {
            chunkSize = chunkSize != 0 ? chunkSize : DaphneSerializer<DenseMatrix<VT>>::length(arg);
            *buffer = new char[chunkSize];
        }
        return serialize(arg, *buffer, chunkSize, serializeFromByte);
    }
    /**
     * @brief Serializes a Daphne object into a vector<char> buffer. 
     * 
     * The user can reserve memory for the buffer before the call. If the capacity of the buffer is 0,
     * memory is reserved for the whole object.
     * 
     * @param arg The daphne object.
     * @param buffer The std::vector<char> buffer.
     * @param serializeFromByte Optional The byte index at which serialization should start (default 0).
    */
    static size_t serialize(const DenseMatrix<VT> *arg, std::vector<char> &buffer, size_t serializeFromByte = 0) {
        // if caller provides an empty buffer, assume we want to serialize the whole object
        size_t chunkSize = buffer.capacity() == 0 ? DaphneSerializer<DenseMatrix<VT>>::length(arg) : buffer.capacity();
        if (buffer.capacity() == 0) 
            buffer.reserve(chunkSize);
        
        return serialize(arg, buffer.data(), chunkSize, serializeFromByte);
    }

    /**
     * @brief Deserializes the header of a buffer containing information about a DenseMatrix.
     * 
     * @param buf The buffer which contains the header.
     * @param matrix The DenseMatrix to initialize with the header information.
     * @return DenseMatrix<VT>* The result matrix.
     */
    static DenseMatrix<VT> *deserializeHeader(const char *buf, DenseMatrix<VT> *matrix = nullptr) {
        size_t bufIdx = 0;
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
        return matrix;
    }

    /**
     * @brief Deserializes a DenseMatrix from a buffer.
     * 
     * Deserialization can be done partially by specifing an byte-index as a starting point in the Matrix. 
     * Notice that index is related to the byte length of the matrix (provided by length(matrix)).
     * 
     * @param buf The buffer containing the serialized data.
     * @param chunkSize The size of the buffer. Since at least one chunk will contain the header, the minimum chunk size should be HEADER_BUFFER_SIZE bytes (so the header won't be partially serialized).
     * @param matrix The result matrix to write data.
     * @param deserializeFromByte (Optional) The index of the @matrix that deserialization should begin writing data.
     * @return DenseMatrix<VT>* The result matrix.
     */
    static DenseMatrix<VT> *deserialize(const char *buf, size_t chunkSize, DenseMatrix<VT> *matrix = nullptr, size_t deserializeFromByte = 0) {
        // Since at least one chunk will contain the header, the minimum chunk size should be HEADER_BUFFER_SIZE bytes (so the header won't be partially serialized).
        if (deserializeFromByte == 0 && chunkSize < HEADER_BUFFER_SIZE)
            throw std::runtime_error("Minimum starting chunk size " + std::to_string(HEADER_BUFFER_SIZE) + " bytes"); // For now..?
        
        size_t bufIdx = 0;     
        size_t serializationIdx = 0;   
        
        if (deserializeFromByte == 0) {
            matrix = deserializeHeader(buf, matrix);
            bufIdx += HEADER_BUFFER_SIZE;
        }
        serializationIdx += HEADER_BUFFER_SIZE;
        
        auto valuesArg = matrix->getValues();
        size_t bytesToCopy = 0;
        size_t valuesSize = matrix->getNumRows() * matrix->getNumCols() * sizeof(VT);

        size_t valuesOffset = deserializeFromByte == 0 ? 0 : deserializeFromByte - HEADER_BUFFER_SIZE;

        if (deserializeFromByte < serializationIdx) {
            bytesToCopy = (chunkSize > (serializationIdx - deserializeFromByte) + valuesSize) ?
                valuesSize : (chunkSize - bufIdx);
        } else {
            bytesToCopy = (deserializeFromByte + chunkSize > valuesSize + serializationIdx) ?
                (valuesSize + serializationIdx - deserializeFromByte) : (chunkSize - bufIdx);
        }

        std::copy(buf + bufIdx, buf + bufIdx + bytesToCopy, reinterpret_cast<char*>(valuesArg) + valuesOffset);
        
        return matrix;
    };
    /**
     * @brief Deserializes a DenseMatrix from a buffer.
     * 
     * Deserialization can be done partially by specifing an byte-index as a starting point in the Matrix. 
     * Notice that index is related to the byte length of the matrix (provided by length(matrix)).
     * 
     * @param buffer An std::vector<char> buffer containg serialized data.
     * @param matrix The result matrix to write data.
     * @param deserializeFromByte (Optional) The index of the @matrix that deserialization should begin writing data.
     * @return DenseMatrix<VT>* The result matrix.
     */
    static DenseMatrix<VT> *deserialize(const std::vector<char> &buffer, DenseMatrix<VT> *matrix = nullptr, size_t deserializeFromByte = 0) {                        
        return deserialize(buffer.data(), buffer.capacity(), matrix, deserializeFromByte);
    }

};

// ----------------------------------------------------------------------------
// const DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct DaphneSerializer<const DenseMatrix<VT>, false> : public DaphneSerializer<DenseMatrix<VT>, false> { };

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------
/**
 * @brief Serialize and deserialize CSRMatrix data types.
 * 
 * Contains static methods for finding the length in bytes, serializing and
 * deserializing CSRMatrix objects.
*/
template <typename VT>
struct DaphneSerializer<CSRMatrix<VT>, false> {
    const CSRMatrix<VT> *matrix;
    CSRMatrix<VT> **matrixPtr;
    size_t chunkSize;
    /**
     * @brief The default serialization chunk size
     */
    static const size_t DEFAULT_SERIALIZATION_BUFFER_SIZE = 1048576;
    // Size of the header
    static const size_t HEADER_BUFFER_SIZE = 53;
    /**
     * @brief Returns the size of the header.
    */
    static size_t headerSize(const CSRMatrix<VT> *arg) { return HEADER_BUFFER_SIZE; }
    
    DaphneSerializer(const CSRMatrix<VT> *matrix, size_t chunkSize = DEFAULT_SERIALIZATION_BUFFER_SIZE) : matrix(matrix), chunkSize(chunkSize) {
        // Since at least one chunk will contain the header, the minimum chunk size should be HEADER_BUFFER_SIZE bytes (so the header won't be partially serialized).
        if (chunkSize < HEADER_BUFFER_SIZE)
            throw std::runtime_error("Minimum chunk size " + std::to_string(HEADER_BUFFER_SIZE) + " bytes"); // For now..?
    };
    DaphneSerializer(CSRMatrix<VT> **matrix, size_t chunkSize = DEFAULT_SERIALIZATION_BUFFER_SIZE) : matrixPtr(matrix), chunkSize(chunkSize) {
        // Since at least one chunk will contain the header, the minimum chunk size should be HEADER_BUFFER_SIZE bytes (so the header won't be partially serialized).
        if (chunkSize < HEADER_BUFFER_SIZE)
            throw std::runtime_error("Minimum chunk size " + std::to_string(HEADER_BUFFER_SIZE) + " bytes"); // For now..?
    };
    /**
     * @brief Calculates the byte length of the object.
     * 
    */
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
        // size_t nzb = 0;
        // for (size_t r = 0; r < arg->getNumRows(); r++){
        //     nzb += arg->getNumNonZeros(r);            
        // }
        size_t nzb = arg->getMaxNumNonZeros();
        len += (nzb * sizeof(size_t));
        // non-zero values
        len += (nzb * sizeof(VT));

        return len;
    };
    /**
     * @brief Creates a header and copies it to the buffer, containing information about the object (dimensions, types, other)
     * 
     * @param arg The object to be serialized.
     * @param buffer A pointer to copy the data.
     * @param bufferIdx (optional) A byte index for the buffer pointer.
    */
    static size_t serializeHeader(const CSRMatrix<VT> *arg, char *buffer, size_t bufferIdx = 0) {
        size_t serializationIdx = 0;

        if (buffer == nullptr){
            throw std::runtime_error("buffer is nullptr");
        }

        // write header
        DF_header h;
        h.version = 1;
        h.dt = (uint8_t)DF_data_t::CSRMatrix_t;
        h.nbrows = (uint64_t) arg->getNumRows();
        h.nbcols = (uint64_t) arg->getNumCols();

        std::copy(reinterpret_cast<const char*>(&h), reinterpret_cast<const char*>(&h) + sizeof(h), buffer);
        bufferIdx += sizeof(h);
        serializationIdx += sizeof(h);

        // value type
        const ValueTypeCode vt = ValueTypeUtils::codeFor<VT>;
        // Check if we actually need this    
        std::copy(reinterpret_cast<const char*>(&vt), reinterpret_cast<const char*>(&vt) + sizeof(vt), buffer + bufferIdx);
        bufferIdx += sizeof(vt);        
        serializationIdx += sizeof(vt);

        // write body
        // single block
        DF_body b;
        b.rx = 0;
        b.cx = 0;

        std::copy(reinterpret_cast<const char*>(&b), reinterpret_cast<const char*>(&b) + sizeof(b), buffer + bufferIdx);
        bufferIdx += sizeof(b);
        serializationIdx += sizeof(b);

        // block header
        DF_body_block bb;
        bb.nbrows = (uint32_t) arg->getNumRows();
        bb.nbcols = (uint32_t) arg->getNumCols();
        bb.bt = (uint8_t)DF_body_t::sparse;

        std::copy(reinterpret_cast<const char*>(&bb), reinterpret_cast<const char*>(&bb) + sizeof(bb), buffer + bufferIdx);
        bufferIdx += sizeof(bb);
        serializationIdx += sizeof(bb);

        // value type
        std::copy(reinterpret_cast<const char*>(&vt), reinterpret_cast<const char*>(&vt) + sizeof(vt), buffer + bufferIdx);
        bufferIdx += sizeof(vt);
        serializationIdx += sizeof(vt);

        // num non-zeros
        size_t nzb = 0;
        for (size_t r = 0; r < arg->getNumRows(); r++){
            nzb += arg->getNumNonZeros(r);            
        }
        std::copy(reinterpret_cast<const char*>(&nzb), reinterpret_cast<const char*>(&nzb) + sizeof(nzb), buffer + bufferIdx);
        bufferIdx += sizeof(nzb);
        serializationIdx += sizeof(nzb);
        return serializationIdx;
    }
    /**
     * @brief Partially serializes an Daphne object into a buffer
     * 
     * @param arg The daphne Matrix
     * @param buffer A pointer to char, the buffer that data will be serialized to.
     * @param chunkSize Optional The size of the buffer (default is DEFAULT_SERIALIZATION_BUFFER_SIZE). Since at least one chunk will contain the header, the minimum chunk size should be HEADER_BUFFER_SIZE bytes (so the header won't be partially serialized).
     * @param serializeFromByte Optional The byte index of the object, at which serialization should begin (default 0). 
    */
    static size_t serialize(const CSRMatrix<VT> *arg, char *buffer, size_t chunkSize = DEFAULT_SERIALIZATION_BUFFER_SIZE, size_t serializeFromByte = 0) {
        size_t bufferIdx = 0;
        size_t serializationIdx = 0;
        chunkSize = chunkSize != 0 ? chunkSize : DaphneSerializer<CSRMatrix<VT>>::length(arg);

        // Since at least one chunk will contain the header, the minimum chunk size should be HEADER_BUFFER_SIZE bytes (so the header won't be partially serialized).
        if (chunkSize < HEADER_BUFFER_SIZE)
            throw std::runtime_error("Minimum chunk size " + std::to_string(HEADER_BUFFER_SIZE) + " bytes"); // For now..?

        if (serializeFromByte == 0) {
            auto bytesWritten = serializeHeader(arg, buffer);            
            bufferIdx += bytesWritten;
        }
        serializationIdx += headerSize(arg);

        // num non-zeros
        size_t nzb = 0;
        for (size_t r = 0; r < arg->getNumRows(); r++){
            nzb += arg->getNumNonZeros(r);            
        }

        const size_t * rowOffsets = arg->getRowOffsets();
        const size_t offset_diff = *arg->getRowOffsets();
        auto new_rows = std::make_unique<size_t[]>(arg->getNumRows() + 1);
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
    /**
     * @brief Partially serializes an Daphne object into a buffer. This overloaded function can allocate memory for the buffer if needed.
     * 
     * @param arg The daphne Matrix
     * @param buffer A pointer to pointer to char, the buffer that data will be serialized to.
     * @param chunkSize Optional The size of the buffer (default is 0 - the size needed for the whole object)
     * @param serializeFromByte Optional The byte index of the object, at which serialization should begin (default 0). 
    */
    static size_t serialize(const CSRMatrix<VT> *arg, char **buffer, size_t chunkSize = 0, size_t serializeFromByte = 0) {
        chunkSize = chunkSize == 0 ? DaphneSerializer<CSRMatrix<VT>>::length(arg) : chunkSize;
        
        if (*buffer == nullptr) // Maybe if is unecessary here..
            *buffer = new char[sizeof(chunkSize)];
        return serialize(arg, *buffer, chunkSize, serializeFromByte);
    }
    /**
     * @brief Partially serializes an Daphne object into a buffer. This overloaded function can allocate memory for the buffer if needed.
     * 
     * @param arg The daphne Matrix
     * @param buffer A pointer to pointer to char, the buffer that data will be serialized to.
     * @param chunkSize Optional The size of the buffer (default is DEFAULT_SERIALIZATION_BUFFER_SIZE)
     * @param serializeFromByte Optional The byte index of the object, at which serialization should begin. 
    */
    static size_t serialize(const CSRMatrix<VT> *arg, std::vector<char> &buffer, size_t serializeFromByte = 0) {
        // if caller provides an empty buffer, assume we want to serialize the whole object
        size_t chunkSize = buffer.capacity() == 0 ? DaphneSerializer<CSRMatrix<VT>>::length(arg) : buffer.capacity();
        if (buffer.capacity() < chunkSize) // Maybe if is unecessary here..
            buffer.reserve(chunkSize);

        return serialize(arg, buffer.data(), chunkSize, serializeFromByte);
    }
    /**
     * @brief Deserializes the header of a buffer containing information about a CSRMatrix.
     * 
     * @param buf The buffer which contains the header.
     * @param matrix The CSRMatrix to initialize with the header information.
     * @return CSRMatrix<VT>* The result matrix.
     */
    static CSRMatrix<VT> *deserializeHeader(const char *buffer, CSRMatrix<VT> *matrix = nullptr) {
        assert((DF_Dtype(buffer) == DF_data_t::CSRMatrix_t) && "CSRMatrix deserialize(): DT mismatch");
        assert((DF_Vtype(buffer) == ValueTypeUtils::codeFor<VT>) && "CSRMatrix deserialize(): VT mismatch");

        size_t bufferIdx = 0;
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
        return matrix;
    }
    /**
     * @brief Deserializes a CSRMatrix from a buffer.
     * 
     * Deserialization can be done partially by specifing an byte-index as a starting point in the Matrix. 
     * Notice that index is related to the byte length of the matrix (provided by length(matrix)).
     * 
     * @param buf The buffer containing the serialized data.
     * @param chunkSize The size of the buffer.
     * @param matrix The result matrix to write data.
     * @param deserializeFromByte (Optional) The index of the @matrix that deserialization should begin writing data.
     * @return CSRMatrix<VT>* The result matrix.
     */
    static CSRMatrix<VT> *deserialize(const char *buffer, size_t chunkSize, CSRMatrix<VT> * matrix = nullptr, size_t deserializeFromByte = 0) {            
        // Since at least one chunk will contain the header, the minimum chunk size should be HEADER_BUFFER_SIZE bytes (so the header won't be partially serialized).
        if (deserializeFromByte == 0 && chunkSize < HEADER_BUFFER_SIZE)
            throw std::runtime_error("Minimum starting chunk size " + std::to_string(HEADER_BUFFER_SIZE) + " bytes"); // For now..?

        size_t bufferIdx = 0;
        size_t serializationIdx = 0;

        if (deserializeFromByte == 0) {
            matrix = deserializeHeader(buffer, matrix);
            bufferIdx += HEADER_BUFFER_SIZE;
        }
        serializationIdx += HEADER_BUFFER_SIZE;
        
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

        size_t nzb = matrix->getMaxNumNonZeros();
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
   /**
     * @brief Deserializes a CSRMatrix from a buffer.
     * 
     * Deserialization can be done partially by specifing an byte-index as a starting point in the Matrix. 
     * Notice that index is related to the byte length of the matrix (provided by length(matrix)).
     * 
     * @param buffer An std::vector<char> buffer containg serialized data.
     * @param matrix The result matrix to write data.
     * @param deserializeFromByte (Optional) The index of the @matrix that deserialization should begin writing data.
     * @return CSRMatrix<VT>* The result matrix.
     */
    static CSRMatrix<VT> *deserialize(const std::vector<char> &buffer, CSRMatrix<VT> * matrix = nullptr, size_t deserializeFromByte = 0) {
        return deserialize(buffer.data(), buffer.capacity(), matrix, deserializeFromByte);
    }
};

// ----------------------------------------------------------------------------
// const CSRMatrix
// ----------------------------------------------------------------------------
template<typename VT>
struct DaphneSerializer<const CSRMatrix<VT>, false> : public DaphneSerializer<CSRMatrix<VT>, false> { };
// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

/**
 * @brief Serialize and deserialize Frame data types.
 * 
 * Contains static methods for finding the length in bytes, serializing and
 * deserializing Frame objects.
*/
template <>
struct DaphneSerializer<Frame> {
    static size_t length(const Frame *arg) {
        throw std::runtime_error("not implemented");
    };

    static size_t serialize(const Frame *arg, char *buf, size_t chunkSize = 0, size_t serializeFromByte = 0) {
        throw std::runtime_error("not implemented");
    };
    static size_t serialize(const Frame *arg, std::vector<char> &buf, size_t chunkSize = 0, size_t serializeFromByte = 0) {
        return serialize(arg, buf.data(), chunkSize, serializeFromByte);
    }


   static Frame *deserialize(const char *buf) {
       throw std::runtime_error("not implemented");
   };
};

// ----------------------------------------------------------------------------
// Structure
// ----------------------------------------------------------------------------
/**
 * @brief Serialize and deserialize Structure types.
 * 
 * Uses dynamic casts to downcast the Structure to a specific Daphne object (Dense, CSR, Frame) and uses the appropriate templated class.
*/
template<>
struct DaphneSerializer<Structure> {
    /**
     * @brief The default serialization chunk size
     */
    static const size_t DEFAULT_SERIALIZATION_BUFFER_SIZE = 1048576;

    // Since at least one chunk will contain the header and we do not know the specific type of the object (Dense, CSR, etc.),
    // the minimum chunk size should be the maximum possible header size in bytes (so the header won't be partially serialized).    
    // TODO Other types?
    static const size_t HEADER_BUFFER_SIZE = std::max(
        DaphneSerializer<DenseMatrix<double>>::HEADER_BUFFER_SIZE,
        DaphneSerializer<CSRMatrix<double>>::HEADER_BUFFER_SIZE
    );
    
    const Structure *obj;
    Structure **objPtr;
    size_t chunkSize;
    DaphneSerializer(const Structure *obj, size_t chunkSize = DEFAULT_SERIALIZATION_BUFFER_SIZE) : obj(obj), chunkSize(chunkSize) {
        if (chunkSize < HEADER_BUFFER_SIZE)
            throw std::runtime_error("Minimum chunk size " + std::to_string(HEADER_BUFFER_SIZE) + " bytes"); // For now..?
    };
    DaphneSerializer(Structure **obj, size_t chunkSize = DEFAULT_SERIALIZATION_BUFFER_SIZE) : objPtr(obj), chunkSize(chunkSize) {
        if (chunkSize < HEADER_BUFFER_SIZE)
            throw std::runtime_error("Minimum chunk size " + std::to_string(HEADER_BUFFER_SIZE) + " bytes"); // For now..?
    };
    // TODO Use macros for dynamic_casts

    /**
     * @brief Returns the size of the header.
    */
    static size_t headerSize(const Structure *arg) { 
        /* DenseMatrix */
        if (auto mat = dynamic_cast<const DenseMatrix<double>*>(arg))
            return DaphneSerializer<DenseMatrix<double>>::headerSize(mat);
        if (auto mat = dynamic_cast<const DenseMatrix<float>*>(arg))
            return DaphneSerializer<DenseMatrix<float>>::headerSize(mat);
        if (auto mat = dynamic_cast<const DenseMatrix<int8_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int8_t>>::headerSize(mat);
        if (auto mat = dynamic_cast<const DenseMatrix<int32_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int32_t>>::headerSize(mat);
        if (auto mat = dynamic_cast<const DenseMatrix<int64_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int64_t>>::headerSize(mat);
        if (auto mat = dynamic_cast<const DenseMatrix<uint8_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint8_t>>::headerSize(mat);
        if (auto mat = dynamic_cast<const DenseMatrix<uint32_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint32_t>>::headerSize(mat);
        if (auto mat = dynamic_cast<const DenseMatrix<uint64_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint64_t>>::headerSize(mat);
        /* CSRMatrix */
        if (auto mat = dynamic_cast<const CSRMatrix<double>*>(arg))
            return DaphneSerializer<CSRMatrix<double>>::headerSize(mat);
        if (auto mat = dynamic_cast<const CSRMatrix<float>*>(arg))
            return DaphneSerializer<CSRMatrix<float>>::headerSize(mat);
        if (auto mat = dynamic_cast<const CSRMatrix<int8_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int8_t>>::headerSize(mat);
        if (auto mat = dynamic_cast<const CSRMatrix<int32_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int32_t>>::headerSize(mat);
        if (auto mat = dynamic_cast<const CSRMatrix<int64_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int64_t>>::headerSize(mat);
        if (auto mat = dynamic_cast<const CSRMatrix<uint8_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint8_t>>::headerSize(mat);
        if (auto mat = dynamic_cast<const CSRMatrix<uint32_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint32_t>>::headerSize(mat);
        if (auto mat = dynamic_cast<const CSRMatrix<uint64_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint64_t>>::headerSize(mat);
        // else   
        throw std::runtime_error("Serialization headerSize: uknown value type");
    };
    /**
     * @brief Calculates the byte length of the object.
     * 
     * @param arg A structure to calculate the length.
     * @return size_t The byte length of the object.
     */
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
    /**
     * @brief Serializes a header.
    */
    static size_t serializeHeader(const Structure *arg, char *buffer) { 
        /* DenseMatrix */
        if (auto mat = dynamic_cast<const DenseMatrix<double>*>(arg))
            return DaphneSerializer<DenseMatrix<double>>::serializeHeader(mat, buffer);
        if (auto mat = dynamic_cast<const DenseMatrix<float>*>(arg))
            return DaphneSerializer<DenseMatrix<float>>::serializeHeader(mat, buffer);
        if (auto mat = dynamic_cast<const DenseMatrix<int8_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int8_t>>::serializeHeader(mat, buffer);
        if (auto mat = dynamic_cast<const DenseMatrix<int32_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int32_t>>::serializeHeader(mat, buffer);
        if (auto mat = dynamic_cast<const DenseMatrix<int64_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int64_t>>::serializeHeader(mat, buffer);
        if (auto mat = dynamic_cast<const DenseMatrix<uint8_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint8_t>>::serializeHeader(mat, buffer);
        if (auto mat = dynamic_cast<const DenseMatrix<uint32_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint32_t>>::serializeHeader(mat, buffer);
        if (auto mat = dynamic_cast<const DenseMatrix<uint64_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint64_t>>::serializeHeader(mat, buffer);
        /* CSRMatrix */
        if (auto mat = dynamic_cast<const CSRMatrix<double>*>(arg))
            return DaphneSerializer<CSRMatrix<double>>::serializeHeader(mat, buffer);
        if (auto mat = dynamic_cast<const CSRMatrix<float>*>(arg))
            return DaphneSerializer<CSRMatrix<float>>::serializeHeader(mat, buffer);
        if (auto mat = dynamic_cast<const CSRMatrix<int8_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int8_t>>::serializeHeader(mat, buffer);
        if (auto mat = dynamic_cast<const CSRMatrix<int32_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int32_t>>::serializeHeader(mat, buffer);
        if (auto mat = dynamic_cast<const CSRMatrix<int64_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int64_t>>::serializeHeader(mat, buffer);
        if (auto mat = dynamic_cast<const CSRMatrix<uint8_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint8_t>>::serializeHeader(mat, buffer);
        if (auto mat = dynamic_cast<const CSRMatrix<uint32_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint32_t>>::serializeHeader(mat, buffer);
        if (auto mat = dynamic_cast<const CSRMatrix<uint64_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint64_t>>::serializeHeader(mat, buffer);
        // else   
        throw std::runtime_error("Serialization serializeHeader: uknown value type");
    };
    /**
     * @brief Serializes a structure object to a buffer.
     * 
     * @param arg A pointer to the object.
     * @param buf The buffer to serialize data.
     * @param chunkSize (Optional) The size of the buffer, default is DEFAULT_SERIALIZATION_BUFFER_SIZE
     * @param serializeFromByte Optional The byte index of the object, at which serialization should begin (default 0)
     * @return size_t 
     */
    static size_t serialize(const Structure *arg, char *buf, size_t chunkSize = 0, size_t serializeFromByte = 0) {
        /* DenseMatrix */
        if (auto mat = dynamic_cast<const DenseMatrix<double>*>(arg))
            return DaphneSerializer<DenseMatrix<double>>::serialize(mat, buf, chunkSize, serializeFromByte);
        if (auto mat = dynamic_cast<const DenseMatrix<float>*>(arg))
            return DaphneSerializer<DenseMatrix<float>>::serialize(mat, buf, chunkSize, serializeFromByte);
        if (auto mat = dynamic_cast<const DenseMatrix<int8_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int8_t>>::serialize(mat, buf, chunkSize, serializeFromByte);
        if (auto mat = dynamic_cast<const DenseMatrix<int32_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int32_t>>::serialize(mat, buf, chunkSize, serializeFromByte);
        if (auto mat = dynamic_cast<const DenseMatrix<int64_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int64_t>>::serialize(mat, buf, chunkSize, serializeFromByte);
        if (auto mat = dynamic_cast<const DenseMatrix<uint8_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint8_t>>::serialize(mat, buf, chunkSize, serializeFromByte);
        if (auto mat = dynamic_cast<const DenseMatrix<uint32_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint32_t>>::serialize(mat, buf, chunkSize, serializeFromByte);
        if (auto mat = dynamic_cast<const DenseMatrix<uint64_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint64_t>>::serialize(mat, buf, chunkSize, serializeFromByte);
            
        /* CSRMatrix */
        if (auto mat = dynamic_cast<const CSRMatrix<double>*>(arg))
            return DaphneSerializer<CSRMatrix<double>>::serialize(mat, buf, chunkSize, serializeFromByte);
        if (auto mat = dynamic_cast<const CSRMatrix<float>*>(arg))
            return DaphneSerializer<CSRMatrix<float>>::serialize(mat, buf, chunkSize, serializeFromByte);
        if (auto mat = dynamic_cast<const CSRMatrix<int8_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int8_t>>::serialize(mat, buf, chunkSize, serializeFromByte);
        if (auto mat = dynamic_cast<const CSRMatrix<int32_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int32_t>>::serialize(mat, buf, chunkSize, serializeFromByte);
        if (auto mat = dynamic_cast<const CSRMatrix<int64_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int64_t>>::serialize(mat, buf, chunkSize, serializeFromByte);
        if (auto mat = dynamic_cast<const CSRMatrix<uint8_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint8_t>>::serialize(mat, buf, chunkSize, serializeFromByte);
        if (auto mat = dynamic_cast<const CSRMatrix<uint32_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint32_t>>::serialize(mat, buf, chunkSize, serializeFromByte);
        if (auto mat = dynamic_cast<const CSRMatrix<uint64_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint64_t>>::serialize(mat, buf, chunkSize, serializeFromByte);
        // else   
        throw std::runtime_error("Serialization serialize: uknown value type");
    };
    // Gets the address of a pointer buffer and if it is nullptr,
    // it allocates chunksize memory
    static size_t serialize(const Structure *arg, std::vector<char> &buffer, size_t chunkSize = 0, size_t serializeFromByte = 0) {
        chunkSize = chunkSize == 0 ? DaphneSerializer<Structure>::length(arg) : chunkSize;
        
        if (buffer.capacity() < chunkSize) // Maybe if is unecessary here..
            buffer.reserve(chunkSize);
        return serialize(arg, buffer.data(), chunkSize, serializeFromByte);
    }
    // Serializes into the vector<char> buffer. If capacity is less than chunksize, it reserves memory.
    static size_t serialize(const Structure *arg, char **buffer,  size_t chunkSize = 0, size_t serializeFromByte = 0) {
        chunkSize = chunkSize == 0 ? DaphneSerializer<Structure>::length(arg) : chunkSize;
        
        if (*buffer == nullptr)
            *buffer = new char[sizeof(chunkSize)];
        return serialize(arg, *buffer, chunkSize, serializeFromByte);
    }

    /**
     * @brief Deserializes a header.
    */
    static Structure *deserializeHeader(const char * buffer, Structure *arg) { 
        if (DF_Dtype(buffer) == DF_data_t::DenseMatrix_t) {
        switch(DF_Vtype(buffer)) {
            case ValueTypeCode::SI8: return DaphneSerializer<DenseMatrix<int8_t>>::deserializeHeader(buffer); break;
            case ValueTypeCode::SI32: return DaphneSerializer<DenseMatrix<int32_t>>::deserializeHeader(buffer); break;
            case ValueTypeCode::SI64: return DaphneSerializer<DenseMatrix<int64_t>>::deserializeHeader(buffer); break;
            case ValueTypeCode::UI8: return DaphneSerializer<DenseMatrix<uint8_t>>::deserializeHeader(buffer); break;
            case ValueTypeCode::UI32: return DaphneSerializer<DenseMatrix<uint32_t>>::deserializeHeader(buffer); break;
            case ValueTypeCode::UI64: return DaphneSerializer<DenseMatrix<uint64_t>>::deserializeHeader(buffer); break;
            case ValueTypeCode::F32: return DaphneSerializer<DenseMatrix<float>>::deserializeHeader(buffer); break;
            case ValueTypeCode::F64: return DaphneSerializer<DenseMatrix<double>>::deserializeHeader(buffer); break;
            default: throw std::runtime_error("unknown value type code");
        }
        } else if (DF_Dtype(buffer) == DF_data_t::CSRMatrix_t) {
        switch(DF_Vtype(buffer)) {
            case ValueTypeCode::SI8: return DaphneSerializer<CSRMatrix<int8_t>>::deserializeHeader(buffer); break;
            case ValueTypeCode::SI32: return DaphneSerializer<CSRMatrix<int32_t>>::deserializeHeader(buffer); break;
            case ValueTypeCode::SI64: return DaphneSerializer<CSRMatrix<int64_t>>::deserializeHeader(buffer); break;
            case ValueTypeCode::UI8: return DaphneSerializer<CSRMatrix<uint8_t>>::deserializeHeader(buffer); break;
            case ValueTypeCode::UI32: return DaphneSerializer<CSRMatrix<uint32_t>>::deserializeHeader(buffer); break;
            case ValueTypeCode::UI64: return DaphneSerializer<CSRMatrix<uint64_t>>::deserializeHeader(buffer); break;
            case ValueTypeCode::F32: return DaphneSerializer<CSRMatrix<float>>::deserializeHeader(buffer); break;
            case ValueTypeCode::F64: return DaphneSerializer<CSRMatrix<double>>::deserializeHeader(buffer); break;
            default: throw std::runtime_error("unknown value type code");
        }
        } else {
            throw std::runtime_error("unknown value type code");
        }
    };
    static Structure *deserialize(const char *buffer, size_t chunkSize, Structure * arg = nullptr, size_t deserializeFromByte = 0) {
        /* DenseMatrix */
        if (auto mat = dynamic_cast<DenseMatrix<double>*>(arg))
            return DaphneSerializer<DenseMatrix<double>>::deserialize(buffer, chunkSize, mat, deserializeFromByte);
        if (auto mat = dynamic_cast<DenseMatrix<float>*>(arg))
            return DaphneSerializer<DenseMatrix<float>>::deserialize(buffer, chunkSize, mat, deserializeFromByte);
        if (auto mat = dynamic_cast<DenseMatrix<int8_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int8_t>>::deserialize(buffer, chunkSize, mat, deserializeFromByte);
        if (auto mat = dynamic_cast<DenseMatrix<int32_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int32_t>>::deserialize(buffer, chunkSize, mat, deserializeFromByte);
        if (auto mat = dynamic_cast<DenseMatrix<int64_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int64_t>>::deserialize(buffer, chunkSize, mat, deserializeFromByte);
        if (auto mat = dynamic_cast<DenseMatrix<uint8_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint8_t>>::deserialize(buffer, chunkSize, mat, deserializeFromByte);
        if (auto mat = dynamic_cast<DenseMatrix<uint32_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint32_t>>::deserialize(buffer, chunkSize, mat, deserializeFromByte);
        if (auto mat = dynamic_cast<DenseMatrix<uint64_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint64_t>>::deserialize(buffer, chunkSize, mat, deserializeFromByte);
            
        /* CSRMatrix */
        if (auto mat = dynamic_cast<CSRMatrix<double>*>(arg))
            return DaphneSerializer<CSRMatrix<double>>::deserialize(buffer, chunkSize, mat, deserializeFromByte);
        if (auto mat = dynamic_cast<CSRMatrix<float>*>(arg))
            return DaphneSerializer<CSRMatrix<float>>::deserialize(buffer, chunkSize, mat, deserializeFromByte);
        if (auto mat = dynamic_cast<CSRMatrix<int8_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int8_t>>::deserialize(buffer, chunkSize, mat, deserializeFromByte);
        if (auto mat = dynamic_cast<CSRMatrix<int32_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int32_t>>::deserialize(buffer, chunkSize, mat, deserializeFromByte);
        if (auto mat = dynamic_cast<CSRMatrix<int64_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int64_t>>::deserialize(buffer, chunkSize, mat, deserializeFromByte);
        if (auto mat = dynamic_cast<CSRMatrix<uint8_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint8_t>>::deserialize(buffer, chunkSize, mat, deserializeFromByte);
        if (auto mat = dynamic_cast<CSRMatrix<uint32_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint32_t>>::deserialize(buffer, chunkSize, mat, deserializeFromByte);
        if (auto mat = dynamic_cast<CSRMatrix<uint64_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint64_t>>::deserialize(buffer, chunkSize, mat, deserializeFromByte);
        // else   
        throw std::runtime_error("Serialization serialize: uknown value type");
    };
};

// ----------------------------------------------------------------------------
// Partial specialization for fundumental types */
// ----------------------------------------------------------------------------

/**
 * @brief Serialize and deserialize value types.
 * 
 * Contains static methods for finding the length in bytes, serializing and
 * deserializing value types.
*/
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


/**
 * @brief Deserializes a buffer to a Daphne object.
 * 
 * @param buf pointer to buffer.
 * @param bufferSize size of buffer in bytes.
 * 
 * @return Pointer to structure of deserialized object.
*/
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

/**
 * @brief Deserializes a vector<char> buffer to a Daphne object.
 * @param buf the vector<char> buffer.
 * 
 * @return Pointer to structure of deserialized object.
*/
inline Structure *DF_deserialize(const std::vector<char> &buf) {
    return DF_deserialize(buf.data(), buf.capacity());
}

/**
 * @brief Serialization out of order.
 * 
 * This class can be used to serialize an object in an out of order fashion. 
 * The serialized chunks produced, can be used to reconstruct the object in any order.
 * All serialized chunks contain a header that is used to determine characteristics of the object (rows, cols, types)
 * and in addition the index of the chunk.
 * 
 * The result chunks should be deserialized using class DaphneDeserializerOutOfOrderChunks
 */
template<class DT>
struct DaphneSerializerOutOfOrderChunks {
private:
    std::mutex lock;    

public:
    /**
     * @brief The default serialization chunk size
     */
    static const size_t DEFAULT_SERIALIZATION_BUFFER_SIZE = 1048576;
    DT *obj;
    size_t chunkSize;
    size_t startOffset = 0; 

    /**
     * @brief Construct a new Daphne Serializer Out Of Order Chunks object
     * 
     * @param obj The object to serialize
     * @param chunkSize_ (optional) The chunk size what will be used for each chunk (default DEFAULT_SERIALIZATION_BUFFER_SIZE)
     */
    DaphneSerializerOutOfOrderChunks(DT *obj, size_t chunkSize_ = DEFAULT_SERIALIZATION_BUFFER_SIZE) : obj(obj), chunkSize(chunkSize_), startOffset(0) {
        chunkSize = chunkSize_ == 0 ? DaphneSerializer<DT>::length(obj) : chunkSize_;
    };

    public:
        size_t numberOfBytesSerialized = 0;
        size_t index;

    /**
     * @brief Serializes the next chunk of data for the given object
     * 
     * @param buffer The buffer to write data
     * @return size_t The number of bytes written to buffer (max = chunkSize)
     */
    size_t SerializeNextChunk(std::vector<char> &buffer) {
        // size_t bytesWritten = 0;

        if (buffer.capacity() < chunkSize)
            buffer.reserve(chunkSize);
        
        // Serialize header (needed for all chunks, we don't know which arrives first)
        auto bufferIdx = DaphneSerializer<DT>::serializeHeader(obj, buffer.data());
        
        // Serialize startOffset index
        std::memcpy(buffer.data() + bufferIdx, reinterpret_cast<char*>(&startOffset), sizeof(startOffset));
        bufferIdx += sizeof(startOffset);

        // TODO use locks for parallel serialization
        size_t len = DaphneSerializer<DT>::serialize(obj, buffer.data() + bufferIdx, chunkSize - bufferIdx, startOffset);
        bufferIdx += len;
        startOffset += len;

        return bufferIdx;
    }
    /**
     * @brief Returns true if there are more chunks to serialize, otherwise false.
     * 
     * @return true 
     * @return false 
     */
    bool HasNextChunk() {
        return startOffset != DaphneSerializer<DT>::length(obj);
    }
};

/**
 * @brief Deserialization out of order.
 * 
 * This class deserializes chunks created with the DaphneSerializationOutOfOrder class.
 * The chunks can be provided in any random order.
 * All chunks should contain a header containing information about the deserialized object along with an index,
 * specifying at which byte-index deserialization should continue for each chunk.
 */
template<class DT>
struct DaphneDeserializerOutOfOrderChunks {
private:
    std::mutex lock;

public:
    DT *obj;
    size_t bytesDeserialized; 

    /**
     * @brief Construct a new Daphne Deserializer Out Of Order Chunks object
     * 
     */
    DaphneDeserializerOutOfOrderChunks() : obj(nullptr), bytesDeserialized(0) {};

    /**
     * @brief Deserializes a chunk and writes the contents to the object.
     * 
     * @param buffer A std::vector<char> containing the serialized data.
     * @return DT* The partially deserialized object.
     */
    DT* DeserializeNextChunk(std::vector<char> &buffer) {
        auto chunkSize = buffer.capacity();
        
        size_t startOffset;        
        size_t bufferIdx = 0;

        if (obj == nullptr) 
            obj = DaphneSerializer<DT>::deserializeHeader(buffer.data(), obj);
        bufferIdx += DaphneSerializer<DT>::headerSize(obj);

        std::memcpy(reinterpret_cast<char*>(&startOffset), buffer.data() + bufferIdx, sizeof(size_t));
        bufferIdx += sizeof(size_t);
        
        // TODO use locks for parallel deserialization
        size_t deserializeLength = chunkSize - bufferIdx;
        obj = DaphneSerializer<DT>::deserialize(buffer.data() + bufferIdx, deserializeLength, obj, startOffset);
        bytesDeserialized += deserializeLength;

        return obj;
    }
    /**
     * @brief Returns true if object is not fully deserialized, else false.
     * 
     * @return true 
     * @return false 
     */
    bool HasNextChunk() {
        if (obj == nullptr)
            return true;
        else
            return bytesDeserialized < DaphneSerializer<DT>::length(obj);
    }
};

/**
 * @brief This class is used to serialize an object using iterators.
 * The result chunks should be deserialized in order.
 * 
 */
template <class DT>
struct DaphneSerializerChunks
{
    /**
     * @brief The default serialization chunk size
     */
    static const size_t DEFAULT_SERIALIZATION_BUFFER_SIZE = 1048576;

    /**
     * @brief The buffer size should be at least big enough to serialize the header.
    */
    static const size_t HEADER_BUFFER_SIZE = DaphneSerializer<DT>::HEADER_BUFFER_SIZE;

    DT *obj;
    size_t chunkSize;
    /**
     * @brief Construct a new Daphne Serializer Chunks object
     * 
     * @param obj The object to serialize
     * @param chunkSize (Optional) The chunk size (default DEFAULT_SERIALIZATION_BUFFER_SIZE). Since at least one chunk will contain the header, the minimum chunk size should be HEADER_BUFFER_SIZE bytes (so the header won't be partially serialized).
     */
    DaphneSerializerChunks(DT *obj, size_t chunkSize = DEFAULT_SERIALIZATION_BUFFER_SIZE) : obj(obj), chunkSize(chunkSize) {
        // Since at least one chunk will contain the header, the minimum chunk size should be HEADER_BUFFER_SIZE bytes (so the header won't be partially serialized).
        if (chunkSize < HEADER_BUFFER_SIZE)
            throw std::runtime_error("Minimum chunk size " + std::to_string(HEADER_BUFFER_SIZE) + " bytes"); // For now..?
    };
    /**
     * @brief An iterator used to serialize the object
     * 
     */
    struct Iterator {
        using iterator_category = std::input_iterator_tag;
        // using difference_type = // todo

        using buffer = std::vector<char>;
        using value_type = std::pair<size_t, buffer>; // TODO verify this
        using pointer = std::pair<size_t, buffer> *;
        using reference = std::pair<size_t, buffer> &; // TODO verify this
    private:
        const DT *obj;
        size_t chunkSize;
        value_type serializedData;

    public:
        size_t numberOfBytesSerialized = 0;
        size_t index;

        // Constructors
        Iterator(){};
        Iterator(const DT *obj, size_t chunkSize) : obj(obj), chunkSize(chunkSize), numberOfBytesSerialized(0), index(0)
        {
            if (chunkSize < HEADER_BUFFER_SIZE)
                throw std::runtime_error("Minimum chunk size " + std::to_string(HEADER_BUFFER_SIZE) + " bytes"); // For now..?
            serializedData.second.reserve(chunkSize);

            serializedData.first = DaphneSerializer<DT>::serialize(obj, serializedData.second.data(), chunkSize, numberOfBytesSerialized);
            numberOfBytesSerialized += serializedData.first;
        };

        reference operator*() { return serializedData; }
        pointer operator->() { return &serializedData; }

        // Prefix increment
        Iterator operator++()
        {
            index++;
            serializedData.first = DaphneSerializer<DT>::serialize(obj, serializedData.second.data(), chunkSize, numberOfBytesSerialized);
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
    /**
     * @brief Returns an iterator containing the first chunk of the serialization process.
     * 
     * @return Iterator 
     */
    Iterator begin() {
        return Iterator(obj, chunkSize);
    }

    /**
     * @brief Returns an iterator at the end of the serialization process (it does not contain any serialized data).
     * 
     * @return Iterator 
     */
    Iterator end() {
        Iterator iter;
        iter.index = DaphneSerializer<DT>::length(obj) / chunkSize + 1;
        return iter;
    }
};

/**
 * @brief A class to deserialize data in order using iterators
 * The iterator expects the chunks in order of serialization, for it to deserialize the object properly.
 */
template <class DT>
struct DaphneDeserializerChunks
{
    /**
     * @brief The buffer size should be at least big enough to serialize the header.
    */
    static const size_t HEADER_BUFFER_SIZE = DaphneSerializer<DT>::HEADER_BUFFER_SIZE;

    DT **objPtr;
    size_t chunkSize;
    /**
     * @brief Construct a new Daphne Deserializer Chunks object
     * 
     * @param obj The object to write data to.
     * @param chunkSize The chunk size used to deserialize
     */
    DaphneDeserializerChunks(DT **obj, size_t chunkSize) : objPtr(obj), chunkSize(chunkSize)
    {
        if (chunkSize < HEADER_BUFFER_SIZE)
            throw std::runtime_error("Minimum chunk size " + std::to_string(HEADER_BUFFER_SIZE) + " bytes"); // For now..?
    };
    struct Iterator
    {
        using iterator_category = std::output_iterator_tag;
        // using difference_type = // todo

        using buffer = std::vector<char>;
        using value_type = std::pair<size_t, buffer>; // TODO verify this
        using pointer = std::pair<size_t, buffer> *;
        using reference = std::pair<size_t, buffer> &; // TODO verify this
        DT **objPtr;

    private:
        size_t chunkSize;
        value_type serializedData;

    public:
        size_t numberOfBytesDeserialized = 0;
        size_t index;

        // Constructors
        Iterator(){};
        Iterator(DT **obj, size_t chunkSize) : objPtr(obj), chunkSize(chunkSize), numberOfBytesDeserialized(0), index(0) {
            if (chunkSize < HEADER_BUFFER_SIZE)
                throw std::runtime_error("Minimum chunk size " + std::to_string(HEADER_BUFFER_SIZE) + " bytes"); // For now..?
            serializedData.second.reserve(chunkSize);
        };

        reference operator*() { return serializedData; }
        pointer operator->() { return &serializedData; }

        // Prefix increment
        Iterator operator++() {
            index++;
            *objPtr = DaphneSerializer<DT>::deserialize(serializedData.second.data(), serializedData.first, *objPtr, numberOfBytesDeserialized);
            numberOfBytesDeserialized += serializedData.first;
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
        return Iterator(objPtr, chunkSize);
    }
    // Iterator end
    Iterator end() {
        Iterator iter;
        // If we just initialized the iterator is possible that we have no information about the deserialized object
        // but the buffer is also uninitialized. If object is uninitialized we simply set Iterator::end() as 1, otherwise
        // we calculate it based on object information.
        if (*objPtr != nullptr)
            iter.index = DaphneSerializer<DT>::length(*objPtr) / chunkSize + 1;
        else
            iter.index = 1;
        return iter;
    }
};