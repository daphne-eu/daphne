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

inline DF_data_t DF_Dtype(const void *buf) {
    return (DF_data_t)((const DF_header *)buf)->dt;
}

inline ValueTypeCode DF_Vtype(const void *buf) {
    const ValueTypeCode *vt = (const ValueTypeCode *)((const char *)buf + sizeof(DF_header));
    return *vt;
}

template <class DTArg>
struct DaphneSerializer {
    static size_t length(const DTArg *arg);
    static void *save(const DTArg *arg, void *buf);
    static DTArg *load(const void *buf);
};

template <typename VT>
struct DaphneSerializer<DenseMatrix<VT>> {
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

    static void *save(const DenseMatrix<VT> *arg, void *buf) {
        size_t idx = 0;

        if (!buf) {
            buf = malloc(DaphneSerializer<DenseMatrix<VT>>::length(arg));
        }

        // write header
        DF_header h;
        h.version = 1;
        h.dt = (uint8_t)DF_data_t::DenseMatrix_t;
        h.nbrows = (uint64_t) arg->getNumRows();
        h.nbcols = (uint64_t) arg->getNumCols();

        memcpy((char *)buf + idx, (const char *)&h, sizeof(h));
        idx += sizeof(h);

        // value type
        const ValueTypeCode vt = ValueTypeUtils::codeFor<VT>;

        memcpy((char *)buf + idx, (const char *)&vt, sizeof(vt));
        idx += sizeof(vt);

        // write body
        // single block
        DF_body b;
        b.rx = 0;
        b.cx = 0;

        memcpy((char *)buf + idx, (const char *)&b, sizeof(b));
        idx += sizeof(b);

        // block header
        DF_body_block bb;
        bb.nbrows = (uint32_t) arg->getNumRows();
        bb.nbcols = (uint32_t) arg->getNumCols();
        bb.bt = (uint8_t)DF_body_t::dense;

        memcpy((char *)buf + idx, (const char *)&bb, sizeof(bb));
        idx += sizeof(bb);

        // value type
        memcpy((char *)buf + idx, (const char *) &vt, sizeof(vt));
        idx += sizeof(vt);

        // block values
        const VT * valuesArg = arg->getValues();
        memcpy((char *)buf + idx, (const char *)valuesArg, arg->getNumRows() * arg->getNumCols() * sizeof(VT));

        return buf;
   };

   static DenseMatrix<VT> *load(const void *buf) {
        const char *ibuf = (const char *)buf;

        assert((DF_Dtype(buf) == DF_data_t::DenseMatrix_t) && "DenseMatrix load(): DT mismatch");
        assert((DF_Vtype(buf) == ValueTypeUtils::codeFor<VT>) && "DenseMatrix load(): VT mismatch");

        // FF to the body
        ibuf += sizeof(DF_header);
        ibuf += sizeof(ValueTypeCode);
        ibuf += sizeof(DF_body);

        const DF_body_block *bb = (const DF_body_block *)ibuf;
        ibuf += sizeof(DF_body_block);

        // empty Matrix
        if (bb->bt == (uint8_t)DF_body_t::empty) {
            return DataObjectFactory::create<DenseMatrix<VT>>(0, 0, false);
        }
        // Dense Matrix
        else if (bb->bt == (uint8_t)DF_body_t::dense) {
            ibuf += sizeof(ValueTypeCode);

            size_t len = bb->nbrows * bb->nbcols * sizeof(VT);
            VT* memblock = (VT*) malloc(len);
            memcpy((char *) memblock, ibuf, len);

            std::shared_ptr<VT[]> data;
            data.reset(memblock);
            return DataObjectFactory::create<DenseMatrix<VT>>((size_t)bb->nbrows,
                        (size_t)bb->nbcols, data);
        } else {
            throw std::runtime_error("unknown body type code");
        }
   };
};

template <typename VT>
struct DaphneSerializer<CSRMatrix<VT>> {
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

    static void *save(const CSRMatrix<VT> *arg, void *buf) {
        size_t idx = 0;

        if (!buf) {
            buf = malloc(DaphneSerializer<CSRMatrix<VT>>::length(arg));
        }

        // write header
        DF_header h;
        h.version = 1;
        h.dt = (uint8_t)DF_data_t::CSRMatrix_t;
        h.nbrows = (uint64_t) arg->getNumRows();
        h.nbcols = (uint64_t) arg->getNumCols();

        memcpy((char *)buf + idx, (const char *)&h, sizeof(h));
        idx += sizeof(h);

        // value type
        const ValueTypeCode vt = ValueTypeUtils::codeFor<VT>;

        memcpy((char *)buf + idx, (const char *)&vt, sizeof(vt));
        idx += sizeof(vt);

        // write body
        // single block
        DF_body b;
        b.rx = 0;
        b.cx = 0;

        memcpy((char *)buf + idx, (const char *)&b, sizeof(b));
        idx += sizeof(b);

        // block header
        DF_body_block bb;
        bb.nbrows = (uint32_t) arg->getNumRows();
        bb.nbcols = (uint32_t) arg->getNumCols();
        bb.bt = (uint8_t)DF_body_t::sparse;

        memcpy((char *)buf + idx, (const char *)&bb, sizeof(bb));
        idx += sizeof(bb);

        // value type
        memcpy((char *)buf + idx, (const char *) &vt, sizeof(vt));
        idx += sizeof(vt);

        // num non-zeros
        size_t nzb = 0;
        for (size_t r = 0; r < arg->getNumRows(); r++){
            nzb += arg->getNumNonZeros(r);            
        }
        memcpy((char *)buf + idx, (const char *) &nzb, sizeof(nzb));
        idx += sizeof(nzb);

        const size_t * rowOffsets = arg->getRowOffsets();
        // memcpy((char *)buf + idx, rowOffsets, sizeof(size_t) * (arg->getNumRows() + 1));
        // idx += sizeof(size_t) * (arg->getNumRows() + 1);
        const size_t offset_diff = *arg->getRowOffsets();
        std::unique_ptr<size_t> new_rows(new size_t[arg->getNumRows() + 1]);
        for (size_t r = 0; r < arg->getNumRows() + 1; r++){
            auto newVal = *(rowOffsets + r) - offset_diff;                        
            new_rows.get()[r] = newVal;
        }
        memcpy((char *)buf + idx, new_rows.get(), sizeof(size_t) * (arg->getNumRows() + 1));
        idx += sizeof(size_t) * (arg->getNumRows() + 1);

        const size_t * colIdxs = arg->getColIdxs(0);
        memcpy((char *)buf + idx, colIdxs, sizeof(size_t) * nzb);
        idx += sizeof(size_t) * nzb;

        const VT * vals = arg->getValues(0);
        memcpy((char *)buf + idx, vals, nzb * sizeof(VT));
        idx += nzb * sizeof(VT);

        return buf;
   };

   static CSRMatrix<VT> *load(const void *buf) {
        const char *ibuf = (const char *)buf;

        assert((DF_Dtype(buf) == DF_data_t::CSRMatrix_t) && "CSRMatrix load(): DT mismatch");
        assert((DF_Vtype(buf) == ValueTypeUtils::codeFor<VT>) && "CSRMatrix load(): VT mismatch");

        // FF to the body
        ibuf += sizeof(DF_header);
        ibuf += sizeof(ValueTypeCode);
        ibuf += sizeof(DF_body);

        const DF_body_block *bb = (const DF_body_block *)ibuf;
        ibuf += sizeof(DF_body_block);

        // empty Matrix
        if (bb->bt == (uint8_t)DF_body_t::empty) {
            return DataObjectFactory::create<CSRMatrix<VT>>(0, 0, 0, false);
        // CSRMatrix
        } else if (bb->bt == (uint8_t)DF_body_t::sparse) {
            ibuf += sizeof(ValueTypeCode);

            size_t nzb;
            memcpy(&nzb, ibuf, sizeof(nzb));
            ibuf += sizeof(nzb);

            auto res = DataObjectFactory::create<CSRMatrix<VT>>(bb->nbrows, bb->nbcols, nzb, true);

            size_t * rowOffsets = res->getRowOffsets();
            memcpy(rowOffsets, ibuf, sizeof(size_t) * (bb->nbrows + 1));
            ibuf += sizeof(size_t) * (bb->nbrows + 1);

            size_t * colIdxs = res->getColIdxs();
            memcpy(colIdxs, ibuf, sizeof(size_t) * nzb);
            ibuf += sizeof(size_t) * nzb;

            VT * vals = res->getValues();
            memcpy(vals, ibuf, sizeof(VT) * nzb);
            
            return res;
        /* TODO MPI: No COO support for write? */
        // COO Matrix
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
   };
};

template <>
struct DaphneSerializer<Frame> {
    static size_t length(const Frame *arg) {
        throw std::runtime_error("not implemented");
    };

    static void *save(const Frame *arg, void *buf) {
        throw std::runtime_error("not implemented");
    };

   static Frame *load(const void *buf) {
       throw std::runtime_error("not implemented");
   };
};

template <>
struct DaphneSerializer<const Structure> {
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

    static void *save(const Structure *arg, void *buf) {
        /* DenseMatrix */
        if (auto mat = dynamic_cast<const DenseMatrix<double>*>(arg))
            return DaphneSerializer<DenseMatrix<double>>::save(mat, buf);
        if (auto mat = dynamic_cast<const DenseMatrix<float>*>(arg))
            return DaphneSerializer<DenseMatrix<float>>::save(mat, buf);
        if (auto mat = dynamic_cast<const DenseMatrix<int8_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int8_t>>::save(mat, buf);
        if (auto mat = dynamic_cast<const DenseMatrix<int32_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int32_t>>::save(mat, buf);
        if (auto mat = dynamic_cast<const DenseMatrix<int64_t>*>(arg))
            return DaphneSerializer<DenseMatrix<int64_t>>::save(mat, buf);
        if (auto mat = dynamic_cast<const DenseMatrix<uint8_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint8_t>>::save(mat, buf);
        if (auto mat = dynamic_cast<const DenseMatrix<uint32_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint32_t>>::save(mat, buf);
        if (auto mat = dynamic_cast<const DenseMatrix<uint64_t>*>(arg))
            return DaphneSerializer<DenseMatrix<uint64_t>>::save(mat, buf);
            
        /* CSRMatrix */
        if (auto mat = dynamic_cast<const CSRMatrix<double>*>(arg))
            return DaphneSerializer<CSRMatrix<double>>::save(mat, buf);
        if (auto mat = dynamic_cast<const CSRMatrix<float>*>(arg))
            return DaphneSerializer<CSRMatrix<float>>::save(mat, buf);
        if (auto mat = dynamic_cast<const CSRMatrix<int8_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int8_t>>::save(mat, buf);
        if (auto mat = dynamic_cast<const CSRMatrix<int32_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int32_t>>::save(mat, buf);
        if (auto mat = dynamic_cast<const CSRMatrix<int64_t>*>(arg))
            return DaphneSerializer<CSRMatrix<int64_t>>::save(mat, buf);
        if (auto mat = dynamic_cast<const CSRMatrix<uint8_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint8_t>>::save(mat, buf);
        if (auto mat = dynamic_cast<const CSRMatrix<uint32_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint32_t>>::save(mat, buf);
        if (auto mat = dynamic_cast<const CSRMatrix<uint64_t>*>(arg))
            return DaphneSerializer<CSRMatrix<uint64_t>>::save(mat, buf);
        // else   
        throw std::runtime_error("Serialization save: uknown value type");
    };

   static Structure *load(const void *buf) {
       throw std::runtime_error("not implemented");
   };
};

inline Structure *DF_load(const void *buf) {
    if (DF_Dtype(buf) == DF_data_t::DenseMatrix_t) {
        switch(DF_Vtype(buf)) {
            case ValueTypeCode::SI8: return DaphneSerializer<DenseMatrix<int8_t>>::load(buf); break;
            case ValueTypeCode::SI32: return DaphneSerializer<DenseMatrix<int32_t>>::load(buf); break;
            case ValueTypeCode::SI64: return DaphneSerializer<DenseMatrix<int64_t>>::load(buf); break;
            case ValueTypeCode::UI8: return DaphneSerializer<DenseMatrix<uint8_t>>::load(buf); break;
            case ValueTypeCode::UI32: return DaphneSerializer<DenseMatrix<uint32_t>>::load(buf); break;
            case ValueTypeCode::UI64: return DaphneSerializer<DenseMatrix<uint64_t>>::load(buf); break;
            case ValueTypeCode::F32: return DaphneSerializer<DenseMatrix<float>>::load(buf); break;
            case ValueTypeCode::F64: return DaphneSerializer<DenseMatrix<double>>::load(buf); break;
            default: throw std::runtime_error("unknown value type code");
        }
    } else if (DF_Dtype(buf) == DF_data_t::CSRMatrix_t) {
        switch(DF_Vtype(buf)) {
            case ValueTypeCode::SI8: return DaphneSerializer<CSRMatrix<int8_t>>::load(buf); break;
            case ValueTypeCode::SI32: return DaphneSerializer<CSRMatrix<int32_t>>::load(buf); break;
            case ValueTypeCode::SI64: return DaphneSerializer<CSRMatrix<int64_t>>::load(buf); break;
            case ValueTypeCode::UI8: return DaphneSerializer<CSRMatrix<uint8_t>>::load(buf); break;
            case ValueTypeCode::UI32: return DaphneSerializer<CSRMatrix<uint32_t>>::load(buf); break;
            case ValueTypeCode::UI64: return DaphneSerializer<CSRMatrix<uint64_t>>::load(buf); break;
            case ValueTypeCode::F32: return DaphneSerializer<CSRMatrix<float>>::load(buf); break;
            case ValueTypeCode::F64: return DaphneSerializer<CSRMatrix<double>>::load(buf); break;
            default: throw std::runtime_error("unknown value type code");
        }
    } else {
        throw std::runtime_error("unknown value type code");
    }
}
