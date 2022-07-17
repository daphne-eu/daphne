/**********************************************************************************************
 * Copyright (C) 2019 by MorphStore-Team                                                      *
 *                                                                                            *
 * This file is part of MorphStore - a compression aware vectorized column store.             *
 *                                                                                            *
 * This program is free software: you can redistribute it and/or modify it under the          *
 * terms of the GNU General Public License as published by the Free Software Foundation,      *
 * either version 3 of the License, or (at your option) any later version.                    *
 *                                                                                            *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
 * See the GNU General Public License for more details.                                       *
 *                                                                                            *
 * You should have received a copy of the GNU General Public License along with this program. *
 * If not, see <http://www.gnu.org/licenses/>.                                                *
 **********************************************************************************************/


#ifndef DAPHNE_PROTOTYPE_CALC_H
#define DAPHNE_PROTOTYPE_CALC_H

#include <cstdint>

#include <core/storage/column.h>
#include <core/operators/otfly_derecompr/calc_uncompr.h>
#include "core/morphing/uncompr.h"


#include <ir/daphneir/Daphne.h>

enum class CalcOperation : uint32_t {
    Add = 1,
    Sub = 2,
    Mul = 3,
    Div = 4,
};

template<class DTRes, class DTIn>
class Calc {
public:
    static void apply(DTRes * & res, const DTIn * inLhs, const DTIn inRhs, const char * inOnLeft, const char * inOnRight, CalcOperation calc) = delete;
};

template<class DTRes, class DTIn>
void calc(DTRes * & res, const DTIn * inLhs, const DTIn * inRhs, const char * inOnLeft, const char * inOnRight, CalcOperation calc) {
    Calc<DTRes, DTIn>::apply(res, inLhs, inRhs, inOnLeft, inOnRight, calc);
}

template<>
class Calc<Frame, Frame> {
public:
    static void apply(Frame * & res, const Frame * inLhs, const Frame * inRhs, const char * inOnLeft, const char * inOnRight, CalcOperation calc) {
        assert((inLhs->getNumRows() == inRhs->getNumRows()) && "number of input rows not the same");

        auto colDataLeft = static_cast<uint64_t const *>(inLhs->getColumnRaw(inLhs->getColumnIdx(inOnLeft)));
        const morphstore::column<morphstore::uncompr_f> * const opColLeft = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * inLhs->getNumRows(), colDataLeft);

        auto colDataRight = static_cast<uint64_t const *>(inRhs->getColumnRaw(inRhs->getColumnIdx(inOnRight)));
        const morphstore::column<morphstore::uncompr_f> * const opColRight = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * inRhs->getNumRows(), colDataRight);

        using ve = vectorlib::scalar<vectorlib::v64<uint64_t> >;

        morphstore::column<morphstore::uncompr_f> *result;

        switch (calc) {
            case CalcOperation::Add:
                result = const_cast<morphstore::column<morphstore::uncompr_f> *>(
                        morphstore::calc_binary<
                        vectorlib::add,
                        ve,
                        morphstore::uncompr_f,
                        morphstore::uncompr_f,
                        morphstore::uncompr_f
                        >(opColLeft, opColRight));
                break;
            case CalcOperation::Sub:
                result = const_cast<morphstore::column<morphstore::uncompr_f> *>(
                        morphstore::calc_binary<
                        vectorlib::sub,
                        ve,
                        morphstore::uncompr_f,
                        morphstore::uncompr_f,
                        morphstore::uncompr_f
                >(opColLeft, opColRight));
                break;
            case CalcOperation::Mul:
                result = const_cast<morphstore::column<morphstore::uncompr_f> *>(
                        morphstore::calc_binary<
                        vectorlib::mul,
                        ve,
                        morphstore::uncompr_f,
                        morphstore::uncompr_f,
                        morphstore::uncompr_f
                >(opColLeft, opColRight));
                break;
            case CalcOperation::Div:
                result = const_cast<morphstore::column<morphstore::uncompr_f> *>(
                        morphstore::calc_binary<
                        vectorlib::div,
                        ve,
                        morphstore::uncompr_f,
                        morphstore::uncompr_f,
                        morphstore::uncompr_f
                >(opColLeft, opColRight));
                break;
        }

        /// Change the persistence type to disable the deletion and deallocation of the data.
        result->set_persistence_type(morphstore::storage_persistence_type::externalScope);

        uint64_t * ptr = result->get_data();

        std::shared_ptr<uint64_t[]> shrdPtr(ptr);

        auto resultMatrix = DataObjectFactory::create<DenseMatrix<uint64_t>>(result->get_count_values(), 1, shrdPtr);

        const std::string columnLabels[] = {"Calc"};

        std::vector<Structure *> resultCols = {resultMatrix};

        res = DataObjectFactory::create<Frame>(resultCols, columnLabels);

        delete result, delete opColLeft, delete opColRight;
    }

};
#endif //DAPHNE_PROTOTYPE_CALC_H
