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


#ifndef DAPHNE_PROTOTYPE_SUM_H
#define DAPHNE_PROTOTYPE_SUM_H

#include <cstdint>

#include <core/storage/column.h>
#include <core/operators/otfly_derecompr/agg_sum_all.h>
#include "core/morphing/uncompr.h"

template<class DTRes, class DTIn>
class AggSum {
public:
    static void apply(DTRes * & res, const DTIn * in, const char * inOn) = delete;
};

template<class DTRes, class DTIn>
void agg_sum(DTRes * & res, const DTIn * in, const char * inOn) {
    AggSum<DTRes, DTIn>::apply(res, in, inOn);
}

template<>
class AggSum<Frame, Frame> {
public:
    static void apply(Frame * & res, const Frame * in, const char * inOn) {
        auto colData = static_cast<uint64_t const *>(in->getColumnRaw(in->getColumnIdx(inOn)));
        const morphstore::column<morphstore::uncompr_f> * const aggCol = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * in->getNumRows(), colData);

        using ve = vectorlib::scalar<vectorlib::v64<uint64_t> >;

        auto* aggResult = const_cast<morphstore::column<morphstore::uncompr_f> *>(morphstore::agg_sum_all<ve, morphstore::uncompr_f, morphstore::uncompr_f>(aggCol));

        /// Change the persistence type to disable the deletion and deallocation of the data.
        aggResult->set_persistence_type(morphstore::storage_persistence_type::externalScope);

        uint64_t * ptr = aggResult->get_data();

        std::shared_ptr<uint64_t[]> shrdPtr(ptr);

        auto result = DataObjectFactory::create<DenseMatrix<uint64_t>>(aggResult->get_count_values(), 1, shrdPtr);

        const std::string columnLabels[] = {"Agg_sum"};

        std::vector<Structure *> resultCols = {result};

        res = DataObjectFactory::create<Frame>(resultCols, columnLabels);

        delete aggResult, delete aggCol;
    }

};
#endif //DAPHNE_PROTOTYPE_SUM_H
