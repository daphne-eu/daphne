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

#ifndef DAPHNE_PROTOTYPE_GROUPSUM_H
#define DAPHNE_PROTOTYPE_GROUPSUM_H

#include <runtime/local/datastructures/Frame.h>
#include <core/operators/otfly_derecompr/project.h>
#include <core/morphing/format.h>
#include "core/morphing/uncompr.h"
#include <core/operators/uncompr/group_first.h>
#include <core/operators/uncompr/group_next.h>
#include <core/operators/reference/agg_sum_grouped.h>

template<class DTRes, class DTIn>
class Groupsum {
public:
    static void apply(DTRes * & res, const DTIn * in, const std::string * groupLabels, const size_t numLabels, const std::string sumColumn) = delete;
};

template<class DTRes, class DTIn, typename ve=vectorlib::scalar<vectorlib::v64<uint64_t>>>
void groupsum(DTRes * & res, const DTIn * in, const std::string * groupLabels, const size_t numLabels, const std::string sumColumn) {
    Groupsum<DTRes, DTIn>::apply(res, in, groupLabels, numLabels, sumColumn);
}

template<>
class Groupsum<Frame, Frame> {
public:
    template<typename ve=vectorlib::scalar<vectorlib::v64<uint64_t>>>
    static void apply(Frame * & res, const Frame * in, const std::string * groupLabels, const size_t numLabels, const std::string sumColumn) {
        std::tuple<const morphstore::column<morphstore::uncompr_f> *, const morphstore::column<morphstore::uncompr_f> *> resultPos;
        for (size_t i = 0; i < numLabels; ++ i) {
            auto colData = static_cast<uint64_t const *>(in->getColumnRaw(in->getColumnIdx(groupLabels[i])));
            const morphstore::column<morphstore::uncompr_f> * const col = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * in->getNumRows(), colData);
            if (i == 0) {
                resultPos = morphstore::group_first<ve, morphstore::uncompr_f, morphstore::uncompr_f, morphstore::uncompr_f>(col);
            } else {
                resultPos = morphstore::group_next<ve, morphstore::uncompr_f, morphstore::uncompr_f, morphstore::uncompr_f, morphstore::uncompr_f>(std::get<0>(resultPos), col);
            }
        }
        auto sumData = static_cast<uint64_t const *>(in->getColumnRaw(in->getColumnIdx(sumColumn)));
        const morphstore::column<morphstore::uncompr_f> * const sumCol = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * in->getNumRows(), sumData);
        morphstore::column <morphstore::uncompr_f> *sumResults = (morphstore::column <morphstore::uncompr_f> *) morphstore::agg_sum_grouped<ve, morphstore::uncompr_f, morphstore::uncompr_f, morphstore::uncompr_f>(
                std::get<0>(resultPos), sumCol, std::get<1>(resultPos)->get_count_values());

        std::vector<Structure *> resultCols = {};
        for (size_t i = 0; i < numLabels; ++ i) {
            auto colProjData = static_cast<uint64_t const *>(in->getColumnRaw(in->getColumnIdx(*(groupLabels + i))));
            auto colProj = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * in->getNumRows(), colProjData);
            auto* projCol = const_cast<morphstore::column<morphstore::uncompr_f> *>(morphstore::my_project_wit_t<ve, morphstore::uncompr_f, morphstore::uncompr_f,
                    morphstore::uncompr_f>::apply(
                    colProj, std::get<1>(resultPos)));

            /// Change the persistence type to disable the deletion and deallocation of the data.
            projCol->set_persistence_type(morphstore::storage_persistence_type::externalScope);

            uint64_t * ptr = projCol->get_data();

            std::shared_ptr<uint64_t[]> shrdPtr(ptr);

            auto result = DataObjectFactory::create<DenseMatrix<uint64_t>>(projCol->get_count_values(), 1, shrdPtr);

            resultCols.push_back(result);
            delete projCol, delete colProj;
        }
        /// Change the persistence type to disable the deletion and deallocation of the data.
        sumResults->set_persistence_type(morphstore::storage_persistence_type::externalScope);

        uint64_t * ptr = sumResults->get_data();

        std::shared_ptr<uint64_t[]> shrdPtr(ptr);

        auto result = DataObjectFactory::create<DenseMatrix<uint64_t>>(sumResults->get_count_values(), 1, shrdPtr);

        resultCols.push_back(result);

        std::vector<std::string> newLabels(groupLabels, groupLabels + numLabels);
        newLabels.push_back("sum-"+sumColumn);
        std::string *labelArray = &newLabels[0];

        res = DataObjectFactory::create<Frame>(resultCols, labelArray);

        delete std::get<0>(resultPos), delete std::get<1>(resultPos);
    }

};


#endif //DAPHNE_PROTOTYPE_GROUPSUM_H