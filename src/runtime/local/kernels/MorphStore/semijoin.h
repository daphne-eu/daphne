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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_MORPHSTORE_SEMIJOIN_H
#define SRC_RUNTIME_LOCAL_KERNELS_MORPHSTORE_SEMIJOIN_H

#include <cstdint>

#include <core/storage/column.h>
#include "core/operators/otfly_derecompr/join_uncompr.h"
#include "core/operators/otfly_derecompr/project.h"
#include "runtime/local/datastructures/Frame.h"
#include <core/operators/otfly_derecompr/intersect.h>
#include "core/morphing/uncompr.h"

/// The implemented Semijoin is a Left Semijoin.
template<class DTRes, class DTInLeft, class DTInRight>
class Semijoin {
public:
    static void apply(DTRes * & res, const DTInLeft * inLeft, const DTInRight * inRight, const char ** inOnLeft, size_t numLhsOn, const char ** inOnRight, size_t numRhsOn) = delete;
};

template<class DTRes, class DTInLeft, class DTInRight>
void semijoin(DTRes * & res, const DTInLeft * inLeft, const DTInRight * inRight, const char ** inOnLeft, size_t numLhsOn, const char ** inOnRight, size_t numRhsOn) {
    Semijoin<DTRes, DTInLeft, DTInRight>::apply(res, inLeft, inRight, inOnLeft, numLhsOn, inOnRight, numRhsOn);
}

template<>
class Semijoin<Frame, Frame, Frame> {
public:
    static void apply(Frame * & res, const Frame * inLeft, const Frame * inRight, const char ** inOnLeft, size_t numLhsOn, const char ** inOnRight, size_t numRhsOn) {

        assert((numLhsOn == numRhsOn) && "incorrect amount of compare values");

        using ve = vectorlib::scalar<vectorlib::v64<uint64_t> >;

        std::vector<const morphstore::column<morphstore::uncompr_f> *> posCols = {};

        for (size_t i = 0; i < numLhsOn; ++ i) {
            auto colDataLeft = static_cast<uint64_t const *>(inLeft->getColumnRaw(inLeft->getColumnIdx(inOnLeft[i])));
            auto colDataRight = static_cast<uint64_t const *>(inRight->getColumnRaw(inRight->getColumnIdx(inOnRight[i])));

            const morphstore::column<morphstore::uncompr_f> * const joinColLeft = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * inLeft->getNumRows(), colDataLeft);
            const morphstore::column<morphstore::uncompr_f> * const joinColRight = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * inRight->getNumRows(), colDataRight);

            auto currentPosCol = morphstore::semi_join<ve, morphstore::uncompr_f, morphstore::uncompr_f, morphstore::uncompr_f>(joinColRight, joinColLeft);

            posCols.push_back(currentPosCol);

            /// @todo Check which elements should be deleted.
            // delete currentPosCol, delete join_col_left, delete join_col_right, delete col_data_left, delete col_data_right;
        }

        const morphstore::column<morphstore::uncompr_f> *selectPos;

//      If necessary, we combine our resulting Position Columns with an intersect, so we only access the data that fulfills all requirements.
        if (numLhsOn < 2) {
            selectPos = posCols[0];
        } else {
            selectPos = morphstore::intersect_sorted<ve, morphstore::uncompr_f, morphstore::uncompr_f, morphstore::uncompr_f >(posCols[0], posCols[1]);
            for (size_t i = 2; i < numLhsOn; ++ i) {
                selectPos = morphstore::intersect_sorted<ve, morphstore::uncompr_f, morphstore::uncompr_f, morphstore::uncompr_f >(selectPos, posCols[i]);
            }
        }

        const std::string *columnLabels = inLeft->getLabels();

        std::vector<Structure *> resultCols = {};

        for (size_t i = 0; i < inLeft->getNumCols(); ++ i) {
            auto colProjData = static_cast<uint64_t const *>(inLeft->getColumnRaw(inLeft->getColumnIdx(*(columnLabels + i))));
            auto colProj = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * inLeft->getNumRows(), colProjData);
            auto* projCol = const_cast<morphstore::column<morphstore::uncompr_f> *>(morphstore::my_project_wit_t<ve, morphstore::uncompr_f, morphstore::uncompr_f,
                    morphstore::uncompr_f>::apply(colProj, selectPos));

            /// Change the persistence type to disable the deletion and deallocation of the data.
            projCol->set_persistence_type(morphstore::storage_persistence_type::externalScope);

            uint64_t * ptr = projCol->get_data();

            std::shared_ptr<uint64_t[]> shrdPtr(ptr);

            auto result = DataObjectFactory::create<DenseMatrix<uint64_t>>(projCol->get_count_values(), 1, shrdPtr);

            resultCols.push_back(result);
            delete projCol, delete colProj;
        }

        res = DataObjectFactory::create<Frame>(resultCols, columnLabels);
        /// @todo Check which delete is more useful.
        /**for (auto i: posCols) {
            delete i;
        }**/
        delete selectPos;

    }

};

#endif //SRC_RUNTIME_LOCAL_KERNELS_MORPHSTORE_SEMIJOIN_H
