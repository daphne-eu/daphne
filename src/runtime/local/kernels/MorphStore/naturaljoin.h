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

#ifndef DAPHNE_PROTOTYPE_NATURALJOIN_H
#define DAPHNE_PROTOTYPE_NATURALJOIN_H

#include <cstdint>

#include <core/storage/column.h>
#include "core/operators/otfly_derecompr/join_uncompr.h"
#include "core/operators/otfly_derecompr/project.h"
#include "runtime/local/datastructures/Frame.h"
#include <core/operators/otfly_derecompr/intersect_tuples.h>
#include "core/morphing/uncompr.h"
#include "core/utils/printing.h"


/// The implemented Semijoin is a Left Semijoin.
template<class DTRes, class DTInLeft, class DTInRight>
class Naturaljoin {
public:
    static void apply(DTRes * & res, const DTInLeft * inLeft, const DTInRight * inRight, const char ** inOnLeft, size_t numLhsOn, const char ** inOnRight, size_t numRhsOn) = delete;
};

template<class DTRes, class DTInLeft, class DTInRight>
void naturaljoin(DTRes * & res, const DTInLeft * inLeft, const DTInRight * inRight, const char ** inOnLeft, size_t numLhsOn, const char ** inOnRight, size_t numRhsOn) {
    Naturaljoin<DTRes, DTInLeft, DTInRight>::apply(res, inLeft, inRight, inOnLeft, numLhsOn, inOnRight, numRhsOn);
}

template<>
class Naturaljoin<Frame, Frame, Frame> {
public:
    static void apply(Frame * & res, const Frame * inLeft, const Frame * inRight, const char ** inOnLeft, size_t numLhsOn, const char ** inOnRight, size_t numRhsOn) {

        assert((numLhsOn == numRhsOn) && "incorrect amount of compare values");

        using ve = vectorlib::scalar<vectorlib::v64<uint64_t> >;

        const morphstore::column<morphstore::uncompr_f> *selectPosLeft= nullptr;
        const morphstore::column<morphstore::uncompr_f> *selectPosRight= nullptr;

        for (size_t i = 0; i < numLhsOn; ++ i) {
            auto colDataLeft = static_cast<uint64_t const *>(inLeft->getColumnRaw(inLeft->getColumnIdx(inOnLeft[i])));
            auto colDataRight = static_cast<uint64_t const *>(inRight->getColumnRaw(inRight->getColumnIdx(inOnRight[i])));

            const morphstore::column<morphstore::uncompr_f> * const joinColLeft = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * inLeft->getNumRows(), colDataLeft);
            const morphstore::column<morphstore::uncompr_f> * const joinColRight = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * inRight->getNumRows(), colDataRight);

            auto currentPosCol = morphstore::natural_equi_join<ve, morphstore::uncompr_f, morphstore::uncompr_f, morphstore::uncompr_f, morphstore::uncompr_f>(joinColLeft, joinColRight);

            /// @todo Check if actually working logic with tests or if order gets mixed up
            if (selectPosLeft && selectPosRight) {
                auto posCols = morphstore::intersect_tuples(std::make_tuple(selectPosLeft, selectPosRight), currentPosCol);
                delete selectPosLeft, delete selectPosRight;
                selectPosLeft = std::get<0>(posCols);
                selectPosRight = std::get<1>(posCols);
            } else {
                selectPosLeft = std::get<0>(currentPosCol);
                selectPosRight = std::get<1>(currentPosCol);
            }

            /// @todo Check which elements should be deleted.
            // delete currentPosCol, delete join_col_left, delete join_col_right, delete col_data_left, delete col_data_right;
        }

        const std::string *columnLabelsLeft = inLeft->getLabels();
        const std::string *columnLabelsRight = inRight->getLabels();
        std::string* columnLabels = NULL;
        columnLabels = new std::string[inLeft->getNumCols()+inRight->getNumCols()];

        std::vector<Structure *> resultCols = {};
        for (size_t i = 0; i < inLeft->getNumCols(); ++ i) {
            auto colProjData = static_cast<uint64_t const *>(inLeft->getColumnRaw(inLeft->getColumnIdx(*(columnLabelsLeft + i))));
            auto colProj = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * inLeft->getNumRows(), colProjData);
            auto* projCol = const_cast<morphstore::column<morphstore::uncompr_f> *>(morphstore::my_project_wit_t<ve, morphstore::uncompr_f, morphstore::uncompr_f,
                    morphstore::uncompr_f>::apply(colProj, selectPosLeft));

            /// Change the persistence type to disable the deletion and deallocation of the data.
            projCol->set_persistence_type(morphstore::storage_persistence_type::externalScope);

            uint64_t * ptr = projCol->get_data();

            std::shared_ptr<uint64_t[]> shrdPtr(ptr);

            auto result = DataObjectFactory::create<DenseMatrix<uint64_t>>(projCol->get_count_values(), 1, shrdPtr);

            resultCols.push_back(result);
            columnLabels[i] = *(columnLabelsLeft + i);
            delete projCol, delete colProj;
        }

        for (size_t i = 0; i < inRight->getNumCols(); ++ i) {
            auto colProjData = static_cast<uint64_t const *>(inRight->getColumnRaw(inRight->getColumnIdx(*(columnLabelsRight + i))));
            auto colProj = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * inRight->getNumRows(), colProjData);
            auto* projCol = const_cast<morphstore::column<morphstore::uncompr_f> *>(morphstore::my_project_wit_t<ve, morphstore::uncompr_f, morphstore::uncompr_f,
                    morphstore::uncompr_f>::apply(colProj, selectPosRight));

            /// Change the persistence type to disable the deletion and deallocation of the data.
            projCol->set_persistence_type(morphstore::storage_persistence_type::externalScope);

            uint64_t * ptr = projCol->get_data();

            std::shared_ptr<uint64_t[]> shrdPtr(ptr);

            auto result = DataObjectFactory::create<DenseMatrix<uint64_t>>(projCol->get_count_values(), 1, shrdPtr);

            resultCols.push_back(result);
            columnLabels[i + inLeft->getNumCols()] = *(columnLabelsRight + i);
            delete projCol, delete colProj;
        }

        res = DataObjectFactory::create<Frame>(resultCols, columnLabels);

        delete selectPosLeft, delete selectPosRight;

    }

};

#endif //DAPHNE_PROTOTYPE_NATURALJOIN_H
