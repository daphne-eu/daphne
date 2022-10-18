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


#ifndef DAPHNE_PROTOTYPE_BETWEEN_H
#define DAPHNE_PROTOTYPE_BETWEEN_H

#include <cstdint>

#include <core/storage/column.h>
#include "core/operators/otfly_derecompr/between.h"
#include "core/operators/otfly_derecompr/project.h"
#include "runtime/local/datastructures/Frame.h"
#include "core/morphing/uncompr.h"

#include <ir/daphneir/Daphne.h>

template<class DTRes, class DTIn>
class Between {
public:
    static void apply(DTRes * & res, const DTIn * in, const char * inOn, uint64_t lowerBound, CompareOperation cmpLower, uint64_t upperBound, CompareOperation cmpUpper) = delete;
};

template<class DTRes, class DTIn, typename ve=vectorlib::scalar<vectorlib::v64<uint64_t>>>
void between(DTRes * & res, const DTIn * in, const char * inOn, uint64_t lowerBound, CompareOperation cmpLower, uint64_t upperBound, CompareOperation cmpUpper) {
    Between<DTRes, DTIn>::template apply<ve>(res, in, inOn, lowerBound, cmpLower, upperBound, cmpUpper);
}

template<>
class Between<Frame, Frame> {
public:
    template<typename ve=vectorlib::scalar<vectorlib::v64<uint64_t>>>
    static void apply(Frame * & res, const Frame * in, const char * inOn, uint64_t lowerBound, CompareOperation cmpLower, uint64_t upperBound, CompareOperation cmpUpper) {
        auto colData = static_cast<uint64_t const *>(in->getColumnRaw(in->getColumnIdx(inOn)));
        const morphstore::column<morphstore::uncompr_f> * const betweenCol = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * in->getNumRows(), colData);

        const morphstore::column<morphstore::uncompr_f> * betweenPos;

        if (cmpLower == CompareOperation::GreaterThan) {
            switch (cmpUpper) {
                case CompareOperation::LessThan:
                    betweenPos = morphstore::between<
                            vectorlib::greater,
                            vectorlib::less,
                            ve,
                            morphstore::uncompr_f,
                            morphstore::uncompr_f
                            >(betweenCol, lowerBound, upperBound);
                    break;
                case CompareOperation::LessEqual:
                    betweenPos = morphstore::between<
                            vectorlib::greater,
                            vectorlib::lessequal,
                            ve,
                            morphstore::uncompr_f,
                            morphstore::uncompr_f
                    >(betweenCol, lowerBound, upperBound);
                    break;
            }
        } else {
            switch (cmpUpper) {
                case CompareOperation::LessThan:
                    betweenPos = morphstore::between<
                            vectorlib::greaterequal,
                            vectorlib::less,
                            ve,
                            morphstore::uncompr_f,
                            morphstore::uncompr_f
                    >(betweenCol, lowerBound, upperBound);
                    break;
                case CompareOperation::LessEqual:
                    betweenPos = morphstore::between<
                            vectorlib::greaterequal,
                            vectorlib::lessequal,
                            ve,
                            morphstore::uncompr_f,
                            morphstore::uncompr_f
                    >(betweenCol, lowerBound, upperBound);
                    break;
            }
        }

        const std::string *columnLabels = in->getLabels();

        std::vector<Structure *> resultCols = {};

        for (size_t i = 0; i < in->getNumCols(); ++ i) {
            auto colProjData = static_cast<uint64_t const *>(in->getColumnRaw(in->getColumnIdx(*(columnLabels + i))));
            auto colProj = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * in->getNumRows(), colProjData);
            auto *projCol = const_cast<morphstore::column<morphstore::uncompr_f> *>(morphstore::my_project_wit_t<ve, morphstore::uncompr_f, morphstore::uncompr_f,
                    morphstore::uncompr_f>::apply(colProj, betweenPos));

            /// Change the persistence type to disable the deletion and deallocation of the data.
            projCol->set_persistence_type(morphstore::storage_persistence_type::externalScope);

            uint64_t * ptr = projCol->get_data();

            std::shared_ptr<uint64_t[]> shrdPtr(ptr);

            auto result = DataObjectFactory::create<DenseMatrix<uint64_t>>(projCol->get_count_values(), 1, shrdPtr);

            resultCols.push_back(result);
            delete projCol, delete colProj;
        }

        res = DataObjectFactory::create<Frame>(resultCols, columnLabels);

        delete betweenPos, delete betweenCol;
    }

};
#endif //DAPHNE_PROTOTYPE_BETWEEN_H
