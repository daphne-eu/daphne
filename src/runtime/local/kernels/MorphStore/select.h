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


#ifndef SRC_RUNTIME_LOCAL_KERNELS_MORPHSTORE_SELECT_H
#define SRC_RUNTIME_LOCAL_KERNELS_MORPHSTORE_SELECT_H

#include <cstdint>

#include <core/storage/column.h>
#include "core/operators/otfly_derecompr/select.h"
#include "core/operators/otfly_derecompr/project.h"
#include "core/operators/otfly_derecompr/merge.h"
#include "core/morphing/uncompr.h"

#include <ir/daphneir/Daphne.h>

enum class CompareOperation : uint32_t {
  Equal = 1,
  LessThan = 2,
  LessEqual = 3,
  GreaterThan = 4,
  GreaterEqual = 5,
  NotEqual = 6,
};

/// later use this CompareOperation Enum
//using mlir::daphne::CompareOperation;


template<class DTRes, class DTIn>
class Select {
  public:
    static void apply(DTRes * & res, const DTIn * in, const char * inOn, CompareOperation cmp, uint64_t selValue) = delete;
};

template<class DTRes, class DTIn, typename ve=vectorlib::scalar<vectorlib::v64<uint64_t>>>
void select(DTRes * & res, const DTIn * in, const char * inOn, CompareOperation cmp, uint64_t selValue) {
    Select<DTRes, DTIn>::apply(res, in, inOn, cmp, selValue);
}

template<>
class Select<Frame, Frame> {
    public:
        template<typename ve=vectorlib::scalar<vectorlib::v64<uint64_t>>>
        static void apply(Frame * & res, const Frame * in, const char * inOn, CompareOperation cmp, uint64_t selValue) {
            auto colData = static_cast<uint64_t const *>(in->getColumnRaw(in->getColumnIdx(inOn)));
            const morphstore::column<morphstore::uncompr_f> * const selectCol = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * in->getNumRows(), colData);

            const morphstore::column<morphstore::uncompr_f> * selectPos;

            switch (cmp) {
                case CompareOperation::Equal:

                    selectPos = morphstore::select<
                            ve,
                            vectorlib::equal,
                            morphstore::uncompr_f,
                            morphstore::uncompr_f
                    >(selectCol, selValue);
                    break;
                case CompareOperation::LessThan:
                    selectPos = morphstore::select<
                            ve,
                            vectorlib::less,
                            morphstore::uncompr_f,
                            morphstore::uncompr_f
                    >(selectCol, selValue);
                    break;
                case CompareOperation::LessEqual:
                    selectPos = morphstore::select<
                            ve,
                            vectorlib::lessequal,
                            morphstore::uncompr_f,
                            morphstore::uncompr_f
                    >(selectCol, selValue);
                    break;
                case CompareOperation::GreaterThan:
                    selectPos = morphstore::select<
                            ve,
                            vectorlib::greater,
                            morphstore::uncompr_f,
                            morphstore::uncompr_f
                    >(selectCol, selValue);
                    break;
                case CompareOperation::GreaterEqual:
                    selectPos = morphstore::select<
                            ve,
                            vectorlib::greaterequal,
                            morphstore::uncompr_f,
                            morphstore::uncompr_f
                    >(selectCol, selValue);
                    break;
                case CompareOperation::NotEqual:
                    auto smallerPos = morphstore::select<
                            ve,
                            vectorlib::less,
                            morphstore::uncompr_f,
                            morphstore::uncompr_f
                    >(selectCol, selValue);
                    auto greaterPos = morphstore::select<
                            ve,
                            vectorlib::greater,
                            morphstore::uncompr_f,
                            morphstore::uncompr_f
                    >(selectCol, selValue);

                    selectPos = morphstore::merge_sorted<ve,
                            morphstore::uncompr_f,
                            morphstore::uncompr_f,
                            morphstore::uncompr_f>(smallerPos, greaterPos);
                    delete smallerPos, delete greaterPos;
                    break;
            }

            const std::string *columnLabels = in->getLabels();

            std::vector<Structure *> resultCols = {};

            for (size_t i = 0; i < in->getNumCols(); ++ i) {
                auto colProjData = static_cast<uint64_t const *>(in->getColumnRaw(in->getColumnIdx(*(columnLabels + i))));
                auto colProj = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * in->getNumRows(), colProjData);
                auto* projCol = const_cast<morphstore::column<morphstore::uncompr_f> *>(morphstore::my_project_wit_t<ve, morphstore::uncompr_f, morphstore::uncompr_f,
                        morphstore::uncompr_f>::apply(
                        colProj, selectPos));

                /// Change the persistence type to disable the deletion and deallocation of the data.
                projCol->set_persistence_type(morphstore::storage_persistence_type::externalScope);

                uint64_t * ptr = projCol->get_data();

                std::shared_ptr<uint64_t[]> shrdPtr(ptr);

                auto result = DataObjectFactory::create<DenseMatrix<uint64_t>>(projCol->get_count_values(), 1, shrdPtr);

                resultCols.push_back(result);
                delete projCol, delete colProj;
            }

            res = DataObjectFactory::create<Frame>(resultCols, columnLabels);

            delete selectPos, delete selectCol;
        }

    };
#endif //SRC_RUNTIME_LOCAL_KERNELS_MORPHSTORE_SELECT_H
