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

#include <vector>
#include <cstdint>
#include <iostream>
#include <chrono>

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/kernels/CheckEq.h>

#include <core/operators/otfly_derecompr/select.h>
#include <core/morphing/uncompr.h>
#include <core/operators/otfly_derecompr/project.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
//using namespace morphstore;
/// @todo Remove before push to main.

int main() {
    long long time_aligned = 0;
    long long time_unaligned = 0;

    //using ve = vectorlib::avx512<vectorlib::v512<uint64_t> >;
    using ve = vectorlib::scalar<vectorlib::v64<uint64_t>>;

    const size_t dataCount = 50 * 1000 * 1000;

    size_t loops = 500;

    const morphstore::column<morphstore::uncompr_f> *const column = morphstore::ColumnGenerator::generate_with_distr(
            dataCount,
            std::uniform_int_distribution<uint64_t>(
                    0,
                    dataCount
            ),
            false
    );
    for (size_t i = 0; i < loops; ++i) {

        auto start = std::chrono::high_resolution_clock::now();
        auto result = morphstore::select<ve, vectorlib::equal, morphstore::uncompr_f, morphstore::uncompr_f>(column,
                                                                                                             rand());
        auto end = std::chrono::high_resolution_clock::now();
        time_aligned += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        delete result;
    }

    std::shared_ptr<uint64_t[]> data = std::shared_ptr<uint64_t[]>(new uint64_t[dataCount]);

    uint64_t * oldData = column->get_data();

    for (size_t i = 0; i < dataCount; ++ i) {
        data[i] = oldData[i];
    }

    uint64_t * ptr = data.get();

    const morphstore::column<morphstore::uncompr_f> * const column_2 = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * dataCount, ptr);
    for (size_t i = 0; i < loops ; ++i) {

        auto start = std::chrono::high_resolution_clock::now();
        auto result = morphstore::select<ve, vectorlib::equal, morphstore::uncompr_f, morphstore::uncompr_f>(column_2, rand());
        auto end = std::chrono::high_resolution_clock::now();
        time_unaligned += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        delete result;
    }

    std::cout << "Elapsed time in milliseconds with alignment: "
         << time_aligned / 1000 / 1000
         << " ms" << std::endl;

    std::cout << "Elapsed time in milliseconds without alignment: "
              << time_unaligned / 1000 / 1000
              << " ms" << std::endl;
}
