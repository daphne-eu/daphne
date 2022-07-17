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
#include <fstream>
#include <chrono>

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/kernels/CheckEq.h>

#include <core/operators/otfly_derecompr/select.h>
#include <core/operators/otfly_derecompr/join_uncompr.h>

#include <core/morphing/uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>

//using namespace morphstore;
/// @todo Remove before push to main.

int main() {
    srand(42215385);
    long long time_aligned_select_500 = 0;
    long long time_unaligned_select_500 = 0;
    long long time_aligned_select_100k = 0;
    long long time_unaligned_select_100k = 0;
    long long time_aligned_join_25 = 0;
    long long time_unaligned_join_25 = 0;
    long long time_aligned_join_100 = 0;
    long long time_unaligned_join_100 = 0;

    //using ve = vectorlib::avx512<vectorlib::v512<uint64_t> >;
    using ve = vectorlib::scalar<vectorlib::v64<uint64_t>>;

    size_t dataCount_500 = 50 * 1000 * 1000;
    size_t dataCount_100k = 1000 * 1000;
    size_t dataCount_25 = 100 * 1000;
    size_t dataCount_100 = 10 * 1000 * 1000;

    size_t loops = 500;

    const morphstore::column<morphstore::uncompr_f> *const column1_500 = morphstore::ColumnGenerator::generate_with_distr(
            dataCount_500,
            std::uniform_int_distribution<uint64_t>(
                    0,
                    dataCount_500
            ),
            false
    );

    const morphstore::column<morphstore::uncompr_f> *const column1_100k = morphstore::ColumnGenerator::generate_with_distr(
            dataCount_100k,
            std::uniform_int_distribution<uint64_t>(
                    0,
                    dataCount_100k
            ),
            false
    );

    const morphstore::column<morphstore::uncompr_f> *const column1_25 = morphstore::ColumnGenerator::generate_with_distr(
            dataCount_25,
            std::uniform_int_distribution<uint64_t>(
                    0,
                    dataCount_25
            ),
            false
    );

    const morphstore::column<morphstore::uncompr_f> *const column1_100 = morphstore::ColumnGenerator::generate_with_distr(
            dataCount_100,
            std::uniform_int_distribution<uint64_t>(
                    0,
                    dataCount_100
            ),
            false
    );

    auto selectValue_500 = new size_t[loops];
    for(size_t i = 0; i < loops; ++i) {
        selectValue_500[i] = rand();
    }
    const morphstore::column<morphstore::uncompr_f> *const column2_25 = morphstore::ColumnGenerator::generate_with_distr(
            dataCount_25,
            std::uniform_int_distribution<uint64_t>(
                    0,
                    dataCount_25
            ),
            false
    );

    const morphstore::column<morphstore::uncompr_f> *const column2_100 = morphstore::ColumnGenerator::generate_with_distr(
            dataCount_100,
            std::uniform_int_distribution<uint64_t>(
                    0,
                    dataCount_100
            ),
            false
    );

    for (size_t i = 0; i < loops; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = morphstore::select<ve, vectorlib::equal, morphstore::uncompr_f, morphstore::uncompr_f>(column1_500,
                                                                                                             selectValue_500[i]);

        auto end = std::chrono::high_resolution_clock::now();
        time_aligned_select_500 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        delete result;
    }

    loops = 100000;
    auto selectValue_100k = new size_t[loops];
    for(size_t i = 0; i < loops; ++i) {
        selectValue_100k[i] = rand();
    }
    for (size_t i = 0; i < loops; ++i) {

        auto start = std::chrono::high_resolution_clock::now();
        auto result = morphstore::select<ve, vectorlib::equal, morphstore::uncompr_f, morphstore::uncompr_f>(column1_100k,
                                                                                                             selectValue_100k[i]);

        auto end = std::chrono::high_resolution_clock::now();
        time_aligned_select_100k += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        delete result;
    }

    loops = 25;
    for (size_t i = 0; i < loops; ++i) {

        auto start = std::chrono::high_resolution_clock::now();
        auto result = morphstore::semi_join<ve,
                morphstore::uncompr_f,
                morphstore::uncompr_f,
                morphstore::uncompr_f>(column2_25, column1_25);
        auto end = std::chrono::high_resolution_clock::now();
        time_aligned_join_25 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        delete result;
    }

    loops = 100;
    for (size_t i = 0; i < loops; ++i) {

        auto start = std::chrono::high_resolution_clock::now();
        auto result = morphstore::semi_join<ve,
                morphstore::uncompr_f,
                morphstore::uncompr_f,
                morphstore::uncompr_f>(column2_100, column1_100);
        auto end = std::chrono::high_resolution_clock::now();
        time_aligned_join_100 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        delete result;
    }

    std::shared_ptr<uint64_t[]> data_500 = std::shared_ptr<uint64_t[]>(new uint64_t[dataCount_500]);
    std::shared_ptr<uint64_t[]> data_100k = std::shared_ptr<uint64_t[]>(new uint64_t[dataCount_100k]);
    std::shared_ptr<uint64_t[]> data_25 = std::shared_ptr<uint64_t[]>(new uint64_t[dataCount_25]);
    std::shared_ptr<uint64_t[]> data_100 = std::shared_ptr<uint64_t[]>(new uint64_t[dataCount_100]);


    uint64_t * oldData_500 = column1_500->get_data();
    uint64_t * oldData_100k = column1_100k->get_data();
    uint64_t * oldData_25 = column1_25->get_data();
    uint64_t * oldData_100 = column1_100->get_data();

    for (size_t i = 0; i < dataCount_500; ++ i) {
        data_500[i] = oldData_500[i];
    }
    for (size_t i = 0; i < dataCount_100k; ++ i) {
        data_100k[i] = oldData_100k[i];
    }
    for (size_t i = 0; i < dataCount_25; ++ i) {
        data_25[i] = oldData_25[i];
    }
    for (size_t i = 0; i < dataCount_100; ++ i) {
        data_100[i] = oldData_100[i];
    }

    uint64_t * ptr_500 = data_500.get();
    uint64_t * ptr_100k = data_100k.get();
    uint64_t * ptr_25 = data_25.get();
    uint64_t * ptr_100 = data_100.get();

    const morphstore::column<morphstore::uncompr_f> * const column_1_500 = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * dataCount_500, ptr_500);
    const morphstore::column<morphstore::uncompr_f> * const column_1_100k = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * dataCount_100k, ptr_100k);
    const morphstore::column<morphstore::uncompr_f> * const column_1_25 = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * dataCount_25, ptr_25);
    const morphstore::column<morphstore::uncompr_f> * const column_1_100 = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * dataCount_100, ptr_100);

    std::shared_ptr<uint64_t[]> data2_25 = std::shared_ptr<uint64_t[]>(new uint64_t[dataCount_25]);
    std::shared_ptr<uint64_t[]> data2_100 = std::shared_ptr<uint64_t[]>(new uint64_t[dataCount_100]);

    uint64_t * oldData2_25 = column2_25->get_data();
    uint64_t * oldData2_100 = column2_100->get_data();

    for (size_t i = 0; i < dataCount_25; ++ i) {
        data2_25[i] = oldData2_25[i];
    }
    for (size_t i = 0; i < dataCount_100; ++ i) {
        data2_100[i] = oldData2_100[i];
    }

    uint64_t * ptr2_25 = data2_25.get();
    uint64_t * ptr2_100 = data2_100.get();

    const morphstore::column<morphstore::uncompr_f> * const column_2_25 = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * dataCount_25, ptr2_25);
    const morphstore::column<morphstore::uncompr_f> * const column_2_100 = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * dataCount_100, ptr2_100);

    loops = 500;
    for (size_t i = 0; i < loops ; ++i) {

        auto start = std::chrono::high_resolution_clock::now();
        auto result = morphstore::select<ve, vectorlib::equal, morphstore::uncompr_f, morphstore::uncompr_f>(column_1_500, selectValue_500[i]);

        auto end = std::chrono::high_resolution_clock::now();
        time_unaligned_select_500 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        delete result;
    }

    loops = 100000;
    for (size_t i = 0; i < loops ; ++i) {

        auto start = std::chrono::high_resolution_clock::now();
        auto result = morphstore::select<ve, vectorlib::equal, morphstore::uncompr_f, morphstore::uncompr_f>(column_1_100k, selectValue_100k[i]);

        auto end = std::chrono::high_resolution_clock::now();
        time_unaligned_select_100k += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        delete result;
    }

    loops = 25;
    for (size_t i = 0; i < loops; ++i) {

        auto start = std::chrono::high_resolution_clock::now();
        auto result = morphstore::semi_join<ve,
                morphstore::uncompr_f,
                morphstore::uncompr_f,
                morphstore::uncompr_f>(column_2_25, column_1_25);
        auto end = std::chrono::high_resolution_clock::now();
        time_unaligned_join_25 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        delete result;
    }

    loops = 100;
    for (size_t i = 0; i < loops; ++i) {

        auto start = std::chrono::high_resolution_clock::now();
        auto result = morphstore::semi_join<ve,
                morphstore::uncompr_f,
                morphstore::uncompr_f,
                morphstore::uncompr_f>(column_2_100, column_1_100);
        auto end = std::chrono::high_resolution_clock::now();
        time_unaligned_join_100 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        delete result;
    }

    std::ofstream out;
    out.open("results_otfly_scalar.txt", std::ofstream::out | std::ofstream::app);
    if (out.fail())
    {
        std::cout << "Failed to open outputfile.\n";
    }
    out << "Elapsed time in milliseconds with alignment for 500 Loops Select: "
        << time_aligned_select_500 / 1000 / 1000
        << " ms" << std::endl;
    out << "Elapsed time in milliseconds without alignment for 500 Loops Select: "
        << time_unaligned_select_500 / 1000 / 1000
        << " ms" << std::endl;

    out << "Elapsed time in milliseconds with alignment for 100k Loops Select: "
        << time_aligned_select_100k / 1000 / 1000
        << " ms" << std::endl;

    out << "Elapsed time in milliseconds without alignment for 100k Loops Select: "
        << time_unaligned_select_100k / 1000 / 1000
        << " ms" << std::endl;

    out << "Elapsed time in milliseconds with alignment for 25 Loops LeftSemiJoin: "
        << time_aligned_join_25 / 1000 / 1000
        << " ms" << std::endl;

    out << "Elapsed time in milliseconds without alignment for 25 Loops LeftSemiJoin: "
        << time_unaligned_join_25 / 1000 / 1000
        << " ms" << std::endl << std::endl;

    out << "Elapsed time in milliseconds with alignment for 100 Loops LeftSemiJoin: "
        << time_aligned_join_25 / 1000 / 1000
        << " ms" << std::endl;

    out << "Elapsed time in milliseconds without alignment for 100 Loops LeftSemiJoin: "
        << time_unaligned_join_25 / 1000 / 1000
        << " ms" << std::endl << std::endl;
    out.close();
    return 0;
}
