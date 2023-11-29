#include "SIMDOperators/operators/project.hpp"
#include "generated/declarations/compare.hpp"
#include <vector>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <chrono>

#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/wrappers/DAPHNE/select.hpp>
#include <SIMDOperators/wrappers/DAPHNE/projectionPath.hpp>
#include <SIMDOperators/wrappers/DAPHNE/aggregate.hpp>
#include <SIMDOperators/wrappers/DAPHNE/between.hpp>
#include <SIMDOperators/wrappers/DAPHNE/calc.hpp>
#include <SIMDOperators/wrappers/DAPHNE/equiJoin.hpp>
#include <SIMDOperators/wrappers/DAPHNE/project.hpp>
#include <SIMDOperators/wrappers/DAPHNE/intersect.hpp>

#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/kernels/CastObj.h>
#include <runtime/local/kernels/Read.h>
#include <runtime/local/kernels/ExtractCol.h>

std::string getCurrentTimeFormatted() {
    std::time_t rawtime;
    std::tm* timeinfo;
    char buffer [80];

    std::time(&rawtime);
    timeinfo = std::localtime(&rawtime);

    std::strftime(buffer,80,"%Y-%m-%d_%H:%M:%S",timeinfo);
    return buffer;
}

template <typename ps>
long runSSBQuery11WOSelPushdown(
    tuddbs::Column<int64_t>* date_d_datekey_cast,
    tuddbs::Column<int64_t>* date_d_year_cast,
    tuddbs::Column<int64_t>* lineorder_lo_orderdate_cast,
    tuddbs::Column<int64_t>* lineorder_lo_discount_cast,
    tuddbs::Column<int64_t>* lineorder_lo_quantity_cast,
    tuddbs::Column<int64_t>* lineorder_lo_extendedprice_cast
    ) {
    auto start = std::chrono::high_resolution_clock::now();

    //smaller join partner needs to be first argument
    auto join = tuddbs::natural_equi_join<ps>(
        date_d_datekey_cast,
        lineorder_lo_orderdate_cast
    );

    tuddbs::Column<int64_t>* lo_pos = std::get<1>(join);
    tuddbs::Column<int64_t>* d_pos = std::get<0>(join);

    tuddbs::daphne_project<ps> project;
    auto project_year = project(date_d_year_cast, d_pos);
    auto project_discount = project(lineorder_lo_discount_cast, lo_pos);
    auto project_quantity = project(lineorder_lo_quantity_cast, lo_pos);

    tuddbs::daphne_select<ps, tsl::functors::equal> select_eq;
    auto pos_year = select_eq(project_year, 1993);

    tuddbs::daphne_select<ps, tsl::functors::less_than> select_lt;
    auto pos_quantity = select_lt(project_quantity, 25);

    tuddbs::daphne_between<ps> between;
    auto pos_discount = between(project_discount, 1, 3);

    tuddbs::daphne_intersect<ps> intersect;
    auto pos_temp = intersect(pos_year, pos_quantity);
    auto pos = intersect(pos_temp, pos_discount);

    tuddbs::daphne_projection_path<ps> projection_path;
    auto proj_final_extended_price = projection_path(lineorder_lo_extendedprice_cast, pos, lo_pos);

    auto proj_final_discount = project(project_discount, pos);

    tuddbs::daphne_calc<ps, tsl::functors::mul> calc;
    auto final_extended_price = calc(proj_final_extended_price, proj_final_discount);

    tuddbs::daphne_aggregate<ps, tsl::functors::add, tsl::functors::hadd> aggregate;
    aggregate(final_extended_price);

    auto end = std::chrono::high_resolution_clock::now();

    return(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}

template <typename ps>
long runSSBQuery11WithSelPushdown(
    tuddbs::Column<int64_t>* date_d_datekey_cast,
    tuddbs::Column<int64_t>* date_d_year_cast,
    tuddbs::Column<int64_t>* lineorder_lo_orderdate_cast,
    tuddbs::Column<int64_t>* lineorder_lo_discount_cast,
    tuddbs::Column<int64_t>* lineorder_lo_quantity_cast,
    tuddbs::Column<int64_t>* lineorder_lo_extendedprice_cast
    ) {
    auto start = std::chrono::high_resolution_clock::now();

    tuddbs::daphne_select<ps, tsl::functors::equal> select_eq;
    auto pos_year = select_eq(date_d_year_cast, 1993);

    tuddbs::daphne_select<ps, tsl::functors::less_than> select_lt;
    auto pos_quantity = select_lt(lineorder_lo_quantity_cast, 25);

    tuddbs::daphne_between<ps> between;
    auto pos_discount = between(lineorder_lo_discount_cast, 1, 3);

    tuddbs::daphne_intersect<ps> intersect;
    auto pos = intersect(pos_quantity, pos_discount);

    tuddbs::daphne_project<ps> project;
    auto project_datekey = project(date_d_datekey_cast, pos_year);
    auto project_orderdate = project(lineorder_lo_orderdate_cast, pos);

    //smaller join partner needs to be first argument
    auto join = tuddbs::natural_equi_join<ps>(
        project_datekey,
        project_orderdate
    );

    tuddbs::Column<int64_t>* lo_pos = std::get<1>(join);

    tuddbs::daphne_projection_path<ps> projection_path;
    auto proj_final_extended_price = projection_path(lineorder_lo_extendedprice_cast, lo_pos, pos);
    auto proj_final_discount = projection_path(lineorder_lo_discount_cast, lo_pos, pos);

    tuddbs::daphne_calc<ps, tsl::functors::mul> calc;
    auto final_extended_price = calc(proj_final_extended_price, proj_final_discount);

    tuddbs::daphne_aggregate<ps, tsl::functors::add, tsl::functors::hadd> aggregate;
    aggregate(final_extended_price);

    auto end = std::chrono::high_resolution_clock::now();

    return(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}

int main() {
    //SF1
    {
        Frame* date = nullptr;
        read(date, "data/ssb/sf1/date.csv", nullptr);
        Frame* lineorder = nullptr;
        read(lineorder, "data/ssb/sf1/lineorder.csv", nullptr);

        Frame* date_d_datekey = nullptr;
        Frame* date_d_year = nullptr;

        extractCol(date_d_datekey, date, "d_datekey", nullptr);
        extractCol(date_d_year, date, "d_year", nullptr);

        Frame* lineorder_lo_orderdate = nullptr;
        Frame* lineorder_lo_discount = nullptr;
        Frame* lineorder_lo_quantity = nullptr;
        Frame* lineorder_lo_extendedprice = nullptr;

        extractCol(lineorder_lo_orderdate, lineorder, "lo_orderdate", nullptr);
        extractCol(lineorder_lo_discount, lineorder, "lo_discount", nullptr);
        extractCol(lineorder_lo_quantity, lineorder, "lo_quantity", nullptr);
        extractCol(lineorder_lo_extendedprice, lineorder, "lo_extendedprice", nullptr);

        tuddbs::Column<int64_t>* date_d_datekey_cast = nullptr;
        tuddbs::Column<int64_t>* date_d_year_cast = nullptr;

        castObj(date_d_datekey_cast, date_d_datekey, nullptr);
        castObj(date_d_year_cast, date_d_year, nullptr);

        tuddbs::Column<int64_t>* lineorder_lo_orderdate_cast = nullptr;
        tuddbs::Column<int64_t>* lineorder_lo_discount_cast = nullptr;
        tuddbs::Column<int64_t>* lineorder_lo_quantity_cast = nullptr;
        tuddbs::Column<int64_t>* lineorder_lo_extendedprice_cast = nullptr;

        castObj(lineorder_lo_orderdate_cast, lineorder_lo_orderdate, nullptr);
        castObj(lineorder_lo_discount_cast, lineorder_lo_discount, nullptr);
        castObj(lineorder_lo_quantity_cast, lineorder_lo_quantity, nullptr);
        castObj(lineorder_lo_extendedprice_cast, lineorder_lo_extendedprice, nullptr);

        long time_scalar_wo_pushdown_sf1 = 0;
        long time_avx512_wo_pushdown_sf1 = 0;
        long time_scalar_with_pushdown_sf1 = 0;
        long time_avx512_with_pushdown_sf1 = 0;

        //Scalar Without Selection Pushdown
        {
            std::ofstream outfile;
            using ps = typename tsl::simd<int64_t, tsl::scalar>;

            time_scalar_wo_pushdown_sf1 = runSSBQuery11WOSelPushdown<ps>(
                date_d_datekey_cast,
                date_d_year_cast,
                lineorder_lo_orderdate_cast,
                lineorder_lo_discount_cast,
                lineorder_lo_quantity_cast,
                lineorder_lo_extendedprice_cast
            );

            outfile.open("../evaluation/scalar_sf1_q11_wo_selection_pushdown_wo_daphne.csv", std::ios_base::app);

            std::string time = getCurrentTimeFormatted();

            outfile << time << "," << time_scalar_wo_pushdown_sf1 / double(1000) << std::endl;
        }

        //AVX512 Without Selection Pushdown
        {
            std::ofstream outfile;
            using ps = typename tsl::simd<int64_t, tsl::avx512>;

            time_avx512_wo_pushdown_sf1 = runSSBQuery11WOSelPushdown<ps>(
                date_d_datekey_cast,
                date_d_year_cast,
                lineorder_lo_orderdate_cast,
                lineorder_lo_discount_cast,
                lineorder_lo_quantity_cast,
                lineorder_lo_extendedprice_cast
            );

            outfile.open("../evaluation/avx512_sf1_q11_wo_selection_pushdown_wo_daphne.csv", std::ios_base::app);

            std::string time = getCurrentTimeFormatted();

            outfile << time << "," << time_avx512_wo_pushdown_sf1 / double(1000) << std::endl;
        }

        //AVX512 With Selection Pushdown
        {
            std::ofstream outfile;
            using ps = typename tsl::simd<int64_t, tsl::avx512>;

            time_avx512_with_pushdown_sf1 = runSSBQuery11WithSelPushdown<ps>(
                date_d_datekey_cast,
                date_d_year_cast,
                lineorder_lo_orderdate_cast,
                lineorder_lo_discount_cast,
                lineorder_lo_quantity_cast,
                lineorder_lo_extendedprice_cast
            );

            outfile.open("../evaluation/avx512_sf1_q11_with_selection_pushdown_wo_daphne.csv", std::ios_base::app);

            std::string time = getCurrentTimeFormatted();

            outfile << time << "," << time_avx512_with_pushdown_sf1 / double(1000) << std::endl;
        }

        //Scalar With Selection Pushdown
        {
            std::ofstream outfile;
            using ps = typename tsl::simd<int64_t, tsl::scalar>;

            time_scalar_with_pushdown_sf1 = runSSBQuery11WithSelPushdown<ps>(
                date_d_datekey_cast,
                date_d_year_cast,
                lineorder_lo_orderdate_cast,
                lineorder_lo_discount_cast,
                lineorder_lo_quantity_cast,
                lineorder_lo_extendedprice_cast
            );

            outfile.open("../evaluation/scalar_sf1_q11_with_selection_pushdown_wo_daphne.csv", std::ios_base::app);

            std::string time = getCurrentTimeFormatted();

            outfile << time << "," << time_scalar_with_pushdown_sf1 / double(1000) << std::endl;
        }

    }

    //SF10
    {
        Frame* date = nullptr;
        read(date, "data/ssb/sf10/date.csv", nullptr);
        Frame* lineorder = nullptr;
        read(lineorder, "data/ssb/sf10/lineorder.csv", nullptr);

        Frame* date_d_datekey = nullptr;
        Frame* date_d_year = nullptr;

        extractCol(date_d_datekey, date, "d_datekey", nullptr);
        extractCol(date_d_year, date, "d_year", nullptr);

        Frame* lineorder_lo_orderdate = nullptr;
        Frame* lineorder_lo_discount = nullptr;
        Frame* lineorder_lo_quantity = nullptr;
        Frame* lineorder_lo_extendedprice = nullptr;

        extractCol(lineorder_lo_orderdate, lineorder, "lo_orderdate", nullptr);
        extractCol(lineorder_lo_discount, lineorder, "lo_discount", nullptr);
        extractCol(lineorder_lo_quantity, lineorder, "lo_quantity", nullptr);
        extractCol(lineorder_lo_extendedprice, lineorder, "lo_extendedprice", nullptr);

        tuddbs::Column<int64_t>* date_d_datekey_cast = nullptr;
        tuddbs::Column<int64_t>* date_d_year_cast = nullptr;

        castObj(date_d_datekey_cast, date_d_datekey, nullptr);
        castObj(date_d_year_cast, date_d_year, nullptr);

        tuddbs::Column<int64_t>* lineorder_lo_orderdate_cast = nullptr;
        tuddbs::Column<int64_t>* lineorder_lo_discount_cast = nullptr;
        tuddbs::Column<int64_t>* lineorder_lo_quantity_cast = nullptr;
        tuddbs::Column<int64_t>* lineorder_lo_extendedprice_cast = nullptr;

        castObj(lineorder_lo_orderdate_cast, lineorder_lo_orderdate, nullptr);
        castObj(lineorder_lo_discount_cast, lineorder_lo_discount, nullptr);
        castObj(lineorder_lo_quantity_cast, lineorder_lo_quantity, nullptr);
        castObj(lineorder_lo_extendedprice_cast, lineorder_lo_extendedprice, nullptr);

        long time_scalar_wo_pushdown_sf10 = 0;
        long time_avx512_wo_pushdown_sf10 = 0;
        long time_scalar_with_pushdown_sf10 = 0;
        long time_avx512_with_pushdown_sf10 = 0;

        //Scalar Without Selection Pushdown
        {
            std::ofstream outfile;
            using ps = typename tsl::simd<int64_t, tsl::scalar>;

            time_scalar_wo_pushdown_sf10 = runSSBQuery11WOSelPushdown<ps>(
                date_d_datekey_cast,
                date_d_year_cast,
                lineorder_lo_orderdate_cast,
                lineorder_lo_discount_cast,
                lineorder_lo_quantity_cast,
                lineorder_lo_extendedprice_cast
            );

            outfile.open("../evaluation/scalar_sf10_q11_wo_selection_pushdown_wo_daphne.csv", std::ios_base::app);

            std::string time = getCurrentTimeFormatted();

            outfile << time << "," << time_scalar_wo_pushdown_sf10 / double(1000) << std::endl;
        }

        //AVX512 Without Selection Pushdown
        {
            std::ofstream outfile;
            using ps = typename tsl::simd<int64_t, tsl::avx512>;

            time_avx512_wo_pushdown_sf10 = runSSBQuery11WOSelPushdown<ps>(
                date_d_datekey_cast,
                date_d_year_cast,
                lineorder_lo_orderdate_cast,
                lineorder_lo_discount_cast,
                lineorder_lo_quantity_cast,
                lineorder_lo_extendedprice_cast
            );

            outfile.open("../evaluation/avx512_sf10_q11_wo_selection_pushdown_wo_daphne.csv", std::ios_base::app);

            std::string time = getCurrentTimeFormatted();

            outfile << time << "," << time_avx512_wo_pushdown_sf10 / double(1000) << std::endl;
        }

        //Scalar With Selection Pushdown
        {
            std::ofstream outfile;
            using ps = typename tsl::simd<int64_t, tsl::scalar>;

            time_scalar_with_pushdown_sf10 = runSSBQuery11WithSelPushdown<ps>(
                date_d_datekey_cast,
                date_d_year_cast,
                lineorder_lo_orderdate_cast,
                lineorder_lo_discount_cast,
                lineorder_lo_quantity_cast,
                lineorder_lo_extendedprice_cast
            );

            outfile.open("../evaluation/scalar_sf10_q11_with_selection_pushdown_wo_daphne.csv", std::ios_base::app);

            std::string time = getCurrentTimeFormatted();

            outfile << time << "," << time_scalar_with_pushdown_sf10 / double(1000) << std::endl;
        }

        //AVX512 With Selection Pushdown
        {
            std::ofstream outfile;
            using ps = typename tsl::simd<int64_t, tsl::avx512>;

            time_avx512_with_pushdown_sf10 = runSSBQuery11WithSelPushdown<ps>(
                date_d_datekey_cast,
                date_d_year_cast,
                lineorder_lo_orderdate_cast,
                lineorder_lo_discount_cast,
                lineorder_lo_quantity_cast,
                lineorder_lo_extendedprice_cast
            );

            outfile.open("../evaluation/avx512_sf10_q11_with_selection_pushdown_wo_daphne.csv", std::ios_base::app);

            std::string time = getCurrentTimeFormatted();

            outfile << time << "," << time_avx512_with_pushdown_sf10 / double(1000) << std::endl;
        }

    }

}