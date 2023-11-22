#include <vector>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <chrono>

#include <runtime/local/kernels/SIMDOperatorsDAPHNE/ColumnProject.h>
#include <runtime/local/kernels/SIMDOperatorsDAPHNE/ColumnProjectionPath.h>
#include <runtime/local/kernels/SIMDOperatorsDAPHNE/ColumnSelect.h>

#include <runtime/local/kernels/RandMatrix.h>
#include <runtime/local/kernels/CastObj.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/DenseMatrix.h>

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
long runProjectProject(
    tuddbs::Column<int64_t>* data,
    tuddbs::Column<int64_t>* pos_list_1,
    tuddbs::Column<int64_t>* pos_list_2
    ) {
    auto start = std::chrono::high_resolution_clock::now();

    tuddbs::daphne_project<ps> project;

    auto first_result = project(data, pos_list_1);

    project(first_result, pos_list_2);

    auto end = std::chrono::high_resolution_clock::now();

    return(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}

template <typename ps>
long runProjectProjectionPath(
    tuddbs::Column<int64_t>* data,
    tuddbs::Column<int64_t>* pos_list_1,
    tuddbs::Column<int64_t>* pos_list_2
    ) {
    auto start = std::chrono::high_resolution_clock::now();

    tuddbs::daphne_project<ps> project;

    auto first_result = project(data, pos_list_1);

    tuddbs::daphne_projection_path<ps> projection_path;

    projection_path(first_result, pos_list_2, pos_list_1);

    auto end = std::chrono::high_resolution_clock::now();

    return(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}

int main() {
    {
        DenseMatrix<int64_t>* data = nullptr;
        randMatrix<DenseMatrix<int64_t>, int64_t>(data, 500'000'000, 1, 0, 1000, 1, 42, nullptr);
        
        tuddbs::Column<int64_t>* data_cast = nullptr;

        castObj(data_cast, data, nullptr);

        tuddbs::daphne_select<tsl::simd<int64_t, tsl::scalar>, tsl::functors::less_than> select;
        auto pos_list_1 = select(data_cast, 600);

        tuddbs::daphne_project<tsl::simd<int64_t, tsl::scalar>> project;
        auto data_projected = project(data_cast, pos_list_1);

        tuddbs::daphne_select<tsl::simd<int64_t, tsl::scalar>, tsl::functors::greater_than> select2;
        auto pos_list_2 = select2(data_projected, 250);

        long time_scalar_project_project = 0;
        long time_avx512_project_project = 0;
        long time_scalar_project_projectionpath = 0;
        long time_avx512_project_projectionpath = 0;

        //Scalar
        {
            std::ofstream outfile;
            using ps = typename tsl::simd<int64_t, tsl::scalar>;

            time_scalar_project_project = runProjectProject<ps>(
                data_cast,
                pos_list_1,
                pos_list_2
            );

            outfile.open("../evaluation/scalar_project_project.csv", std::ios_base::app);

            std::string time = getCurrentTimeFormatted();

            outfile << time << "," << time_scalar_project_project / double(1000) << std::endl;
        }

        //AVX512
        {
            std::ofstream outfile;
            using ps = typename tsl::simd<int64_t, tsl::avx512>;

            time_avx512_project_project = runProjectProject<ps>(
                data_cast,
                pos_list_1,
                pos_list_2
            );

            outfile.open("../evaluation/avx512_project_project.csv", std::ios_base::app);

            std::string time = getCurrentTimeFormatted();

            outfile << time << "," << time_avx512_project_project / double(1000) << std::endl;
        }

        //Scalar
        {
            std::ofstream outfile;
            using ps = typename tsl::simd<int64_t, tsl::scalar>;

            time_scalar_project_projectionpath = runProjectProjectionPath<ps>(
                data_cast,
                pos_list_1,
                pos_list_2
            );

            outfile.open("../evaluation/scalar_project_projectionpath.csv", std::ios_base::app);

            std::string time = getCurrentTimeFormatted();

            outfile << time << "," << time_scalar_project_projectionpath / double(1000) << std::endl;
        }

        //AVX512
        {
            std::ofstream outfile;
            using ps = typename tsl::simd<int64_t, tsl::avx512>;

            time_avx512_project_projectionpath = runProjectProjectionPath<ps>(
                data_cast,
                pos_list_1,
                pos_list_2
            );

            outfile.open("../evaluation/avx512_project_projectionpath.csv", std::ios_base::app);

            std::string time = getCurrentTimeFormatted();

            outfile << time << "," << time_avx512_project_projectionpath / double(1000) << std::endl;
        }

    }

}