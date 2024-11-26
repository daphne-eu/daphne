#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <thread>

class DaphneContext;

extern "C" {

// This struct contains parameters that control how the calculation is performed
// hardware targets, coprocessing share and other parameters can be changed here
struct config {
    size_t vector_size = 1024 * 1024 * 256;
    int omp_threads = 6;    // OpenMP creates this many threads for multi core CPU part
    float share_cpu = 0.5f; // share of data processed on cpu using openMP. The rest ist processed on GPU
    size_t start_index = 0;
    std::string processing_mode = "Co-processing";
};

/**
 * SYCL based element-wise addition
 * for each element of two input array (a, b) a sum is calculated and written to array c.
 * SYCL parallel for is called to process the operation
 */
void VectorAdd(sycl::queue &q, const int64_t *a, const int64_t *b, int64_t *sum, size_t size) {

    sycl::range<1> num_items{size};
    auto e = q.parallel_for(num_items, [=](auto i) { sum[i] = a[i] + b[i]; });
    e.wait();
}

/**
 * This function prints a number of parameters:
 * element count of arrays, the share of CPU processing between 0 and 1
 * starting index of GPU processing within the unified memory space
 * number of openMP threads used on CPU side
 */
void printcfg(config conf) {
    std::cout << "PRINT CONFIG" << std::endl;
    std::cout << "vector_size: " << conf.vector_size << std::endl;
    std::cout << "CPU Share: " << conf.share_cpu << std::endl;
    std::cout << "GPU Start index: " << conf.start_index << std::endl;
    std::cout << "omp threads: " << conf.omp_threads << std::endl;
    std::cout << "Processing mode: " << conf.processing_mode << std::endl;
}
void omp_add(int64_t *a, int64_t *b, int64_t *sum_parallel, config conf) {
    int n_per_thread = conf.vector_size / conf.omp_threads;
    int i;
#pragma omp parallel num_threads(conf.omp_threads)
    {
#pragma omp parallel for shared(a, b, sum_parallel) private(i) schedule(dynamic, n_per_thread),
        for (i = 0; i < conf.start_index; i++) {
            sum_parallel[i] = a[i] + b[i];
        }
    }
}

/**
 * Coproc is called as daphne Extension function to perform an element-wise addition of integer values over two input
 * arrays The calculation is performed in parallel both on GPU and CPU. The CPU side uses OpenMP to perform multicore
 * CPU calculations The GPU side runs as SYCL kernel which can be executed on any SYCL capable device To parallelize the
 * Co-Processing, OpenMP is called in a seperate thread Both CPU and GPU access SYCL unified memory which is a pointer
 * that is valid both in CPU and GPU code parts.
 *
 */
void coproc(DenseMatrix<int64_t> *&res, const DenseMatrix<int64_t> *lhs, const DenseMatrix<int64_t> *rhs,
            DaphneContext *ctx) {
    std::cerr << "Starting coproc() function" << std::endl;

    // preparing pointer to access daphne supplied data objects
    const int64_t *in_a = lhs->getValues();
    const int64_t *in_b = rhs->getValues();
    res = DataObjectFactory::create<DenseMatrix<int64_t>>(lhs->getNumRows(), lhs->getNumCols(), false);
    int64_t *out_res = res->getValues();

    auto selector = sycl::gpu_selector_v; // Force selection of a GPU as Sycl device
    config conf;
    conf.vector_size = lhs->getNumRows() * lhs->getNumCols(); // Fetch number of elements from daphne data

    printcfg(conf);

    // Data is partitioned into a GPU and a CPU part.
    // GPU starts at array index 0
    // CPU starting index is calculated by the percentage of data it calculates
    conf.start_index = (conf.vector_size - 1) * conf.share_cpu;
    if (conf.share_cpu >= 0.99f) {
        conf.start_index = conf.vector_size;
    }

    sycl::queue q(selector, sycl::property::queue::enable_profiling{});

    // Print out the SYCL device information
    std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // allocate unified memory buffers
    int64_t *a = sycl::malloc_shared<int64_t>(conf.vector_size, q);
    int64_t *b = sycl::malloc_shared<int64_t>(conf.vector_size, q);
    int64_t *sum_sequential = sycl::malloc_shared<int64_t>(conf.vector_size, q);
    int64_t *sum_parallel = sycl::malloc_shared<int64_t>(conf.vector_size, q);

    // copy over daphne data into unified memory buffers
    for (int i = 0; i < conf.vector_size; i++) {
        a[i] = in_a[i];
        b[i] = in_b[i];
    }

    int n_per_thread = conf.vector_size / conf.omp_threads;
    std::thread tt(omp_add, a, b, sum_parallel, conf); // spawn additional thread for CPU based openMP part
    VectorAdd(q, a + conf.start_index, b + conf.start_index, sum_parallel + conf.start_index, // call SYCL kernel on GPU
              conf.vector_size - conf.start_index);
    tt.join();

    // copy over data to daphne result object
    for (int i = 0; i < conf.vector_size; i++) {
        out_res[i] = sum_parallel[i];
    }
}
}