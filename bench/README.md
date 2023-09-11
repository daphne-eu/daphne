# Benchmarking

This document provides key information about benchmarking the implemented in-place update approach to reproduce our results.

**E2E-Benchmark:**

There are currently following cases with different matrix sizes benchmarked:

* [addition.daph](addition.daph)
* [addition_readMatrix.daph](addition_readMatrix.daph)
* [normalize_matrix.daph](normalize_matrix.daph)
* [tranpose.daph](tranpose.daph)
* [kmeans.daph](kmeans.daph)

 We are capturing the following indicators:

* *timings* from DAPHNE by using `--timing` as an argument
* *Peak memory* *consumption*
* (*perf*)

To execute the benchmark, run `﻿$ python3 bench.py.` Prior to that, it is necessary to run [create_matrix_files.daph](create_matrix_files.daph) in order to generate static matrix files that will be stored on disk. The resulting matrices have a total size of 2.3GB and are used in *addition_readMatrix.daph*.

The fine-granular results can be displayed as boxplots with the [draw_graphs.ipynb](draw_graphs.ipynb).

**Microbenchmark:**

The implemented kernels are tested directly in Catch2 using the BENCHMARK feature. This demonstrates the impact of not allocating new memory for a data object, especially in combination when another algorithm is used.

The tag `[inplace-bench]` is deactivated by default, we can run it by executing:

```bash
$ ./test.sh [inplace-bench]
```

The result of the run on the bench VM can be found in file: [catch2_microbench.txt](/bench/results/catch2_microbench.txt).

## System Information

The benchmark was done inside a VM (CX41, scaled up) on hetzner.cloud

**CPU:**

Architecture: x86_64
CPUs: 4x @ 2.1 Ghz
Model name: Intel Xeon Processor (Skylake, IBRS)
L1 cache: 128 KiB
L2 cache: 16 MiB
L3 cache: 16 MiB

Hypervisor vendor: KVM
Virtualization type:  full

**RAM**: 16GB
**Disk Storage**: SSD 80GB

**OS**: Ubuntu 22.04.2 LTS (GNU/Linux 5.15.0-75-generic x86_64)
