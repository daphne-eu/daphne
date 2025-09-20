<!--
Copyright 2021 The DAPHNE Consortium

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# FPGA Configuration

## System Requirements

Using DAPHNE's build script with FPGA kernels support requires the additional `QUARTUSDIR` environment variable to be defined.
An example command is presented in `fpga-build-env.sh` or in the following command:

```bash
export QUARTUSDIR=/opt/intel/intelFPGA_pro/21.4
```

To build DAPHNE with FPGA kernels support, the flag `--fpgaopencl` has to be used:

```bash
./build.sh --fpgaopencl
```

To run the pre-compiled FPGA OpenCL kernels included in the DAPHNE repository, an installed and configured FPGA device is required.
Our example kernels have been tested using an [Intel(R) PAC D5005 card](https://www.intel.com/content/www/us/en/products/sku/193921/intel-fpga-pac-d5005/specifications.html).

DAPHNE contains some example linear algebra kernels developed using the [T2SP framework](https://github.com/IntelLabs/t2sp/blob/master/README.md).
The example pre-compiled FPGA kernels can be used at DaphneDSL level.
To prepare the system for the pre-compiled FPGA kernels, some FPGA and OpenCL environment variables are required.
The easiest way to set up the required variables is to use the `init_opencl.sh` script from installed Intel(R) Quartus software or from the
Intel(R) OpenCL RTE or Intel(R) OpenCL SDK packages.

Example script usage:

```bash
source /opt/intel/intelFPGA_pro/21.4/hld/init_opencl.sh
```

For additional details, see the [Intel guide](https://www.intel.com/content/www/us/en/docs/programmable/683550/18-1/standard-edition-getting-started-guide.html)
or the [SDK for OpenCL](https://www.intel.com/content/www/us/en/software/programmable/sdk-for-opencl/overview.html).

### Pre-compiled FPGA Kernels

To use a pre-compiled FPGA kernel, an FPGA image is required (`*.aocx`). The FPGA device has to be programmed with a particular image which contains the required kernel implementation.
An example FPGA programming command using example FPGA image:

```bash
aocl program acl0 src/runtime/local/kernels/FPGAOPENCL/bitstreams/sgemm.aocx
```

Additionally, the `BITSTREAM` environment variable has to be defind in the system as follows:

```bash
export BITSTREAM=src/runtime/local/kernels/FPGAOPENCL/bitstreams/sgemm.aocx
```

When another FPGA image contains an implementation for another required computational kernel, then the FPGA device has to be reprogrammed and the `BITSTREAM` environment variable has to be changed.
