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

FPGA configuration for usage in DAPHNE

## System requirments

Daphne build script for FPGA kernels support requires additional QUARTUSDIR system variable definition.
Example command is presented in fpga-build-env.sh or in the following command:

`export QUARTUSDIR=/opt/intel/intelFPGA_pro/21.4`

To build the Daphne with the FPGA support `-fpgaopencl` flag has to be used:

`./build.sh --fpgaopenc`

To run developed or precompiled, included in Daphne repository FPGA OpenCL kernels an installedand configured  FPGA device is required.
Our example kernels have been tested using [Intel(R) PAC D5005 card](https://www.intel.com/content/www/us/en/products/sku/193921/intel-fpga-pac-d5005/specifications.html)

DAPHNE contains some example linear algebra kernels developed using [T2SP framework](https://github.com/IntelLabs/t2sp/blob/master/README.md).
Example precompiled FPGA kernels can be usedon  DAPHNE DSL description level.
To prepare the system for the precompiled FPGA kernels some FPGA and OpenCL system variables are required.
The easiest way to set up required varables is to use the init_opencl.sh script from installed Intel(R) Quartus sowtware or from the
Intel(R) OpenCL RTE or Intel(R) OpenCL SDK packages.

Example script usage:
`source /opt/intel/intelFPGA_pro/21.4/hld/init_opencl.sh`

For additional details please look into the [Intel guide](https://www.intel.com/content/www/us/en/docs/programmable/683550/18-1/standard-edition-getting-started-guide.html)
or [SDK for openCL](https://www.intel.com/content/www/us/en/software/programmable/sdk-for-opencl/overview.html).

### Precompiled FPGA Kernels

To use a precompiled FPGA kernel a FPGA image is required (*.aocx). FPGA device has to programmed with particular image which contains required kernel implementation.
Example FPGA programming command using example FPGA image:

`aocl program acl0 src/runtime/local/kernels/FPGAOPENCL/bitstreams/sgemm.aocx`

Additionally the BITSTREAM variable has to be defind in the system.
Please look into the following example:

`export BITSTREAM=src/runtime/local/kernels/FPGAOPENCL/bitstreams/sgemm.aocx`

When another FPGA image contains implementation for another required computational kernel then FPGA device has to be reprogrammed and BITSTREAM variable value has to be changed.
