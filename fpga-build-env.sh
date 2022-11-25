#!/usr/bin/env bash
#initialize Intel FPGA OpenCL environment
export QUARTUSDIR=/opt/intel/intelFPGA_pro/21.4
source $QUARTUSDIR/hld/init_opencl.sh
echo $INTELFPGAOCLSDKROOT
export ALTERAOCLSDKROOT=$INTELFPGAOCLSDKROOT
#set up BITSTREAM variable for required FPGA image (can be different different for varius implemented kernels)
export BITSTREAM=src/runtime/local/kernels/FPGAOPENCL/bitstreams/sgemm.aocx  # SGEMM computational kernel

