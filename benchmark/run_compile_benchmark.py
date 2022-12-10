#!/usr/bin/python3

import sys
import subprocess
import re

from typing import List


def run_compile_benchmark(file: str):
    time_spent_precompiled = 0
    time_spent_codegen = 0

    ROUNDS = 2

    for i in range(ROUNDS):
        # precompiled
        popen = subprocess.Popen(("../bin/daphne " + file).split(), stdout=subprocess.PIPE)
        popen.wait()
        output = str(popen.stdout.read())
        time_spent_precompiled += int(re.findall(r"\d+", output)[0])

        # codegen
        popen = subprocess.Popen(("../bin/daphne --codegen " + file).split(), stdout=subprocess.PIPE)
        popen.wait()
        output = str(popen.stdout.read())
        time_spent_codegen += int(re.findall(r"\d+", output)[0])

    print(f"Total time spent precompiled: {time_spent_precompiled}, average:"
          f"{time_spent_precompiled/ROUNDS}")
    print(f"Total time spent codegen: {time_spent_codegen}, average:"
          f"{time_spent_codegen/ROUNDS}")



def run_runtime_benchmark(file: str):
    raise NotImplementedError

if __name__ == "__main__":
    print(f"Running compile benchmark for Float32")
    run_compile_benchmark("f32_100_sum.daphne")
    print("")

    print(f"Running compile benchmark for Float64")
    run_compile_benchmark("f64_100_sum.daphne")
    print("")

    run_runtime_benchmark(sys.argv[2])
