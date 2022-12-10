#!/usr/bin/python3

import sys
import subprocess
import re

from typing import List


def run_compile_benchmark(file: str):
    time_spent_precompiled = 0
    time_spent_codegen = 0

    for i in range(100):
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

    print(f"Total time spent precompiled: {time_spent_precompiled}, average: {time_spent_precompiled/100}")
    print(f"Total time spent codegen: {time_spent_codegen}, average:"
          f"{time_spent_codegen/100}")



def run_runtime_benchmark(file: str):
    raise NotImplementedError

if __name__ == "__main__":
    if sys.argv[1] == "compile":
        run_compile_benchmark(sys.argv[2])
    if sys.argv[1] == "runtime":
        run_runtime_benchmark(sys.argv[2])
