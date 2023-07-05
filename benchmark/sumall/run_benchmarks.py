#!/usr/bin/python3

import sys
import subprocess
import re
import pandas as pd

from typing import List

tests = [
    "f32_10x10.daphne",
    "f32_1kb.daphne",
    "f32_10kb.daphne",
    "f32_100kb.daphne",
    "f32_1mb.daphne",
    "f32_10mb.daphne",
    "f32_100mb.daphne",
    "f32_1gb.daphne",
    "f32_5gb.daphne",
    "f32_10gb.daphne",
    #  "f64_10x10.daphne",
    #  "f64_1kb.daphne",
    #  "f64_10kb.daphne",
    #  "f64_100kb.daphne",
    #  "f64_1mb.daphne",
    #  "f64_10mb.daphne",
    #  "f64_100mb.daphne",
    #  "f64_1gb.daphne",
    #  "f64_5gb.daphne",
    #  "f64_10gb.daphne",
]

precompiled_results = []
codegen_results = []


def run_benchmark(file: str):
    time_spent_precompiled = 0
    time_spent_codegen = 0

    ROUNDS = 3

    for i in range(ROUNDS):
        # precompiled
        popen = subprocess.Popen(
            ("daphne " + file).split(), stdout=subprocess.PIPE
        )
        popen.wait()
        result = str(popen.stdout.readline())
        output = str(popen.stdout.readline())
        #  print(output)
        time_spent_precompiled += float(re.findall(r"\d+", output)[0]) / 1e6
        #  print(time_spent_precompiled)

    precompiled_results.append(round(time_spent_precompiled / ROUNDS, 3))
    print(
        f"Total time spent precompiled: {time_spent_precompiled:.3}, average:"
        f"{(time_spent_precompiled/ROUNDS):.3}"
    )

    for i in range(ROUNDS):
        # codegen
        popen = subprocess.Popen(
            ("daphne --codegen " + file).split(), stdout=subprocess.PIPE
        )
        popen.wait()

        result = str(popen.stdout.readline())
        output = str(popen.stdout.readline())
        time_spent_codegen += float(re.findall(r"\d+", output)[0]) / 1e6

    codegen_results.append(round(time_spent_codegen / ROUNDS, 3))

    print(f"Total time spent codegen: {time_spent_codegen}, average:"
          f"{time_spent_codegen/ROUNDS}")


if __name__ == "__main__":
    print(f"Running tests.")
    for test in tests:
        run_benchmark(test)
    print(f"Finished tests.\n")

    df = pd.DataFrame(
        {
            "test_name": [t.replace(".daphne", "") for t in tests],
            "precompiled": precompiled_results,
            "codegen": codegen_results,
        },
        columns=["test_name", "precompiled", "codegen"],
    )
    df.to_csv("results.csv", index=False)

    # since we can't run codegen and codegen_optimized with the same binary I
    # stitch the results together after a seperate run for codegen_optimized
    #  df = pd.DataFrame(
    #      {
    #          'codegen_optimized': codegen_results
    #      }, columns=['codegen_optimized']
    #  )
    #  df.to_csv('codegen.csv', index=False)

    print(df)
