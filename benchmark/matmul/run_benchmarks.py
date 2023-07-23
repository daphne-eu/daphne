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
    #  "f32_1gb.daphne",
    #  "f32_5gb.daphne",
    #  "f32_10gb.daphne",
    #  "f64_10x10.daphne",
    #  "f64_1kb.daphne",
    #  "f64_10kb.daphne",
    #  "f64_100kb.daphne",
    #  "f64_1mb.daphne",
    #  "f64_10mb.daphne",
    #  "f64_1gb.daphne",
    #  "f64_5gb.daphne",
    #  "f64_10gb.daphne",
    #  "f64_32x32.daphne",
    #  "f64_64x64.daphne",
    #  "f64_512x512.daphne",
    #  "f64_1024x1024.daphne",
    #  "f64_2048x2048.daphne",
    #  "f64_4096x4096.daphne"
]

precompiled_results = []
codegen_results = []


def run_benchmark(file: str):
    time_spent_precompiled = 0
    time_spent_codegen = 0

    ROUNDS = 3

    print(file)
    for i in range(ROUNDS):
        # precompiled
        popen = subprocess.Popen(
            ("daphne " + file).split(), stdout=subprocess.PIPE
        )
        popen.wait()
        popen.stdout.readline()
        result = str(popen.stdout.readline())
        output = str(popen.stdout.read())
        time_spent_precompiled += float(re.findall(r"\d+", output)[0]) / 1e6

    precompiled_results.append(round(time_spent_precompiled / ROUNDS, 3))
    print(
        f"Total time spent precompiled: {time_spent_precompiled:.3}"
    )

    for i in range(ROUNDS):
        # codegen
        popen = subprocess.Popen(
            ("daphne --codegen " + file).split(), stdout=subprocess.PIPE
        )
        popen.wait()
        popen.stdout.readline()
        result = str(popen.stdout.readline())
        output = str(popen.stdout.read())
        time_spent_codegen += float(re.findall(r"\d+", output)[0]) / 1e6

    codegen_results.append(round(time_spent_codegen / ROUNDS, 3))

    print(f"Total time spent codegen: {time_spent_codegen:.3}")


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
    df.to_csv("results_handwritten_mm.csv", index=False)

    print(df)
