# Initial DAPHNE Prototype [WIP]

Initial prototype of the DAPHNE system.
Takes a plain text file written in *DaphneDSL* as input, parses, optimizes, JIT-compiles, and executes it.
Prints script outputs to `stdout`.

## 0. Requirements

Please ensure that your system meets the following requirements before trying to build the prototype.

- operation system
  - GNU/Linux
- software
  - cmake
  - ninja
  - clang
  - lld
  - uuid-dev
- hardware
  - about 3.5 GB disk space (only for LLVM/MLIR)

## 1. Build

Simply build the prototype using `./build.sh`.
When you do this the *first* time, it will also download and build antlr4 as well as clone and build LLVM/MLIR.
The latter takes very long (can be about 30 minutes).
All *following* build should take only a few seconds.
To clean, simply delete the `build`-directory.

## 2. Run

Write a little DaphneDSL script or use `example.daphne`...

```
def main() {
    let x = 1;
    let y = 2;
    print(x + y);

    let m = rand(2, 3, 0, 1.0);
    print(m);
    print(m + m);
}
```

... and execute it as follows: `build/bin/daphnec example.daphne`.
