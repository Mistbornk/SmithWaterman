# SmithWaterman CUDA Version

C++17 implementations of scalar overlap alignment, SIMD local alignment and CUDA overlap alignment.
The backends currently have different semantics and known correctness limitations; see [test design and audit](test/README.md).

## Build and test

CPU-only build (CMake 3.18+, C++17 compiler):

```sh
cmake -S . -B build -DSW_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

CUDA build (add CUDA toolkit and a working GPU driver for GPU execution):

```sh
cmake -S . -B build_cuda -DSW_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_cuda --parallel 2
ctest --test-dir build_cuda --output-on-failure
```

`sw.cuda` is explicitly **skipped** (exit 77) if no usable CUDA device exists.
CPU tests never require a GPU. Correctness tests do not run benchmarks or modify FASTA files.
`SW_NATIVE_ARCH=ON` opts into host-specific SIMD compilation; default builds remain portable across the compiler target architecture.
`BUILD_TESTING=OFF` disables tests; `SW_BUILD_BENCHMARKS=OFF` disables the benchmark.

## Benchmark

```sh
./build/test/sw-benchmark --backend scalar --batch 16 --length 128 --threads 4
./build/test/sw-benchmark --backend simd --batch 16 --length 128 --threads 4
./build_cuda/test/sw-benchmark --backend cuda --mode batch --batch 16 --length 128
```

Use `--help` for seed, warmup, repetitions and workload limits. Output is CSV.
Measurements include host allocations, dispatch, transfers, computation and traceback; they are not kernel-only timings.
Compare the same backend, semantics, dataset, scoring, machine and compiler configuration when evaluating an optimization.
SIMD/scalar speed ratios are not equivalent-algorithm speedups.

## Optional FASTA export

```sh
./test/generate_fasta.sh 1000 70 --seed 42 --out-dir /tmp/sw-fixture-new
```

Requires Python 3. The output directory must not already exist. Tests and benchmarks generate their own deterministic inputs in memory.
