# SparseInferC

Pointer-Based Sparse-Matrix Architecture for Energy-Efficient Neural-Network Inference.

SparseInferC stores only the non-zero weights of a pruned neural network in a
compressed sparse row (CSR) layout and walks them with raw pointers, so
inference never fetches or multiplies a zero. It is checked against a dense
reference kernel and benchmarked from 0% to 99% sparsity.

## Files

| File | Purpose |
| --- | --- |
| `sparseinferc.c` | Complete implementation, unit tests and benchmark (one C file) |
| `results.csv` | Benchmark output used for the report |
| `README.md` | This file |

## Build and run

```bash
gcc -O2 -std=c11 -Wall -Wextra -o sparseinferc sparseinferc.c -lm
./sparseinferc results.csv
```

Needs only a C compiler and a POSIX system (Linux or macOS). The program runs
six unit tests, then nine sparsity levels, prints a table, writes the CSV and
exits with a non-zero code if any check fails.

## What is measured

Network: 784 -> 256 -> 128 -> 10, ReLU, 234,752 random weights, magnitude
pruned. For every sparsity level:

- correctness: sparse output equals dense output (max difference and predictions)
- MACs and word fetches per inference, counted inside the kernels
- memory: dense bytes against CSR bytes
- energy: analytic model, weights in DRAM (640 pJ/word) and on-chip SRAM (5 pJ/word)
- run time: best of five passes over 200 inputs

## Results (from `results.csv`)

| Sparsity | Stored weights | CSR memory (KB) | DRAM energy sparse / dense | Speedup |
| --- | --- | --- | --- | --- |
| 0% | 234,752 | 1,835.6 | 1.99x | 0.93x |
| 50% | 117,376 | 918.6 | 1.00x | 2.03x |
| 90% | 23,475 | 184.9 | 0.20x | 12.6x |
| 99% | 2,348 | 19.9 | 0.02x | 57.1x |

Dense reference: 234,752 MACs, 917.0 KB, about 151.3 uJ per inference.
Break-even is about 50% sparsity for memory and DRAM energy.

## Limitations

- Weights are random and untrained, so this shows kernel equivalence, not the
  accuracy cost of pruning.
- Energy is an analytic estimate using approximate 45 nm per-operation figures
  (Horowitz, ISSCC 2014), not a power measurement.
- Timings come from a virtual machine and will differ on other hardware.
