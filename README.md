# SparseInferC

**Pointer-Based Sparse-Matrix Architecture for Energy-Efficient
Neural-Network Inference**

## Team Details

-   **Team Number:** Team 4
-   **Batch:** 09
-   **Member 1:** Aryaman Dheeraj Kumar Yadav --- SAP ID: 590032850
-   **Member 2:** Mayank --- SAP ID:590038953
-   **Member 3:** Nisha Dager --- SAP ID: 590031992
    

## Problem Statement

Dense neural-network inference stores and processes every weight, even
when many weights are zero after pruning. Multiplying by zero does not
contribute to the output, but it still consumes computation and memory
access.

SparseInferC demonstrates a pointer-based sparse-matrix approach in C.
It stores only the non-zero weights of a pruned neural network using
Compressed Sparse Row (CSR) format and compares the sparse
implementation with a dense reference implementation.

## Objectives

-   Implement dense and sparse neural-network inference in C.
-   Apply magnitude pruning at different sparsity levels.
-   Store non-zero weights efficiently using CSR.
-   Use pointer-based traversal to process only stored non-zero weights.
-   Verify that sparse and dense implementations produce equivalent
    outputs.
-   Compare MAC operations, memory, estimated energy, execution time,
    and speedup.
-   Save benchmark results in a CSV file.

## Main Features

-   Network: **784 → 256 → 128 → 10**
-   **234,752 total weights**
-   ReLU activation
-   Magnitude pruning
-   CSR sparse-matrix representation
-   Pointer-based sparse matrix-vector multiplication
-   Dense vs. sparse correctness verification
-   Six built-in unit tests
-   Sparsity levels: **0%, 30%, 50%, 60%, 70%, 80%, 90%, 95%, 99%**
-   200 inputs per benchmark pass
-   MAC, memory, energy, runtime, and speedup measurements
-   Automatic output to `results.csv`

## Algorithm

1.  Start the program.
2.  Run the built-in unit tests.
3.  Initialize the **784 → 256 → 128 → 10** neural network.
4.  Generate weights, biases, and test inputs.
5.  Select a sparsity level.
6.  Apply magnitude pruning by setting the smallest-magnitude weights to
    zero.
7.  Convert the pruned dense matrices into CSR format.
8.  Run dense inference, which processes all weights.
9.  Run sparse inference using CSR and pointers, which processes only
    stored non-zero weights.
10. Compare dense and sparse outputs for correctness.
11. Measure MACs, memory footprint, estimated energy, and execution
    time.
12. Repeat for every sparsity level.
13. Save results to `results.csv`.
14. Print `ALL CHECKS PASSED` when all checks succeed.
15. End.

## Simplified Program Flow

``` text
Start
  ↓
Run Unit Tests
  ↓
Initialize Network
  ↓
Choose Sparsity Level
  ↓
Magnitude Pruning
  ↓
Convert to CSR
  ↓
Dense Inference + Sparse Inference
  ↓
Compare Outputs
  ↓
Measure MACs, Memory, Energy and Time
  ↓
Repeat for All Sparsity Levels
  ↓
Save results.csv
  ↓
End
```

## CSR Representation

CSR means **Compressed Sparse Row**. SparseInferC stores:

-   `vals` --- non-zero weight values
-   `col_idx` --- column index of each stored value
-   `row_ptr` --- positions showing where each row starts and ends

This lets the sparse kernel traverse only useful non-zero weights.

## C Programming Concepts Used

-   **Structures (`struct`)** --- CSR matrices, network data,
    statistics, and benchmark rows.
-   **Pointers** --- traversal of non-zero weights and column indices.
-   **Arrays** --- layer sizes, weights, biases, inputs, outputs, and
    sparsity levels.
-   **Functions** --- including `csr_from_dense()`, `csr_matvec()`,
    `dense_matvec()`, `infer()`, and `prune_magnitude()`.
-   **Loops** --- matrix operations, network layers, tests, inputs, and
    benchmark levels.
-   **Conditional statements** --- dense/sparse selection, ReLU,
    pruning, and checks.
-   **Dynamic memory allocation** --- `malloc()` and `free()`.
-   **File handling** --- `fopen()`, `fprintf()`, and `fclose()` for
    `results.csv`.
-   **Mathematical functions** --- `fabsf()`, `floor()`, `sqrtf()`.
-   **Timing** --- `clock_gettime()` for execution-time benchmarking.

## Files

| File | Purpose |
| --- | --- |
| `sparseinferc.c` | Complete C implementation, unit tests, and benchmark |
| `results.csv` | Benchmark results |
| `README.md` | Project documentation |
| `presentation.pptx` | Mini Hackathon presentation |
| `flowchart.png` | Program flowchart |
| `screenshots/sample_output.png` | Screenshot of a full program run |

## Build and Run

### Compile

``` bash
gcc -O2 -std=c11 -Wall -Wextra -o sparseinferc sparseinferc.c -lm
```

### Run

``` bash
./sparseinferc results.csv
```

## Sample Test Cases

### Test Case 1 --- Known 2 × 3 Sparse Matrix

Matrix:

``` text
1  0  2
0  3  0
```

Input: `[1, 2, 3]`\
Bias: `[0.5, -1]`\
**Expected Output:** `[7.5, 5.0]`\
**Actual Output:** `[7.5, 5.0]`\
**Result:** PASS

### Test Case 2 --- Identity Matrix

Input: `[4, -5, 6]`\
**Expected Output:** `[4, -5, 6]`\
**Actual Output:** `[4, -5, 6]`\
**Result:** PASS

### Test Case 3 --- All-Zero Matrix

Input: `[1, 2, 3]`\
Bias: `[7, 8]`\
**Expected Output:** `[7, 8]`\
**Actual Output:** `[7, 8]`\
**Result:** PASS

### Test Case 4 --- 1 × 1 Matrix

Weight: `2.5`\
Input: `4`\
**Expected Output:** `10.0`\
**Actual Output:** `10.0`\
**Result:** PASS

### Test Case 5 --- Magnitude Pruning

100 weights are generated and 75% sparsity is requested.\
**Expected:** 75 weights become zero.\
**Actual:** 75 weights become zero.\
**Result:** PASS

### Test Case 6 --- Empty Middle Row

Matrix:

``` text
1  2  3
0  0  0
4  5  6
```

Input: `[1, 1, 1]`\
**Expected Output:** `[6, 0, 15]`\
**Actual Output:** `[6, 0, 15]`\
**Result:** PASS

This gives a total of **six built-in unit tests**.

## Sample Output

Full program output (6 unit tests, 9 sparsity levels, `ALL CHECKS PASSED`):

![Sample output](file:///C:/Users/aryaman/Downloads/Sparse%20Inference%20Benchmark%20Results.pdf)

## Correctness Verification

At every sparsity level, dense and sparse inference are run on the same
200 inputs. The program checks the maximum absolute output difference
and whether both implementations select the same maximum-output index.

## What Is Measured?

-   Stored non-zero weights
-   Dense and sparse MAC operations
-   Dense and CSR memory footprint
-   Estimated DRAM and SRAM energy
-   Dense and sparse execution time
-   Speedup
-   Maximum absolute output error
-   Output-index match count

## Selected Results

Values from `results.csv`.

| Sparsity | Stored Weights | CSR Memory (KB) | DRAM Energy Sparse / Dense | Speedup |
| --- | --- | --- | --- | --- |
| 0% | 234,752 | 1,835.6 | 1.99x | 1.00x |
| 50% | 117,376 | 918.6 | 1.00x | 2.01x |
| 90% | 23,475 | 184.9 | 0.20x | 10.6x |
| 99% | 2,348 | 19.9 | 0.02x | 92.7x |

Dense reference: **234,752 MACs**, approximately **917.0 KB**, and about
**151.3 µJ** estimated DRAM energy per inference.

## Conclusion

SparseInferC demonstrates that pointer-based CSR representation can
avoid unnecessary processing of zero-valued neural-network weights. At
sufficiently high sparsity, it reduces operations, memory requirements,
estimated energy, and execution time while remaining computationally
equivalent to the dense reference implementation.

## Limitations

-   The benchmark uses generated weights and inputs rather than a
    trained real-world neural-network model.
-   It demonstrates computational equivalence and efficiency, not
    real-world model accuracy after pruning.
-   Energy is analytically estimated, not physically measured.
-   Runtime can vary across computers and between runs, so speedup values
    change slightly each time; correctness, MAC, memory and energy
    values are exact.
-   At 0% sparsity the sparse version is not faster than dense, because
    CSR also stores an index for every weight.

