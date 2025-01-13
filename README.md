# Triton cache and no cache comparison

This Proof of Concept (PoC) compares the startup time and GPU memory usage of a custom `vllm` configuration that uses Triton flash attention. It evaluates the performance difference between two scenarios:

1. **With Triton cache pre-loaded**: The cache is already present.
2. **Without Triton cache**: The cache is not present.

The results demonstrate that the Triton cache significantly improves performance, speeding up startup time by approximately **20%**.

![Performance Plot](gpu_memory_usage_comparison.png)  
*As shown in the plot, the Triton cache improves performance by ~20%.*

---

## Prerequisites

To run the benchmark, you need to:
1. Install **Triton**.
2. Install the edited `vllm` version with Triton flash attention support:
   - [https://github.com/cmagina/vllm/tree/triton](https://github.com/cmagina/vllm/tree/triton)

---

## Installation

1. Clone this repository:
``` bash
git clone git@github.com:fulvius31/triton-cache-comparison.git && cd triton-cache-comparison
```

2. Ensure you have the required dependencies installed:
- **Triton**
- Edited `vllm` (linked above)

---

## Usage

Run the benchmark by executing the `benchmark.sh` script:

``` bash
./benchmark.sh
```
