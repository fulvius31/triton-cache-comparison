# Triton Cache Performance Comparison

![Performance Plot](gpu_memory_usage_comparison.png)  
*Triton cache improves startup performance by ~20%*

## Proof of Concept

This benchmark compares GPU memory usage and startup performance of a custom `vllm` configuration using Triton flash attention in two scenarios:

1. **With Triton cache pre-loaded** - Cache exists from previous run
2. **Without Triton cache** - Clean cache state

Key findings:
- Triton cache reduces startup time by approximately **20%**
- More consistent memory usage patterns with cached kernels
- Improved resource utilization during initial model loading

## Prerequisites

### Mandatory Requirements
- [Triton](https://openai.com/research/triton) installed
- Custom `vllm` fork with Triton support:
  ```bash
  git clone -b triton https://github.com/cmagina/vllm.git
  cd vllm && pip install -e .
  ```

### Hardware Requirements
- NVIDIA GPU (CUDA) or AMD GPU (ROCm)

## Usage

### Basic Benchmark
```bash
./benchmark.sh --arch [cuda|rocm]
```

### Advanced Options
```bash
# Custom cache location and script
./benchmark.sh \
  --arch cuda \
  --triton-cache-dir ~/alternate_cache \
  --script ./custom_script.py
```

### Expected Output
1. `gpu_usage_log.csv` - Time-series memory data
2. `gpu_memory_usage_comparison.png` - Visualization plot

## Technical Details

### Benchmark Process
1. **Cold Start** (no cache):
   - Purge existing Triton cache
   - Run inference script
   - Log GPU memory at 1Hz frequency

2. **Warm Start** (with cache):
   - Reuse generated kernels
   - Run identical inference script
   - Compare memory/time metrics

### Key Configuration
```bash
export VLLM_ATTENTION_BACKEND=TRITON_FLASH  # Required for Triton support
export TRITON_CACHE_DIR="~/.triton/cache"  # Default cache location
```

## License
Apache 2.0 [LICENSE](LICENSE)
