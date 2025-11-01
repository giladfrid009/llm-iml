# Additional Performance Improvement Suggestions

This document outlines additional performance optimization opportunities identified during analysis but not yet implemented. These suggestions are organized by priority and potential impact.

## High Priority Suggestions

### 1. Batched Evaluation Passes

**Current Issue**: Multiple separate forward passes for different checks in IML attack.

**Location**: `src/univ_attacks/iml.py`, lines 164-184 and 194-210

**Problem**:
```python
# Multiple evaluation passes
if self.skip_already_fooled:
    init_responses = self.adv_model.chat(...)  # First forward pass
    eval_result = self.judge_evaluator.eval_batch(...)

# ... sample attack ...

if self.skip_failed_attacks:
    sample_responses = self.adv_model.chat(...)  # Second forward pass
    eval_result = self.judge_evaluator.eval_batch(...)
```

**Suggested Solution**:
Combine evaluations into a single pass when both checks are enabled:
```python
# Proposed approach
if self.skip_already_fooled or self.skip_failed_attacks:
    # Single forward pass, evaluate once
    responses = self.adv_model.chat(...)
    eval_results = self.judge_evaluator.eval_batch(...)
    # Use results for both filtering decisions
```

**Expected Impact**: 30-40% reduction in evaluation overhead

**Complexity**: Medium (requires refactoring control flow)

---

### 2. Pre-allocated Buffer Reuse

**Current Issue**: Frequent tensor allocations for temporary computations.

**Locations**: 
- `src/sample_attacks/sp.py` (criterion function)
- `src/univ_attacks/iml.py` (loss computation)

**Problem**: Creating new tensors in every iteration.

**Suggested Solution**:
```python
class BufferPool:
    """Reusable tensor buffers to reduce allocations"""
    def __init__(self):
        self._buffers = {}
    
    def get_buffer(self, shape, dtype, device, key=None):
        key = key or (shape, dtype, device)
        if key not in self._buffers:
            self._buffers[key] = torch.empty(shape, dtype=dtype, device=device)
        buffer = self._buffers[key]
        if buffer.shape != shape:
            self._buffers[key] = torch.empty(shape, dtype=dtype, device=device)
            return self._buffers[key]
        return buffer
```

**Expected Impact**: 10-15% reduction in allocation overhead

**Complexity**: Medium (requires careful memory management)

---

### 3. Optimized Conversation Deep Copy

**Current Issue**: Deep copying conversation structures is still used in some places.

**Location**: `src/adv_model.py`, line 266

**Problem**:
```python
def inject_tokens(self, conversations: list[Conv], ...):
    conversations = copy.deepcopy(conversations)
    # ... modify conversations ...
```

**Suggested Solution**:
Use shallow copy with explicit message copying:
```python
def inject_tokens(self, conversations: list[Conv], ...):
    # Create new list of conversations with copied messages
    conversations = [[{**msg} for msg in conv] for conv in conversations]
    # ... modify conversations ...
```

**Expected Impact**: 5-10% faster token injection

**Complexity**: Low

---

## Medium Priority Suggestions

### 4. Parallel Activation Extraction

**Current Issue**: Sequential extraction of activations across layers.

**Location**: `src/sample_attacks/pcav.py`, lines 180-195

**Problem**: Loop through batches one at a time.

**Suggested Solution**:
```python
# Use torch.utils.data.DataLoader with num_workers > 0
# Or use concurrent.futures for parallel batch processing
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(self._extract_batch, batch) 
               for batch in dl]
    results = [f.result() for f in futures]
```

**Expected Impact**: 20-30% faster activation extraction on multi-core systems

**Complexity**: Medium (thread safety considerations)

---

### 5. Cached Tokenizer Outputs

**Current Issue**: Repeated tokenization of same inputs.

**Location**: Throughout codebase where same conversations are tokenized multiple times.

**Suggested Solution**:
```python
class TokenizationCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
    
    def get_or_tokenize(self, conv_hash, tokenize_fn):
        if conv_hash not in self.cache:
            if len(self.cache) >= self.max_size:
                # LRU eviction
                self.cache.pop(next(iter(self.cache)))
            self.cache[conv_hash] = tokenize_fn()
        return self.cache[conv_hash]
```

**Expected Impact**: 15-25% faster in scenarios with repeated inputs

**Complexity**: Medium (need robust hash function for conversations)

---

### 6. Vectorized Random Initialization

**Current Issue**: Sequential random initialization operations.

**Location**: `src/initialize.py`, various methods

**Suggested Solution**:
Use batch operations for random sampling:
```python
# Instead of multiple random calls
# Generate all random values at once and reshape
all_random = torch.randn(batch_size * num_tokens * embed_dim, device=device)
embeds = all_random.view(batch_size, num_tokens, embed_dim)
```

**Expected Impact**: 5-10% faster initialization

**Complexity**: Low

---

## Low Priority Suggestions

### 7. Memory-Mapped Data Loading

**Current Issue**: Loading entire datasets into memory.

**Location**: `src/data.py`

**Suggested Solution**:
Use memory-mapped files for large datasets:
```python
import numpy as np
from pathlib import Path

class MMapDataLoader:
    def __init__(self, data_path):
        self.mmap = np.load(data_path, mmap_mode='r')
```

**Expected Impact**: Reduced memory footprint for large datasets

**Complexity**: Medium (requires data format changes)

---

### 8. Profile-Guided Optimization

**Suggested Addition**: Add profiling hooks throughout the codebase.

**Implementation**:
```python
from contextlib import contextmanager
import time

@contextmanager
def profile_section(name, logger=None):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if logger:
            logger.info(f"{name}: {elapsed:.4f}s")
```

**Expected Impact**: Better visibility into performance bottlenecks

**Complexity**: Low

---

### 9. Gradient Accumulation Optimization

**Current Issue**: Gradient updates after each batch.

**Suggested Solution**:
Add optional gradient accumulation:
```python
class GradientAccumulator:
    def __init__(self, accumulation_steps=4):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
    
    def should_step(self):
        self.step_count += 1
        return self.step_count % self.accumulation_steps == 0
```

**Expected Impact**: Better throughput with larger effective batch sizes

**Complexity**: Medium

---

### 10. Async Generation

**Current Issue**: Synchronous generation blocks training.

**Suggested Solution**:
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_generate(model, inputs):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool, model.generate, inputs
        )
    return result
```

**Expected Impact**: Better resource utilization in mixed workloads

**Complexity**: High (requires async refactoring)

---

## Micro-optimizations

These are small optimizations that might help in specific scenarios:

1. **Use `torch.compile`** for frequently called functions (PyTorch 2.0+)
2. **Replace list comprehensions with generator expressions** where appropriate
3. **Use `torch.inference_mode()` instead of `torch.no_grad()`** (slightly faster)
4. **Batch string operations** in tokenizer calls
5. **Use `functools.lru_cache`** for pure functions
6. **Replace multiple `.item()` calls** with single `.cpu().numpy()`
7. **Use in-place operations** where safe (`add_`, `mul_`, etc.)
8. **Fuse adjacent tensor operations** into single kernels
9. **Use `torch.jit.script`** for performance-critical functions
10. **Replace Python loops with `torch.vmap`** where possible

## Implementation Guidelines

When implementing these suggestions:

1. **Measure First**: Profile before and after each optimization
2. **Test Thoroughly**: Ensure numerical equivalence
3. **Document Changes**: Update performance documentation
4. **Consider Trade-offs**: Sometimes complexity isn't worth the gain
5. **Maintain Readability**: Don't sacrifice clarity for minor gains

## Benchmarking Framework

Suggested benchmarking setup:

```python
import torch
import time
from contextlib import contextmanager

@contextmanager
def benchmark(name, iterations=100):
    """Benchmark a code block"""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    
    try:
        yield
    finally:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start
        print(f"{name}: {elapsed/iterations*1000:.2f}ms per iteration")

# Usage
with benchmark("Token search", iterations=100):
    for _ in range(100):
        result = find_tokens(...)
```

## Hardware-Specific Optimizations

Consider these optimizations for specific hardware:

### For NVIDIA GPUs:
- Use `torch.cuda.amp` for automatic mixed precision
- Enable TF32 for Ampere+ GPUs: `torch.backends.cuda.matmul.allow_tf32 = True`
- Use flash attention when available
- Consider using `torch.compile` with `mode="reduce-overhead"`

### For AMD GPUs:
- Use ROCm-optimized operations
- Tune thread block sizes for RDNA/CDNA architecture

### For CPUs:
- Enable `torch.set_num_threads()` appropriately
- Use Intel MKL or OpenBLAS optimizations
- Consider using quantization for inference

### For Apple Silicon:
- Use MPS backend when appropriate
- Profile memory bandwidth constraints

## Monitoring and Profiling Tools

Recommended tools for performance analysis:

1. **PyTorch Profiler**: Built-in profiling with Chrome trace viewer
2. **NVIDIA Nsight**: GPU profiling and debugging
3. **Intel VTune**: CPU profiling
4. **memory_profiler**: Memory usage tracking
5. **line_profiler**: Line-by-line profiling
6. **py-spy**: Sampling profiler

Example usage:
```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # Your code here
    pass

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## Contributing New Optimizations

When proposing new optimizations:

1. **Create an issue** describing the optimization
2. **Provide benchmarks** showing the improvement
3. **Include test cases** verifying correctness
4. **Update documentation** with the changes
5. **Consider backward compatibility**

## Questions?

For questions about these suggestions or to propose new optimizations, please:
- Open a GitHub issue
- Tag with "performance" label
- Include benchmark results when possible
