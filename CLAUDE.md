# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is `tpu-inference`, a hardware plugin for vLLM that provides high-performance LLM inference on Google TPUs. It unifies JAX and PyTorch model execution under a single backend, allowing PyTorch models to run efficiently on TPU without code changes while also supporting native JAX implementations.

**Key Design Principles:**
- Unified backend: JAX and PyTorch models share the same lowering path
- Framework flexibility: Run PyTorch model definitions on TPU via torchax, or use native JAX implementations
- vLLM standardization: Maintains the same user experience, telemetry, and interface as vLLM

**Compatible TPU Generations:**
- Recommended: v5e, v6e
- Experimental: v3, v4, v5p

## Architecture

The codebase follows a layered architecture with framework-specific and common components:

### Directory Structure

**Core Components:**
- `tpu_inference/core/`: Core TPU execution engine
  - `core_tpu.py`: Main TPU execution logic, disaggregated execution support
  - `sched/`: Scheduling logic for request handling
- `tpu_inference/runner/`: Model execution orchestration
  - `tpu_runner.py`: Main runner coordinating model execution
  - `compilation_manager.py`: JAX compilation and caching
  - `kv_cache_manager.py`: KV cache management
  - `input_batch.py`: Input batch handling
- `tpu_inference/worker/`: Worker processes for distributed execution
- `tpu_inference/executors/`: Distributed execution via Ray

**Model and Layer Implementations:**
- `tpu_inference/models/`:
  - `common/`: Shared model utilities
  - `jax/`: Native JAX model implementations (llama3, llama4, qwen2/3, deepseek_v3, etc.)
  - `vllm/`: PyTorch model wrappers (via torchax)
- `tpu_inference/layers/`:
  - `common/`: Framework-agnostic layers
  - `jax/`: JAX-specific layers
  - `vllm/`: vLLM/PyTorch-specific layers

**Kernels:**
- `tpu_inference/kernels/`: High-performance TPU kernels
  - `flash_attention/`: Flash attention implementations
  - `ragged_paged_attention/`: Paged attention for variable-length sequences
  - `fused_moe/`: Fused mixture-of-experts kernels
  - `quantized_matmul/`: Quantization kernels
  - `mla/`: Multi-head latent attention (DeepSeek)
  - `collectives/`: Distributed communication primitives

**Configuration and Utilities:**
- `tpu_inference/envs.py`: Centralized environment variable management (ALL env var access should route through this)
- `tpu_inference/env_override.py`: Environment variable overrides (imported first)
- `tpu_inference/tpu_info.py`: TPU hardware information utilities
- `tpu_inference/utils.py`: General utilities
- `tpu_inference/platforms/`: Platform detection and configuration

### Model Implementation Types

The codebase supports two model implementation strategies:

1. **JAX Native Models** (`tpu_inference/models/jax/`):
   - Written in JAX/Flax for maximum TPU performance
   - Examples: llama3, llama4, qwen2, qwen3, deepseek_v3, llama_guard_4
   - Require manual model definition but offer best performance

2. **vLLM Native Models** (`tpu_inference/models/vllm/`):
   - Use PyTorch model definitions from vLLM upstream
   - Converted to JAX via torchax at runtime
   - Broader model support with minimal code changes
   - Managed by `vllm_model_wrapper.py`

## Development Commands

### Installation

```bash
# Standard installation
pip install -e .

# For TPU v7x support (requires additional dependencies)
pip install -r requirements_v7x.txt
```

### Testing

**Run all tests:**
```bash
pytest
```

**Run specific test file:**
```bash
pytest tests/test_envs.py
```

**Run specific test:**
```bash
pytest tests/test_envs.py::test_env_variable_name
```

**Run tests by category:**
```bash
# Core functionality tests
pytest tests/core/

# Model tests
pytest tests/models/

# Layer tests
pytest tests/layers/

# Kernel tests
pytest tests/kernels/

# End-to-end tests
pytest tests/e2e/
```

### Linting and Formatting

**Install pre-commit hooks:**
```bash
pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

**Run linting/formatting manually:**
```bash
pre-commit run --all-files
```

**Individual formatters:**
- `yapf`: Python code formatting
- `isort`: Import sorting
- `ruff`: Linting
- `clang-format`: C++/CUDA formatting
- `pymarkdown`: Markdown linting

### Running Models

**Offline inference (basic usage):**
```bash
python examples/offline_inference.py --model meta-llama/Llama-3.2-1B-Instruct
```

**Multi-modal inference:**
```bash
python examples/multi_modal_inference.py --model <MODEL_NAME>
```

**LoRA inference:**
```bash
python examples/offline_lora_inference.py --model <BASE_MODEL> --lora <LORA_PATH>
```

**Using vLLM API:**
```python
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
outputs = llm.generate(["Hello, my name is"], sampling_params)
```

### Environment Variables

Access ALL environment variables through `tpu_inference.envs` module (never use `os.getenv` directly):

**Important Environment Variables:**
- `JAX_PLATFORMS`: Platform selection ("tpu", "cpu", "proxy")
- `TPU_ACCELERATOR_TYPE`: TPU type (e.g., "v5litepod-16", "v6e-8")
- `TPU_NAME`: TPU resource name
- `TPU_WORKER_ID`: Worker ID for multi-host setups
- `SKIP_JAX_PRECOMPILE`: Skip JAX precompilation (useful for quick tests)
- `MODEL_IMPL_TYPE`: Model implementation type (default: "flax_nnx")
- `NEW_MODEL_DESIGN`: Enable new model design (set to "1")
- `VLLM_XLA_CHECK_RECOMPILATION`: Check for unexpected recompilations

**Disaggregated Serving:**
- `PREFILL_SLICES`: Slice configuration for prefill workers
- `DECODE_SLICES`: Slice configuration for decode workers

## CI/CD and Buildkite

The project uses Buildkite for continuous integration. Configuration lives in `.buildkite/`:

**Pipeline Files:**
- `pipeline_jax.yml`: JAX model testing pipeline
- `pipeline_torch.yml`: PyTorch model testing pipeline
- `main.yml`: Main CI pipeline
- `nightly_releases.yml`: Nightly release builds

**Adding New Models to CI:**

For JAX/TPU-optimized models:
```bash
cd .buildkite/pipeline_generation
python add_model_to_ci.py --model-name meta-llama/Llama-3.1-8B --queue tpu_v6e_queue
```

For vLLM-native models:
```bash
python add_model_to_ci.py --model-name <MODEL_NAME> --queue tpu_v6e_queue --type vllm-native
```

After generation, fill in TODOs in `.buildkite/models/<MODEL>.yml`:
1. Unit test command
2. Accuracy target
3. Performance benchmark target (TPU-optimized only)

**Adding New Features to CI:**
```bash
python add_feature_to_ci.py --feature-name "my feature" --queue tpu_v6e_queue
```

## Code Organization Patterns

### Framework-Specific Code

The codebase segregates JAX and vLLM code into separate directories:
- Place JAX-only code in `*/jax/` subdirectories
- Place vLLM/PyTorch-only code in `*/vllm/` subdirectories
- Place shared code in `*/common/` subdirectories

### Environment Variable Management

**CRITICAL:** Always access environment variables through `tpu_inference.envs`:

```python
# Correct
from tpu_inference import envs
if envs.SKIP_JAX_PRECOMPILE:
    ...

# Incorrect (will break centralized management)
import os
if os.getenv("SKIP_JAX_PRECOMPILE"):
    ...
```

### Model Development

When implementing new JAX models:
1. Add model implementation to `tpu_inference/models/jax/<model_name>.py`
2. Add corresponding layers to `tpu_inference/layers/jax/` if needed
3. Add unit tests to `tests/models/jax/test_<model_name>.py`
4. Add to CI pipeline using `add_model_to_ci.py`
5. Update model config in vLLM if needed

When supporting new vLLM models:
1. Ensure the PyTorch model definition exists in vLLM upstream
2. Test compatibility via `vllm_model_wrapper.py`
3. Add unit tests to `tests/models/vllm/`
4. Add to CI pipeline using `add_model_to_ci.py --type vllm-native`

## Important Implementation Details

### JAX Compilation

- JAX uses JIT compilation with caching in `compilation_manager.py`
- First run triggers compilation (can be slow), subsequent runs are fast
- Use `SKIP_JAX_PRECOMPILE=1` to skip warmup for quick testing
- Check for unexpected recompilation with `VLLM_XLA_CHECK_RECOMPILATION=1`

### Multi-host Execution

- TPU pods span multiple hosts with multiple chips per host
- Worker 0 coordinates I/O operations
- All workers must execute the same compiled code
- Communication via JAX collectives (all-reduce, all-gather, etc.)

### KV Cache Management

- Paged attention with block-based KV cache
- Managed by `kv_cache_manager.py`
- Blocks allocated dynamically based on sequence length
- Supports variable-length sequences efficiently

### Disaggregated Serving

- Separate prefill and decode workers for better resource utilization
- Configure via `PREFILL_SLICES` and `DECODE_SLICES`
- Managed by `core/disagg_executor.py` and `core/disagg_utils.py`

### Quantization Support

- Quantized matmul kernels in `kernels/quantized_matmul/`
- Test with `QUANTIZATION=True` environment variable
- Integration tests in `tests/test_quantization.py`

## Testing Best Practices

- Add unit tests for all new layers and models
- Add integration tests for end-to-end functionality
- Use `tests/test_base.py` utilities for vLLM config setup
- For models, test both accuracy and performance
- Mock vLLM components when testing tpu_inference-specific logic
- Use `parameterized` for testing multiple configurations

## External Resources

- [Documentation](https://docs.vllm.ai/projects/tpu/en/latest/)
- [vLLM Project](https://github.com/vllm-project/vllm)
- [Developer Slack](https://slack.vllm.ai) (#sig-tpu channel)
- [User Forum](https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27)
- [TPU Documentation](https://cloud.google.com/tpu/docs)
- [JAX Documentation](https://jax.readthedocs.io)
