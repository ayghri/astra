# Benchmarking Plan: ADMM vs SparseGPT vs Wanda for 2:4 LLM Pruning

## Goal

Compare three pruning methods on common LLMs under **2:4 structured sparsity**, measuring wikitext perplexity and common benchmarks (MMLU). All pruning is layer-by-layer, one-shot, calibration-based.

## Methods

### 1. SparseGPT (baseline)
- **Code**: `astra/pruners/sparsegpt.py` -> `sparsegpt_prune(weights, hessian, ...)`
- **Interface**: takes `(weights, hessian, blocksize, sparsity, prune_n=2, prune_m=4, percdamp)`
- **Pipeline**: already implemented in `scripts/prune_sparsegpt.py`
- **Propagation**: layer inputs are re-captured after each layer is pruned (error propagation)

### 2. Wanda
- **Code**: needs implementation — `astra/pruners/wanda.py` is empty
- **Algorithm**: saliency = `|W| * ||X||_2` per-output-row, then prune lowest 2-of-4
- **Interface**: `wanda_prune(weight, input_activations, prune_n=2, prune_m=4)` -> pruned weight
- **Key**: no Hessian needed, only activation norms. Cheapest method.

### 3. ADMM (ours)
- **Code**: `autoresearch/admm.py` — currently single-layer standalone
- **Must extract into**: `admm_prune(W0, H, num_iter, kappa, ...)` -> pruned weight
- **Key advantage**: can target `||WX - Y||^2` for arbitrary Y, not just `||(W-W0)X||^2`
  - **Mode A** (fair comparison): Y = W0 @ X (same objective as SparseGPT)
  - **Mode B** (error-correcting): Y = dense model output at this layer (Y from dense forward pass). This corrects upstream pruning error.
- **Needs**: extract ADMM core loop from `main()` into a callable function with signature:
  ```python
  def admm_prune(W0, H, num_iter=2000, kappa=2, psi_beta=0.995, dtype=torch.float32) -> Tensor
  ```

---

## Models

Dense decoder-only architectures only (no MoE). Three tiers:

| Tier   | Model               | HuggingFace ID              | Params | Notes                          |
|--------|---------------------|-----------------------------|--------|--------------------------------|
| Small  | Qwen3.5 0.8B        | `Qwen/Qwen3.5-0.8B`        | 0.8B   | Quick debug loop               |
| Small  | Qwen3.5 2B          | `Qwen/Qwen3.5-2B`          | 2B     | Small but meaningful           |
| Medium | Qwen3.5 4B          | `Qwen/Qwen3.5-4B`          | 4B     | Main small-scale target        |
| Medium | Qwen3.5 9B          | `Qwen/Qwen3.5-9B`          | 9B     | Mid-range                      |
| Medium | Llama 3.1 8B        | `meta-llama/Llama-3.1-8B`   | 8B     | Standard pruning benchmark     |
| Large  | Qwen3.5 27B         | `Qwen/Qwen3.5-27B`         | 27B    | Dense, top of target range     |
| Large  | Gemma 4 31B         | `google/gemma-4-31B`        | 31B    | Dense, latest Google model     |

Avoid MoE variants (Qwen3.5-35B-A3B, Gemma-4-26B-A4B) — structured pruning on routed experts is a different problem.

Start with Qwen3.5-0.8B for debugging, Qwen3.5-4B + Llama-3.1-8B for main results, scale to 27B/31B if time allows.

---

## Pipeline Architecture

One unified script: `autoresearch/bench_prune_llm.py`

### Step 1: Setup
```python
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
c4_data = get_c4(num_samples=1024, seq_len=2048, tokenizer=tokenizer)
```

### Step 2: Baseline evaluation
Use `quantkit/experiments/eval_ppl.py`'s `evaluate(model, tokenizer)` for wikitext PPL.

### Step 3: Layer-by-layer pruning loop
For each decoder layer:
1. **Capture inputs**: use `LLMIOCatcher` to get layer inputs from the (partially pruned) model
2. **For each linear sublayer** (q/k/v/o_proj, gate/up/down_proj):
   a. Compute H = X^T X / N from captured activations
   b. Apply chosen pruning method:
      - **SparseGPT**: `sparsegpt_prune(W, H, blocksize=128, prune_n=2, prune_m=4, percdamp=0.01)`
      - **Wanda**: `wanda_prune(W, X_norms, prune_n=2, prune_m=4)`
      - **ADMM**: `admm_prune(W, H, num_iter=2000, kappa=2)`
   c. Write pruned weights back to model
3. **Propagate**: re-capture inputs for next layer through the now-pruned model
4. **Periodic eval**: wikitext PPL every N layers (from `prune_sparsegpt.py` pattern)

### Step 4: Final evaluation

**Wikitext perplexity** (primary metric):
- `evaluate(model, tokenizer)` from `quantkit/experiments/eval_ppl.py`
- Reports: `word_perplexity`, `byte_perplexity`, `bits_per_byte`

**Common benchmarks** (pattern from `quantkit/experiments/quant_full_lmeval_multi.py`):
```python
TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande",
         "boolq", "lambada_openai"]
```
Run via `lm_eval.evaluator.simple_evaluate(model=hf_model, tasks=[task], num_fewshot=0)`.
Use `acc_norm` for hellaswag, arc_challenge, winogrande; `acc` for the rest.

**MMLU** (optional, expensive):
```python
simple_evaluate(model=hf_model, tasks=["mmlu"], num_fewshot=5)
```

### Step 5: Results collection
CSV with columns: `model, method, word_ppl, byte_ppl, bits_per_byte, hellaswag, arc_easy, arc_challenge, piqa, winogrande, boolq, lambada, mmlu, prune_time_s`

---

## Implementation Tasks

### Task 1: Extract ADMM into callable function
From `autoresearch/admm.py`, extract core into `astra/pruners/admm.py`:
```python
def admm_prune(
    W0: Tensor,          # [M, K] original weight
    H: Tensor,           # [K, K] Hessian (X^T X / N)
    num_iter: int = 2000,
    kappa: int = 2,       # nnz per group of 4
    psi_beta: float = 0.995,
    k_weight: float = 2999.0,
    dtype=torch.float32,
) -> Tensor:             # [M, K] pruned weight
```
This wraps the ADMM loop + optional batched refinement solve.

### Task 2: Implement Wanda
In `astra/pruners/wanda.py`:
```python
def wanda_prune(
    W: Tensor,           # [M, K] weight
    X: Tensor,           # [N, K] input activations
    prune_n: int = 2,
    prune_m: int = 4,
) -> Tensor:             # [M, K] pruned weight
```
Saliency = `|W_ij| * ||X_j||_2`, prune lowest `prune_n` per group of `prune_m`.

### Task 3: Build unified benchmark script
`autoresearch/bench_prune_llm.py` following the pipeline above. Key design:
- `--model`: HuggingFace model ID
- `--method`: one of `sparsegpt`, `wanda`, `admm`
- `--device`: cuda device
- `--eval-tasks`: `wikitext`, `mmlu`, or `wikitext,mmlu`
- Results to CSV

### Task 4: Run experiments

Phase 1 — Debug on Qwen3-0.6B (1 GPU, ~30 min total):
```bash
for method in sparsegpt wanda admm; do
    python autoresearch/bench_prune_llm.py --model Qwen/Qwen3-0.6B --method $method
done
```

Phase 2 — Main results on Qwen3-4B + Llama-3.1-8B (parallel on 2 GPUs):
```bash
# GPU 0: Qwen3-4B
# GPU 1: Llama-3.1-8B
```

Phase 3 — Scale to 14B/27B if results are promising.

---

## GPU Memory Budget

| Model   | bf16 size | + H matrices | + activations | Total est. |
|---------|-----------|-------------|--------------|------------|
| 0.8B    | 1.6 GB    | ~0.1 GB     | ~2 GB        | ~4 GB      |
| 4B      | 8 GB      | ~0.5 GB     | ~4 GB        | ~14 GB     |
| 8-9B    | 18 GB     | ~1 GB       | ~6 GB        | ~26 GB     |
| 27B     | 54 GB     | ~3 GB       | ~10 GB       | ~70 GB     |
| 31B     | 62 GB     | ~3 GB       | ~10 GB       | ~76 GB     |

- Up to 9B: single 80GB A100/H100
- 27B+: offload model to CPU, prune one layer at a time on GPU (existing pattern in `prune_sparsegpt.py`), or 2-GPU split

---

## Expected Output

A comparison table per model:

| Method       | PPL  | HellaSwag | ARC-C | WinoGrande | BoolQ | Time/layer |
|-------------|------|-----------|-------|------------|-------|------------|
| Dense       | ?    | ?         | ?     | ?          | ?     | -          |
| SparseGPT   | ?    | ?         | ?     | ?          | ?     | ~1s        |
| Wanda       | ?    | ?         | ?     | ?          | ?     | <0.1s      |
| ADMM        | ?    | ?         | ?     | ?          | ?     | ~10s       |
| ADMM (corr) | ?    | ?         | ?     | ?          | ?     | ~10s       |

The key hypothesis: ADMM in error-correcting mode (Mode B: Y = dense output) should show meaningfully better PPL than all other methods, especially on deeper models where error compounds more.
