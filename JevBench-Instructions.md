# JevBench Measurement & Benchmark Guide for Gevva0 (Linux / CUDA)

This guide provides complete instructions for faithfully measuring **Gevva0** on Linux machines with NVIDIA CUDA GPUs.

---

## 1. Executive Summary & Recommended Measurement Configurations

We request measuring Gevva0 under two official configurations on the JevBench leaderboard:

| Configuration | Activated Tiers | Models Required | Key Benchmark Characteristics |
| :--- | :--- | :--- | :--- |
| **Gevva0 (4-Tier Cascade)**<br/>*(Primary Submission)* | **Tier -1 + Tier 0 + Tier 1 + Tier 2** | `Gemma 4 26B-A4B MoE` (LLM)<br/>`EmbeddingGemma-2` (ONNX) | **#1 Intelligence (84.62) & Overall Accuracy (89.17%).**<br/>• Absorbing Markov Chains exit early on trivial cases (~16–25 ms)<br/>• Hot-KV Bounded CoT rescues hard temporal/arithmetic tasks (79.28% hard accuracy)<br/>• Median latency (`p50`): **44.0 ms** |
| **Gevva0 (Tier 1 Fast-Path)**<br/>*(Zero-Token Baseline)* | **Tier 1 Only** | `Gemma 4 26B-A4B MoE` (LLM) | **#1 Composite Score (87.34 raw / 81.40 GPU-adjusted).**<br/>• **0 generated tokens** (pure logit readout, zero token cost)<br/>• Median latency (`p50`): **37.5 ms** |

> **If your protocol only allows measuring a single configuration:**
> Please measure **Gevva0 (4-Tier Cascade)** (`./run_linux.sh --full-cascade`). It represents the complete cascading architecture and shows how local geometric routing combined with hot-KV verification outperforms frontier reasoning models on hard multi-hop tasks.

---

## 2. Environment Setup (Linux + CUDA)

### Prerequisites
* Linux x86_64 with NVIDIA GPU (CUDA 12.1+ / 12.4+ supported)
* Python 3.12 (or 3.11+)

### Installation
```bash
# Clone the repository
git clone https://github.com/your-org/gevva0-linux.git
cd gevva0-linux

# Install dependencies (fetches pre-built CUDA wheels for llama-cpp-python)
pip install -r requirements.txt
```

---

## 3. Model Weights Directory Structure

Ensure model weights are placed or symlinked into the `models/` directory:

```
models/
├── gemma-4-26B-A4B-it-qat-UD-Q4_K_XL/
│   ├── gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf   # (~14 GB, Required)
│   └── mmproj-BF16.gguf                         # (~1.2 GB, Optional, for vision tasks)
│
└── embeddings/                                  # (Required for Full Cascade)
    └── embeddinggemma-2-onnx/                   # (~1.8 GB ONNX)
        ├── model.onnx
        ├── model.onnx_data
        ├── tokenizer.json
        └── tokenizer.model
```

*Quick symlink if weights reside elsewhere:*
```bash
ln -s /path/to/downloaded/models/* models/
```

---

## 4. Launching the Gateway

### Option A: 4-Tier Cascade (Primary Submission)
Launches the full cascading pipeline (Rules → Markov Router → Fast-Path → Hot-KV CoT):

```bash
./run_linux.sh --full-cascade
```

*Equivalent explicit Python command:*
```bash
python -m gevva0.cli serve \
  --host 0.0.0.0 --port 8000 \
  --model models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  --mmproj models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL/mmproj-BF16.gguf \
  --threshold 0.45 \
  --tier0
```

---

### Option B: Tier 1 Fast-Path Only (Zero-Token Baseline)
Runs pure logit extraction with zero autoregressive token generation:

```bash
python -m gevva0.cli serve \
  --host 0.0.0.0 --port 8000 \
  --model models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  --mmproj models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL/mmproj-BF16.gguf \
  --threshold 0.0
```

---

### Option C: Minimum Tier 1 + Tier 2 (LLM Only, No ONNX Embedding)
If you wish to evaluate the cascade without mounting the ONNX embedding model:

```bash
./run_linux.sh
```

---

## 5. Decision API Specification

Send decision requests via HTTP POST to `http://127.0.0.1:8000/v1/decision`:

```bash
curl -X POST http://127.0.0.1:8000/v1/decision \
  -H "Content-Type: application/json" \
  -d '{
    "context": "Procurement Policy 4.1: Purchases >$1,500 require VP sign-off. Developer tooling pre-authorized up to $10,000 under platform budget. Scenario: Annual $4,200 profiling tool requested under platform budget.\n\nQuestion: What approval level is required?",
    "options": {
      "A": "Formal VP signature and secondary review",
      "B": "CFO escalation",
      "C": "Engineering Manager approval under platform budget",
      "D": "Automatic rejection"
    }
  }'
```

### JSON Response Example:
```json
{
  "test_id": null,
  "decision": "C",
  "decision_text": "Engineering Manager approval under platform budget",
  "confidence": 0.8921,
  "latency_ms": 37.45,
  "duration_ms": 37.45,
  "candidate_letter": "C",
  "probabilities": {
    "A": 0.0382,
    "B": 0.0094,
    "C": 0.8921,
    "D": 0.0603
  },
  "tier": 1,
  "mode": "fast_path",
  "cot_scratchpad": null
}
```

---

## 6. Answers to Technical Measurement Questions

### 1. Linux Entry Point & Sparse Head Patch
* **Functional Equivalence:** The $K=64$ sparse micro-head is an in-memory GEMV output projection optimization (~5 ms latency savings on Windows). It does not alter model weights, representations, prompt ingest, KV caching, or candidate logit values.
* **Linux Fallback:** On Linux using standard CUDA wheels, Gevva0 automatically falls back to native full-vocabulary extraction (`_extract_candidate_logits_from_ptr`). The choice predictions, calibrated confidence values, and accuracy are **100% bit-exact and mathematically identical**.

### 2. Evaluated Mode & Hyperparameters
* **Cyclic Debiasing:** `cyclic_debias = False`. Shipped configuration evaluates single-pass candidate extraction with leading space / no-space marginalization (`logsumexp(logit(" X"), logit("X"))`).
* **Routing Rules:** `enable_rules = True`. All active rules in `RoutingRules/routing_rules.json` solely route complex queries (`FORCE_COT` or `SKIP_TIER0`). **Zero `DIRECT_EXIT` rules are active during benchmarks**, ensuring 100% of decisions are produced by neural inference.
* **CoT Escalation Threshold:** **`0.45`** (calibrated operating point). At $C < 0.45$, exactly ~17.1% of complex traffic escalates to Tier 2 CoT (matching published reports). Setting this to 0.85 would over-escalate and inflate token pricing.
* **CoT Generation Budget:** `max_cot_tokens = 256`, `cot_temp = 0.0`.
* **Hot-KV Cache Reuse:** `reuse_kv = True` (CoT continues directly from the KV cache filled during Tier 1 prefill; zero duplicate context processing).

### 3. Startup Calibration Data
* **Committed File:** The repository includes [`calibration.json`](file:///c:/code/demos/gevva0-linux/calibration.json) with fitted parameters ($T = 4.1451$, `cot_threshold = 0.45`). The server loads this file instantly with zero startup auto-calibration delay.
* **Dataset Used for Fitting:** Fitted exclusively on [`benchmarks/calibration_slice_100.json`](file:///c:/code/demos/gevva0-linux/benchmarks/calibration_slice_100.json), a 125-item slice drawn from **Decision Index** (50 MMLU-Pro, 50 When2Call, 25 BBH boolean expressions).
* **Zero Contamination:** **No JevBench items were used for calibration.** JevBench items are strictly held out and unseen.
