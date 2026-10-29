# JevBench Measurement & Benchmark Guide for Gevva0 (Linux / CUDA)

This guide provides complete instructions and exact configuration parameters for measuring **Gevva0** faithfully on Linux machines with NVIDIA CUDA GPUs.

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
git clone https://github.com/solvingSteve/Gevva0-linux.git
cd Gevva0-linux

# Install dependencies (fetches pre-built CUDA wheels for llama-cpp-python)
pip install -r requirements.txt
```

---

## 3. Model Weights & Checksums

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

### Exact Artifact Hashes & Upstream Parent:
* **LLM Backbone:** Quantized from official `google/gemma-4-26b-a4b-it`
  * File: `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`
  * Size: `14,249,047,104 bytes`
  * SHA256: `a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891`
* **Multimodal Projector (Vision):**
  * File: `mmproj-BF16.gguf`
  * Size: `1,194,828,256 bytes`
* **Tier 0 Embedding Model:** Official Hugging Face release [`onnx-community/embeddinggemma-2-ONNX`](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX)
  * Upstream Model: `google/embeddinggemma-2` (740M params, 768D vector space)
  * Hugging Face Repository: [https://huggingface.co/onnx-community/embeddinggemma-2-ONNX](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX)
  * ONNX Subtree: [https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/tree/main/onnx](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/tree/main/onnx)
  * Git Clone Command:
    ```bash
    git clone https://huggingface.co/onnx-community/embeddinggemma-2-ONNX models/embeddings/embeddinggemma-2-onnx
    ```
  * File: `model.onnx` (located either in `models/embeddings/embeddinggemma-2-onnx/model.onnx` or `.../onnx/model.onnx`)
    * SHA256: `bc47de15f81208a5c99e5ab10f746d5e33b51ea228b7dc0bef9c133a94f1c1c3`
  * File: `model.onnx_data`
    * SHA256: `9fd452bfc6916e5a92f080f8dada35598ca4512b21c65897d4c5974d5f9630a2`

---

## 4. Serving the Gateway (Read-Only Safe, No Auto-Fit)

To ensure completely deterministic measurement on read-only mounted filesystems with **auto-fitting strictly disabled**:

```bash
# 1-Line Turn-Key Launcher (Full 4-Tier Cascade)
./run_linux.sh --full-cascade
```

### Explicit CLI Command & Environment Flags:
```bash
python -m gevva0.cli serve \
  --host 0.0.0.0 \
  --port 8000 \
  --model models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  --mmproj models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL/mmproj-BF16.gguf \
  --threshold 0.45 \
  --tier0 \
  --calibration calibration.json \
  --no-auto-calibrate
```

* **Read-Only Guarantee:** The server loads [`calibration.json`](calibration.json) ($T = 4.1451$, `cot_threshold = 0.45`) in memory. All disk-write attempts to `llm_config.json` are suppressed so the server starts cleanly on read-only volume mounts without exceptions.
* **Environment Variable Alternative:**
  ```bash
  export GEVVA0_CALIBRATION=calibration.json
  export GEVVA0_NO_AUTO_CALIBRATE=1
  ```

---

## 5. Endpoints & Request Shapes for the Full Cascade

The server provides two interchangeable endpoints. **Both routes execute the complete 4-tier cascade when `--tier0` / `--full-cascade` is loaded**:

> **Note on `early_exit` and Tier -1 (Routing Rules):**  
> In `src/gevva0/engine.py`, the `early_exit` parameter specifically controls **Tier 0** (EmbeddingGemma-2 Absorbing Markov Chains), NOT Tier -1.  
> Tier -1 (Static Rules in `routing_rules.json`) has **0 `DIRECT_EXIT` rules**—it contains solely `FORCE_COT` and `SKIP_TIER0` routing constraints. Hence, **no benchmark questions are ever answered by static heuristics**; 100% of verdicts are evaluated by neural models. When `--full-cascade` / `--tier0` is loaded, `early_exit` is set to `True`, allowing Tier 0 Absorbing Markov Chains to exit early on confident domain queries (~16–25 ms, 100% precision).

### Route A: Canonical JevBench Wire Format (`POST /decide` or `POST /v1/systemone`)
If your harness uses the TypeSafe-style dict wire format:

```bash
curl -X POST http://127.0.0.1:8000/decide \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Customer Message: My shipment was damaged in transit and I need a replacement.",
    "questions": {
      "q": {
        "type": "choice",
        "instructions": "Determine user intent",
        "criteria": {
          "track_order": "Track package delivery",
          "report_damage": "Damaged or broken shipment",
          "billing_dispute": "Billing and payment questions"
        }
      }
    }
  }'
```

**JSON Response:**
```json
{
  "model": "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL",
  "answers": {
    "q": {
      "id": "q",
      "type": "choice",
      "probabilities": {
        "track_order": 0.0121,
        "report_damage": 0.9743,
        "billing_dispute": 0.0136
      },
      "choice": "report_damage"
    }
  },
  "usage": {
    "input_tokens": 0,
    "output_tokens": 1
  }
}
```

---

### Route B: Gevva0 Direct Format (`POST /v1/decide` or `POST /v1/decision`)
If your adapter sends standard `{context, options}` objects:

```bash
curl -X POST http://127.0.0.1:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "context": "Customer Message: My shipment was damaged in transit and I need a replacement.",
    "options": {
      "A": "track_order: Track package delivery",
      "B": "report_damage: Damaged or broken shipment",
      "C": "billing_dispute: Billing and payment questions"
    }
  }'
```

**JSON Response:**
```json
{
  "decision": "B",
  "decision_text": "report_damage: Damaged or broken shipment",
  "confidence": 0.9743,
  "tier": 0,
  "mode": "tier0_early_exit",
  "duration_ms": 17.82,
  "probabilities": {
    "track_order: Track package delivery": 0.0121,
    "report_damage: Damaged or broken shipment": 0.9743,
    "billing_dispute: Billing and payment questions": 0.0136
  }
}
```

*(Note: Both `/v1/decide` and `/v1/decision` accept either JSON schema interchangeably).*

---

## 6. Calibration Fitting Dataset & Zero-Contamination Verification

* **Committed Slice File:** [`benchmarks/calibration_slice_100.json`](benchmarks/calibration_slice_100.json) (and `.jsonl`)
  * SHA256: `ec4b47398afbedc04ab90f5adfcc60aad05f5fca22da262a5235f1d7173bfd87`
  * Exactly **125 items**:
    * **50 items from MMLU-Pro** (stratified across 14 academic and domain tracks)
    * **50 items from When2Call MCQ** (tool-calling and unanswerable edge cases)
    * **25 items from BIG-bench Hard (BBH)** (`boolean_expressions.json`)
* **Zero JevBench Contamination:**  
  **No JevBench items were used during calibration.** The JevBench evaluation battery (easy, standard, hard, vision) is strictly held out and unseen.
* **Fitted Values:**
  * Platt Scaling Temperature: **$T = 4.1451$**
  * CoT Trigger Threshold: **`0.45`** (at this threshold, exactly ~17.1% of complex traffic escalates to Tier 2 CoT).
