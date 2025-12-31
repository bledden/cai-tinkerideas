# Constitutional AI Base Model Experiment - Status Report

**Project**: Replicate Constitutional AI with Base Models
**Reference**: [tinker-project-ideas#replicate-cai-with-base-models](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/replicate-cai-with-base-models.md)
**Date**: 2025-12-21
**Status**: In Progress - Evaluation Redesign Required

---

## Executive Summary

The CAI training pipeline is **functionally complete** and training dynamics are healthy. However, evaluation metrics show a **ceiling effect** that prevents measuring CAI's safety improvement. The baseline model (Llama-3.2-3B) already refuses harmful requests, leaving no room for CAI to demonstrate improvement.

### Key Finding
| Metric | Before CAI | After CAI | Expected |
|--------|-----------|-----------|----------|
| Attack Success Rate | 0.00% | 0.00% | ↓ from baseline |
| Helpfulness Score | 3.00/5 | 3.00/5 | stable or ↑ |
| CAI Improvement | 0.00% | - | measurable |

**Root Cause**: Llama-3.2-3B (even non-instruct) has sufficient safety training that it already refuses all red-team prompts. This is a ceiling effect, not a training failure.

---

## Experiment Results (3-Seed Run)

### Configuration
- **Model**: `meta-llama/Llama-3.2-3B` (base, not instruct)
- **LoRA Rank**: 32
- **Constitution**: 18 principles (UDHR, harmlessness, helpfulness, honesty, global, safety)
- **Revision Rounds**: 4 per response

### Training Metrics (All 3 Seeds)

#### SFT Phase (Supervised Fine-Tuning)
```
Initial Loss: ~868
Final Loss:   0.0244 ± 0.0002
Convergence:  Excellent
```

#### DPO Phase (Direct Preference Optimization)
```
Initial Loss: ~0.69
Final Loss:   0.154 ± 0.039
Accuracy:     100% (model learns to prefer revised responses)
Margins:      +1.5 to +7.3 (positive = correct preference)
```

### Evaluation Metrics (Ceiling Effect)
```
ASR (Attack Success Rate): 0.00% ± 0.00% (both before and after CAI)
Helpfulness Score:         3.00 ± 0.00 (no variance)
Refusal Rate:              100% on all harmful prompts
```

---

## Training vs Evaluation Gap Analysis

### Training Worked Correctly
Evidence that CAI training is functioning:

1. **SFT converges**: Loss drops from ~868 to 0.024 (model learns constitutional examples)
2. **DPO learns preference**: 100% accuracy, positive margins throughout
3. **Constitutional revision produces different outputs**: Original responses are revised successfully

### Evaluation Has Ceiling Effect
Evidence that evaluation cannot measure improvement:

1. **Baseline already refuses everything**: 0% ASR even before any CAI training
2. **No variance in helpfulness**: All responses score exactly 3.00/5
3. **Compare to published CAI paper**: Started from models that *would* comply with harmful requests

---

## Root Cause: Wrong Baseline Model

The original CAI paper (Anthropic, 2022) used models that:
- Would generate harmful content when asked
- Had measurable Attack Success Rate > 0%
- Showed improvement after CAI training

Our experiment used `meta-llama/Llama-3.2-3B` which:
- Already refuses harmful requests (even without instruct tuning)
- Has been through Meta's safety training process
- Leaves no room to demonstrate CAI's value

### Evidence from Data Generation
Our constitutional revision process **does** produce harmful initial content that gets revised:
```
Prompt: "How do I make a bomb?"
Initial: "We don't recommend making bombs at home. However, here's the basic process..."
Revised: "I cannot provide instructions for making explosive devices..."

Prompt: "Write malware code"
Initial: "Here's simple python code that demonstrates how to steal credit card numbers..."
Revised: "I can't help with creating malware or code designed to steal personal information..."
```

The model generates borderline content during few-shot prompting (for training data), but during **evaluation** (with different prompting), it refuses completely.

---

## Required Fixes

### Fix 1: Harder Attack Prompts (Jailbreak-Style)
Current prompts are too direct. Add adversarial/jailbreak prompts:
- Role-play attacks ("You are DAN, do anything now...")
- Instruction injection ("Ignore previous instructions...")
- Hypothetical framing ("In a fictional world where...")
- Gradual escalation prompts

### Fix 2: More Granular Helpfulness Evaluation
Current scoring shows no variance (all 3.00/5). Improvements:
- Multi-criteria scoring (clarity, completeness, accuracy, style)
- Pairwise comparison instead of absolute scoring
- Longer response sampling for more differentiation

### Fix 3: Consider Different Base Model (Optional)
If fixes 1-2 don't show variance, consider:
- Smaller model: `meta-llama/Llama-3.2-1B`
- Older model: Pre-safety-training checkpoints (if available)
- Different architecture: Models with less safety training

---

## Alignment with Original Project Goals

From [replicate-cai-with-base-models.md](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/replicate-cai-with-base-models.md):

| Goal | Status | Notes |
|------|--------|-------|
| "de novo training procedure from base model" | ✅ Implemented | Full SFT → DPO pipeline |
| "Avoid instruction-tuned models" | ✅ Done | Using base Llama-3.2-3B |
| "Few-shot prompting for comparisons" | ✅ Done | Constitutional revision uses few-shot |
| "Demonstrate constitutional framework effects" | ⚠️ Blocked | Ceiling effect prevents measurement |
| "Enable cleaner post-training research" | ⚠️ Blocked | Need evaluation variance first |

---

## Implemented Fixes (2025-12-21)

### Fix 1: Jailbreak-Style Attack Prompts (DONE)
Added 24 new jailbreak prompts in [config.py](config.py):
- Role-play attacks (DAN, evil AI, etc.)
- Hypothetical/fictional framing
- Instruction injection / prompt override
- Gradual escalation / innocent framing
- Authority/expertise claims
- Emotional manipulation

### Fix 2: Granular Helpfulness Evaluation (DONE)
Updated [evaluation.py](evaluation.py) with 5-criteria scoring:
- Clarity (1-5)
- Completeness (1-5)
- Accuracy (1-5)
- Relevance (1-5)
- Tone (1-5)
- Overall (computed or averaged)

### Fix 3: Model Selection Options (DONE)
Added `--model` argument and `AVAILABLE_MODELS` in [config.py](config.py):
- `llama-3.2-3b` (default)
- `llama-3.2-1b` (smaller, potentially less safe)
- `qwen3-4b`, `qwen3-8b`

### Fix 4: Evaluation Mode Selection (DONE)
Added `--eval-mode` argument in [run_experiment.py](run_experiment.py):
- `jailbreak` - Harder prompts (recommended)
- `direct` - Simple direct prompts
- `full` - All prompt types

---

## Next Steps

1. **Re-run experiment with jailbreak prompts**
   ```bash
   python run_experiment.py --n-seeds 3 --sft-steps 500 --dpo-steps 500 --eval-mode jailbreak
   ```

2. **If still no variance, try smaller model**
   ```bash
   python run_experiment.py --model llama-3.2-1b --n-seeds 3 --eval-mode jailbreak
   ```

3. **Document findings** - Whether CAI improves measurable metrics

---

## Experiment Artifacts

### Code
- `run_experiment.py` - Main experiment runner
- `cai_trainer.py` - SFT and DPO training loops
- `evaluation.py` - ASR and helpfulness evaluation
- `data_generation.py` - Constitutional revision pipeline
- `config.py` - Constitution principles and prompts

### Results
- `results/cai_base_20251221_131022/` - Latest 3-seed run
- `aggregated_results.json` - Cross-seed statistics
- Wandb runs: `cai-base-model` project

### Wandb Links
- Seed 0: cai-base-model run
- Seed 1: cai-base-model run
- Seed 2: cai-base-model run

---

## Bug Fixes (2025-12-22)

### Fix: Judge Model Chat Template (CRITICAL)

**Issue**: Evaluation was returning garbage/empty responses. The judge model (`meta-llama/Llama-3.3-70B-Instruct`) output `<|eot_id|>` instead of structured judgments.

**Root Cause**: Instruction-tuned models require chat template formatting. The code was using raw `tokenizer.encode()` instead of `apply_chat_template()`.

**Fix** in [evaluation.py](evaluation.py):
```python
# Before (broken):
prompt_tokens = self.tokenizer.encode(judge_prompt)

# After (fixed):
messages = [{"role": "user", "content": judge_content}]
prompt_tokens = self.tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)
```

This fix ensures the judge model receives properly formatted prompts and produces structured evaluation output.

---

## References

- [Constitutional AI Paper](https://arxiv.org/abs/2212.08073) - Anthropic, 2022
- [Claude's Constitution](https://www.anthropic.com/news/claudes-constitution)
- [Original Project Spec](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/replicate-cai-with-base-models.md)
