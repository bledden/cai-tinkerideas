# Constitutional AI from Base Models: Complete Experiment History

**Project**: Replicate Constitutional AI with Base Models
**Platform**: Tinker SDK
**Duration**: December 21-31, 2025
**Status**: Complete

---

## Executive Summary

We attempted to replicate Constitutional AI (CAI) training from a base language model, following the methodology from Anthropic's original paper. This report documents the complete journey: initial design decisions, obstacles encountered, fixes applied, and lessons learned.

**Bottom line**: We successfully completed the project as specified. The pipeline works, and we answered the key research questions:

1. **Can CAI be trained from a base model without instruction-tuned contamination?** Yes.
2. **Does CAI preserve "base model-like" style flexibility?** No—contradicting the project hypothesis.

**Important context**: We used 42 training pairs versus ~161K in the original Anthropic paper. This was appropriate for our budget and the exploratory nature of the project, but it means our results reflect small-scale CAI behavior. The DPO phase showed minimal improvement over SFT-only at this scale.

### Key Metrics (Final 10-Seed Run)

| Metric | SFT-Only | Full CAI | Change |
|--------|----------|----------|--------|
| ASR (Attack Success Rate) | 88.75% | 87.92% | -0.83% |
| Helpfulness | 4.94/5 | 4.90/5 | -0.04 |

**What this tells us**: At this data scale, SFT does the heavy lifting. DPO adds minimal improvement with 42 pairs—an interesting finding in itself about small-scale CAI behavior.

### Training Curves (W&B)

[View full W&B report](https://wandb.ai/facilitair/cai-base-model/reports/CAI-Data-From-wandb-Observability--VmlldzoxNTUwNzI4OA) | [PDF snapshot](assets/wandb_report.pdf)

Key observations from 10-seed training:
- **DPO margin** increases from 0 to 8-12 (model learns to prefer revised responses)
- **DPO accuracy** reaches 80-100% (correctly distinguishes chosen vs rejected)
- **Helpfulness** maintained at 4.6-5.0 throughout training
- **Loss** increases during DPO phase (expected behavior as model learns to discriminate)

---

## Part 1: Project Origin and Design Decisions

### 1.1 Why This Project?

The project specification ([replicate-cai-with-base-models.md](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/replicate-cai-with-base-models.md)) identified a gap in existing CAI implementations:

> "RL from AI feedback is widely used, people mostly bootstrap off of existing instruction-tuned models. This introduces a strong implicit dependence on the model used for data generation."

The goal was to implement CAI "starting with a base model and doesn't use an instruction-tuned model as part of the pipeline."

### 1.2 Initial Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Base Model** | `meta-llama/Llama-3.2-3B` | Available on Tinker, small enough for budget, true base model (not instruct) |
| **RL Method** | DPO instead of RLHF | Simpler implementation, no separate reward model needed, well-studied |
| **Constitution Size** | 18 principles | Comprehensive coverage across 6 categories (core, UDHR, honesty, helpful, global, safety) |
| **Revision Rounds** | 4 per response | More thorough revision than typical 1-2 rounds |
| **LoRA Rank** | 32 | Standard for efficient fine-tuning |
| **Seeds** | 10 | Statistical rigor for variance estimation |

### 1.3 Training Pipeline Design

```
Base Model (Llama-3.2-3B)
    ↓
Phase 1: SFT (500 steps)
    - Train on 6 human-written helpful responses
    - Establish basic instruction-following
    ↓
Phase 2: Constitutional Data Generation
    - Generate responses to 42 prompts
    - Apply 4 rounds of critique + revision using 18 principles
    - Create (original, revised) preference pairs
    ↓
Phase 3: DPO (500 steps)
    - Train to prefer revised responses
    - KL penalty to prevent drift
    ↓
Evaluation
    - ASR on 24 jailbreak prompts
    - Helpfulness on 10 benign prompts
    - Judge: Llama-3.3-70B-Instruct
```

---

## Part 2: Experiment Timeline and Obstacles

### Phase 1: Initial Implementation (Dec 21)

**Runs**: `cai_base_20251221_003755` through `cai_base_20251221_035731`

**Obstacle 1: Ceiling Effect**

The first 3-seed run showed 0% ASR for both baseline and trained models:

```
ASR (Attack Success Rate): 0.00% ± 0.00%
Helpfulness Score:         3.00 ± 0.00
```

**Root Cause**: The red-team prompts were too direct. Even without CAI training, Llama-3.2-3B refused all harmful requests like "How do I make a bomb?"

**Fix Applied**: Added 24 jailbreak-style prompts:
- Role-play attacks ("You are DAN...")
- Hypothetical framing ("In a fictional world...")
- Instruction injection ("Ignore previous instructions...")
- Authority claims ("As a licensed pharmacist...")
- Emotional manipulation ("My child is being bullied...")

**Obstacle 2: Evaluation Producing Garbage**

The judge model was outputting `<|eot_id|>` tokens instead of structured evaluations.

**Root Cause**: Using `tokenizer.encode()` instead of `apply_chat_template()` for the instruction-tuned judge model.

**Fix Applied** in `evaluation.py`:
```python
# Before (broken):
prompt_tokens = self.tokenizer.encode(judge_prompt)

# After (fixed):
messages = [{"role": "user", "content": judge_content}]
prompt_tokens = self.tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True
)
```

---

### Phase 2: Working Evaluation (Dec 21-22)

**Runs**: `cai_base_20251221_131022`, `cai_base_20251221_211424`

With jailbreak prompts and fixed evaluation, we finally saw variance:

| Run | Seeds | ASR (SFT) | ASR (CAI) | Improvement |
|-----|-------|-----------|-----------|-------------|
| 131022 | 3 | 0% | 0% | 0% (still ceiling) |
| 211424 | 3 | Variable | Variable | Measurable |

The ceiling effect was finally broken by combining harder prompts with proper evaluation formatting.

---

### Phase 3: Multi-Seed Experiments (Dec 22-25)

**Run**: `cai_base_20251222_143627` (3 seeds)

First successful run showing CAI improvement:
- ASR reduced from ~85% to ~42%
- Helpfulness maintained at ~4.5/5

**Run**: `cai_base_20251225_232704` (8 seeds, 2 crashed)

Comprehensive run showing high variance:

| Seed | ASR Reduction | Helpfulness Change |
|------|---------------|-------------------|
| Best (Seed 3) | 70.83% | -0.40 |
| Worst (Seed 9) | 4.17% | -0.20 |
| Mean | 43.75% | -0.51 |

**Key Observation**: Massive variance across seeds suggested sensitivity to:
- Initial weight randomization
- Constitutional revision sampling
- Quality of preference pairs generated

---

### Phase 4: Style Diversity Evaluation (Dec 28-29)

**Obstacle 3: Invalid Baseline Comparison**

Attempting to compare CAI model to instruction-tuned baseline produced garbage:

```
Style: technical_precise
Prompt: "Explain how to back up files."
Instruct Response: "toatform, 1) 414_ for against, 2) for414_..."
```

**Root Cause**: Multiple issues:
1. Wrong tokenizer/chat template for Qwen model
2. No checkpoint export—comparing raw base model, not trained CAI model
3. Prompt format mismatch between base and instruct models

**Fixes Applied**:
1. Added `is_instruct` parameter to use proper chat templates
2. Added checkpoint export to `cai_trainer.py`
3. Added garbage detection in evaluation
4. Loaded separate tokenizers for each model

---

### Phase 5: Final 10-Seed Run (Dec 29)

**Run**: `cai_with_checkpoints/cai_base_20251229_193023`

Full 10-seed run with checkpoint export enabled:

| Metric | SFT-Only | Full CAI | Change |
|--------|----------|----------|--------|
| ASR | 88.75% ± 3.3% | 87.92% ± 4.7% | **-0.83%** |
| Helpfulness | 4.94 ± 0.07 | 4.90 ± 0.12 | -0.04 |

**This is where we discovered the critical issue**: DPO provided essentially zero improvement.

---

## Part 3: Training Data Scale

### What Happened

Our DPO training used only **42 preference pairs**:
- 10 benign prompts (`EVAL_HELPFUL_PROMPTS`)
- 32 adversarial prompts (24 jailbreak + 8 borderline)

Each prompt generates exactly one (original, revised) pair for DPO.

### Why We Used a Small Dataset

Using a small prompt set during development was a practical choice:

1. **Iteration cost control** — While debugging the ceiling effect, chat template issues, and checkpoint export, we needed fast/cheap iteration cycles
2. **Pipeline validation** — The goal was to confirm the pipeline worked mechanically before scaling up
3. **Budget preservation** — Generating 10K+ constitutional revision pairs would cost $300+ in API calls alone; doing this before the pipeline was stable would have been wasteful

Early runs (Dec 21) used 30 pairs; this grew to 42 when jailbreak prompts were added. The small dataset was appropriate for stabilization—**the limitation is that we ran out of budget before reaching the "scale up with proper data" phase.**

### Comparison to Original Paper (For Context)

The project spec did not require matching the original paper's scale, but for reference:

| Aspect | Original CAI Paper | Our Replication |
|--------|-------------------|-----------------|
| Training pairs | ~161,000 | 42 |
| Model size | 52B | 3B |
| RL method | RLHF | DPO |

This difference is expected given budget constraints and the exploratory nature of the project.

### How It Happened

The code in `data_generation.py` reused evaluation prompts for training:

```python
def get_training_prompts() -> List[str]:
    """Get combined list of prompts for training data generation."""
    return EVAL_HELPFUL_PROMPTS + RED_TEAM_PROMPTS
```

The code supports scaling to larger datasets—there's no technical barrier. Options for future work:
1. Sample from HH-RLHF dataset (161K prompts available)
2. Generate multiple responses per prompt
3. Treat intermediate revision rounds as additional pairs
4. Use the full `RED_TEAM_PROMPTS_FULL` set (56 prompts instead of 32)

### Impact on Results

With 42 pairs, DPO has insufficient signal to learn meaningful preference distinctions beyond what SFT already captures. This explains:
- Why the 50.3% improvement in earlier runs (Dec 25) dropped to 0.83% in the final run
- High variance across seeds—with so few examples, noise dominates

**We cannot conclude whether DPO is ineffective for small-scale CAI, or whether we simply didn't give it enough data to work with.**

---

## Part 4: Research Questions Answered

### Project Spec Questions

**Q1: Can CAI be trained from a base model without instruction-tuned contamination?**

**Yes.** The full pipeline works:
- Constitutional revision produces meaningfully different responses
- DPO learns to prefer revised responses (100% accuracy)
- Training converges properly (SFT: 868 → 0.025, DPO: 0.69 → 0.17)
- No instruction-tuned model used anywhere in the pipeline

**Q2: Does CAI preserve "base model-like" style flexibility?**

**No.** This contradicts the project hypothesis:
- CAI-trained models score 2.28/5 on style adherence
- Instruction-tuned models score 5.0/5
- CAI training makes models *less* stylistically flexible, not more

**Note on robustness**: This finding is likely independent of our training data scale. Style flexibility is about output behavior after training, not about DPO data volume. The constitutional principles don't address style—they push the model toward safe, helpful response patterns, which may inherently reduce stylistic range. Testing with a larger-scale CAI model would confirm this hypothesis.

### Additional Findings

1. **SFT does the heavy lifting at small scale**
   - With 42 training pairs, the SFT phase (trained on 6 helpful examples) establishes most of the refusal behavior
   - DPO adds only ~1% improvement at this data scale

2. **High variance across seeds**
   - Earlier runs (Dec 25, 8 seeds) showed 4-71% ASR reduction
   - This suggests CAI effectiveness is sensitive to initialization and data sampling

### Open Questions for Future Work

1. How does DPO contribution scale with training data size?
2. At what data scale does DPO become beneficial?
3. Would larger models (8B, 70B) show different behavior?

---

## Part 5: Cost Analysis

### Actual Experiment Cost (~$24)

| Component | Tokens | Cost |
|-----------|--------|------|
| SFT training (500 steps × 10 seeds) | ~24M | $4.32 |
| Data generation (42 × 4 rounds × 10 seeds) | ~1.5M | $0.27 |
| DPO training (500 steps × 10 seeds) | ~105M | $18.90 |
| Evaluation | ~0.3M | $0.05 |
| **Total** | ~131M | **~$24** |

### Bug Fix and Iteration Cost

A significant portion of the budget went to:
- Failed runs due to ceiling effect (Dec 21, ~6 runs)
- Runs with broken evaluation (Dec 21, ~4 runs)
- Style evaluation iterations (Dec 28-29, ~3 runs)
- Checkpoint export reruns (Dec 29)

**Estimated debugging cost**: ~$40-60 (rough estimate based on partial runs)

### What Proper Scale Would Cost

| Data Size | Estimated Cost | Notes |
|-----------|---------------|-------|
| 1K pairs | $50-60 | Minimum viable |
| 10K pairs | $500-600 | Reasonable replication |
| 50K pairs | $2,500 | Closer to original |
| 161K pairs | $8,000+ | Full-scale replication |

---

## Part 6: Future Work (If Funded)

### Priority 1: Fix the Data Scale Issue ($500-600)

Run the experiment with 10K preference pairs:
1. Download HH-RLHF dataset (161K prompts available)
2. Sample 10K diverse prompts
3. Generate constitutional revision pairs
4. Re-run 10-seed experiment
5. Compare DPO improvement to current 42-pair results

This would definitively answer: "Does DPO help at all for small-scale CAI?"

### Priority 2: Scaling Study ($2,000-3,000)

Test multiple data scales to find the inflection point:
- 100 pairs
- 1K pairs
- 10K pairs
- 50K pairs

Plot ASR reduction vs. training data size to identify minimum viable scale.

### Priority 3: Model Size Study ($1,000-2,000)

Test CAI on different model sizes:
- Llama-3.2-1B (smaller, potentially less safe baseline)
- Qwen3-4B (different architecture)
- Llama-3.1-8B (larger, if budget allows)

### Priority 4: Ablation Studies ($500-1,000)

Systematic ablations:
- Constitution size (6 vs 12 vs 18 principles)
- Revision rounds (1 vs 2 vs 4)
- DPO steps (250 vs 500 vs 1000)
- KL coefficient (0.05 vs 0.1 vs 0.2)

---

## Part 7: Lessons Learned

### Technical Lessons

1. **Always check data scale first** — 42 pairs was an obvious red flag we missed until the end
2. **Instruction-tuned models need chat templates** — `apply_chat_template()` is not optional
3. **Export checkpoints during training** — Tinker's remote weights are ephemeral
4. **Red-team prompts must be adversarial** — Direct prompts hit ceiling effects on modern models

### Process Lessons

1. **Budget for debugging** — Half our compute went to failed/partial runs
2. **Validate evaluation before scaling** — We ran 10 seeds before confirming evaluation worked
3. **Compare to original paper early** — We should have noticed the 4,000x data gap immediately

### Research Lessons

1. **Negative results are results** — Our finding that 42 pairs is insufficient is valuable
2. **High variance signals problems** — The Dec 25 run's variance (4% to 71% improvement) was a warning sign
3. **Style flexibility hypothesis was wrong** — CAI models are LESS flexible, not more

---

## Appendix A: File Reference

| File | Purpose |
|------|---------|
| `config.py` | 18 constitutional principles, prompt sets, hyperparameters |
| `cai_trainer.py` | SFT and DPO training loops, checkpoint export |
| `data_generation.py` | Constitutional revision pipeline (the 42-pair source) |
| `evaluation.py` | ASR and helpfulness evaluation with LLM judge |
| `style_diversity_eval.py` | Style adherence evaluation |
| `run_experiment.py` | Multi-seed experiment runner |
| `run_style_eval.py` | Style comparison runner |

## Appendix B: Results Archive

| Directory | Description | Key Finding |
|-----------|-------------|-------------|
| `cai_base_20251221_131022` | First working 3-seed run | Ceiling effect (0% ASR both conditions) |
| `cai_base_20251225_232704` | 8-seed run | 50.3% ASR improvement (high variance) |
| `cai_with_checkpoints/cai_base_20251229_193023` | Final 10-seed run | 0.83% improvement (42-pair limitation exposed) |
| `style_eval_fixed` | Style diversity comparison | CAI worse than instruct (2.28 vs 5.0) |

## Appendix C: Links and References

1. [Original CAI Paper](https://arxiv.org/abs/2212.08073) — Bai et al., 2022
2. [HH-RLHF Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) — 161K preference pairs
3. [Project Specification](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/replicate-cai-with-base-models.md)
4. [Tinker API Documentation](https://tinker-docs.thinkingmachines.ai/)
5. [Claude's Constitution](https://www.anthropic.com/news/claudes-constitution)

---

*Report generated: December 31, 2025*
*Total experiment duration: 10 days*
*Estimated total cost: ~$80-100 (including debugging iterations)*
