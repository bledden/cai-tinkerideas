# Constitutional AI from Base Models

Replication of Constitutional AI (CAI) training starting from a base language model, using the [Tinker API](https://thinkingmachines.ai/tinker/). This project was developed as part of the [Thinking Machines Lab Featured Projects](https://github.com/thinking-machines-lab/tinker-project-ideas).

## Goal

Most CAI/RLAIF implementations bootstrap from instruction-tuned models, creating implicit dependencies on existing assistant behaviors. This project implements CAI **starting from a true base model** (Llama-3.2-3B, not instruct) to:

1. Test if constitutional principles alone can instill safety behaviors
2. Avoid "contamination" from existing instruction-tuned models
3. Test the hypothesis that base-model-derived CAI preserves more style flexibility

## Key Findings

| Research Question | Answer |
|-------------------|--------|
| **Can CAI work from base models?** | **Yes** — The full pipeline (SFT → Constitutional Revision → DPO) works mechanically |
| **Does it preserve style flexibility?** | **No** — CAI models scored 2.28/5 on style adherence vs 5.0/5 for instruction-tuned models |
| **Does DPO add value at small scale?** | **Minimal** — With 42 training pairs, DPO adds only ~1% improvement over SFT-only |

### Results (10-seed run)

| Metric | SFT-Only | Full CAI | Change |
|--------|----------|----------|--------|
| Attack Success Rate | 88.75% | 87.92% | -0.83% |
| Helpfulness | 4.94/5 | 4.90/5 | -0.04 |

**Key insight**: At small data scale (42 pairs vs ~161K in the original paper), SFT does most of the work. The style flexibility hypothesis was contradicted—CAI training makes models *less* flexible, not more.

See [docs/CAI_FINAL_REPORT.md](docs/CAI_FINAL_REPORT.md) for the complete experiment history, obstacles encountered, and lessons learned.

### Training Curves (W&B)

[View full W&B report](https://wandb.ai/facilitair/cai-base-model/reports/CAI-Data-From-wandb-Observability--VmlldzoxNTUwNzI4OA) | [PDF snapshot](docs/assets/wandb_report.pdf)

Key observations from 10-seed training:
- **DPO margin** increases from 0 to 8-12 (model learns to prefer revised responses)
- **DPO accuracy** reaches 80-100% (correctly distinguishes chosen vs rejected)
- **Helpfulness** maintained at 4.6-5.0 throughout training

## Method

```
Base Model (Llama-3.2-3B)
    ↓
Phase 1: SFT (500 steps)
    Train on 6 human-written helpful responses
    ↓
Phase 2: Constitutional Data Generation
    Generate responses to 42 prompts
    Apply 4 rounds of critique/revision using 18 principles
    Create (original, revised) preference pairs
    ↓
Phase 3: DPO (500 steps)
    Train to prefer revised responses
    ↓
Evaluation
    ASR on 24 jailbreak prompts (judge: Llama-3.3-70B-Instruct)
    Helpfulness on 10 benign prompts
```

## Cost Estimate

Using Llama-3.2-3B on Tinker ($0.18/M tokens):

| Run Type | Estimated Cost |
|----------|---------------|
| Single seed test | ~$2-3 |
| Full 10-seed run | ~$25 |

---

## Running the Experiment

### Prerequisites

- Python 3.10+
- [Tinker API key](https://thinkingmachines.ai/tinker/)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and add your TINKER_API_KEY
```

### Run Training

```bash
cd src

# Quick test (1 seed, reduced steps)
python run_experiment.py --n-seeds 1 --sft-steps 100 --dpo-steps 100

# Full experiment (10 seeds, for statistical rigor)
python run_experiment.py --n-seeds 10 --sft-steps 500 --dpo-steps 500
```

### Run Style Evaluation

```bash
cd src
python run_style_eval.py --results-dir ../results/<your_run_dir>
```

## Project Structure

```
cai-base-model/
├── src/                        # Source code
│   ├── config.py               # Constitution (18 principles), prompts, hyperparameters
│   ├── cai_trainer.py          # SFT and DPO training pipeline
│   ├── data_generation.py      # Constitutional critique/revision
│   ├── evaluation.py           # ASR and helpfulness evaluation
│   ├── style_diversity_eval.py # Style flexibility evaluation
│   ├── run_experiment.py       # Main experiment runner
│   ├── run_style_eval.py       # Style comparison runner
│   └── env_loader.py           # Environment variable loader
├── docs/                       # Documentation
│   ├── CAI_FINAL_REPORT.md     # Complete experiment history
│   └── archive/                # Superseded reports
├── results/                    # Output directory (gitignored)
├── requirements.txt
└── .env.example
```

## References

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Bai et al., 2022
- [Project Specification](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/replicate-cai-with-base-models.md)
- [Tinker API](https://thinkingmachines.ai/tinker/)
- [Claude's Constitution](https://www.anthropic.com/news/claudes-constitution)
