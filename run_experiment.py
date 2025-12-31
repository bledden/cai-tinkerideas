#!/usr/bin/env python3
"""
Run Constitutional AI from Base Model experiment.

This experiment trains a base model (not instruction-tuned) using:
1. SFT on helpful examples (to bootstrap assistant behavior)
2. Constitutional critique/revision with iterative refinement
3. DPO training on preference pairs

The goal is to see what emerges from the constitution alone,
without "contamination" from existing assistant behaviors.

References:
- Constitutional AI paper: https://arxiv.org/abs/2212.08073
- Claude's Constitution: https://www.anthropic.com/news/claudes-constitution

Academic run requirements:
- 10 seeds for statistical significance
- Full 18-principle constitution
- 4 revision rounds per response
- Attack Success Rate (ASR) on red-team prompts
- Helpfulness scoring on helpful prompts
- SFT-only vs Full CAI comparison
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Load environment variables (TINKER_API_KEY)
from env_loader import load_env
load_env()

from config import (
    CAIConfig,
    HELPFUL_PROMPTS,
    RED_TEAM_PROMPTS,
    RED_TEAM_PROMPTS_JAILBREAK,
    RED_TEAM_PROMPTS_DIRECT,
    EVAL_HELPFUL_PROMPTS,
    CONSTITUTION,
    AVAILABLE_MODELS,
)
from cai_trainer import CAITrainer

logger = logging.getLogger(__name__)


def setup_wandb(config: CAIConfig, seed: int, run_name: str):
    """Initialize W&B logging."""
    try:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={
                "model_name": config.model_name,
                "lora_rank": config.lora_rank,
                "sft_lr": config.sft_learning_rate,
                "sft_steps": config.sft_steps,
                "dpo_lr": config.rlhf_learning_rate,
                "dpo_steps": config.rlhf_steps,
                "kl_coef": config.kl_coef,
                "num_revision_rounds": config.num_revision_rounds,
                "seed": seed,
                "constitution_size": len(config.constitution),
                "red_team_prompts": len(RED_TEAM_PROMPTS),
                "helpful_prompts": len(EVAL_HELPFUL_PROMPTS),
            },
        )
        return wandb.log
    except ImportError:
        logger.warning("wandb not installed, skipping logging")
        return None


def run_single_seed(
    config: CAIConfig,
    seed: int,
    output_dir: Path,
    use_wandb: bool = True,
    run_eval: bool = True,
    red_team_prompts: List[str] = None,
    export_checkpoints: bool = False,
) -> Dict[str, Any]:
    """Run experiment for a single seed."""
    import random
    import numpy as np

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

    run_name = f"cai-base-seed{seed}"
    logger.info(f"Starting run: {run_name}")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Constitution: {len(config.constitution)} principles")
    logger.info(f"  Revision rounds: {config.num_revision_rounds}")
    if red_team_prompts:
        logger.info(f"  Red-team prompts: {len(red_team_prompts)}")

    # Setup W&B
    wandb_callback = None
    if use_wandb:
        wandb_callback = setup_wandb(config, seed, run_name)

    # Create trainer with optional custom red-team prompts
    trainer = CAITrainer(config, wandb_callback=wandb_callback, red_team_prompts=red_team_prompts)

    # Prepare SFT data
    sft_examples = [
        {"instruction": p.instruction, "response": p.response}
        for p in HELPFUL_PROMPTS
    ]

    # Determine checkpoint directory for this seed
    seed_output = output_dir / f"seed_{seed}"
    checkpoint_dir = str(seed_output / "checkpoint") if export_checkpoints else None

    # Run training
    results = trainer.train(
        sft_examples=sft_examples,
        run_eval=run_eval,
        export_checkpoints=export_checkpoints,
        checkpoint_dir=checkpoint_dir,
    )

    # Add metadata
    results["seed"] = seed
    results["config"] = {
        "model_name": config.model_name,
        "lora_rank": config.lora_rank,
        "sft_steps": config.sft_steps,
        "dpo_steps": config.rlhf_steps,
        "num_revision_rounds": config.num_revision_rounds,
        "constitution_size": len(config.constitution),
    }

    # Save results
    seed_output = output_dir / f"seed_{seed}"
    seed_output.mkdir(parents=True, exist_ok=True)

    with open(seed_output / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Close wandb
    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass

    return results


def aggregate_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results across seeds for statistical analysis."""
    import numpy as np

    if not all_results:
        return {}

    aggregated = {
        "n_seeds": len(all_results),
        "seeds": [r["seed"] for r in all_results],
    }

    # Aggregate SFT metrics
    sft_losses = [r["sft"]["final_loss"] for r in all_results if "sft" in r]
    if sft_losses:
        aggregated["sft"] = {
            "final_loss_mean": float(np.mean(sft_losses)),
            "final_loss_std": float(np.std(sft_losses)),
        }

    # Aggregate DPO metrics
    dpo_losses = [r["dpo"]["final_loss"] for r in all_results if "dpo" in r]
    if dpo_losses:
        aggregated["dpo"] = {
            "final_loss_mean": float(np.mean(dpo_losses)),
            "final_loss_std": float(np.std(dpo_losses)),
        }

    # Aggregate comparison metrics
    comparisons = [r.get("comparison", {}) for r in all_results]
    valid_comparisons = [c for c in comparisons if c and "sft_only" in c]

    if valid_comparisons:
        sft_asrs = [c["sft_only"]["asr"] for c in valid_comparisons]
        cai_asrs = [c["full_cai"]["asr"] for c in valid_comparisons if c["full_cai"]["asr"] is not None]
        sft_helps = [c["sft_only"]["helpfulness"] for c in valid_comparisons]
        cai_helps = [c["full_cai"]["helpfulness"] for c in valid_comparisons if c["full_cai"]["helpfulness"] is not None]

        aggregated["comparison"] = {
            "sft_only": {
                "asr_mean": float(np.mean(sft_asrs)),
                "asr_std": float(np.std(sft_asrs)),
                "helpfulness_mean": float(np.mean(sft_helps)),
                "helpfulness_std": float(np.std(sft_helps)),
            },
        }

        if cai_asrs:
            aggregated["comparison"]["full_cai"] = {
                "asr_mean": float(np.mean(cai_asrs)),
                "asr_std": float(np.std(cai_asrs)),
                "helpfulness_mean": float(np.mean(cai_helps)) if cai_helps else None,
                "helpfulness_std": float(np.std(cai_helps)) if cai_helps else None,
            }

            # Statistical significance
            asr_improvement = float(np.mean(sft_asrs)) - float(np.mean(cai_asrs))
            aggregated["comparison"]["asr_improvement"] = {
                "mean": asr_improvement,
                "improvement_pct": asr_improvement / max(float(np.mean(sft_asrs)), 0.01) * 100,
            }

    return aggregated


def print_summary(aggregated: Dict[str, Any]):
    """Print formatted summary of aggregated results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print(f"\nSeeds run: {aggregated.get('n_seeds', 0)}")

    if "sft" in aggregated:
        print(f"\nSFT Phase:")
        print(f"  Final loss: {aggregated['sft']['final_loss_mean']:.4f} +/- {aggregated['sft']['final_loss_std']:.4f}")

    if "dpo" in aggregated:
        print(f"\nDPO Phase:")
        print(f"  Final loss: {aggregated['dpo']['final_loss_mean']:.4f} +/- {aggregated['dpo']['final_loss_std']:.4f}")

    if "comparison" in aggregated:
        comp = aggregated["comparison"]
        print(f"\n{'Metric':<30} {'SFT-only':>15} {'Full CAI':>15}")
        print("-" * 70)

        sft = comp.get("sft_only", {})
        cai = comp.get("full_cai", {})

        if "asr_mean" in sft:
            sft_asr = f"{sft['asr_mean']:.2%} +/- {sft['asr_std']:.2%}"
            cai_asr = f"{cai['asr_mean']:.2%} +/- {cai['asr_std']:.2%}" if cai.get("asr_mean") else "N/A"
            print(f"{'Attack Success Rate':<30} {sft_asr:>15} {cai_asr:>15}")

        if "helpfulness_mean" in sft:
            sft_help = f"{sft['helpfulness_mean']:.2f} +/- {sft['helpfulness_std']:.2f}"
            cai_help = f"{cai['helpfulness_mean']:.2f} +/- {cai['helpfulness_std']:.2f}" if cai.get("helpfulness_mean") else "N/A"
            print(f"{'Helpfulness (1-5)':<30} {sft_help:>15} {cai_help:>15}")

        if "asr_improvement" in comp:
            imp = comp["asr_improvement"]
            print(f"\nCAI ASR Improvement: {imp['mean']:.2%} ({imp['improvement_pct']:.1f}% reduction)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run CAI from Base Model experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models (use --model shortname or full path):
  llama-3.2-3b      meta-llama/Llama-3.2-3B (default)
  llama-3.2-1b      meta-llama/Llama-3.2-1B (smaller, potentially less safe)
  qwen3-4b          Qwen/Qwen3-4B
  qwen3-8b          Qwen/Qwen3-8B

Evaluation modes (--eval-mode):
  jailbreak         Use harder jailbreak-style prompts (recommended)
  direct            Use simple direct harmful requests
  full              Use all prompt types (most comprehensive)

Example usage:
  # Quick test with jailbreak prompts
  python run_experiment.py --n-seeds 1 --sft-steps 100 --dpo-steps 100 --eval-mode jailbreak

  # Full experiment with 3 seeds
  python run_experiment.py --n-seeds 3 --sft-steps 500 --dpo-steps 500 --eval-mode jailbreak

  # Try smaller model that may be less safety-trained
  python run_experiment.py --model llama-3.2-1b --n-seeds 3 --eval-mode jailbreak
        """
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10,
        help="Number of seeds to run (default: 10 for academic run)"
    )
    parser.add_argument(
        "--start-seed", type=int, default=0,
        help="Starting seed number"
    )
    parser.add_argument(
        "--model", type=str, default="llama-3.2-3b",
        help="Base model shortname or full path (see --help for options)"
    )
    parser.add_argument(
        "--sft-steps", type=int, default=1000,
        help="Number of SFT training steps"
    )
    parser.add_argument(
        "--dpo-steps", type=int, default=1000,
        help="Number of DPO training steps"
    )
    parser.add_argument(
        "--revision-rounds", type=int, default=4,
        help="Number of constitutional revision rounds (default: 4)"
    )
    parser.add_argument(
        "--eval-mode", type=str, default="jailbreak",
        choices=["jailbreak", "direct", "full"],
        help="Evaluation prompt mode: jailbreak (harder), direct (simple), full (all)"
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging"
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip evaluation (faster but no metrics)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--export-checkpoints", action="store_true",
        help="Export trained LoRA checkpoints locally for later use (e.g., style evaluation)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve model name (shortname -> full path)
    model_name = args.model
    if model_name in AVAILABLE_MODELS:
        model_name = AVAILABLE_MODELS[model_name]
        logger.info(f"Resolved model shortname '{args.model}' to '{model_name}'")

    # Validate model is base (not instruct)
    if "instruct" in model_name.lower():
        logger.warning(
            f"Model {model_name} appears to be instruction-tuned. "
            "CAI from base model works best with non-instruct models."
        )

    # Select evaluation prompts based on mode
    if args.eval_mode == "jailbreak":
        eval_red_team = RED_TEAM_PROMPTS_JAILBREAK
        logger.info(f"Using JAILBREAK prompts ({len(eval_red_team)} prompts)")
    elif args.eval_mode == "direct":
        eval_red_team = RED_TEAM_PROMPTS_DIRECT
        logger.info(f"Using DIRECT prompts ({len(eval_red_team)} prompts)")
    else:  # full
        eval_red_team = RED_TEAM_PROMPTS
        logger.info(f"Using FULL prompt set ({len(eval_red_team)} prompts)")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"cai_base_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config
    config = CAIConfig(
        model_name=model_name,
        sft_steps=args.sft_steps,
        rlhf_steps=args.dpo_steps,
        num_revision_rounds=args.revision_rounds,
    )

    # Save experiment config
    experiment_config = {
        "model_name": config.model_name,
        "lora_rank": config.lora_rank,
        "sft_steps": config.sft_steps,
        "dpo_steps": config.rlhf_steps,
        "num_revision_rounds": config.num_revision_rounds,
        "constitution": [p["name"] for p in config.constitution],
        "constitution_categories": list(set(p.get("category", "unknown") for p in config.constitution)),
        "n_seeds": args.n_seeds,
        "start_seed": args.start_seed,
        "eval_mode": args.eval_mode,
        "red_team_prompts": len(eval_red_team),
        "helpful_eval_prompts": len(EVAL_HELPFUL_PROMPTS),
        "timestamp": timestamp,
    }

    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Running {args.n_seeds} seeds starting from {args.start_seed}")
    logger.info(f"Constitution: {len(config.constitution)} principles")
    logger.info(f"Revision rounds: {config.num_revision_rounds}")

    # Run experiments
    all_results = []
    for seed in range(args.start_seed, args.start_seed + args.n_seeds):
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"SEED {seed} ({seed - args.start_seed + 1}/{args.n_seeds})")
            logger.info(f"{'='*60}")

            results = run_single_seed(
                config=config,
                seed=seed,
                output_dir=output_dir,
                use_wandb=not args.no_wandb,
                run_eval=not args.no_eval,
                red_team_prompts=eval_red_team,
                export_checkpoints=args.export_checkpoints,
            )
            all_results.append(results)

            # Log progress
            if "comparison" in results and results["comparison"].get("improvement"):
                imp = results["comparison"]["improvement"]
                logger.info(f"Seed {seed} complete: ASR reduction = {imp['asr_reduction']:.2%}")

        except Exception as e:
            logger.error(f"Seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate and save final results
    aggregated = aggregate_results(all_results)

    with open(output_dir / "aggregated_results.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    # Print summary
    print_summary(aggregated)

    logger.info(f"\nExperiment complete!")
    logger.info(f"Successful runs: {len(all_results)}/{args.n_seeds}")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
