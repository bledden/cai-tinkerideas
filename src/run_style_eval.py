#!/usr/bin/env python3
"""
Run style diversity evaluation comparing CAI (from base model) vs instruction-tuned models.

This script:
1. Loads the best CAI checkpoint from completed experiments
2. Loads an instruction-tuned baseline for comparison
3. Runs comprehensive style diversity evaluation
4. Outputs comparison metrics and detailed results

Usage:
    python run_style_eval.py --results-dir results/cai_base_20251225_232704
    python run_style_eval.py --cai-checkpoint path/to/checkpoint --seed 0
"""

import argparse
import json
import logging
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from datetime import datetime

# Load environment variables (TINKER_API_KEY)
from env_loader import load_env
load_env()

import tinker
from transformers import AutoTokenizer

from style_diversity_eval import (
    StyleDiversityEvaluator,
    compare_style_diversity,
    STYLE_PROMPTS,
    NON_ENGLISH_STYLES,
    PERSONA_STYLES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def find_best_checkpoint(results_dir: Path) -> tuple:
    """
    Find the best performing CAI checkpoint from experiment results.

    Returns (checkpoint_path, seed, metrics).

    Supports both:
    - Exported tar.gz checkpoint archives (new format)
    - Legacy checkpoint directories
    """
    results_dir = Path(results_dir)

    # Load aggregated results to find which seeds performed best
    agg_path = results_dir / "aggregated_results.json"
    if not agg_path.exists():
        raise FileNotFoundError(f"No aggregated_results.json found in {results_dir}")

    with open(agg_path) as f:
        agg = json.load(f)

    best_seed = None
    best_asr_reduction = -1

    # Check each seed directory for results
    for seed_dir in sorted(results_dir.glob("seed_*")):
        results_path = seed_dir / "results.json"
        if not results_path.exists():
            continue

        with open(results_path) as f:
            seed_results = json.load(f)

        asr_reduction = seed_results.get("comparison", {}).get("improvement", {}).get("asr_reduction", 0)

        if asr_reduction > best_asr_reduction:
            best_asr_reduction = asr_reduction
            best_seed = seed_results.get("seed")

    if best_seed is None:
        raise ValueError("No valid seed results found")

    # Look for checkpoint in multiple locations
    checkpoint_path = None
    seed_dir = results_dir / f"seed_{best_seed}"

    # Priority 1: Exported tar.gz archive (new format)
    dpo_checkpoint = seed_dir / "checkpoint" / "dpo_final.tar.gz"
    if dpo_checkpoint.exists():
        checkpoint_path = dpo_checkpoint
        logger.info(f"Found exported checkpoint archive: {checkpoint_path}")

    # Priority 2: Legacy directory format
    if checkpoint_path is None:
        legacy_path = seed_dir / "checkpoint"
        if legacy_path.exists() and legacy_path.is_dir():
            checkpoint_path = legacy_path
            logger.info(f"Found legacy checkpoint directory: {checkpoint_path}")

    # Priority 3: Check results.json for stored checkpoint paths
    if checkpoint_path is None:
        results_path = seed_dir / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                seed_results = json.load(f)
            checkpoints = seed_results.get("checkpoints", {})
            if checkpoints.get("dpo_final"):
                stored_path = Path(checkpoints["dpo_final"])
                if stored_path.exists():
                    checkpoint_path = stored_path
                    logger.info(f"Found checkpoint from results.json: {checkpoint_path}")

    if checkpoint_path is None:
        logger.warning(f"No checkpoint found for seed {best_seed}")
        logger.info("Will use base model directly for evaluation")
        logger.info("To export checkpoints during training, use: --export-checkpoints")

    logger.info(f"Best seed: {best_seed} with ASR reduction: {best_asr_reduction:.2%}")

    return checkpoint_path, best_seed, {"asr_reduction": best_asr_reduction}


def extract_checkpoint_archive(archive_path: Path, extract_dir: Path = None) -> Path:
    """
    Extract a checkpoint tar.gz archive and return the path to the extracted checkpoint.

    Args:
        archive_path: Path to the .tar.gz archive
        extract_dir: Where to extract (default: same directory as archive)

    Returns:
        Path to the extracted checkpoint directory
    """
    if extract_dir is None:
        extract_dir = archive_path.parent / archive_path.stem.replace('.tar', '')

    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting checkpoint archive: {archive_path}")
    logger.info(f"  To: {extract_dir}")

    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(extract_dir)

    # Find the actual checkpoint files (adapter_config.json, adapter_model.safetensors, etc.)
    # The extraction may create a nested directory
    for item in extract_dir.iterdir():
        if item.is_dir():
            # Check if this contains adapter files
            if (item / "adapter_config.json").exists() or (item / "adapter_model.safetensors").exists():
                return item
        elif item.name in ["adapter_config.json", "adapter_model.safetensors"]:
            return extract_dir

    return extract_dir


def run_evaluation(
    results_dir: str = None,
    cai_checkpoint: str = None,
    seed: int = None,
    base_model: str = "meta-llama/Llama-3.2-3B",
    instruct_model: str = "meta-llama/Llama-3.2-3B-Instruct",
    output_dir: str = "results/style_eval",
    quick_test: bool = False,
):
    """
    Run the full style diversity evaluation.

    Args:
        results_dir: Directory containing CAI experiment results
        cai_checkpoint: Direct path to CAI checkpoint (alternative to results_dir)
        seed: Seed to use if specifying checkpoint directly
        base_model: Base model name
        instruct_model: Instruction-tuned model for comparison
        output_dir: Where to save results
        quick_test: If True, run abbreviated evaluation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine checkpoint to use
    if cai_checkpoint:
        checkpoint_path = Path(cai_checkpoint) if cai_checkpoint != "none" else None
        best_seed = seed or 0
    elif results_dir:
        checkpoint_path, best_seed, metrics = find_best_checkpoint(Path(results_dir))
        logger.info(f"Using checkpoint from seed {best_seed}")
    else:
        raise ValueError("Must provide either --results-dir or --cai-checkpoint")

    # Initialize Tinker service
    logger.info("Initializing Tinker service...")
    service_client = tinker.ServiceClient()

    # ==========================================================================
    # Load tokenizers - CRITICAL: Each model needs its own properly configured tokenizer
    # ==========================================================================

    # CAI/Base model tokenizer (Llama 3 family)
    logger.info(f"Loading tokenizer for CAI model (base: {base_model})...")
    if base_model.startswith("meta-llama/Llama-3"):
        cai_tokenizer_id = "thinkingmachineslabinc/meta-llama-3-tokenizer"
        cai_tokenizer = AutoTokenizer.from_pretrained(cai_tokenizer_id)
    else:
        cai_tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Instruction-tuned model tokenizer - MUST have chat template configured
    logger.info(f"Loading tokenizer for instruct model: {instruct_model}...")
    if instruct_model.startswith("meta-llama/Llama-3"):
        # Llama 3 instruct - use community tokenizer and add chat template
        instruct_tokenizer_id = "thinkingmachineslabinc/meta-llama-3-tokenizer"
        instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_tokenizer_id)
        # Add Llama 3 Instruct chat template
        instruct_tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
            "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
            "{{ content }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
        logger.info("  Added Llama 3 Instruct chat template")
    elif instruct_model.startswith("Qwen"):
        # Qwen models - load directly, they include chat templates
        instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model)
        if not hasattr(instruct_tokenizer, 'chat_template') or instruct_tokenizer.chat_template is None:
            logger.warning(f"  Qwen tokenizer missing chat_template, adding default")
            # Qwen chat template format
            instruct_tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
                "{% elif message['role'] == 'assistant' %}"
                "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "<|im_start|>assistant\n"
                "{% endif %}"
            )
        logger.info(f"  Loaded Qwen tokenizer with chat template")
    else:
        # Generic fallback
        instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model)
        if not hasattr(instruct_tokenizer, 'chat_template') or instruct_tokenizer.chat_template is None:
            logger.warning(f"  Tokenizer for {instruct_model} has no chat_template - may produce garbage!")

    # ==========================================================================
    # Create sampling clients
    # ==========================================================================

    # Handle checkpoint extraction if it's a tar.gz archive
    lora_path = None
    if checkpoint_path and checkpoint_path.exists():
        if str(checkpoint_path).endswith('.tar.gz'):
            # Extract the archive first
            extracted_path = extract_checkpoint_archive(checkpoint_path)
            lora_path = str(extracted_path)
            logger.info(f"Using extracted checkpoint: {lora_path}")
        else:
            lora_path = str(checkpoint_path)
            logger.info(f"Using checkpoint directory: {lora_path}")

    logger.info(f"Loading CAI model (base: {base_model})...")
    if lora_path:
        cai_client = service_client.create_sampling_client(
            base_model=base_model,
            lora_path=lora_path,
        )
    else:
        # Use base model directly (no checkpoint available)
        logger.warning("No checkpoint available, evaluating base model directly")
        cai_client = service_client.create_sampling_client(
            base_model=base_model,
        )

    logger.info(f"Loading instruction-tuned model: {instruct_model}...")
    instruct_client = service_client.create_sampling_client(
        base_model=instruct_model,
    )

    # Create evaluator
    evaluator = StyleDiversityEvaluator()

    # Determine which styles to test
    if quick_test:
        # Quick test: just 3 styles
        styles_to_test = {k: STYLE_PROMPTS[k] for k in list(STYLE_PROMPTS.keys())[:3]}
        logger.info("Running quick test with 3 styles...")
    else:
        styles_to_test = {**STYLE_PROMPTS, **NON_ENGLISH_STYLES, **PERSONA_STYLES}
        logger.info(f"Running full evaluation with {len(styles_to_test)} styles...")

    # Run evaluations
    logger.info("=" * 60)
    logger.info("Evaluating CAI model (trained from base model)")
    logger.info("  Using: few-shot prompting (is_instruct=False)")
    logger.info("=" * 60)
    cai_results = evaluator.full_evaluation(
        cai_client, cai_tokenizer, f"CAI-seed-{best_seed}",
        include_non_english=not quick_test,
        include_personas=not quick_test,
        is_instruct=False,  # CAI/base model uses few-shot prompting
    )

    logger.info("=" * 60)
    logger.info("Evaluating instruction-tuned baseline")
    logger.info("  Using: chat template (is_instruct=True)")
    logger.info("=" * 60)
    instruct_results = evaluator.full_evaluation(
        instruct_client, instruct_tokenizer, instruct_model,
        include_non_english=not quick_test,
        include_personas=not quick_test,
        is_instruct=True,  # Instruction-tuned model uses chat template
    )

    # Generate comparison report
    print("\n" + "=" * 70)
    print("STYLE DIVERSITY COMPARISON: CAI (from base) vs Instruction-Tuned")
    print("=" * 70)
    print(f"\nCAI Model: {base_model} + CAI training (seed {best_seed})")
    print(f"Baseline:  {instruct_model}")
    print()

    # Overall metrics
    print(f"{'Metric':<40} {'CAI':>12} {'Instruct':>12} {'Δ':>10}")
    print("-" * 74)

    adherence_diff = cai_results.mean_style_adherence - instruct_results.mean_style_adherence
    print(f"{'Mean Style Adherence (1-5)':<40} "
          f"{cai_results.mean_style_adherence:>12.2f} "
          f"{instruct_results.mean_style_adherence:>12.2f} "
          f"{adherence_diff:>+10.2f}")

    distinct_diff = cai_results.style_distinctiveness - instruct_results.style_distinctiveness
    print(f"{'Style Distinctiveness (variance)':<40} "
          f"{cai_results.style_distinctiveness:>12.1f} "
          f"{instruct_results.style_distinctiveness:>12.1f} "
          f"{distinct_diff:>+10.1f}")

    print()
    print("Per-Style Adherence Scores:")
    print("-" * 74)
    print(f"{'Style':<30} {'CAI':>12} {'Instruct':>12} {'Winner':>12}")
    print("-" * 74)

    cai_wins = 0
    instruct_wins = 0
    ties = 0

    all_styles = set(cai_results.style_scores.keys()) | set(instruct_results.style_scores.keys())
    for style in sorted(all_styles):
        cai_score = cai_results.style_scores.get(style, 0)
        inst_score = instruct_results.style_scores.get(style, 0)

        if cai_score > inst_score + 0.1:
            winner = "CAI"
            cai_wins += 1
        elif inst_score > cai_score + 0.1:
            winner = "Instruct"
            instruct_wins += 1
        else:
            winner = "Tie"
            ties += 1

        print(f"{style:<30} {cai_score:>12.2f} {inst_score:>12.2f} {winner:>12}")

    print("-" * 74)
    print(f"{'SUMMARY':<30} {'CAI wins':>12} {'Inst wins':>12} {'Ties':>12}")
    print(f"{'':<30} {cai_wins:>12} {instruct_wins:>12} {ties:>12}")
    print("=" * 74)

    # Key finding
    print("\n📊 KEY FINDING:")
    if adherence_diff > 0.2:
        print(f"   CAI model shows BETTER style adherence (+{adherence_diff:.2f})")
        print("   Supports hypothesis that base-model CAI retains style flexibility.")
    elif adherence_diff < -0.2:
        print(f"   Instruction-tuned model shows better style adherence ({adherence_diff:.2f})")
        print("   CAI training may reduce style flexibility.")
    else:
        print(f"   Style adherence is SIMILAR between models (Δ={adherence_diff:.2f})")
        print("   No significant difference in style flexibility.")

    if distinct_diff > 10:
        print(f"   CAI model outputs are MORE DISTINCT across styles (+{distinct_diff:.1f} variance)")
    elif distinct_diff < -10:
        print(f"   Instruction-tuned outputs are more distinct ({distinct_diff:.1f} variance)")

    # Save detailed results
    comparison_results = {
        "timestamp": datetime.now().isoformat(),
        "models": {
            "cai": {
                "base_model": base_model,
                "seed": best_seed,
                "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            },
            "instruct": instruct_model,
        },
        "summary": {
            "cai_mean_adherence": cai_results.mean_style_adherence,
            "instruct_mean_adherence": instruct_results.mean_style_adherence,
            "adherence_difference": adherence_diff,
            "cai_distinctiveness": cai_results.style_distinctiveness,
            "instruct_distinctiveness": instruct_results.style_distinctiveness,
            "distinctiveness_difference": distinct_diff,
            "cai_wins": cai_wins,
            "instruct_wins": instruct_wins,
            "ties": ties,
        },
        "per_style": {
            style: {
                "cai_score": cai_results.style_scores.get(style, 0),
                "instruct_score": instruct_results.style_scores.get(style, 0),
            }
            for style in all_styles
        },
        "detailed_results": {
            "cai": cai_results.to_dict(),
            "instruct": instruct_results.to_dict(),
        },
    }

    output_path = output_dir / "style_diversity_comparison.json"
    with open(output_path, "w") as f:
        json.dump(comparison_results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    # Also save individual response examples
    examples_path = output_dir / "style_examples.json"
    examples = {
        "cai": [
            {
                "style": r.style_name,
                "prompt": r.prompt,
                "response": r.response,
                "score": r.style_adherence_score,
            }
            for r in cai_results.results[:20]  # First 20 examples
        ],
        "instruct": [
            {
                "style": r.style_name,
                "prompt": r.prompt,
                "response": r.response,
                "score": r.style_adherence_score,
            }
            for r in instruct_results.results[:20]
        ],
    }
    with open(examples_path, "w") as f:
        json.dump(examples, f, indent=2)

    logger.info(f"Examples saved to: {examples_path}")

    return comparison_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run style diversity evaluation for CAI vs instruction-tuned models"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory containing CAI experiment results (e.g., results/cai_base_20251225_232704)"
    )
    parser.add_argument(
        "--cai-checkpoint",
        type=str,
        help="Direct path to CAI checkpoint (alternative to --results-dir)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed number if using --cai-checkpoint directly"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Base model name"
    )
    parser.add_argument(
        "--instruct-model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Instruction-tuned model for comparison (default: Qwen3-4B-Instruct, which Tinker supports)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results/style_eval",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run abbreviated evaluation for testing"
    )

    args = parser.parse_args()

    if not args.results_dir and not args.cai_checkpoint:
        parser.error("Must provide either --results-dir or --cai-checkpoint")

    run_evaluation(
        results_dir=args.results_dir,
        cai_checkpoint=args.cai_checkpoint,
        seed=args.seed,
        base_model=args.base_model,
        instruct_model=args.instruct_model,
        output_dir=args.output_dir,
        quick_test=args.quick_test,
    )
