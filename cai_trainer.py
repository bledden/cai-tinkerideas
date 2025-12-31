"""
Constitutional AI Trainer using Tinker.

Implements the full CAI pipeline from base model:
1. Helpful-only SFT (also serves as baseline)
2. Constitutional critique/revision data generation (iterative)
3. DPO training on preference pairs

Key insight: Starting from base model instead of instruction-tuned
removes "contamination" from existing assistant behaviors.

References:
- Constitutional AI paper: https://arxiv.org/abs/2212.08073
- Claude's Constitution: https://www.anthropic.com/news/claudes-constitution
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import asyncio

import tinker
from tinker import types
import torch

from config import CAIConfig, HELPFUL_PROMPTS, RED_TEAM_PROMPTS, EVAL_HELPFUL_PROMPTS
from data_generation import (
    ConstitutionalDataGenerator,
    ConstitutionalPair,
    format_few_shot_prompt,
    get_training_prompts,
)
from evaluation import CAIEvaluator, EvaluationSummary

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Tracks training progress across phases."""
    phase: str = "init"
    global_step: int = 0
    sft_steps_completed: int = 0
    dpo_steps_completed: int = 0


@dataclass
class TrainingResult:
    """Results from a training run."""
    phase: str
    final_loss: float
    steps_completed: int
    eval_summary: Optional[EvaluationSummary] = None


class CAITrainer:
    """
    Constitutional AI trainer starting from base model.

    Training Pipeline:
    1. SFT Phase: Train on helpful examples to get assistant behavior
       - Can save weights here for SFT-only baseline comparison
    2. Data Generation: Generate constitutional pairs (iterative revision)
    3. DPO Phase: Train to prefer revised over original responses
    """

    def __init__(
        self,
        config: CAIConfig,
        wandb_callback: Optional[callable] = None,
        red_team_prompts: Optional[List[str]] = None,
    ):
        self.config = config
        self.wandb_callback = wandb_callback
        self.red_team_prompts = red_team_prompts or RED_TEAM_PROMPTS
        self.state = TrainingState()

        # Clients initialized lazily
        self._service_client: Optional[tinker.ServiceClient] = None
        self._training_client: Optional[tinker.TrainingClient] = None
        self._sampling_client: Optional[tinker.SamplingClient] = None
        self._sft_baseline_client: Optional[tinker.SamplingClient] = None
        self._tokenizer = None

        # Data generator for constitutional pairs
        self.data_generator: Optional[ConstitutionalDataGenerator] = None

        # Generated training data
        self.constitutional_pairs: List[ConstitutionalPair] = []

        # Evaluator
        self.evaluator = CAIEvaluator(config)

    def _init_clients(self):
        """Initialize Tinker clients using ServiceClient pattern."""
        if self._training_client is not None:
            return

        logger.info(f"Initializing Tinker clients...")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  LoRA rank: {self.config.lora_rank}")

        # Initialize service client
        self._service_client = tinker.ServiceClient()

        # Create training client for LoRA training
        self._training_client = self._service_client.create_lora_training_client(
            base_model=self.config.model_name,
            rank=self.config.lora_rank,
        )

        # Create initial sampling client
        self._sampling_client = self._training_client.save_weights_and_get_sampling_client(
            name="init"
        )

        # Get tokenizer
        self._tokenizer = self._training_client.get_tokenizer()

        logger.info("Tinker clients initialized successfully")

    @property
    def training_client(self) -> tinker.TrainingClient:
        if self._training_client is None:
            self._init_clients()
        return self._training_client

    @property
    def sampling_client(self) -> tinker.SamplingClient:
        if self._sampling_client is None:
            self._init_clients()
        return self._sampling_client

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._init_clients()
        return self._tokenizer

    def refresh_sampling_client(self, name: Optional[str] = None):
        """Refresh sampling client with latest trained weights."""
        self._sampling_client = self.training_client.save_weights_and_get_sampling_client(
            name=name or f"step_{self.state.global_step}"
        )

    def save_sft_baseline(self):
        """Save SFT weights for baseline comparison."""
        logger.info("Saving SFT-only baseline weights...")
        self._sft_baseline_client = self.training_client.save_weights_and_get_sampling_client(
            name="sft_baseline"
        )
        # Store model path for potential export
        self._sft_model_path = getattr(self._sft_baseline_client, 'model_path', None)

    def export_checkpoint(self, output_dir: str, checkpoint_name: str = "final") -> Optional[str]:
        """
        Export current model checkpoint to local directory.

        Uses Tinker's download_checkpoint_archive_from_tinker_path to save
        LoRA weights locally for later use with the style evaluation.

        Args:
            output_dir: Directory to save checkpoint files
            checkpoint_name: Name for the checkpoint (e.g., 'sft', 'dpo_final')

        Returns:
            Path to saved checkpoint archive, or None if export failed
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get the model path from current sampling client
        if self._sampling_client is None:
            logger.error("No sampling client available for export")
            return None

        model_path = getattr(self._sampling_client, 'model_path', None)
        if not model_path:
            logger.error("Sampling client has no model_path attribute")
            return None

        logger.info(f"Exporting checkpoint from {model_path}...")

        try:
            # Create REST client to download weights
            rest_client = self._service_client.create_rest_client()

            # Download checkpoint archive
            future = rest_client.download_checkpoint_archive_from_tinker_path(model_path)
            archive_data = future.result()

            # Save to file
            checkpoint_path = output_dir / f"{checkpoint_name}.tar.gz"
            with open(checkpoint_path, "wb") as f:
                f.write(archive_data)

            logger.info(f"Checkpoint exported to: {checkpoint_path}")
            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"Failed to export checkpoint: {e}")
            return None

    def export_all_checkpoints(self, output_dir: str) -> Dict[str, Optional[str]]:
        """
        Export both SFT and DPO checkpoints to local directory.

        Returns dict mapping checkpoint names to paths.
        """
        results = {}

        # Export SFT baseline if available
        if self._sft_baseline_client:
            sft_path = getattr(self._sft_baseline_client, 'model_path', None)
            if sft_path:
                logger.info(f"Exporting SFT baseline from {sft_path}...")
                try:
                    rest_client = self._service_client.create_rest_client()
                    future = rest_client.download_checkpoint_archive_from_tinker_path(sft_path)
                    archive_data = future.result()

                    output_path = Path(output_dir) / "sft_baseline.tar.gz"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(archive_data)

                    results["sft_baseline"] = str(output_path)
                    logger.info(f"SFT baseline exported to: {output_path}")
                except Exception as e:
                    logger.error(f"Failed to export SFT baseline: {e}")
                    results["sft_baseline"] = None

        # Export final DPO checkpoint
        final_path = self.export_checkpoint(output_dir, "dpo_final")
        results["dpo_final"] = final_path

        return results

    # =========================================================================
    # Phase 1: Supervised Fine-Tuning on Helpful Examples
    # =========================================================================

    def sft_step(self, examples: List[dict]) -> float:
        """
        Single SFT training step.

        examples: List of {"instruction": str, "response": str}
        """
        datums = []

        for ex in examples:
            # Format as few-shot prompt to teach assistant behavior
            prompt = format_few_shot_prompt(ex["instruction"])
            full_text = prompt + " " + ex["response"]

            # Tokenize using tokenizer from training client
            tokens = self.tokenizer.encode(full_text)

            # Token shifting for next-token prediction
            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]

            # Create mask: only train on response tokens
            prompt_tokens = self.tokenizer.encode(prompt)
            prompt_len = len(prompt_tokens) - 1  # Adjusted for shift

            mask = [0.0] * prompt_len + [1.0] * (len(input_tokens) - prompt_len)

            datum = tinker.Datum(
                model_input=types.ModelInput.from_ints(input_tokens),
                loss_fn_inputs={
                    "target_tokens": target_tokens,
                    "weights": mask,
                },
            )
            datums.append(datum)

        # Forward-backward pass
        fwd_bwd_future = self.training_client.forward_backward(
            datums, loss_fn="cross_entropy"
        )
        optim_future = self.training_client.optim_step(
            types.AdamParams(learning_rate=self.config.sft_learning_rate)
        )

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        # Metrics use suffixes - try all formats (loss:sum, loss:mean, loss)
        metrics = fwd_bwd_result.metrics
        loss = metrics.get("loss:sum", metrics.get("loss:mean", metrics.get("loss", 0.0)))

        # Debug: Log full metrics on first step
        if self.state.global_step == 0:
            logger.info(f"[DEBUG] SFT metrics keys: {list(metrics.keys())}")
            logger.info(f"[DEBUG] SFT full metrics: {metrics}")

        return loss

    def run_sft_phase(
        self,
        examples: List[dict],
        run_eval: bool = True,
    ) -> TrainingResult:
        """Run full SFT phase."""
        logger.info(f"Starting SFT phase with {len(examples)} examples...")
        self.state.phase = "sft"
        losses = []

        for step in range(self.config.sft_steps):
            # Sample batch
            batch_start = (step * self.config.sft_batch_size) % len(examples)
            batch = examples[batch_start:batch_start + self.config.sft_batch_size]

            # Wrap around if needed
            if len(batch) < self.config.sft_batch_size:
                batch.extend(examples[:self.config.sft_batch_size - len(batch)])

            loss = self.sft_step(batch)
            losses.append(loss)

            self.state.global_step += 1
            self.state.sft_steps_completed += 1

            if step % 50 == 0:
                logger.info(f"SFT Step {step}: loss={loss:.4f}")

            if self.wandb_callback:
                self.wandb_callback({
                    "phase": "sft",
                    "step": step,
                    "loss": loss,
                })

            # Periodic evaluation
            if run_eval and step > 0 and step % self.config.eval_interval == 0:
                self.refresh_sampling_client()
                eval_summary = self.evaluator.full_evaluation(
                    self._sampling_client,
                    red_team_prompts=self.red_team_prompts,
                    model_tokenizer=self._tokenizer
                )
                logger.info(f"  Eval: ASR={eval_summary.attack_success_rate:.2%}, "
                           f"Helpfulness={eval_summary.avg_helpfulness_score:.2f}")
                if self.wandb_callback:
                    self.wandb_callback({
                        "phase": "sft",
                        "step": step,
                        "eval/asr": eval_summary.attack_success_rate,
                        "eval/helpfulness": eval_summary.avg_helpfulness_score,
                    })

        self.refresh_sampling_client()

        # Save SFT baseline for comparison
        self.save_sft_baseline()

        # Final evaluation
        eval_summary = None
        if run_eval:
            eval_summary = self.evaluator.full_evaluation(
                self._sampling_client,
                red_team_prompts=self.red_team_prompts,
                model_tokenizer=self._tokenizer
            )

        logger.info(f"SFT phase complete. Final loss: {losses[-1]:.4f}")

        return TrainingResult(
            phase="sft",
            final_loss=losses[-1] if losses else 0.0,
            steps_completed=self.state.sft_steps_completed,
            eval_summary=eval_summary,
        )

    # =========================================================================
    # Phase 2-3: Constitutional Data Generation
    # =========================================================================

    def generate_constitutional_data(
        self,
        instructions: Optional[List[str]] = None,
    ) -> List[ConstitutionalPair]:
        """Generate constitutional critique/revision pairs with iterative refinement."""
        if instructions is None:
            instructions = get_training_prompts()

        logger.info(f"Generating constitutional data for {len(instructions)} instructions...")
        logger.info(f"Using {self.config.num_revision_rounds} revision rounds per response")

        # Use current fine-tuned model for generation
        self.refresh_sampling_client()

        if self.data_generator is None:
            self.data_generator = ConstitutionalDataGenerator(self.config)
            # Point to our trained sampling client and tokenizer
            self.data_generator._sampling_client = self._sampling_client
            self.data_generator._tokenizer = self._tokenizer

        pairs = self.data_generator.generate_preference_dataset(instructions)
        self.constitutional_pairs = pairs

        # Log statistics
        total_revisions = sum(p.num_revisions for p in pairs)
        principles_used = {}
        for p in pairs:
            for principle in p.principles_used:
                principles_used[principle] = principles_used.get(principle, 0) + 1

        logger.info(f"Generated {len(pairs)} constitutional pairs")
        logger.info(f"Total revisions: {total_revisions}")
        logger.info(f"Principles distribution: {principles_used}")

        return pairs

    # =========================================================================
    # Phase 4-5: DPO Training
    # =========================================================================

    def dpo_step(self, pairs: List[ConstitutionalPair]) -> Dict[str, float]:
        """
        Single DPO training step using forward_backward_custom.

        DPO loss directly optimizes for preferred (revised) over
        rejected (original) responses.

        Returns dict with loss and metrics (accuracy, margin, etc.)
        """
        # We need to create datums for both preferred and rejected responses
        # The cookbook pattern alternates: [chosen_0, rejected_0, chosen_1, rejected_1, ...]
        all_datums = []
        pair_metadata = []  # Track prompt lengths for masking

        for pair in pairs:
            # Preferred response (final, after all revisions)
            preferred_prompt = format_few_shot_prompt(pair.instruction)
            preferred_full = preferred_prompt + " " + pair.final_response
            preferred_tokens = self.tokenizer.encode(preferred_full)

            # Rejected response (original, before revisions)
            rejected_full = preferred_prompt + " " + pair.original_response
            rejected_tokens = self.tokenizer.encode(rejected_full)

            # Token shifting for next-token prediction
            pref_input = preferred_tokens[:-1]
            pref_target = preferred_tokens[1:]
            rej_input = rejected_tokens[:-1]
            rej_target = rejected_tokens[1:]

            # Mask for response only (don't compute loss on prompt)
            prompt_tokens = self.tokenizer.encode(preferred_prompt)
            prompt_len = len(prompt_tokens) - 1

            pref_mask = [0.0] * prompt_len + [1.0] * (len(pref_input) - prompt_len)
            rej_mask = [0.0] * prompt_len + [1.0] * (len(rej_input) - prompt_len)

            # Create datums for preferred (chosen) response
            pref_datum = tinker.Datum(
                model_input=types.ModelInput.from_ints(pref_input),
                loss_fn_inputs={
                    "target_tokens": pref_target,
                    "weights": pref_mask,
                },
            )

            # Create datums for rejected response
            rej_datum = tinker.Datum(
                model_input=types.ModelInput.from_ints(rej_input),
                loss_fn_inputs={
                    "target_tokens": rej_target,
                    "weights": rej_mask,
                },
            )

            # Add in alternating order: chosen, rejected, chosen, rejected...
            all_datums.append(pref_datum)
            all_datums.append(rej_datum)

            pair_metadata.append({
                "pref_mask": pref_mask,
                "rej_mask": rej_mask,
            })

        # Get reference log probs from current model (before this step's update)
        # We need a reference snapshot - use the sampling client which has the pre-step weights
        ref_logprobs = self._get_reference_logprobs(all_datums)

        # Define DPO loss function
        dpo_beta = self.config.kl_coef

        def dpo_loss_fn(
            data: List[tinker.Datum],
            logprobs_list: List[torch.Tensor]
        ) -> tuple[torch.Tensor, Dict[str, float]]:
            """Custom DPO loss computed on log probabilities."""
            # Split into chosen and rejected (alternating pattern)
            chosen_logprobs = []
            rejected_logprobs = []
            chosen_ref_logprobs = []
            rejected_ref_logprobs = []

            for i in range(0, len(data), 2):
                # Get weights from datum - handle TensorData from Tinker
                chosen_weights_raw = data[i].loss_fn_inputs["weights"]
                rejected_weights_raw = data[i + 1].loss_fn_inputs["weights"]
                if hasattr(chosen_weights_raw, 'tolist'):
                    chosen_weights_raw = chosen_weights_raw.tolist()
                elif hasattr(chosen_weights_raw, 'data'):
                    chosen_weights_raw = list(chosen_weights_raw.data)
                if hasattr(rejected_weights_raw, 'tolist'):
                    rejected_weights_raw = rejected_weights_raw.tolist()
                elif hasattr(rejected_weights_raw, 'data'):
                    rejected_weights_raw = list(rejected_weights_raw.data)
                chosen_weights = torch.tensor(chosen_weights_raw, dtype=torch.float32)
                rejected_weights = torch.tensor(rejected_weights_raw, dtype=torch.float32)

                # Policy log probs (current model)
                chosen_lp = logprobs_list[i]
                rejected_lp = logprobs_list[i + 1]

                # Reference log probs
                chosen_ref_lp = ref_logprobs[i]
                rejected_ref_lp = ref_logprobs[i + 1]

                # Compute weighted sum of log probs for each sequence
                # Truncate to match lengths
                min_len_c = min(len(chosen_lp), len(chosen_weights))
                min_len_r = min(len(rejected_lp), len(rejected_weights))

                chosen_logprob = torch.dot(
                    chosen_lp[:min_len_c].float(),
                    chosen_weights[:min_len_c].float()
                )
                rejected_logprob = torch.dot(
                    rejected_lp[:min_len_r].float(),
                    rejected_weights[:min_len_r].float()
                )

                chosen_ref_logprob = torch.dot(
                    chosen_ref_lp[:min_len_c].float(),
                    chosen_weights[:min_len_c].float()
                )
                rejected_ref_logprob = torch.dot(
                    rejected_ref_lp[:min_len_r].float(),
                    rejected_weights[:min_len_r].float()
                )

                chosen_logprobs.append(chosen_logprob)
                rejected_logprobs.append(rejected_logprob)
                chosen_ref_logprobs.append(chosen_ref_logprob)
                rejected_ref_logprobs.append(rejected_ref_logprob)

            # Compute DPO loss: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
            # where log_ratio = log_pi - log_ref
            chosen_log_ratios = torch.stack([
                lp - ref_lp for lp, ref_lp in zip(chosen_logprobs, chosen_ref_logprobs)
            ])
            rejected_log_ratios = torch.stack([
                lp - ref_lp for lp, ref_lp in zip(rejected_logprobs, rejected_ref_logprobs)
            ])

            # DPO loss
            losses = -torch.log(torch.sigmoid(dpo_beta * (chosen_log_ratios - rejected_log_ratios)))
            loss = losses.mean()

            # Metrics
            accuracy = (chosen_log_ratios > rejected_log_ratios).float().mean().item()
            chosen_rewards = dpo_beta * chosen_log_ratios
            rejected_rewards = dpo_beta * rejected_log_ratios
            margin = (chosen_rewards - rejected_rewards).mean().item()

            metrics = {
                "dpo_loss": loss.item(),
                "accuracy": accuracy,
                "margin": margin,
                "chosen_reward": chosen_rewards.mean().item(),
                "rejected_reward": rejected_rewards.mean().item(),
            }

            return loss, metrics

        # Forward-backward with custom DPO loss
        fwd_bwd_result = self.training_client.forward_backward_custom(
            all_datums, dpo_loss_fn
        ).result()

        # Optimizer step
        self.training_client.optim_step(
            types.AdamParams(learning_rate=self.config.rlhf_learning_rate)
        ).result()

        return fwd_bwd_result.metrics

    def _get_reference_logprobs(self, datums: List[tinker.Datum]) -> List[torch.Tensor]:
        """Get log probabilities from the reference model (sampling client)."""
        async def compute_all_logprobs():
            tasks = []
            for datum in datums:
                # Get the full sequence including the last target token
                target_tokens = datum.loss_fn_inputs["target_tokens"]
                # Handle TensorData from Tinker - convert to list if needed
                if hasattr(target_tokens, 'tolist'):
                    target_tokens = target_tokens.tolist()
                elif hasattr(target_tokens, 'data'):
                    target_tokens = list(target_tokens.data)
                if target_tokens:
                    full_sequence = datum.model_input.append_int(int(target_tokens[-1]))
                else:
                    full_sequence = datum.model_input
                tasks.append(self.sampling_client.compute_logprobs_async(full_sequence))
            return await asyncio.gather(*tasks)

        # Run async computation
        loop = asyncio.new_event_loop()
        try:
            all_logprobs = loop.run_until_complete(compute_all_logprobs())
        finally:
            loop.close()

        # Convert to tensors (skip first token which is the prompt start)
        return [torch.tensor(lp[1:]) for lp in all_logprobs]

    def run_dpo_phase(
        self,
        pairs: List[ConstitutionalPair],
        run_eval: bool = True,
    ) -> TrainingResult:
        """Run DPO training phase."""
        logger.info(f"Starting DPO phase with {len(pairs)} pairs...")
        self.state.phase = "dpo"
        losses = []
        accuracies = []

        # Create a fresh reference snapshot before DPO training
        # This is the "frozen" reference model used throughout DPO
        logger.info("Creating reference model snapshot for DPO...")
        self.refresh_sampling_client("dpo_reference")

        for step in range(self.config.rlhf_steps):
            # Sample batch
            batch_start = (step * self.config.rlhf_batch_size) % len(pairs)
            batch = pairs[batch_start:batch_start + self.config.rlhf_batch_size]

            if len(batch) < self.config.rlhf_batch_size:
                batch.extend(pairs[:self.config.rlhf_batch_size - len(batch)])

            metrics = self.dpo_step(batch)
            loss = metrics.get("dpo_loss", 0.0)
            accuracy = metrics.get("accuracy", 0.0)
            losses.append(loss)
            accuracies.append(accuracy)

            self.state.global_step += 1
            self.state.dpo_steps_completed += 1

            if step % 50 == 0:
                logger.info(f"DPO Step {step}: loss={loss:.4f}, accuracy={accuracy:.2%}, margin={metrics.get('margin', 0):.3f}")

            if self.wandb_callback:
                self.wandb_callback({
                    "phase": "dpo",
                    "step": step,
                    "loss": loss,
                    "dpo/accuracy": accuracy,
                    "dpo/margin": metrics.get("margin", 0),
                    "dpo/chosen_reward": metrics.get("chosen_reward", 0),
                    "dpo/rejected_reward": metrics.get("rejected_reward", 0),
                })

            # Periodic evaluation
            if run_eval and step > 0 and step % self.config.eval_interval == 0:
                self.refresh_sampling_client()
                eval_summary = self.evaluator.full_evaluation(
                    self._sampling_client,
                    red_team_prompts=self.red_team_prompts,
                    model_tokenizer=self._tokenizer
                )
                logger.info(f"  Eval: ASR={eval_summary.attack_success_rate:.2%}, "
                           f"Helpfulness={eval_summary.avg_helpfulness_score:.2f}")
                if self.wandb_callback:
                    self.wandb_callback({
                        "phase": "dpo",
                        "step": step,
                        "eval/asr": eval_summary.attack_success_rate,
                        "eval/helpfulness": eval_summary.avg_helpfulness_score,
                    })

        self.refresh_sampling_client()

        # Final evaluation
        eval_summary = None
        if run_eval:
            eval_summary = self.evaluator.full_evaluation(
                self._sampling_client,
                red_team_prompts=self.red_team_prompts,
                model_tokenizer=self._tokenizer
            )

        final_loss = losses[-1] if losses else 0.0
        final_accuracy = accuracies[-1] if accuracies else 0.0
        logger.info(f"DPO phase complete. Final loss: {final_loss:.4f}, accuracy: {final_accuracy:.2%}")

        return TrainingResult(
            phase="dpo",
            final_loss=final_loss,
            steps_completed=self.state.dpo_steps_completed,
            eval_summary=eval_summary,
        )

    # =========================================================================
    # Full Pipeline
    # =========================================================================

    def train(
        self,
        sft_examples: Optional[List[dict]] = None,
        constitutional_prompts: Optional[List[str]] = None,
        run_eval: bool = True,
        export_checkpoints: bool = True,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run full CAI training pipeline.

        1. SFT on helpful examples
        2. Generate constitutional pairs (iterative revision)
        3. DPO training on pairs
        4. Compare SFT-only vs Full CAI
        5. Export checkpoints locally (if enabled)

        Args:
            sft_examples: Optional list of SFT examples
            constitutional_prompts: Optional list of prompts for constitutional data generation
            run_eval: Whether to run evaluations during/after training
            export_checkpoints: If True, export trained checkpoints locally for later use
            checkpoint_dir: Directory to save checkpoints (required if export_checkpoints=True)

        Returns comprehensive results including:
        - Training losses
        - Evaluation metrics at each phase
        - Model comparison
        - Checkpoint paths (if exported)
        """
        # Prepare SFT data
        if sft_examples is None:
            sft_examples = [
                {"instruction": p.instruction, "response": p.response}
                for p in HELPFUL_PROMPTS
            ]

        results: Dict[str, Any] = {}

        # Phase 1: SFT
        sft_result = self.run_sft_phase(sft_examples, run_eval=run_eval)
        results["sft"] = {
            "final_loss": sft_result.final_loss,
            "steps": sft_result.steps_completed,
        }
        if sft_result.eval_summary:
            results["sft"]["eval"] = {
                "asr": sft_result.eval_summary.attack_success_rate,
                "helpfulness": sft_result.eval_summary.avg_helpfulness_score,
                "harmlessness": sft_result.eval_summary.harmlessness_score,
            }

        # Phase 2-3: Generate constitutional data
        pairs = self.generate_constitutional_data(constitutional_prompts)
        results["data_generation"] = {
            "num_pairs": len(pairs),
            "num_revision_rounds": self.config.num_revision_rounds,
            "principles_used": len(self.config.constitution),
        }

        # Phase 4-5: DPO training
        dpo_result = self.run_dpo_phase(pairs, run_eval=run_eval)
        results["dpo"] = {
            "final_loss": dpo_result.final_loss,
            "steps": dpo_result.steps_completed,
        }
        if dpo_result.eval_summary:
            results["dpo"]["eval"] = {
                "asr": dpo_result.eval_summary.attack_success_rate,
                "helpfulness": dpo_result.eval_summary.avg_helpfulness_score,
                "harmlessness": dpo_result.eval_summary.harmlessness_score,
            }

        # Compare SFT-only vs Full CAI
        if self._sft_baseline_client and run_eval:
            logger.info("Running baseline comparison...")

            sft_only_eval = self.evaluator.full_evaluation(
                self._sft_baseline_client,
                red_team_prompts=self.red_team_prompts,
                model_tokenizer=self._tokenizer
            )
            full_cai_eval = dpo_result.eval_summary

            results["comparison"] = {
                "sft_only": {
                    "asr": sft_only_eval.attack_success_rate,
                    "helpfulness": sft_only_eval.avg_helpfulness_score,
                    "harmlessness": sft_only_eval.harmlessness_score,
                },
                "full_cai": {
                    "asr": full_cai_eval.attack_success_rate if full_cai_eval else None,
                    "helpfulness": full_cai_eval.avg_helpfulness_score if full_cai_eval else None,
                    "harmlessness": full_cai_eval.harmlessness_score if full_cai_eval else None,
                },
            }

            # Calculate improvement
            if full_cai_eval:
                asr_reduction = sft_only_eval.attack_success_rate - full_cai_eval.attack_success_rate
                helpfulness_change = full_cai_eval.avg_helpfulness_score - sft_only_eval.avg_helpfulness_score

                results["comparison"]["improvement"] = {
                    "asr_reduction": asr_reduction,
                    "asr_reduction_pct": asr_reduction / max(sft_only_eval.attack_success_rate, 0.01) * 100,
                    "helpfulness_change": helpfulness_change,
                }

                logger.info(f"CAI reduced ASR by {asr_reduction:.2%} ({results['comparison']['improvement']['asr_reduction_pct']:.1f}%)")
                logger.info(f"Helpfulness change: {helpfulness_change:+.2f}")

        # Export checkpoints locally
        if export_checkpoints:
            if checkpoint_dir is None:
                logger.warning("export_checkpoints=True but no checkpoint_dir provided, skipping export")
            else:
                logger.info(f"Exporting checkpoints to {checkpoint_dir}...")
                checkpoint_paths = self.export_all_checkpoints(checkpoint_dir)
                results["checkpoints"] = checkpoint_paths
                logger.info(f"Checkpoints exported: {checkpoint_paths}")

        return results


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    config = CAIConfig(
        sft_steps=10,  # Quick test
        rlhf_steps=10,
        num_revision_rounds=2,
    )

    trainer = CAITrainer(config)

    print("CAI Trainer initialized successfully!")
    print(f"Model: {config.model_name}")
    print(f"Constitution has {len(config.constitution)} principles")
    print(f"Revision rounds: {config.num_revision_rounds}")
    print(f"Red-team prompts: {len(RED_TEAM_PROMPTS)}")
    print(f"Helpful eval prompts: {len(EVAL_HELPFUL_PROMPTS)}")
