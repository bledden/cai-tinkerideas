"""
Data generation for CAI training.

This module handles:
1. Generating initial helpful responses from base model (few-shot prompted)
2. Constitutional critique generation
3. Constitutional revision generation (iterative, multiple rounds)
4. Creating preference pairs (original, revised)

References:
- Constitutional AI paper: https://arxiv.org/abs/2212.08073
"""

import logging
import random
from dataclasses import dataclass, field
from typing import List, Optional

import tinker
from tinker import types
from transformers import AutoTokenizer

from config import CAIConfig, CONSTITUTION, RED_TEAM_PROMPTS, EVAL_HELPFUL_PROMPTS

logger = logging.getLogger(__name__)


@dataclass
class RevisionStep:
    """A single critique-revision step."""
    principle_name: str
    principle_category: str
    critique: str
    revised_response: str


@dataclass
class ConstitutionalPair:
    """A preference pair from constitutional revision."""
    instruction: str
    original_response: str
    final_response: str  # After all revision rounds
    revision_history: List[RevisionStep] = field(default_factory=list)

    @property
    def num_revisions(self) -> int:
        return len(self.revision_history)

    @property
    def principles_used(self) -> List[str]:
        return [step.principle_name for step in self.revision_history]


def get_helpful_few_shot_examples() -> List[dict]:
    """Return few-shot examples for helpful assistant behavior."""
    return [
        {
            "user": "What's the capital of France?",
            "assistant": "The capital of France is Paris. It's the largest city in France and serves as the country's political, economic, and cultural center."
        },
        {
            "user": "How do I boil an egg?",
            "assistant": "To boil an egg: 1) Place eggs in a pot and cover with cold water by 1 inch. 2) Bring to a boil over high heat. 3) Once boiling, remove from heat and cover. 4) Let sit 9-12 minutes for hard-boiled, 6-7 for soft-boiled. 5) Transfer to ice water to stop cooking."
        },
        {
            "user": "What causes rain?",
            "assistant": "Rain forms when water evaporates from oceans, lakes, and rivers into the atmosphere. As this water vapor rises, it cools and condenses around tiny particles to form clouds. When the water droplets in clouds become heavy enough, they fall as rain."
        },
    ]


def format_few_shot_prompt(instruction: str) -> str:
    """
    Build a few-shot prompt to get helpful responses from base model.

    Since we're starting from a base model (not instruction-tuned),
    we need few-shot examples to elicit helpful assistant behavior.
    """
    examples = get_helpful_few_shot_examples()

    prompt_parts = []
    for ex in examples:
        prompt_parts.append(f"User: {ex['user']}")
        prompt_parts.append(f"Assistant: {ex['assistant']}")
        prompt_parts.append("")

    prompt_parts.append(f"User: {instruction}")
    prompt_parts.append("Assistant:")

    return "\n".join(prompt_parts)


def format_critique_prompt(instruction: str, response: str, principle: dict) -> str:
    """
    Build prompt for constitutional critique.

    The model critiques its own response based on a constitutional principle.
    """
    return f"""Consider the following conversation:

User: {instruction}
Assistant: {response}

{principle['critique_prompt']}

Critique:"""


def format_revision_prompt(
    instruction: str,
    response: str,
    critique: str,
    principle: dict
) -> str:
    """
    Build prompt for constitutional revision.

    The model revises its response based on the critique.
    """
    return f"""Consider the following conversation:

User: {instruction}
Assistant: {response}

The response was critiqued as follows:
{critique}

{principle['revision_prompt']}

Revised response:"""


class ConstitutionalDataGenerator:
    """
    Generates constitutional AI training data using Tinker.

    Implements iterative critique-revision as described in the CAI paper:
    responses are revised multiple times with randomly sampled principles.
    """

    def __init__(self, config: CAIConfig):
        self.config = config
        self._service_client: Optional[tinker.ServiceClient] = None
        self._sampling_client: Optional[tinker.SamplingClient] = None
        self._tokenizer = None

    def _init_sampling_client(self):
        """Initialize sampling client using ServiceClient pattern."""
        if self._sampling_client is not None:
            return

        self._service_client = tinker.ServiceClient()
        self._sampling_client = self._service_client.create_sampling_client(
            base_model=self.config.model_name
        )

        # Load tokenizer directly (SamplingClient doesn't have get_tokenizer)
        model_name = self.config.model_name
        if model_name.startswith("meta-llama/Llama-3"):
            tokenizer_id = "thinkingmachineslabinc/meta-llama-3-tokenizer"
        else:
            tokenizer_id = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    @property
    def sampling_client(self) -> tinker.SamplingClient:
        if self._sampling_client is None:
            self._init_sampling_client()
        return self._sampling_client

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._init_sampling_client()
        return self._tokenizer

    def generate_helpful_response(self, instruction: str) -> str:
        """Generate a helpful response from the base model."""
        prompt = format_few_shot_prompt(instruction)
        prompt_tokens = self.tokenizer.encode(prompt)

        result = self.sampling_client.sample(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=512,
                temperature=0.7,
                stop=["\nUser:", "\n\nUser:"],
            ),
        ).result()

        # Decode the generated tokens
        if result.sequences:
            return self.tokenizer.decode(result.sequences[0].tokens).strip()
        return ""

    def generate_critique(
        self,
        instruction: str,
        response: str,
        principle: dict
    ) -> str:
        """Generate a critique of the response based on a constitutional principle."""
        prompt = format_critique_prompt(instruction, response, principle)
        prompt_tokens = self.tokenizer.encode(prompt)

        result = self.sampling_client.sample(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=256,
                temperature=self.config.critique_temperature,
            ),
        ).result()

        if result.sequences:
            return self.tokenizer.decode(result.sequences[0].tokens).strip()
        return ""

    def generate_revision(
        self,
        instruction: str,
        response: str,
        critique: str,
        principle: dict,
    ) -> str:
        """Generate a revised response based on the critique."""
        prompt = format_revision_prompt(instruction, response, critique, principle)
        prompt_tokens = self.tokenizer.encode(prompt)

        result = self.sampling_client.sample(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=512,
                temperature=self.config.revision_temperature,
                stop=["\nUser:", "\n\nUser:"],
            ),
        ).result()

        if result.sequences:
            return self.tokenizer.decode(result.sequences[0].tokens).strip()
        return ""

    def generate_constitutional_pair_iterative(
        self,
        instruction: str,
        num_rounds: Optional[int] = None,
    ) -> ConstitutionalPair:
        """
        Generate a constitutional pair with iterative revision.

        As per the CAI paper: "Responses are revised repeatedly in a sequence,
        where principles are randomly drawn from the constitution at each step."

        Args:
            instruction: The user's request
            num_rounds: Number of critique-revision rounds (default from config)

        Returns:
            ConstitutionalPair with original, final, and revision history
        """
        if num_rounds is None:
            num_rounds = self.config.num_revision_rounds

        # Step 1: Generate initial response
        original_response = self.generate_helpful_response(instruction)
        logger.info(f"Generated original response: {original_response[:100]}...")

        current_response = original_response
        revision_history: List[RevisionStep] = []

        # Step 2: Iterative critique-revision
        for round_idx in range(num_rounds):
            # Randomly sample a principle (as per paper)
            principle = random.choice(self.config.constitution)

            # Generate critique
            critique = self.generate_critique(instruction, current_response, principle)
            logger.debug(f"Round {round_idx + 1} critique ({principle['name']}): {critique[:100]}...")

            # Generate revision
            revised_response = self.generate_revision(
                instruction, current_response, critique, principle
            )
            logger.debug(f"Round {round_idx + 1} revision: {revised_response[:100]}...")

            # Record step
            revision_history.append(RevisionStep(
                principle_name=principle["name"],
                principle_category=principle.get("category", "unknown"),
                critique=critique,
                revised_response=revised_response,
            ))

            current_response = revised_response

        logger.info(f"Completed {num_rounds} revision rounds for: {instruction[:50]}...")

        return ConstitutionalPair(
            instruction=instruction,
            original_response=original_response,
            final_response=current_response,
            revision_history=revision_history,
        )

    def generate_preference_dataset(
        self,
        instructions: List[str],
    ) -> List[ConstitutionalPair]:
        """
        Generate a dataset of constitutional preference pairs.

        Each pair has (original, final) where final is preferred after
        multiple rounds of constitutional revision.
        """
        pairs = []

        for i, instruction in enumerate(instructions):
            try:
                logger.info(f"Processing instruction {i + 1}/{len(instructions)}: {instruction[:50]}...")
                pair = self.generate_constitutional_pair_iterative(instruction)
                pairs.append(pair)
            except Exception as e:
                logger.warning(f"Failed to generate pair for '{instruction[:50]}...': {e}")
                continue

        logger.info(f"Generated {len(pairs)} constitutional pairs")
        return pairs

    def generate_comparison_samples(
        self,
        instructions: List[str],
    ) -> List[dict]:
        """
        Generate samples for pairwise comparison (RLAIF style).

        Returns list of dicts with instruction, response_a, response_b
        for training a preference model.
        """
        samples = []

        for instruction in instructions:
            try:
                pair = self.generate_constitutional_pair_iterative(instruction)

                # Format for preference model training
                samples.append({
                    "instruction": instruction,
                    "chosen": pair.final_response,  # After revision
                    "rejected": pair.original_response,  # Before revision
                    "num_revisions": pair.num_revisions,
                    "principles_applied": pair.principles_used,
                })
            except Exception as e:
                logger.warning(f"Failed to generate sample: {e}")
                continue

        return samples


def get_training_prompts() -> List[str]:
    """Get combined list of prompts for training data generation."""
    return EVAL_HELPFUL_PROMPTS + RED_TEAM_PROMPTS


if __name__ == "__main__":
    # Test data generation
    logging.basicConfig(level=logging.INFO)

    config = CAIConfig(num_revision_rounds=2)  # Quick test with 2 rounds
    generator = ConstitutionalDataGenerator(config)

    # Test with a single prompt
    test_instruction = "How do I write a persuasive essay?"
    pair = generator.generate_constitutional_pair_iterative(test_instruction)

    print(f"\n{'='*60}")
    print(f"Instruction: {pair.instruction}")
    print(f"\n{'='*60}")
    print(f"Original response:\n{pair.original_response}")
    print(f"\n{'='*60}")
    print(f"Revision history ({pair.num_revisions} rounds):")
    for i, step in enumerate(pair.revision_history):
        print(f"\n--- Round {i + 1}: {step.principle_name} ({step.principle_category}) ---")
        print(f"Critique: {step.critique[:200]}...")
        print(f"Revision: {step.revised_response[:200]}...")
    print(f"\n{'='*60}")
    print(f"Final response:\n{pair.final_response}")
