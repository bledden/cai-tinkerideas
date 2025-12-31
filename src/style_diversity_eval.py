"""
Style Diversity Evaluation for Constitutional AI Models.

Tests the hypothesis that CAI models trained from base models retain
more "base model-like" qualities, specifically the ability to write
in diverse styles that instruction-tuned models struggle to adopt.

This evaluation measures:
1. Style adherence: Can the model follow style instructions?
2. Style distinctiveness: Are outputs measurably different across styles?
3. Style quality: Is the quality maintained across different styles?

Reference: The CAI paper suggests base-model-derived CAI models may be
"better at writing in different styles that are hard to steer existing
instruction-tuned models to follow."
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import statistics

import tinker
from tinker import types
from transformers import AutoTokenizer

from config import CAIConfig
from data_generation import format_few_shot_prompt

logger = logging.getLogger(__name__)


# =============================================================================
# Style Evaluation Prompts
# =============================================================================

STYLE_PROMPTS = {
    # Creative/literary styles
    "formal_academic": {
        "instruction": "Write in a formal academic style with precise terminology.",
        "test_prompts": [
            "Explain why exercise is good for health.",
            "Describe the water cycle.",
            "Discuss the importance of education.",
        ],
        "markers": ["furthermore", "consequently", "methodology", "empirical", "hypothesis"],
    },
    "casual_conversational": {
        "instruction": "Write in a casual, conversational tone like you're chatting with a friend.",
        "test_prompts": [
            "Explain why exercise is good for health.",
            "Describe the water cycle.",
            "Discuss the importance of education.",
        ],
        "markers": ["hey", "you know", "pretty much", "like", "cool", "stuff"],
    },
    "poetic_literary": {
        "instruction": "Write in a poetic, literary style with vivid imagery and metaphors.",
        "test_prompts": [
            "Describe a sunset.",
            "Write about the feeling of loneliness.",
            "Describe autumn leaves falling.",
        ],
        "markers": ["metaphor", "imagery", "like a", "as if", "dances", "whispers"],
    },
    "technical_precise": {
        "instruction": "Write in a technical, precise style suitable for documentation.",
        "test_prompts": [
            "Explain how to back up files.",
            "Describe the function of a CPU.",
            "Explain what an API is.",
        ],
        "markers": ["execute", "parameter", "function", "implementation", "specification"],
    },
    "storytelling_narrative": {
        "instruction": "Write in a storytelling narrative style, as if beginning a tale.",
        "test_prompts": [
            "Describe how computers were invented.",
            "Explain the discovery of penicillin.",
            "Describe the first moon landing.",
        ],
        "markers": ["once upon", "it all began", "little did", "story", "journey"],
    },
    "minimalist_terse": {
        "instruction": "Write in a minimalist, terse style. Be extremely brief and direct.",
        "test_prompts": [
            "Explain why sleep is important.",
            "Describe what a computer does.",
            "Explain democracy.",
        ],
        "markers": [],  # Measured by brevity
    },
    "enthusiastic_excited": {
        "instruction": "Write with enthusiasm and excitement, like you're sharing amazing news!",
        "test_prompts": [
            "Explain why learning is fun.",
            "Describe your favorite season.",
            "Tell me about a cool science fact.",
        ],
        "markers": ["amazing", "incredible", "wow", "exciting", "!", "fantastic"],
    },
    "socratic_questioning": {
        "instruction": "Write in a Socratic style, using questions to guide understanding.",
        "test_prompts": [
            "Help someone understand ethics.",
            "Explain the concept of justice.",
            "Teach about critical thinking.",
        ],
        "markers": ["?", "what if", "consider", "have you thought", "why do you think"],
    },
}

# Non-English style tests
NON_ENGLISH_STYLES = {
    "french_phrases": {
        "instruction": "Incorporate French phrases naturally into your English response.",
        "test_prompts": [
            "Describe a romantic dinner.",
            "Discuss fashion and style.",
            "Write about fine cuisine.",
        ],
        "markers": ["c'est", "bon", "je", "très", "mais", "merci"],
    },
    "shakespeare_archaic": {
        "instruction": "Write in the style of Shakespeare with archaic English.",
        "test_prompts": [
            "Write about love.",
            "Describe ambition.",
            "Write about betrayal.",
        ],
        "markers": ["thou", "thee", "hath", "doth", "'tis", "wherefore", "prithee"],
    },
}

# Persona-based styles (harder for instruction-tuned models)
PERSONA_STYLES = {
    "grumpy_old_person": {
        "instruction": "Write as a grumpy elderly person who thinks everything was better in the old days.",
        "test_prompts": [
            "What do you think about smartphones?",
            "How should young people behave?",
            "What's your opinion on modern music?",
        ],
        "markers": ["back in my day", "kids these days", "used to be", "ridiculous", "nonsense"],
    },
    "overexcited_child": {
        "instruction": "Write as an overexcited 8-year-old who finds everything amazing.",
        "test_prompts": [
            "Tell me about dinosaurs.",
            "What's space like?",
            "Describe ice cream.",
        ],
        "markers": ["and then", "so cool", "wow", "awesome", "really really", "!"],
    },
    "noir_detective": {
        "instruction": "Write in the style of a hard-boiled noir detective narrating.",
        "test_prompts": [
            "Describe a rainy night.",
            "Tell me about a mysterious stranger.",
            "Describe entering a dark room.",
        ],
        "markers": ["dame", "rain", "shadows", "trouble", "walked in", "cigarette"],
    },
}


@dataclass
class StyleResult:
    """Result of evaluating a single style prompt."""
    style_name: str
    prompt: str
    response: str
    style_adherence_score: float  # 1-5: How well did it follow the style?
    marker_count: int  # Number of style markers found
    word_count: int
    explanation: str


@dataclass
class StyleEvaluationSummary:
    """Summary of style diversity evaluation."""
    model_name: str

    # Per-style metrics
    style_scores: Dict[str, float] = field(default_factory=dict)
    style_marker_rates: Dict[str, float] = field(default_factory=dict)
    style_word_counts: Dict[str, float] = field(default_factory=dict)

    # Aggregate metrics
    mean_style_adherence: float = 0.0
    style_adherence_std: float = 0.0
    style_distinctiveness: float = 0.0  # Variance in outputs across styles

    # Detailed results
    results: List[StyleResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "style_scores": self.style_scores,
            "style_marker_rates": self.style_marker_rates,
            "style_word_counts": self.style_word_counts,
            "mean_style_adherence": self.mean_style_adherence,
            "style_adherence_std": self.style_adherence_std,
            "style_distinctiveness": self.style_distinctiveness,
            "n_evaluations": len(self.results),
        }


class StyleDiversityEvaluator:
    """
    Evaluates a model's ability to write in diverse styles.

    Uses a judge model to assess style adherence and measures
    statistical distinctiveness across style outputs.
    """

    DEFAULT_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

    def __init__(
        self,
        judge_model: Optional[str] = None,
    ):
        self.judge_model = judge_model or self.DEFAULT_JUDGE_MODEL
        self._service_client: Optional[tinker.ServiceClient] = None
        self._judge_client: Optional[tinker.SamplingClient] = None
        self._tokenizer = None

    def _init_judge_client(self):
        """Initialize judge client."""
        if self._judge_client is not None:
            return

        self._service_client = tinker.ServiceClient()

        model_name = self.judge_model
        if model_name.startswith("meta-llama/Llama-3"):
            tokenizer_id = "thinkingmachineslabinc/meta-llama-3-tokenizer"
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            self._tokenizer.chat_template = (
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
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        self._judge_client = self._service_client.create_sampling_client(
            base_model=self.judge_model
        )

    @property
    def judge_client(self) -> tinker.SamplingClient:
        if self._judge_client is None:
            self._init_judge_client()
        return self._judge_client

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._init_judge_client()
        return self._tokenizer

    def _count_markers(self, text: str, markers: List[str]) -> int:
        """Count how many style markers appear in the text."""
        text_lower = text.lower()
        count = 0
        for marker in markers:
            count += text_lower.count(marker.lower())
        return count

    def _judge_style_adherence(
        self,
        style_name: str,
        style_instruction: str,
        prompt: str,
        response: str,
    ) -> Tuple[float, str]:
        """
        Judge how well the response adheres to the requested style.

        Returns (score 1-5, explanation).
        """
        judge_content = f"""You are evaluating whether an AI assistant followed a specific writing style instruction.

STYLE INSTRUCTION: "{style_instruction}"

USER PROMPT: "{prompt}"

ASSISTANT RESPONSE: "{response}"

Rate how well the response adheres to the requested style from 1 to 5:
1 = Completely ignored the style instruction
2 = Barely followed the style, mostly standard response
3 = Partial style adherence, some elements present
4 = Good style adherence, clearly attempted the style
5 = Excellent style adherence, fully embodied the requested style

Consider:
- Does the vocabulary match the requested style?
- Does the tone/register match?
- Does the structure/format match?
- Is it natural or forced?

Format your response as:
SCORE: [1-5]
EXPLANATION: [Brief explanation of your rating]"""

        messages = [{"role": "user", "content": judge_content}]
        prompt_tokens = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )

        result = self.judge_client.sample(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=256,
                temperature=0.1,
            ),
        ).result()

        judgment = self.tokenizer.decode(
            result.sequences[0].tokens,
            skip_special_tokens=True
        ).strip() if result.sequences else ""

        # Extract score
        score = 3.0  # default
        score_match = re.search(r"SCORE:\s*(\d)", judgment)
        if score_match:
            try:
                score = float(score_match.group(1))
                score = min(5.0, max(1.0, score))
            except ValueError:
                pass

        # Extract explanation
        explanation = ""
        exp_match = re.search(r"EXPLANATION:\s*(.+)", judgment, re.DOTALL)
        if exp_match:
            explanation = exp_match.group(1).strip()

        return score, explanation

    def _is_garbage_response(self, response: str) -> bool:
        """
        Detect if a response is garbage/malformed output.

        This catches cases where the model produces nonsensical repetitive
        patterns instead of coherent text.
        """
        if not response or len(response) < 10:
            return True

        # Check for excessive repetition of short patterns
        words = response.split()
        if len(words) < 5:
            return False  # Too short to judge

        # Count unique words - garbage often has very low uniqueness
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.15:  # Less than 15% unique words
            logger.warning(f"[GARBAGE DETECTION] Low unique word ratio: {unique_ratio:.2f}")
            return True

        # Check for repetitive number sequences (common garbage pattern)
        import re
        number_sequences = re.findall(r'\d+\)', response)
        if len(number_sequences) > 10:
            logger.warning(f"[GARBAGE DETECTION] Excessive numbered sequences: {len(number_sequences)}")
            return True

        # Check for common garbage indicators
        garbage_patterns = [
            r'(\b\w+\b)(\s+\1){5,}',  # Same word repeated 5+ times
            r'(\d+\)\s*){10,}',  # Numbered lists that go on too long
            r'(User:|Assistant:){3,}',  # Repetitive conversation markers
        ]
        for pattern in garbage_patterns:
            if re.search(pattern, response):
                logger.warning(f"[GARBAGE DETECTION] Matched pattern: {pattern[:30]}...")
                return True

        return False

    def evaluate_style(
        self,
        sampling_client: tinker.SamplingClient,
        model_tokenizer,
        style_name: str,
        style_config: dict,
        is_instruct: bool = False,
    ) -> List[StyleResult]:
        """
        Evaluate a model on a single style.

        Args:
            sampling_client: The model's sampling client
            model_tokenizer: Tokenizer for the model
            style_name: Name of the style being tested
            style_config: Style configuration dict
            is_instruct: If True, use chat template formatting instead of few-shot
        """
        results = []

        instruction = style_config["instruction"]
        markers = style_config.get("markers", [])

        for prompt in style_config["test_prompts"]:
            try:
                # Create the styled prompt
                styled_prompt = f"{instruction}\n\nPrompt: {prompt}"

                if is_instruct:
                    # Use chat template for instruction-tuned models
                    messages = [{"role": "user", "content": styled_prompt}]
                    if hasattr(model_tokenizer, 'apply_chat_template'):
                        prompt_tokens = model_tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                        )
                    else:
                        # Fallback if no chat template
                        logger.warning(f"No chat template for tokenizer, using raw encoding")
                        prompt_tokens = model_tokenizer.encode(styled_prompt)
                else:
                    # Use few-shot prompting for base/CAI models
                    formatted = format_few_shot_prompt(styled_prompt)
                    prompt_tokens = model_tokenizer.encode(formatted)

                # Generate response
                response = sampling_client.sample(
                    prompt=types.ModelInput.from_ints(prompt_tokens),
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=300,
                        temperature=0.7,
                    ),
                ).result()

                response_text = model_tokenizer.decode(
                    response.sequences[0].tokens,
                    skip_special_tokens=True
                ).strip() if response.sequences else ""

                # Validate response is not garbage
                if self._is_garbage_response(response_text):
                    logger.error(
                        f"[GARBAGE OUTPUT] Model produced invalid response for style '{style_name}':\n"
                        f"  First 200 chars: {response_text[:200]!r}"
                    )
                    # Still record it but with score 0 and flag it
                    results.append(StyleResult(
                        style_name=style_name,
                        prompt=prompt,
                        response=response_text,
                        style_adherence_score=0.0,  # Invalid response
                        marker_count=0,
                        word_count=len(response_text.split()),
                        explanation="INVALID: Garbage/malformed output detected",
                    ))
                    continue

                # Judge style adherence
                score, explanation = self._judge_style_adherence(
                    style_name, instruction, prompt, response_text
                )

                # Count markers
                marker_count = self._count_markers(response_text, markers)
                word_count = len(response_text.split())

                results.append(StyleResult(
                    style_name=style_name,
                    prompt=prompt,
                    response=response_text,
                    style_adherence_score=score,
                    marker_count=marker_count,
                    word_count=word_count,
                    explanation=explanation,
                ))

                logger.debug(f"Style eval [{style_name}]: score={score}, markers={marker_count}")

            except Exception as e:
                logger.warning(f"Failed to evaluate style '{style_name}' on prompt: {e}")
                continue

        return results

    def full_evaluation(
        self,
        sampling_client: tinker.SamplingClient,
        model_tokenizer,
        model_name: str,
        include_non_english: bool = True,
        include_personas: bool = True,
        is_instruct: bool = False,
    ) -> StyleEvaluationSummary:
        """
        Run full style diversity evaluation.

        Args:
            sampling_client: Client for the model being evaluated
            model_tokenizer: Tokenizer for the model
            model_name: Name of the model (for logging)
            include_non_english: Include non-English style tests
            include_personas: Include persona-based styles
            is_instruct: If True, use chat template formatting (for instruction-tuned models)

        Returns:
            StyleEvaluationSummary with all metrics
        """
        logger.info(f"Starting style diversity evaluation for {model_name}")
        logger.info(f"  Mode: {'instruction-tuned (chat template)' if is_instruct else 'base model (few-shot)'}")

        all_results = []
        style_scores = defaultdict(list)
        style_markers = defaultdict(list)
        style_words = defaultdict(list)

        # Collect all styles to test
        styles_to_test = dict(STYLE_PROMPTS)
        if include_non_english:
            styles_to_test.update(NON_ENGLISH_STYLES)
        if include_personas:
            styles_to_test.update(PERSONA_STYLES)

        # Evaluate each style
        for style_name, style_config in styles_to_test.items():
            logger.info(f"  Evaluating style: {style_name}")

            results = self.evaluate_style(
                sampling_client, model_tokenizer, style_name, style_config,
                is_instruct=is_instruct,
            )

            all_results.extend(results)

            for r in results:
                style_scores[style_name].append(r.style_adherence_score)
                style_markers[style_name].append(r.marker_count)
                style_words[style_name].append(r.word_count)

        # Compute per-style averages
        avg_style_scores = {
            name: statistics.mean(scores) if scores else 0.0
            for name, scores in style_scores.items()
        }
        avg_marker_rates = {
            name: statistics.mean(counts) if counts else 0.0
            for name, counts in style_markers.items()
        }
        avg_word_counts = {
            name: statistics.mean(counts) if counts else 0.0
            for name, counts in style_words.items()
        }

        # Compute aggregate metrics
        all_scores = [r.style_adherence_score for r in all_results]
        mean_adherence = statistics.mean(all_scores) if all_scores else 0.0
        std_adherence = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0

        # Style distinctiveness: variance in word counts across styles
        # Higher variance = more distinct outputs per style
        style_word_means = list(avg_word_counts.values())
        distinctiveness = statistics.variance(style_word_means) if len(style_word_means) > 1 else 0.0

        summary = StyleEvaluationSummary(
            model_name=model_name,
            style_scores=avg_style_scores,
            style_marker_rates=avg_marker_rates,
            style_word_counts=avg_word_counts,
            mean_style_adherence=mean_adherence,
            style_adherence_std=std_adherence,
            style_distinctiveness=distinctiveness,
            results=all_results,
        )

        logger.info(f"Style evaluation complete: mean_adherence={mean_adherence:.2f}, "
                   f"distinctiveness={distinctiveness:.1f}")

        return summary


def compare_style_diversity(
    cai_client: tinker.SamplingClient,
    cai_tokenizer,
    instruct_client: tinker.SamplingClient,
    instruct_tokenizer,
    output_dir: Optional[Path] = None,
) -> Dict[str, StyleEvaluationSummary]:
    """
    Compare style diversity between CAI model and instruction-tuned baseline.

    Returns dict with results for both models.
    """
    evaluator = StyleDiversityEvaluator()

    results = {}

    # Evaluate CAI model
    logger.info("=" * 60)
    logger.info("Evaluating CAI model style diversity...")
    logger.info("=" * 60)
    results["cai"] = evaluator.full_evaluation(
        cai_client, cai_tokenizer, "CAI-from-base"
    )

    # Evaluate instruction-tuned model
    logger.info("=" * 60)
    logger.info("Evaluating instruction-tuned model style diversity...")
    logger.info("=" * 60)
    results["instruct"] = evaluator.full_evaluation(
        instruct_client, instruct_tokenizer, "Instruction-tuned"
    )

    # Print comparison
    print("\n" + "=" * 70)
    print("STYLE DIVERSITY COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<35} {'CAI':>15} {'Instruct':>15}")
    print("-" * 70)
    print(f"{'Mean Style Adherence (1-5)':<35} {results['cai'].mean_style_adherence:>15.2f} "
          f"{results['instruct'].mean_style_adherence:>15.2f}")
    print(f"{'Style Adherence Std Dev':<35} {results['cai'].style_adherence_std:>15.2f} "
          f"{results['instruct'].style_adherence_std:>15.2f}")
    print(f"{'Style Distinctiveness':<35} {results['cai'].style_distinctiveness:>15.1f} "
          f"{results['instruct'].style_distinctiveness:>15.1f}")
    print("-" * 70)

    # Per-style comparison
    print("\nPer-Style Adherence Scores:")
    print(f"{'Style':<25} {'CAI':>10} {'Instruct':>10} {'Δ':>10}")
    print("-" * 55)

    for style in results["cai"].style_scores:
        cai_score = results["cai"].style_scores.get(style, 0)
        inst_score = results["instruct"].style_scores.get(style, 0)
        delta = cai_score - inst_score
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        print(f"{style:<25} {cai_score:>10.2f} {inst_score:>10.2f} {delta_str:>10}")

    print("=" * 70)

    # Save results if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        comparison = {
            "cai": results["cai"].to_dict(),
            "instruct": results["instruct"].to_dict(),
            "comparison": {
                "adherence_difference": results["cai"].mean_style_adherence - results["instruct"].mean_style_adherence,
                "distinctiveness_difference": results["cai"].style_distinctiveness - results["instruct"].style_distinctiveness,
                "cai_better_styles": [
                    s for s in results["cai"].style_scores
                    if results["cai"].style_scores[s] > results["instruct"].style_scores.get(s, 0)
                ],
                "instruct_better_styles": [
                    s for s in results["instruct"].style_scores
                    if results["instruct"].style_scores[s] > results["cai"].style_scores.get(s, 0)
                ],
            }
        }

        with open(output_dir / "style_diversity_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Results saved to {output_dir}/style_diversity_comparison.json")

    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="Evaluate style diversity of CAI models")
    parser.add_argument("--cai-checkpoint", type=str, required=True,
                       help="Path to CAI model checkpoint")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-3B",
                       help="Base model name")
    parser.add_argument("--instruct-model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                       help="Instruction-tuned model for comparison")
    parser.add_argument("--output-dir", type=str, default="results/style_eval",
                       help="Output directory for results")

    args = parser.parse_args()

    # Initialize clients
    service_client = tinker.ServiceClient()

    # CAI model (with LoRA checkpoint)
    cai_client = service_client.create_sampling_client(
        base_model=args.base_model,
        lora_path=args.cai_checkpoint,
    )

    # Instruction-tuned baseline
    instruct_client = service_client.create_sampling_client(
        base_model=args.instruct_model,
    )

    # Load tokenizers
    tokenizer_id = "thinkingmachineslabinc/meta-llama-3-tokenizer"
    cai_tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    instruct_tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    # Run comparison
    results = compare_style_diversity(
        cai_client, cai_tokenizer,
        instruct_client, instruct_tokenizer,
        output_dir=Path(args.output_dir),
    )
