import re
import random
from typing import Dict, List, Tuple
from .base import BaseTask
from datasets import load_dataset

SYSTEM_PROMPT = """You are a chemistry expert. Reason step by step, then provide your final answer as JSON, e.g., "answer": "C"."""


class SciKnowEvalChemistryTask(BaseTask):
    """SciKnowEval Chemistry multiple-choice question answering task."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.num_train = config.get('train_size', 200)
        self.num_test = config.get('test_size', 2000)
        self.system_prompt = SYSTEM_PROMPT

    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load SciKnowEval Chemistry MCQ data from HuggingFace."""
        dataset = load_dataset('hicai-zju/SciKnowEval', 'v2')

        # Filter to Chemistry domain and MCQ type
        chem_mcq = [ex for ex in dataset['test']
                    if ex['domain'] == 'Chemistry' and ex['type'] == 'mcq-4-choices']

        # Shuffle with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(chem_mcq)

        # Split: first num_train for train, next num_test for test (no overlap)
        train_examples = chem_mcq[:self.num_train]
        test_examples = chem_mcq[self.num_train:self.num_train + self.num_test]

        def format_example(ex):
            """Format a single example into the expected structure."""
            # Build choices string
            choices = ex['choices']
            choices_str = ""
            for label, text in zip(choices['label'], choices['text']):
                choices_str += f"{label}) {text}\n"

            # Format context with question and choices
            user_content = f"""Question: {ex['question']}

{choices_str.strip()}

Please show your choice in the answer field with only the choice letter, e.g., "answer": "C"."""

            return {
                'context': user_content,
                'question': ex['question'],
                'choices': ex['choices'],
                'answer': ex['answerKey'],
                'level': ex['details']['level'],
            }

        train_data = [format_example(ex) for ex in train_examples]
        test_data = [format_example(ex) for ex in test_examples]

        return train_data, test_data

    def compute_reward(self, prompt: str, response: str, data_item: Dict) -> Dict[str, float]:
        """Compute reward based on whether the predicted answer matches the correct answer."""
        correct_answer = data_item['answer'].upper()

        # Try to extract answer from response using multiple patterns
        predicted_answer = None

        # Pattern 1: "answer": "C" or "answer": "c" (with or without spaces)
        json_pattern = r'"answer"\s*:\s*"?([A-Da-d])"?'
        matches = re.findall(json_pattern, response, re.IGNORECASE)
        if matches:
            predicted_answer = matches[-1].upper()

        # Pattern 2: answer: C (no quotes)
        if predicted_answer is None:
            no_quote_pattern = r'answer\s*:\s*([A-Da-d])\b'
            matches = re.findall(no_quote_pattern, response, re.IGNORECASE)
            if matches:
                predicted_answer = matches[-1].upper()

        # Pattern 3: Fallback - find last standalone A/B/C/D (word boundary)
        if predicted_answer is None:
            fallback_pattern = r'\b([A-Da-d])\b'
            matches = re.findall(fallback_pattern, response)
            # Filter to only A, B, C, D
            valid_matches = [m.upper() for m in matches if m.upper() in ['A', 'B', 'C', 'D']]
            if valid_matches:
                predicted_answer = valid_matches[-1]

        # Check if prediction matches correct answer
        if predicted_answer == correct_answer:
            return {'reward': 1.0, 'correct': 1.0}

        return {'reward': 0.0, 'correct': 0.0}

    def get_metrics_description(self) -> Dict[str, str]:
        return {
            'reward': 'SciKnowEval Chemistry task reward (1.0 if MCQ answer is correct, 0.0 otherwise)',
            'correct': 'Whether the multiple choice answer is correct',
        }
