import re
from typing import Dict, List, Tuple
from .base import BaseTask
from datasets import load_dataset

SYSTEM_PROMPT = """You are a reading assistant. Answer yes or no based on the passage.

Please reason step by step, and put your final answer within \\boxed{}."""


class BoolQTask(BaseTask):
    """BoolQ binary yes/no question answering task."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.num_train = config.get('train_size', 1000)
        self.num_test = config.get('test_size', 200)
        self.system_prompt = SYSTEM_PROMPT

    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load BoolQ data from HuggingFace."""
        dataset = load_dataset("google/boolq")

        train_data = []
        for ex in dataset['train']:
            user_content = f"Passage: {ex['passage']}\n\nQuestion: {ex['question']}"

            train_data.append({
                'context': user_content,
                'passage': ex['passage'],
                'question': ex['question'],
                'answer': ex['answer'],
            })

        train_data = train_data[:self.num_train]

        test_data = []
        for ex in dataset['validation']:
            user_content = f"Passage: {ex['passage']}\n\nQuestion: {ex['question']}"

            test_data.append({
                'context': user_content,
                'passage': ex['passage'],
                'question': ex['question'],
                'answer': ex['answer'],
            })

        test_data = test_data[:self.num_test]

        return train_data, test_data

    def normalize_answer(self, text: str) -> str:
        """Normalize answer text for comparison."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def compute_reward(self, prompt: str, response: str, data_item: Dict) -> Dict[str, float]:
        """Compute BoolQ-specific reward."""
        correct_answer = data_item['answer']  # Boolean

        # Extract answer from \boxed{...}
        boxed_regex = r"\\boxed\{([^}]*)\}"
        matches = re.findall(boxed_regex, response)

        if not matches:
            return {'reward': 0.0, 'correct': 0.0}

        predicted_answer = self.normalize_answer(matches[-1].strip())

        # Check if prediction matches
        # Accept "yes", "true", "1" for True
        # Accept "no", "false", "0" for False
        if correct_answer:
            if predicted_answer in ['yes', 'true', '1', 'correct', 'right']:
                return {'reward': 1.0, 'correct': 1.0}
        else:
            if predicted_answer in ['no', 'false', '0', 'incorrect', 'wrong']:
                return {'reward': 1.0, 'correct': 1.0}

        return {'reward': 0.0, 'correct': 0.0}

    def get_metrics_description(self) -> Dict[str, str]:
        return {
            'reward': 'BoolQ task reward (1.0 if yes/no answer is correct, 0.0 otherwise)',
            'correct': 'Whether the yes/no answer is correct',
        }
