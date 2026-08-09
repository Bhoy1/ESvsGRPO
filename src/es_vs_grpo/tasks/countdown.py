#binary reward structure


import re
from typing import Dict, List, Tuple
from .base import BaseTask
from datasets import load_dataset

SYSTEM_PROMPT = """You are a math assistant. Using the given numbers, create an arithmetic expression that equals the target.

Rules:
- Use each number exactly once
- Use +, -, *, / operations

Please reason step by step, and put your final answer within \\boxed{}."""


class CountdownTaskNew(BaseTask):
    """Countdown numbers game - reach target using arithmetic operations."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.num_train = config.get('train_size', 1000)
        self.num_test = config.get('test_size', 200)
        self.num_numbers = config.get('num_numbers', None)
        self.system_prompt = SYSTEM_PROMPT

    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load countdown data from HuggingFace."""
        dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

        if self.num_numbers is not None:
            dataset = dataset.filter(lambda x: len(x["nums"]) == self.num_numbers)

        all_data = []
        for ex in dataset:
            user_content = f"Numbers: {ex['nums']}\nTarget: {ex['target']}"

            all_data.append({
                'context': user_content,
                'target': ex['target'],
                'numbers': ex['nums'],
            })

        train_data = all_data[:self.num_train]
        test_data = all_data[self.num_train:self.num_train + self.num_test]

        return train_data, test_data

    def compute_reward(self, prompt: str, response: str, data_item: Dict) -> Dict[str, float]:
        """Compute countdown-specific reward."""
        numbers = data_item['numbers']
        target = data_item['target']

        # Extract answer from \boxed{...}
        # Handles both \boxed{answer} and \boxed{answer}
        boxed_regex = r"\\boxed\{([^}]*)\}"
        matches = re.findall(boxed_regex, response)

        if not matches:
            return {'reward': 0.0, 'correct': 0.0}

        answer_content = matches[-1].strip()

        # TODO: Convert LaTeX notation (\div, \times, \cdot, ×, ÷) to Python operators (/, *)
        # Currently \boxed{57 + (36 \div 9)} fails because \div is not in allowed chars

        # Check if answer contains only valid characters
        allowed_chars = r"^[0-9+\-*/() ]+$"
        if not re.match(allowed_chars, answer_content):
            return {'reward': 0.0, 'correct': 0.0}

        # Check if all numbers are used exactly once
        used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
        if sorted(used_numbers) != sorted(numbers):
            return {'reward': 0.0, 'correct': 0.0}

        # Try to evaluate the expression
        try:
            result = eval(answer_content, {"__builtins__": None}, {})
            if abs(float(result) - float(target)) < 1e-5:
                return {'reward': 1.0, 'correct': 1.0}
        except:
            return {'reward': 0.0, 'correct': 0.0}

        return {'reward': 0.0, 'correct': 0.0}

    def get_metrics_description(self) -> Dict[str, str]:
        return {
            'reward': 'Countdown task reward (1.0 if correct expression, 0.0 otherwise)',
            'correct': 'Whether the answer is correct',
        }
