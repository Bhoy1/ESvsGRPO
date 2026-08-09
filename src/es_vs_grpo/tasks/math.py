import re
from typing import Dict, List, Tuple
from .base import BaseTask
from datasets import load_dataset

SYSTEM_PROMPT = """You are a math expert. Solve the following competition math problem.

Please reason step by step, and put your final answer within \\boxed{}."""


class MATHTask(BaseTask):
    """MATH dataset - competition-level math problems."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.num_train = config.get('train_size', 200)
        self.num_test = config.get('test_size', 500)
        self.stratify_by_level = config.get('stratify_by_level', True)
        self.system_prompt = SYSTEM_PROMPT

    def _extract_boxed_answer(self, text: str) -> str:
        """Extract answer from \\boxed{...}, handling nested braces."""
        # Find \boxed{ and then match braces
        idx = text.rfind('\\boxed{')
        if idx == -1:
            return ""

        # Start after \boxed{
        start = idx + 7
        brace_count = 1
        end = start

        while end < len(text) and brace_count > 0:
            if text[end] == '{':
                brace_count += 1
            elif text[end] == '}':
                brace_count -= 1
            end += 1

        if brace_count == 0:
            return text[start:end-1]
        return ""

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        answer = answer.strip()

        # Remove \text{...} wrappers
        answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)

        # Remove spaces
        answer = answer.replace(' ', '')

        # Remove trailing zeros after decimal
        if '.' in answer and not any(c.isalpha() for c in answer):
            try:
                val = float(answer)
                if val == int(val):
                    answer = str(int(val))
            except:
                pass

        # Normalize common latex
        answer = answer.replace('\\$', '')
        answer = answer.replace('$', '')
        answer = answer.replace('\\%', '')
        answer = answer.replace('%', '')
        answer = answer.replace('^{\\circ}', '')
        answer = answer.replace('^\\circ', '')
        answer = answer.replace('\\circ', '')

        # Handle dfrac same as frac
        answer = answer.replace('\\dfrac', '\\frac')
        answer = answer.replace('\\tfrac', '\\frac')

        return answer

    def _latex_frac_to_float(self, s: str) -> float:
        """Convert \\frac{a}{b} to float."""
        match = re.match(r'\\frac\{([^}]+)\}\{([^}]+)\}', s)
        if match:
            try:
                num = float(match.group(1))
                den = float(match.group(2))
                return num / den
            except:
                pass
        return None

    def _answers_equal(self, pred: str, gold: str) -> bool:
        """Check if two answers are equivalent."""
        pred = self._normalize_answer(pred)
        gold = self._normalize_answer(gold)

        # Direct string match
        if pred == gold:
            return True

        # Try numeric comparison
        try:
            # Handle fractions
            pred_val = self._latex_frac_to_float(pred)
            if pred_val is None:
                pred_val = float(pred.replace(',', ''))

            gold_val = self._latex_frac_to_float(gold)
            if gold_val is None:
                gold_val = float(gold.replace(',', ''))

            if abs(pred_val - gold_val) < 1e-6:
                return True
        except:
            pass

        # Try simple fraction notation (1/2 vs 0.5)
        try:
            if '/' in pred and '/' not in gold:
                parts = pred.split('/')
                pred_val = float(parts[0]) / float(parts[1])
                gold_val = float(gold)
                if abs(pred_val - gold_val) < 1e-6:
                    return True
            elif '/' in gold and '/' not in pred:
                parts = gold.split('/')
                gold_val = float(parts[0]) / float(parts[1])
                pred_val = float(pred)
                if abs(pred_val - gold_val) < 1e-6:
                    return True
        except:
            pass

        return False

    def _stratified_sample(self, dataset, num_samples: int) -> List[Dict]:
        """Sample evenly across difficulty levels 1-5."""
        # Group by level
        by_level = {i: [] for i in range(1, 6)}
        for ex in dataset:
            level = int(ex['level'])
            by_level[level].append(ex)

        # Calculate samples per level
        per_level = num_samples // 5
        remainder = num_samples % 5

        sampled = []
        for level in range(1, 6):
            n = per_level + (1 if level <= remainder else 0)
            level_data = by_level[level][:n]
            sampled.extend(level_data)

        return sampled

    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load MATH data from HuggingFace."""
        # Load dataset
        dataset = load_dataset("nlile/hendrycks-MATH-benchmark")
        train_dataset = dataset['train']
        test_dataset = dataset['test']

        # Sample train data
        if self.stratify_by_level:
            train_samples = self._stratified_sample(train_dataset, self.num_train)
        else:
            train_samples = list(train_dataset)[:self.num_train]

        # Use full test split (500 problems) - no stratification needed for eval
        test_samples = list(test_dataset)[:self.num_test]

        # Convert to our format
        train_data = []
        for ex in train_samples:
            train_data.append({
                'context': ex['problem'],
                'answer': ex['answer'],
                'full_solution': ex['solution'],
                'level': ex['level'],
                'subject': ex['subject'],
            })

        test_data = []
        for ex in test_samples:
            test_data.append({
                'context': ex['problem'],
                'answer': ex['answer'],
                'full_solution': ex['solution'],
                'level': ex['level'],
                'subject': ex['subject'],
            })

        return train_data, test_data

    def compute_reward(self, prompt: str, response: str, data_item: Dict) -> Dict[str, float]:
        """Compute MATH reward - binary (correct/incorrect)."""
        ground_truth = data_item['answer']

        # Extract answer from \boxed{...}
        predicted = self._extract_boxed_answer(response)

        if not predicted:
            return {'reward': 0.0, 'correct': 0.0}

        # Check if answers are equivalent
        if self._answers_equal(predicted, ground_truth):
            return {'reward': 1.0, 'correct': 1.0}

        return {'reward': 0.0, 'correct': 0.0}

    def get_metrics_description(self) -> Dict[str, str]:
        return {
            'reward': 'MATH task reward (1.0 if correct answer, 0.0 otherwise)',
            'correct': 'Whether the answer is correct',
        }
