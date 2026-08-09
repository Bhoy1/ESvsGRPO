from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

class BaseTask(ABC):
    """Abstract base class for all tasks in continual learning."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Task-specific configuration dict
        """
        self.config = config
        self.name = config.get('name', 'unnamed_task')
        self.system_prompt = None  # Subclasses can set this for chat template

    @abstractmethod
    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load and return train/test data.

        Returns:
            (train_data, test_data) where each is a list of dicts with:
                - 'context': str (the prompt)
                - ... (task-specific fields)
        """
        pass

    @abstractmethod
    def compute_reward(self, prompt: str, response: str, data_item: Dict) -> Dict[str, float]:
        """
        Compute reward for a generated response.

        Args:
            prompt: The input prompt
            response: Model's generated response
            data_item: Original data dict (contains ground truth)

        Returns:
            Dict with at least {'reward': float}, can include other metrics
        """
        pass

    def preprocess_prompt(self, data_item: Dict) -> str:
        """
        Optional: Transform data_item into prompt text.
        Default just returns data_item['context'].
        """
        return data_item.get('context', '')

    def get_metrics_description(self) -> Dict[str, str]:
        """
        Optional: Return description of metrics this task computes.

        Returns:
            Dict mapping metric names to descriptions
        """
        return {
            'reward': 'Primary task reward'
        }
