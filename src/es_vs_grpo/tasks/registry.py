"""Task registry for the four-task sequential-learning experiments."""

from typing import Any, Mapping, Type

from .base import BaseTask
from .boolq import BoolQTask
from .chemistry import SciKnowEvalChemistryTask
from .countdown import CountdownTaskNew
from .math import MATHTask


TASK_REGISTRY: Mapping[str, Type[BaseTask]] = {
    "countdown": CountdownTaskNew,
    "math": MATHTask,
    "sciknoweval_chemistry": SciKnowEvalChemistryTask,
    "boolq": BoolQTask,
}


def create_task(config: Mapping[str, Any]) -> BaseTask:
    """Construct a task from its experiment configuration."""
    task_type = config["type"]
    try:
        task_class = TASK_REGISTRY[task_type]
    except KeyError as error:
        available = ", ".join(sorted(TASK_REGISTRY))
        raise ValueError(
            f"Unknown task type: {task_type}. Available tasks: {available}"
        ) from error
    return task_class(config)
