"""Continual-learning experiment for the task sequence used by Shuttleworth et al."""

from .data import TASK_SPECS, load_and_sample_tasks
from .modeling import ContinualClassifier

__all__ = ["TASK_SPECS", "ContinualClassifier", "load_and_sample_tasks"]
