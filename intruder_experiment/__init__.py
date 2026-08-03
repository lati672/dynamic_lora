"""Continual-learning experiment for analyzing intruder dimensions."""

from .data import TASK_SPECS, load_and_sample_tasks
from .modeling import ContinualClassifier

__all__ = ["TASK_SPECS", "ContinualClassifier", "load_and_sample_tasks"]
