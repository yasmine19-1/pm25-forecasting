"""Public interfaces for the shared validation splits."""
from src.folds import generate, load_folds, main, verify

__all__ = ["generate", "load_folds", "main", "verify"]
