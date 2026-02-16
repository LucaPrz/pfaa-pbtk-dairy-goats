"""
Shared helpers for resolving project- and data-level paths in pfaa-pbtk-dairy-goats.

Centralising these here allows scripts across the project to use a
single source of truth for where data, results, and docs live.
"""

from pathlib import Path


def get_project_root() -> Path:
    """
    Return the project root directory (the repository root).

    Resolves to the directory containing this file's parent (auxiliary/),
    so the repo folder can be renamed without breaking paths.
    """
    return Path(__file__).resolve().parents[1]


def get_clean_root() -> Path:
    """
    Return the root directory of the project (same as get_project_root()).
    Kept for backwards compatibility.
    """
    return get_project_root()


def get_data_root() -> Path:
    """
    Return the root directory for clean data:
        <clean_root>/data
    """
    return get_clean_root() / "data"


def get_results_root() -> Path:
    """
    Return the root directory for clean analysis/optimization results:
        <clean_root>/results
    """
    return get_clean_root() / "results"

