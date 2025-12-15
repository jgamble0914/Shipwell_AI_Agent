"""Utility functions for the AI agent."""

import os
from pathlib import Path


def load_env_file(env_path: str = ".env") -> None:
    """
    Load environment variables from a .env file.

    Args:
        env_path: Path to the .env file
    """
    env_path = Path(env_path)
    if not env_path.exists():
        return

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()


def validate_folder(folder_path: str) -> bool:
    """
    Validate that a folder path exists.

    Args:
        folder_path: Path to validate

    Returns:
        True if folder exists, False otherwise
    """
    return Path(folder_path).exists() and Path(folder_path).is_dir()
