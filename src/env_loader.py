"""
Environment variable loader for Tinker experiments.

Loads .env file from the current directory or project root.
"""

import os
from pathlib import Path


def load_env():
    """Load environment variables from .env file."""
    # Check current directory first, then parent
    for env_path in [Path(".env"), Path(__file__).parent / ".env"]:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        # Only set if not already in environment
                        if key and value and key not in os.environ:
                            os.environ[key] = value
            return True
    return False


def get_tinker_api_key():
    """Get Tinker API key, loading from .env if needed."""
    load_env()
    return os.environ.get("TINKER_API_KEY")


# Auto-load on import
load_env()
