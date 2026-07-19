import os
from enum import Enum
from pathlib import Path

UNIVERSE_ENV_VAR = "PYSYS_UNIVERSE"

class Universe(str, Enum):
    live      = "live"
    backtest  = "backtest"
    synthetic = "synthetic"

_DEFAULT = Universe.synthetic


def get_universe() -> Universe:
    """Return active Universe from PYSYS_UNIVERSE env var, defaulting to synthetic.

    Raises ValueError on unrecognised value — no silent fallback.
    """
    raw = os.environ.get(UNIVERSE_ENV_VAR, _DEFAULT.value)
    try:
        return Universe(raw)
    except ValueError:
        valid = [u.value for u in Universe]
        raise ValueError(
            f"PYSYS_UNIVERSE={raw!r} is not a valid universe; must be one of {valid}"
        )


def universe_subdir() -> str:
    """Return the universe name string for use as a path component."""
    return get_universe().value


def scoped_path(base_env_var: str) -> str:
    """Return os.environ[base_env_var] / <universe>, creating the directory.

    Raises KeyError if base_env_var is unset.
    Raises ValueError if PYSYS_UNIVERSE is invalid.
    """
    base = os.environ[base_env_var].rstrip("/")
    path = Path(base) / universe_subdir()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
