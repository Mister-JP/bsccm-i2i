"""Small helpers for silencing noisy third-party stdout/stderr output."""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def run_quietly(fn: Callable[[], T]) -> T:
    """Run a callable while suppressing stdout/stderr."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn()
