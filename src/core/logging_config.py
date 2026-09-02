#!/usr/bin/env python3
"""
Global logging and console output configuration module.
Provides context managers and flags to hide or reveal detailed logs.
"""

import contextlib
import os
import sys

_VERBOSE = False


def set_verbose(enabled: bool):
    global _VERBOSE
    _VERBOSE = bool(enabled)


def is_verbose() -> bool:
    return _VERBOSE


def log_msg(msg: str):
    """Prints message only if verbose mode is enabled."""
    if _VERBOSE:
        print(msg)


@contextlib.contextmanager
def suppress_stdout(suppress: bool = True):
    """Context manager to silence stdout if suppress is True."""
    if not suppress:
        yield
        return

    old_stdout = sys.stdout
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
