"""Logging for trnrt.

Library modules log through `logging.getLogger(__name__)` (or `get(name)` for a fixed name); they never configure
handlers. Each process's entry point calls `configure()` once: the CLI (`python -m trnrt`), the standalone router
(`python -m trnrt.serve.router`) and every tensor-parallel rank a spawn start method begins afresh. Messages keep the
plain one-line form operators grep for ("trnrt.router: worker 0 (...) is down"), one per line on stderr.
"""

from __future__ import annotations

import logging
import os
import sys

ROOT = "trnrt"
_MARK = "_trnrt_handler"


def get(name: str) -> logging.Logger:
    """The logger for a module or area, under trnrt's root ("trnrt.router", "trnrt.engine.tp")."""
    return logging.getLogger(name if name == ROOT or name.startswith(ROOT + ".") else f"{ROOT}.{name}")


def configure(level: int | str | None = None) -> None:
    """Send trnrt's records to stderr, one message per line. Idempotent. The level is `level`, else TRNRT_LOG_LEVEL,
    else INFO."""
    root = logging.getLogger(ROOT)
    root.setLevel(level if level is not None else os.environ.get("TRNRT_LOG_LEVEL", "INFO").upper())
    if any(getattr(h, _MARK, False) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    setattr(handler, _MARK, True)
    root.addHandler(handler)


def _reaches_output(logger: logging.Logger) -> bool:
    """Whether a record from logger reaches any handler other than a NullHandler."""
    lg: logging.Logger | None = logger
    while lg is not None:
        if any(not isinstance(h, logging.NullHandler) for h in lg.handlers):
            return True
        if not lg.propagate:
            return False
        lg = lg.parent
    return False


def always(logger: logging.Logger, level: int, msg: str, *args: object) -> None:
    """Log a message that must never vanish (a hang report right before the process exits): if no entry point has
    configured logging, it goes to stderr as well."""
    logger.log(level, msg, *args)
    if not _reaches_output(logger) or not logger.isEnabledFor(level):
        sys.stderr.write((msg % args if args else msg) + "\n")
        sys.stderr.flush()


# A library adds no output of its own until an entry point configures it (the logging HOWTO's rule for libraries).
logging.getLogger(ROOT).addHandler(logging.NullHandler())
