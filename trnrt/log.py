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
    if _configured():
        return
    # An application that configured the root logger already (logging.basicConfig, a framework) receives trnrt's
    # records through it; a handler of our own as well would print each one twice.
    if logging.getLogger().handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    setattr(handler, _MARK, True)
    root.addHandler(handler)


def _configured() -> bool:
    """Whether configure() installed trnrt's own stderr handler in this process."""
    return any(getattr(h, _MARK, False) for h in logging.getLogger(ROOT).handlers)


def _handlers(logger: logging.Logger) -> list[logging.Handler]:
    """Every handler a record from logger passes, in order (propagation followed)."""
    out: list[logging.Handler] = []
    lg: logging.Logger | None = logger
    while lg is not None:
        out += lg.handlers
        if not lg.propagate:
            break
        lg = lg.parent
    return out


def always(logger: logging.Logger, level: int, msg: str, *args: object) -> None:
    """Log a message that must never vanish (a hang report right before the process ends with os._exit).

    Every handler on the way is flushed, since nothing runs after the exit. Unless trnrt's own stderr handler
    (configure()) takes the record, the message is written to stderr directly as well: another handler may filter,
    buffer or drop it, and a duplicate line beats a lost one here."""
    logger.log(level, msg, *args)
    handlers = _handlers(logger)
    for h in handlers:
        h.flush()
    ours = logger.isEnabledFor(level) and any(getattr(h, _MARK, False) and h.level <= level for h in handlers)
    if not ours:
        sys.stderr.write((msg % args if args else msg) + "\n")
        sys.stderr.flush()


# A library adds no output of its own until an entry point configures it (the logging HOWTO's rule for libraries).
logging.getLogger(ROOT).addHandler(logging.NullHandler())
