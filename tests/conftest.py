"""Suite-wide fixtures."""

import logging

import pytest

from trnrt import log


@pytest.fixture(autouse=True)
def _restore_trnrt_logging():
    """Entry points (`main()` of the CLI, the router, check_mixed) configure trnrt's logger for their process; tests run
    several in one process, so each test gets the logger back as it found it."""
    root = logging.getLogger(log.ROOT)
    handlers, level, propagate = list(root.handlers), root.level, root.propagate
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    root.propagate = propagate
