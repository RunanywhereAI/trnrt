"""trnrt/log.py: names stay under trnrt's root; configure() is idempotent, honours TRNRT_LOG_LEVEL and never doubles an
application's own handler; always() never loses a message, whatever the handlers do."""

import io
import logging
import logging.handlers

import pytest

from trnrt import log


@pytest.fixture
def bare_root(monkeypatch):
    """A root logger of our own for the test (put back after it), and trnrt's level unset. pytest adds its capture
    handler to the root when the test body starts, so each test clears the root first with _no_root_handlers()."""
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    monkeypatch.delenv("TRNRT_LOG_LEVEL", raising=False)
    return logging.getLogger(log.ROOT)


def _no_root_handlers():
    logging.getLogger().handlers.clear()


def test_names_stay_under_the_root():
    assert log.get("router").name == "trnrt.router"
    assert log.get("trnrt.engine.tp").name == "trnrt.engine.tp"
    assert log.get("trnrt").name == "trnrt"


def test_configure_once_bare_messages_on_stderr(bare_root, capsys):
    _no_root_handlers()
    log.configure()
    log.configure()
    assert sum(getattr(h, log._MARK, False) for h in bare_root.handlers) == 1
    assert bare_root.level == logging.INFO
    log.get("router").info("trnrt.router: worker 0 is down")
    log.get("router").debug("hidden at INFO")
    assert capsys.readouterr().err == "trnrt.router: worker 0 is down\n"


def test_configure_leaves_an_application_handler_alone(bare_root):
    """An application that configured the root logger first gets each record once, through its own handler."""
    _no_root_handlers()
    seen = io.StringIO()
    logging.getLogger().addHandler(logging.StreamHandler(seen))
    log.configure()
    assert not any(getattr(h, log._MARK, False) for h in bare_root.handlers)
    log.get("router").warning("trnrt.router: worker 1 is down")
    assert seen.getvalue() == "trnrt.router: worker 1 is down\n"


def test_level_from_the_environment(bare_root, monkeypatch):
    _no_root_handlers()
    monkeypatch.setenv("TRNRT_LOG_LEVEL", "warning")
    log.configure()
    assert bare_root.level == logging.WARNING


def test_always_reaches_stderr_without_a_configured_handler(bare_root, capsys):
    _no_root_handlers()
    log.always(log.get("watchdog"), logging.ERROR, "EXEC WATCHDOG rank %s: hung", 3)
    assert capsys.readouterr().err == "EXEC WATCHDOG rank 3: hung\n"


def test_always_logs_once_when_configured(bare_root, capsys):
    _no_root_handlers()
    log.configure()
    log.always(log.get("watchdog"), logging.ERROR, "EXEC WATCHDOG rank %s: hung", 3)
    assert capsys.readouterr().err == "EXEC WATCHDOG rank 3: hung\n"


def test_always_survives_a_handler_or_level_that_drops_the_record(bare_root, capsys):
    """A handler above the record's level, or a logger level that disables it, does not count as output."""
    _no_root_handlers()
    strict = logging.StreamHandler(io.StringIO())
    strict.setLevel(logging.CRITICAL)
    logging.getLogger().addHandler(strict)
    log.always(log.get("watchdog"), logging.ERROR, "EXEC WATCHDOG rank %s: hung", 5)
    assert capsys.readouterr().err == "EXEC WATCHDOG rank 5: hung\n"
    log.configure()
    bare_root.setLevel(logging.CRITICAL)
    log.always(log.get("watchdog"), logging.ERROR, "EXEC WATCHDOG rank %s: hung", 6)
    assert capsys.readouterr().err == "EXEC WATCHDOG rank 6: hung\n"


def test_always_flushes_a_buffering_handler(bare_root, capsys):
    """The process exits right after: a buffered record must be out before always() returns."""
    _no_root_handlers()
    sink = io.StringIO()
    logging.getLogger().addHandler(logging.handlers.MemoryHandler(capacity=100, target=logging.StreamHandler(sink)))
    log.always(log.get("watchdog"), logging.ERROR, "EXEC WATCHDOG rank %s: hung", 7)
    assert sink.getvalue() == "EXEC WATCHDOG rank 7: hung\n"
    assert capsys.readouterr().err == "EXEC WATCHDOG rank 7: hung\n"  # not trnrt's own handler: stderr as well
