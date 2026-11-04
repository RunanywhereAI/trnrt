"""trnrt/log.py: names stay under trnrt's root, configure() is idempotent and honours TRNRT_LOG_LEVEL, messages are bare."""

import logging

from trnrt import log


def _cleanup(root):
    for h in [h for h in root.handlers if getattr(h, log._MARK, False)]:
        root.removeHandler(h)
    root.setLevel(logging.NOTSET)


def test_names_stay_under_the_root():
    assert log.get("router").name == "trnrt.router"
    assert log.get("trnrt.engine.tp").name == "trnrt.engine.tp"
    assert log.get("trnrt").name == "trnrt"


def test_configure_once_bare_messages_on_stderr(capsys, monkeypatch):
    root = logging.getLogger(log.ROOT)
    monkeypatch.delenv("TRNRT_LOG_LEVEL", raising=False)
    try:
        log.configure()
        log.configure()
        assert sum(getattr(h, log._MARK, False) for h in root.handlers) == 1
        assert root.level == logging.INFO
        log.get("router").info("trnrt.router: worker 0 is down")
        log.get("router").debug("hidden at INFO")
        err = capsys.readouterr().err
        assert err == "trnrt.router: worker 0 is down\n"
    finally:
        _cleanup(root)


def test_level_from_the_environment(monkeypatch):
    root = logging.getLogger(log.ROOT)
    monkeypatch.setenv("TRNRT_LOG_LEVEL", "warning")
    try:
        log.configure()
        assert root.level == logging.WARNING
    finally:
        _cleanup(root)
