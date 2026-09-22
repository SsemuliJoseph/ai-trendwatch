"""
FILE: tests/conftest.py
Shared pytest fixtures.

`configure_logging` installs handlers on the ROOT logger. A test that calls main() (or configure_logging
directly) would otherwise leak those handlers into every later test, so they are removed after each test.
Only handlers created by app.utils.logging_setup are touched (pytest's own capture handlers are left alone).
"""
import pytest

from app.utils.logging_setup import reset_logging


@pytest.fixture(autouse=True)
def _restore_logging_after_each_test():
    yield
    reset_logging()
