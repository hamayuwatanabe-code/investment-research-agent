"""Phase 3E.2.1: a socket-level guard shared by every *_live_smoke_offline
test file (SEC/ClinicalTrials/Form4) -- defense in depth on top of
``FakeHttpClient``/env-monkeypatching discipline, catching exactly the
class of bug this phase fixed: a test that implicitly assumed an ambient
environment variable (``IRA_SEC_USER_AGENT``) was unset, and silently
made a real network request when it genuinely was set on the machine
running the suite.

Patches ``socket.socket.connect``/``connect_ex`` for the duration of a
test so any attempted connection to a host OTHER than loopback raises
``AssertionError`` immediately, regardless of which HTTP client class
(``urllib``-based ``AllowlistedHttpClient`` included) ends up being
constructed. Never applied to ``tests/integration/`` -- those tests
deliberately talk to a local ``ThreadingHTTPServer`` on 127.0.0.1, which
this guard still allows.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

_ALLOWED_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _host_of(address: object) -> object:
    return address[0] if isinstance(address, tuple) else address


@contextmanager
def forbid_external_network() -> Iterator[None]:
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guarded_connect(self: socket.socket, address: object, *args: object, **kwargs: object) -> object:
        host = _host_of(address)
        if host not in _ALLOWED_HOSTS:
            raise AssertionError(
                f"forbid_external_network: refused a real socket connection to {address!r} -- "
                "offline tests must never reach a real external host"
            )
        return real_connect(self, address, *args, **kwargs)  # type: ignore[arg-type]

    def guarded_connect_ex(self: socket.socket, address: object, *args: object, **kwargs: object) -> object:
        host = _host_of(address)
        if host not in _ALLOWED_HOSTS:
            raise AssertionError(
                f"forbid_external_network: refused a real socket connect_ex to {address!r} -- "
                "offline tests must never reach a real external host"
            )
        return real_connect_ex(self, address, *args, **kwargs)  # type: ignore[arg-type]

    socket.socket.connect = guarded_connect  # type: ignore[method-assign,assignment]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign,assignment]
    try:
        yield
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = real_connect_ex  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def forbid_external_network_autouse() -> Iterator[None]:
    """Import this fixture's NAME into an offline test module (``from
    ._network_guard import forbid_external_network_autouse  # noqa: F401``)
    to apply it, autouse, to every test in that module."""
    with forbid_external_network():
        yield
