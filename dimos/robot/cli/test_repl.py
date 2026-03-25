# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import pytest
import typer

from dimos.core.repl_server import DEFAULT_REPL_PORT, ReplServer
from dimos.robot.cli.repl import repl_command


@pytest.fixture
def force_stdlib_repl(mocker):
    mocker.patch("dimos.robot.cli.repl._has_ipython", return_value=False)


@pytest.fixture
def no_registry(mocker):
    mocker.patch("dimos.robot.cli.repl.get_most_recent", return_value=None)


@pytest.fixture
def mock_connect(mocker):
    return mocker.patch("dimos.robot.cli.repl.rpyc.connect", side_effect=ConnectionRefusedError)


@pytest.fixture
def repl_server(find_free_port, wait_until_rpyc_connectable, make_stub_coordinator):
    """Start a real ReplServer on a free port and tear it down after the test."""
    coordinator = make_stub_coordinator(
        modules={"ModuleA": ("127.0.0.1", 0), "ModuleB": ("127.0.0.1", 0)}
    )
    port = find_free_port()
    server = ReplServer(coordinator, port=port, host="127.0.0.1")
    server.start()

    wait_until_rpyc_connectable("127.0.0.1", port)

    yield port

    thread = server._thread
    server.stop()
    if thread is not None:
        thread.join(timeout=2.0)


def test_explicit_port_is_used(mock_connect):
    """An explicit port is forwarded to rpyc.connect as-is."""
    with pytest.raises(typer.Exit):
        repl_command(host="127.0.0.1", port=12345)

    assert mock_connect.call_args[0] == ("127.0.0.1", 12345)


def test_port_from_registry(mocker, mock_connect):
    """When port is None the registry entry's repl_port is used."""
    entry = mocker.MagicMock(repl_port=9999)
    mocker.patch("dimos.robot.cli.repl.get_most_recent", return_value=entry)

    with pytest.raises(typer.Exit):
        repl_command(host="127.0.0.1", port=None)

    assert mock_connect.call_args[0] == ("127.0.0.1", 9999)


def test_port_defaults_when_no_registry(mock_connect, no_registry):
    """Falls back to DEFAULT_REPL_PORT when no registry entry exists."""

    with pytest.raises(typer.Exit):
        repl_command(host="127.0.0.1", port=None)

    assert mock_connect.call_args[0] == ("127.0.0.1", DEFAULT_REPL_PORT)


def test_port_defaults_when_entry_has_no_repl_port(mocker, mock_connect):
    """Registry entry with repl_port=None still falls back to DEFAULT_REPL_PORT."""
    entry = mocker.MagicMock(repl_port=None)
    mocker.patch("dimos.robot.cli.repl.get_most_recent", return_value=entry)

    with pytest.raises(typer.Exit):
        repl_command(host="127.0.0.1", port=None)

    assert mock_connect.call_args[0] == ("127.0.0.1", DEFAULT_REPL_PORT)


def test_connection_refused_exits_with_helpful_message(capsys, find_free_port):
    """Real refused connection exits with code 1 and shows host:port."""
    port = find_free_port()

    with pytest.raises(typer.Exit) as exc_info:
        repl_command(host="127.0.0.1", port=port)

    assert exc_info.value.exit_code == 1
    assert f"127.0.0.1:{port}" in capsys.readouterr().err


def test_modules_lists_deployed_names(repl_server, mocker, force_stdlib_repl, no_registry):
    """modules() returns the names provided by the coordinator."""
    result = {}

    def _interact(banner, local):
        result["modules"] = local["modules"]()

    mocker.patch("code.interact", side_effect=_interact)
    repl_command(host="127.0.0.1", port=repl_server)

    assert set(result["modules"]) == {"ModuleA", "ModuleB"}


def test_get_raises_for_unknown_module(repl_server, mocker, force_stdlib_repl, no_registry):
    """get() raises KeyError when the module is not deployed."""
    ran = []

    def _interact(banner, local):
        with pytest.raises(KeyError, match="NoSuchModule"):
            local["get"]("NoSuchModule")
        ran.append(True)

    mocker.patch("code.interact", side_effect=_interact)
    repl_command(host="127.0.0.1", port=repl_server)

    assert ran  # guard: the assertion inside _interact actually executed
