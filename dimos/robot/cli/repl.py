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

import sys
from typing import Any

import rpyc
import typer

from dimos.core.repl_server import DEFAULT_REPL_PORT
from dimos.core.run_registry import get_most_recent


def repl_command(host: str, port: int | None) -> None:
    if port is None:
        entry = get_most_recent(alive_only=True)
        if entry and entry.repl_port:
            port = entry.repl_port
        else:
            port = DEFAULT_REPL_PORT

    try:
        conn = rpyc.connect(host, port, config={"sync_request_timeout": None})
    except ConnectionRefusedError:
        typer.echo(
            f"Error: cannot connect to {host}:{port}\n"
            "Is DimOS running? (REPL is enabled by default with 'dimos run')",
            err=True,
        )
        raise typer.Exit(1)

    coordinator = conn.root.get_coordinator()

    # Cache worker connections so each worker is connected to at most once.
    _worker_conns: dict[tuple[str, int], rpyc.Connection] = {}

    def modules() -> list[str]:
        """List deployed module names."""
        return list(conn.root.list_modules())

    def get(name: str):  # type: ignore[no-untyped-def]
        """Get a module instance by class name (connects directly to its worker)."""
        location = conn.root.get_module_location(name)
        if location is None:
            available = modules()
            raise KeyError(f"Module '{name}' not found. Available: {available}")
        w_host = str(location[0])
        w_port = int(location[1])
        key = (w_host, w_port)
        if key not in _worker_conns or _worker_conns[key].closed:
            _worker_conns[key] = rpyc.connect(w_host, w_port, config={"sync_request_timeout": None})
        return _worker_conns[key].root.get_instance_by_name(name)

    ns: dict[str, Any] = {
        "coordinator": coordinator,
        "modules": modules,
        "get": get,
        "conn": conn,
        "rpyc": rpyc,
    }

    banner = (
        "DimOS REPL\n"
        f"Connected to {host}:{port}\n"
        "\n"
        "  coordinator  ModuleCoordinator instance\n"
        "  modules()    List deployed module names\n"
        "  get(name)    Get module instance by class name\n"
    )

    use_ipython = _has_ipython() and _is_interactive()

    try:
        if use_ipython:
            import IPython

            print(banner)
            IPython.start_ipython(argv=[], user_ns=ns, display_banner=False)  # type: ignore[no-untyped-call]

        else:
            import code

            code.interact(banner, local=ns)
    finally:
        for wc in _worker_conns.values():
            try:
                wc.close()
            except Exception:
                pass
        conn.close()


def _has_ipython() -> bool:
    try:
        import IPython  # noqa: F401
    except ImportError:
        return False
    return True


def _is_interactive() -> bool:
    return bool(hasattr(sys, "ps1") or sys.flags.interactive or sys.stdin.isatty())
