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

import threading
from typing import TYPE_CHECKING, Any

import rpyc
from rpyc.utils.server import ThreadedServer

from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.core.module_coordinator import ModuleCoordinator

logger = setup_logger()

DEFAULT_REPL_PORT = 18861

_RPYC_PROTOCOL_CONFIG = {
    "allow_public_attrs": True,
    "allow_all_attrs": True,
    "allow_getattr": True,
    "allow_setattr": True,
    "allow_delattr": True,
}

_THREAD_NAME = "worker-repl-server"


def start_worker_repl_server(instances: dict[int, Any], host: str = "localhost") -> int:
    """Start an RPyC server inside a worker process.

    Returns the port the server is listening on (uses port 0 for auto-assign).
    """

    class WorkerReplService(rpyc.Service):  # type: ignore[misc]
        def on_connect(self, conn: rpyc.Connection) -> None:
            conn._config.update(_RPYC_PROTOCOL_CONFIG)

        def exposed_get_instance_by_name(self, name: str) -> Any:
            for inst in instances.values():
                if type(inst).__name__ == name:
                    return inst
            available = [type(inst).__name__ for inst in instances.values()]
            raise KeyError(f"'{name}' not found on this worker. Available: {available}")

        def exposed_list_instances(self) -> dict[int, str]:
            return {mid: type(inst).__name__ for mid, inst in instances.items()}

    server = ThreadedServer(
        WorkerReplService,
        hostname=host,
        port=0,
        protocol_config=_RPYC_PROTOCOL_CONFIG,
    )
    thread = threading.Thread(target=server.start, daemon=True, name=_THREAD_NAME)
    thread.start()
    return int(server.port)


def _make_service(coordinator: ModuleCoordinator) -> type[rpyc.Service]:
    """Create an RPyC service class bound to *coordinator*."""

    class DimosReplService(rpyc.Service):  # type: ignore[misc]
        ALIASES = ["dimos"]

        def on_connect(self, conn: rpyc.Connection) -> None:
            conn._config.update(_RPYC_PROTOCOL_CONFIG)

        def exposed_get_coordinator(self) -> ModuleCoordinator:
            return coordinator

        def exposed_list_modules(self) -> list[str]:
            return coordinator.list_modules()

        def exposed_get_module_location(self, name: str) -> tuple[str, int] | None:
            """Return (host, port) of the worker RPyC server hosting *name*."""
            return coordinator.get_module_location(name)

    return DimosReplService


class ReplServer:
    """Manages an RPyC server for interactive REPL access to a running coordinator."""

    def __init__(
        self,
        coordinator: ModuleCoordinator,
        port: int = DEFAULT_REPL_PORT,
        host: str = "localhost",
    ) -> None:
        self._coordinator = coordinator
        self._port = port
        self._host = host
        self._server: ThreadedServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> None:
        service_cls = _make_service(self._coordinator)
        self._server = ThreadedServer(
            service_cls,
            hostname=self._host,
            port=self._port,
            protocol_config=_RPYC_PROTOCOL_CONFIG,
        )
        self._thread = threading.Thread(target=self._server.start, daemon=True, name="repl-server")
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
            self._thread = None
