# Copyright 2025-2026 Dimensional Inc.
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

from concurrent.futures import ThreadPoolExecutor
import threading
from typing import TYPE_CHECKING, Any

from dimos.core.global_config import GlobalConfig, global_config
from dimos.core.module import ModuleBase, ModuleSpec
from dimos.core.repl_server import DEFAULT_REPL_PORT, ReplServer
from dimos.core.resource import Resource
from dimos.core.worker_manager import WorkerManager
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.core.resource_monitor.monitor import StatsMonitor
    from dimos.core.rpc_client import ModuleProxy
    from dimos.core.worker import Worker

logger = setup_logger()


class ModuleCoordinator(Resource):  # type: ignore[misc]
    _client: WorkerManager | None = None
    _global_config: GlobalConfig
    _n: int | None = None
    _memory_limit: str = "auto"
    _deployed_modules: dict[type[ModuleBase], ModuleProxy]
    _stats_monitor: StatsMonitor | None = None
    _repl_server: ReplServer | None = None
    _module_locations: dict[str, tuple[str, int]]

    def __init__(
        self,
        n: int | None = None,
        cfg: GlobalConfig = global_config,
    ) -> None:
        self._n = n if n is not None else cfg.n_workers
        self._memory_limit = cfg.memory_limit
        self._global_config = cfg
        self._deployed_modules = {}
        self._module_locations = {}

    @property
    def workers(self) -> list[Worker]:
        """Active worker processes."""
        if self._client is None:
            return []
        return self._client.workers

    @property
    def n_workers(self) -> int:
        """Number of active workers."""
        return len(self.workers)

    def health_check(self) -> bool:
        """Verify all workers are alive after build.

        Since ``blueprint.build()`` is synchronous, every module should be
        started by the time this runs.  We just confirm no worker has died.
        """
        if self.n_workers == 0:
            logger.error("health_check: no workers found")
            return False

        for w in self.workers:
            if w.pid is None:
                logger.error("health_check: worker died", worker_id=w.worker_id)
                return False

        return True

    @property
    def n_modules(self) -> int:
        """Number of deployed modules."""
        return len(self._deployed_modules)

    @property
    def client(self) -> WorkerManager:
        if self._client is None:
            raise ValueError("Client not started yet")
        return self._client

    def suppress_console(self) -> None:
        """Silence console output in all worker processes."""
        if self._client is not None:
            self._client.suppress_console()

    def start(self) -> None:
        n = self._n if self._n is not None else 2
        self._client = WorkerManager(n_workers=n)
        self._client.start()

        if self._global_config.dtop:
            from dimos.core.resource_monitor.monitor import StatsMonitor

            self._stats_monitor = StatsMonitor(self._client)
            self._stats_monitor.start()

    def stop(self) -> None:
        if self._repl_server is not None:
            self._repl_server.stop()
            self._repl_server = None

        if self._stats_monitor is not None:
            self._stats_monitor.stop()
            self._stats_monitor = None

        for module_class, module in reversed(self._deployed_modules.items()):
            logger.info("Stopping module...", module=module_class.__name__)
            try:
                module.stop()
            except Exception:
                logger.error("Error stopping module", module=module_class.__name__, exc_info=True)
            logger.info("Module stopped.", module=module_class.__name__)

        self._client.close_all()  # type: ignore[union-attr]

    def deploy(
        self,
        module_class: type[ModuleBase[Any]],
        global_config: GlobalConfig = global_config,
        **kwargs: Any,
    ) -> ModuleProxy:
        module = self.client.deploy(module_class, global_config, kwargs)
        self._deployed_modules[module_class] = module  # type: ignore[assignment]
        return module  # type: ignore[return-value]

    def deploy_parallel(self, module_specs: list[ModuleSpec]) -> list[ModuleProxy]:
        modules = self.client.deploy_parallel(module_specs)
        for (module_class, _, _), module in zip(module_specs, modules, strict=True):
            self._deployed_modules[module_class] = module  # type: ignore[assignment]
        return modules  # type: ignore[return-value]

    def start_all_modules(self) -> None:
        modules = list(self._deployed_modules.values())
        if isinstance(self._client, WorkerManager):
            with ThreadPoolExecutor(max_workers=len(modules)) as executor:
                list(executor.map(lambda m: m.start(), modules))
        else:
            for module in modules:
                module.start()

        module_list = list(self._deployed_modules.values())
        for module in modules:
            if hasattr(module, "on_system_modules"):
                module.on_system_modules(module_list)

    def get_instance(self, module: type[ModuleBase]) -> ModuleProxy:
        return self._deployed_modules.get(module)  # type: ignore[return-value, no-any-return]

    def list_modules(self) -> list[str]:
        """Return the class names of all deployed modules."""
        return [cls.__name__ for cls in self._deployed_modules]

    def get_module(self, name: str) -> ModuleProxy:
        """Look up a deployed module by class name."""
        for cls, proxy in self._deployed_modules.items():
            if cls.__name__ == name:
                return proxy
        raise KeyError(f"Module '{name}' not found. Available: {self.list_modules()}")

    def get_module_location(self, name: str) -> tuple[str, int] | None:
        """Return ``(host, port)`` of the worker RPyC server hosting *name*."""
        return self._module_locations.get(name)

    def start_repl_server(self, port: int | None = None) -> None:
        """Start RPyC REPL servers in every worker and the main coordinator."""
        port = port if port is not None else DEFAULT_REPL_PORT

        # Start an RPyC server inside each worker process.
        for worker in self.client.workers:
            worker_port = worker.start_repl_server()
            if worker_port is None:
                logger.error(
                    "Worker failed to start REPL server, skipping...", worker_id=worker.worker_id
                )
                continue
            for module_name in worker.module_names:
                self._module_locations[module_name] = ("localhost", worker_port)

        self._repl_server = ReplServer(self, port=port)
        self._repl_server.start()

    def loop(self) -> None:
        stop = threading.Event()
        try:
            stop.wait()
        except KeyboardInterrupt:
            return
        finally:
            self.stop()
