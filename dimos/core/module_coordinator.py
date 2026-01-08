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

import time
from typing import TypeVar

from dimos import core
from dimos.core import DimosCluster, Module
from dimos.core.global_config import GlobalConfig
from dimos.core.resource import Resource

T = TypeVar("T", bound="Module")


class ModuleCoordinator(Resource):
    """Orchestrate distributed module lifecycle on a Dask cluster.

    ModuleCoordinator manages cluster startup, module deployment as distributed
    actors, and graceful shutdown. It maintains a type-indexed registry where
    each module class maps to at most one deployed instance.

    Most users should use `blueprint.build()` instead, which handles cluster
    startup, module deployment, and inter-module connections automatically.
    Direct use of ModuleCoordinator is for custom configuration or manual
    lifecycle control.

    Examples:
        Recommended approach using blueprints:

        >>> from dimos.core.blueprints import autoconnect  # doctest: +SKIP
        >>> blueprint = autoconnect(SomeModule.blueprint(), OtherModule.blueprint())  # doctest: +SKIP
        >>> coordinator = blueprint.build()  # doctest: +SKIP
        >>> coordinator.loop()  # doctest: +SKIP

        Direct instantiation for custom control:

        >>> coordinator = ModuleCoordinator(n=4, memory_limit="4GB")  # doctest: +SKIP
        >>> coordinator.start()  # doctest: +SKIP
        >>> module = coordinator.deploy(MyModule)  # doctest: +SKIP
        >>> coordinator.start_all_modules()  # doctest: +SKIP
        >>> coordinator.loop()  # doctest: +SKIP
    """

    _client: DimosCluster | None = None
    _n: int | None = None
    _memory_limit: str = "auto"
    _deployed_modules: dict[type[Module], Module] = {}

    def __init__(
        self,
        n: int | None = None,
        memory_limit: str = "auto",
        global_config: GlobalConfig | None = None,
    ) -> None:
        """Initialize coordinator with cluster configuration.

        Args:
            n: Number of Dask worker processes. Falls back to
                `global_config.n_dask_workers` if None.
            memory_limit: Memory limit per worker (e.g., "4GB", "500MB", "auto").
            global_config: System-wide settings. If None, uses defaults.
        """
        cfg = global_config or GlobalConfig()
        self._n = n if n is not None else cfg.n_dask_workers
        self._memory_limit = memory_limit

    def start(self) -> None:
        """Low-level API: Start the underlying Dask cluster.

        Spawns worker processes and initializes the distributed actor system.
        After this returns, modules can be deployed via `deploy()`.

        Note:
            Calling `start()` on an already-started coordinator overwrites the
            existing cluster reference. Use `stop()` first if restarting.
        """
        self._client = core.start(self._n, self._memory_limit)

    def stop(self) -> None:
        """Shut down all modules and the cluster.

        Stops modules in reverse deployment order, then closes the Dask cluster.

        Note:
            Raises `AttributeError` if called before `start()`. For automatic
            lifecycle management, use `loop()` instead.
        """
        for module in reversed(self._deployed_modules.values()):
            module.stop()

        self._client.close_all()  # type: ignore[union-attr]

    def deploy(self, module_class: type[T], *args, **kwargs) -> T:  # type: ignore[no-untyped-def]
        """Low-level API: Deploy a module class as a distributed actor.

        Creates an instance of `module_class` on a Dask worker and returns an
        RPC proxy for remote method calls. The coordinator tracks one instance
        per module class; deploying the same class again overwrites the registry
        entry (the original actor remains in the cluster but becomes untracked).

        Args:
            module_class: The Module subclass to deploy.
            *args: Positional arguments for `module_class.__init__`.
            **kwargs: Keyword arguments for `module_class.__init__`.

        Returns:
            RPCClient proxy to the deployed actor.

        Raises:
            ValueError: If `start()` has not been called.
        """
        if not self._client:
            raise ValueError("Not started")

        module = self._client.deploy(module_class, *args, **kwargs)  # type: ignore[attr-defined]
        self._deployed_modules[module_class] = module
        return module  # type: ignore[no-any-return]

    def start_all_modules(self) -> None:
        """Low-level API: Call `start()` on all deployed modules.

        Initializes each module's RPC server and event loop in deployment order.
        After this completes, modules are ready to process messages.

        If a module's `start()` raises, the exception propagates immediately.
        Modules started before the failure remain running; no rollback occurs.
        """
        for module in self._deployed_modules.values():
            module.start()

    def get_instance(self, module: type[T]) -> T | None:
        """Retrieve a deployed module by its class.

        Args:
            module: The module class to look up.

        Returns:
            The RPCClient proxy if deployed, or None if not found.
        """
        return self._deployed_modules.get(module)  # type: ignore[return-value]

    def loop(self) -> None:
        """Block until interrupted, then shut down gracefully.

        Sleeps indefinitely until Ctrl+C (SIGINT), then calls `stop()` to clean
        up all modules and the cluster. This is the standard way to run a
        long-lived DimOS application.

        Examples:
            >>> coordinator = blueprint.build()  # doctest: +SKIP
            >>> coordinator.loop()  # doctest: +SKIP
        """
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            return
        finally:
            self.stop()
