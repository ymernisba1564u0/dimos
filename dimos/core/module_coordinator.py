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

import threading
from typing import TYPE_CHECKING, Any, TypeAlias

from dimos.core.global_config import GlobalConfig, global_config
from dimos.core.module import ModuleBase, ModuleSpec
from dimos.core.resource import Resource
from dimos.core.worker_manager import WorkerManager
from dimos.core.worker_manager_docker import WorkerManagerDocker
from dimos.utils.logging_config import setup_logger
from dimos.utils.safe_thread_map import safe_thread_map

if TYPE_CHECKING:
    from dimos.core.blueprints import DeploySpec, ModuleRefWiring, RpcWiringPlan, StreamWiring
    from dimos.core.rpc_client import ModuleProxy, ModuleProxyProtocol

logger = setup_logger()

DeploymentManager: TypeAlias = WorkerManagerDocker | WorkerManager


class ModuleCoordinator(Resource):  # type: ignore[misc]
    """
    There should only ever be one module coordinator instance (this is a singleton)
    - Module (classes) should be able to be deployed, stopped, and re-deployed in on one instance of ModuleCoordinator
    - Arguably ModuleCoordinator could be called the "DimosRuntime"
    - ModuleCoordinator is responsible for all global "addresses".
      Ex: it should make sure all modules are using the same LCM url, the same rerun port, etc
      (it may not do all of that at time of writing but that is the intention/job of this class)
    - Modules shouldn't be deployed on their own (except for testing)
    """

    _managers: dict[str, DeploymentManager]
    _global_config: GlobalConfig
    _deploy_spec: DeploySpec | None
    _deployed_modules: dict[type[ModuleBase], ModuleProxyProtocol]

    def __init__(
        self,
        g: GlobalConfig = global_config,
        deploy_spec: DeploySpec | None = None,
    ) -> None:
        self._global_config = g
        self._deploy_spec = deploy_spec
        manager_types: list[type[DeploymentManager]] = [WorkerManagerDocker, WorkerManager]
        self._managers: dict[str, DeploymentManager] = {
            cls.deployment_identifier: cls(g=g) for cls in manager_types
        }
        self._deployed_modules = {}

    def start(self) -> None:
        from dimos.core.o3dpickle import register_picklers

        register_picklers()
        for m in self._managers.values():
            m.start()

        if self._deploy_spec is not None:
            spec = self._deploy_spec
            self.deploy_parallel(spec.module_specs)
            self._wire_streams(spec.stream_wiring)
            self._wire_rpc_methods(spec.rpc_wiring)
            self._wire_module_refs(spec.module_ref_wiring)
            self._build_all_modules()
            self.start_all_modules()

    def health_check(self) -> bool:
        return all(m.health_check() for m in self._managers.values())

    @property
    def n_modules(self) -> int:
        return len(self._deployed_modules)

    def suppress_console(self) -> None:
        for m in self._managers.values():
            m.suppress_console()

    def stop(self) -> None:
        for module_class, module in reversed(self._deployed_modules.items()):
            logger.info("Stopping module...", module=module_class.__name__)
            try:
                module.stop()
            except Exception:
                logger.error("Error stopping module", module=module_class.__name__, exc_info=True)
            logger.info("Module stopped.", module=module_class.__name__)

        def _stop_manager(m: DeploymentManager) -> None:
            try:
                m.stop()
            except Exception:
                logger.error("Error stopping manager", manager=type(m).__name__, exc_info=True)

        safe_thread_map(tuple(self._managers.values()), _stop_manager)

    def deploy(
        self,
        module_class: type[ModuleBase[Any]],
        global_config: GlobalConfig = global_config,
        **kwargs: Any,
    ) -> ModuleProxy:
        if not self._managers:
            raise ValueError("Trying to dimos.deploy before the client has started")

        deployed_module = self._managers[module_class.deployment].deploy(
            module_class, global_config, kwargs
        )
        self._deployed_modules[module_class] = deployed_module  # type: ignore[assignment]
        return deployed_module  # type: ignore[return-value]

    def deploy_parallel(self, module_specs: list[ModuleSpec]) -> list[ModuleProxy]:
        if not self._managers:
            raise ValueError("Not started")

        # Group specs by deployment type, tracking original indices for reassembly
        indices_by_deployment: dict[str, list[int]] = {}
        specs_by_deployment: dict[str, list[ModuleSpec]] = {}
        for index, spec in enumerate(module_specs):
            dep = spec[0].deployment
            indices_by_deployment.setdefault(dep, []).append(index)
            specs_by_deployment.setdefault(dep, []).append(spec)

        results: list[Any] = [None] * len(module_specs)

        def _deploy_group(dep: str) -> None:
            deployed = self._managers[dep].deploy_parallel(specs_by_deployment[dep])
            for index, module in zip(indices_by_deployment[dep], deployed, strict=True):
                results[index] = module

        try:
            safe_thread_map(list(specs_by_deployment.keys()), _deploy_group)
        except:
            self.stop()
            raise

        self._deployed_modules.update(
            {
                cls: mod
                for (cls, _, _), mod in zip(module_specs, results, strict=True)
                if mod is not None
            }
        )
        return results

    def _wire_streams(self, wiring: list[StreamWiring]) -> None:
        """Apply stream transports to deployed modules."""
        for w in wiring:
            instance = self.get_instance(w.module_class)
            instance.set_transport(w.stream_name, w.transport)  # type: ignore[union-attr]

    def _wire_rpc_methods(self, plan: RpcWiringPlan) -> None:
        """Wire RPC methods between modules using the compiled plan."""
        # Build callable registry from deployed instances
        callables: dict[str, Any] = {}
        for rpc_key, (module_class, method_name) in plan.registry.items():
            proxy = self.get_instance(module_class)
            callables[rpc_key] = getattr(proxy, method_name)

        # Apply set_ methods
        for module_class, set_method, linked_key in plan.set_methods:
            if linked_key in callables:
                instance = self.get_instance(module_class)
                getattr(instance, set_method)(callables[linked_key])

        # Apply rpc_call bindings
        for module_class, requested_name, rpc_key in plan.rpc_call_bindings:
            if rpc_key in callables:
                instance = self.get_instance(module_class)
                instance.set_rpc_method(requested_name, callables[rpc_key])  # type: ignore[union-attr]

    def _wire_module_refs(self, wiring: list[ModuleRefWiring]) -> None:
        """Set module references between deployed modules."""
        for w in wiring:
            base_proxy = self.get_instance(w.base_module)
            target_proxy = self.get_instance(w.target_module)
            setattr(base_proxy, w.ref_name, target_proxy)
            base_proxy.set_module_ref(w.ref_name, target_proxy)  # type: ignore[union-attr]

    def _build_all_modules(self) -> None:
        """Call build() on all deployed modules in parallel.

        build() handles heavy one-time work (docker builds, LFS downloads, etc.)
        with a very long timeout. Must be called after deploy and stream wiring
        but before start_all_modules().
        """
        modules = list(self._deployed_modules.values())
        if not modules:
            raise ValueError("No modules deployed. Call deploy() before build_all_modules().")

        try:
            safe_thread_map(modules, lambda m: m.build())
        except:
            self.stop()
            raise

    def start_all_modules(self) -> None:
        modules = list(self._deployed_modules.values())
        if not modules:
            raise ValueError("No modules deployed. Call deploy() before start_all_modules().")

        safe_thread_map(modules, lambda m: m.start())

        for module in modules:
            if hasattr(module, "on_system_modules"):
                module.on_system_modules(modules)

    def get_instance(self, module: type[ModuleBase]) -> ModuleProxy:
        return self._deployed_modules.get(module)  # type: ignore[return-value, no-any-return]

    def loop(self) -> None:
        stop = threading.Event()
        try:
            stop.wait()
        except KeyboardInterrupt:
            return
        finally:
            self.stop()
