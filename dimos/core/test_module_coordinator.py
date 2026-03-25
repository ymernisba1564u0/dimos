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

# type: ignore
from __future__ import annotations

import pytest

from dimos.core.module_coordinator import ModuleCoordinator


class _SensorModule:
    pass


class _MotorModule:
    pass


def test_health_check_fails_with_no_workers():
    coord = ModuleCoordinator()
    assert coord.health_check() is False


def test_health_check_fails_when_worker_died(mocker):
    coord = ModuleCoordinator()
    dead_worker = mocker.MagicMock(pid=None, worker_id=1)
    coord._client = mocker.MagicMock(workers=[dead_worker])
    assert coord.health_check() is False


def test_health_check_passes_when_all_alive(mocker):
    coord = ModuleCoordinator()
    coord._client = mocker.MagicMock(
        workers=[mocker.MagicMock(pid=100, worker_id=1), mocker.MagicMock(pid=101, worker_id=2)]
    )
    assert coord.health_check() is True


def test_list_modules():
    coord = ModuleCoordinator()
    coord._deployed_modules = {_SensorModule: object(), _MotorModule: object()}
    assert set(coord.list_modules()) == {"_SensorModule", "_MotorModule"}


def test_get_module_by_name(mocker):
    coord = ModuleCoordinator()
    proxy = mocker.MagicMock()
    coord._deployed_modules = {_SensorModule: proxy}
    assert coord.get_module("_SensorModule") is proxy


def test_get_module_unknown_raises():
    coord = ModuleCoordinator()
    coord._deployed_modules = {_SensorModule: object()}
    with pytest.raises(KeyError, match="NoSuch"):
        coord.get_module("NoSuch")


def test_get_module_location():
    coord = ModuleCoordinator()
    coord._module_locations = {"Sensor": ("localhost", 5000)}
    assert coord.get_module_location("Sensor") == ("localhost", 5000)
    assert coord.get_module_location("Unknown") is None


def test_stop_calls_stop_on_all_modules(mocker):
    coord = ModuleCoordinator()
    proxy_a = mocker.MagicMock()
    proxy_b = mocker.MagicMock()
    coord._deployed_modules = {_SensorModule: proxy_a, _MotorModule: proxy_b}
    coord._client = mocker.MagicMock()

    coord.stop()

    proxy_a.stop.assert_called_once()
    proxy_b.stop.assert_called_once()
    coord._client.close_all.assert_called_once()


def test_stop_resilient_to_module_error(mocker):
    """A module raising during stop() must not prevent other modules from stopping."""
    coord = ModuleCoordinator()
    proxy_a = mocker.MagicMock()
    proxy_a.stop.side_effect = RuntimeError("boom")
    proxy_b = mocker.MagicMock()
    coord._deployed_modules = {_SensorModule: proxy_a, _MotorModule: proxy_b}
    coord._client = mocker.MagicMock()

    coord.stop()

    proxy_b.stop.assert_called_once()


def test_start_repl_server_populates_locations(mocker):
    coord = ModuleCoordinator()
    worker = mocker.MagicMock()
    worker.start_repl_server.return_value = 9999
    worker.module_names = ["Sensor", "Motor"]
    coord._client = mocker.MagicMock(workers=[worker])
    mocker.patch("dimos.core.module_coordinator.ReplServer")

    coord.start_repl_server(port=12345)

    assert coord.get_module_location("Sensor") == ("localhost", 9999)
    assert coord.get_module_location("Motor") == ("localhost", 9999)


def test_start_repl_server_skips_failed_worker(mocker):
    coord = ModuleCoordinator()
    worker = mocker.MagicMock()
    worker.start_repl_server.return_value = None
    worker.module_names = ["Sensor"]
    coord._client = mocker.MagicMock(workers=[worker])
    mocker.patch("dimos.core.module_coordinator.ReplServer")

    coord.start_repl_server()

    assert coord.get_module_location("Sensor") is None
