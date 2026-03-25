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

import pytest
import rpyc
from rpyc.utils.server import ThreadedServer

from dimos.core.repl_server import _THREAD_NAME, ReplServer, start_worker_repl_server


@pytest.fixture
def coordinator_server(find_free_port, wait_until_rpyc_connectable, make_stub_coordinator):
    port = find_free_port()
    coord = make_stub_coordinator(
        modules=["Sensor", "Motor"],
        locations={"Sensor": ("10.0.0.1", 5000)},
    )
    server = ReplServer(coord, port=port, host="127.0.0.1")
    server.start()
    wait_until_rpyc_connectable("127.0.0.1", port)
    yield port
    thread = server._thread
    server.stop()
    if thread is not None:
        thread.join(timeout=2.0)


@pytest.fixture
def coordinator_conn(coordinator_server):
    conn = rpyc.connect("127.0.0.1", coordinator_server, config={"sync_request_timeout": 5})
    yield conn
    conn.close()


@pytest.fixture
def worker_server(mocker, wait_until_rpyc_connectable):
    # Keep track of all the servers so we can close them.
    servers = []
    real_init = ThreadedServer.__init__

    def tracking_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        servers.append(self)

    mocker.patch.object(ThreadedServer, "__init__", tracking_init)

    class ModuleA:
        value = 42

    class ModuleB:
        pass

    instances = {1: ModuleA(), 2: ModuleB()}
    port = start_worker_repl_server(instances, host="127.0.0.1")
    wait_until_rpyc_connectable("127.0.0.1", port)

    yield port

    for srv in servers:
        srv.close()

    for t in threading.enumerate():
        if t.name == _THREAD_NAME:
            t.join(timeout=2.0)
            break


@pytest.fixture
def worker_conn(worker_server):
    conn = rpyc.connect("127.0.0.1", worker_server, config={"sync_request_timeout": 5})
    yield conn
    conn.close()


def test_list_modules(coordinator_conn):
    assert set(coordinator_conn.root.list_modules()) == {"Sensor", "Motor"}


def test_get_module_location_known(coordinator_conn):
    loc = coordinator_conn.root.get_module_location("Sensor")
    assert (str(loc[0]), int(loc[1])) == ("10.0.0.1", 5000)


def test_get_module_location_unknown(coordinator_conn):
    assert coordinator_conn.root.get_module_location("NoSuch") is None


def test_stop_prevents_new_connections(
    find_free_port, wait_until_rpyc_connectable, make_stub_coordinator
):
    port = find_free_port()
    server = ReplServer(make_stub_coordinator(), port=port, host="127.0.0.1")
    server.start()
    wait_until_rpyc_connectable("127.0.0.1", port)
    thread = server._thread
    server.stop()
    if thread is not None:
        thread.join(timeout=2.0)

    with pytest.raises(ConnectionRefusedError):
        rpyc.connect("127.0.0.1", port)


def test_worker_get_instance_by_name(worker_conn):
    result = worker_conn.root.get_instance_by_name("ModuleA")
    assert result.value == 42


def test_worker_get_instance_unknown_raises(worker_conn):
    with pytest.raises(KeyError, match="NoSuchModule"):
        worker_conn.root.get_instance_by_name("NoSuchModule")


def test_worker_list_instances(worker_conn):
    mapping = worker_conn.root.list_instances()
    assert str(mapping[1]) == "ModuleA"
    assert str(mapping[2]) == "ModuleB"
