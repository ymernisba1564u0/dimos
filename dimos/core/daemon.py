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

"""Daemonization and health-check support for DimOS processes."""

from __future__ import annotations

import os
import signal
import sys
import time
from typing import TYPE_CHECKING

from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from pathlib import Path

    from dimos.core.module_coordinator import ModuleCoordinator
    from dimos.core.run_registry import RunEntry

logger = setup_logger()

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

_DEFAULT_HEALTH_TIMEOUT: float = 30.0
_POLL_INTERVAL: float = 0.25


def health_check(
    coordinator: ModuleCoordinator,
    timeout: float = _DEFAULT_HEALTH_TIMEOUT,
    poll_interval: float = _POLL_INTERVAL,
) -> bool:
    """Poll coordinator workers until *timeout*, return True if all stay alive.

    The check fails immediately when any worker's ``pid`` becomes ``None``
    (meaning the underlying process exited).  It also fails if there are
    zero workers (nothing to monitor).
    """
    client = coordinator._client
    if client is None:
        logger.error("health_check: coordinator has no WorkerManager")
        return False

    workers = client.workers
    if not workers:
        logger.error("health_check: no workers found")
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for w in workers:
            if w.pid is None:
                logger.error(
                    "health_check: worker died",
                    worker_id=w.worker_id,
                )
                return False
        time.sleep(poll_interval)

    # Final check after timeout elapsed
    for w in workers:
        if w.pid is None:
            logger.error(
                "health_check: worker died at deadline",
                worker_id=w.worker_id,
            )
            return False

    return True


# ---------------------------------------------------------------------------
# Daemonize (double-fork)
# ---------------------------------------------------------------------------


def daemonize(log_dir: Path) -> None:
    """Double-fork daemonize the current process.

    After this call the *caller* is the daemon grandchild.
    stdin/stdout/stderr are redirected to ``/dev/null`` — all real
    logging goes through structlog's FileHandler to ``main.jsonl``.
    The two intermediate parents call ``os._exit(0)``.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # First fork — detach from terminal
    pid = os.fork()
    if pid > 0:
        os._exit(0)

    os.setsid()

    # Second fork — can never reacquire a controlling terminal
    pid = os.fork()
    if pid > 0:
        os._exit(0)

    # Redirect all stdio to /dev/null — structlog FileHandler is the log path
    sys.stdout.flush()
    sys.stderr.flush()

    devnull = open(os.devnull)
    os.dup2(devnull.fileno(), sys.stdin.fileno())
    os.dup2(devnull.fileno(), sys.stdout.fileno())
    os.dup2(devnull.fileno(), sys.stderr.fileno())
    devnull.close()


# ---------------------------------------------------------------------------
# Signal handler for clean shutdown
# ---------------------------------------------------------------------------


def install_signal_handlers(entry: RunEntry, coordinator: ModuleCoordinator) -> None:
    """Install SIGTERM/SIGINT handlers that stop the coordinator and clean the registry."""

    def _shutdown(signum: int, frame: object) -> None:
        logger.info("Received signal, shutting down", signal=signum)
        try:
            coordinator.stop()
        except Exception:
            logger.error("Error during coordinator stop", exc_info=True)
        entry.remove()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
