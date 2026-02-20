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

"""Test that blueprint modules can be imported without hardware-specific dependencies.

This is NOT an integration test — it runs in every CI pass.  It catches the class
of bug where a module-level ``import pyrealsense2`` (or similar) prevents loading
*any* blueprint defined in the same file, even simulation blueprints that never
touch the hardware.

The fix for each failure is the same: lazy-load the hardware dep via
``TYPE_CHECKING`` + local imports inside the methods that actually need it.
"""

import importlib

import pytest

from dimos.robot.all_blueprints import all_blueprints

# Blueprint *modules* that still eagerly import a missing optional dep.
# Goal: shrink this dict to empty.  Each entry is module_path -> reason.
KNOWN_IMPORT_FAILURES: dict[str, str] = {
    "dimos.perception.demo_object_scene_registration": "zed camera eager import",
}


def _unique_modules() -> list[tuple[str, str]]:
    """Deduplicate blueprint entries by module path."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for name, full_path in sorted(all_blueprints.items()):
        mod_path = full_path.split(":")[0]
        if mod_path not in seen:
            seen.add(mod_path)
            out.append((name, mod_path))
    return out


@pytest.mark.parametrize(
    ("blueprint_name", "module_path"),
    _unique_modules(),
    ids=[m for _, m in _unique_modules()],
)
def test_blueprint_module_importable(blueprint_name: str, module_path: str) -> None:
    """Every blueprint module must import without hardware-only dependencies.

    If this fails, the module has a top-level ``import <hardware_lib>`` that
    should be deferred to runtime.  See the realsense lazy-loading pattern:
    ``from __future__ import annotations`` + ``TYPE_CHECKING`` guard +
    local ``import`` inside methods that use the dep.
    """
    if module_path in KNOWN_IMPORT_FAILURES:
        pytest.skip(KNOWN_IMPORT_FAILURES[module_path])

    try:
        importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise AssertionError(
            f"Blueprint module '{module_path}' (blueprint '{blueprint_name}') "
            f"eagerly imports '{e.name}' which is not installed.\n"
            f"Use lazy imports (TYPE_CHECKING + local imports) for hardware-specific deps."
        ) from e
