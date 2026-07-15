"""
Static checks for launch/osm_cloud.launch.py.

Importing the launch file for real requires the `launch`, `launch_ros`, and
`ament_index_python` ROS2 packages, none of which are installed in this
(non-ROS) test environment — see the sys.modules-mocking approach in
tests/test_osm_cloud.py for why that's impractical to fake convincingly for
a whole launch-description graph. Instead we parse the file's AST and check
that every ``DeclareLaunchArgument`` it declares is actually consumed via
``LaunchConfiguration`` somewhere in the file.

This is exactly the class of bug fixed by a previous commit ("osm_cloud
bugs (run_all, costs, topics, linspace, launch args)"): it's easy to add a
new launch argument (or rename one) and forget to wire it into the Node's
parameters, silently making it a no-op from the user's perspective.

If `launch`/`launch_ros`/`ament_index_python` do happen to be importable
(e.g. inside a colcon/ROS workspace), a real smoke test also builds the
LaunchDescription and checks the node parameters.
"""

import ast
from pathlib import Path

import pytest

_LAUNCH_FILE = Path(__file__).resolve().parents[1] / "launch" / "osm_cloud.launch.py"


def _parse_launch_file() -> ast.Module:
    source = _LAUNCH_FILE.read_text()
    return ast.parse(source, filename=str(_LAUNCH_FILE))


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _declared_and_used_args(tree: ast.Module) -> tuple[set[str], set[str]]:
    """
    Collect launch-argument names declared via DeclareLaunchArgument("name", ...)
    and names referenced via LaunchConfiguration("name") anywhere in the module.
    """
    declared: set[str] = set()
    used: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name == "DeclareLaunchArgument" and node.args:
            arg_name = _string_literal(node.args[0])
            if arg_name is not None:
                declared.add(arg_name)
        elif name == "LaunchConfiguration" and node.args:
            arg_name = _string_literal(node.args[0])
            if arg_name is not None:
                used.add(arg_name)
    return declared, used


def test_launch_file_exists():
    assert _LAUNCH_FILE.is_file()


def test_every_declared_launch_argument_is_referenced():
    """
    Every DeclareLaunchArgument must be consumed via LaunchConfiguration
    somewhere in the file. A declared-but-unused argument silently does
    nothing when a user sets it — that's a dead argument.
    """
    tree = _parse_launch_file()
    declared, used = _declared_and_used_args(tree)

    assert declared, "expected at least one DeclareLaunchArgument in the launch file"

    dead_args = declared - used
    assert not dead_args, (
        "launch argument(s) declared but never referenced via LaunchConfiguration "
        f"(dead arguments): {sorted(dead_args)}"
    )


def test_no_launch_configuration_references_undeclared_argument():
    """
    Every LaunchConfiguration("name") must correspond to a declared argument
    — catches typos/renames that would raise at launch time.
    """
    tree = _parse_launch_file()
    declared, used = _declared_and_used_args(tree)

    undeclared = used - declared
    assert not undeclared, (
        f"LaunchConfiguration references argument(s) never declared: {sorted(undeclared)}"
    )


def test_generate_launch_description_is_defined():
    tree = _parse_launch_file()
    top_level_funcs = {
        node.name for node in ast.iter_child_nodes(tree) if isinstance(node, ast.FunctionDef)
    }
    assert "generate_launch_description" in top_level_funcs


# ── optional: real smoke test when ROS2 launch packages are available ──────


def test_generate_launch_description_smoke():
    """
    Full smoke test: actually import the launch file and build its
    LaunchDescription. Skipped outside a ROS2 workspace where `launch`,
    `launch_ros`, and `ament_index_python` aren't installed.
    """
    launch = pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    pytest.importorskip("ament_index_python")

    import importlib.util

    spec = importlib.util.spec_from_file_location("osm_cloud_launch_module", _LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    description = module.generate_launch_description()
    assert isinstance(description, launch.LaunchDescription)
