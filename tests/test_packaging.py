"""The version is spelled out in four files; none of them may drift."""

import ast
import re
import sys
import tomllib
from pathlib import Path
from xml.etree import ElementTree

import yaml

PKG = Path(__file__).resolve().parents[1]


def _package_xml_version(path: Path) -> str:
    version = ElementTree.parse(path).getroot().findtext("version")
    assert version is not None, f"{path} declares no <version>"
    return version


def test_the_version_is_the_same_everywhere():
    pyproject = tomllib.loads((PKG / "pyproject.toml").read_text())["project"]["version"]
    citation = yaml.safe_load((PKG / "CITATION.cff").read_text())["version"]
    versions = {
        "pyproject.toml": pyproject,
        "CITATION.cff": str(citation),
        "package.xml": _package_xml_version(PKG / "package.xml"),
        "map_data_interfaces/package.xml": _package_xml_version(
            PKG / "map_data_interfaces" / "package.xml",
        ),
    }
    assert len(set(versions.values())) == 1, f"the version has drifted: {versions}"


def test_the_changelog_documents_the_released_version():
    """A bump without its changelog section is a release nobody can read."""
    version = tomllib.loads((PKG / "pyproject.toml").read_text())["project"]["version"]
    changelog = (PKG / "docs" / "dev" / "changelog.md").read_text()
    section = rf"^## \[{re.escape(version)}\] — \d{{4}}-\d{{2}}-\d{{2}}$"
    assert re.search(section, changelog, re.M), (
        f"docs/dev/changelog.md has no dated section for {version}"
    )


def test_top_level_ros_imports_are_declared():
    """ROS package dependencies must not work only by accident of the system."""
    declared = {
        node.text.strip()
        for node in ElementTree.parse(PKG / "package.xml").getroot().findall("exec_depend")
        if node.text
    }
    imported = set()
    for pattern in ("map_data/**/*.py", "launch/*.py"):
        for path in PKG.glob(pattern):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    imported.add(node.module.split(".")[0])

    import_to_rosdep = {
        "yaml": "python3-yaml",
        "cv2": "python3-opencv",
        "flask": "python3-flask",
        "flask_socketio": "python3-flask-socketio",
        "gpxpy": "python3-gpxpy",
        "numpy": "python3-numpy",
        "overpy": "python3-overpy",
        "shapely": "python3-shapely",
        "utm": "python3-utm",
        "scipy": "python3-scipy",
    }
    required = {import_to_rosdep.get(name, name) for name in imported}
    allowed = set(sys.stdlib_module_names) | {"map_data", "werkzeug"} | declared
    undeclared = required - allowed
    assert not undeclared, f"packages imported but not declared: {sorted(undeclared)}"
