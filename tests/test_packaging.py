"""The version is spelled out in four files; none of them may drift."""

import re
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
