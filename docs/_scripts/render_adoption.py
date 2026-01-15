from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SoftwareEntry:
    """A single “software using DerivKit” entry for the docs adoption list."""

    name: str
    description: str
    repo: str


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML file and return its top-level mapping.

    This is a small helper for reading adoption entry files. We expect each YAML
    file to be a single mapping (a dict) that contains the fields we need to
    render the docs.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at top level.")
    return data


def _require_str(data: dict[str, Any], key: str, *, path: Path) -> str:
    """Return a required string field from a YAML mapping.

    This enforces that a given key exists and is a non-empty string, so the docs
    renderer can rely on well-formed data and fail with a clear error message.
    """
    val = data.get(key)
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"{path}: missing/invalid {key!r} (must be a non-empty string).")
    return val.strip()


def _iter_entry_files(root: Path) -> list[Path]:
    """List adoption entry YAML files in a directory.

    Files whose names start with an underscore are treated as templates or
    placeholders and are ignored.
    """
    return sorted(p for p in root.glob("*.yml") if not p.name.startswith("_"))


def load_software_entries(root: Path) -> list[SoftwareEntry]:
    """Load all software adoption entries from a directory of YAML files.

    Each YAML file is expected to define one software entry with the required
    fields (name, description, repo). The returned list is used to render the
    “Software using DerivKit” section in the docs.
    """
    entries: list[SoftwareEntry] = []
    for path in _iter_entry_files(root):
        data = _load_yaml(path)
        entries.append(
            SoftwareEntry(
                name=_require_str(data, "name", path=path),
                description=_require_str(data, "description", path=path),
                repo=_require_str(data, "repo", path=path),
            )
        )
    return entries


def render_software(entries: list[SoftwareEntry]) -> str:
    """Render software adoption entries as reStructuredText.

    The output is an RST snippet that can be included directly in a docs page.
    If there are no entries, a short placeholder message is returned.
    """
    if not entries:
        return "*No external software listed yet.*\n"

    lines: list[str] = []
    for e in entries:
        lines += [
            f"- **{e.name}**",
            "",
        ]
        for para in e.description.splitlines():
            if para.strip():
                lines.append(f"  {para.rstrip()}")
            else:
                lines.append("")
        lines += [
            f"  Repository: {e.repo}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def render_publications_stub() -> str:
    """Return the placeholder text for the publications section.

    We keep publications as a simple stub for now and can extend the script later
    to render publication YAML entries in the same way as software entries.
    """
    return "*No publications listed yet.*\n"


def main() -> None:
    """Generate the adoption RST snippets used by the docs.

    This script reads adoption entries from ``docs/adoption/`` and writes the
    rendered RST fragments into ``docs/_generated/``. The Sphinx build includes
    those fragments in the Citation/Adoption page, so new merged entries appear
    automatically in the rendered documentation.
    """
    docs = Path(__file__).resolve().parents[1]
    outdir = docs / "_generated"
    outdir.mkdir(parents=True, exist_ok=True)

    software_dir = docs / "adoption" / "software"
    pubs_dir = docs / "adoption" / "publications"
    software_dir.mkdir(parents=True, exist_ok=True)
    pubs_dir.mkdir(parents=True, exist_ok=True)

    software = load_software_entries(software_dir)
    (outdir / "adoption_software.rst").write_text(render_software(software), encoding="utf-8")

    (outdir / "adoption_publications.rst").write_text(
        render_publications_stub(),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
