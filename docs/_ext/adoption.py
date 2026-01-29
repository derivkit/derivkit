from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docutils import nodes
import yaml

from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective, SphinxRole
from sphinx.util.typing import ExtensionMetadata


@dataclass(frozen=True)
class Entry:
    """A single adoption entry for the docs adoption list."""

    name: str
    description: str
    link: str
    citation: str


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


def _iter_entry_files(path: Path) -> list[Path]:
    """List adoption entry YAML files in a directory.

    Files whose names start with an underscore are treated as templates or
    placeholders and are ignored.
    """
    return sorted(p for p in path.glob("*.yml"))


def _require_str(data: dict[str, Any], key: str, *, path: Path) -> str:
    """Return a required string field from a YAML mapping.

    This enforces that a given key exists and is a non-empty string, so the docs
    renderer can rely on well-formed data and fail with a clear error message.
    """
    val = data.get(key)
    if not isinstance(val, str) or not val.strip():
        raise ValueError(
            f"{path}: missing/invalid {key!r} (must be a non-empty string)."
        )
    return val.strip()


def load_adoption_entries(path: Path) -> list[Entry]:
    """Load all adoption entries from a directory of YAML files.

    Each YAML file is expected to define one entry with the required
    fields (name, description, link).
    """
    entries: list[Entry] = []
    for entry in _iter_entry_files(path):
        data = _load_yaml(entry)
        entries.append(
            Entry(
                name=_require_str(data, "name", path=entry),
                description=_require_str(data, "description", path=entry),
                link=_require_str(data, "link", path=entry),
                citation=_require_str(data, "citation", path=entry),
            )
        )
    return entries


class AdoptionDirective(SphinxDirective):
    """A directive to add adoptions."""

    required_arguments = 1

    def run(self) -> list[nodes.Node]:
        docs_dir = Path(__file__).resolve().parents[1]
        adoption_dir = docs_dir / "adoption"

        match self.arguments[0]:
            case "publications":
                adoption_type = "publications"
            case "software":
                adoption_type = "software"
            case _:
                raise ValueError("Unexpected adoption type.")

        adoption_entries = load_adoption_entries(adoption_dir / adoption_type)

        node_list = []
        for entry in adoption_entries:
                entry_section = nodes.section(ids=[entry.name])
                entry_section += nodes.title(text=entry.name)
                entry_section += nodes.paragraph(text=entry.description)
                if entry.citation is not None:
                    entry_section += nodes.paragraph(text=entry.description)
                if entry.link is not None:
                    entry_section += nodes.paragraph(
                        "",
                        "",
                        nodes.reference("", "Project website", refuri=entry.link),
                    )

                node_list += entry_section

        if len(node_list) == 0:
            node_list.append(nodes.paragraph(text=f"No {adoption_type} adoptions yet."))

        return node_list


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_directive("adoption", AdoptionDirective)

    return {
        "version": "0.2",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
