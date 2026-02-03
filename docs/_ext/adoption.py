"""Utilities for rendering adoption entries in the docs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
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
    return sorted(
        p for p in path.glob("*.yml")
        if not p.name.startswith("_")
    )


def _require_str(data: dict[str, Any], key: str, *, path: Path) -> str:
    """Return a required string field from a YAML mapping.

    This enforces that a given key exists and is a valid string, so the docs
    renderer can rely on well-formed data and fail with a clear error message.
    """
    val = data.get(key)
    if val and not isinstance(val, str):
        raise ValueError(
            f"{path}: invalid {key!r}."
        )
    return val.strip() if val else ""


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
            case "publications" | "software":
                adoption_type = self.arguments[0]
            case _:
                raise ValueError("Unexpected adoption type.")

        adoption_entries = load_adoption_entries(adoption_dir / adoption_type)

        if not adoption_entries:
            return [nodes.paragraph(text=f"No {adoption_type} adoptions yet.")]

        dl = nodes.definition_list(classes=["adoption-list"])

        for entry in adoption_entries:
            item = nodes.definition_list_item()

            # term (the "name") — keep this clean: no ids/targets near it
            term = nodes.term()
            term += nodes.strong(text=entry.name)
            item += term

            definition = nodes.definition()

            # put the anchor *inside* the definition, so themes don't style it as a heading
            target_id = nodes.make_id(entry.name)
            definition += nodes.target(ids=[target_id])

            definition += nodes.paragraph(text=entry.description)

            if entry.citation:
                definition += nodes.paragraph(text=entry.citation)

            if entry.link:
                p = nodes.paragraph()
                p += nodes.reference("", "Project website", refuri=entry.link)
                definition += p

            item += definition
            dl += item

        return [dl]


def setup(app: Sphinx) -> ExtensionMetadata:
    """Set up the extension."""
    app.add_directive("adoption", AdoptionDirective)

    return {
        "version": "0.2",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
