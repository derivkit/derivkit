"""Script to generate adoption.rst from adoption_from_issue.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
ADOPTION_DIR = DOCS_DIR / "adoption"
OUT_DIR = DOCS_DIR / "_generated"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at top level.")
    return data


def _req_str(d: dict[str, Any], key: str, *, path: Path) -> str:
    v = d.get(key, "")
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{path}: missing/empty required field '{key}'")
    return v.strip()


def _opt_str(d: dict[str, Any], key: str) -> str | None:
    v = d.get(key)
    if not isinstance(v, str):
        return None
    v = v.strip()
    return v if v else None


def _rst_block(text: str, indent: int = 2) -> str:
    pad = " " * indent
    lines = [pad + ln for ln in text.strip().splitlines()]
    return "\n".join(lines)


def _render_software_entry(d: dict[str, Any], *, path: Path) -> tuple[str, str]:
    name = _req_str(d, "name", path=path)
    repo = _req_str(d, "repo", path=path)
    desc = _req_str(d, "description", path=path)
    rendered = (
        f"- `{name} <{repo}>`_\n\n"
        f"{_rst_block(desc)}\n"
    )
    return (name.casefold(), rendered)


def _render_publication_entry(d: dict[str, Any], *, path: Path) -> tuple[str, str]:
    name = _req_str(d, "name", path=path)
    desc = _req_str(d, "description", path=path)
    link = _opt_str(d, "link") or _opt_str(d, "source_issue")
    if link is None:
        # Shouldn’t happen because adoption_from_issue always writes source_issue,
        # but keep it robust.
        link = ""
    citation = _opt_str(d, "citation")

    head = f"- `{name} <{link}>`_\n\n" if link else f"- {name}\n\n"
    rendered = head + f"{_rst_block(desc)}\n"
    if citation:
        rendered += "\n" + _rst_block("**Citation:**") + "\n\n" + _rst_block(citation) + "\n"
    return (name.casefold(), rendered)


def _write(out_path: Path, title: str, entries: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = title + "\n" + ("-" * len(title)) + "\n\n"
    if entries:
        text += "\n".join(entries).rstrip() + "\n"
    else:
        text += "No entries yet.\n"
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    software_paths = sorted((ADOPTION_DIR / "software").glob("*.yml")) + sorted((ADOPTION_DIR / "software").glob("*.yaml"))
    pub_paths = sorted((ADOPTION_DIR / "publications").glob("*.yml")) + sorted((ADOPTION_DIR / "publications").glob("*.yaml"))

    software_entries: list[tuple[str, str]] = []
    for p in software_paths:
        d = _load_yaml(p)
        if d.get("kind") != "software":
            continue
        software_entries.append(_render_software_entry(d, path=p))
    software_entries.sort(key=lambda t: t[0])

    pub_entries: list[tuple[str, str]] = []
    for p in pub_paths:
        d = _load_yaml(p)
        if d.get("kind") != "publication":
            continue
        pub_entries.append(_render_publication_entry(d, path=p))
    pub_entries.sort(key=lambda t: t[0])

    _write(OUT_DIR / "adoption_software.rst", "Software", [r for _, r in software_entries])
    _write(OUT_DIR / "adoption_publications.rst", "Publications", [r for _, r in pub_entries])

    print(f"Wrote {OUT_DIR / 'adoption_software.rst'} and {OUT_DIR / 'adoption_publications.rst'}")


if __name__ == "__main__":
    main()
