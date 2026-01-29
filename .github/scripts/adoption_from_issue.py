"""Create adoption YAML entries from GitHub adoption issues.

This script is used by a GitHub Action. It parses the issue form body and writes a
YAML file into ``docs/adoption/`` so the docs can render the adoption list.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdoptionIssue:
    """Structured fields extracted from a DerivKit adoption issue."""
    entry_type: str  # "software" or "publication"
    name: str
    description: str
    repo: str | None
    link: str | None
    citation: str | None
    contact: str | None
    issue_number: int
    issue_url: str


_FIELD_RE = re.compile(r"^###\s+(?P<label>.+?)\s*$")


def _normalize_type(s: str) -> str:
    """Normalize entry type string to lowercase and remove "software" / "publication"."""
    s = s.strip().lower()
    if "software" in s:
        return "software"
    if "publication" in s:
        return "publication"
    raise ValueError(f"Unrecognized entry type: {s!r}")


def _slugify(name: str) -> str:
    """Generate a slug for a given entry name."""
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug or "adoption-entry"


def _parse_issue_form(body: str) -> dict[str, str]:
    """Parse a GitHub issue-form body into a mapping of field labels to values.

    The issue form is rendered as markdown sections of the form:

        ### Field label
        value
    """
    out: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for raw in body.splitlines():
        m = _FIELD_RE.match(raw)
        if m:
            if current is not None:
                out[current] = "\n".join(lines).strip()
            current = m.group("label").strip()
            lines = []
            continue
        if current is not None:
            lines.append(raw)
    if current is not None:
        out[current] = "\n".join(lines).strip()
    return out


def _yaml_quote_block(s: str) -> str:
    # use YAML folded style
    s = s.rstrip()
    # indent by two spaces under key
    return ">\n" + "\n".join(f"  {line}".rstrip() for line in s.splitlines())


def build_yaml(issue: AdoptionIssue) -> str:
    """Render a single adoption entry as YAML text."""
    if issue.entry_type == "software":
        if not issue.repo:
            raise ValueError("Software entry requires a repository URL (repo).")

        return "\n".join(
            [
                f"name: {issue.name}",
                "kind: software",
                "description: " + _yaml_quote_block(issue.description),
                f"repo: {issue.repo}",
                f"source_issue: {issue.issue_url}",
                "",
            ]
        )

    # publication
    parts = [
        f"name: {issue.name}",
        "kind: publication",
        "description: " + _yaml_quote_block(issue.description),
    ]
    if issue.citation:
        parts.append("citation: " + _yaml_quote_block(issue.citation))
    if issue.link:
        parts.append(f"link: {issue.link}")
    parts.append(f"source_issue: {issue.issue_url}")
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    """Entry point for GitHub Actions to generate the adoption YAML file."""
    issue_body = os.environ["ISSUE_BODY"]
    issue_number = int(os.environ["ISSUE_NUMBER"])
    issue_url = os.environ["ISSUE_URL"]

    fields = _parse_issue_form(issue_body)
    print(issue_body)

    entry_type = _normalize_type(fields.get("Entry type", ""))
    name = fields.get("Name", "").strip()
    description = fields.get("Description", "").strip()
    repo = fields.get("Repository URL (software only)", "").strip() or None
    link = fields.get("Publication link (publication only)", "").strip() or None
    citation = fields.get("Citation / reference (publication only)", "").strip() or None
    contact = fields.get("Contact (optional)", "").strip() or None

    if not name:
        raise ValueError("Missing required field: Name")
    if not description:
        raise ValueError("Missing required field: Description")

    issue = AdoptionIssue(
        entry_type=entry_type,
        name=name,
        description=description,
        repo=repo,
        link=link,
        citation=citation,
        contact=contact,
        issue_number=issue_number,
        issue_url=issue_url,
    )

    slug = _slugify(name)
    suffix = f"-{issue_number}"

    if entry_type == "software":
        base = Path("docs/adoption/software") / slug
    else:
        base = Path("docs/adoption/publications") / slug

    out = base.with_suffix(".yml")
    if out.exists():
        out = Path(f"{base}{suffix}.yml")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_yaml(issue), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
