from __future__ import annotations

from docutils import nodes

from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective, SphinxRole
from sphinx.util.typing import ExtensionMetadata


class AdoptionDirective(SphinxDirective):
    """A directive to add adoptions"""

    required_arguments = 1

    def run(self) -> list[nodes.Node]:
        paragraph_node = nodes.paragraph(text="Hello, world!")
        return [paragraph_node]


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_directive("adoption", AdoptionDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
