"""
Input/Output utilities for the Erebors Priors package.
"""
from __future__ import annotations
import yaml
import re

class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads ``1e-4`` as a float (YAML 1.2 rule; PyYAML gives a str)."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":  # `<<: *anchor`, expanded by SafeLoader
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    f"found duplicate key {key!r}", key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)

_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?[0-9][0-9_]*(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)

if __name__ == "__main__":
    filename = "prior_example.yaml"

    loaded = yaml.load(open(filename), Loader=_Loader)

    breakpoint()