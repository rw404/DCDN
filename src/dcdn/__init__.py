"""DCDN — Deep Cross-Division Network, pre-release subset.

This package is the minimal module set the published results were produced
with. The full research tree, including data acquisition and materialization,
lives on the project's experiment branch.
"""

from dcdn.core import DCDNCore, DCDNCoreBlock

__all__ = ["DCDNCore", "DCDNCoreBlock"]
