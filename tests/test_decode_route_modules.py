"""tests/conftest.py's ROUTE_MODULES, which the decode_routes fixture restores, is every module that
engine.set_decode_routes imports and writes: a route added in another module would leak between tests unseen."""

import ast
import inspect

from tests.conftest import ROUTE_MODULES
from trnrt.engine import engine as E


def test_route_modules_are_set_decode_routes_imports():
    tree = ast.parse(inspect.getsource(E.set_decode_routes))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = {"..kernels": "trnrt.kernels", "..models": "trnrt.models"}[("." * node.level) + (node.module or "")]
            imported |= {f"{base}.{a.name}" for a in node.names}
    assert imported == set(ROUTE_MODULES)
