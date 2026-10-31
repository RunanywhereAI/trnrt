"""Every command line parses without abbreviations (argparse allow_abbrev=False).

An argv handed from one command to another (serve's to each engine, an engine's to engine_config, check_mixed's to
engine_config) is read by parsers that know different arguments. With abbreviations, a parser reads an argument it
lacks as one it has that the argument is a prefix of: serve's `--host 127.0.0.1` became each engine's --host-cache-gb,
and no engine started. So every ArgumentParser(...) and add_parser(...) call under trnrt/, tools/ and docker/ must pass
allow_abbrev=False; a positive control requires the scan to flag a parser without it.
"""

import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = ("trnrt", "tools", "docker")
# Its parsers change with the compile pipeline's own fix (which parses capture's engine arguments strictly); remove it
# from here with that change.
EXEMPT = {"trnrt/compile/farm.py"}


def abbreviating(src: str, where: str) -> list[str]:
    """`where:line` of each parser construction in `src` without allow_abbrev=False."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
        if name in ("ArgumentParser", "add_parser"):
            v = {k.arg: k.value for k in node.keywords}.get("allow_abbrev")
            if not (isinstance(v, ast.Constant) and v.value is False):
                out.append(f"{where}:{node.lineno}")
    return out


def test_every_parser_refuses_abbreviations():
    found, seen = [], 0
    for d in DIRS:
        for dirpath, _, files in os.walk(os.path.join(ROOT, d)):
            for fn in files:
                rel = os.path.relpath(os.path.join(dirpath, fn), ROOT)
                if fn.endswith(".py") and rel not in EXEMPT:
                    with open(os.path.join(ROOT, rel)) as f:
                        src = f.read()
                    seen += src.count("ArgumentParser(")
                    found += abbreviating(src, rel)
    assert seen >= 6, seen  # the scan reads the parsers (trnrt/__main__.py alone has three)
    assert found == [], found


def test_the_scan_flags_a_parser_that_abbreviates():
    src = ("import argparse\n"
           "a = argparse.ArgumentParser(prog='x')\n"
           "b = argparse.ArgumentParser(prog='y', allow_abbrev=False)\n"
           "c = argparse.ArgumentParser(allow_abbrev=True)\n"
           "s = b.add_subparsers().add_parser('run')\n")
    assert abbreviating(src, "f.py") == ["f.py:2", "f.py:4", "f.py:5"]
