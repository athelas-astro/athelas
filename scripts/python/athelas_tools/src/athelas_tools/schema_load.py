"""
schema_load.py
--------------
Shared schema loading logic for schema_to_docs.py and schema_explorer.py.

Requires: lupa  (uv add lupa)
"""

from pathlib import Path

import lupa
from lupa import LuaRuntime


def lua_table_to_dict(lua_table) -> dict:
  """Recursively convert a lupa Lua table to a Python dict."""
  result = {}
  for k, v in lua_table.items():
    key = str(k)
    if lupa.lua_type(v) == "table":
      result[key] = lua_table_to_dict(v)
    elif isinstance(v, bool):
      result[key] = v
    else:
      result[key] = v
  return result


def load_schema(lua_path: Path) -> dict:
  """Execute schema.lua via lupa and return a plain Python dict."""
  rt = LuaRuntime(unpack_returned_tuples=True)
  lua_table = rt.execute(lua_path.read_text())
  return lua_table_to_dict(lua_table)


def is_leaf(node: dict) -> bool:
  """A schema leaf descriptor contains a 'doc' key."""
  return isinstance(node, dict) and "doc" in node
