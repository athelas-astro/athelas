#!/usr/bin/env python3
"""
schema_to_docs.py
-----------------
Generate a Sphinx RST reference page from schema.lua.

Usage:
    python schema_to_docs.py schema.lua                  # writes schema_reference.rst
    python schema_to_docs.py schema.lua output.rst       # explicit output path

Requires: lupa  (uv add lupa)

Output format:
    One .. list-table:: per schema section, with columns:
        Key | Type | Default | Required | Description
    Nested sections become subsections with their own tables.
    The result is ready for .. include:: in a Sphinx .rst file.
"""

import sys
from pathlib import Path

from athelas_tools.schema_load import load_schema, is_leaf

# -----------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------


def format_mandatory(m) -> str:
  if m is True:
    return "Yes"
  if isinstance(m, dict):
    when = m.get("when", "?")
    if m.get("is_true"):
      return f"When ``{when}`` is true"
    equals = m.get("equals")
    if equals is not None:
      return f"When ``{when}`` = ``{equals}``"
  return "No"


def format_default(d) -> str:
  if d is None:
    return "—"
  if isinstance(d, bool):
    return "``true``" if d else "``false``"
  if isinstance(d, float):
    if d != 0.0 and (abs(d) >= 1e6 or abs(d) < 1e-3):
      return f"``{d:e}``"
    return f"``{d}``"
  if isinstance(d, int):
    return f"``{d}``"
  return f"``{d}``"


def format_type(t) -> str:
  return t if t else "—"


# -----------------------------------------------------------------------
# RST generation
# -----------------------------------------------------------------------

HEADER_CHARS = ["=", "-", "~", "^", '"']


def rst_section_header(title: str, level: int) -> str:
  char = HEADER_CHARS[min(level, len(HEADER_CHARS) - 1)]
  return f"{title}\n{char * len(title)}\n"


def rst_list_table(rows: list[tuple], section_path: str) -> str:
  """Emit a .. list-table:: for the leaf entries in this section."""
  if not rows:
    return ""

  lines = [
    ".. list-table::",
    "   :widths: 10 8 12 20 50",
    "   :header-rows: 1",
    "",
    "   * - Key",
    "     - Type",
    "     - Default",
    "     - Required",
    "     - Description",
  ]
  for key, typ, default, mandatory, doc in rows:
    lines.append(f"   * - ``{key}``")
    lines.append(f"     - {format_type(typ)}")
    lines.append(f"     - {format_default(default)}")
    lines.append(f"     - {format_mandatory(mandatory)}")
    lines.append(f"     - {doc if doc else '—'}")
  lines.append("")
  return "\n".join(lines) + "\n"


def walk(node: dict, path: str, level: int, out: list[str]) -> None:
  """Recursively walk the schema dict, emitting RST sections and tables."""
  leaves: list[tuple] = []
  children: list[tuple[str, dict]] = []

  for key, val in node.items():
    if not isinstance(val, dict):
      continue
    if val.get("ignore"):
      leaves.append(
        (
          key,
          "table",
          None,
          val.get("mandatory", False),
          val.get("doc", "Problem-specific. Not schema-validated."),
        )
      )
      continue
    if is_leaf(val):
      leaves.append(
        (
          key,
          val.get("type"),
          val.get("default"),
          val.get("mandatory", False),
          val.get("doc", ""),
        )
      )
    else:
      children.append((key, val))

  if leaves:
    out.append(rst_list_table(leaves, path))

  for child_key, child_val in children:
    child_path = f"{path}.{child_key}" if path else child_key
    out.append(rst_section_header(f"``{child_key}``", level))
    walk(child_val, child_path, level + 1, out)


def generate_rst(schema: dict, source_name: str) -> str:
  out: list[str] = []
  out.append(rst_section_header("Input Deck Reference", 0))
  out.append(
    f"Auto-generated from ``{source_name}``. "
    "Each table below documents one section of the ``config`` table.\n\n"
  )
  for section_key, section_val in sorted(schema.items()):
    if not isinstance(section_val, dict):
      continue
    out.append(rst_section_header(f"``config.{section_key}``", 1))
    walk(section_val, section_key, 2, out)

  return "\n".join(out)


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------


def main() -> None:
  if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

  schema_path = Path(sys.argv[1])
  if not schema_path.exists():
    print(f"Error: {schema_path} not found")
    sys.exit(1)

  out_path = (
    Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("schema_reference.rst")
  )

  schema = load_schema(schema_path)
  rst = generate_rst(schema, schema_path.name)
  out_path.write_text(rst)
  print(f"Wrote {out_path}")


if __name__ == "__main__":
  main()
