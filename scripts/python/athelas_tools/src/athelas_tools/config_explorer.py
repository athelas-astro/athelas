#!/usr/bin/env python3
"""
schema_explorer.py
------------------
Interactive TUI explorer for the Athelas input deck schema.

Usage:
    python schema_explorer.py schema.lua

Navigation:
    Arrow keys / j/k   Move through sections and fields
    Tab / Shift+Tab     Switch focus between panes
    q / Ctrl+C          Quit
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field

from athelas_tools.schema_load import load_schema as _load_raw, is_leaf

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, ListItem, ListView, Static
from textual.reactive import reactive


# ---------------------------------------------------------------------------
# Schema loading and data model
# ---------------------------------------------------------------------------


@dataclass
class LeafNode:
  path: str  # full dotted path, e.g. "opacity.floors.type"
  key: str  # bare key, e.g. "type"
  typ: str | None
  default: object
  mandatory: object  # True, False, or dict with 'when'/'equals'/'is_true'
  doc: str
  ignore: bool = False


@dataclass
class SectionNode:
  path: str  # full dotted path, e.g. "opacity.floors"
  label: str  # display label, e.g. "opacity.floors"
  leaves: list[LeafNode] = field(default_factory=list)
  children: list["SectionNode"] = field(default_factory=list)


def walk_schema(node: dict, path: str) -> SectionNode:
  """Recursively build a SectionNode tree from the schema dict."""
  section = SectionNode(path=path, label=path)
  for key, val in node.items():
    if not isinstance(val, dict):
      continue
    full_path = f"{path}.{key}" if path else key

    if val.get("ignore"):
      section.leaves.append(
        LeafNode(
          path=full_path,
          key=key,
          typ="table",
          default=None,
          mandatory=False,
          doc=val.get("doc", "Problem-specific. Not schema-validated."),
          ignore=True,
        )
      )
      continue

    if is_leaf(val):
      mandatory = val.get("mandatory", False)
      section.leaves.append(
        LeafNode(
          path=full_path,
          key=key,
          typ=val.get("type"),
          default=val.get("default"),
          mandatory=mandatory,
          doc=val.get("doc", ""),
        )
      )
    else:
      section.children.append(walk_schema(val, full_path))

  return section


def load_schema(lua_path: Path) -> list[SectionNode]:
  """Load schema.lua and return a flat list of SectionNodes (depth-first)."""
  schema_dict = _load_raw(lua_path)
  sections: list[SectionNode] = []
  for key, val in sorted(schema_dict.items()):
    if not isinstance(val, dict):
      continue
    sections.extend(flatten(walk_schema(val, key)))
  return sections


def flatten(node: SectionNode) -> list[SectionNode]:
  """Depth-first flattening of a SectionNode tree."""
  result = [node]
  for child in node.children:
    result.extend(flatten(child))
  return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_mandatory(mandatory: object) -> str:
  if mandatory is True:
    return "[bold red]always required[/bold red]"
  if isinstance(mandatory, dict):
    when = mandatory.get("when", "?")
    if mandatory.get("is_true"):
      return f"[yellow]when [italic]{when}[/italic] is true[/yellow]"
    equals = mandatory.get("equals")
    if equals is not None:
      return f"[yellow]when [italic]{when}[/italic] = [bold]{equals!r}[/bold][/yellow]"
  return "[dim]optional[/dim]"


def fmt_default(default: object) -> str:
  if default is None:
    return "[dim]—[/dim]"
  if isinstance(default, bool):
    return f"[cyan]{'true' if default else 'false'}[/cyan]"
  if isinstance(default, float):
    if default != 0.0 and (abs(default) >= 1e6 or abs(default) < 1e-3):
      return f"[cyan]{default:e}[/cyan]"
    return f"[cyan]{default}[/cyan]"
  return f"[cyan]{default!r}[/cyan]"


def fmt_type(typ: str | None) -> str:
  if typ is None:
    return "[dim]—[/dim]"
  return f"[green]{typ}[/green]"


def fmt_field_row(leaf: LeafNode) -> str:
  """One-line summary for the field list."""
  if leaf.ignore:
    return f"  [bold]{leaf.key}[/bold]  [dim]passthrough[/dim]"
  m = leaf.mandatory
  if m is True:
    badge = "[bold red]required[/bold red]"
  elif isinstance(m, dict):
    badge = "[yellow]conditional[/yellow]"
  else:
    d = leaf.default
    if d is None:
      badge = "[dim]optional[/dim]"
    else:
      badge = f"[dim]default:[/dim] {fmt_default(d)}"
  typ = f"[green]{leaf.typ}[/green]" if leaf.typ else "[dim]—[/dim]"
  return f"  [bold]{leaf.key}[/bold]   {typ}   {badge}"


def build_detail(leaf: LeafNode) -> str:
  """Rich-formatted detail panel content for a leaf."""
  lines = [
    f"[bold underline]{leaf.path}[/bold underline]",
    "",
    f"[dim]Type:[/dim]      {fmt_type(leaf.typ)}",
    f"[dim]Required:[/dim]  {fmt_mandatory(leaf.mandatory)}",
    f"[dim]Default:[/dim]   {fmt_default(leaf.default)}",
    "",
  ]
  if leaf.ignore:
    lines.append(
      "[italic]This subtable is passed through without schema validation.[/italic]"
    )
  else:
    doc = leaf.doc.strip() if leaf.doc else "[dim]No description.[/dim]"
    lines.append(doc)
  return "\n".join(lines)


# ---------------------------------------------------------------------------
# Textual widgets
# ---------------------------------------------------------------------------


class BrandingBanner(Static):
  DEFAULT_CSS = """
    BrandingBanner {
        content-align: center middle;
        color: $primary;
        height: 3;
        text-style: bold;
        padding: 1 2;
    }
    """

  def render(self) -> str:
    return "Athelas  v26.03\nSchema Explorer"


class DetailPanel(Static):
  """Bottom-right panel showing full details for the selected field."""

  DEFAULT_CSS = """
    DetailPanel {
        border: solid $primary-darken-2;
        padding: 1 2;
        height: 1fr;
    }
    """

  def show(self, leaf: LeafNode | None) -> None:
    if leaf is None:
      self.update("[dim]Select a field to see details.[/dim]")
    else:
      self.update(build_detail(leaf))


class FieldList(ListView):
  """Top-right pane: list of fields in the selected section."""

  DEFAULT_CSS = """
    FieldList {
        border: solid $primary-darken-2;
        height: 1fr;
    }
    """


class SectionList(ListView):
  """Left pane: list of schema sections."""

  DEFAULT_CSS = """
    SectionList {
        border: solid $primary-darken-2;
        width: 28;
        height: 1fr;
    }
    """


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class SchemaExplorer(App):
  TITLE = "Athelas Config Explorer"
  CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        layout: horizontal;
        height: 1fr;
    }
    #right {
        layout: vertical;
        width: 1fr;
    }
    SectionList ListItem {
        padding: 0 1;
    }
    FieldList ListItem {
        padding: 0 1;
    }
    """

  BINDINGS = [
    Binding("q", "quit", "Quit"),
    Binding("ctrl+p", "command_palette", "Themes"),
    Binding("j", "cursor_down_vi", "Down", show=True),
    Binding("k", "cursor_up_vi", "Up", show=True),
  ]

  selected_section: reactive[int] = reactive(0)
  selected_field: reactive[int] = reactive(0)

  def __init__(self, sections: list[SectionNode]):
    super().__init__()
    self.sections = sections

  def compose(self) -> ComposeResult:
    yield Header()
    yield BrandingBanner()
    with Horizontal(id="main"):
      yield SectionList(
        *[ListItem(Static(f"[bold]{s.label}[/bold]")) for s in self.sections],
        id="section-list",
      )
      with Vertical(id="right"):
        yield FieldList(id="field-list")
        yield DetailPanel(id="detail")
    yield Footer()

  def on_mount(self) -> None:
    self._populate_fields(0)
    self.query_one("#section-list").focus()

  def _populate_fields(self, section_idx: int) -> None:
    field_list = self.query_one("#field-list", FieldList)
    field_list.clear()
    if not self.sections:
      return
    section = self.sections[section_idx]
    all_leaves = section.leaves[:]
    # Also show leaves from immediate children inline, prefixed
    for child in section.children:
      for leaf in child.leaves:
        all_leaves.append(leaf)
    self._current_leaves = all_leaves
    for leaf in all_leaves:
      field_list.append(ListItem(Static(fmt_field_row(leaf))))
    # Reset detail
    detail = self.query_one("#detail", DetailPanel)
    detail.show(all_leaves[0] if all_leaves else None)

  def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
    list_id = event.list_view.id

    if list_id == "section-list":
      idx = event.list_view.index
      if idx is not None:
        self._populate_fields(idx)

    elif list_id == "field-list":
      idx = event.list_view.index
      if idx is not None and hasattr(self, "_current_leaves"):
        leaves = self._current_leaves
        if 0 <= idx < len(leaves):
          detail = self.query_one("#detail", DetailPanel)
          detail.show(leaves[idx])

  def action_cursor_down_vi(self) -> None:
    focused = self.focused
    if isinstance(focused, ListView):
      focused.action_cursor_down()

  def action_cursor_up_vi(self) -> None:
    focused = self.focused
    if isinstance(focused, ListView):
      focused.action_cursor_up()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
  if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

  schema_path = Path(sys.argv[1])
  if not schema_path.exists():
    print(f"Error: {schema_path} not found")
    sys.exit(1)

  sections = load_schema(schema_path)
  app = SchemaExplorer(sections)
  app.run()


if __name__ == "__main__":
  main()
