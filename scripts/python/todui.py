#!/usr/bin/env python3
"""
Todui: TODO Parser TUI - A tool to find and organize TODO comments in C++ codebases
"""

import os
import re
import argparse
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict
import curses


@dataclass
class TodoItem:
  """Represents a single TODO item"""

  file_path: str
  line_number: int
  user: str
  message: str
  line_content: str

  def __str__(self):
    return f"{self.user}: {self.message}"


class TodoParser:
  """Parses codebase for TODO comments"""

  def __init__(self, extensions=None):
    self.extensions = extensions or [
      ".cpp",
      ".cc",
      ".cxx",
      ".c",
      ".h",
      ".hpp",
      ".hxx",
      ".py",
    ]
    # Pattern for single-line comments (// and #)
    self.single_line_pattern = re.compile(
      r"(?://|#)\s*TODO\(([^)]+)\):\s*(.*)", re.IGNORECASE
    )
    # Pattern for block comments (/* */)
    self.block_comment_pattern = re.compile(
      r"/\*\*?\s*\n?\s*\*?\s*TODO\(([^)]+)\):\s*(.*?)(?:\*/|$)",
      re.IGNORECASE | re.DOTALL,
    )

  def scan_file(self, file_path: Path) -> List[TodoItem]:
    """Scan a single file for TODO comments"""
    todos = []
    try:
      with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

        # Scan for single-line comments (// and #)
        for line_num, line in enumerate(content.splitlines(), 1):
          match = self.single_line_pattern.search(line)
          if match:
            user = match.group(1).strip()
            message = match.group(2).strip()
            todos.append(
              TodoItem(
                file_path=str(file_path),
                line_number=line_num,
                user=user,
                message=message,
                line_content=line.strip(),
              )
            )

        # Scan for block comments (/* */)
        for match in self.block_comment_pattern.finditer(content):
          user = match.group(1).strip()
          message = match.group(2).strip()

          # Clean up the message (remove extra whitespace and asterisks)
          message = re.sub(r"\s*\*\s*", " ", message).strip()

          # Find the line number where this block comment starts
          line_num = content[: match.start()].count("\n") + 1

          # Get the first line of the block comment for display
          block_start = match.start()
          block_end = content.find("\n", block_start)
          if block_end == -1:
            block_end = len(content)
          first_line = content[block_start:block_end].strip()

          todos.append(
            TodoItem(
              file_path=str(file_path),
              line_number=line_num,
              user=user,
              message=message,
              line_content=first_line,
            )
          )

    except Exception as e:
      # Skip files that can't be read
      print(f"Exception: {e}")
      pass
    return todos

  def scan_directory(self, directory: Path) -> List[TodoItem]:
    """Recursively scan directory for TODO comments"""
    all_todos = []

    for file_path in directory.rglob("*"):
      if file_path.is_file() and file_path.suffix in self.extensions:
        todos = self.scan_file(file_path)
        all_todos.extend(todos)

    return all_todos


class TodoTUI:
  """Terminal User Interface for TODO management"""

  def __init__(self, todos: List[TodoItem]):
    self.todos = todos
    self.grouped_todos = self._group_todos()
    self.current_view = "users"  # 'users', 'files', 'all'
    self.current_selection = 0
    self.current_user = None
    self.current_file = None
    self.scroll_offset = 0

    # Search functionality
    self.search_mode = False
    self.search_query = ""
    self.filtered_todos = todos
    self.search_history = []

  def _group_todos(self) -> Dict:
    """Group TODOs by user and file"""
    by_user = defaultdict(list)
    by_file = defaultdict(list)

    for todo in self.todos:
      by_user[todo.user].append(todo)
      by_file[todo.file_path].append(todo)

    return {"by_user": dict(by_user), "by_file": dict(by_file)}

  def _filter_todos(self, query: str) -> List[TodoItem]:
    """Filter TODOs based on search query"""
    if not query:
      return self.todos

    query_lower = query.lower()
    filtered = []

    for todo in self.todos:
      # Search in user, message, file path, and line content
      if (
        query_lower in todo.user.lower()
        or query_lower in todo.message.lower()
        or query_lower in todo.file_path.lower()
        or query_lower in todo.line_content.lower()
      ):
        filtered.append(todo)

    return filtered

  def run(self, stdscr):
    """Main TUI loop"""
    curses.curs_set(0)  # Hide cursor
    stdscr.timeout(100)  # Non-blocking input

    # Color pairs
    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Header
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Selected
    curses.init_pair(3, curses.COLOR_MAGENTA, curses.COLOR_BLACK)  # User
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # File
    curses.init_pair(5, curses.COLOR_RED, curses.COLOR_BLACK)  # TODO count
    curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_CYAN)  # Selected

    while True:
      stdscr.clear()
      self._draw_screen(stdscr)
      stdscr.refresh()

      if self.search_mode:
        self._handle_search_input(stdscr)
      else:
        key = stdscr.getch()
        if key == ord("q"):
          break
        elif key == ord("s"):
          self._start_search()
        elif key == curses.KEY_UP or key == ord("k"):
          self._move_selection(-1)
        elif key == curses.KEY_DOWN or key == ord("j"):
          self._move_selection(1)
        elif (
          key == ord("\n")
          or key == curses.KEY_ENTER
          or key == 10
          or key == curses.KEY_RIGHT
          or key == ord("l")
        ):
          self._handle_enter()
        elif key == ord("u"):
          self._switch_view("users")
        elif key == ord("f"):
          self._switch_view("files")
        elif key == ord("a"):
          self._switch_view("all")
        elif key == ord("b") or key == curses.KEY_LEFT or key == ord("h"):
          self._go_back()
        elif key == ord("e"):
          self._edit_current_todo()
        elif key == ord("c"):  # Clear search
          self._clear_search()

  def _draw_screen(self, stdscr):
    """Draw the current screen"""
    height, width = stdscr.getmaxyx()

    # Draw header
    header = f"Todui - Total: {len(self.todos)} TODOs"
    stdscr.attron(curses.color_pair(1))
    stdscr.addstr(0, 0, header.ljust(width))
    stdscr.attroff(curses.color_pair(1))

    # Draw navigation help
    help_text = "[u]sers [f]iles [a]ll [s]earch [e]dit [c]lear [q]uit"
    if self.current_user or self.current_file:
      help_text += " [b]ack"
    if self.search_query:
      help_text += f" [filter: '{self.search_query}']"
    stdscr.addstr(1, 0, help_text)

    # Draw current view
    if self.current_view == "users" and not self.current_user:
      self._draw_users_view(stdscr, height, width)
    elif self.current_view == "files" and not self.current_file:
      self._draw_files_view(stdscr, height, width)
    elif self.current_view == "all":
      self._draw_all_todos(stdscr, height, width)
    elif self.current_user:
      self._draw_user_todos(stdscr, height, width)
    elif self.current_file:
      self._draw_file_todos(stdscr, height, width)

  def _draw_users_view(self, stdscr, height, width):
    """Draw the users overview"""
    stdscr.addstr(3, 0, "Users:")

    users = list(self.grouped_todos["by_user"].keys())
    start_row = 4
    visible_rows = height - start_row - 1

    for i, user in enumerate(
      users[self.scroll_offset : self.scroll_offset + visible_rows]
    ):
      actual_index = i + self.scroll_offset
      count = len(self.grouped_todos["by_user"][user])
      line = f"  {user} ({count} TODOs)"

      if actual_index == self.current_selection:
        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(start_row + i, 0, line.ljust(width))
        stdscr.attroff(curses.color_pair(2))
      else:
        stdscr.attron(curses.color_pair(3))
        stdscr.addstr(start_row + i, 0, f"  {user}")
        stdscr.attroff(curses.color_pair(3))
        stdscr.attron(curses.color_pair(5))
        stdscr.addstr(f" ({count} TODOs)")
        stdscr.attroff(curses.color_pair(5))

  def _draw_files_view(self, stdscr, height, width):
    """Draw the files overview"""
    stdscr.addstr(3, 0, "Files:")

    files = list(self.grouped_todos["by_file"].keys())
    start_row = 4
    visible_rows = height - start_row - 1

    for i, file_path in enumerate(
      files[self.scroll_offset : self.scroll_offset + visible_rows]
    ):
      actual_index = i + self.scroll_offset
      count = len(self.grouped_todos["by_file"][file_path])
      filename = os.path.basename(file_path)
      line = f"  {filename} ({count} TODOs)"

      if actual_index == self.current_selection:
        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(start_row + i, 0, line.ljust(width))
        stdscr.attroff(curses.color_pair(2))
      else:
        stdscr.attron(curses.color_pair(4))
        stdscr.addstr(start_row + i, 0, f"  {filename}")
        stdscr.attroff(curses.color_pair(4))
        stdscr.attron(curses.color_pair(5))
        stdscr.addstr(f" ({count} TODOs)")
        stdscr.attroff(curses.color_pair(5))

  def _draw_all_todos(self, stdscr, height, width):
    """Draw all TODOs"""
    display_todos = self.filtered_todos if self.search_query else self.todos
    title = f"All TODOs ({len(display_todos)}/{len(self.todos)})"
    if self.search_query:
      title += f" [filtered by: '{self.search_query}']"
    stdscr.addstr(3, 0, title)

    start_row = 4
    visible_rows = height - start_row - 1

    for i, todo in enumerate(
      display_todos[self.scroll_offset : self.scroll_offset + visible_rows]
    ):
      actual_index = i + self.scroll_offset
      filename = os.path.basename(todo.file_path)
      line = f"  {filename}:{todo.line_number} [{todo.user}] {todo.message}"

      if len(line) > width:
        line = line[: width - 3] + "..."

      if actual_index == self.current_selection:
        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(start_row + i, 0, line.ljust(width))
        stdscr.attroff(curses.color_pair(2))
      else:
        stdscr.addstr(start_row + i, 0, line)

  def _draw_user_todos(self, stdscr, height, width):
    """Draw TODOs for specific user"""
    stdscr.addstr(3, 0, f"TODOs for {self.current_user}:")

    user_todos = self.grouped_todos["by_user"][self.current_user]
    start_row = 4
    visible_rows = height - start_row - 1

    for i, todo in enumerate(
      user_todos[self.scroll_offset : self.scroll_offset + visible_rows]
    ):
      actual_index = i + self.scroll_offset
      filename = os.path.basename(todo.file_path)
      line = f"  {filename}:{todo.line_number} {todo.message}"

      if len(line) > width:
        line = line[: width - 3] + "..."

      if actual_index == self.current_selection:
        stdscr.attron(curses.color_pair(6))
        stdscr.addstr(start_row + i, 0, line.ljust(width))
        stdscr.attroff(curses.color_pair(6))
      else:
        stdscr.attron(curses.color_pair(4))
        stdscr.addstr(start_row + i, 0, line)
        stdscr.attroff(curses.color_pair(4))

  def _draw_file_todos(self, stdscr, height, width):
    """Draw TODOs for specific file"""
    filename = os.path.basename(self.current_file)
    stdscr.addstr(3, 0, f"TODOs in {filename}:")

    file_todos = self.grouped_todos["by_file"][self.current_file]
    start_row = 4
    visible_rows = height - start_row - 1

    for i, todo in enumerate(
      file_todos[self.scroll_offset : self.scroll_offset + visible_rows]
    ):
      actual_index = i + self.scroll_offset
      line = f"  Line {todo.line_number}: [{todo.user}] {todo.message}"

      if len(line) > width:
        line = line[: width - 3] + "..."

      if actual_index == self.current_selection:
        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(start_row + i, 0, line.ljust(width))
        stdscr.attroff(curses.color_pair(2))
      else:
        stdscr.addstr(start_row + i, 0, line)

  def _move_selection(self, direction):
    """Move selection up or down"""
    if self.current_view == "users" and not self.current_user:
      max_items = len(self.grouped_todos["by_user"])
    elif self.current_view == "files" and not self.current_file:
      max_items = len(self.grouped_todos["by_file"])
    elif self.current_view == "all":
      # Use filtered results if search is active
      display_todos = self.filtered_todos if self.search_query else self.todos
      max_items = len(display_todos)
    elif self.current_user:
      max_items = len(self.grouped_todos["by_user"][self.current_user])
    elif self.current_file:
      max_items = len(self.grouped_todos["by_file"][self.current_file])
    else:
      return

    self.current_selection = max(
      0, min(max_items - 1, self.current_selection + direction)
    )

    # Adjust scroll offset
    if self.current_selection < self.scroll_offset:
      self.scroll_offset = self.current_selection
    elif (
      self.current_selection >= self.scroll_offset + 20
    ):  # Assume 20 visible rows
      self.scroll_offset = self.current_selection - 19

  def _handle_enter(self):
    """Handle enter key press"""
    if self.current_view == "users" and not self.current_user:
      users = list(self.grouped_todos["by_user"].keys())
      if self.current_selection < len(users):
        self.current_user = users[self.current_selection]
        self.current_selection = 0
        self.scroll_offset = 0
    elif self.current_view == "files" and not self.current_file:
      files = list(self.grouped_todos["by_file"].keys())
      if self.current_selection < len(files):
        self.current_file = files[self.current_selection]
        self.current_selection = 0
        self.scroll_offset = 0

  def _switch_view(self, view):
    """Switch between different views"""
    self.current_view = view
    self.current_user = None
    self.current_file = None
    self.current_selection = 0
    self.scroll_offset = 0

  def _go_back(self):
    """Go back to previous view"""
    if self.current_user:
      self.current_user = None
      self.current_selection = 0
      self.scroll_offset = 0
    elif self.current_file:
      self.current_file = None
      self.current_selection = 0
      self.scroll_offset = 0

  def _get_current_todo(self) -> Optional[TodoItem]:
    """Get the currently selected TODO item"""
    if self.current_view == "all":
      # Use filtered results if search is active
      display_todos = self.filtered_todos if self.search_query else self.todos
      if 0 <= self.current_selection < len(display_todos):
        return display_todos[self.current_selection]
    elif self.current_user:
      user_todos = self.grouped_todos["by_user"][self.current_user]
      if 0 <= self.current_selection < len(user_todos):
        return user_todos[self.current_selection]
    elif self.current_file:
      file_todos = self.grouped_todos["by_file"][self.current_file]
      if 0 <= self.current_selection < len(file_todos):
        return file_todos[self.current_selection]
    return None

  def _edit_current_todo(self):
    """Open current TODO in editor"""
    todo = self._get_current_todo()
    if not todo:
      return

    # Get editor from environment, fallback to common editors
    editor = os.environ.get("EDITOR")
    if not editor:
      # Try to find a common editor
      for candidate in ["nvim", "vim", "nano", "emacs", "code", "subl"]:
        editor_path = shutil.which(candidate)
        if editor_path:
          editor = editor_path
          break

    if not editor:
      return

    # Get the base name of the editor for comparison
    editor_name = os.path.basename(editor).lower()

    # Store current curses state
    curses.def_prog_mode()

    try:
      # End curses mode completely
      curses.endwin()

      # Reset terminal to normal mode
      os.system("reset")

      # Different editors have different syntax for going to a line
      if editor_name in ["vim", "nvim", "vi"]:
        # Use -c command to ensure we go to the line after opening
        cmd = [editor, "-c", f"{todo.line_number}", todo.file_path]
      elif editor_name in ["emacs", "emacsclient"]:
        cmd = [editor, f"+{todo.line_number}", todo.file_path]
      elif editor_name == "nano":
        cmd = [editor, f"+{todo.line_number}", todo.file_path]
      elif editor_name == "code":  # VS Code
        cmd = [editor, "-g", f"{todo.file_path}:{todo.line_number}"]
      elif editor_name in ["subl", "sublime_text"]:  # Sublime Text
        cmd = [editor, f"{todo.file_path}:{todo.line_number}"]
      else:
        # Generic fallback - just open the file
        cmd = [editor, todo.file_path]

      # Execute the editor and wait for it to complete
      _ = subprocess.run(cmd, stdin=None, stdout=None, stderr=None)

    except Exception as e:
      # If editor fails, just continue
      pass
    finally:
      # Clear screen and reinitialize curses properly
      print("\033[2J\033[H")  # Clear screen and move cursor to top
      curses.reset_prog_mode()
      curses.curs_set(0)  # Hide cursor again

  def _start_search(self):
    """Start search mode"""
    self.search_mode = True
    self.search_query = ""
    self.filtered_todos = self.todos
    self.current_selection = 0
    self.scroll_offset = 0

  def _clear_search(self):
    """Clear current search filter"""
    self.search_query = ""
    self.filtered_todos = self.todos
    self.current_selection = 0
    self.scroll_offset = 0

  def _apply_search(self):
    """Apply current search query"""
    self.filtered_todos = self._filter_todos(self.search_query)
    self.current_selection = 0
    self.scroll_offset = 0

  def _handle_search_input(self, stdscr):
    """Handle search input mode"""
    stdscr.timeout(-1)  # Blocking input for search
    curses.curs_set(1)  # Show cursor for search

    # Draw search prompt
    height, _ = stdscr.getmaxyx()
    prompt = f"Search: {self.search_query}"
    stdscr.addstr(height - 1, 0, prompt)
    stdscr.addstr(height - 1, len(prompt), "_")  # Cursor

    key = stdscr.getch()

    if key == 27:  # ESC key
      self.search_mode = False
      curses.curs_set(0)  # Hide cursor
      stdscr.timeout(100)  # Non-blocking input
    elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:
      # Backspace
      if self.search_query:
        self.search_query = self.search_query[:-1]
        self._apply_search()
    elif key == curses.KEY_ENTER or key == 10 or key == ord("\n"):
      # Enter - apply search and exit search mode
      self.search_mode = False
      curses.curs_set(0)  # Hide cursor
      stdscr.timeout(100)  # Non-blocking input
    elif 32 <= key <= 126:  # Printable characters
      self.search_query += chr(key)
      self._apply_search()


def print_summary(todos: List[TodoItem]):
  """Print a summary of found TODOs"""
  if not todos:
    print("No TODO comments found.")
    return

  print(f"\nFound {len(todos)} TODO comments:")

  # Group by user
  by_user = defaultdict(list)
  for todo in todos:
    by_user[todo.user].append(todo)

  for user, user_todos in by_user.items():
    print(f"\n{user} ({len(user_todos)} TODOs):")
    for todo in user_todos:
      filename = os.path.basename(todo.file_path)
      print(f"  {filename}:{todo.line_number} - {todo.message}")


def main():
  parser = argparse.ArgumentParser(
    description="Parse and display TODO comments"
  )
  parser.add_argument(
    "directory",
    nargs="?",
    default=".",
    help="Directory to scan (default: current directory)",
  )
  parser.add_argument(
    "--extensions",
    nargs="+",
    default=[".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx", ".py"],
    help="File extensions to scan",
  )
  parser.add_argument(
    "--no-tui", action="store_true", help="Print summary instead of showing TUI"
  )

  args = parser.parse_args()

  # Parse TODOs
  todo_parser = TodoParser(extensions=args.extensions)
  directory = Path(args.directory)

  if not directory.exists():
    print(f"Error: Directory '{directory}' does not exist.")
    return os.EX_SOFTWARE

  print(f"Scanning {directory} for TODO comments...")
  todos = todo_parser.scan_directory(directory)

  if args.no_tui:
    print_summary(todos)
  else:
    if not todos:
      print("No TODO comments found.")
      return os.EX_OK

    # Launch TUI
    tui = TodoTUI(todos)
    try:
      curses.wrapper(tui.run)
    except KeyboardInterrupt:
      pass

  return os.EX_OK


if __name__ == "__main__":
  exit(main())
