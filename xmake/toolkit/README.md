# xmake toolkit

Three reusable xmake rules, with no assumptions about the project that uses
them. Copy this directory into any xmake project and add one `includes` line.

- **`toolkit.provenance`** — stamps the git commit, compiler version, build
  timestamp, mode, architecture, and platform into the binary.
- **`toolkit.sanitizers`** — adds `asan`, `tsan`, `ubsan`, `lsan`, and `allsan`
  as ordinary build modes.
- **`toolkit.modes`** — adds `relwithdebinfo` (CMake's `RelWithDebInfo`) and
  `perf` (sampling profiles without `-pg`), and enables LTO for optimized modes.

Each file carries its own documentation; this is the short version.

## Install

Copy `xmake/toolkit/` into the target project, then in its `xmake.lua`:

```lua
set_allowedmodes(
  "debug", "release", "releasedbg", "profile",   -- xmake's own
  "relwithdebinfo", "perf",                      -- toolkit.modes
  "asan", "tsan", "ubsan", "lsan", "allsan"      -- toolkit.sanitizers
)

includes("xmake/toolkit")

add_rules("mode.debug", "mode.release", "mode.releasedbg", "mode.profile")
add_rules("toolkit.modes", "toolkit.sanitizers")
```

`includes("xmake/toolkit")` loads `xmake/toolkit/xmake.lua`, which pulls in the
three rule files.

Note that `set_allowedmodes` must list every mode you intend to use — xmake
rejects an unlisted one during configuration. `toolkit.modes` and
`toolkit.sanitizers` are per-target rules, so a target that sets its own
`optimize` or `symbols` keeps them.

## Build provenance

Add a template that renders to a source file. Anything the compiler accepts
works; a C++ example:

```cpp
// src/build_info.cpp.in
#include <string>
#include "build_info.hpp"
namespace build_info {
const std::string GIT_HASH = "@GIT_HASH@";
const std::string BUILD_TIMESTAMP = "@BUILD_TIMESTAMP@";
const std::string COMPILER = "@COMPILER@";
const std::string OPTIMIZATION = "@OPTIMIZATION@";
const std::string ARCH = "@ARCH@";
const std::string OS = "@OS@";
} // namespace build_info
```

Then, on the target that should carry it:

```lua
target("app")
  set_kind("binary")
  add_rules("toolkit.provenance")
  set_values("toolkit.provenance.template", "src/build_info.cpp.in")
```

The rendered file is added to the target's sources automatically, so it needs no
`add_files`. It lands in `<builddir>/provenance/build_info.cpp` — the template's
name with a trailing `.in` removed. Override the directory with
`set_values("toolkit.provenance.outputdir", "...")`.

Placeholders are `GIT_HASH`, `COMPILER`, `OPTIMIZATION`, `ARCH`, `OS`, and
`BUILD_TIMESTAMP`. An unknown `@NAME@` in the template is an error naming the
available set, rather than a mystery compile failure later. Values are escaped
for a C++ string literal.

`COMPILER` is the first line of the compiler's own `--version` output, so it
records the real toolchain, not the name xmake was asked for. If git or the
compiler cannot be run, the fields fall back to `unknown` and the tool name; the
build does not fail.

### Why it does not relink every build

The obvious version of this rule rewrites the generated file on every build,
which changes its timestamp, which relinks the binary every time. This rule
records the substituted values, excluding `BUILD_TIMESTAMP`, in a `.state` file
beside the output. If nothing but the time has changed, the file is left alone.

So `BUILD_TIMESTAMP` is the time of the last build that changed something —
commit, compiler, mode, architecture, platform, or the template itself. That is
usually what you want in provenance. If you need the wall-clock time of every
build, this rule is the wrong tool, and you should accept the relink.

## Sanitizer modes

```sh
xmake f -m asan && xmake
xmake f -m tsan && xmake
```

AddressSanitizer and ThreadSanitizer cannot coexist in one executable, so
`allsan` combines address, leak, and undefined behavior only. Run `tsan`
separately. `allsan` also drops to `-O0`, since stacked sanitizer reports are
hard to read through inlining; a single sanitizer stays at `-O3`.

Instrumentation reaches only the targets the rule is applied to. Prebuilt or
externally built dependencies stay uninstrumented, which is usually right but
does mean a bug inside one of them may surface only at its boundary.

## Requirements

Developed against xmake 3.0.7. `toolkit.sanitizers` relies on the
`build.sanitizer.*` policies, which need xmake 2.8.6 or later.

Some caveats worth knowing if you extend these:

- xmake's *description* scope (the top level of `xmake.lua` and anything
  `includes()`d) runs in a restricted sandbox with no `io` module and no
  `os.iorunv`. Rule scripts (`on_load`, `before_build`, ...) do not have that
  limit. That is why the provenance rule resolves paths at description scope and
  captures them as locals, then does its file and process work inside the rule.
- Locals captured as upvalues by the rule closures work across that boundary;
  file-scope globals do not reliably.
