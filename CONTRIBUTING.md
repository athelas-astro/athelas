# Contributing to Athelas

Thank you for your interest in contributing to `Athelas`!
This document provides current guidelines and information for contributors.
As a general rule, these are not not set in stone but subject to possible change, dicussion, and revision.

## Table of Contents

- [Contributing Code](#contributing-code)
- [Pull Request Process](#pull-request-process)
- [Code Style and Formatting](#code-style-and-formatting)
- [Testing](#testing)
- [Versioning](#versioning)
- [Documentation](#documentation)

## Contributing Code

The general workflow is to create a branch (or fork, as relevant),
implement the feature there, and open a PR into the main branch.
Please try to keep PRs small and targeted, as this makes the reviewing process
simpler, faster, and helps avoid things slipping into the codebase that we otherwise
might not want.

Even if your PR is not ready you may feel free to open a draft PR marked
with "[WIP]" in the title. These are not merged until [WIP] is removed and
can be convenient as GitHub's tests will launch even on the draft PR.

### Branching Strategy

- Create feature branches from the main development branch
- To keep things orderly, contributors are encouraged to create branches starting with their username or
    initials followed by a "/" and ending with a brief description, e.g., `astrobarker/multigroup`.
- Keep branches focused on a single feature or bugfix

### Commit Messages

- Use clear, descriptive commit messages
- Follow conventional commit format when possible:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `test:` for test additions/changes
  - `refactor:` for code refactoring
  - `style:` for formatting changes

## Pull Request Process

All PRs are subject to review by at least one core maintainer.
PRs must include a thorough description of the implemented feature or fix.
If this breaks existing APIs or code, this must be noted here.
For a PR to be merged, the following must:

- obey the style guide (test with CPPLINT) (coming soon)
- pass the linting test (test with CPPLINT) (coming soon)
- pass the formatting test (see "Formatting Code" below)
- pass the existing test suite
- include tests that cover the new feature (if applicable)
- include documentation in the `doc/` folder (feature or developer; if applicable)
- include a brief summary in `CHANGELOG.md`

**All code review will be done by humans and will not involve agentic AI.**

## Code Style and Formatting

### C++ Code

We use `clang-format` with a specific configuration (`.clang-format`). Key requirements:

- **Column limit:** 80 characters
- **Indentation:** 2 spaces
- **Pointer alignment:** Right-aligned (`int *ptr`)
- **Brace style:** Attach braces to function/class declarations
- **Include sorting:** Case-sensitive with specific priority rules

In general you should not worry too much about formatting guidelines.
There exists a script to automatically format code:

**Formatting your code:**
```bash
# Format all C++ files
./scripts/bash/format.sh

# Or format specific files
clang-format -i src/your_file.cpp
```

### Python Code

We use `ruff` for Python formatting and linting. Configuration is in `ruff.toml`:

- **Line length:** 80 characters
- **Indentation:** 2 spaces
- **Quote style:** Double quotes

**Formatting your Python code:**
```bash
# Format all Python files
ruff format .

# Format specific file
ruff format your_script.py

# Check for linting issues
ruff check .
```

### Pre-commit Hooks

We provide pre-commit hooks that automatically format your code before commits.
Enable them with:

```bash
./scripts/hooks/install-hooks.sh
```

The hooks will:
- Format C++ and Python files using clang-format and ruff
- Check for trailing whitespace
- Prevent commits with formatting issues

## Testing

### Test Structure

- **Unit tests:** Located in `test/unit/` (C++ tests using Catch2)
- **Regression tests:** Located in `test/regression/` (Python-based integration tests)

### Running Tests

**Unit tests:**
`Athelas` must be built with `-DATHELAS_ENABLE_UNIT_TESTS=On` to build the unit tests.
```bash
cd build
make test
# or
ctest
```

**Regression tests:**
```bash
# Run all regression tests
python test/regression/run_regression_tests.py

# Run with existing executable (faster)
python test/regression/run_regression_tests.py -e /path/to/athelas/executable

# Run specific test
python test/regression/run_regression_tests.py --test test_sod -e /path/to/athelas/executable
```

### Adding Tests

Please add appropriate tests to the suite for new features.


## Versioning

We use **CalVer** (Calendar Versioning) with the format `vYY.0M`,
that is, a short year and a zero-padded month.

Version tags will be created by maintainers for stable releases.

## Documentation

### Code Documentation

- Use Doxygen-style comments for C++ code
- Document all public APIs
- Include parameter descriptions and return values
- Add usage examples for complex functions

**This is an ongoing effort**

### Example C++ Documentation:
```cpp
/**
 * @brief Implements an explicit update for a physics package
 * @param state State object
 * @param dU Right hand side delta (populated in this function)
 * @param grid Mesh object
 * @param dt_info Struct ocntianing timestep information
 */
void update_explicit(const State *const state, AthelasArray3D<double> dU,
                     const GridStructure &grid,
                     const TimeStepInfo &dt_info) const;
```

## Getting Help

- **Issues:** Use GitHub issues for bug reports and feature requests
- **Discussions:** Use GitHub discussions for questions and general discussion
- **Code review:** Ask questions in PR comments

## License

By contributing to Athelas, you agree that your contributions will be licensed under the same GPL license as the project.

---
