.. _Catch2: https://github.com/catchorg/Catch2
.. _singularity-eos: https://lanl.github.io/singularity-eos
.. _JOSS: https://joss.theoj.org/about#ai-policy
.. _cc: https://www.conventionalcommits.org/en/v1.0.0/
.. _git-cliff: https://git-cliff.org/
.. _semver: https://semver.org/

Contributing
=============================

Thank you for your interest in contributing to ``Athelas``! 
This document provides current guidelines and information for contributors. 
As a general rule, these are not not set in stone but subject to possible 
change, dicussion, and revision.

Overview
--------

To contribute to ``athelas``, feel free to submit a pull
request or open an issue.

1. Create a new fork or branch off of ``main`` where your changes can be made.
2. When ready, create a pull request, describe what you've done
   and ensure the branch has no conflicts.
3. At least one Maintainer will review the PR.
4. Once comments/feedback is addressed, the PR will be merged into the
   main branch. User-facing changes are collected automatically from the
   commit history for the next release.
5. At present, Releases (with a git version tag) for the ``main`` branch
   of ``athelas`` will occur at a 6 to 12 month cadence or following
   implementation of a major enhancement or capability to the code base.

Branching Strategy
``````````````````
* Create feature branches from the main development branch.
* To keep things orderly, contributors are encouraged to create branches 
  starting with their username or initials followed by a "/" and ending 
  with a brief description, e.g., ``blb/implicit_transport``.
* Keep branches focused on a single feature or fix.

Please try to keep PRs small and targeted, as this makes the reviewing 
process simpler, faster, and helps avoid things slipping into the codebase 
that we otherwise might not want.

Even if your PR is not ready you may feel free to open a draft PR marked with 
"[WIP]" in the title. These are not merged until [WIP] is removed and can be 
convenient as GitHub's tests will launch even on the draft PR.

Commit Messages
````````````````

The commit that lands on ``main`` must use the `Conventional Commits <cc_>`_
format. In particular, use a Conventional Commit title for pull requests that
will be squash-merged:

.. code-block:: text

   type(optional-scope): short description

The changelog publishes the following user-facing types:

* ``feat``: new features
* ``fix``: bug fixes
* ``perf``: performance improvements
* ``docs``: documentation changes
* ``refactor``: code changes that neither add a feature nor fix a bug

The types ``test``, ``style``, ``chore``, ``build``, and ``ci`` are valid for
internal work but are omitted from release notes. Scopes are optional; for
example, ``fix(eos): handle the energy floor``. Mark a breaking change with
``!`` before the colon and explain it in a ``BREAKING CHANGE:`` footer:

.. code-block:: text

   feat(io)!: replace the checkpoint layout

   BREAKING CHANGE: Checkpoints written by earlier releases cannot be read.

Pull request protocol
----------------------

.. note::

   All code review will be conducted by humans and not by agentic AI.

When submitting a pull request, there is a default template that is automatically
populated. The pull request should sufficiently summarize all changes.
As necessary, tests should be added for new features of bugs fixed.

Before a pull request will be merged, the code should be formatted. We
use clang-format for this, pinned to version 20.1.0.
The script ``scripts/bash/format.sh`` will apply ``clang-format``
to C++ source files in the repository as well as ``ruff`` to python files, if available.
The script takes three CLI arguments
that may be useful, ``CFM``, which can be set to the path for your
clang-format binary, ``PFM`` which can point to ``ruff``, 
and ``VERBOSE``, which if set to ``1`` adds useful output. For example:

.. code-block:: bash

    CFM=clang-format VERBOSE=1 ./scripts/bash/format.sh

In order for a pull request to merge, we require:

- Provide a thorough summary of changes, especially if they are breaking changes
  or new features.
- Obey style guidleines (format with ``clang-format`` and pass the necessary test)
- Pass the existing test suite
- Have at least one approval from a Maintainer
- If generative or agentic AI was used, add an appropriate disclosure (:ref:`ai`).
- If applicable:

  - Write new tests for new features or bugs
  - Include or update documentation.

Versioning and Releases
-----------------------

Athelas uses `Semantic Versioning <semver_>`_.
Release notes are generated with `git-cliff <git-cliff_>`_. Contributors do
not edit ``CHANGELOG.md`` for individual pull requests.

.. note::

   The versioning and release workflow is manual. It is not expected to be 
   a large burden, but in the future the process may be partially automated.

Release preparation
```````````````````

This workflow prepares a stable release. It does not currently define release
candidate tags such as ``-rc.1``.

1. Start from an up-to-date ``main`` with no unrelated changes. Inspect the
   changes since the latest release, choose the new version, and create a
   dedicated release branch:

   .. code-block:: bash

      git switch main
      git pull --ff-only
      git status --short
      git log "$(git describe --tags --abbrev=0)"..HEAD --oneline
      release_version=0.11.0
      previous_version=0.10.0
      release_branch="blb/release-v${release_version}"
      git switch -c "${release_branch}"

   Replace the example versions and ``blb`` branch prefix as appropriate.

2. Update the synchronized version in the following locations:

   * ``CMakeLists.txt``
   * ``docs/conf.py``
   * ``scripts/python/athelas_tools/pyproject.toml``
   * ``scripts/python/athelas_tools/uv.lock``
   * ``scripts/python/athelas_tools/src/athelas_tools/__init__.py``

   Regenerate the lockfile rather than editing it by hand, then confirm the old
   version is gone and the new version appears in all version sources:

   .. code-block:: bash

      (cd scripts/python/athelas_tools && uv lock)
      rg "${previous_version}" CMakeLists.txt docs/conf.py scripts/python/athelas_tools
      rg "${release_version}" CMakeLists.txt docs/conf.py scripts/python/athelas_tools

   The first search should produce no project-version matches. Dependency
   versions in ``uv.lock`` are unrelated and may coincidentally match.

3. Preview the exact dated changelog entry without modifying any files. Review
   its categories, PR links, and breaking-change notices:

   .. code-block:: bash

      git cliff --unreleased --tag "v${release_version}"

4. Once the preview is correct, prepend the entry to ``CHANGELOG.md`` exactly
   once:

   .. code-block:: bash

      git cliff --unreleased --tag "v${release_version}" \
        --prepend CHANGELOG.md

5. Inspect the complete release diff, commit it, push the branch, and open a
   pull request. Do not create the release tag from the branch:

   .. code-block:: bash

      git diff
      git status --short
      git add CHANGELOG.md CMakeLists.txt docs/conf.py \
        scripts/python/athelas_tools/pyproject.toml \
        scripts/python/athelas_tools/uv.lock \
        scripts/python/athelas_tools/src/athelas_tools/__init__.py
      git commit -m "chore(release): prepare v${release_version}"
      git push -u origin "${release_branch}"

   The release PR must pass the normal review and CI requirements. Check that
   the PR contains only the version updates and generated changelog entry.

Publishing the release
``````````````````````

After the release PR is merged, a maintainer tags the resulting commit on
``main`` and pushes only that tag:

.. code-block:: bash

   release_version=0.11.0
   git switch main
   git pull --ff-only
   git tag -a "v${release_version}" \
     -m "Athelas v${release_version}"
   git show --no-patch "v${release_version}"
   git push origin "v${release_version}"

Create the corresponding GitHub release from the tag and use the matching
``CHANGELOG.md`` section as its release notes.

Test Suite
----------

Several sets of tests are triggered on a pull request: a static format
check, a docs buld, build on multiple compilers, and a suite of unit and regression tests.
These are run through GitHub's CI infrastructure. These tests
help ensure that modifications to the code do not break existing capabilities
and ensure a consistent code style.

Adding Tests
````````````

There are two primary categories of tests written in ``athelas``:
unit tests and regression tests.

Unit
^^^^

Unit tests live in ``test/unit/``. They are implemented using the
`Catch2`_ unit testing framework. They are integrated with ``cmake``
and can be run, when enabled, with ``ctest``. ``Athelas`` must be built
with unit tests enabled (see :ref:`build-opts`.).

Regression
^^^^^^^^^^
Regression tests run existing simulations and test against saved output
in order to verify sustained capabilities.
They are implemented in Python in
``test/regression/``. To run the tests you will need a Python environment with
at least ``numpy`` and ``h5py``. Tests can be ran manually as, e.g.,

.. code-block:: bash

   python run_regression_tests.py --test test_marshak -e ../../build/athelas


This will use an existing build of ``athelas`` located in ``build/athelas``.
Without the ``-e`` option the runner will build ``Athelas`` locally.
Each script ``test_problem.py`` has a correspodning "gold file" ``problem.gold``.
The gold files contain the gold standard data that the output of the regression test
is compared against. To generate new gold data, for example if a change is implemented
that changes the behavior of a test (not erroneously) or a new test is created, run the test
script with the ``--upgold`` option. This will create or update the corresponding ``.gold`` file.
To add a new test:

1. Create a new test script.

   - Copy an existing test module, e.g., ``test_sod.py``
   - Set the ``variables`` list to contain the quantities to test against
   - Optionally, change the ``compression_factor`` to avoid overly large gold files.
2. Run the script with the ``--upgold`` option
3. Commit the test script and gold file
4. Update the CI to include the new test (``athelas/.github/workflows/regression_testing.yml``)


Expectations for code review
-----------------------------
.. note::
   Much of what follows is adapted from `singularity-eos`_.

From the perspective of the contributor
````````````````````````````````````````

Code review is an integral part of the development process
for ``athelas``. You can expect at least one, perhaps many,
core developers to read your code and offer suggestions.
You should treat this much like scientific or academic peer review.
You should listen to suggestions but also feel entitled to push back
if you believe the suggestions or comments are incorrect or
are requesting too much effort.

Reviewers may offer conflicting advice, if this is the case, it's an
opportunity to open a discussion and communally arrive at a good
approach. You should feel empowered to argue for which of the
solutions you prefer or to suggest a compromise. If you
don't feel strongly, that's fine too, but it's best to say so to keep
the lines of communication open.

Big contributions may be difficult to review in one piece and you may
be requested to split your pull request into two or more separate
contributions. You may also receive many "nitpicky" comments about
code style or structure. These comments help keep a broad codebase
with many contributors uniform in style and maintainable with
consistent expectations accross the code base. While there is no
formal style guide for now, the regular contributors have a sense for
the broad style of the project. You should take these stylistic and
"nitpicky" suggestions seriously, but you should also feel free to
push back.

As with any creative endeavor, we put a lot of ourselves into our
code. It can be painful to receive criticism on your contribution and
easy to take it personally. While you should resist the urge to take
offense, it is also partly code reviewer's responsiblity to create a
constructive environment, as discussed below.

Expectations of code reviewers
````````````````````````````````

A good code review is similar to a good review of an academic paper:
it builds a contribution up, rather than tearing it
down. Here are a few rules to keep code reviews constructive and
congenial:

* You should take the time needed to review a contribution and offer
  meaningful advice. Unless a contribution is very small, limit
  the times you simply click "approve" with a "looks good to me."

* You should keep your comments constructive. For example, rather than
  saying "this pattern is bad," try saying "at this point, you may
  want to try this other pattern."

* Avoid language that can be misconstrued, even if it's common
  notation in the commnunity. For example, avoid phrases like "code
  smell."

* Explain why you make a suggestion. In addition to saying "try X
  instead of Y" explain why you like pattern X more than pattern Y.

* A contributor may push back on your suggestion. Be open to the
  possibility that you're either asking too much or are incorrect in
  this instance. Code review is an opportunity for everyone to learn.

* Don't just highlight what you don't like. Also highlight the parts
  of the pull request you do like and thank the contributor for their
  effort.

General principle for everyone
```````````````````````````````

It's hard to convey tone in text correspondance. Try to read what
others write favorably and try to write in such a way that your tone
can't be mis-interpreted as malicious.

.. _ai:

AI Policy
---------

``Athelas`` strives to maintain a high standard of code style, design, 
and functionality. Generative AI is rapidly changing the way code is written.
As a scientific tool for the community, we must ensure that the code base 
stays maintainable, understandable, clean, and correct.
The following guidelines apply to contributed code (pull requests) 
as well as issues.

.. note::

   The following is inspired by the generative AI policy used in the 
   Journal of Open Source Software (`JOSS`_).

The use of generative AI for code submissions is allowed, however, 
all such contributions must be disclosed in the pull request. This includes:

* Tool use: The tools/models used (and versions) and where 
  they were used (code, text, docs, etc).
* The nature and scope of assistance: e.g., code generation, refactoring, 
  test scaffolding, documenting, drafting, prototyping.
* Confirmation of review: Authors must assert that human contributors reviewed, 
  edited, validated all AI-assisted outputs and made the core design decisions.

Failure to provide a disclosure may result in closing of the pull request 
and rejection of the contributions. Repeated offenses may result in the user 
being blocked from the repository.

All code review, including conversation, will be conducted by humans.
