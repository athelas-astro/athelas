.. _Catch2: https://github.com/catchorg/Catch2
.. _singularity-eos: https://lanl.github.io/singularity-eos

Contributing
=============================

To contribute to ``athelas``, feel free to submit a pull
request or open an issue.

1. Create a new fork or branch of off ``main`` where your changes can be made.
2. When ready, create a pull request, describe what you've done
   and ensure the branch has no conflicts.
3. At least one Maintainer will review the PR.
4. Once comments/feedback is addressed, the PR will be merged into the
   main branch and changes will be added to list of changes in the next
   release.
5. At present, Releases (with a git version tag) for the ``main`` branch
   of ``athelas`` will occur at a 6 to 12 month cadence or following
   implementation of a major enhancement or capability to the code base.


Pull request protocol
----------------------

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
- Update `CHANGELOG.md`
- If applicable:

  - Write new tests for new features or bugs
  - Include or update documentation in ``docs/``

Test Suite
----------

Several sets of tests are triggered on a pull request: a static format
check, a docs buld, build on multiple compilers, and a suite of unit and regression tests.
These are run through github's CI infrastructure. These tests
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

A good code review builds a contribution up, rather than tearing it
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
