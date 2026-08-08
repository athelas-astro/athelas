Building
========

Obtaining the Source Code
-------------------------

``athelas`` uses git submodules for managing several dependencies. 
To make sure you get them all, clone it as

.. code:: bash

   git clone --recursive https://github.com/athelas-astro/athelas.git

or as

.. code:: bash

   git clone https://github.com/athelas-astro/athelas.git
   cd athelas
   git submodule update --init --recursive


Prerequisites
-------------

To build ``athelas``, you need to create a build directly, as in-source builds are not supported.
After cloning the repository,

.. code:: bash

   cd athelas
   mkdir build
   cd build

``Cmake`` is used as the build system. For a standard build:

.. code:: bash

   cmake ..
   make -j4 # or cmake --build .

In the above, you may adjust ``-j4`` to reflect the number of cores available on your machine.
It is not generally a good idea to set this to all of your available cores.

Experimental xmake Build
------------------------

.. note::

   The xmake build capability is a work in progress. CMake remains supported
   while xmake support is introduced incrementally.

Requirements
~~~~~~~~~~~~

Install `xmake <https://xmake.io/#/guide/installation>`__ first; it is not
vendored with Athelas.

Most dependencies come from the git submodules and are built for you. Two must
already be present on the system:

* **Lua**, version 5.4 or earlier. sol2 does not yet accept Lua 5.5, so the
  build searches backwards from 5.4 and takes the first version it finds.
* **HDF5**, with both the C++ and high-level libraries.

Configuring and Building
~~~~~~~~~~~~~~~~~~~~~~~~

Run from the repository root. xmake keeps its output in ``build/``, so
there is no build directory to create:

.. code:: bash

   xmake f
   xmake build athelas

The default mode is ``relwithdebinfo``, which matches the CMake default. Pass
``-m`` to choose another, for example ``xmake f -m release``. The configuration
is sticky: a later ``xmake f`` keeps the mode until you change it.

``xmake build`` with no target builds everything enabled. xmake uses every core
by default, which is not always what you want. Limit it with ``-j``:

.. code:: bash

   xmake build -j4

The first build of the submodule dependencies takes a while, Kokkos and Kokkos
Kernels especially. They are cached afterwards; see `Dependencies and Package
Cache`_.

To clean:

.. code:: bash

   xmake clean         # object files for the default targets
   xmake clean --all   # everything xmake generated
   xmake f -c          # discard the configuration and check it again

Running the Executable
~~~~~~~~~~~~~~~~~~~~~~

``xmake run`` runs ``athelas`` from the directory you invoke it in, so paths
behave as they would if you ran the binary yourself:

.. code:: bash

   xmake run athelas -i inputs/sod.lua

Both the input path and the output location are relative to that directory.
Athelas writes its output to the working directory unless ``-o`` says otherwise,
so to keep results out of the source tree, work from a scratch directory:

.. code:: bash

   mkdir -p ~/runs/sod && cd ~/runs/sod
   xmake -P ~/src/athelas run athelas -i ~/src/athelas/inputs/sod.lua

Relative paths work the same way. From a directory one level below the project
root:

.. code:: bash

   xmake run athelas -i ../inputs/sod.lua

``-w`` overrides this and sets the working directory explicitly. It is resolved
against the project directory rather than your shell, so ``-w .`` always means
the project root, from anywhere in the tree:

.. code:: bash

   xmake run -w . athelas -i inputs/sod.lua

The binary can equally be run on its own:

.. code:: bash

   ./build/linux/x86_64/relwithdebinfo/athelas -i inputs/sod.lua

Build Modes
~~~~~~~~~~~

Athelas supports xmake's standard modes, adds ``relwithdebinfo`` and ``perf``,
and adds five sanitizer modes.

.. list-table::
   :header-rows: 1
   :widths: 18 34 48

   * - Mode
     - Athelas flags
     - Purpose
   * - ``debug``
     - ``-O0 -g``
     - Development, with Athelas and Kokkos debug checks
   * - ``release``
     - ``-O3 -flto -DNDEBUG``
     - Production build
   * - ``releasedbg``
     - xmake built-in plus LTO
     - Optimized build that keeps debug information
   * - ``profile``
     - xmake built-in
     - Instrumented ``gprof`` build using ``-pg``
   * - ``relwithdebinfo``
     - ``-O2 -g -flto -DNDEBUG``
     - CMake ``RelWithDebInfo`` equivalent; the default
   * - ``perf``
     - ``-O3 -g -flto -fno-omit-frame-pointer -DNDEBUG``
     - Sampling profilers, without ``-pg`` instrumentation

The CMake-backed Kokkos, Kokkos Kernels, and Spiner packages are built
``Debug`` in ``debug`` mode, ``RelWithDebInfo`` in ``releasedbg`` and
``relwithdebinfo``, and ``Release`` in every other mode.

.. list-table::
   :header-rows: 1
   :widths: 18 34 48

   * - Mode
     - Sanitizer
     - Notes
   * - ``asan``
     - AddressSanitizer
     - Enables ``ATHELAS_DEBUG`` and the Kokkos debug checks
   * - ``tsan``
     - ThreadSanitizer
     - Cannot be combined with AddressSanitizer
   * - ``ubsan``
     - UndefinedBehaviorSanitizer
     - Also available combined, in ``allsan``
   * - ``lsan``
     - LeakSanitizer
     - Also available combined, in ``allsan``
   * - ``allsan``
     - Address, leak, and undefined behavior
     - Uses ``-O0``; enables ``ATHELAS_DEBUG`` and the Kokkos debug checks

AddressSanitizer and ThreadSanitizer cannot coexist in one executable, so
``allsan`` covers address, leak, and undefined behavior only. Run a separate
``tsan`` configuration for thread races. ``allsan`` also drops to ``-O0``, since
stacked sanitizer reports are hard to read through inlining; a single sanitizer
stays optimized.

Sanitizer instrumentation reaches only the targets built here. The CMake-backed
dependencies stay uninstrumented, so a fault inside one of them may surface only
at its boundary.

``ATHELAS_DEBUG`` and the Kokkos debug and bounds checks are always enabled
together: in ``debug``, ``asan``, and ``allsan``.

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

``xmake f --help`` lists every option with its default, but not the values each
one accepts, so those are given here.

.. list-table::
   :header-rows: 1
   :widths: 26 40 34

   * - Option
     - Values (default first)
     - Purpose
   * - ``--backend``
     - ``openmp``, ``cuda``, ``hip``
     - Kokkos execution backend
   * - ``--par_loop_layout``
     - ``MANUAL1D_LOOP``, ``SIMDFOR_LOOP``, ``MDRANGE_LOOP``, ``TPTTR_LOOP``,
       ``TPTVR_LOOP``, ``TPTTRTVR_LOOP``
     - Layout for the general ``parallel_for`` wrapper
   * - ``--par_loop_flat_layout``
     - ``MANUAL1D_LOOP``, ``SIMDFOR_LOOP``
     - Layout for the one-dimensional ``parallel_for`` wrapper
   * - ``--par_loop_inner_layout``
     - ``SIMDFOR_INNER_LOOP``, ``TVR_INNER_LOOP``
     - Layout for the inner ``parallel_for`` wrapper
   * - ``--unit_tests``
     - ``n``, ``y``
     - Build the Catch2 unit test suite
   * - ``--regression_tests``
     - ``n``, ``y``
     - Register the Python regression test suite

An unrecognized loop-layout value stops configuration with an error naming the
accepted ones, rather than producing a broken header.

.. warning::

   Only the default ``MANUAL1D_LOOP`` layout is exercised. ``MDRANGE_LOOP``
   currently fails to compile, because Kokkos rejects a rank-1
   ``MDRangePolicy``, and the team-policy layouts are untested. This is a
   limitation of the loop wrappers rather than of the build, so the equivalent
   CMake ``PAR_LOOP_LAYOUT`` values behave the same way.

The backend defaults to OpenMP, which is the only one currently supported:

.. code:: bash

   xmake f --backend=openmp

``cuda`` and ``hip`` are accepted names, so their build paths can be added later
without changing the user-facing option, but selecting either stops
configuration with an unsupported-backend error. The xmake build supports Linux
only.

Prefer xmake's toolchain option when switching compilers, so that Athelas and
its locally built dependencies use the same one:

.. code:: bash

   xmake f --toolchain=clang

The equivalents of the CMake options in :ref:`build-opts` are:

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - xmake
     - CMake
   * - ``-m relwithdebinfo``
     - ``-DCMAKE_BUILD_TYPE=RelWithDebInfo``
   * - ``--unit_tests=y``
     - ``-DATHELAS_ENABLE_UNIT_TESTS=ON``
   * - ``--par_loop_layout=<value>``
     - ``-DPAR_LOOP_LAYOUT=<value>``
   * - ``--toolchain=clang``
     - ``-DCMAKE_CXX_COMPILER=clang++``
   * - ``-m asan``
     - ``-DATHELAS_ENABLE_SANITIZERS=ON`` (address and undefined behavior only)

Tests
~~~~~

Unit and regression tests are disabled by default. Either family can be enabled
on its own, or both in one configuration:

.. code:: bash

   xmake f --unit_tests=y --regression_tests=y

The two groups stay independently runnable:

.. code:: bash

   # Run only the Catch2 unit-test executable.
   xmake test -g unit

   # Run only the Python regression suite.
   xmake test -g regression

   # Run one regression case.
   xmake test regression_tests/sod

The regression cases are read from ``test/regression/test_*.py``, so a new test
file needs no build-file change. Each case is named after its module with the
``test_`` prefix removed: ``test_radeq.py`` runs as ``regression_tests/radeq``.

When ``uv`` is available, xmake runs the regression suite in the locked Python
environment under ``scripts/python/athelas_tools``. Otherwise the active Python
installation must provide the regression-test dependencies.

Dependencies and Package Cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dependencies provided as submodules are described by local package recipes under
``xmake/packages/``. These recipes invoke the dependencies' existing CMake
builds, including those for Kokkos and Kokkos Kernels, and expose the resulting
libraries to the native xmake targets. They build out of the submodule directory
and write nothing into it. The built packages are kept under
``build/.packages/``, so a fresh clone always builds them from the
submodule it has.

Each package cache key includes the submodule revision, so a submodule bump does
not reuse the previously built library. Two limits are worth knowing:

* A dependency is cached as one unit, so any submodule change rebuilds it
  completely rather than incrementally. Expect a full Kokkos rebuild after a
  bump.
* The revision is read from the submodule's detached ``HEAD``. A submodule left
  on a branch does not change ``HEAD`` between commits, so a bump there is not
  detected. ``git submodule update`` detaches by default.

The compiler is not part of the cache key beyond the toolchain that xmake itself
records. After a system compiler upgrade with no ``--toolchain`` change, run
``xmake f -c`` so that the dependencies are rebuilt.

Build Layout and Provenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything xmake produces stays under ``build/``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Path
     - Contents
   * - ``build/<plat>/<arch>/<mode>/``
     - Executables
   * - ``build/generated/``
     - Generated headers: ``loop_layout.hpp``, ``lua_schema.hpp``
   * - ``build/provenance/``
     - Generated ``build_info.cpp``
   * - ``build/.packages/``
     - Locally built dependencies

Nothing depends on that directory name, so ``-o`` may be used to build
elsewhere.

Build provenance is checked before every executable build. A new Git commit,
compiler, mode, platform, architecture, or template refreshes
``build/provenance/build_info.cpp``. An otherwise unchanged build leaves it
alone, so the timestamp does not force a relink on every build. Athelas reports
these values at startup and writes them to the ``/metadata/build`` group of its
HDF5 output.

The ``relwithdebinfo`` and ``perf`` modes, the sanitizer modes, and the
provenance rule come from ``xmake/toolkit/``, which carries no Athelas
assumptions and can be copied into another xmake project as-is. See
``xmake/toolkit/README.md``.


.. _build-opts:

Build Options
-------------

The build options explicitly provided by ``athelas`` are:

+---------------------------+---------+------------------------------------------------------+
| Option                    | Default | Comment                                              |
+===========================+=========+======================================================+
| ATHELAS_ENABLE_UNIT_TESTS | OFF     | Build the unit testing suite                         |
+---------------------------+---------+------------------------------------------------------+
| ATHELAS_ENABLE_SANITIZERS | OFF     | Build with address and undefined behavior sanitizers |
+---------------------------+---------+------------------------------------------------------+
| MACHINE_CFG               | None    | Sets a custom config file.                           |
+---------------------------+---------+------------------------------------------------------+

A few other relevant compile options not specific to ``Athelas``:

+---------------------+----------------+---------------------------------------------+
| Option              | Default        | Comment                                     |
+=====================+================+=============================================+
| CMAKE_BUILD_TYPE    | RelWithDebInfo | Used to set the optimization level          |
+---------------------+----------------+---------------------------------------------+
| CMAKE_CXX_COMPILER  | None           | Can be used to specify cxx compiler         |
+---------------------+----------------+---------------------------------------------+
| Kokkos_ARCH_XXXX    | OFF            | Can be used to set the machine architecture |
+---------------------+----------------+---------------------------------------------+

You can see all the kokkos build options
`here <https://github.com/kokkos/kokkos/wiki/Compiling>`__

For example, you might get a debug build of ``Athelas`` with unit tests 
using ``clang++`` as


.. code:: bash

   cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++ -DATHELAS_ENABLE_UNIT_TESTS=On ..
   make -j4 # or cmake --build .

Running
-------

Run ``Athelas`` from the ``build`` directory as

.. code:: bash

   ./athelas -i path/to/input/file.lua -o output/dir

The Lua input decks are in ``athelas/inputs/*.lua``. The output directory is
optional (the default is the current working directory) but the input deck is
required.

Individual values in the input deck can be overridden from the command line
using ``--<dotted.key>=<lua_expr>``:

.. code:: bash

   ./athelas -i ../inputs/marshak.lua --mesh.nx=16 --radiation.newton.tol=1e-12

See :ref:`cli-overrides` for the full syntax.

.. _restart:

Restarting from a checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any ``.ath`` HDF5 dump can be used to resume a run via ``-r``. The
checkpoint embeds the processed input deck (under ``/params``) plus the full
simulation state, so no Lua file is needed. 
``-r`` is mutually exclusive with ``-i``:

.. code:: bash

   ./athelas -r run/sedov_000050.ath
   ./athelas -r run/sedov_000050.ath -o new_run/
   ./athelas -r run/sedov_final.ath  --time.t_end=0.1

Both numbered dumps (``sedov_000050.ath``) and the post-loop ``_final``
file are valid restart sources. CLI ``--<key>=<value>`` overrides apply on
top of the checkpoint's params, parsed the same way as for new runs (see
:ref:`cli-overrides`), so a run can be extended (``--time.t_end=...``,
``--time.nlim=...``) or retuned without editing the checkpoint or
re-running the deck.

A few currently-unsupported configurations to be aware of:

* ``-r`` cannot be combined with ``-i``. If both are passed the run aborts
  with an error.
* CLI ``--<key>=<value>`` overrides on restart only accept scalar values
  (``bool``, ``int``, ``double``, ``string``); table-valued params
  (e.g. ``bc.fluid.dirichlet_values_i``) cannot currently be overridden
  from the CLI on restart.

Dependencies
------------

Submodules
~~~~~~~~~~

-  `Sol`_ is a C++ convenient C++ Lua binding.

- `eigen`_ is a C++ linear algebra library.

-  ``Kokkos`` provides performance portable shared-memory parallelism.
   It allows our loops to be CUDA, OpenMP, or something else. 

.. _Sol: https://sol2.readthedocs.io/en/latest/index.html
.. _eigen: https://github.com/PX4/eigen

External (Required)
~~~~~~~~~~~~~~~~~~~

-  ``cmake`` for building
-  ``Lua`` for input configuration
-  ``hdf5`` for output

Optional
~~~~~~~~

-  ``python3`` for reading output data.
