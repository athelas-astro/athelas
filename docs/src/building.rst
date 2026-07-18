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
