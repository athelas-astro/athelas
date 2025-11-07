Building
========

Obtaining the Source Code
-------------------------

``athelas`` uses git submodules for managing several dependencies. 
To make sure you get them all, clone it as

.. code:: bash

   git clone --recursive git@github.com:athelas-astro/athelas.git

or as

.. code:: bash

   git clone git@github.com:athelas-astro/athelas.git
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

   ./athelas -i path/to/input/file.toml

The input files are in ``athelas/inputs/*``. 

Dependencies
------------

Submodules
~~~~~~~~~~

-  `tomlplusplus`_ is a C++ file parser for toml file format. 
   Used for input decks and data.

- `eigen`_ is a C++ linear algebra library.

-  ``Kokkos`` provides performance portable shared-memory parallelism.
   It allows our loops to be CUDA, OpenMP, or something else. 

.. _tomlplusplus: https://github.com/marzer/tomlplusplus 
.. _eigen: https://github.com/PX4/eigen

External (Required)
~~~~~~~~~~~~~~~~~~~

-  ``cmake`` for building
-  ``hdf5`` for output

Optional
~~~~~~~~

-  ``python3`` for visualization
