History Output
==============

``Athelas`` writes a scalar history file for time-series quantities such as
total mass, total energy, boundary fluxes, and optional diagnostic trackers.
History output is intended for inexpensive quantities that can be represented
by one value per output time.

The history file is a plain text table. The first line is a header beginning
with ``#``. Each column is described by an integer index followed by the column
name:

.. code-block:: text

   # 0 Time [s] 1 Total Mass [g] 2 Total Energy [erg] ...

All subsequent rows contain floating-point values in the same order. The first
column is always simulation time.

Input Deck
----------

History output is configured under ``config.output.history``:

.. code-block:: lua

   config.output = {
     dt_hdf5 = 1.0e-2,
     history = {
       fn = "model.hst",
       dt = 1.0e-3,
     },
   }

The relevant controls are:

* ``history.fn`` -- history filename. The default is based on the problem name.
* ``history.dt`` -- history output cadence. If omitted, it defaults to a
  fraction of the HDF5 output cadence.

The output directory is controlled by the normal command-line ``-o`` option or
the corresponding parsed ``output.dir`` parameter.

Built-In Quantities
-------------------

The driver registers a standard set of global conservation and boundary
quantities, including:

* ``Total Mass [g]``
* ``Total Energy [erg]``
* ``Total Fluid Energy [erg]``
* ``Total Fluid Momentum [g cm / s]``
* ``Total Internal Energy [erg]``
* ``Total Kinetic Energy [erg]``
* ``Total Momentum [g cm / s]``
* ``Fluid Boundary Energy Rate [erg / s]``

Additional columns are registered when their physics packages are enabled. For
example, radiation runs add radiation energy, radiation momentum, and radiation
boundary energy rate columns. Nickel heating runs add total isotope masses.

Diagnostics can also add history columns. For example, enabling the shock
tracker adds:

* ``Shock Radius [cm]``
* ``Shock Cell Index``
* ``Shock Compression [1 / s]``

See :doc:`diagnostics` for the available tracker quantities and their meanings.

Developer Notes
---------------

History quantities are registered with ``HistoryOutput`` during driver
initialization. The simple path is ``add_quantity``, which registers one column
from a callback:

.. code-block:: cpp

   history.add_quantity("Total Mass [g]", analysis::total_mass);

The callback has the signature:

.. code-block:: cpp

   double(const MeshState &, const Mesh &)

For related columns that share an expensive calculation, use
``add_quantities``. It registers several adjacent columns from one callback:

.. code-block:: cpp

   history.add_quantities(
       {"Shock Radius [cm]", "Shock Cell Index", "Shock Compression [1 / s]"},
       [](const MeshState &state, const Mesh &mesh) {
         const auto shock = diagnostics::detect_shock(state, mesh);
         return std::vector<double>{shock.radius, shock.cell,
                                    shock.compression};
       });

This keeps ``HistoryOutput`` stateless across writes while avoiding duplicate
work within a single row. The callback is evaluated once per history write and
must return exactly one value for each registered column name.

Output-Time State
-----------------

Before a history row is written, the driver refreshes derived quantities and
then runs output-only diagnostic preparation. For example, the optical-depth
profile is filled at output cadence before the photosphere history tracker reads
it. This keeps history callbacks simple: they should read the current
``MeshState`` and ``Mesh`` and return scalar values, but they should not advance
the solution or mutate timestep state.
