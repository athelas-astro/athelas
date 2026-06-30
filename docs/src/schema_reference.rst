Input Deck Reference
====================

Auto-generated from ``schema.lua``. Each table below documents one section of the ``config`` table.


``config.basis``
----------------

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``nnodes``
     - double
     - —
     - No
     - Number of DG nodes per cell.


``config.bc``
-------------

``fluid``
~~~~~~~~~

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``surface_pressure_o``
     - number
     - —
     - No
     - Prescribed pressure at an outer surface boundary. Defaults to zero.
   * - ``surface_pressure_i``
     - number
     - —
     - No
     - Prescribed pressure at an inner surface boundary. Defaults to zero.
   * - ``bc_o``
     - string
     - —
     - No
     - Outer fluid boundary condition. Options: 'reflecting', 'outflow', 'surface', 'periodic'.
   * - ``bc_i``
     - string
     - —
     - No
     - Inner fluid boundary condition. Options: 'reflecting', 'outflow', 'surface', 'periodic'.


``radiation``
~~~~~~~~~~~~~

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``marshak_incoming_energy_i``
     - number
     - —
     - No
     - Incoming volumetric radiation energy for an inner Marshak boundary.
   * - ``bc_o``
     - string
     - —
     - No
     - Outer radiation boundary condition. Options: 'reflecting', 'interior', 'free_streaming', 'periodic'. 'interior' uses the interior-state physical flux, analogous to fluid outflow but not a no-incoming-characteristics condition.
   * - ``bc_i``
     - string
     - —
     - No
     - Inner radiation boundary condition. Options: 'reflecting', 'interior', 'free_streaming', 'marshak', 'periodic'. 'interior' uses the interior-state physical flux, analogous to fluid outflow but not a no-incoming-characteristics condition.


``config.composition``
----------------------

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``ncomps``
     - int
     - —
     - No
     - Number of composition species.


``config.diagnostics``
----------------------

``photosphere``
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``enabled``
     - bool
     - ``false``
     - No
     - Enable photosphere history tracking from the optical-depth profile.
   * - ``tau``
     - double
     - ``0.6666666666666666``
     - No
     - Optical-depth threshold used by the photosphere tracker. Must be > 0.


``shock``
~~~~~~~~~

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``enabled``
     - bool
     - ``false``
     - No
     - Enable shock history tracking using the strongest velocity compression.


``optical_depth``
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``enabled``
     - bool
     - ``false``
     - No
     - Enable Rosseland optical depth output integrated inward from the outer boundary.


``config.engine``
-----------------

``thermal``
~~~~~~~~~~~

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``operator_split``
     - bool
     - ``false``
     - No
     - Apply engine as operator-split source.
   * - ``enabled``
     - bool
     - —
     - No
     - Enable thermal energy injection engine.
   * - ``energy``
     - double
     - —
     - No
     - Total injected energy in erg.
   * - ``mode``
     - string
     - —
     - No
     - Injection mode. Options: 'direct', 'asymptotic'.
   * - ``mend``
     - double
     - —
     - No
     - Mass coordinate of injection upper boundary in Msun. Must be > 0.
   * - ``tend``
     - double
     - —
     - No
     - End time for energy injection. Must be > 0.


``config.eos``
--------------

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``n``
     - double
     - —
     - No
     - Polytropic index n. Required for polytropic EOS.
   * - ``gamma``
     - double
     - ``1.4``
     - No
     - Adiabatic index. Default: 1.4.
   * - ``type``
     - string
     - —
     - No
     - Equation of state type. Options:. 'ideal', 'paczynski', 'marshak', 'polytropic'.
   * - ``k``
     - double
     - —
     - No
     - Polytropic constant K. Required for polytropic EOS.


``config.fluid``
----------------

``limiter``
~~~~~~~~~~~

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``m_tvb``
     - double
     - ``0.0``
     - No
     - TVB parameter M. Used with minmod limiter.
   * - ``enabled``
     - bool
     - ``true``
     - No
     - Enable slope limiter for fluid.
   * - ``characteristic``
     - bool
     - ``false``
     - No
     - Enable characteristic limiting.
   * - ``type``
     - string
     - ``minmod``
     - No
     - Limiter type. Options: 'minmod', 'moment', 'weno [experimental]'.
   * - ``weno_r``
     - double
     - ``2.0``
     - No
     - WENO smoothness exponent. Must be > 0.
   * - ``tci_val``
     - double
     - —
     - No
     - Troubled-cell indicator threshold.
   * - ``tci_opt``
     - bool
     - ``false``
     - No
     - Enable troubled-cell indicator.
   * - ``gamma_l``
     - double
     - —
     - No
     - WENO left weight. Inferred from gamma_i if omitted.
   * - ``b_tvd``
     - double
     - ``1.0``
     - No
     - TVD parameter b. Used with minmod limiter.
   * - ``gamma_r``
     - double
     - —
     - No
     - WENO right weight. Inferred from gamma_i if omitted.
   * - ``gamma_i``
     - double
     - —
     - No
     - WENO central weight. Required for WENO limiter.


``config.gravity``
------------------

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``operator_split``
     - bool
     - ``false``
     - No
     - Apply gravity as an operator-split source.
   * - ``model``
     - string
     - —
     - No
     - Gravity model. Options: 'constant', 'spherical'.
   * - ``gval``
     - double
     - —
     - No
     - Gravitational acceleration (constant model). Must be > 0.


``config.heating``
------------------

``nickel``
~~~~~~~~~~

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``operator_split``
     - bool
     - ``false``
     - No
     - Apply nickel heating as operator-split source.
   * - ``enabled``
     - bool
     - —
     - No
     - Enable Ni56 decay heating.
   * - ``model``
     - string
     - —
     - No
     - Nickel heating model. Options. 'jeffery' and 'full_trapping'.


``config.ionization``
---------------------

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``fn_ionization``
     - string
     - —
     - No
     - Path to ionization atomic data file.
   * - ``ncomps``
     - int
     - —
     - No
     - Number of species for Saha solver.
   * - ``solver``
     - string
     - ``linear``
     - No
     - Saha solver mode. Options: 'linear', 'log'.
   * - ``fn_degeneracy``
     - string
     - —
     - No
     - Path to degeneracy factors atomic data file.


``config.opacity``
------------------

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``filename``
     - string
     - —
     - No
     - Path to tabular opacity HDF5 file.
   * - ``t_exp``
     - double
     - —
     - No
     - Temperature exponent for powerlaw opacity.
   * - ``type``
     - string
     - —
     - No
     - Opacity model. Options: 'tabular', 'constant', 'powerlaw'.
   * - ``kR``
     - double
     - —
     - No
     - Rosseland mean opacity (cm^2/g). Required for constant and powerlaw.
   * - ``kP``
     - double
     - —
     - No
     - Planck mean opacity (cm^2/g). Required for constant and powerlaw.
   * - ``kP_offset``
     - double
     - ``0.0``
     - No
     - Planck opacity additive offset. Powerlaw only.
   * - ``rho_exp``
     - double
     - —
     - No
     - Density exponent for powerlaw opacity.
   * - ``kR_offset``
     - double
     - ``0.0``
     - No
     - Rosseland opacity additive offset. Powerlaw only.


``floors``
~~~~~~~~~~

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``env_planck``
     - double
     - ``0.01``
     - No
     - Envelope Planck floor. Used with core_envelope model.
   * - ``rosseland``
     - double
     - ``0.001``
     - No
     - Rosseland floor value. Used with constant floor model.
   * - ``planck``
     - double
     - ``0.001``
     - No
     - Planck floor value. Used with constant floor model.
   * - ``env_rosseland``
     - double
     - ``0.01``
     - No
     - Envelope Rosseland floor. Used with core_envelope model.
   * - ``core_planck``
     - double
     - ``0.24``
     - No
     - Core Planck floor. Used with core_envelope model.
   * - ``type``
     - string
     - ``core_envelope``
     - No
     - Opacity floor model. Options: 'core_envelope', 'constant'.
   * - ``core_rosseland``
     - double
     - ``0.24``
     - No
     - Core Rosseland floor. Used with core_envelope model.


``config.output``
-----------------

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``dt_hdf5``
     - double
     - —
     - No
     - Time interval between HDF5 outputs. Default: t_end / 100.
   * - ``dt_fixed``
     - double
     - —
     - No
     - Fixed timestep override. Optional.
   * - ``ncycle_out``
     - double
     - ``100``
     - No
     - Number of cycles between stdout output.
   * - ``dt_growth_frac``
     - double
     - ``1.05``
     - No
     - Initial timestep growth factor. Must be > 1.
   * - ``dt_init``
     - double
     - ``1.000000e-16``
     - No
     - Initial timestep. Must be > 0.


``history``
~~~~~~~~~~~

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``fn``
     - string
     - ``athelas.hst``
     - No
     - History output filename.
   * - ``dt``
     - double
     - —
     - No
     - Time interval between history writes. Default: dt_hdf5 / 10.


``config.physics``
------------------

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``radiation``
     - bool
     - ``false``
     - No
     - Enable radiation transport.
   * - ``engine``
     - bool
     - ``false``
     - No
     - Enable energy injection engine.
   * - ``heating``
     - bool
     - ``false``
     - No
     - Enable nuclear heating sources.
   * - ``ionization``
     - bool
     - ``false``
     - No
     - Enable Saha ionization. Requires composition = true.
   * - ``composition``
     - bool
     - ``false``
     - No
     - Enable multi-species composition.
   * - ``gravity``
     - bool
     - ``false``
     - No
     - Enable gravitational source terms.


``config.problem``
------------------

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``geometry``
     - string
     - —
     - No
     - Domain geometry. Options: 'planar', 'spherical'.
   * - ``nx``
     - double
     - —
     - No
     - Number of grid cells. Must be > 0.
   * - ``xr``
     - double
     - —
     - No
     - Right boundary of the domain. Must be > xl.
   * - ``xl``
     - double
     - —
     - No
     - Left boundary of the domain.
   * - ``name``
     - string
     - —
     - No
     - Unique identifier for this simulation problem.
   * - ``t_end``
     - double
     - —
     - No
     - End time of the simulation.
   * - ``params``
     - —
     - —
     - No
     - Problem-specific parameters. Validated by the problem generator.
   * - ``grid_type``
     - string
     - —
     - No
     - Grid spacing type. Options: 'uniform', 'logarithmic'.
   * - ``nlim``
     - double
     - ``-1``
     - No
     - Maximum double of cycles. -1 for unlimited.
   * - ``cfl``
     - double
     - —
     - No
     - CFL double for timestep control. Must be > 0.


``config.radiation``
--------------------

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``ap_coefficient``
     - double
     - ``0.0``
     - No
     - Coefficient C in the AP LLF damping factor 1 / (1 + C tau). C = 0 recovers standard LLF (no damping); C > 0 enables the optical-depth correction. Must be >= 0.
   * - ``discretization``
     - string
     - —
     - No
     - Spatial discretization of the transport term. Options: 'implicit' or 'explicit'.


``timestep``
~~~~~~~~~~~~

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``max_change_f``
     - double
     - —
     - No
     - Maximum allowed absolute change in radiation reduced flux. Timestep control for implicit transport.
   * - ``max_fractional_change_e``
     - double
     - —
     - No
     - Maximum allowed fractional change in radiation energy. Timestep control for implicit transport.
   * - ``energy_change_scale``
     - double
     - ``1.000000e-10``
     - No
     - Specific radiation energy scale used in the implicit transport relative-change timestep controller.


``limiter``
~~~~~~~~~~~

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``m_tvb``
     - double
     - ``0.0``
     - No
     - TVB parameter M. Used with minmod limiter.
   * - ``enabled``
     - bool
     - ``true``
     - No
     - Enable slope limiter for radiation.
   * - ``characteristic``
     - bool
     - ``false``
     - No
     - Enable characteristic limiting. Currently unsupported for radiation.
   * - ``type``
     - string
     - ``minmod``
     - No
     - Limiter type. Options: 'minmod', 'moment', 'weno [experimental]'.
   * - ``weno_r``
     - double
     - ``2.0``
     - No
     - WENO smoothness exponent. Must be > 0.
   * - ``tci_val``
     - double
     - —
     - No
     - Troubled-cell indicator threshold.
   * - ``tci_opt``
     - bool
     - ``false``
     - No
     - Enable troubled-cell indicator.
   * - ``gamma_l``
     - double
     - —
     - No
     - WENO left weight. Inferred from gamma_i if omitted.
   * - ``b_tvd``
     - double
     - ``1.0``
     - No
     - TVD parameter b. Used with minmod limiter.
   * - ``gamma_r``
     - double
     - —
     - No
     - WENO right weight. Inferred from gamma_i if omitted.
   * - ``gamma_i``
     - double
     - —
     - No
     - WENO central weight. Required for WENO limiter.


``newton``
~~~~~~~~~~

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``max_iter``
     - int
     - ``10``
     - No
     - Maximum Newton iterations for implicit transport solve.
   * - ``tol``
     - double
     - ``1.000000e-08``
     - No
     - Convergence tolerance for implicit transport Newton iteration.


``config.time``
---------------

.. list-table::
   :widths: 10 8 12 20 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Required
     - Description
   * - ``integrator``
     - string
     - —
     - No
     - Time integration method. E.g. 'IMEX_SSPRK11', 'IMEX_ARK32_ESDIRK'.

