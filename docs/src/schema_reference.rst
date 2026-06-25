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
   * - ``bc_o``
     - string
     - —
     - No
     - Outer radiation boundary condition. Options: 'reflecting', 'interior', 'free_streaming', 'periodic'. 'interior' uses the interior-state physical flux, analogous to fluid outflow but not a no-incoming-characteristics condition.
   * - ``marshak_incoming_energy_i``
     - number
     - —
     - No
     - Incoming volumetric radiation energy for an inner Marshak boundary.
   * - ``bc_i``
     - string
     - —
     - No
     - Inner radiation boundary condition. Options: 'reflecting', 'interior', 'free_streaming', 'marshak', 'periodic'. 'interior' uses the interior-state physical flux, analogous to fluid outflow but not a no-incoming-characteristics condition.


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
   * - ``bc_o``
     - string
     - —
     - No
     - Outer fluid boundary condition. Options: 'reflecting', 'outflow', 'surface', 'periodic'.
   * - ``surface_pressure_i``
     - number
     - —
     - No
     - Prescribed pressure at an inner surface boundary. Defaults to zero.
   * - ``bc_i``
     - string
     - —
     - No
     - Inner fluid boundary condition. Options: 'reflecting', 'outflow', 'surface', 'periodic'.


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
   * - ``mend``
     - double
     - —
     - No
     - Mass coordinate of injection upper boundary in Msun. Must be > 0.
   * - ``mode``
     - string
     - —
     - No
     - Injection mode. Options: 'direct', 'asymptotic'.
   * - ``operator_split``
     - bool
     - ``false``
     - No
     - Apply engine as operator-split source.
   * - ``energy``
     - double
     - —
     - No
     - Total injected energy in erg.
   * - ``enabled``
     - bool
     - —
     - No
     - Enable thermal energy injection engine.
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
   * - ``k``
     - double
     - —
     - No
     - Polytropic constant K. Required for polytropic EOS.
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
   * - ``characteristic``
     - bool
     - ``false``
     - No
     - Enable characteristic limiting.
   * - ``enabled``
     - bool
     - ``true``
     - No
     - Enable slope limiter for fluid.
   * - ``gamma_i``
     - double
     - —
     - No
     - WENO central weight. Required for WENO limiter.
   * - ``weno_r``
     - double
     - ``2.0``
     - No
     - WENO smoothness exponent. Must be > 0.
   * - ``type``
     - string
     - ``minmod``
     - No
     - Limiter type. Options: 'minmod', 'moment', 'weno [experimental]'.
   * - ``tci_opt``
     - bool
     - ``false``
     - No
     - Enable troubled-cell indicator.
   * - ``tci_val``
     - double
     - —
     - No
     - Troubled-cell indicator threshold.
   * - ``m_tvb``
     - double
     - ``0.0``
     - No
     - TVB parameter M. Used with minmod limiter.
   * - ``gamma_l``
     - double
     - —
     - No
     - WENO left weight. Inferred from gamma_i if omitted.


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
   * - ``model``
     - string
     - —
     - No
     - Nickel heating model. Options. 'jeffery' and 'full_trapping'.
   * - ``enabled``
     - bool
     - —
     - No
     - Enable Ni56 decay heating.


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
   * - ``solver``
     - string
     - ``linear``
     - No
     - Saha solver mode. Options: 'linear', 'log'.
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
   * - ``kR_offset``
     - double
     - ``0.0``
     - No
     - Rosseland opacity additive offset. Powerlaw only.
   * - ``kR``
     - double
     - —
     - No
     - Rosseland mean opacity (cm^2/g). Required for constant and powerlaw.


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
   * - ``core_rosseland``
     - double
     - ``0.24``
     - No
     - Core Rosseland floor. Used with core_envelope model.
   * - ``type``
     - string
     - ``core_envelope``
     - No
     - Opacity floor model. Options: 'core_envelope', 'constant'.
   * - ``core_planck``
     - double
     - ``0.24``
     - No
     - Core Planck floor. Used with core_envelope model.
   * - ``env_rosseland``
     - double
     - ``0.01``
     - No
     - Envelope Rosseland floor. Used with core_envelope model.
   * - ``planck``
     - double
     - ``0.001``
     - No
     - Planck floor value. Used with constant floor model.
   * - ``rosseland``
     - double
     - ``0.001``
     - No
     - Rosseland floor value. Used with constant floor model.


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
   * - ``dt_growth_frac``
     - double
     - ``1.05``
     - No
     - Initial timestep growth factor. Must be > 1.
   * - ``dt_fixed``
     - double
     - —
     - No
     - Fixed timestep override. Optional.
   * - ``dt_hdf5``
     - double
     - —
     - No
     - Time interval between HDF5 outputs. Default: t_end / 100.
   * - ``dt_init``
     - double
     - ``1.000000e-16``
     - No
     - Initial timestep. Must be > 0.
   * - ``ncycle_out``
     - double
     - ``100``
     - No
     - Number of cycles between stdout output.


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
   * - ``dt``
     - double
     - —
     - No
     - Time interval between history writes. Default: dt_hdf5 / 10.
   * - ``fn``
     - string
     - ``athelas.hst``
     - No
     - History output filename.


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
   * - ``radiation``
     - bool
     - ``false``
     - No
     - Enable radiation transport.
   * - ``gravity``
     - bool
     - ``false``
     - No
     - Enable gravitational source terms.
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
   * - ``xl``
     - double
     - —
     - No
     - Left boundary of the domain.
   * - ``nx``
     - double
     - —
     - No
     - Number of grid cells. Must be > 0.
   * - ``geometry``
     - string
     - —
     - No
     - Domain geometry. Options: 'planar', 'spherical'.
   * - ``params``
     - —
     - —
     - No
     - Problem-specific parameters. Validated by the problem generator.
   * - ``t_end``
     - double
     - —
     - No
     - End time of the simulation.
   * - ``name``
     - string
     - —
     - No
     - Unique identifier for this simulation problem.
   * - ``xr``
     - double
     - —
     - No
     - Right boundary of the domain. Must be > xl.
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
   * - ``grid_type``
     - string
     - —
     - No
     - Grid spacing type. Options: 'uniform', 'logarithmic'.


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
   * - ``max_fractional_change_e``
     - double
     - —
     - No
     - Maximum allowed fractional change in radiation energy. Timestep control for implicit transport.
   * - ``max_change_f``
     - double
     - —
     - No
     - Maximum allowed absolute change in radiation reduced flux. Timestep control for implicit transport.
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
   * - ``characteristic``
     - bool
     - ``false``
     - No
     - Enable characteristic limiting. Currently unsupported for radiation.
   * - ``enabled``
     - bool
     - ``true``
     - No
     - Enable slope limiter for radiation.
   * - ``gamma_i``
     - double
     - —
     - No
     - WENO central weight. Required for WENO limiter.
   * - ``weno_r``
     - double
     - ``2.0``
     - No
     - WENO smoothness exponent. Must be > 0.
   * - ``type``
     - string
     - ``minmod``
     - No
     - Limiter type. Options: 'minmod', 'moment', 'weno [experimental]'.
   * - ``tci_opt``
     - bool
     - ``false``
     - No
     - Enable troubled-cell indicator.
   * - ``tci_val``
     - double
     - —
     - No
     - Troubled-cell indicator threshold.
   * - ``m_tvb``
     - double
     - ``0.0``
     - No
     - TVB parameter M. Used with minmod limiter.
   * - ``gamma_l``
     - double
     - —
     - No
     - WENO left weight. Inferred from gamma_i if omitted.


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

