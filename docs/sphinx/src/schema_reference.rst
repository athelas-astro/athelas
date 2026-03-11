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
     - Yes
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
   * - ``bc_o``
     - string
     - —
     - Yes
     - Outer fluid boundary condition. Options: 'reflecting', 'outflow', 'dirichlet', 'periodic'.
   * - ``bc_i``
     - string
     - —
     - Yes
     - Inner fluid boundary condition. Options: 'reflecting', 'outflow', 'dirichlet', 'periodic'.
   * - ``dirichlet_values_i``
     - array
     - —
     - When ``bc.fluid.bc_i`` = ``dirichlet``
     - Dirichlet state values at inner boundary.
   * - ``dirichlet_values_o``
     - array
     - —
     - When ``bc.fluid.bc_o`` = ``dirichlet``
     - Dirichlet state values at outer boundary.


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
     - When ``physics.radiation`` is true
     - Outer radiation boundary condition. Options: 'reflecting', 'outflow', 'dirichlet'.
   * - ``bc_i``
     - string
     - —
     - When ``physics.radiation`` is true
     - Inner radiation boundary condition. Options: 'reflecting', 'outflow', 'marshak', 'dirichlet'.
   * - ``dirichlet_values_i``
     - array
     - —
     - When ``bc.radiation.bc_i`` = ``dirichlet``
     - Dirichlet state values at inner boundary.
   * - ``dirichlet_values_o``
     - array
     - —
     - When ``bc.radiation.bc_o`` = ``dirichlet``
     - Dirichlet state values at outer boundary.


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
     - When ``physics.composition`` is true
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
     - When ``physics.engine`` is true
     - Mass coordinate of injection upper boundary in Msun. Must be > 0.
   * - ``mode``
     - string
     - —
     - When ``physics.engine`` is true
     - Injection mode. Options: 'direct', 'asymptotic'.
   * - ``energy``
     - double
     - —
     - When ``physics.engine`` is true
     - Total injected energy in erg.
   * - ``enabled``
     - bool
     - —
     - When ``physics.engine`` is true
     - Enable thermal energy injection engine.
   * - ``tend``
     - double
     - —
     - When ``physics.engine`` is true
     - End time for energy injection. Must be > 0.
   * - ``operator_split``
     - bool
     - ``false``
     - No
     - Apply engine as operator-split source.


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
   * - ``type``
     - string
     - —
     - Yes
     - Equation of state type. Options:. 'ideal', 'paczynski', 'marshak', 'polytropic'.
   * - ``gamma``
     - double
     - ``1.4``
     - No
     - Adiabatic index. Default: 1.4.
   * - ``n``
     - double
     - —
     - When ``eos.type`` = ``polytropic``
     - Polytropic index n. Required for polytropic EOS.
   * - ``k``
     - double
     - —
     - When ``eos.type`` = ``polytropic``
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
   * - ``type``
     - string
     - ``minmod``
     - No
     - Limiter type. Options: 'minmod', 'weno [experimental]'.
   * - ``b_tvd``
     - double
     - ``1.0``
     - No
     - TVD parameter b. Used with minmod limiter.
   * - ``tci_val``
     - double
     - —
     - When ``fluid.limiter.tci_opt`` is true
     - Troubled-cell indicator threshold.
   * - ``weno_r``
     - double
     - ``2.0``
     - No
     - WENO smoothness exponent. Must be > 0.
   * - ``gamma_i``
     - double
     - —
     - When ``fluid.limiter.type`` = ``weno``
     - WENO central weight. Required for WENO limiter.
   * - ``do_limiter``
     - bool
     - ``true``
     - No
     - Enable slope limiter for fluid.
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
   * - ``tci_opt``
     - bool
     - ``false``
     - No
     - Enable troubled-cell indicator.


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
   * - ``gval``
     - double
     - ``1.0``
     - No
     - Gravitational acceleration (constant model). Must be > 0.
   * - ``model``
     - string
     - ``constant``
     - No
     - Gravity model. Options: 'constant', 'spherical'.
   * - ``operator_split``
     - bool
     - ``false``
     - No
     - Apply gravity as an operator-split source.


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
   * - ``enabled``
     - bool
     - —
     - When ``physics.heating`` is true
     - Enable Ni56 decay heating.
   * - ``model``
     - string
     - —
     - When ``physics.heating`` is true
     - Nickel heating model. Options. 'jeffery' and 'full_trapping'.
   * - ``operator_split``
     - bool
     - ``false``
     - No
     - Apply nickel heating as operator-split source.


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
     - When ``physics.ionization`` is true
     - Path to ionization atomic data file.
   * - ``fn_degeneracy``
     - string
     - —
     - When ``physics.ionization`` is true
     - Path to degeneracy factors atomic data file.
   * - ``ncomps``
     - int
     - —
     - When ``physics.ionization`` is true
     - Number of species for Saha solver.


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
   * - ``type``
     - string
     - —
     - When ``physics.radiation`` is true
     - Opacity model. Options: 'tabular', 'constant', 'powerlaw'.
   * - ``kP``
     - double
     - —
     - When ``opacity.type`` = ``constant``
     - Planck mean opacity (cm^2/g). Required for constant and powerlaw.
   * - ``kR_offset``
     - double
     - ``0.0``
     - No
     - Rosseland opacity additive offset. Powerlaw only.
   * - ``t_exp``
     - double
     - —
     - When ``opacity.type`` = ``powerlaw``
     - Temperature exponent for powerlaw opacity.
   * - ``kR``
     - double
     - —
     - When ``opacity.type`` = ``constant``
     - Rosseland mean opacity (cm^2/g). Required for constant and powerlaw.
   * - ``rho_exp``
     - double
     - —
     - When ``opacity.type`` = ``powerlaw``
     - Density exponent for powerlaw opacity.
   * - ``filename``
     - string
     - —
     - When ``opacity.type`` = ``tabular``
     - Path to tabular opacity HDF5 file.
   * - ``kP_offset``
     - double
     - ``0.0``
     - No
     - Planck opacity additive offset. Powerlaw only.


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
   * - ``type``
     - string
     - ``core_envelope``
     - No
     - Opacity floor model. Options: 'core_envelope', 'constant'.
   * - ``env_planck``
     - double
     - ``0.01``
     - No
     - Envelope Planck floor. Used with core_envelope model.
   * - ``planck``
     - double
     - ``0.001``
     - No
     - Planck floor value. Used with constant floor model.
   * - ``core_rosseland``
     - double
     - ``0.24``
     - No
     - Core Rosseland floor. Used with core_envelope model.
   * - ``env_rosseland``
     - double
     - ``0.01``
     - No
     - Envelope Rosseland floor. Used with core_envelope model.
   * - ``rosseland``
     - double
     - ``0.001``
     - No
     - Rosseland floor value. Used with constant floor model.
   * - ``core_planck``
     - double
     - ``0.24``
     - No
     - Core Planck floor. Used with core_envelope model.


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
   * - ``dt_init_frac``
     - double
     - ``1.05``
     - No
     - Initial timestep growth factor. Must be > 1.
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
   * - ``dt_init``
     - double
     - ``1.000000e-16``
     - No
     - Initial timestep. Must be > 0.
   * - ``ncycle_out``
     - double
     - ``1``
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
   * - ``radiation``
     - bool
     - ``false``
     - No
     - Enable radiation transport.
   * - ``heating``
     - bool
     - ``false``
     - No
     - Enable nuclear heating sources.
   * - ``gravity``
     - bool
     - ``false``
     - No
     - Enable gravitational source terms.
   * - ``composition``
     - bool
     - ``false``
     - No
     - Enable multi-species composition.
   * - ``engine``
     - bool
     - ``false``
     - No
     - Enable energy injection engine.
   * - ``ionization``
     - bool
     - ``false``
     - No
     - Enable Saha ionization. Requires composition = true.


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
   * - ``params``
     - table
     - —
     - No
     - Problem-specific parameters. Validated by the problem generator.
   * - ``name``
     - string
     - —
     - Yes
     - Unique identifier for this simulation problem.
   * - ``geometry``
     - string
     - —
     - Yes
     - Domain geometry. Options: 'planar', 'spherical'.
   * - ``xr``
     - double
     - —
     - Yes
     - Right boundary of the domain. Must be > xl.
   * - ``nx``
     - double
     - —
     - Yes
     - Number of grid cells. Must be > 0.
   * - ``grid_type``
     - string
     - —
     - Yes
     - Grid spacing type. Options: 'uniform', 'logarithmic'.
   * - ``cfl``
     - double
     - —
     - Yes
     - CFL double for timestep control. Must be > 0.
   * - ``t_end``
     - double
     - —
     - Yes
     - End time of the simulation.
   * - ``xl``
     - double
     - —
     - Yes
     - Left boundary of the domain.
   * - ``nlim``
     - double
     - ``-1``
     - No
     - Maximum double of cycles. -1 for unlimited.


``config.radiation``
--------------------

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
   * - ``type``
     - string
     - ``minmod``
     - No
     - Limiter type. Options: 'minmod', 'weno'.
   * - ``b_tvd``
     - double
     - ``1.0``
     - No
     - TVD parameter b. Used with minmod limiter.
   * - ``tci_val``
     - double
     - —
     - When ``radiation.limiter.tci_opt`` is true
     - Troubled-cell indicator threshold.
   * - ``weno_r``
     - double
     - ``2.0``
     - No
     - WENO smoothness exponent. Must be > 0.
   * - ``gamma_i``
     - double
     - —
     - When ``radiation.limiter.type`` = ``weno``
     - WENO central weight. Required for WENO limiter.
   * - ``do_limiter``
     - bool
     - ``true``
     - No
     - Enable slope limiter for radiation.
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
   * - ``tci_opt``
     - bool
     - ``false``
     - No
     - Enable troubled-cell indicator.


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
     - Yes
     - Time integration method. E.g. 'IMEX_SSPRK11', 'IMEX_ARK32_ESDIRK'.

