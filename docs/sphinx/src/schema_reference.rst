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
   * - ``bc_i``
     - string
     - —
     - No
     - Inner fluid boundary condition. Options: 'reflecting', 'outflow', 'dirichlet', 'periodic'.
   * - ``dirichlet_values_o``
     - array
     - —
     - No
     - Dirichlet state values at outer boundary.
   * - ``dirichlet_values_i``
     - array
     - —
     - No
     - Dirichlet state values at inner boundary.
   * - ``bc_o``
     - string
     - —
     - No
     - Outer fluid boundary condition. Options: 'reflecting', 'outflow', 'dirichlet', 'periodic'.


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
   * - ``bc_i``
     - string
     - —
     - No
     - Inner radiation boundary condition. Options: 'reflecting', 'outflow', 'marshak', 'dirichlet'.
   * - ``dirichlet_values_o``
     - array
     - —
     - No
     - Dirichlet state values at outer boundary.
   * - ``dirichlet_values_i``
     - array
     - —
     - No
     - Dirichlet state values at inner boundary.
   * - ``bc_o``
     - string
     - —
     - No
     - Outer radiation boundary condition. Options: 'reflecting', 'outflow', 'dirichlet'.


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
   * - ``tend``
     - double
     - —
     - No
     - End time for energy injection. Must be > 0.
   * - ``energy``
     - double
     - —
     - No
     - Total injected energy in erg.
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
   * - ``mode``
     - string
     - —
     - No
     - Injection mode. Options: 'direct', 'asymptotic'.


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
   * - ``characteristic``
     - bool
     - ``false``
     - No
     - Enable characteristic limiting.
   * - ``b_tvd``
     - double
     - ``1.0``
     - No
     - TVD parameter b. Used with minmod limiter.
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
   * - ``type``
     - string
     - ``minmod``
     - No
     - Limiter type. Options: 'minmod', 'weno [experimental]'.
   * - ``gamma_i``
     - double
     - —
     - No
     - WENO central weight. Required for WENO limiter.
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
   * - ``do_limiter``
     - bool
     - ``true``
     - No
     - Enable slope limiter for fluid.
   * - ``weno_r``
     - double
     - ``2.0``
     - No
     - WENO smoothness exponent. Must be > 0.
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
   * - ``ncomps``
     - int
     - —
     - No
     - Number of species for Saha solver.
   * - ``fn_ionization``
     - string
     - —
     - No
     - Path to ionization atomic data file.
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
   * - ``env_planck``
     - double
     - ``0.01``
     - No
     - Envelope Planck floor. Used with core_envelope model.
   * - ``env_rosseland``
     - double
     - ``0.01``
     - No
     - Envelope Rosseland floor. Used with core_envelope model.
   * - ``core_rosseland``
     - double
     - ``0.24``
     - No
     - Core Rosseland floor. Used with core_envelope model.
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
   * - ``dt_init``
     - double
     - ``1.000000e-16``
     - No
     - Initial timestep. Must be > 0.
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
   * - ``dt_init_frac``
     - double
     - ``1.05``
     - No
     - Initial timestep growth factor. Must be > 1.


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
   * - ``engine``
     - bool
     - ``false``
     - No
     - Enable energy injection engine.
   * - ``radiation``
     - bool
     - ``false``
     - No
     - Enable radiation transport.
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
   * - ``geometry``
     - string
     - —
     - No
     - Domain geometry. Options: 'planar', 'spherical'.
   * - ``cfl``
     - double
     - —
     - No
     - CFL double for timestep control. Must be > 0.
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
   * - ``t_end``
     - double
     - —
     - No
     - End time of the simulation.
   * - ``nx``
     - double
     - —
     - No
     - Number of grid cells. Must be > 0.
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
   * - ``characteristic``
     - bool
     - ``false``
     - No
     - Enable characteristic limiting. Currently unsupported for radiation.
   * - ``b_tvd``
     - double
     - ``1.0``
     - No
     - TVD parameter b. Used with minmod limiter.
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
   * - ``type``
     - string
     - ``minmod``
     - No
     - Limiter type. Options: 'minmod', 'weno'.
   * - ``gamma_i``
     - double
     - —
     - No
     - WENO central weight. Required for WENO limiter.
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
   * - ``do_limiter``
     - bool
     - ``true``
     - No
     - Enable slope limiter for radiation.
   * - ``weno_r``
     - double
     - ``2.0``
     - No
     - WENO smoothness exponent. Must be > 0.
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
     - No
     - Time integration method. E.g. 'IMEX_SSPRK11', 'IMEX_ARK32_ESDIRK'.

