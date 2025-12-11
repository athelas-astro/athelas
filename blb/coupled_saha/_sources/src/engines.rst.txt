Engines
========
``Athelas`` supports methods for artificially driving a supernova called ``engines``.
These ``engines`` are ``packages`` which may implement one or more source terms 
and other options. Currently, we implmenent a ``ThermalEngine`` to drive a supernova. 
This is often referred to as a "thermal bomb."

Thermal Engine
-------------------------
The ``ThermalEngine`` model for driving an explosion injects a fixed amount of energy 
through a time dependent heating source term over a fixed interval in mass coordinate 
and in time. The energy injection profile is an exponential decay in mass and in time,
described delow.

We chose the energy per unit time injected into the star to be given by:

.. math::
    Q_{\text{engine}}(t) = d' e^{-c' t}

We set the ratio

.. math::
    \frac{Q_{\text{engine}}(t_{\text{start}})}{Q_{\text{engine}}(t_{\text{end}})} = R_t,

where R\_t controls the steepness of the exponential decay in time 
and is set to 100.0 in ``engines/thermal.hpp``. Note that the start time 
of energy injection is taken to be time 0.0. Then, using this ratio and 
integrating over the injection timescale,

.. math::
    c' = \frac{\ln R_t}{t_{\text{end}} - t_{\text{start}}}
    \quad d' = \frac{c' E_{\text{tot}}}{e^{-c' t_{\text{start}}} - e^{-c' t_{\text{end}}}}

where E\_tot is the total energy to be deposited. 
The energy injected per unit time per unit mass is then of the form

.. math::
    Q_{\text{engine}, i}(m_i) = b' e^{-a' m_i}.

The above is the source term implemented in the ``ThermalEngine`` package. 
Similarly to above, we control the steepness of the decay in mass coordinate

.. math::
    \frac{Q_{\text{engine}, m_i}(m_{\text{start}})}{Q_{\text{engine}, m_i}(m_{\text{end}})} 
    = R_m,

where R_m is again set to 100.0 in ``engines/thermal.hpp``. 
we obtain the final coefficients

.. math::
    a' = \frac{\ln R_m}{m_{\text{end}} - m_{\text{start}}}
    \quad b' = \frac{d' e^{-c' t}}{\sum_i e^{-a' m_i} \Delta m}

where the summation in the expression for b' is performed over the mass extent of the injection.

Note that to determine d' we perform an integration assuming a 
continuous time coordinate. This approach is valid because the timestep 
at the beginning of the calculation is much smaller than the duration of 
energy injection by the engine. However, individual cell masses are generally 
not small enough to justify a continuum approximation, so we instead use a sum over 
discrete mass cells.

Once the injection timescale has passed the ``ThermalEngine`` package is disabled
and the source term no longer computed or applied.


.. admonition:: Timestep Restriction

  The ``ThermalEngine`` requires that the energy injection is spread out over 
  at least 2500 timesteps.

Configuration
--------------
The ``ThermalEngine`` has a few relevant controls.
The extent of injection in time and mass space can be controlled.
Most importantly, the *mode* of calculating the injected energy can be set.
The ``direct`` mode specifies the amount of energy to be injected and uses 
that for E\_tot. This may not be sufficient energy to unbind the star, 
resulting in a failed supernova. More useful is ``asymptotic`` where the 
desired *final* energy is specified. In this mode the total energy of 
the stellar model is calculated and the injection energy is adjust such that 
the energy of the ejecta reaches the desired energy after unbinding the 
star. If for some reason the calculated energy is determined to be less than 
the binding energy of the progenitor the code will crash.

.. note::

   We do not at present allow runtime configuration of the steepness 
   parameters R\_t and R\_m. This is simple to add.

Input Deck
----------
The ``ThermalEngine`` is controlled in the input deck as follows:

.. code::

   [physics]
   engine = true

   [engine.thermal]
   enabled = true 
   mode = "asymptotic" # or "direct"
   energy = 1.5e51 # erg
   tend = 0.1 # seconds
   mend = 0.1 # Msun

