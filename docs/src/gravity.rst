Gravity
=======

Athelas evolves Newtonian self-gravity in spherical symmetry, with a constant
gravitational acceleration available for test problems. Gravity is implemented
as a *package*: it contributes a momentum source and an energy source to every
explicit stage, and is registered automatically when ``physics.gravity = true``.
The configuration options are collected under :ref:`gravity-configuration` at
the end of this page.

Physical model
--------------

In spherical symmetry the gravitational acceleration at radius :math:`r` is

.. math::

   g(r) = -\frac{G M(r)}{r^2},

where :math:`M(r)` is the mass enclosed within :math:`r`. For the ``constant``
model :math:`g` is a fixed input, independent of :math:`r`.

Gravity enters the evolution as source terms in the momentum and total-energy
equations. Athelas is Lagrangian, so the independent spatial variable is the
mass coordinate :math:`m` rather than the radius, and the evolved fluid state is
the specific volume :math:`\tau`, velocity :math:`v`, and specific total energy
:math:`E`. Neglecting the pressure terms (which the hydro package supplies),
gravity accelerates each fluid element and does work on it as it moves:

.. math::

   \left(\frac{\partial v}{\partial t}\right)_{\!g} = g ,
   \qquad
   \left(\frac{\partial E}{\partial t}\right)_{\!g} = g\,v .

These two terms --- the acceleration :math:`g` and the rate of work
:math:`g\,v` --- are what the package supplies to each stage. Everything in the
numerical model below is how they are discretized.

Because the mesh is Lagrangian, each element carries a fixed mass, so the
enclosed mass :math:`M(r)` at a given element is materially conserved --- it does
not change as the star expands or contracts. Self-gravity is thus a function of
the *fixed* mass distribution and the *evolving* nodal radii alone.

The total energy the scheme conserves is the sum of fluid (kinetic plus
internal), radiation, and gravitational potential energy,

.. math::

   W = -\int \frac{G M}{r}\,\mathrm{d}m .

The numerical construction below is built so that the discrete analogue of
:math:`W` is conserved together with the fluid and radiation energy.

Numerical model
---------------

A pointwise discretization of :math:`g` and :math:`g\,v` would neither preserve
hydrostatic balance nor conserve the discrete potential energy. Athelas instead
recasts both sources through a single discrete object, the *gravity pressure*
:math:`\tilde\Pi` (the symbol :math:`\Pi` is reserved for the stress tensor, of
which this is nominally the gravitational part), defined so that in the continuum

.. math::

   g = \sqrt{\gamma} \, \partial_m \tilde\Pi ,

with :math:`\sqrt{\gamma} = r^2` the spherical metric factor and :math:`m` the
mass coordinate. Discretely, :math:`\tilde\Pi` is built by a reverse scan over
cells, inward from the outer boundary, as the transpose of the mesh
reconstruction:

.. math::

   M \tilde\Pi = J^{\mathsf T} M f , \qquad f = -g/\sqrt{\gamma} ,

where :math:`M` is the diagonal nodal mass matrix, with entries
:math:`M_q = w_q \mu_q`, and :math:`J` is the tangent of the map from specific
volume :math:`\tau` to the volume coordinate :math:`X = r^3/3`. Making
:math:`\tilde\Pi` the exact discrete adjoint of the reconstruction --- rather
than a merely consistent approximation --- is what lets the energy source
conserve energy at second order and above.

.. note::

   :math:`M_q` (a bare node subscript) is the mass-matrix diagonal; :math:`M(r)`
   (a function of radius) is the enclosed mass. They are different objects that
   share the letter :math:`M`.

The **momentum source** discretizes :math:`g` by applying the same weak pressure
operator to :math:`\tilde\Pi` that the hydro package applies to the fluid
pressure :math:`P`. Consequently any state with

.. math::

   P_h - \tilde\Pi_h = \mathrm{constant}

is annihilated exactly, to roundoff. This is the well-balancing property, and it
holds for **any** :math:`\tilde\Pi` --- what matters is that the initial data be
consistent with the *discrete* :math:`\tilde\Pi_h`.

.. note::

   A hydrostatic profile obtained by integrating the ODE
   :math:`\mathrm{d}P/\mathrm{d}r = -G M \rho / r^2` and sampling it at the
   nodes is **not** exactly :math:`\tilde\Pi_h + \text{const}`. It agrees only to
   truncation order, so such an initial condition carries a residual
   acceleration. Machine-precision well-balancing requires projecting the
   initial state onto :math:`\tilde\Pi_h`.

The frozen reference mass
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The material invariance of :math:`M(r)` is realized by freezing the
reference-mass measure :math:`\mu_q` and the enclosed mass at every node at their
initial values; ``Mesh::reconstruct_mesh`` never recomputes them. The exact
semi-discrete cancellation of the energy source relies on this, so the mass
measure must not be recomputed mid-evolution (see ``Mesh::compute_mass_measure``).

Gauge freedom
^^^^^^^^^^^^^

:math:`\tilde\Pi` is only defined up to an additive constant, and both operators
are exactly invariant under :math:`\tilde\Pi \to \tilde\Pi + c`:

* In the momentum operator the flux and curvature terms cancel term by term.
* In the energy source the cell total of a constant :math:`\tilde\Pi` is
  :math:`c\,[(\sqrt{\gamma}_R\, v^*_R - \sqrt{\gamma}_L\, v^*_L) -
  \sum_q M_q \dot\tau_q]`, which vanishes identically because the bracket is an
  exact identity of the discrete specific-volume equation.

The reverse scan anchors :math:`\tilde\Pi = 0` at the stellar surface. For a
star this is the well-conditioned choice, since :math:`P \to 0` there as well, so
the constant is small and :math:`P - \tilde\Pi` involves no catastrophic
cancellation.

The energy source
^^^^^^^^^^^^^^^^^

The gravitational work :math:`g\,v` is applied in weak form, as a discrete
product rule rather than pointwise:

.. math::

   M \dot E_g = -\mathcal{D}_h(\sqrt{\gamma}\, v \tilde\Pi)
                + M (\tilde\Pi \dot\tau) ,

assembled per test function as ``(face - volume - dilatation)``. Summed over the
nodal test functions of a cell:

* the face term telescopes across cells, because
  :math:`\sum_q \phi_q = 1` at each face, leaving only boundary work;
* the volume term vanishes identically, because
  :math:`\sum_q \phi_q' = 0`;
* the dilatation term reproduces :math:`\tilde\Pi^{\mathsf T} M \dot\tau`.

The result is that the total gravitational work is exactly
:math:`-\mathrm{d}W_h/\mathrm{d}t`, where

.. math::

   W_h = \sum_q M_q \varphi_q , \qquad \varphi_q = -G\, M(r_q) / r_q

is the discrete gravitational potential energy reported as
``Total Gravitational Energy`` in the history file (here :math:`M_q` is the
mass-matrix diagonal and :math:`M(r_q)` the enclosed mass). **This identity is
exact in the semi-discrete sense**, not merely consistent, which is why the weak
form is preferred over a direct :math:`g\,v` source.

The :math:`\dot\tau` used in the dilatation term is the complete
mass-matrix-scaled specific-volume right-hand side --- including both face lifts
and the volume term --- which the hydro (or IMEX rad-hydro) package publishes
into the ``dtau_dt`` field earlier in the same stage. Hydro must therefore be
registered before gravity, and the source cannot run operator-split (it would
consume a stale right-hand side).

Sources of residual error
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since the semi-discrete identity is exact, any measured drift in total energy
comes from one of the following. Each is diagnosed by a history column.

Limiter mesh work
   The limiters preserve the mass-weighted cell average, so cell *faces* do not
   move. But changing the :math:`\tau` profile within a cell relocates the
   *interior nodes*, since node position is defined by
   :math:`X_q = X_L + \sum_p I_{qp} \mu_p \tau_p`. Moving nodes changes
   :math:`W_h` with no compensating source --- a form of limiter dissipation.
   ``Cumulative Limiter Mesh Work`` reports this for the **end-of-step limiter
   block on the committed state**, and is the quantity the limiter energy
   correction below cancels. The per-stage limiters act on discarded
   intermediate RK states, so their :math:`W_h` changes are not committed
   non-conservation --- they reach the committed state only indirectly, through
   the stage right-hand sides, and are not reported here.

   .. important::

      This argument requires the limiter to be conservative with respect to the
      **mass-weighted** average :math:`\sum_q w_q \mu_q U_q`, which is what
      ``conservative_correction`` enforces. A limiter that instead preserves the
      plain average :math:`\sum_q w_q U_q` will change the cell volume increment
      and displace every face outward of it.

Per-stage limiting and temporal error
   :math:`W_h` is a nonlinear function of the nodal radii, so a Runge-Kutta step
   conserves it only to the order of the integrator; and the per-stage limiters
   perturb the stage right-hand sides. Neither is captured by the committed-state
   mesh-work metric or its correction, so a residual drift remains even with the
   correction on. Both converge away under timestep refinement.

Limiter energy correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The committed-state limiter mesh work above is a genuine non-conservation: the
limiter changes :math:`W_h` with no source. Enabling
``gravity.limiter_energy_correction`` (default off) returns it. After the
end-of-step limiter block, for each cell the correction computes the exact
potential change the limiter caused,

.. math::

   \delta W_i = 4\pi \sum_q w_q \mu_q
       \big[ \varphi(r_q^{\text{new}}) - \varphi(r_q^{\text{old}}) \big],

using the same potential :math:`\varphi` as ``Total Gravitational Energy``
(:math:`-G\, M(r) / r` for spherical self-gravity, :math:`+G\,g_{\rm val}\,r`
for the constant model), and subtracts :math:`\delta W_i / m_i` uniformly from
that cell's specific energy. Because only :math:`\tau` moves the mesh, correcting
:math:`E` has no geometric feedback: it changes only the cell-average level of
:math:`E`, not the limiter's shape, and cannot move :math:`W_h`.

The correction is clamped so the specific internal energy stays at or above the
EOS floor ``min_sie(eos, rho, lambda)`` at every node --- it never removes more
than a cell's available internal energy. The applied amount is reported as
``Cumulative Limiter Energy Correction`` and the shortfall the clamp prevents
(the residual non-conservation) as ``Cumulative Limiter Energy Clamp Residual``.
By construction the reported quantities satisfy
:math:`\text{correction} + \text{clamp residual} = -\,\text{mesh work}`, so the
correction cancels the committed-state limiter work exactly up to the floor. It
does **not** remove the per-stage-limiting and temporal residuals described
above, which is why some drift remains with the correction on.

On the production supernova configuration the correction reduces the
post-injection total-energy drift by two to three orders of magnitude,
depending on resolution and integrator.

.. warning::

   The correction is **opt-in and off by default** because it is not robust in
   every regime. It adjusts the total energy :math:`E` at the gravitational
   scale, and in flows where the specific internal energy
   :math:`e = E - \tfrac{1}{2}v^2` is a small difference of large numbers, that
   adjustment can destabilize the solution even though the EOS floor is
   respected. Two regimes seen to misbehave:

   * **Supersonic cold collapse** (e.g. the pressureless dust-collapse test).
     As the collapse accelerates, the limiter fires hard and the per-cell
     correction grows relative to the tiny internal energy, tipping the run into
     a nonfinite state before the uncorrected run fails.
   * **Violent radiation-coupled phases with weak dissipation.** With
     ``ap_coefficient = 0`` the asymptotic-preserving dissipation is off, so
     violent phases (e.g. thermal-engine injection) leave the radiation solve
     noisy and the correction cannot cleanly offset the (then very large)
     limiter mesh work. Use ``ap_coefficient > 0`` on radiation-coupled runs.

   Enable the correction on smooth, subsonic, well-resolved problems where
   conservation is the priority, and watch ``Cumulative Limiter Energy Clamp
   Residual`` --- a growing clamp residual signals the correction is being
   fought by the internal-energy floor.

.. _gravity-configuration:

Configuration
-------------

Enable the package and select a model:

.. code-block:: lua

   config.physics = { gravity = true }
   config.gravity = {
     model = "spherical",              -- or "constant"
     gval  = 1000.0,                   -- required for model = "constant"
     limiter_energy_correction = false, -- opt-in; see "Limiter energy correction"
   }

.. warning::

   ``gravity.operator_split = true`` is not supported. The energy source
   consumes the specific-volume right-hand side published by the hydro package
   within the same stage, which an operator-split package cannot see.

See :doc:`schema_reference` for the full ``config.gravity`` table.
