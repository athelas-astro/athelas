Gravity
=======

Overview
--------

Athelas provides Newtonian gravity as an explicit physics package. The default
*coupled* formulation is designed for spherical self-gravity: it holds
hydrostatic balance to the order of the scheme and conserves fluid plus
gravitational energy in the semi-discrete limit. A constant acceleration model
is also available for test problems.

Enable gravity with ``physics.gravity = true``. Coupled gravity is the
recommended mode. ``gravity.operator_split = true`` is a deliberately simpler
compatibility mode; it uses a direct velocity kick and gives up well-balancing
and total-energy conservation. The available options are summarized in
:ref:`gravity-configuration`.

Physical model
--------------

Gravity enters through the Newtonian potential :math:`\varphi`. In spherical
symmetry,

.. math::

   \varphi(r) = -\frac{G M(r)}{r},

where :math:`M(r)` is the mass enclosed within :math:`r`; the ``constant``
model uses :math:`\varphi(r) = g_{\rm val}\,r`. The acceleration is minus its
gradient,

.. math::

   g = -\frac{\partial\varphi}{\partial r},

recovering :math:`g = -GM/r^2` and :math:`g = -g_{\rm val}` respectively.

Gravity contributes acceleration and work to the Lagrangian fluid equations:

.. math::

   \frac{\partial v}{\partial t} = -\frac{\partial\varphi}{\partial r} ,
   \qquad
   \frac{\partial E}{\partial t} = -v\,\frac{\partial\varphi}{\partial r} .

Here :math:`v` is velocity and :math:`E` is specific total fluid energy.

Numerical model
---------------

Coupled gravity
^^^^^^^^^^^^^^^

A pointwise discretization of these source terms would neither preserve
hydrostatic balance nor conserve the discrete potential energy.
We seek a representation of gravitational sources that is consistent with the
operators used for the hydrodynamic flux divergence and geometric source.
Moreover, in the Lagrangian picture, the gravitational potential evolves
through mesh advection, so any treatment of gravity must be consistent with
mesh motion.
The coupled package builds a *gravity pressure* :math:`\tilde\Pi`, defined in
the continuum by

.. math::

   \frac{\partial\tilde\Pi}{\partial r}
     = -\rho\,\frac{\partial\varphi}{\partial r} .

This is precisely the equation of hydrostatic equilibrium: :math:`\tilde\Pi` is
the pressure profile that balances the potential :math:`\varphi`. Equilibrium is
therefore the statement :math:`P = \tilde\Pi + \text{constant}`.
In the mass coordinate, dropping factors of :math:`4\pi` so that
:math:`\mathrm{d}m = \rho\sqrt{\gamma}\,\mathrm{d}r` and :math:`\sqrt{\gamma}=r^2`
in spherical geometry, this reads

.. math::

   \partial_m \tilde\Pi = -\rho\,\partial_m\varphi .

Discretely, :math:`\tilde\Pi` is built as the gravitational analogue of a
pressure. A gas pressure is how the internal energy resists compression;
:math:`\tilde\Pi` is how the *gravitational* energy resists it. Let
:math:`W_h=\sum_q M_q\,\varphi(r_q)` be the discrete gravitational energy
(nodal mass :math:`M_q=w_q\mu_q` times the potential at each node), and define

.. math::

   M\tilde\Pi = \frac{\partial W_h}{\partial\tau} .

Nudging the specific volume :math:`\tau` in a cell repacks the gas, which moves
the nodes and so shifts where mass sits in the potential; :math:`\tilde\Pi` is
precisely how much :math:`W_h` responds. That is exactly the sense in which the
gas and gravity pressures balance in equilibrium.

Computing that response is a single **inward sweep**.
The change in :math:`W_h` at a cell accumulates the
gravitational pull of everything exterior to it, which the package assembles as
one surface-to-centre running sum. Formally that sum is the transpose of the
mesh reconstruction :math:`r(\tau)` -- the map that integrates :math:`\mu\tau`
to the volume coordinate :math:`X=r^3/3`. Writing
:math:`J_X=\partial X/\partial\tau` for its tangent (the same object that
advances the grid, :math:`\dot r=J_X\dot\tau/r^2`) and
:math:`f=\rho\,\partial_m\varphi=\partial\varphi/\partial X` for the force per
unit volume coordinate,

.. math::

   M\tilde\Pi = J_X^{\mathsf T} M f ,

which is the **adjoint** of the reconstruction.

Building :math:`\tilde\Pi` this way is what makes the scheme *conservative*.
Any :math:`\tilde\Pi` obeying the
continuum relation above would balance the momentum equation for a hydrostatic
state, but only the exact adjoint satisfies
:math:`\dot W_h=\tilde\Pi^{\mathsf T} M\dot\tau`: the gravity work returned to the
fluid then equals the rate of change of the discrete potential energy, term for
term. That identity is the engine of the energy source below.

Well-balancing
~~~~~~~~~~~~~~

The momentum source applies the same weak divergence operator to
:math:`\tilde\Pi` that hydro applies to fluid pressure :math:`P`. For a discrete
state satisfying

.. math::

   P_h - \tilde\Pi_h = \mathrm{constant} ,

the interior (volume) and curvature parts of the two operators cancel term for
term to roundoff -- the payoff of building :math:`\tilde\Pi` as the exact
adjoint. What survives is a residual confined to the cell faces, because the two
packages compute the face pressure by different rules: gravity uses a single
continuous scan value, while hydro uses a Riemann solve of two independently
reconstructed traces. These agree at the boundaries -- the inner face carries
:math:`\sqrt\gamma = 0`, and the outer face is exact under the surface gauge
with a matching surface pressure -- but only to truncation order at interior
faces.

The balance is therefore high order in the mesh spacing, not exact to
roundoff: the spurious equilibrium acceleration converges away under
refinement at the order of the scheme. That convergence, rather than a
machine-zero residual, is the operative well-balancing criterion.
Making the balance exact would additionally require the fluid flux to reuse
:math:`\tilde\Pi` at the face -- a hydrostatic-reconstruction-style well-balanced
flux -- which is a possible extension, not the current scheme.

Gauge choice
~~~~~~~~~~~~

:math:`\tilde\Pi` is defined up to a constant. Both the momentum and energy
operators are invariant under :math:`\tilde\Pi\to\tilde\Pi+c`: the constant
cancels in the weak momentum operator, and the energy contribution reduces to
the discrete specific-volume identity. We take
:math:`\tilde\Pi=0` at the stellar surface, where :math:`P\to0`; this avoids
subtractive cancellation in :math:`P-\tilde\Pi`.

Energy conservation
~~~~~~~~~~~~~~~~~~~

The coupled energy source uses a weak product rule,

.. math::

   M\dot E_g = \mathcal{D}_h(\sqrt{\gamma}\,v\tilde\Pi)
                - M(\tilde\Pi\dot\tau),

assembled as ``(face - volume - dilatation)``. The face term telescopes across
cells, the summed volume term vanishes, and the dilatation term is
:math:`\tilde\Pi^{\mathsf T}M\dot\tau`. Therefore the total gravity work is
exactly :math:`-\mathrm{d}W_h/\mathrm{d}t` in the semi-discrete sense.

The hydro or IMEX rad-hydro package publishes the complete mass-matrix-scaled
specific-volume RHS in ``dtau_dt`` earlier in the same Runge--Kutta stage.
Hydro must be registered before coupled gravity. This same-stage dependency is
why the weak source cannot be used as an operator-split package.

Operator-split gravity
^^^^^^^^^^^^^^^^^^^^^^

Split gravity holds the mesh fixed and applies a direct kick after the hydro
step. With :math:`g` evaluated on that post-hydro mesh, it uses

.. math::

   v^{n+1}=v^n+g\Delta t, \qquad
   E^{n+1}=E^n+\tfrac12\big[(v^{n+1})^2-(v^n)^2\big].

This leaves specific internal energy unchanged to roundoff. It does *not* move
the mesh or update :math:`W_h`, so it is not well-balanced and does not conserve
fluid plus gravitational energy. The limiter-energy correction is unavailable
in split mode.

.. note::

   Operator splitting is not recommended for gravity. The gravity update is not
   a large computational expense, so there is no real gain and a demonstrable
   loss of accuracy.

Diagnostics and conservation
----------------------------

The history file reports ``Total Gravitational Energy [erg]`` as the discrete
form of :math:`W = \int\varphi\,\mathrm{d}m`, using the same potential
:math:`\varphi` as the source. Together with total fluid and radiation energy,
this is the conservation audit for coupled gravity.

Limiter mesh work
^^^^^^^^^^^^^^^^^

Limiters preserve each cell's mass-weighted average, so they do not move cell
faces. They can nevertheless reshape :math:`\tau` within a cell, relocate its
interior nodes, and change :math:`W_h` without a physical energy source.
``Cumulative Limiter Mesh Work [erg]`` measures this change for the
end-of-step limiter block on the committed state.

Limiter energy correction
^^^^^^^^^^^^^^^^^^^^^^^^^

``gravity.limiter_energy_correction`` returns the committed limiter's potential
change to the fluid. For each cell it computes

.. math::

   \delta W_i = 4\pi\sum_q w_q\mu_q
      \left[\varphi(r_q^{\rm new})-\varphi(r_q^{\rm old})\right]

and adds :math:`-\delta W_i/m_i` uniformly to the cell's specific energy. The
correction is clamped at the EOS internal-energy floor. The associated history
columns are ``Cumulative Limiter Energy Correction [erg]`` and
``Cumulative Limiter Energy Clamp Residual [erg]``. They satisfy

.. math::

   \text{correction}+\text{clamp residual}=-\text{mesh work}.

.. _gravity-configuration:

Configuration
-------------

Enable the package and select a model:

.. code-block:: lua

   config.physics = { gravity = true }
   config.gravity = {
     model = "spherical",               -- or "constant"
     gval = 1000.0,                     -- acceleration [cm s^-2], for "constant"
     operator_split = false,            -- coupled gravity is recommended
     limiter_energy_correction = false, -- opt-in
   }

See :doc:`schema_reference` for the complete option table.
