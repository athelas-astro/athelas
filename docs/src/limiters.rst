.. [Shu1987] Shu, C.-W. (1987).
   `TVB uniformly high-order schemes for conservation laws
   <https://doi.org/10.1090/S0025-5718-1987-0890256-5>`__.
   Mathematics of Computation, 49(179), 105-121.

.. [CS1989] Cockburn, B., & Shu, C.-W. (1989).
   `TVB Runge-Kutta local projection discontinuous Galerkin finite element
   method for conservation laws II: General framework
   <https://doi.org/10.1090/S0025-5718-1989-0983311-4>`__.
   Mathematics of Computation, 52(186), 411-435.

.. [Kri2007] Krivodonova, L. (2007).
   `Limiters for high-order discontinuous Galerkin methods
   <https://doi.org/10.1016/j.jcp.2007.05.011>`__.
   Journal of Computational Physics, 226(1), 879-896.

Slope Limiters
==============

Athelas damps spurious oscillations in the discontinuous Galerkin solution with
*slope limiters*, applied to the modal (Legendre) coefficients of each field
after every explicit stage. A limiter is selected per physics block
(``fluid`` and ``radiation``) in the input deck; see `Input Deck`_ below.

Setting ``enabled = false`` installs a no-op limiter. Limiting is never applied
to a first-order (``basis.nnodes = 1``) run.

Available limiters
------------------

``minmod`` -- TVD minmod
   The classic Cockburn & Shu TVD(B) limiter. It limits the cell *slope*
   (mode ``k = 1``) with a minmod of the slope against the forward/backward
   differences of the neighboring cell averages, and **zeros every higher
   mode** when the slope is touched. Robust, but it clips smooth extrema and
   degrades the order of the scheme when it activates.

``moment`` -- hierarchical moment limiter
   Limits from the highest mode downward and stops as soon as a mode is
   left unchanged, so it leaves resolved smooth data alone while 
   still controlling oscillations at discontinuities.

``weno`` -- WENO
   Weighted essentially non-oscillatory reconstruction. (Currently unreliable;
   prefer ``minmod`` or ``moment``.)

The fluid limiters support optional characteristic-variable limiting
(``characteristic = true``). Radiation characteristic limiting is currently
unsupported. All limiters support a troubled-cell indicator
(``tci_opt = true``) that restricts limiting to flagged cells.

The TVD minmod limiter
----------------------

The minmod limiter is the total-variation-diminishing slope limiter of the
Runge-Kutta DG framework of Cockburn & Shu [CS1989]_, with the TVB
relaxation of Shu [Shu1987]_. Writing the cell average as
:math:`\bar{u}_i = u_i^{(0)}` and the slope coefficient as :math:`u_i^{(1)}`,
the limited slope on a non-uniform mesh is

.. math::

   \tilde{u}_i^{(1)} = \widetilde{m}\!\left(
       u_i^{(1)},\;
       b\,{h_i \over 2}\,{(\bar{u}_{i+1} - \bar{u}_i) \over d_{i,+}},\;
       b\,{h_i \over 2}\,{(\bar{u}_i - \bar{u}_{i-1}) \over d_{i,-}}
   \right),

where :math:`h_i = \Delta x_i` is the local cell width,
:math:`d_{i,+} = (h_i + h_{i+1})/2`, and
:math:`d_{i,-} = (h_i + h_{i-1})/2`. Neighbor differences are first formed as
physical slopes between cell centroids and then rescaled by the half width of
the target cell. The factor :math:`b` is the compression parameter
(``b_tvd``), and the
TVB-modified minmod function is

.. math::

   \widetilde{m}(a_1, a_2, a_3) =
   \begin{cases}
     a_1 & \text{if } |a_1| \le M\,h_i^2, \\
     \mathrm{minmod}(a_1, a_2, a_3) & \text{otherwise,}
   \end{cases}

with the standard minmod returning :math:`s\,\min_j |a_j|` when all three
arguments share the sign :math:`s` and zero otherwise. The threshold
:math:`M` (``m_tvb``) is the TVB constant: when it is comparable to the local
second derivative of the solution the limiter leaves genuine smooth extrema
untouched, recovering full accuracy there; with :math:`M = 0` the scheme is
strictly TVD and clips extrema.

When the slope is modified the limiter discards every higher mode
(:math:`k \ge 2`), so a limited cell falls back to a (limited) linear profile.
This local-width form is what makes the limiter appropriate for the moving
Lagrangian mesh, where neighboring zones generally do not have equal widths.

The hierarchical moment limiter
-------------------------------

The moment limiter follows Krivodonova [Kri2007]_. Writing the solution on a
cell as an expansion in Legendre modes :math:`u_i = \sum_k u_i^{(k)} P_k`, it
limits the coefficients from the highest mode down to the slope :math:`k = 1`,
comparing mode :math:`k` against the differences of the *next-lower* mode in
the neighboring cells:

.. math::

   \tilde{u}_i^{(k)} = \widetilde{m}\!\left(
       u_i^{(k)},\;
       b\,\beta_k\,{h_i \over 2}\,
         {(u_{i+1}^{(k-1)} - u_i^{(k-1)}) \over d_{i,+}},\;
       b\,\beta_k\,{h_i \over 2}\,
         {(u_i^{(k-1)} - u_{i-1}^{(k-1)}) \over d_{i,-}}
   \right).

Here :math:`\beta_k = 1/(2k-1)` for the un-normalized Legendre coefficients.
On a uniform mesh the half-width scaling gives
the usual inter-mode factor :math:`1/[2(2k-1)]`. At :math:`k = 1`,
:math:`\beta_k = 1`, so this reduces exactly to the TVD minmod slope step
above.

The cascade stops the moment a coefficient is returned unchanged: if the
highest mode is already acceptable the cell is deemed smooth and **all** lower
modes are kept. This is the key difference from the TVD minmod limiter, which
unconditionally discards every mode above the slope. By limiting only the modes
that fail the hierarchical test, the moment limiter preserves high-order
accuracy at smooth extrema. The ``b_tvd`` and ``m_tvb`` controls carry the same
meaning as in the TVD minmod limiter; the TVB threshold is applied to each modal
coefficient with the local width :math:`h_i`. The cell average
(:math:`k = 0`) is never modified, so the scheme remains conservative.


Input Deck
----------
A limiter is configured per physics block (here ``fluid``):

.. code:: lua

   config.fluid = {
     limiter = {
       enabled = true,
       type = "moment",   -- "minmod", "moment", or "weno"
       b_tvd = 1.0,
       m_tvb = 0.0,
       characteristic = false,
       tci_opt = false,
       -- tci_val = 1.0,   -- required when tci_opt = true
     },
   }

The controls are:

* ``b_tvd`` -- minmod compression factor on the neighbor-difference arguments.
* ``m_tvb`` -- TVB threshold :math:`M`; coefficients with
  :math:`|u_i^{(k)}| \le M\,h_i^2` are left untouched, which prevents
  limiting of genuine smooth extrema. A suitable :math:`M` scales with the
  second derivative of the solution; too large a value disables limiting at
  real discontinuities, so it should be tuned per problem.
* ``characteristic`` -- limit in characteristic variables rather than
  conserved variables. This is currently supported for fluid limiting only.
* ``tci_opt`` / ``tci_val`` -- enable the troubled-cell indicator and set its
  threshold, restricting limiting to flagged cells.
