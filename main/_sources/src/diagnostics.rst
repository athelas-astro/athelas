Diagnostics
===========

``Athelas`` supports optional diagnostics for derived quantities that are useful
for interpreting a run but are not part of the evolved state. Diagnostics are
opt-in and are configured under ``config.diagnostics`` in the input deck.

The current diagnostics are:

``optical_depth``
   Computes the Rosseland optical depth from the outer boundary inward. The
   profile is written to the HDF5 ``.ath`` output in a field named
   ``diagnostics/optical_depth``.

``photosphere``
   Tracks the radius where the optical depth crosses a chosen threshold,
   defaulting to :math:`\tau = 2/3`. The tracker is written to the history
   output.

``shock``
   Tracks the strongest velocity compression and records its radius, cell
   index, and compression strength in the history output.

Optical Depth
-------------

The optical depth diagnostic computes

.. math::

   \tau(r) = \int_r^{R_\mathrm{out}} \rho \kappa_R \, dr,

where :math:`\kappa_R` is the Rosseland mean opacity provided by the active
opacity model. The integration is performed from the outer boundary inward, so
the outermost boundary has :math:`\tau = 0` and deeper points have larger
optical depth.

When enabled, the optical-depth profile is stored in the HDF5 checkpoint output
as a diagnostic field.

.. note::

   Optical depth requires opacity data, so it is only valid when radiation
   physics is enabled. The input parser rejects ``diagnostics.optical_depth`` if
   ``physics.radiation = false``.

Photosphere Tracker
-------------------

The photosphere tracker uses the optical-depth profile to locate the radius
where

.. math::

   \tau(r_\mathrm{ph}) = \tau_\mathrm{ph}.

The threshold ``tau`` is configurable and defaults to ``2.0 / 3.0``. If the
threshold is bracketed by the computed optical-depth profile, the tracker
linearly interpolates between neighboring diagnostic samples and writes the
photosphere radius to the history output.

The history quantities are:

* ``Photosphere Radius [cm]``
* ``Photosphere Cell Index``
* ``Photosphere Valid``

``Photosphere Valid`` is ``1`` when the requested optical depth was found and
``0`` otherwise.

.. note::

   The photosphere tracker depends on the optical-depth diagnostic. The input
   parser rejects ``diagnostics.photosphere`` unless
   ``diagnostics.optical_depth.enabled = true``.

Shock Tracker
-------------

The shock tracker locates the strongest velocity compression. It examines both
nodal velocity gradients and velocity jumps across cell faces, then records the
largest positive value of :math:`-\partial v / \partial r`.

Within each cell, the velocity is treated as the DG polynomial represented by
the nodal values in that element. The tracker applies the basis differentiation
matrix to those nodal values to estimate the reference-coordinate derivative,

.. math::

   \left.{\partial v \over \partial \eta}\right|_{i,q}
     = \sum_p D_{q p} v_{i,p},

then converts to a physical gradient using the cell width,

.. math::

   \left.{\partial v \over \partial r}\right|_{i,q}
     = {1 \over \Delta r_i}
       \left.{\partial v \over \partial \eta}\right|_{i,q}.

The nodal compression candidate is therefore
:math:`-\partial v / \partial r` evaluated at each interpolation node. This is
different from a cell-average shock indicator based only on neighboring average
velocities; it uses the local DG representation directly and can see compression
inside a high-order element.

The tracker also checks jumps across cell faces. It evaluates the left cell
polynomial at the outer face and the right cell polynomial at the inner face,
then estimates

.. math::

   - {v_R - v_L \over \Delta r_\mathrm{face}},

where :math:`\Delta r_\mathrm{face}` is the average width of the two cells
adjacent to the face. This catches compression concentrated at an element
interface, where a purely interior nodal derivative can miss the sharpest part
of the jump.

The reported shock position is the node or face with the largest positive
compression. This is a lightweight tracker rather than a troubled-cell
indicator or limiter; it does not modify the solution.

The history quantities are:

* ``Shock Radius [cm]``
* ``Shock Cell Index``
* ``Shock Compression [1 / s]``

Shock Breakout
--------------

The shock and photosphere trackers are most useful together. *Shock breakout*
-- the point at which the shock reaches the stellar surface and its radiation is
no longer trapped behind an optically thick layer -- corresponds to the shock
position crossing the photosphere. In terms of the tracked quantities, breakout
is the first time

.. math::

   r_\mathrm{shock} \gtrsim r_\mathrm{ph},

i.e. when ``Shock Radius [cm]`` first reaches or exceeds ``Photosphere Radius
[cm]``. Before breakout the shock sits below the photosphere
(:math:`r_\mathrm{shock} < r_\mathrm{ph}`); the ordering reverses at breakout.

Breakout is detected in post-processing rather than flagged by the solver:
enable both ``shock`` and ``photosphere`` (and therefore ``optical_depth``), and
the Python ``athelas_tools`` interface locates the crossing from the history
file. The breakout time is often a convenient analysis ``t = 0`` reference.

.. code-block:: python

   from athelas_tools.athelas import Athelas

   ds = Athelas("model_final.ath")
   t_breakout = ds.breakout_time()    # None if breakout never occurs
   t_rel = ds.time_since_breakout     # this checkpoint's time minus t_breakout

The crossing time is linearly interpolated between the bracketing history
samples (pass ``interpolate=False`` for the first post-crossing sample time).
Only rows with a valid photosphere and finite shock radius are considered.

Plotting Trackers
-----------------

The Python ``athelas_tools`` plotting interface can overplot tracker positions
from the history file on checkpoint profiles:

.. code-block:: python

   from athelas_tools.athelas import Athelas

   ds = Athelas("sod_final.ath")
   ax = ds.plot("density", overlay_trackers=True)

By default, this overlays all known trackers that are present in the history
output. Tracker lines use black dashed vertical markers with labels such as
``Shock`` and ``Photosphere``.

Input Deck
----------

Diagnostics are configured in a dedicated input-deck table:

.. code-block:: lua

   config.diagnostics = {
     optical_depth = {
       enabled = true,
     },
     photosphere = {
       enabled = true,
       tau = 2.0 / 3.0,
     },
     shock = {
       enabled = true,
     },
   }

The controls are:

* ``optical_depth.enabled`` -- compute and write the optical-depth profile.
  Requires ``physics.radiation = true``.
* ``photosphere.enabled`` -- write photosphere tracker quantities to the
  history output. Requires ``optical_depth.enabled = true``.
* ``photosphere.tau`` -- optical-depth threshold for the photosphere tracker.
  Must be positive. Defaults to ``2.0 / 3.0``.
* ``shock.enabled`` -- write shock tracker quantities to the history output.
