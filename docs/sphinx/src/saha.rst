.. _Zaghloul: https://iopscience.iop.org/article/10.1088/0022-3727/33/8/314/pdf

Saha Ionization Model
========================
``Athelas`` models ionization equilibrium with the non-degenerate Saha-Boltzmann formulation based on
`Zaghloul`_ et al. (2000). This model is used to compute ionization fractions and the mean
charge state of each atomic species.
The ionization state enters the equation of state through contributions to the internal
energy and pressure and is therefore solved repeatedly as part
of the temperature inversion. It is a large computational expense.

.. math::
  \frac{n_{i+1}}{n_i}
  \frac{2 U_{i+1}(T)}{U_i(T)}
  \left( \frac{2 \pi m_e k_B T}{h^2} \right)^{3/2}
  \frac{1}{n_e}
  \exp\left(-\frac{I_i}{k_B T}\right).

Zaghloul Saha Formulation
--------------------------
Consider an atomic species with charge :math:`Z`, temperature :math:`T`, and
number density :math:`n_k`. Let :math:`n_i` denote the number density of the ion in
charge state :math:`i`, where :math:`i = 0` corresponds to the neutral atom.


.. math::
   :label: saha

   1 - \bar{\mathbb{Z}}_k
   \left(
     \sum_{i=1}^{Z_k}
     \frac{
       i \displaystyle\prod_{j=1}^{i} f_{k,j}
     }{
       (\bar{\mathbb{Z}}_k n_k)^i
     }
   \right)^{-1}
   \left[
     1 + \sum_{i=1}^{Z_k}
     \frac{
       \displaystyle\prod_{j=1}^{i} f_{k,j}
     }{
       (\bar{\mathbb{Z}}_k n_k)^i
     }
   \right]
   = 0 \, .

.. math::

   \begin{aligned}
   f_{k,i+1} &= 2 \frac{g_{k,i+1}}{g_{k,i}}
   \left[
     \frac{2 \pi m_e k_{\rm B} T}{h^2}
   \right]^{3/2}
   \exp\left(
     -\frac{I_{k,i}}{k_{\rm B} T}
   \right) \, , \\
   i &= 0,1,\ldots,(Z_k-1) \, .
   \end{aligned}

The ratio of successive ionization states is given by the Saha relation

.. math::
  \frac{n_{i+1}}{n_i} = f_i(T, n_k),

where

.. math::
  f_i(T, n_k) =
  \frac{2 U_{i+1}(T)}{U_i(T)}
  \left( \frac{2 \pi m_e k_B T}{h^2} \right)^{3/2}
  \frac{1}{n_k}
  \exp\left(-\frac{I_i}{k_B T}\right).

Here:

:math:`U_i(T)` is the partition function of charge state :math:`i`,
:math:`I_i` is the ionization energy from state :math:`i` to :math:`i+1`,
:math:`m_e` is the electron mass,
:math:`k_B` is Boltzmann’s constant,
:math:`h` is Planck’s constant.

Following Zaghloul et al., we define the cumulative product

.. math::
  F_i = \prod_{j=0}^{i-1} f_j,

with :math:`F_0 = 1`. The number density of each charge state can then be written as

.. math::
  n_i = \frac{F_i}{Z_k^i} n_0,

where :math:`Z_k` is the mean ion charge and :math:`n_0` is the neutral density.

Mean Charge Constraint
-----------------------
The mean charge state :math:`\bar{Z}` is defined by

.. math::
  \bar{Z} = \frac{1}{n_k} \sum_{i=0}^{Z} i n_i.

Substituting the Saha ladder expression yields a transcendental equation for
:math:`\bar{Z}`:

.. math::
  \bar{Z} =
  \frac{\displaystyle \sum_{i=1}^{Z} i \frac{F_i}{\bar{Z}^i}}
  {\displaystyle \sum_{i=0}^{Z} \frac{F_i}{\bar{Z}^i}}.

This equation is solved numerically for :math:`\bar{Z}`. Once :math:`\bar{Z}` is known,
the individual ion fractions follow directly from the Saha relations.

Numerical Challenges
---------------------
For species with many ionization states or at high temperatures, the cumulative products
:math:`F_i` can span many orders of magnitude. Direct evaluation of the Saha ladder may
therefore suffer from floating-point overflow or underflow, even when the final physical
result is well-behaved.

Since the ionization solve is performed repeatedly inside the temperature inversion,
numerical robustness is essential.

Logarithmic Reformulation
-------------------------
To improve numerical stability, Athelas evaluates the Saha ladder in logarithmic
space. We define

.. math::
  \ell_i = \ln f_i, \quad
  L_i = \sum_{j=0}^{i-1} \ell_j = \ln F_i.

The transcendental equation for :math:`\bar{Z}` is rewritten in terms of
:math:`x = \ln \bar{Z}`. Defining the sums

.. math::
  N(x) = \sum_{i=1}^{Z} \exp\left(L_i - i(x - \ln n_k)\right), \\
  D(x) = \sum_{i=1}^{Z} i \exp\left(L_i - i(x - \ln n_k)\right),

the mean charge condition becomes

.. math::
  G(x) = x + \ln N(x) - \ln D(x) = 0.

This form avoids explicit products and powers of :math:`\bar{Z}`, and all summations
are evaluated using stable log-sum-exp techniques.

Newton–Raphson Solve
------------------------
The equation :math:`G(x) = 0` is solved using a Newton–Raphson iteration in logarithmic
space. Both the function value and its derivative are computed simultaneously:

.. math::
  G'(x) = 1 - \frac{D(x)}{N(x)} + \frac{E(x)}{D(x)},

where

.. math::
  E(x) = \sum_{i=1}^{Z} i^2 \exp\left(L_i - i(x - \ln n_k)\right).

Evaluating the function and derivative together avoids duplicated work and ensures
consistent numerical behavior.

The iteration typically converges in one or two steps when initialized from the
previous timestep or spatially neighboring solution.

The linear Saha solver is faster but numerically fragile and intended for light elements only.
The logarithmic solver is robust for all species and thermodynamic conditions, at increased cost.
