.. _Zaghloul: https://iopscience.iop.org/article/10.1088/0022-3727/33/8/314/pdf

Saha Ionization Model
========================
``Athelas`` models ionization equilibrium with the non-degenerate Saha-Boltzmann 
equations based on the formalism of `Zaghloul`_ et al. (2000). 
This model is used to compute ionization fractions and the mean charge state of 
each atomic species. The ionization state enters the equation of state 
through contributions to the internal energy and pressure and is therefore 
solved repeatedly as part of the temperature inversion. 
It is a large computational expense.

The Saha-Boltzmann equations have the form

.. math::
   :label: saha

  \frac{n_{s+1}n_{e}}{n_s} = 
  \frac{2 g_{s+1}}{g_s}
  \left[ \frac{2 \pi m_e k_B T}{h^2} \right]^{3/2}
  e^{-\chi_s/(k_B T)}, \,\,
   s = 0, 1, ..., \mathbb{Z} - 1

where :math:`n_s` is the number density of atoms in the :math:`s`-th ionization state, 
:math:`g_s` is the associated statistical weight, :math:`m_e` is the electron rest mass, 
:math:`h` is Planck's constant, 
:math:`k_B` is Boltzmann's constant, :math:`\chi_s` is the ionization energy for the 
:math:`s \to (s+1)` process, and :math:`\mathbb{Z}` is the atomic number of the species.

The Saha-Boltzmann equations can be combined with the condition of charge neutrality

.. math::
   :label: charge_neutrality

   \sum_{s=1}^{\mathbb{Z}} s\, n_s = n_e

and conservation of nuclei

.. math::
   :label: nuclei

   \sum_{i=0}^{\mathbb{Z}} n_i = n_k = \text{constant},

where :math:`n_k` is the number density of nuclei of species :math:`k`, 
can be formed into a single set of transcendental equations that can be solved
for the mean charge :math:`\bar{\mathbb{Z}}`. Let us denote the fraction of atoms 
in the :math:`s`-th ionization state as :math:`y_s = n_s/n_k` and express the 
mean charge as :math:`\bar{\mathbb{Z}} = n_e/n_k` we can, 
following `Zaghloul`_ , reexpress the above equations in the following forms

.. math::
   :label: conservation

   \sum_{s=0}^{\mathbb{Z}} y_s = 1

.. math::
   :label: charge

   \sum_{s=0}^{\mathbb{Z}} s\, y_s = \bar{\mathbb{Z}}

.. math::
   :label: new_saha

  \frac{y_{s+1}\bar{\mathbb{Z}}n_{k}}{y_s} = 
  2\frac{g_{s+1}}{g_s}
  \left[ \frac{2 \pi m_e k_B T}{h^2} \right]^{3/2}
  e^{-\chi_s/(k_B T)} = f_s, \,\,
   s = 0, 1, ..., \mathbb{Z} - 1

and observe the recurrence relation

.. math::
   y_{s+1} = y_s \frac{f_s}{\bar{\mathbb{Z}}n_k}.

Substituting this into :eq:`charge` we get
an expression for the neutral fraction

.. math::
   y_0 = \bar{\mathbb{Z}}
   \left[ \sum_{s=1}^{\mathbb{Z}} \frac{s\prod_{j=1}^{s} f_j}{(\bar{\mathbb{Z}} n_k)^s} \right].

Finally, combining the relations for the ionization fractions :math:`y` with the 
condition for conservation :eq:`conservation` of nuclei we 
arrive at a transcendental equation for :math:`\bar{\mathbb{Z}}`

.. math::
   :label: saha_eq

   F(\bar{\mathbb{Z}}) = 1 - \bar{\mathbb{Z}}_k
   \left[
     \sum_{s=1}^{Z_k}
     \frac{
       s \displaystyle\prod_{j=1}^{s} f_{k,j}
     }{
       (\bar{\mathbb{Z}}_k n_k)^i
     }
   \right]^{-1}
   \left[
     1 + \sum_{s=1}^{Z_k}
     \frac{
       \displaystyle\prod_{j=1}^{i} f_{k,j}
     }{
       (\bar{\mathbb{Z}}_k n_k)^s
     }
   \right]

Equation :eq:`saha_eq` may be solved iteratively for :math:`\bar{\mathbb{Z}}` using a any 
root finding algorithm. ``Athelas`` implements a couple of method for finding 
:math:`\bar{\mathbb{Z}}`.

Once :math:`\bar{\mathbb{Z}}` is known then the constribution from this species 
to the electron number density can be tallied as :math:`n_e = \bar{\mathbb{Z}}n_k`.

Numerical Methods
---------------------
``Athelas`` implements two methods for solving equation :eq:`saha_eq`.


Numerical Challenges
---------------------
For species with many ionization states, at very high temperatures, 
or at low number densities, the cumulative products
can grow quickly. Direct evaluation of the Saha ladder may
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

Atomic Data
------------

Configuration
--------------

Input Deck
-----------
