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
   e^{-\chi_s/(k_B T)} = f_{s+1}, \,\,
   s = 0, 1, ..., \mathbb{Z} - 1

and observe the recurrence relation

.. math::
   :label: y

   y_{s+1} = y_s \frac{f_{s+1}}{\bar{\mathbb{Z}}n_k}.

Substituting this into :eq:`charge` we get
an expression for the neutral fraction

.. math::
   :label: y0

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
       (\bar{\mathbb{Z}}_k n_k)^s
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
Both solvers use a Newton-Raphson iteration so derivatives are necessary.
The first, which we refer to as the ``linear`` model solves Eq. :eq:`saha_eq`
directly. The second, referred to as ``log`` in ``Athelas``, reformaulates 
Eq. :eq:`saha_eq` by taking natural logs and expressing :math:`\prod f` as 
exponentials.

The linear Saha solver is faster but can be numerically fragile 
for large :math:`\mathbb{Z}` and intended for light elements. 
For the simulations that ``Athelas`` is designed to carry out 
this is usually fine -- it is stable through CNO. The logarithmic 
solver is robust for all species and thermodynamic conditions, 
at increased cost.

Linear
^^^^^^
The original Saha solver in ``Athelas``, which we refer to as ``linear``, 
solves Eq :eq:`saha_eq` directly using a Newton-Raphson iteration. 
Before writing down the derivative :math:`F'(\mathbb{\bar{Z}})` we note that 
this formulation can be subject to numerical instabilities. An attempt to 
circumvent this, inspired by the implementation in ``SNEC``, is to introduce thresholds on 
prospective electron number densities such that they are ignored. This serves 
both as a guard to instability and improves performance.
We rewrite Eqs :eq:`saha_eq` and :eq:`y0` as

.. math::

   y_{\rm min} = \bar{\mathbb{Z}}
   \left[ s_{\rm min} - 1 + \sum_{s=s_{\rm min}}^{s_{\rm max}} 
   \frac{s\prod_{j=s_{\rm min}}^{s} f_j}{(\bar{\mathbb{Z}} n_k)^s} \right].

.. math::

   F(\bar{\mathbb{Z}}) = 1 - \bar{\mathbb{Z}}_k
   \left[
     s_{\rm min} - 1 +
     \sum_{s=s_{\rm min}}^{s_{\rm max}}
     \frac{
       s \displaystyle\prod_{j=s_{\rm min}}^{s} f_{k,j}
     }{
       (\bar{\mathbb{Z}}_k n_k)^s
     }
   \right]^{-1}
   \left[
     1 + \sum_{s=s_{\rm min}}^{s_{\rm max}}
     \frac{
       \displaystyle\prod_{j=1}^{i} f_{k,j}
     }{
       (\bar{\mathbb{Z}}_k n_k)^s
     }
   \right]

:math:`s_{\rm min}` and :math:`s_{\rm max}` are set based on thresholds 
``ZBARTOL`` and ``ZBARTOLINV`` as compared to :math:`f_s/(\bar{\mathbb{Z}} n_k)`.
If :math:`f_s/(\bar{\mathbb{Z}} n_k) <` ``ZBARTOL`` then we do not solve for the 
:math:`(s+1)`-th ionization state and assume that state and all following are empty.
Similarly, if :math:`f_s/(\bar{\mathbb{Z}} n_k) >` ``ZBARTOLINV`` we assume that the 
:math:`s`-th ionization state and all preceeding are empty.

With this in hand we may write down the derivative :math:`F'(\bar{\mathbb{Z}})`

.. math::
   F'(\bar{\mathbb{Z}}) = \left[ \Sigma_1 - 
   \left( 1 + \Sigma_0 \right)
   \left[ 1 + \Sigma_3(s_{\rm min} - 1 + \Sigma_1)^{-1} \right]
   \right] (s_{\rm min} - 1 + \Sigma_1)

where we have defined

.. math::
   \begin{aligned}
   \Sigma_0 &= \sum_{s=s_{\rm min}}^{s=s_{\rm max}} 
     \left( \prod_{j=s_{\rm min}}^{s} \frac{f_j}{\bar{\mathcal{Z}}n_k}\right) \\
   \Sigma_1 &= \sum_{s=s_{\rm min}}^{s=s_{\rm max}} 
     \left( s\prod_{j=s_{\rm min}}^{s} 
   \frac{f_j}{\bar{\mathcal{Z}}n_k}\right) \\
   \Sigma_2 &= \sum_{s=s_{\rm min}}^{s=s_{\rm max}} 
     \left( (s-s_{\rm min} + 1)\prod_{j=s_{\rm min}}^{s} 
   \frac{f_j}{\bar{\mathcal{Z}}n_k}\right) \\
   \Sigma_3 &= \sum_{s=s_{\rm min}}^{s=s_{\rm max}} 
     \left( s(s-s_{\rm min} + 1)\prod_{j=s_{\rm min}}^{s} 
   \frac{f_j}{\bar{\mathcal{Z}}n_k}\right)
   \end{aligned}


Log
^^^
The products of exponentials in Eq. :eq:`saha_eq` can overflow for sufficiently 
high-:math:`\mathbb{Z}` atoms.
To improve numerical stability, Athelas can express the Saha ladder in logarithmic
space. Begin by compactifying Eq :eq:`saha_eq` as 

.. math::
   F(\bar{\mathbb{Z}}) = 1 - \bar{\mathbb{Z}}\frac{\mathcal{N}}{\mathcal{D}}

where :math:`\mathcal{N}` and :math:`\mathcal{D}` can be identified from 
Eq :eq:`saha_eq`.
Taking the natural logarithm of :math:`F(\bar{\mathbb{Z}})` and rearranging, 
we notice that a root of :math:`F` corresponds to a root of

.. math::
   G(\bar{\mathbb{Z}}) = \ln \bar{\mathbb{Z}} + \ln \mathcal{N} - \ln \mathcal{D}

In what follows we will exploit the "LogSumExp" trick borrowed from 
machine learning. The core idea is to notice that

.. math::
   \ln \left( \sum_i e^{a_i} \right) 
   = a_{\rm max} + \ln \left( \sum_i e^{a_i - a_{\rm max}} \right).

All that remains is to express :math:`\mathcal{N}` and :math:`\mathcal{D}`
in a useful form. Note that the above can be written into an 
inline algorithm that doesn't require a-priori knowledge of 
:math:`a_{\rm max}`.

We define

.. math::
  \ell_i = \ln f_i, \quad
  L_i = \sum_{j=0}^{i-1} \ell_j = \ln F_i.

The transcendental equation for :math:`\bar{\mathbb{Z}}` is rewritten in terms of
:math:`x = \ln \bar{\mathbb{Z}}`. Defining the sums

.. math::
  \mathcal{N}(x) = \sum_{i=1}^{Z} \exp\left(L_i - i(x - \ln n_k)\right), \\
  \mathcal{D}(x) = \sum_{i=1}^{Z} i \exp\left(L_i - i(x - \ln n_k)\right).

With manipulation these can be identified as the numerator and denominator of 
Eq :eq:`saha_eq`. The mean charge equation then becomes

.. math::
  G(x) = x + \ln \mathcal{N}(x) - \ln \mathcal{D}(x).

This form avoids explicit products and powers of :math:`\bar{\mathbb{Z}}`, 
and all summations are evaluated using stable log-sum-exp techniques.
The equation :math:`G(x) = 0` is solved using a Newton–Raphson iteration in logarithmic
space. This requires the derivative which, with respect to :math:`x`, can easily be 
written down:

.. math::
  G'(x) = 1 - \frac{\mathcal{D}(x)}{\mathcal{N}(x)} 
  + \frac{\mathcal{E}(x)}{\mathcal{D}(x)},

where

.. math::
  \mathcal{E}(x) = \sum_{i=1}^{Z} i^2 \exp\left(L_i - i(x - \ln n_k)\right).

Then we may recover :math:`\bar{\mathbb{Z}} = e^{x}`. 
The iteration typically converges in one or two steps.

.. note::
   The minimum/maximum ionization state tricks used above in the ``linear``
   solver are not yet implemented in the ``log`` solver.

.. tip::
   We recommend the ``linear`` solver for most use cases. It is slightly 
   faster and stable for most species of concern in the context of 
   supernova ejecta.

Atomic Data
------------
We take ionization potentials and statistical weights from the NIST database.
Ionization potentials are stored in ``athelas/data/atomic_data_ionization.dat`` 
and statistical weights in ``athelas/data/atomic_data_degeneracy_factors.dat``.
These are provided for convenience but the user can supply their own.

.. attention::
   We assume, as is common, that the ground states are the dominant 
   contributors in the Saha-Boltzmann equation. Therefore, there is no
   temperature dependence in the partition functions and we 
   simply use the ground state statistical weights.

Configuration
--------------
The ``ionization`` physics has a few controls in the ``[ionization]`` block.
The paths to the ionization potentials and degeneracy factors must 
be specified. The provided data are stored in ``athelas/data`` and can be set 
with ``fn_ionization`` and ``fn_degeneracy``. These are required when 
ionization is enabled. Then, the number of species to solve Saha ionization for 
may be set with the ``ncomps`` option. Finally, the solver method may be set
as ``solver = linear`` or ``solver = log``.

As is inferred above, we do not require that we do Saha ionization solves 
for all species present on the domain. Instead, we can do ionization for 
any number of species.

.. note::
   We currently assume full ionization for species which 
   we do not do Saha solves. This can be extended with little work 
   if desired.

.. note::
   The number of Saha species must be less than or equal 
   to the total number of species set in the ``composition`` block.
   If neutrons are present in the compositional profile then this 
   inequality is strict.

Input Deck
-----------
Ionization is controlled in the input deck as follows:

.. code::

   [physics]
   ionization = true

   [ionization]
   fn_ionization = "../data/atomic_data_ionization.dat"
   fn_degeneracy = "../data/atomic_data_degeneracy_factors.dat"
   ncomps = 3 # must be <= [composition.ncomps] !!
   solver = "linear" # or "log"
