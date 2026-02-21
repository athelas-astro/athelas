#!/usr/bin/env python3

import glob

from astropy import constants as consts
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

from athelas import Athelas


class ThermalEquilibrium:
  """
  Class for constructing thermal equilibrium solution
  See https://ui.adsabs.harvard.edu/abs/2001ApJS..135...95T/abstract

  Args:
    e_gas (float) : initial gas internal energy densty
    kappa (float) : absorption opacity
    T (float)     : temperature
    e_rad (float) : initial radiation energy densty
    t_end (float) : simulation time
  """

  def __init__(self, e_gas, kappa, T, e_rad, t_end):
    self.e_gas = e_gas
    self.kappa = kappa
    self.temperature = T
    self.e_rad = e_rad
    self.t_end = t_end
    self.t = None  # time array, gets filled later.
    self.sol = None

  # End __init__

  def __str__(self):
    print(f"e_gas = {self.e_gas:.5e}")
    print(f"kappa = {self.kappa:.5e}")
    print(f"temperature = {self.temperature:.5e}")
    print(f"e_rad = {self.e_rad:.5e}")
    print(f"t_end = {self.t_end:.5e}")
    return "\n"

  # End __str__

  def rhs_(self, e_gas, _):
    c = consts.c.cgs.value
    a_rad = 4.0 * consts.sigma_sb.cgs.value / c
    return (
      -c * self.kappa * (a_rad * np.power(self.T_(e_gas), 4.0) - self.e_rad)
    )

  # End rhs_

  def T_(self, e_gas):
    """
    Ideal gas temperature from internal energy
    TODO: don't hardcode gamma, mu, rho
    """
    k = consts.k_B.cgs.value
    gamma = 5.0 / 3.0
    rho = 1.0e-7
    sie = e_gas / rho
    mu = 1.0 + consts.m_e.cgs.value / consts.m_p.cgs.value
    return (gamma - 1.0) * mu * consts.m_p.cgs.value * sie / (k)

  # End T_

  def evolve(self):
    # t = np.linspace(0, self.t_end, 2049)  # 1024 time values
    t = np.logspace(-16, np.log10(self.t_end), 2049)  # 1024 time values
    self.t = t
    sol = odeint(self.rhs_, self.e_gas, t)
    self.sol = sol

  # End evolve


# End ThermalEquilibrium
def main():
  # TODO: make input params with argparse
  e_gas = 10**10
  e_rad = 10**12
  T = 4.81e8
  kappa = 4 * 1.0e-8  # * 1.0e-7# 1 / cm weird units
  t_end = 1.0e-7
  te = ThermalEquilibrium(e_gas, kappa, T, e_rad, t_end)
  te.evolve()

  files = sorted(glob.glob("rad_eq*.ath"))[:]
  athelas_time = np.zeros(len(files) - 0)
  athelas_ener = np.zeros(len(files) - 0)

  fig, ax = plt.subplots()
  i = 0
  for fn in files:
    if "basis" in fn:
      continue
    a = Athelas(fn)
    athelas_time[i] = a.time
    athelas_ener[i] = a.get("fluid_energy")[0] / a.get("tau")[0]
    i += 1
  print(athelas_time)
  print(athelas_ener)
  ax.loglog(
    athelas_time,
    athelas_ener,
    marker="x",
    color="#98A785",
    label="Athelas",
    ls=" ",
  )
  ax.loglog(te.t, te.sol, color="k", ls="--", label="Analytic")
  ax.legend(frameon=False)
  ax.set(ylabel=r"$\rho \varepsilon$ [erg cm$^{-3}$]", xlabel="Time [s]")

  plt.savefig("radeq.png")


if __name__ == "__main__":
  main()
