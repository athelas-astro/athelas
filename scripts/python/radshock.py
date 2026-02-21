#!/usr/bin/env python3

from astropy import constants as consts
import matplotlib.pyplot as plt
import numpy as np

from athelas import Athelas

plt.style.use("style.mplstyle")


def plot_rad_shock(chk):
  problem = "rad_shock"
  fn = f"{problem}_{chk}.ath"
  basis_fn = f"{problem}_basis.ath"

  c = consts.c.cgs.value
  sigma = consts.sigma_sb.cgs.value
  a_rad = 4.0 * sigma / c
  # epsilon = 1.0
  # alpha_so = 4.0 * a_rad / epsilon

  a = Athelas(fn, basis_fn)
  r = a.r
  # tau = a.uCF[0, :, 0]
  # rho = 1.0 / tau
  vel = a.uCF[:, 0, 1]
  emT = a.uCF[:, 0, 2]
  em = emT - 0.5 * vel * vel
  # ev = em * rho
  gamma = 5.0 / 3.0
  # p = (gamma - 1.0) * em / tau
  m_e = consts.m_e.cgs.value
  m_p = consts.m_p.cgs.value
  mu = 1.0 + m_e / m_p
  kb = consts.k_B.cgs.value

  T_g = (gamma - 1.0) * mu * m_p * em / kb

  # rad
  ev_r = a.uCF[:, 0, 3]
  T_r = np.power(ev_r / a_rad, 0.25)

  fig, ax = plt.subplots(figsize=(3.5, 3.5))
  plt.minorticks_on()
  # rho_color = "#86e3a1"  # green
  # vel_color = "#ff9a8b"  # orange
  sie_color = "#d287ef"  # purple
  pre_color = "#8cc8f3"  # blue

  ## --- analytic solution ---
  # EV_TO_K = 1.0 / consts.k_B.to("eV / K").value
  # print(EV_TO_K)
  # t_final = a.time
  # rmin = a.r[0]
  # rax = 3.466205e-3
  # print(rmax)
  # print(a.r[-1])
  # r_sol = np.linspace(rmin, rmax, 1024)
  # kappa = 577.0
  # solver = SuOlson()
  # t_bndry = 3.481334e6  # K
  # soln = solver(r_sol, t_final)
  ## rad_sol, fluid_sol = suolson(t_final, r_sol, t_bndry, kappa, alpha_so)
  ## rad_sol *= EV_TO_K
  ## fluid_sol *= EV_TO_K
  ## ax.plot(r_sol, rad_sol, ls = " ", marker="o", color=pre_color)
  ## ax.plot(r_sol, fluid_sol, ls = " ", marker="o", color=vel_color)
  # print(soln["temperature_rad"])
  # print(soln["temperature_mat"])
  # ax.plot(
  #  r_sol,
  #  soln["temperature_rad"] * EV_TO_K,
  #  ls=" ",
  #  marker="o",
  #  color=pre_color,
  # )
  # ax.plot(
  #  r_sol,
  #  soln["temperature_mat"] * EV_TO_K,
  #  ls=" ",
  #  marker="o",
  #  color=vel_color,
  # )

  # --- athelas ---
  ax.plot(r, T_g / T_g[0], label="Fluid", color=sie_color)
  ax.plot(r, T_r / T_r[0], label="Radiation", color=pre_color, ls="--")
  #  ax.plot(r, p, label="Pressure", color=pre_color)

  ## limiting
  # for i in range(len(r)):
  #  if a.slope_limiter[i] == 1:
  #    ax.axvline(r[i], color="#7c8c8c", alpha=0.25)

  ax.legend(frameon=False)
  ax.set(ylabel=r"Temperature [K]", xlabel="x [cm]")

  svname = f"{problem}_{chk}.png"
  print(f"Saving figure {svname}")
  plt.savefig(svname, dpi=300)
  plt.close(fig)


def main():
  chk = "final"
  plot_rad_shock(chk)


if __name__ == "__main__":
  main()
