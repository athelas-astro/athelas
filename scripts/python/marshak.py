#!/usr/bin/env python3

from astropy import constants as consts
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from athelas import Athelas

plt.style.use("style.mplstyle")


# needs a massive cleanup
def plot_marshak(chk):
  problem = "marshak"
  fn = f"{problem}_{chk}.ath"

  kappa = 577
  rho0 = 10.0
  chi = rho0 * kappa

  c = consts.c.cgs.value
  sigma = consts.sigma_sb.cgs.value
  a_rad = 4.0 * sigma / c
  epsilon = 1.0
  alpha_so = 4.0 * a_rad / epsilon

  a = Athelas(fn)
  r = a.r
  tau = a.get("tau")
  print(len(tau))
  rho = 1.0 / tau
  vel = a.get("vel")
  emT = a.get("fluid_energy")
  em = emT - 0.5 * vel * vel
  ev = em * rho

  # use Su-Olson form for temperature
  T_g = np.power(4.0 * ev / alpha_so, 0.25)

  # rad
  ev_r = a.get("rad_energy") * rho
  #  print(ev_r)
  T_r = np.power(ev_r / a_rad, 0.25)

  fig, ax = plt.subplots(figsize=(3.5, 3.5))
  plt.minorticks_on()
  # rho_color = "#86e3a1"  # green
  # vel_color = "#ff9a8b"  # orange
  sie_color = "#d287ef"  # purple
  pre_color = "#8cc8f3"  # blue

  # --- analytic solution ---
  kappa = 577.0
  t_bndry = 3.481334e6  # K
  x_sol, t_rad_sol, t_fluid_sol = np.loadtxt(
    "marshak.dat", unpack=True, usecols=(1, 4, 5)
  )
  x_sol = np.sqrt(3) * x_sol / chi
  plt.scatter(
    (x_sol),
    t_fluid_sol * t_bndry,
    s=12,
    facecolor=mcolors.to_rgba(sie_color, alpha=0.05),
    edgecolor=mcolors.to_rgba(sie_color, alpha=0.5),
    linewidth=0.5,
    label="Analytic Solution",
  )
  plt.scatter(
    (x_sol),
    t_rad_sol * t_bndry,
    s=12,
    facecolor=mcolors.to_rgba(pre_color, alpha=0.05),
    edgecolor=mcolors.to_rgba(pre_color, alpha=0.5),
    linewidth=0.5,
    label="Analytic Solution",
  )

  # --- athelas ---
  #  try to buid x vars
  new_r = np.zeros_like(r)
  for i in range(len(r)):
    new_r[i] = np.sqrt(3) * r[i]
  ax.semilogx(new_r, T_g, label="Fluid", color=sie_color)
  ax.semilogx(new_r, T_r, label="Radiation", color=pre_color, ls="--")

  ax.legend(frameon=False)
  ax.set(ylabel=r"Temperature [K]", xlabel="x [cm]")

  svname = f"{problem}_{chk}.png"
  print(f"Saving figure {svname}")
  plt.savefig(svname, dpi=300)
  plt.close(fig)


def main():
  chk = "final"
  plot_marshak(chk)


if __name__ == "__main__":
  main()
