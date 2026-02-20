#!/usr/bin/env python3

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from athelas import Athelas
from exactpack.solvers.sedov.sedov import Sedov

plt.style.use("style.mplstyle")


def plot_sedov(chk):
  problem = "sedov"
  fn = f"{problem}_{chk}.h5"
  basis_fn = f"{problem}_basis.h5"

  a = Athelas(fn)
  r = a.r
  print(r[0])
  tau = a.get("tau")
  # vel = a.uCF[:, 0, 1]
  # emT = a.uCF[:, 0, 2]
  # em = emT - 0.5 * vel * vel
  rho = 1.0 / tau
  # gamma = 1.4
  # p = (gamma - 1.0) * em / tau

  fig, ax = plt.subplots(figsize=(3.5, 3.5))
  plt.minorticks_on()
  rho_color = "#86e3a1"  # green
  # vel_color = "#ff9a8b"  # orange
  # sie_color = "#d287ef"  # purple
  # pre_color = "#8cc8f3"  # blue

  # --- analytic solution ---
  t_final = a.time
  solver = Sedov(
    geometry=3,  # spherical
    eblast=0.5,
    gamma=1.66667,
    omega=0.0,
  )

  xsol = np.linspace(0.0, 1.0, 64)
  sol = solver._run(xsol, t_final)
  plt.scatter(
    xsol,
    sol["density"],
    s=18,
    facecolor=mcolors.to_rgba(rho_color, alpha=0.25),
    edgecolor=mcolors.to_rgba(rho_color, alpha=1.0),
    linewidth=0.5,
    label="Analytic Solution",
  )

  # --- athelas ---
  ax.plot(r, rho, label="Density", color=rho_color)

  ax.legend(frameon=False)
  ax.set(ylabel=r"$\rho$ [a.u.]", xlabel="x [a.u.]")

  plt.savefig(f"{problem}_{chk}.png")
  plt.close(fig)


def main():
  chk = "final"
  plot_sedov(chk)


if __name__ == "__main__":
  main()
