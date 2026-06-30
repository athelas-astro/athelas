#!/usr/bin/env python3
"""Plot an Athelas Sod shock-tube checkpoint against the exact solution.

A small, self-contained example of the ``athelas_tools`` API: load a
checkpoint, derive primitive variables from the evolved state, and overlay the
analytic Riemann solution and any tracker positions recorded in the run
history.

Usage::

    python shocktube.py             # plots the "final" checkpoint
    python shocktube.py 000010         # plots shocktube_0010.ath
    python shocktube.py 000010 -o out.png --no-trackers
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from exactpack.solvers.riemann.ep_riemann import IGEOS_Solver

try:
  from .athelas import Athelas
except ImportError:  # allow running this file directly, not just as a module
  from athelas import Athelas

plt.style.use(Path(__file__).resolve().parents[3] / "style.mplstyle")

# Sod shock tube initial state, matching inputs/sod.lua.
# TODO(astrobarker): [sod; python] pull Sod params in from .ath
GAMMA = 1.4
SOD_LEFT = {"rho": 1.0, "vel": 0.0, "pre": 1.0}
SOD_RIGHT = {"rho": 0.125, "vel": 0.0, "pre": 0.1}
SOD_INTERFACE = 0.5

# One color per primitive variable, shared between the numerical lines and the
# analytic markers. Keys match the exactpack solution field names.
COLORS = {
  "density": "#86e3a1",  # green
  "velocity": "#ff9a8b",  # orange
  "pressure": "#8cc8f3",  # blue
}


def sod_exact(t: float, npoints: int = 32) -> tuple[np.ndarray, dict]:
  """Return ``(x, solution)`` for the exact Sod Riemann problem at time ``t``."""
  solver = IGEOS_Solver(
    rl=SOD_LEFT["rho"],
    ul=SOD_LEFT["vel"],
    pl=SOD_LEFT["pre"],
    gl=GAMMA,
    rr=SOD_RIGHT["rho"],
    ur=SOD_RIGHT["vel"],
    pr=SOD_RIGHT["pre"],
    gr=GAMMA,
    xmin=0.0,
    xd0=SOD_INTERFACE,
    xmax=1.0,
    t=t,
  )
  x = np.linspace(0.0, 1.0, npoints)
  return x, solver._run(x, t)


def plot_shocktube(
  chk: str = "final",
  overlay_trackers: Union[bool, str, list[str]] = True,
  output: Union[str, Path, None] = None,
) -> Path:
  """Plot shock-tube checkpoint ``chk`` against the exact solution.

  Returns the path of the saved figure.
  """
  a = Athelas(f"shocktube_{chk}.ath")

  # Derive primitive variables from the evolved cell averages.
  vel = a.get("velocity")
  tau = a.get("specific_volume")  # specific volume = 1 / density
  sie = a.get("specific_total_fluid_energy") - 0.5 * vel * vel
  primitives = {
    "density": 1.0 / tau,
    "velocity": vel,
    "pressure": (GAMMA - 1.0) * sie / tau,
  }

  fig, ax = plt.subplots(figsize=(3.5, 3.5))
  ax.minorticks_on()

  # Analytic solution as markers (label only the first so the legend shows a
  # single "Analytic Solution" entry).
  x, sol = sod_exact(a.time)
  for i, (name, color) in enumerate(COLORS.items()):
    ax.scatter(
      x,
      sol[name],
      s=18,
      facecolor=mcolors.to_rgba(color, 0.25),
      edgecolor=mcolors.to_rgba(color, 1.0),
      linewidth=0.5,
      label="Analytic Solution" if i == 0 else None,
    )

  # Numerical solution as lines.
  for name, color in COLORS.items():
    ax.plot(a.r, primitives[name], label=name.capitalize(), color=color)

  if overlay_trackers:
    a.plot_trackers(ax=ax, trackers=overlay_trackers)

  ax.legend(frameon=False, fontsize=6)
  ax.set(xlabel="x", ylabel="Solution")

  out = Path(output) if output else Path(f"shocktube_{chk}.png")
  fig.savefig(out)
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "chk",
    nargs="?",
    default="final",
    help="checkpoint id, e.g. '000010' or 'final' (default: final)",
  )
  parser.add_argument("-o", "--output", help="output image path")
  parser.add_argument(
    "--no-trackers",
    action="store_true",
    help="do not overlay history tracker positions",
  )
  args = parser.parse_args()

  out = plot_shocktube(
    args.chk,
    overlay_trackers=not args.no_trackers,
    output=args.output,
  )
  print(f"wrote {out}")


if __name__ == "__main__":
  main()
