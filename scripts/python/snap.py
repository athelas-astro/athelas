#!/usr/bin/env python3
"""
Generic plotting script for Athelas output files.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from athelas3 import Athelas, AthelasError


def parse_args():
  parser = argparse.ArgumentParser(
    description="Plot a quantity from one or more Athelas files."
  )

  parser.add_argument(
    "quantity",
    type=str,
    help="Quantity to plot (e.g. rho, tau, pressure, energy)",
  )

  parser.add_argument(
    "files",
    nargs="+",
    help="List of Athelas HDF5 output files",
  )

  parser.add_argument(
    "--mode",
    type=int,
    default=0,
    help="DG mode to plot (default: 0)",
  )

  parser.add_argument(
    "--xlim",
    nargs=2,
    type=float,
    metavar=("XMIN", "XMAX"),
    help="x-axis limits",
  )

  parser.add_argument(
    "--ylim",
    nargs=2,
    type=float,
    metavar=("YMIN", "YMAX"),
    help="y-axis limits",
  )

  parser.add_argument(
    "--logx",
    action="store_true",
    help="Use logarithmic x-axis",
  )

  parser.add_argument(
    "--logy",
    action="store_true",
    help="Use logarithmic y-axis",
  )

  return parser.parse_args()


def main():
  args = parse_args()

  n_files = len(args.files)
  ndigits = len(str(n_files - 1))

  for i, filename in enumerate(args.files):
    filename = Path(filename)

    try:
      ds = Athelas(filename)
    except AthelasError as e:
      print(f"[ERROR] Failed to load {filename}: {e}")
      continue

    fig, ax = plt.subplots(figsize=(8, 5))

    try:
      ds.plot(
        args.quantity,
        ax=ax,
        mode=args.mode,
        logx=args.logx,
        logy=args.logy,
      )
    except AthelasError as e:
      print(f"[ERROR] Failed to plot {filename}: {e}")
      plt.close(fig)
      continue

    if args.xlim is not None:
      ax.set_xlim(args.xlim)

    if args.ylim is not None:
      ax.set_ylim(args.ylim)

    ax.set_title(f"{args.quantity}, t = {ds.time:.3e}\n{filename.name}")

    output = f"{args.quantity}_{i:0{ndigits}d}.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved {output}")


if __name__ == "__main__":
  main()
