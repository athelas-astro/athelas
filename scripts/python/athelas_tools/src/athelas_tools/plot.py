import argparse
import concurrent.futures

import matplotlib.pyplot as plt

from athelas import Athelas

plt.style.use("style.mplstyle")


def parse_args():
  parser = argparse.ArgumentParser(
    description="Plot variable from multiple files."
  )
  parser.add_argument("var", type=str, help="Variable to plot")
  parser.add_argument(
    "files", nargs="+", help="List of data files to load and plot"
  )
  parser.add_argument(
    "--logx", action="store_true", help="Use log scale for X axis"
  )
  parser.add_argument(
    "--logy", action="store_true", help="Use log scale for Y axis"
  )
  parser.add_argument(
    "-c", "--color", type=str, default="#b07aa1", help="Color for plotting"
  )
  parser.add_argument(
    "--num-procs",
    type=int,
    default=2,
    help="Maximum number of parallel processes (default: 2)",
  )

  return parser.parse_args()


def plot_file(var, filename, file_idx, color, logx=False, logy=False):
  data = Athelas(filename)
  print(f"Plotting {filename}: '{var}'")

  # Example dummy data
  x = data.r
  y = data.get(var, k=0)

  fig, ax = plt.subplots()
  if logx:
    plt.xscale("log")
  if logy:
    plt.yscale("log")

  plt.minorticks_on()
  ax.plot(x, y, label=filename, color=color)
  ax.set(xlabel="x", ylabel=var)
  fig.savefig(f"fig_{var}_{file_idx:05d}.png")
  plt.close(fig)


def main():
  args = parse_args()
  with concurrent.futures.ThreadPoolExecutor(
    max_workers=args.num_procs
  ) as executor:
    futures = [
      executor.submit(
        plot_file, args.var, f, i, args.color, args.logx, args.logy
      )
      for i, f in enumerate(args.files)
    ]
    concurrent.futures.wait(futures)


if __name__ == "__main__":
  main()
