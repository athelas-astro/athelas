from concurrent.futures import ProcessPoolExecutor
import glob
import re

from shocktube import plot_shocktube


def plot_worker(chk):
  print(chk)
  plot_shocktube(chk)


if __name__ == "__main__":
  # Get list of matching .h5 files
  file_list = glob.glob("sod_*.h5")

  # Extract the identifier part using regex
  identifiers = []
  for filename in file_list:
    match = re.search(r"sod_(.*)\.h5", filename)
    if match:
      identifiers.append(match.group(1))

  # Sort numerically where possible
  identifiers.sort(key=lambda x: int(x) if x.isdigit() else float("inf"))

  with ProcessPoolExecutor(max_workers=4) as executor:
    executor.map(plot_worker, identifiers)
