from concurrent.futures import ProcessPoolExecutor
import glob
import re

from shocktube import plot_shocktube
from plot_supernova_profiles import plot_supernova_profiles
from plot_supernova_comps import plot_supernova_comps

def plot_worker(chk):
  print(chk)
  plot_supernova_profiles(chk)


if __name__ == "__main__":
  # Get list of matching .h5 files
  file_list = glob.glob("supernova_*.h5")

  # Extract the identifier part using regex
  identifiers = []
  for filename in file_list:
    match = re.search(r"supernova_(.*)\.h5", filename)
    if match:
      identifiers.append(match.group(1))

  # Sort numerically where possible
  identifiers.sort(key=lambda x: int(x) if x.isdigit() else float("inf"))

  with ProcessPoolExecutor(max_workers=2) as executor:
    executor.map(plot_worker, identifiers)
