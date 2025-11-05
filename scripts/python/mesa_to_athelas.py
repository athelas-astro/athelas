import argparse
import re
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import mesa_reader as mr
from astropy import constants as consts

# globals
Rsun = consts.R_sun.cgs.value

# Dictionary of element symbols to atomic numbers
# fmt: off
ELEMENTS = {
    'h': 1, 'he': 2, 'li': 3, 'be': 4, 'b': 5, 'c': 6, 'n': 7, 'o': 8, 'f': 9, 
    'ne': 10, 'na': 11, 'mg': 12, 'al': 13, 'si': 14, 'p': 15, 's': 16, 
    'cl': 17, 'ar': 18, 'k': 19, 'ca': 20, 'sc': 21, 'ti': 22, 'v': 23, 
    'cr': 24, 'mn': 25, 'fe': 26, 'co': 27, 'ni': 28, 'cu': 29, 'zn': 30, 
    'ga': 31, 'ge': 32, 'as': 33, 'se': 34, 'br': 35, 'kr': 36, 'rb': 37, 
    'sr': 38, 'y': 39, 'zr': 40, 'nb': 41, 'mo': 42, 'tc': 43, 'ru': 44, 
    'rh': 45, 'pd': 46, 'ag': 47, 'cd': 48, 'in': 49, 'sn': 50, 'sb': 51, 
    'te': 52, 'i': 53, 'xe': 54, 'cs': 55, 'ba': 56, 'la': 57, 'ce': 58, 
    'pr': 59, 'nd': 60, 'pm': 61, 'sm': 62, 'eu': 63, 'gd': 64, 'tb': 65, 
    'dy': 66, 'ho': 67, 'er': 68, 'tm': 69, 'yb': 70, 'lu': 71, 'hf': 72, 
    'ta': 73, 'w': 74, 're': 75, 'os': 76, 'ir': 77, 'pt': 78, 'au': 79, 
    'hg': 80, 'tl': 81, 'pb': 82, 'bi': 83, 'po': 84, 'at': 85, 'rn': 86, 
    'fr': 87, 'ra': 88, 'ac': 89, 'th': 90, 'pa': 91, 'u': 92, 'np': 93, 
    'pu': 94, 'am': 95, 'cm': 96, 'bk': 97, 'cf': 98, 'es': 99, 'fm': 100,
    'md': 101, 'no': 102, 'lr': 103, 'rf': 104, 'db': 105, 'sg': 106, 'bh': 107,
    'hs': 108, 'mt': 109, 'ds': 110, 'rg': 111, 'cn': 112, 'nh': 113, 'fl': 114, 
    'mc': 115, 'lv': 116, 'ts': 117, 'og': 118
}
# fmt: on


def parse_mesa_species(species_str):
  """Parse MESA species string into Z and N using a custom element dictionary."""
  if species_str == "neut":
    return 0, 1
  if species_str == "prot":
    return 1, 0

  match = re.match(r"([a-z]+)(\d+)", species_str.lower())
  if not match:
    raise ValueError(f"Invalid species format: {species_str}")

  element_symbol, mass_number = match.groups()
  mass_number = int(mass_number)

  if element_symbol not in ELEMENTS:
    raise ValueError(f"Unknown element: {element_symbol}")

  Z = ELEMENTS[element_symbol]
  N = mass_number - Z

  return Z, N


class MESAProfile:
  """
  Wrapper around mesa_reader to simplify MESA profile reading and Athelas
  input construction.

  Attributes:
      profile: mesa_reader.MesaData object
      header (dict): Header information from the profile file
      columns (list): List of available column names
  """

  def __init__(
    self,
    logs_dir: Optional[Union[str, Path]] = None,
    profile_number: Optional[int] = None,
  ) -> None:
    """
    Initialize and load a MESA profile.

    Args:
        logs_dir: Path to LOGS directory. If None, uses current directory.
        profile_number: Specific profile number to load.
            If None, loads the final profile.
    """
    self.logs_dir: Path = Path(logs_dir) if logs_dir else Path(".")

    if profile_number is not None:
      # Load specific profile
      self.profile: mr.MesaData = mr.MesaData(
        str(self.logs_dir / f"profile{profile_number}.data")
      )
    else:
      # Load final profile using mesa_reader's logic
      self.profile: mr.MesaData = self._load_final_profile()

    # Cache header and columns for easy access
    self.header: Dict[str, Any] = self._extract_header()
    self.columns: List[str] = list(self.profile.bulk_names)

  def _load_final_profile(self) -> mr.MesaData:
    """
    Load the final profile using mesa_reader's MesaLogDir.
    This properly handles MESA's profile numbering.
    """
    log_dir = mr.MesaLogDir(str(self.logs_dir))
    # Get the final profile number from profiles.index (last element)
    final_profile_num: int = log_dir.profile_numbers[-1]
    profile_path: Path = self.logs_dir / f"profile{final_profile_num}.data"
    return mr.MesaData(str(profile_path))

  def _extract_header(self) -> Dict[str, Any]:
    """Extract header information into a dictionary."""
    header: Dict[str, Any] = {}
    for name in self.profile.header_names:
      header[name] = self.profile.header_data[name]
    return header

  def get(self, column_name: str) -> np.ndarray:
    """
    Get data for a specific column.

    Args:
        column_name: Name of the column

    Returns:
        Column data as numpy array
    """
    if column_name not in self.profile.bulk_names:
      raise KeyError(
        f"Column '{column_name}' not found. "
        f"Available columns: {self.columns[:20]}..."
      )
    return self.profile.data(column_name)

  def __getitem__(self, key: str) -> np.ndarray:
    """Allow dictionary-style string access to columns."""
    return self.get(key)

  @property
  def n_zones(self) -> int:
    """Number of zones in the profile."""
    return len(self.profile.data(self.columns[0]))

  def __repr__(self) -> str:
    model_num = self.header.get("model_number", "unknown")
    return (
      f"MESAProfile(model={model_num}, "
      f"{len(self.columns)} columns, {self.n_zones} zones)"
    )

  def get_mass_fraction_columns(self) -> List[str]:
    """
    Get all mass fraction column names from the profile.

    Returns:
        List of mass fraction column names (e.g., 'h1', 'he4', 'c12', etc.)
    """

    # hacky..
    badlist = ["pnhe4", "z2bar"]
    extralist = ["neut", "prot"]
    # Mass fractions typically don't have underscores and are lowercase isotope names
    # Common patterns: h1, he3, he4, c12, n14, o16, ne20, mg24, si28, fe56, etc.
    mass_fractions = []
    for col in self.columns:
      # Check if it looks like an isotope name
      if (
        col.islower()
        and len(col) <= 5
        and any(c.isdigit() for c in col)
        and col not in badlist
        or col in extralist
      ):
        # Additional check: starts with element symbol pattern
        if col[0].isalpha():
          mass_fractions.append(col)
    return mass_fractions

  def write_athelas_inputs(
    self, hydro_file: str = "hydro.prof", comps_file: str = "comps.prof"
  ) -> None:
    """
    Write MESA profile data to separate output files.

    Args:
        hydro_file: Filename for hydrodynamic quantities
            (radius, density, velocity, pressure)
        comps_file: Filename for composition (all mass fractions)

    Note:
        Arrays are flipped so that the center (interior) comes first,
        since MESA profiles start at the surface.
    """
    # Get hydrodynamic quantities and flip them (MESA goes surface->center)
    # Here we access data using the __getitem__ accessors.
    # We can equivalently do density = self.get("density")
    radius = self["radius"][::-1] * Rsun
    density = self["rho"][::-1]
    velocity = self["velocity"][::-1]
    pressure = np.power(10.0, self["logPgas"][::-1])
    temperature = np.power(10.0, self["logT"][::-1])
    luminosity = self["luminosity"][::-1]

    # Write hydrodynamic file with units in CGS
    hydro_data = np.column_stack(
      [radius, density, velocity, pressure, temperature, luminosity]
    )
    hydro_header = "# radius [cm] density [g/cm^3] velocity [cm/s] pressure [erg/cm^3] temperature [K] luminosity [erg/cm^3/s]"
    np.savetxt(hydro_file, hydro_data, header=hydro_header, comments="")
    print(f"Wrote hydrodynamic data to {hydro_file}")

    # Get mass fractions
    mass_frac_cols = self.get_mass_fraction_columns()
    if not mass_frac_cols:
      print("Warning: No mass fraction columns found")
      return

    # Build composition array and flip

    # combine prot into h1
    if "prot" in mass_frac_cols:
      self["h1"][::-1] += self["prot"][::-1]
      mass_frac_cols.remove("prot")

    Z_values = []
    N_values = []

    for species in mass_frac_cols:
      Z, N = parse_mesa_species(species)
      Z_values.append(str(Z))
      N_values.append(str(N))

    Z_line = " ".join(Z_values)
    N_line = " ".join(N_values)
    mass_frac_data = np.column_stack(
      [self[col][::-1] for col in mass_frac_cols]
    )

    # We replace 0.0 mass fractions with a small, effectively zero value.
    # There are some places in the code (Saha ionization) where these values
    # can be prolematic
    SMALL = 1.0e-50
    mass_frac_data[mass_frac_data == 0.0] = SMALL
    comps_header = Z_line + "\n" + N_line
    np.savetxt(comps_file, mass_frac_data, header=comps_header, comments="")
    print(
      f"Wrote composition data ({len(mass_frac_cols)} species) to {comps_file}"
    )


def load_final_profile(
  logs_dir: Optional[Union[str, Path]] = None,
) -> MESAProfile:
  """
  Convenience function to load the final profile from a MESA run.

  Args:
      logs_dir: Path to LOGS directory. If None, uses current directory.

  Returns:
      Loaded profile object
  """
  return MESAProfile(logs_dir=logs_dir)


def main() -> None:
  """Main function with argument parsing."""
  parser = argparse.ArgumentParser(
    description="Load and display MESA stellar profile data"
  )
  parser.add_argument("logs_dir", type=str, help="Path to MESA LOGS directory")
  parser.add_argument(
    "-p",
    "--profile-number",
    type=int,
    default=None,
    help="Specific profile number to load (default: final profile)",
  )
  parser.add_argument(
    "--hydro-file",
    type=str,
    default="profile_hydro.dat",
    help="Output filename for hydrodynamic data (default: profile_hydro.dat)",
  )
  parser.add_argument(
    "--comps-file",
    type=str,
    default="profile_comps.dat",
    help="Output filename for composition data (default: profile_comps.dat)",
  )
  parser.add_argument("--write", action="store_true", help="Write output files")

  args = parser.parse_args()

  # Load the profile
  profile = MESAProfile(
    logs_dir=args.logs_dir, profile_number=args.profile_number
  )

  # Access header information
  print("Model number:", profile.header.get("model_number"))
  print("Star mass (Msun):", profile.header.get("star_mass"))
  print("Star age (years):", profile.header.get("star_age"))
  print("Profile:", profile)

  # Access specific columns
  density: np.ndarray = profile["rho"]

  print(
    f"\nLoaded profile with {len(profile.columns)} columns and {profile.n_zones} zones"
  )
  print(f"\nFirst few columns: {profile.columns[:]}")

  # Example: print central vs surface values
  print(f"\nCentral density: {density[-1]:.3e} g/cm^3")
  print(f"Surface density: {density[0]:.3e} g/cm^3")

  # Write output files if requested
  if args.write:
    print("\nWriting output files...")
    profile.write_athelas_inputs(
      hydro_file=args.hydro_file, comps_file=args.comps_file
    )


if __name__ == "__main__":
  main()
