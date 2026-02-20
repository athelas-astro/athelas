#!/usr/bin/env python3
"""
Athelas HDF5 loader with dynamic field/variable discovery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union, Any

from astropy import constants as consts
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Errors
# ============================================================================


class AthelasError(RuntimeError):
  pass


# ============================================================================
# Data containers
# ============================================================================

ParamValue = Union[
  int,
  float,
  bool,
  str,
  np.typing.NDArray[Any],
  Dict[str, Any],
]


class Quadrature:
  def __init__(self, qs: np.typing.NDArray[Any], ws: np.typing.NDArray[Any]):
    self.nodes = qs
    self.weights = ws


class Field:
  def __init__(self, name: str, dataset: h5py.Dataset, meta: dict):
    self.name = name
    self.dataset = dataset
    self.nvars = int(meta["nvars"])
    self.rank = int(meta["rank"])
    self.policy = meta["policy"]
    self.description = meta["description"]

  def slice(self, sl):
    return self.dataset[sl, :, :]


class Variable:
  def __init__(self, name: str, field: Field, index: int):
    self.name = name
    self.field = field
    self.index = index

  def get(self, sl):
    return self.field.dataset[sl, :, self.index]


# ============================================================================
# Main dataset
# ============================================================================


class Athelas:
  def __init__(
    self, filename: Union[str, Path], load_ghost_cells: bool = False
  ):
    self.filename = Path(filename)
    self.load_ghost_cells = load_ghost_cells

    self.fields: Dict[str, Field] = {}
    self.variables: Dict[str, Variable] = {}
    self._derived = {}
    self.params: Dict[str, ParamValue] = {}

    self._load()

  # end __init__

  def __str__(self) -> str:
    return self.info()

  # end __str__

  def __repr__(self):
    return (
      f"Athelas('{self.filename.name}', cycle={self.cycle}, t={self.time:.6e})"
    )

  # end __repr__

  def _load(self):
    try:
      self._f = h5py.File(self.filename, "r")
    except Exception as e:
      raise AthelasError(f"Failed to open {self.filename}: {e}")

    f = self._f

    # ----------------------
    # Simulation info
    # ----------------------
    self.time = float(f["info/time"][()])
    self.n_stages = int(f["info/n_stages"][()])
    self.cycle = int(f["info/cycle"][()])

    self.params = read_hdf5_group(f["params"])
    self.geometry = self.params["problem.geometry"]

    # ----------------------
    # Grid
    # ----------------------
    if self.load_ghost_cells:
      self.sl = slice(None)
    else:
      self.sl = slice(1, -1)

    self.r = f["mesh/r"][self.sl]
    self.r_q = f["mesh/r_q"][self.sl, 1:-1].reshape(-1)
    self.r_q_u = f["mesh/r_q"][self.sl].reshape(-1)
    self.dr = f["mesh/dr"][self.sl]
    self.sqrt_gm = f["mesh/sqrt_gm"][self.sl]
    self.dm = f["mesh/dm"][self.sl]
    self.mass = f["mesh/enclosed_mass"][self.sl]
    if self.geometry == "spherical":
      self.dm *= 4.0 * np.pi

    # ----------------------
    # Load fields
    # ----------------------
    reg = f["metadata/field_registry"]

    for field_name in reg:
      if f"fields/{field_name}" not in f:
        continue

      meta = {
        "nvars": reg[field_name]["nvars"][()],
        "rank": reg[field_name]["rank"][()],
        "policy": reg[field_name]["policy"][()],
        "description": reg[field_name]["description"][()],
      }

      ds = f[f"fields/{field_name}"]
      self.fields[field_name] = Field(field_name, ds, meta)

    # ----------------------
    # Load variables
    # ----------------------
    for varname in f["metadata/variables"]:
      vgrp = f["metadata/variables"][varname]
      index = int(vgrp["index"][()])
      field_name = vgrp["location"][()].astype(str)[0]

      if field_name not in self.fields.keys():
        continue

      self.variables[varname] = Variable(
        varname, self.fields[field_name], index
      )

    # ----------------------
    # Basis
    # ----------------------
    self.basis = f[f"basis/"]
    self.quadrature = Quadrature(
      self.basis["nodes"][:], self.basis["weights"][:]
    )

    # ----------------------
    # Derived
    # ----------------------
    self._derived_registry = {
      "flux_factor": self._derive_flux_factor,
    }

  # end _load

  # ------------------------------------------------------------------
  # Derived
  # ------------------------------------------------------------------

  def _derive_flux_factor(self) -> np.ndarray:
    """
    Radiation flux factor f = |F| / (c E)
    """
    # Ensure required physics exists
    self.require(
      "rad_energy",
      "rad_momentum",
      basis="radiation",
    )

    E = self.get("rad_energy")
    F = self.get("rad_momentum")

    return np.abs(F) / (consts.c.cgs.value * E)

  # end _derive_flux_factor

  # ------------------------------------------------------------------
  # Accessors
  # ------------------------------------------------------------------

  #  def get(self, name: str) -> np.ndarray:
  #    if name in self.variables:
  #      return self.variables[name].get(self.sl)
  #
  #    if name in self.fields:
  #      return self.fields[name].slice(self.sl)
  #
  #    if name in self._derived_registry:
  #      key = name
  #      if key not in self._derived:
  #        self._derived[key] = self._derived_registry[name]()
  #      return self._derived[key]
  #
  #    raise AthelasError(f"Unknown variable or field '{name}'")

  def get(self, name: str, average: bool = True) -> np.ndarray:
    """
    Get variable, field, or derived quantity.

    Parameters
    ----------
    name : str
        Variable, field, or derived quantity name
    average : bool, default=True
        If True, return cell averages for nodal data.
        If False, return full nodal data.

    Returns
    -------
    np.ndarray
        Shape (nx,) if average=True, (nx, nnodes) if average=False
    """
    # Get the raw data
    if name in self.variables:
      data = self.variables[name].get(self.sl)
    elif name in self.fields:
      data = self.fields[name].slice(self.sl)
    elif name in self._derived_registry:
      key = (name,)
      if key not in self._derived:
        self._derived[key] = self._derived_registry[name]()
      data = self._derived[key]
    else:
      raise AthelasError(f"Unknown variable or field '{name}'")

    # If not averaging or data is already 1D, return as-is
    if not average or data.ndim == 1:
      return data

    # Handle variables with interface points (nnodes + 2)
    nnodes = len(self.quadrature.weights)
    if data.shape[1] > nnodes:
      # Strip interface points (first and last in second dimension)
      data = data[:, 1:-1]

    # Check cache for cell-averaged version
    cache_key = (name, "average")
    if cache_key not in self._derived:
      # Compute cell averages: shape (nx, nnodes) -> (nx,)
      weights = self.quadrature.weights
      self._derived[cache_key] = np.sum(
        data * weights[np.newaxis, :], axis=1
      ) / np.sum(weights[np.newaxis, :])
      # self._derived[cache_key] = np.sum(data * weights[np.newaxis, :] * self.sqrt_gm[:, 1:-1], axis=1) / np.sum(weights[np.newaxis, :] * self.sqrt_gm[:, 1:-1])

    return self._derived[cache_key]

  def __getitem__(self, name: str) -> np.ndarray:
    return self.get(name)

  def info(self) -> str:
    lines = []

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    lines.append("Athelas dataset")
    lines.append("-" * 60)
    lines.append(f"File:        {self.filename}")
    lines.append(f"Cycle:       {self.cycle}")
    lines.append(f"Time:        {self.time:.6e}")
    lines.append(f"DG stages:   {self.n_stages}")
    lines.append(f"Zones:       {self.r.size}")
    lines.append("")

    lines.append("Fields:")
    for name, field in self.fields.items():
      lines.append(
        f"  {name:6s} | nvars={field.nvars:3d} "
        f"| rank={field.rank} | policy={field.policy}"
      )
    lines.append("")

    lines.append("Variables:")
    for name, var in self.variables.items():
      lines.append(f"  {name:20s} → {var.field.name}[{var.index}]")
    lines.append("")

    if self.basis:
      lines.append("Basis functions:")
      for phys, comps in self.basis.items():
        names = ", ".join(comps.keys())
        lines.append(f"  {phys}: {names}")
      lines.append("")

    return "\n".join(lines)

  # end info

  def require(
    self,
    *vars: str,
    field: Optional[str] = None,
    basis: Optional[str] = None,
    composition: bool = False,
    ionization: bool = False,
  ) -> None:
    """
    Assert required data are present in the dataset.

    Examples
    --------
    ds.require("density", "pressure")
    ds.require(field="u_cf")
    ds.require(basis="radiation")
    ds.require(composition=True)
    """

    missing = []

    # ------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------
    for v in vars:
      if v not in self.variables:
        missing.append(f"variable '{v}'")

    # ------------------------------------------------------------
    # Field
    # ------------------------------------------------------------
    if field is not None:
      if field not in self.fields:
        missing.append(f"field '{field}'")

    # ------------------------------------------------------------
    # Basis
    # ------------------------------------------------------------
    if basis is not None:
      if not hasattr(self, "basis") or basis not in self.basis:
        missing.append(f"basis '{basis}'")

    # ------------------------------------------------------------
    # Optional physics modules
    # ------------------------------------------------------------
    if composition and "composition" not in self._f:
      missing.append("composition data")

    if ionization and "ionization" not in self._f:
      missing.append("ionization data")

    # ------------------------------------------------------------
    if missing:
      raise AthelasError(
        "Dataset is missing required data:\n  - " + "\n  - ".join(missing)
      )

  # end require

  # ------------------------------------------------------------------
  # Plotting
  # ------------------------------------------------------------------

  def plot(
    self,
    name: str,
    *,
    ax: Optional[plt.Axes] = None,
    logx: bool = False,
    logy: bool = False,
    label: Optional[str] = None,
    **kwargs,
  ) -> plt.Axes:
    if ax is None:
      _, ax = plt.subplots(figsize=(8, 5))

    x = self.r
    y = self.get(name)

    if logx:
      x = np.log10(np.abs(x))
      ax.set_xlabel("log10(r)")
    else:
      ax.set_xlabel("r")

    if logy:
      y = np.log10(np.abs(y))
      ax.set_ylabel(f"log10({name})")
    else:
      ax.set_ylabel(name)

    if label is None:
      label = f"{name} "

    ax.plot(x, y, label=label, **kwargs)
    ax.grid(alpha=0.3)
    ax.legend()

    return ax

  # ------------------------------------------------------------------


def read_hdf5_group(group: h5py.Group) -> Dict[str, ParamValue]:
  """
  Recursively read an HDF5 group into a nested Python dict.
  """
  result: Dict[str, ParamValue] = {}

  for key, item in group.items():
    if isinstance(item, h5py.Group):
      result[key] = read_hdf5_group(item)

    elif isinstance(item, h5py.Dataset):
      data = item[()]

      # Variable-length string
      if isinstance(data, bytes):
        data = str(data.decode())

      # NumPy array
      elif isinstance(data, np.ndarray):
        # Scalar array (shape == ())
        if data.shape == ():
          data = data.item()
        # Length-1 vector → scalar
        elif data.size == 1:
          data = data.reshape(()).item()

      result[key] = data

  return result
