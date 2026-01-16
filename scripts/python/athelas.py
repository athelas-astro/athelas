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


class Field:
  def __init__(self, name: str, dataset: h5py.Dataset, meta: dict):
    self.name = name
    self.dataset = dataset
    self.nvars = int(meta["nvars"])
    self.rank = int(meta["rank"])
    self.policy = meta["policy"]
    self.description = meta["description"]

  def slice(self, sl, mode: int):
    return self.dataset[sl, mode, :]


class Variable:
  def __init__(self, name: str, field: Field, index: int):
    self.name = name
    self.field = field
    self.index = index

  def get(self, sl, mode: int):
    return self.field.dataset[sl, mode, self.index]


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
    self.time = float(f["simulation_info/time"][()])
    self.n_stages = int(f["simulation_info/n_stages"][()])
    self.cycle = int(f["simulation_info/cycle"][()])

    self.params = read_hdf5_group(f["params"])

    # ----------------------
    # Grid
    # ----------------------
    if self.load_ghost_cells:
      self.sl = slice(None)
    else:
      self.sl = slice(1, -1)

    self.r = f["mesh/r"][self.sl]
    self.r_q = f["mesh/r_q"][self.sl]
    self.dr = f["mesh/dr"][self.sl]

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
    self.basis = {}
    if "basis" in f:
      for phys in f["basis"]:
        self.basis[phys] = {}
        for name in f[f"basis/{phys}"]:
          self.basis[phys][name] = f[f"basis/{phys}/{name}"][...]

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

  def _derive_flux_factor(self, mode: int = 0) -> np.ndarray:
    """
    Radiation flux factor f = |F| / (c E)
    """
    # Ensure required physics exists
    self.require(
      "rad_energy",
      "rad_momentum",
      basis="radiation",
    )

    E = self.get("rad_energy", mode=mode)
    F = self.get("rad_momentum", mode=mode)

    return np.abs(F) / (consts.c.cgs.value * E)

  # end _derive_flux_factor

  # ------------------------------------------------------------------
  # Accessors
  # ------------------------------------------------------------------

  def get(self, name: str, mode: int = 0) -> np.ndarray:
    if name in self.variables:
      return self.variables[name].get(self.sl, mode)

    if name in self.fields:
      return self.fields[name].slice(self.sl, mode)

    if name in self._derived_registry:
      key = (name, mode)
      if key not in self._derived:
        self._derived[key] = self._derived_registry[name](mode)
      return self._derived[key]

    raise AthelasError(f"Unknown variable or field '{name}'")

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
    mode: int = 0,
    ax: Optional[plt.Axes] = None,
    logx: bool = False,
    logy: bool = False,
    label: Optional[str] = None,
    **kwargs,
  ) -> plt.Axes:
    if ax is None:
      _, ax = plt.subplots(figsize=(8, 5))

    x = self.r
    y = self.get(name, mode)

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
      label = f"{name} (mode {mode})" if mode else name

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
