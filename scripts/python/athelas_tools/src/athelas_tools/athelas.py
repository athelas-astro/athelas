#!/usr/bin/env python3
"""
Athelas HDF5 loader with dynamic field/variable discovery.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Callable, Dict, Optional, Sequence, Union, Any

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

TRACKER_LABELS = {
  "photosphere": "Photosphere",
  "shock": "Shock",
}


class Quadrature:
  def __init__(self, qs: np.typing.NDArray[Any], ws: np.typing.NDArray[Any]):
    self.nodes = qs
    self.weights = ws


class Field:
  def __init__(self, name: str, dataset: h5py.Dataset, meta: dict):
    self.name = name
    self.dataset = dataset
    self.nvars = int(meta["nvars"][0])
    self.rank = int(meta["rank"][0])
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


class History:
  def __init__(self, path: Path):
    self.path = path
    self.columns, self.data = load_history(path)

  def nearest_row(self, time: float) -> np.typing.NDArray[Any]:
    return self.data[np.argmin(np.abs(self.data[:, 0] - time))]

  def value(self, name: str, time: float) -> float:
    if name not in self.columns:
      raise AthelasError(f"History file {self.path} has no column '{name}'")
    return float(self.nearest_row(time)[self.columns[name]])

  def breakout_time(self, *, interpolate: bool = True) -> Optional[float]:
    """Time of shock breakout: the first time the shock reaches the photosphere.

    Breakout is defined as the shock radius first crossing (reaching or
    exceeding) the photosphere radius. Only rows with a finite shock radius and
    a valid, finite photosphere radius are considered. Returns ``None`` if the
    required tracker columns are absent or breakout never occurs in the run.

    When ``interpolate`` is true (default) the crossing time is linearly
    interpolated between the bracketing history samples, since the trackers are
    only recorded at output cadence; otherwise the first post-crossing sample
    time is returned.
    """
    needed = ("Shock Radius [cm]", "Photosphere Radius [cm]")
    if any(col not in self.columns for col in needed):
      return None

    t = self.data[:, 0]
    r_shock = self.data[:, self.columns["Shock Radius [cm]"]]
    r_ph = self.data[:, self.columns["Photosphere Radius [cm]"]]

    valid = np.isfinite(r_shock) & np.isfinite(r_ph)
    if "Photosphere Valid" in self.columns:
      valid &= self.data[:, self.columns["Photosphere Valid"]] >= 0.5
    if not np.any(valid):
      return None

    tv = t[valid]
    # gap = shock - photosphere; >= 0 once the shock has reached the surface.
    gap = r_shock[valid] - r_ph[valid]
    above = gap >= 0.0
    if not np.any(above):
      return None

    first = int(np.argmax(above))
    if first == 0 or not interpolate:
      return float(tv[first])

    g0, g1 = gap[first - 1], gap[first]
    if g1 == g0:
      return float(tv[first])
    frac = (0.0 - g0) / (g1 - g0)
    return float(tv[first - 1] + frac * (tv[first] - tv[first - 1]))


# ============================================================================
# Main dataset
# ============================================================================


class Athelas:
  def __init__(
    self,
    filename: Union[str, Path],
    load_ghost_cells: bool = False,
    history: Union[None, str, Path, History] = None,
  ):
    """Load an Athelas checkpoint.

    ``history`` controls how the run's history (``.hst``) is resolved, which the
    tracker and breakout helpers read:

    * ``None`` (default) -- derive the location from the output parameters next
      to the checkpoint (see :meth:`history_path`).
    * ``str`` / ``Path`` -- load the history from this explicit location.
    * ``History`` -- reuse an already-parsed history object. History is a
      per-run artifact, so sharing one instance across many checkpoints avoids
      re-reading and re-parsing the same file.
    """
    self.filename = Path(filename)
    self.load_ghost_cells = load_ghost_cells

    self.fields: Dict[str, Field] = {}
    self.variables: Dict[str, Variable] = {}
    self._derived = {}
    self.params: Dict[str, ParamValue] = {}

    self._history: Optional[History] = None
    self._history_path: Optional[Path] = None
    if isinstance(history, History):
      self._history = history
    elif history is not None:
      self._history_path = Path(history)

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
    self.time = float(f["info/time"][0])
    self.n_stages = int(f["info/n_stages"][0])
    self.cycle = int(f["info/last_cycle"][0])

    self.params = read_hdf5_group(f["params"])
    self.geometry = self.params["mesh.geometry"]

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
      index = int(vgrp["index"][0])
      field_name = vgrp["location"][0]
      field_name = field_name.decode("utf-8")

      if field_name not in self.fields.keys():
        continue

      self.variables[varname] = Variable(
        varname, self.fields[field_name], index
      )

    # ----------------------
    # Basis
    # ----------------------
    self.basis = f["basis/"]
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
      "specific_radiation_energy",
      "specific_radiation_flux",
      basis="radiation",
    )

    E = self.get("specific_radiation_energy")
    F = self.get("specific_radiation_flux")

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
    ds.require(field="evolved")
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

  def history_path(self) -> Path:
    hist_fn = as_string(self.params.get("output.hist_fn", "athelas.hst"))
    path = Path(hist_fn)
    if path.is_absolute() and path.exists():
      return path

    candidates = [
      self.filename.parent / path,
      Path(as_string(self.params.get("output.dir", self.filename.parent)))
      / path,
    ]
    for candidate in candidates:
      if candidate.exists():
        return candidate

    raise AthelasError(
      "Could not find history file for tracker overlay. "
      f"Tried: {', '.join(str(candidate) for candidate in candidates)}"
    )

  def history(self) -> History:
    if self._history is None:
      path = self._history_path or self.history_path()
      self._history = History(path)
    return self._history

  def breakout_time(self, *, interpolate: bool = True) -> Optional[float]:
    """Shock breakout time from the run's history (see History.breakout_time).

    Useful as an analysis ``t = 0`` reference. Returns ``None`` if the shock and
    photosphere trackers are not both present or breakout never occurs.
    """
    return self.history().breakout_time(interpolate=interpolate)

  @property
  def time_since_breakout(self) -> Optional[float]:
    """This checkpoint's time relative to shock breakout (``t - t_breakout``).

    Returns ``None`` if breakout cannot be determined from the history.
    """
    t_breakout = self.breakout_time()
    if t_breakout is None:
      return None
    return self.time - t_breakout

  def tracker_position(self, tracker: str) -> Optional[float]:
    tracker = tracker.lower()
    hist = self.history()

    if tracker == "photosphere":
      if "Photosphere Valid" in hist.columns:
        valid = hist.value("Photosphere Valid", self.time)
        if valid < 0.5:
          return None
      return hist.value("Photosphere Radius [cm]", self.time)

    if tracker == "shock":
      return hist.value("Shock Radius [cm]", self.time)

    raise AthelasError(f"Unknown tracker '{tracker}'")

  def tracker_positions(
    self, trackers: Union[bool, str, Sequence[str]] = True
  ) -> Dict[str, float]:
    if trackers is True:
      requested = ("photosphere", "shock")
    elif trackers is False:
      return {}
    elif isinstance(trackers, str):
      requested = (trackers,)
    else:
      requested = tuple(trackers)

    positions: Dict[str, float] = {}
    for tracker in requested:
      try:
        position = self.tracker_position(tracker)
      except AthelasError:
        continue
      if position is None or not np.isfinite(position):
        continue
      positions[tracker] = position
    return positions

  def plot_trackers(
    self,
    ax: Optional[plt.Axes] = None,
    trackers: Union[bool, str, Sequence[str]] = True,
    *,
    color: str = "k",
    linestyle: str = "--",
    linewidth: float = 1.0,
    alpha: float = 0.9,
    label: bool = True,
    position_transform: Optional[Callable[[float], float]] = None,
    **kwargs,
  ) -> plt.Axes:
    if ax is None:
      ax = plt.gca()

    positions = self.tracker_positions(trackers)
    for tracker, position in positions.items():
      if position_transform is not None:
        position = position_transform(position)
      line_label = TRACKER_LABELS.get(tracker, tracker) if label else None
      ax.axvline(
        position,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label=line_label,
        **kwargs,
      )

    if label and positions:
      ax.legend()

    return ax

  def plot(
    self,
    name: str,
    *,
    ax: Optional[plt.Axes] = None,
    logx: bool = False,
    logy: bool = False,
    label: Optional[str] = None,
    overlay_trackers: Union[bool, str, Sequence[str]] = False,
    tracker_kwargs: Optional[dict[str, Any]] = None,
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
    if overlay_trackers:
      if logx:
        tracker_kwargs = dict(tracker_kwargs or {})
        tracker_kwargs.setdefault(
          "position_transform", lambda value: np.log10(np.abs(value))
        )
      self.plot_trackers(
        ax=ax, trackers=overlay_trackers, **(tracker_kwargs or {})
      )
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

      if isinstance(data, bytes):
        data = str(data.decode())

      result[key] = data

  return result


def parse_history_header(line: str) -> Dict[str, int]:
  # The header is guaranteed to start with '#', followed by "<index> <name>"
  # pairs, e.g. "# 0 Time [s] 1 Total Mass [g] ...". Names may contain spaces
  # and brackets; column indices are standalone integers. Match each name
  # lazily up to the next index token (or end of line).
  body = line.lstrip("#").strip()
  pairs = re.findall(r"(\d+)\s+(.+?)(?=\s+\d+\s+|$)", body)
  return {name.strip(): int(index) for index, name in pairs}


def load_history(path: Path) -> tuple[Dict[str, int], np.typing.NDArray[Any]]:
  columns: Optional[Dict[str, int]] = None
  rows: list[list[float]] = []
  for line in path.read_text().splitlines():
    if not line.strip():
      continue
    if line.startswith("#"):
      columns = parse_history_header(line)
    else:
      rows.append([float(value) for value in line.split()])

  if columns is None:
    raise AthelasError(f"History file {path} has no header")
  if not rows:
    raise AthelasError(f"History file {path} has no rows")

  return columns, np.asarray(rows)


def as_string(value: Any) -> str:
  if isinstance(value, bytes):
    return value.decode("utf-8")
  if isinstance(value, np.ndarray) and value.shape == ():
    return as_string(value.item())
  return str(value)
