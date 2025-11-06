#!/usr/bin/env python3
"""
This module provides tools for loading, analyzing, and visualizing Athelas results.
"""

from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path
import warnings

import h5py
import numpy as np
import matplotlib.pyplot as plt

try:
  from basis import ModalBasis  # type: ignore
except ImportError:
  warnings.warn(
    "basis module not found. ModalBasis functionality will be disabled."
  )


class AthelasError(Exception):
  """Custom exception for Athelas-related errors."""

  pass


class Athelas:
  """
  A class for loading and analyzing Athelas data.

  Attributes:
      time (float): Simulation time
      sOrder (int): Spatial order of the DG scheme
      nX (int): Number of spatial cells
      r (np.ndarray): Cell center coordinates
      r_nodal (np.ndarray): Nodal coordinates
      dr (np.ndarray): Cell widths
      uCF (np.ndarray): Conserved variables [nX, sOrder, nVars]
      uAF (np.ndarray): Auxiliary variables [nX, sOrder, nVars]
      basis (ModalBasis): Modal basis functions (optional)
  """

  # Default variable indices - can be customized
  DEFAULT_VARIABLES = {
    "tau": 0,
    "velocity": 1,
    "energy": 2,
    "rad_energy": 3,
    "rad_flux": 4,
  }

  AUXILIARY = {
    "pressure": 0,
    "t_gas": 1,
  }

  def __init__(
    self,
    filename: Union[str, Path],
    basis_filename: Optional[Union[str, Path]] = None,
    load_ghost_cells: bool = False,
    variable_indices: Optional[Dict[str, int]] = None,
  ):
    """
    Initialize Athelas object by loading simulation data from HDF5 file.

    Args:
        filename: Path to HDF5 simulation output file
        basis_filename: Optional path to basis function file
        load_ghost_cells: Whether to include ghost cells in loaded data
        variable_indices: Custom mapping of variable names to indices

    Raises:
        AthelasError: If file loading fails or data is invalid
    """
    # Initialize attributes
    self.uCF: Optional[np.ndarray] = None
    self.uAF: Optional[np.ndarray] = None
    self.time: Optional[float] = None
    self.sOrder: Optional[int] = None
    self.nX: Optional[int] = None
    self.r: Optional[np.ndarray] = None
    self.r_nodal: Optional[np.ndarray] = None
    self.dr: Optional[np.ndarray] = None
    self.basis: Optional[ModalBasis] = None  # type: ignore

    # Set up variable indices
    self.idx = (
      variable_indices
      if variable_indices is not None
      else self.DEFAULT_VARIABLES.copy() | self.AUXILIARY.copy()
    )

    # Load data
    self._filename = Path(filename)
    self._load_ghost_cells = load_ghost_cells
    self._load_simulation_data()

    # Load basis if provided
    if basis_filename is not None:
      self._load_basis(basis_filename)

  def __str__(self) -> str:
    """Return string representation of Athelas object."""
    ghost_info = " (with ghost cells)" if self._load_ghost_cells else ""
    return (
      f"--- Athelas DG Data ---\n"
      f"File: {self._filename.name}\n"
      f"Time: {self.time:.6e}\n"
      f"Spatial Order: {self.sOrder}\n"
      f"Cells: {self.nX}{ghost_info}\n"
      f"Variables: {list(self.idx.keys())}\n"
      f"Basis loaded: {self.basis is not None}\n"
      f"Domain: [{self.r[0]:.3e}, {self.r[-1]:.3e}]"  # type: ignore
    )

  def __repr__(self) -> str:
    """Return detailed representation."""
    return f"Athelas('{self._filename}', nX={self.nX}, sOrder={self.sOrder}, t={self.time})"

  def _load_simulation_data(self) -> None:
    """Load simulation data from HDF5 file."""
    try:
      with h5py.File(self._filename, "r") as f:
        # Load metadata
        self.time = float(f["metadata/time"][0])  # type: ignore
        self.sOrder = int(f["metadata/order"][0])  # type: ignore
        self.nX = int(f["metadata/nx"][0])  # type: ignore

        # Determine slice for ghost cells
        if self._load_ghost_cells:
          slice_idx = slice(None)  # Load all cells
        else:
          slice_idx = slice(1, -1)  # Skip first and last (ghost) cells
          self.nX -= 2  # Adjust cell count

        # Load grid data
        self.r = f["grid/x"][slice_idx]  # type: ignore
        self.r_nodal = f["grid/x_nodal"][slice_idx]  # type: ignore
        self.dr = f["grid/dx"][slice_idx]  # type: ignore
        self.mass_coordinate = f["grid/enclosed_mass"][slice_idx]  # type: ignore

        # Load variable data
        self.uCF = f["variables/conserved"][slice_idx, ...]  # type: ignore
        self.uAF = f["variables/auxiliary"][slice_idx, ...]  # type: ignore

    except (OSError, KeyError, ValueError) as e:
      raise AthelasError(
        f"Failed to load simulation data from '{self._filename}': {e}"
      )

  def _load_basis(self, basis_filename: Union[str, Path]) -> None:
    """Load modal basis functions."""
    if ModalBasis is None:  # type: ignore
      warnings.warn("ModalBasis not available. Skipping basis loading.")
      return

    try:
      self.basis = ModalBasis(basis_filename)  # type: ignore
    except Exception as e:
      warnings.warn(f"Failed to load basis from '{basis_filename}': {e}")

  @property
  def variables(self) -> List[str]:
    """Get list of available variable names."""
    return list(self.idx.keys())

  @property
  def n_conserved_vars(self) -> int:
    """Number of conserved variables."""
    return self.uCF.shape[2] if self.uCF is not None else 0

  @property
  def n_auxiliary_vars(self) -> int:
    """Number of auxiliary variables."""
    return self.uAF.shape[2] if self.uAF is not None else 0

  @property
  def domain_bounds(self) -> Tuple[float, float]:
    """Get domain bounds as (min, max)."""
    return float(self.r[0]), float(self.r[-1])  # type: ignore

  def add_variable(
    self, name: str, index: int, var_type: str = "conserved"
  ) -> None:
    """
    Add a new variable to the index mapping.

    Args:
        name: Variable name
        index: Variable index in the data array
        var_type: Either "conserved" or "auxiliary"
    """
    if var_type not in ["conserved", "auxiliary"]:
      raise ValueError("var_type must be 'conserved' or 'auxiliary'")

    max_idx = (
      self.n_conserved_vars
      if var_type == "conserved"
      else self.n_auxiliary_vars
    )
    if index >= max_idx:
      raise ValueError(
        f"Index {index} exceeds available {var_type} variables ({max_idx})"
      )

    self.idx[name] = index

  def get_variable(
    self, var: str, mode: int = 0, var_type: str = "conserved"
  ) -> np.ndarray:
    """
    Get variable data for a specific mode.

    Args:
        var: Variable name
        mode: Modal coefficient index (default: 0 for cell average)
        var_type: Either "conserved" or "auxiliary"

    Returns:
        1D array of variable values across all cells

    Raises:
        AthelasError: If variable not found or invalid parameters
    """
    if var not in self.idx:
      available = ", ".join(self.variables)
      raise AthelasError(f"Unknown variable '{var}'. Available: {available}")

    if var_type == "conserved":
      data = self.uCF
    elif var_type == "auxiliary":
      data = self.uAF
    else:
      raise ValueError("var_type must be 'conserved' or 'auxiliary'")

    if mode >= self.sOrder:
      raise AthelasError(f"Mode {mode} exceeds spatial order {self.sOrder}")

    var_idx = self.idx[var]
    if var_idx >= data.shape[2]:
      raise AthelasError(
        f"Variable index {var_idx} exceeds available variables ({data.shape[2]})"
      )

    return data[:, mode, var_idx]

  def get(self, var: str, k: int = 0) -> np.ndarray:
    """Get conserved variable (backward compatibility method)."""
    return self.get_variable(var, mode=k, var_type="conserved")

  def evaluate_at_points(
    self, var: str, points: np.ndarray, var_type: str = "conserved"
  ) -> np.ndarray:
    """
    Evaluate variable at arbitrary points using basis functions.

    Args:
        var: Variable name
        points: Array of x-coordinates where to evaluate
        var_type: Either "conserved" or "auxiliary"

    Returns:
        Array of variable values at the specified points
    """
    if self.basis is None:
      raise AthelasError(
        "Basis functions not loaded. Cannot evaluate at arbitrary points."
      )

    # This is a placeholder - actual implementation would depend on basis structure
    raise NotImplementedError("Point evaluation not yet implemented")

  def basis_eval(self, var_idx: int, cell_idx: int, quad_point: int) -> float:
    """
    Evaluate polynomial for quantity at quadrature point using basis functions.

    Args:
        var_idx: Variable index
        cell_idx: Cell index
        quad_point: Quadrature point index

    Returns:
        Evaluated value
    """
    if self.basis is None:
      raise AthelasError("Basis functions not loaded")

    if cell_idx >= self.nX:
      raise AthelasError(
        f"Cell index {cell_idx} exceeds number of cells ({self.nX})"
      )

    result = 0.0
    for k in range(self.sOrder):
      result += (
        self.basis.phi[cell_idx, quad_point, k] * self.uCF[cell_idx, k, var_idx]
      )
    return result

  def plot_variable(
    self,
    var: str,
    ax: Optional[plt.Axes] = None,
    mode: int = 0,
    var_type: str = "conserved",
    logx: bool = False,
    logy: bool = False,
    label: Optional[str] = None,
    **kwargs,
  ) -> plt.Axes:
    """
    Plot a variable against spatial coordinate.

    Args:
        var: Variable name to plot
        ax: Matplotlib axes (creates new if None)
        mode: Modal coefficient to plot (0 = cell average)
        var_type: Either "conserved" or "auxiliary"
        logx: Use log scale for x-axis
        logy: Use log scale for y-axis
        label: Line label for legend
        **kwargs: Additional arguments passed to plot()

    Returns:
        The matplotlib axes object
    """
    if ax is None:
      fig, ax = plt.subplots(figsize=(10, 6))

    # Get data
    q = self.get_variable(var, mode=mode, var_type=var_type)
    x = self.r.copy()

    # Apply log transforms
    if logy:
      q = np.log10(np.abs(q))
      ax.set_ylabel(f"log10({var})")
    else:
      ax.set_ylabel(var)

    if logx:
      x = np.log10(np.abs(x))
      ax.set_xlabel("log10(x)")
    else:
      ax.set_xlabel("x")

    # Set default styling
    plot_kwargs = {"color": "#b07aa1", "linewidth": 2}
    plot_kwargs.update(kwargs)

    if label is None:
      label = f"{var} (mode {mode})" if mode > 0 else var

    ax.plot(x, q, label=label, **plot_kwargs)
    ax.grid(True, alpha=0.3)

    return ax

  # Backward compatibility
  def plot(self, ax: plt.Axes, var: str, **kwargs) -> None:
    """Plot method for backward compatibility."""
    self.plot_variable(var, ax=ax, **kwargs)

  def summary_plot(
    self,
    variables: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
  ) -> plt.Figure:
    """
    Create a summary plot showing multiple variables.

    Args:
        variables: List of variables to plot (uses all if None)
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    if variables is None:
      variables = self.variables[:4]  # Plot first 4 variables by default

    n_vars = len(variables)
    cols = 2
    rows = (n_vars + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
      axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i, var in enumerate(variables):
      try:
        self.plot_variable(var, ax=axes[i])
        axes[i].set_title(f"{var} at t = {self.time:.3e}")
      except AthelasError as e:
        axes[i].text(
          0.5,
          0.5,
          f"Error: {e}",
          transform=axes[i].transAxes,
          ha="center",
          va="center",
        )
        axes[i].set_title(f"{var} (error)")

    # Hide unused subplots
    for i in range(n_vars, len(axes)):
      axes[i].set_visible(False)

    fig.suptitle(f"Athelas Summary - {self._filename.name}")
    fig.tight_layout()

    return fig

  def save_data(
    self,
    filename: Union[str, Path],
    variables: Optional[List[str]] = None,
    format: str = "hdf5",
    mode: int = 0,
    var_type: str = "conserved",
    delimiter: str = "\t",
    include_grid: bool = True,
  ) -> None:
    """
    Save selected variables to file in various formats.

    Args:
        filename: Output filename (extension determines format if format='auto')
        variables: Variables to save (saves all if None)
        format: Output format - 'hdf5', 'numpy', 'text', or 'auto'
        mode: Modal coefficient to save (default: 0 for cell averages)
        var_type: Either "conserved" or "auxiliary"
        delimiter: Delimiter for text format (default: tab)
        include_grid: Whether to include grid coordinates

    Supported formats:
        - 'hdf5': HDF5 format with metadata and structure
        - 'numpy': NumPy .npz format (binary, compact)
        - 'text': Plain text/CSV format (human readable)
        - 'auto': Determine from file extension (.h5/.hdf5 -> hdf5,
                 .npz -> numpy, .txt/.csv/.dat -> text)
    """
    filepath = Path(filename)

    if variables is None:
      variables = self.variables

    # Auto-detect format from extension
    if format == "auto":
      ext = filepath.suffix.lower()
      if ext in [".h5", ".hdf5"]:
        format = "hdf5"
      elif ext == ".npz":
        format = "numpy"
      elif ext in [".txt", ".csv", ".dat"]:
        format = "text"
      else:
        # Default to text for unknown extensions
        format = "text"
        warnings.warn(f"Unknown extension '{ext}', defaulting to text format")

    if format == "hdf5":
      self._save_hdf5(filepath, variables, mode, var_type, include_grid)
    elif format == "numpy":
      self._save_numpy(filepath, variables, mode, var_type, include_grid)
    elif format == "text":
      self._save_text(
        filepath, variables, mode, var_type, delimiter, include_grid
      )
    else:
      raise ValueError(
        f"Unsupported format: {format}. Use 'hdf5', 'numpy', 'text', or 'auto'"
      )

  def _save_hdf5(
    self,
    filepath: Path,
    variables: List[str],
    mode: int,
    var_type: str,
    include_grid: bool,
  ) -> None:
    """Save data in HDF5 format with full metadata."""
    with h5py.File(filepath, "w") as f:
      # Save metadata
      meta_grp = f.create_group("metadata")  # type: ignore
      meta_grp["time"] = self.time  # type: ignore
      meta_grp["order"] = self.sOrder  # type: ignore
      meta_grp["nx"] = self.nX  # type: ignore
      meta_grp["mode"] = mode  # type: ignore
      meta_grp["var_type"] = var_type  # type: ignore
      meta_grp["source_file"] = str(self._filename)  # type: ignore

      if include_grid:
        # Save grid
        grid_grp = f.create_group("grid")  # type: ignore
        grid_grp["x"] = self.r  # type: ignore
        grid_grp["x_nodal"] = self.r_nodal  # type: ignore
        grid_grp["dx"] = self.dr  # type: ignore

      # Save variables
      var_grp = f.create_group("variables")  # type: ignore
      for var in variables:
        if var in self.idx:
          try:
            var_data = self.get_variable(var, mode=mode, var_type=var_type)
            var_grp[var] = var_data  # type: ignore
          except AthelasError as e:
            warnings.warn(f"Skipping variable '{var}': {e}")

  def _save_numpy(
    self,
    filepath: Path,
    variables: List[str],
    mode: int,
    var_type: str,
    include_grid: bool,
  ) -> None:
    """Save data in NumPy .npz format (binary, compact)."""
    data_dict = {}

    # Add metadata
    data_dict["_metadata_time"] = self.time
    data_dict["_metadata_order"] = self.sOrder
    data_dict["_metadata_nx"] = self.nX
    data_dict["_metadata_mode"] = mode
    data_dict["_metadata_var_type"] = var_type
    data_dict["_metadata_source_file"] = str(self._filename)

    if include_grid:
      data_dict["grid_x"] = self.r
      data_dict["grid_x_nodal"] = self.r_nodal
      data_dict["grid_dx"] = self.dr

    # Save variables
    for var in variables:
      if var in self.idx:
        try:
          var_data = self.get_variable(var, mode=mode, var_type=var_type)
          data_dict[var] = var_data
        except AthelasError as e:
          warnings.warn(f"Skipping variable '{var}': {e}")

    np.savez_compressed(filepath, **data_dict)

  def _save_text(
    self,
    filepath: Path,
    variables: List[str],
    mode: int,
    var_type: str,
    delimiter: str,
    include_grid: bool,
  ) -> None:
    """Save data in plain text format (human readable)."""
    # Collect data
    data_arrays = []
    headers = []

    if include_grid:
      data_arrays.extend([self.r, self.dr])
      headers.extend(["x", "dx"])

    # Collect variable data
    for var in variables:
      if var in self.idx:
        try:
          var_data = self.get_variable(var, mode=mode, var_type=var_type)
          data_arrays.append(var_data)
          var_name = f"{var}_mode{mode}" if mode > 0 else var
          headers.append(var_name)
        except AthelasError as e:
          warnings.warn(f"Skipping variable '{var}': {e}")

    if not data_arrays:
      raise AthelasError("No valid variables to save")

    # Stack data into columns
    data_matrix = np.column_stack(data_arrays)

    # Create header with metadata
    header_lines = [
      f"# Athelas simulation data extracted at t = {self.time:.6e}",
      f"# Source file: {self._filename}",
      f"# Spatial order: {self.sOrder}, Mode: {mode}, Type: {var_type}",
      f"# Number of cells: {self.nX}",
      f"# Columns: {delimiter.join(headers)}",
    ]
    header_text = "\n".join(header_lines)

    # Save to file
    np.savetxt(
      filepath,
      data_matrix,
      delimiter=delimiter,
      header=header_text,
      fmt="%.12e",
    )

  def export_modes(
    self,
    filename: Union[str, Path],
    variables: Optional[List[str]] = None,
    modes: Optional[List[int]] = None,
    format: str = "numpy",
  ) -> None:
    """
    Export multiple modes for selected variables.

    Args:
        filename: Output filename
        variables: Variables to export (all if None)
        modes: List of modes to export (all if None)
        format: Output format ('numpy', 'hdf5', or 'text')
    """
    if variables is None:
      variables = self.variables

    if modes is None:
      modes = list(range(self.sOrder))

    filepath = Path(filename)

    if format == "numpy":
      data_dict = {
        "_metadata_time": self.time,
        "_metadata_order": self.sOrder,
        "_metadata_nx": self.nX,
        "_metadata_source_file": str(self._filename),
        "grid_x": self.r,
        "grid_dx": self.dr,
      }

      for var in variables:
        if var in self.idx:
          for mode in modes:
            try:
              var_data = self.get_variable(var, mode=mode)
              data_dict[f"{var}_mode_{mode}"] = var_data
            except AthelasError as e:
              warnings.warn(f"Skipping {var} mode {mode}: {e}")

      np.savez_compressed(filepath, **data_dict)

    elif format == "text":
      # For text format, create separate files for each variable
      for var in variables:
        if var not in self.idx:
          continue

        var_filepath = (
          filepath.parent / f"{filepath.stem}_{var}{filepath.suffix}"
        )
        data_arrays = [self.r, self.dr]  # Always include grid
        headers = ["x", "dx"]

        for mode in modes:
          try:
            var_data = self.get_variable(var, mode=mode)
            data_arrays.append(var_data)
            headers.append(f"{var}_mode_{mode}")
          except AthelasError:
            continue

        if len(data_arrays) > 2:  # More than just grid
          data_matrix = np.column_stack(data_arrays)
          header_text = (
            f"# Variable: {var}, Time: {self.time:.6e}\n# Columns: "
            + "\t".join(headers)
          )
          np.savetxt(
            var_filepath,
            data_matrix,
            delimiter="\t",
            header=header_text,
            fmt="%.12e",
          )

    elif format == "hdf5":
      with h5py.File(filepath, "w") as f:
        # Metadata
        meta_grp = f.create_group("metadata")  # type: ignore
        meta_grp["time"] = self.time  # type: ignore
        meta_grp["order"] = self.sOrder  # type: ignore
        meta_grp["nx"] = self.nX  # type: ignore
        meta_grp["source_file"] = str(self._filename)  # type: ignore

        # Grid
        grid_grp = f.create_group("grid")  # type: ignore
        grid_grp["x"] = self.r  # type: ignore
        grid_grp["dx"] = self.dr  # type: ignore

        # Variables and modes
        var_grp = f.create_group("variables")  # type: ignore
        for var in variables:
          if var not in self.idx:
            continue
          var_subgrp = var_grp.create_group(var)  # type: ignore
          for mode in modes:
            try:
              var_data = self.get_variable(var, mode=mode)
              var_subgrp[f"mode_{mode}"] = var_data  # type: ignore
            except AthelasError:
              continue
    else:
      raise ValueError(f"Unsupported format for export_modes: {format}")

  @staticmethod
  def load_saved_data(filename: Union[str, Path]) -> Dict:
    """
    Load data that was saved with save_data().

    Args:
        filename: Path to saved file

    Returns:
        Dictionary containing the loaded data

    Note:
        This is a utility function to load exported data back.
        For HDF5 and NumPy formats, returns structured data.
        For text format, you may want to use pandas or np.loadtxt directly.
    """
    filepath = Path(filename)
    ext = filepath.suffix.lower()

    if ext in [".npz"]:
      # NumPy format
      loaded = np.load(filepath)
      return dict(loaded)

    elif ext in [".h5", ".hdf5"]:
      # HDF5 format - return nested dict structure
      result = {}
      with h5py.File(filepath, "r") as f:

        def _recursive_load(group, prefix=""):
          for key in group.keys():
            full_key = f"{prefix}/{key}" if prefix else key
            item = group[key]  # type: ignore
            if hasattr(item, "keys"):  # It's a group
              _recursive_load(item, full_key)  # type: ignore
            else:  # It's a dataset
              result[full_key] = item[...]  # type: ignore

        _recursive_load(f)

      return result

    elif ext in [".txt", ".csv", ".dat"]:
      # Text format - simple load
      data = np.loadtxt(filepath, delimiter=None)  # Auto-detect delimiter
      return {
        "data": data,
        "note": "Use np.loadtxt() for more control over text loading",
      }

    else:
      raise ValueError(
        f"Cannot load file with extension '{ext}'. Supported: .npz, .h5/.hdf5, .txt/.csv/.dat"
      )

  def info(self) -> Dict:
    """
    Return dictionary with simulation information.

    Returns:
        Dictionary containing simulation metadata and statistics
    """
    info_dict = {
      "filename": str(self._filename),
      "time": self.time,
      "spatial_order": self.sOrder,
      "num_cells": self.nX,
      "domain_bounds": self.domain_bounds,
      "variables": self.variables,
      "num_conserved_vars": self.n_conserved_vars,
      "num_auxiliary_vars": self.n_auxiliary_vars,
      "has_basis": self.basis is not None,
      "ghost_cells_loaded": self._load_ghost_cells,
    }

    # Add basic statistics for first few variables
    for var in self.variables[:3]:  # First 3 variables
      try:
        data = self.get_variable(var, mode=0)
        info_dict[f"{var}_stats"] = {
          "min": float(np.min(data)),
          "max": float(np.max(data)),
          "mean": float(np.mean(data)),
          "std": float(np.std(data)),
        }
      except AthelasError:
        continue

    return info_dict


def load_multiple_files(
  file_pattern: str, basis_filename: Optional[str] = None, **kwargs
) -> List[Athelas]:
  """
  Load multiple Athelas files matching a pattern.

  Args:
      file_pattern: Glob pattern for files to load
      basis_filename: Optional basis file to use for all
      **kwargs: Additional arguments passed to Athelas constructor

  Returns:
      List of Athelas objects sorted by time
  """
  from glob import glob

  files = sorted(glob(file_pattern))
  if not files:
    raise AthelasError(f"No files found matching pattern: {file_pattern}")

  athelas_objects = []
  for filename in files:
    try:
      athelas_objects.append(Athelas(filename, basis_filename, **kwargs))
    except AthelasError as e:
      warnings.warn(f"Failed to load {filename}: {e}")
      continue

  # Sort by time
  athelas_objects.sort(key=lambda a: a.time)

  return athelas_objects


if __name__ == "__main__":
  # Example usage
  try:
    # Basic usage
    sim = Athelas("sod_final.h5")
    print(sim)
    print("\nSimulation info:")
    for key, value in sim.info().items():
      print(f"  {key}: {value}")

    # Create summary plot
    fig = sim.summary_plot()
    fig.savefig("summary.png", dpi=150, bbox_inches="tight")

    sim.save_data("test.txt", variables=["tau", "energy"], format="text")
    sim.export_modes(
      "all_modes.txt",
      variables=["tau", "velocity"],
      modes=[0, 1, 2],
      format="text",
    )

  except AthelasError as e:
    print(f"Error: {e}")
