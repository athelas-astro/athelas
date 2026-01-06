import unittest
import shutil
import subprocess
import os
import sys
import warnings

import h5py  # type: ignore[import]
import numpy as np


# -- Compare two values up to some floating point tolerance
def soft_equiv(val: float, ref: float, rtol: float = 1.0e-5) -> bool:
  atol = ref / 1000.0 + np.finfo(float).eps
  error = abs(val - ref) / (atol + rtol * abs(ref))

  if abs(error) > 1.0:
    print(f"val ref error  {val}, {ref}, {error}")
    return False
  else:
    return True


class AthelasRegressionTest(unittest.TestCase):
  def __init__(
    self,
    test_name="test_sod",
    src_dir="",
    build_dir="./build",
    executable="./athelas",
    infile="test_inputs/sod.toml",
    varlist=["grid/x"],
    run_dir="./run",
    build_type="Release",
    num_procs=4,
    goldfile=None,
    upgold=False,
    tolerance=1.0e-5,
    build_required=True,
    compression_factor=2,
  ):
    super().__init__(test_name)
    self.src_dir = src_dir
    self.build_dir = build_dir
    self.build_type = build_type
    self.executable = executable
    self.infile = infile
    self.varlist = varlist
    self.run_dir = os.path.relpath(run_dir)
    self.num_procs = num_procs
    self.goldfile = goldfile
    self.upgold = upgold
    self.tolerance = tolerance
    self.build_required = build_required
    self.compression_factor = compression_factor

  # End __init__

  def build_code(self):
    if os.path.isdir(self.build_dir):
      print("Build dir already exists! Clean up before regression testing!")
      sys.exit(os.EX_SOFTWARE)
    os.mkdir(self.build_dir)
    os.chdir(self.build_dir)

    # Base configure options
    configure_options = ""

    if self.build_type == "Release":
      configure_options += "-DCMAKE_BUILD_TYPE=Release "
    elif self.build_type == "Debug":
      configure_options += "-DCMAKE_BUILD_TYPE=Debug "
    else:  # TODO: more cmake build types
      print(f"Build type '{self.build_type}' not known!")
      sys.exit(os.EX_SOFTWARE)
    configure_options += "-DENABLE_UNIT_TESTS=OFF "

    cmake_call = ""
    cmake_call += "cmake "
    for option in configure_options:
      cmake_call += option
    # cmake opts
    cmake_call += self.src_dir

    # Configure
    print("Configuring the source...")
    try:
      subprocess.run(
        cmake_call,
        shell=True,
        check=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
      )
    except subprocess.CalledProcessError as e:
      self.fail(f"Configure failed with error: {e.stderr.decode()}")

    # Compile
    print("Compiling the source...")
    try:
      subprocess.run(
        "cmake --build . --parallel " + str(self.num_procs),
        shell=True,
        check=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
      )
    except subprocess.CalledProcessError as e:
      self.fail(f"Compilation failed with error: {e.stderr.decode()}")

    # Return to standard directory
    os.chdir("..")

  # End build_code

  def setUp(self):
    """Set up the test environment by compiling the code if required."""
    if self.build_required:
      self.build_code()

  # end setUp

  def tearDown(self):
    """Clean up the generated executable after the test."""
    self.cleanup()

  def run_code(self):
    if self.build_required and not os.path.isdir(self.build_dir):
      self.build_code()

    # Check if executable exists
    if os.path.isabs(self.executable):
      # For absolute path, check directly
      if not os.path.isfile(self.executable):
        print(f"Executable '{self.executable}' does not exist!")
        sys.exit(os.EX_SOFTWARE)
    else:
      if self.build_required:
        executable_path = os.path.join(self.build_dir, self.executable)
        if not os.path.isfile(executable_path):
          print(
            f"Executable '{executable_path}' does not exist after building!"
          )
          sys.exit(os.EX_SOFTWARE)
        # Update executable path to be relative to run directory
        self.executable = os.path.join("../", self.build_dir, self.executable)

    if os.path.isdir(self.run_dir):
      print("Run dir already exists! Clean up before regression testing!")
      sys.exit(os.EX_SOFTWARE)
    os.mkdir(self.run_dir)
    os.chdir(self.run_dir)

    run_cmd = ""  # empty now, can accomodate mpi runs
    outfile = open("out.dat", "w")
    print("\nRunning athelas...")

    try:
      if os.path.isabs(self.executable):
        # For absolute executable path, use absolute path for input file too
        abs_infile = os.path.abspath(self.infile)
        subprocess.run(
          run_cmd + self.executable + " -i " + abs_infile,
          shell=True,
          check=True,
          stdout=outfile,
          stderr=subprocess.PIPE,
        )
      else:
        subprocess.run(
          run_cmd + self.executable + f" -i {self.infile}",
          shell=True,
          check=True,
          stdout=outfile,
          stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
      self.fail(f"Execution failed with error: {e.stderr.decode()}")
    outfile.close()

  # End run_code

  def load_output(self, fn, varlist):
    """
    load athelas output
    """
    with h5py.File(fn, "r") as f:
      outs = []
      for v in varlist:
        outs.append(f[v][1:-1, ...].flatten())  # type: ignore
    return np.concatenate(outs)

  # End load_output

  def compare_gold(self):
    """
    compare to gold data
    """
    # check if sim data exist
    if not os.path.isfile("out.dat"):
      print("Simulation data do not exist!")
      sys.exit(os.EX_SOFTWARE)

    # load sim data
    basename = os.path.splitext(os.path.basename(self.infile))[0]
    filename = basename + "_final.h5"
    variables_data = self.load_output(filename, self.varlist)

    # Compress results, if desired
    compressed_variables = np.zeros(
      len(variables_data) // self.compression_factor
    )
    for n in range(len(compressed_variables)):
      compressed_variables[n] = variables_data[self.compression_factor * n]
    variables_data = compressed_variables

    success = True

    # Use the goldfile path if provided, otherwise construct it
    if self.goldfile:
      gold_name = self.goldfile
    else:
      gold_name = os.path.join("../goldfiles", basename) + ".gold"

    if self.upgold:
      np.savetxt(gold_name, variables_data, newline="\n")
    else:
      gold_variables = np.loadtxt(gold_name)
      if not len(gold_variables) == len(variables_data):
        print("Length of gold variables does not match calculated variables!")
        success = False
      else:
        for n in range(len(gold_variables)):
          if not soft_equiv(
            variables_data[n], gold_variables[n], rtol=self.tolerance
          ) or np.isnan(variables_data[n]):
            success = False
            print(f"failure at n = {n}")

    # Report upgolding, success, or failure
    if self.upgold:
      print(f"Gold file {gold_name} updated!")
      return True
    else:
      if success:
        return True
      else:
        mean_error = np.mean(variables_data - gold_variables)
        max_error = np.max(np.fabs(variables_data - gold_variables))
        max_frac_error = np.max(
          np.fabs(variables_data - gold_variables)
          / np.clip(np.fabs(gold_variables), 1.0e-100, None)
        )

        print(f"Mean error:           {mean_error}")
        print(f"Max error:            {max_error}")
        print(f"Max fractional error: {max_frac_error}")
        return False

  # End compare_gold

  def cleanup(self):
    """
    Clean up working directory after testing.
    Only cleans up build directory if we actually built the code (build_required=True).
    """
    if (
      os.getcwd().split(os.sep)[-1] == self.build_dir
      or os.getcwd().split(os.sep)[-1] == self.run_dir
    ):
      os.chdir("..")

    if os.path.isabs(self.build_dir):
      print(
        "Absolute paths not allowed for build dir -- unsafe when cleaning up!"
      )
      sys.exit(os.EX_SOFTWARE)

    if os.path.isabs(self.run_dir):
      print(
        "Absolute paths not allowed for run dir -- unsafe when cleaning up!"
      )
      sys.exit(os.EX_SOFTWARE)

    # Only clean up build directory if we actually built the code
    if self.build_required and os.path.exists(self.build_dir):
      try:
        shutil.rmtree(self.build_dir)
      except Exception:
        warnings.warn(f"Error cleaning up build directory '{self.build_dir}'!")

    if os.path.exists(self.run_dir):
      try:
        shutil.rmtree(self.run_dir)
      except Exception:
        warnings.warn(f"Error cleaning up run directory '{self.run_dir}'!")

  # End cleanup
