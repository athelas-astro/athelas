import sys
import os
import unittest

from regression_test import AthelasRegressionTest, soft_equiv

sys.path.insert(1, "../../scripts/python/")
import radeq


class RadiationEquilibriumTest(AthelasRegressionTest):
  """Test for the Sod shock tube problem"""

  def __init__(self, methodName="test_radeq", executable_path=None):
    # Get the absolute path to the regression test directory
    regression_dir = os.path.dirname(os.path.abspath(__file__))

    # Set up paths relative to the regression directory
    src_dir = os.path.abspath(os.path.join(regression_dir, "../../"))

    # Use relative paths for build and run directories to ensure safe cleanup
    build_dir = "build_rad_equilibrium"
    run_dir = "run_rad_equilibrium"

    # Use absolute paths for input and gold files to ensure they can be found
    infile = os.path.join(regression_dir, "test_inputs", "rad_equilibrium.toml")
    goldfile = os.path.join(regression_dir, "goldfiles", "rad_equilibrium.gold")

    # If executable_path is provided, use it directly
    if executable_path:
      # Convert to absolute path if it's not already
      if not os.path.isabs(executable_path):
        executable_path = os.path.abspath(executable_path)
      executable = executable_path
      # We don't need to build if using an existing executable
      build_required = False
    else:
      # For built executable, use relative path
      executable = "athelas"
      build_required = True

    # Initialize the parent class with test-specific parameters
    super().__init__(
      test_name=methodName,
      src_dir=src_dir,
      build_dir=build_dir,
      executable=executable,
      infile=infile,
      run_dir=run_dir,
      build_type="Release",
      num_procs=2,
      goldfile=goldfile,
      upgold=False,
      tolerance=1.0e-5,
      build_required=build_required,
      compression_factor=1,
    )

  def test_radeq(self):
    """Test radiation equilibrium simulation against gold standard data."""
    self.run_code()

    # vars to test
    varlist = [
      "fields/u_cf",
    ]
    fn = "rad_equilibrium_final.ath"
    data = self.load_output(fn, varlist)

    # analytic sol
    e_gas = 10**10
    e_rad = 10**12
    T = 4.81e8
    rho = 1.0e-7
    kappa = 4 * 1.0e-8  # * 1.0e-7# 1 / cm weird units
    t_end = 1.0e-7
    te = radeq.ThermalEquilibrium(e_gas, kappa, T, e_rad, t_end)
    te.evolve()
    sol = te.sol[-1]  # scale by density

    self.assertTrue(soft_equiv(sol, data[2] * rho, rtol=1.0e-2))  # low tol


def create_test_suite(executable_path=None):
  """Create a test suite for the test"""
  suite = unittest.TestSuite()
  suite.addTest(RadiationEquilibriumTest(executable_path=executable_path))
  return suite


if __name__ == "__main__":
  # Run this test directly if executed as a script
  import argparse

  parser = argparse.ArgumentParser(description="Run rad equilibrium test")
  parser.add_argument(
    "--executable",
    "-e",
    help="Path to an existing executable to use instead of building",
  )
  args = parser.parse_args()

  runner = unittest.TextTestRunner(verbosity=2)
  runner.run(create_test_suite(executable_path=args.executable))
