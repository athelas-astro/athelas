import os
import unittest
from regression_test import AthelasRegressionTest


class SedovBlastWaveTest(AthelasRegressionTest):
  """Test for the Sedov blast wave"""

  def __init__(self, methodName="test_sedov", executable_path=None):
    # Get the absolute path to the regression test directory
    regression_dir = os.path.dirname(os.path.abspath(__file__))

    # Set up paths relative to the regression directory
    src_dir = os.path.abspath(os.path.join(regression_dir, "../../"))

    # Use relative paths for build and run directories to ensure safe cleanup
    build_dir = "build_sedov"
    run_dir = "run_sedov"

    # Use absolute paths for input and gold files to ensure they can be found
    infile = os.path.join(regression_dir, "test_inputs", "sedov.toml")
    goldfile = os.path.join(regression_dir, "goldfiles", "sedov.gold")

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

    # vars to test
    varlist = [
      "mesh/r",
      "mesh/dr",
      "fields/u_cf",
    ]

    # Initialize the parent class with test-specific parameters
    super().__init__(
      test_name=methodName,
      src_dir=src_dir,
      build_dir=build_dir,
      executable=executable,
      infile=infile,
      varlist=varlist,
      run_dir=run_dir,
      build_type="Release",
      num_procs=2,
      goldfile=goldfile,
      upgold=False,
      tolerance=1.0e-5,
      build_required=build_required,
      compression_factor=2,
    )

  def test_sedov(self):
    """Test Sedov blast wave: 3rd order in spherical symmetry"""
    self.run_code()
    self.assertTrue(self.compare_gold())
    if self.upgold:
      print(f"Gold file {self.goldfile} successfully updated!")
    self.assertTrue(True)


def create_test_suite(executable_path=None):
  suite = unittest.TestSuite()
  suite.addTest(SedovBlastWaveTest(executable_path=executable_path))
  return suite


if __name__ == "__main__":
  # Run this test directly if executed as a script
  import argparse

  parser = argparse.ArgumentParser(description="Run Sedov blast wavee test")
  parser.add_argument(
    "--executable",
    "-e",
    help="Path to an existing executable to use instead of building",
  )
  args = parser.parse_args()

  runner = unittest.TextTestRunner(verbosity=2)
  runner.run(create_test_suite(executable_path=args.executable))
