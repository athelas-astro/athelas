import math
import os
import unittest

import numpy as np

from regression_test import AthelasRegressionTest, soft_equiv
from athelas_tools.athelas import Athelas


class RadDiffusionWaveTest(AthelasRegressionTest):
  """Test optically thick radiation diffusion in the AP transport limit."""

  def __init__(
    self, methodName="test_rad_diffusion_wave", executable_path=None
  ):
    regression_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(regression_dir, "../../"))

    infile = os.path.join(
      regression_dir, "test_inputs", "rad_diffusion_wave.lua"
    )

    if executable_path:
      if not os.path.isabs(executable_path):
        executable_path = os.path.abspath(executable_path)
      executable = executable_path
      build_required = False
    else:
      executable = "athelas"
      build_required = True

    super().__init__(
      test_name=methodName,
      src_dir=src_dir,
      build_dir="build_rad_diffusion_wave",
      executable=executable,
      infile=infile,
      run_dir="run_rad_diffusion_wave",
      build_type="Release",
      num_procs=2,
      build_required=build_required,
    )

  def test_rad_diffusion_wave(self):
    self.run_code()

    # Pull the problem setup straight from the output so the analytic
    # expectation can never drift out of sync with the input deck.
    ds = Athelas("rad_diffusion_wave_final.ath")
    rho = ds.params["problem.params.rho"]
    kappa_r = ds.params["opacity.kR"]
    mode = ds.params["problem.params.mode"]
    perturbation = ds.params["problem.params.perturbation"]
    length = ds.params["problem.xr"] - ds.params["problem.xl"]
    t_end = ds.params["problem.tf"]
    c_cgs = 2.99792458e10  # constants::c_cgs

    # Diffusion-mode amplitude decays as exp(-D k^2 t), D = c / (3 rho kappa_R).
    wave_number = mode * math.pi / length
    diffusion_coeff = c_cgs / (3.0 * rho * kappa_r)
    expected = perturbation * math.exp(
      -diffusion_coeff * wave_number * wave_number * t_end
    )

    # Cell-averaged specific radiation energy at cell centers (ghosts stripped).
    x = ds.r
    e_rad = ds.get("specific_radiation_energy")

    # Least-squares projection of the perturbation onto cos(k x), normalized by
    # the mean so the comparison is the relative amplitude (independent of the
    # background level and density).
    basis = np.cos(wave_number * x)
    e_mean = np.mean(e_rad)
    amplitude = np.sum((e_rad - e_mean) * basis) / np.sum(basis * basis)

    self.assertTrue(soft_equiv(amplitude / e_mean, expected, rtol=1.0e-5))


def create_test_suite(executable_path=None):
  suite = unittest.TestSuite()
  suite.addTest(RadDiffusionWaveTest(executable_path=executable_path))
  return suite


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Run radiation diffusion test")
  parser.add_argument(
    "--executable",
    "-e",
    help="Path to an existing executable to use instead of building",
  )
  args = parser.parse_args()

  runner = unittest.TextTestRunner(verbosity=2)
  runner.run(create_test_suite(executable_path=args.executable))
