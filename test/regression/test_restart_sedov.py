"""Restart regression test.

Verify that a run resumed from a midpoint .ath checkpoint reproduces the
final state of an identical end-to-end run.

The two runs share an input deck; we just take a midpoint dump from the
baseline and restart from it into a separate output dir, then compare each
run's sedov_final.ath.
"""

import argparse
import os
import shutil
import subprocess
import sys
import unittest

import h5py  # type: ignore[import]
import numpy as np

from regression_test import soft_equiv


class RestartSedovTest(unittest.TestCase):
  """Restart-resumed run must match the direct run within tolerance."""

  def __init__(self, methodName="test_restart_sedov", executable_path=None):
    super().__init__(methodName)
    regression_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(regression_dir, "../../"))

    self.regression_dir = regression_dir
    self.src_dir = src_dir
    self.infile = os.path.join(
      regression_dir, "test_inputs", "restart_sedov.lua"
    )

    self.varlist = [
      "mesh/r",
      "mesh/dr",
      "mesh/x_l",
      "mesh/dm",
      "mesh/enclosed_mass",
      "mesh/sqrt_gm",
      "mesh/r_q",
      "fields/u_cf",
      "fields/u_pf",
    ]
    # Tolerance accounts for fp non-associativity in the restart-resumed
    # half of the integration; this is much tighter than the gold tests.
    self.tolerance = 1.0e-10

    self.run_dir = "run_restart_sedov"
    self.build_dir = "build_restart_sedov"

    if executable_path:
      self.executable = os.path.abspath(executable_path)
      self.build_required = False
    else:
      self.executable = None  # built in setUp
      self.build_required = True

  def setUp(self):
    if self.build_required:
      self._build()
    if os.path.isdir(self.run_dir):
      shutil.rmtree(self.run_dir)
    os.makedirs(os.path.join(self.run_dir, "baseline"))
    os.makedirs(os.path.join(self.run_dir, "restart"))

  def tearDown(self):
    if os.path.isdir(self.run_dir):
      shutil.rmtree(self.run_dir)
    if self.build_required and os.path.isdir(self.build_dir):
      shutil.rmtree(self.build_dir)

  def _build(self):
    if os.path.isdir(self.build_dir):
      self.fail(
        f"Build dir '{self.build_dir}' already exists; clean before testing"
      )
    os.mkdir(self.build_dir)
    try:
      subprocess.run(
        ["cmake", "-DCMAKE_BUILD_TYPE=Release",
         "-DATHELAS_ENABLE_UNIT_TESTS=OFF", self.src_dir],
        cwd=self.build_dir, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      )
      subprocess.run(
        ["cmake", "--build", ".", "--parallel"],
        cwd=self.build_dir, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      )
    except subprocess.CalledProcessError as e:
      self.fail(f"Build failed: {e.stderr.decode(errors='replace')}")
    self.executable = os.path.abspath(
      os.path.join(self.build_dir, "athelas")
    )

  def _run(self, args, cwd, logname):
    with open(os.path.join(cwd, logname), "w") as out:
      try:
        subprocess.run(
          [self.executable, *args],
          cwd=cwd, check=True, stdout=out, stderr=subprocess.PIPE,
        )
      except subprocess.CalledProcessError as e:
        self.fail(
          f"athelas failed ({' '.join(args)}): "
          f"{e.stderr.decode(errors='replace')}"
        )

  def test_restart_sedov(self):
    baseline = os.path.join(self.run_dir, "baseline")
    restart = os.path.join(self.run_dir, "restart")

    # Baseline: full end-to-end run, leaves a midpoint dump and a _final.
    self._run(["-i", self.infile, "-o", "."], baseline, "baseline.log")

    midpoint = os.path.join(baseline, "sedov_000001.ath")
    self.assertTrue(
      os.path.isfile(midpoint),
      f"baseline did not produce expected midpoint dump {midpoint}",
    )

    # Restart from the midpoint into a clean dir.
    self._run(
      ["-r", os.path.abspath(midpoint), "-o", "."], restart, "restart.log"
    )

    baseline_final = os.path.join(baseline, "sedov_final.ath")
    restart_final = os.path.join(restart, "sedov_final.ath")
    self.assertTrue(os.path.isfile(baseline_final), "baseline final missing")
    self.assertTrue(os.path.isfile(restart_final), "restart final missing")

    self._compare(baseline_final, restart_final)

  def _compare(self, file_a, file_b):
    failures = []
    with h5py.File(file_a, "r") as fa, h5py.File(file_b, "r") as fb:
      for v in self.varlist:
        arr_a = np.asarray(fa[v]).flatten()
        arr_b = np.asarray(fb[v]).flatten()
        self.assertEqual(
          arr_a.shape, arr_b.shape, f"shape mismatch for '{v}'"
        )
        mismatches = 0
        worst = (0.0, 0.0, 0.0)  # (val, ref, frac_error)
        for n in range(len(arr_a)):
          if np.isnan(arr_b[n]) or not soft_equiv(
            arr_b[n], arr_a[n], rtol=self.tolerance
          ):
            mismatches += 1
            denom = max(abs(arr_a[n]), 1e-300)
            frac = abs(arr_a[n] - arr_b[n]) / denom
            if frac > worst[2]:
              worst = (arr_b[n], arr_a[n], frac)
        if mismatches:
          failures.append((v, mismatches, len(arr_a), worst))
    if failures:
      lines = ["Restart drift exceeds tolerance:"]
      for v, m, n, w in failures:
        lines.append(
          f"  {v}: {m}/{n} entries diverge; worst frac error "
          f"{w[2]:.3e} (restart={w[0]:.6e}, baseline={w[1]:.6e})"
        )
      self.fail("\n".join(lines))


def create_test_suite(executable_path=None):
  suite = unittest.TestSuite()
  suite.addTest(RestartSedovTest(executable_path=executable_path))
  return suite


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Verify restart reproducibility on the sedov problem"
  )
  parser.add_argument(
    "--executable", "-e",
    help="Path to a prebuilt athelas executable (skips build)",
  )
  args = parser.parse_args()
  runner = unittest.TextTestRunner(verbosity=2)
  result = runner.run(create_test_suite(executable_path=args.executable))
  sys.exit(0 if result.wasSuccessful() else 1)
