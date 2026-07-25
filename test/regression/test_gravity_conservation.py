import os
import shutil
import subprocess
import unittest

import numpy as np


def _read_history(path):
  """Parse an athelas .hst file into a name->column-array dict.

  The header is a single comment line of ``<index> <name with spaces> ...``
  tuples; walk it, splitting whenever the running index appears as a token.
  """
  with open(path) as f:
    header = f.readline().lstrip("#").split()
  columns = {}
  nxt, current = 0, []
  for tok in header:
    if tok == str(nxt) and (nxt == 0 or current):
      if current:
        columns[nxt - 1] = " ".join(current)
      current = []
      nxt += 1
    else:
      current.append(tok)
  if current:
    columns[nxt - 1] = " ".join(current)

  data = np.loadtxt(path, comments="#")
  name_to_index = {name: idx for idx, name in columns.items()}
  return {name: data[:, idx] for name, idx in name_to_index.items()}


class GravityEnergyConservationTest(unittest.TestCase):
  """Regression test for the gravity limiter energy correction.

  A small P1 hydrostatic-balance breathing mode keeps the slope limiter active,
  so the limiter relocates interior mesh nodes and moves the discrete
  gravitational potential energy W_h with no physical source (reported as
  ``Cumulative Limiter Mesh Work``). This is a differential test: it runs the
  same problem with ``gravity.limiter_energy_correction`` off and on and asserts
  that the correction fires, offsets the limiter mesh work, and measurably
  reduces the total-energy drift. No gold file -- it tests a conservation
  property directly, so it is robust to benign floating-point changes.
  """

  def __init__(
    self, methodName="test_gravity_conservation", executable_path=None
  ):
    super().__init__(methodName)
    regression_dir = os.path.dirname(os.path.abspath(__file__))
    self.infile = os.path.join(
      regression_dir, "test_inputs", "gravity_energy_conservation.lua"
    )
    self.run_dir = os.path.join(regression_dir, "run_gravity_conservation")
    self.hist_name = "gravity_energy_conservation.hst"
    self.executable = (
      os.path.abspath(executable_path) if executable_path else None
    )

  def setUp(self):
    # This differential test drives an existing binary directly; it does not
    # build its own. Run the suite with -e /path/to/athelas.
    if self.executable is None:
      self.skipTest(
        "requires an existing executable (run with -e /path/to/athelas)"
      )

  def tearDown(self):
    if os.path.isdir(self.run_dir):
      shutil.rmtree(self.run_dir)

  def _run(self, correction, tag):
    """Run the deck with the correction flag forced to `correction`; return
    the parsed history."""
    out_dir = os.path.join(self.run_dir, tag)
    os.makedirs(out_dir)

    # Materialize a deck variant with the correction flag forced.
    with open(self.infile) as f:
      deck = f.read()
    value = "true" if correction else "false"
    deck = deck.replace(
      "limiter_energy_correction = true",
      f"limiter_energy_correction = {value}",
    )
    deck_path = os.path.join(out_dir, "deck.lua")
    with open(deck_path, "w") as f:
      f.write(deck)

    env = dict(os.environ, OMP_NUM_THREADS="1")
    result = subprocess.run(
      f"{self.executable} -i {deck_path} -o {out_dir}",
      shell=True,
      cwd=out_dir,
      env=env,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
      self.fail(
        f"athelas ({tag}) failed:\n{result.stderr.decode(errors='replace')}"
      )
    return _read_history(os.path.join(out_dir, self.hist_name))

  def test_gravity_conservation(self):
    os.makedirs(self.run_dir)

    off = self._run(correction=False, tag="off")
    on = self._run(correction=True, tag="on")

    def drift(h):
      e = h["Total Energy [erg]"]
      return (e[-1] - e[0]) / abs(e[0])

    def finite(h):
      return all(np.all(np.isfinite(v)) for v in h.values())

    self.assertTrue(finite(off), "off run produced non-finite history values")
    self.assertTrue(finite(on), "on run produced non-finite history values")

    # The limiter-energy diagnostics are only present when the correction is
    # enabled, so they come from the on run.
    mesh_work_on = on["Cumulative Limiter Mesh Work [erg]"][-1]
    correction = on["Cumulative Limiter Energy Correction [erg]"][-1]
    clamp = on["Cumulative Limiter Energy Clamp Residual [erg]"][-1]

    # The problem must actually exercise the limiter, otherwise the test is
    # vacuous.
    self.assertGreater(
      abs(mesh_work_on),
      1.0e40,
      "limiter moved no gravitational energy; problem does not exercise the "
      "correction",
    )

    # The correction fired and has the opposite sign to the mesh work it offsets.
    self.assertNotEqual(correction, 0.0, "correction did not fire")
    self.assertLess(
      correction * mesh_work_on,
      0.0,
      "correction has the same sign as the limiter mesh work it should offset",
    )

    # Exact cancellation invariant: the reported mesh work is the end-of-step
    # limiter block that the correction targets, so
    #   correction + clamp_residual = -mesh_work
    # holds to roundoff (the clamp accounts for any part the EOS floor blocked).
    self.assertLess(
      abs(correction + clamp + mesh_work_on),
      1.0e-6 * abs(mesh_work_on),
      "correction does not cancel the reported limiter mesh work: "
      f"correction={correction:.4e}, clamp={clamp:.4e}, "
      f"mesh_work={mesh_work_on:.4e}",
    )

    # On this well-behaved problem the EOS internal-energy floor should not
    # bind, so essentially all of the correction is applied.
    self.assertLess(
      abs(clamp),
      0.05 * abs(correction),
      "EOS clamp residual dominates the correction",
    )

    # The correction must improve conservation: total-energy drift strictly
    # smaller with it on. (Boundary outflow is identical in both runs, so the
    # difference isolates the correction.)
    self.assertLess(
      abs(drift(on)),
      abs(drift(off)),
      f"correction did not reduce drift: off={drift(off):.3e}, "
      f"on={drift(on):.3e}",
    )


def create_test_suite(executable_path=None):
  suite = unittest.TestSuite()
  suite.addTest(GravityEnergyConservationTest(executable_path=executable_path))
  return suite


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(
    description="Run the gravity energy conservation test"
  )
  parser.add_argument(
    "--executable",
    "-e",
    required=True,
    help="Path to an existing athelas executable",
  )
  args = parser.parse_args()

  runner = unittest.TextTestRunner(verbosity=2)
  runner.run(create_test_suite(executable_path=args.executable))
