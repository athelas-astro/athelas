#pragma once

namespace athelas {

class GridStructure;
class MeshState;
class PackageManager;
struct TimeStepInfo;

/**
 * @class OperatorSplitStepper
 * @brief Updates operator split packages: U -> U + dU dt
 * @note We make some asusmptions on the packages that go here.
 *   1. Hydro will never be in here. The grid advection will never be in here.
 *   2. We are not limiting fields in these packages. This should be remedied
 *      when limiters are moved into packages.
 */
class OperatorSplitStepper {
 public:
  OperatorSplitStepper() = default;

  static void step(PackageManager *pkgs, MeshState &mesh_state,
                   const GridStructure &grid, TimeStepInfo &dt_info);
};

} // namespace athelas
