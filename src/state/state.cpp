#include "state/state.hpp"
#include "compdata.hpp"
#include "utils/error.hpp"

namespace athelas {

using atom::CompositionData;
using atom::IonizationState;
State::State(const ProblemIn *const pin, const int nstages)
    : params_(std::make_unique<Params>()) {
  static const bool rad_enabled = pin->param()->get<bool>("physics.rad_active");
  static const bool composition_enabled =
      pin->param()->get<bool>("physics.composition_enabled");
  static const bool ionization_enabled =
      pin->param()->get<bool>("physics.ionization_enabled");
  // NOTE: This will need to be extended when mixing is added.
  static const bool composition_evolved =
      pin->param()->get<bool>("physics.heating.nickel.enabled");
  static const bool nickel_evolved =
      pin->param()->get<bool>("physics.heating.nickel.enabled");
  int nvars_cons = (rad_enabled) ? 5 : 3;
  if (composition_enabled) {
    const int ncomps = pin->param()->get<int>("composition.ncomps");
    nvars_cons += ncomps;
    params_->add("ncomps", ncomps);
  }
  static const int nvars_prim = 3; // Maybe this can be smarter
  static const int nvars_aux = (rad_enabled) ? 5 : 3;
  static const int nx = pin->param()->get<int>("problem.nx");
  static const int n_nodes = pin->param()->get<int>("fluid.nnodes");
  static const int porder = pin->param()->get<int>("fluid.porder");

  uCF_ = AthelasArray3D<double>("uCF", nx + 2, porder, nvars_cons);
  uCF_s_ = AthelasArray4D<double>("uCF_s", nstages, nx + 2, porder, nvars_cons);
  uPF_ = AthelasArray3D<double>("uPF", nx + 2, n_nodes + 2, nvars_prim);
  uAF_ = AthelasArray3D<double>("uAF", nx + 2, n_nodes + 2, nvars_aux);

  params_->add("nvars_cons", nvars_cons);
  params_->add("nvars_prim", nvars_prim);
  params_->add("nvars_aux", nvars_aux);
  params_->add("n_nodes", n_nodes);
  params_->add("p_order", porder);
  params_->add("n_stages", nstages);
  params_->add("composition_enabled", composition_enabled);
  params_->add("ionization_enabled", ionization_enabled);
  params_->add("composition_evolved", composition_evolved);
  params_->add("nickel_evolved", nickel_evolved);
}

void State::setup_composition(std::shared_ptr<CompositionData> comps) {
  if (!composition_enabled()) {
    THROW_ATHELAS_ERROR(
        "Trying to set composition but composition is not enabled!");
  }
  comps_ = std::move(comps);
}

void State::setup_ionization(std::shared_ptr<IonizationState> ion) {
  if (!ionization_enabled()) {
    THROW_ATHELAS_ERROR(
        "Trying to set ionization but ionization is not enabled!");
  }
  ionization_state_ = std::move(ion);
}

[[nodiscard]] auto State::comps() const -> CompositionData * {
  if (!composition_enabled()) {
    THROW_ATHELAS_ERROR("Composition not enabled!");
  }
  return comps_.get();
}

[[nodiscard]] auto State::ionization_state() const -> IonizationState * {
  if (!ionization_enabled()) {
    THROW_ATHELAS_ERROR("Ionization not enabled!");
  }
  return ionization_state_.get();
}

[[nodiscard]] auto State::composition_enabled() const noexcept -> bool {
  return params_->get<bool>("composition_enabled");
}

[[nodiscard]] auto State::ionization_enabled() const noexcept -> bool {
  return params_->get<bool>("ionization_enabled");
}

[[nodiscard]] auto State::composition_evolved() const noexcept -> bool {
  return params_->get<bool>("composition_evolved");
}

[[nodiscard]] auto State::nickel_evolved() const noexcept -> bool {
  return params_->get<bool>("nickel_evolved");
}

[[nodiscard]] auto State::mass_fractions() const noexcept
    -> AthelasArray3D<double> {
  return Kokkos::subview(uCF_, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::make_pair(nvars(), nvars() + ncomps()));
}

[[nodiscard]] auto State::mass_fractions_stages() const noexcept
    -> AthelasArray4D<double> {
  return Kokkos::subview(uCF_s_, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::make_pair(nvars(), nvars() + ncomps()));
}

auto State::params() noexcept -> Params * { return params_.get(); }

// num var accessors
auto State::n_cf() const noexcept -> int {
  return params_->get<int>("nvars_cons");
}
auto State::n_pf() const noexcept -> int {
  return params_->get<int>("nvars_prim");
}
auto State::n_af() const noexcept -> int {
  return params_->get<int>("nvars_aux");
}
auto State::ncomps() const noexcept -> int {
  if (composition_enabled()) {
    return params_->get<int>("ncomps");
  }
  return 0;
}
/**
 * @brief return number of rad hydro variables
 */
auto State::nvars() const noexcept -> int { return n_cf() - ncomps(); }
auto State::p_order() const noexcept -> int {
  return params_->get<int>("p_order");
}

// view accessors
auto State::u_cf() const noexcept -> AthelasArray3D<double> { return uCF_; }
auto State::u_cf_stages() const noexcept -> AthelasArray4D<double> {
  return uCF_s_;
}
auto State::u_pf() const noexcept -> AthelasArray3D<double> { return uPF_; }
auto State::u_af() const noexcept -> AthelasArray3D<double> { return uAF_; }
auto State::vars() const noexcept -> AthelasArray3D<double> {
  return Kokkos::subview(uCF_, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::make_pair(0, nvars()));
}

} // namespace athelas
