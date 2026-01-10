#include "state/state.hpp"
#include "composition/compdata.hpp"
#include "utils/error.hpp"

namespace athelas {

using atom::CompositionData;
using atom::IonizationState;

// --- StageData ---

[[nodiscard]] auto StageData::ionization_enabled() const noexcept -> bool {
  return parent_->ionization_enabled();
}
[[nodiscard]] auto StageData::composition_enabled() const noexcept -> bool {
  return parent_->composition_enabled();
}

auto StageData::get_field(const std::string &name) const
    -> AthelasArray3D<double> {
  assert(parent_->is_allocated(name) && "Field not allocated!");
  const auto &metadata = parent_->get_metadata(name);

  if (metadata.policy == DataPolicy::Staged) {
    return parent_->get_field_at_stage(name, stage_);
  }
  return parent_->get_field<AthelasArray3D<double>>(name);
}

auto StageData::get_var(const std::string &field, const std::string &var_name,
                        const int i, const int q) const -> double {
  const int var_idx = parent_->var_index(field, var_name);
  auto var = get_field(field);
  return var(i, q, var_idx);
}

[[nodiscard]] auto StageData::nvars(const std::string &field) const -> int {
  return parent_->nvars(field);
}

auto StageData::comps() const -> atom::CompositionData * {
  return parent_->comps();
}

auto StageData::ionization_state() const -> atom::IonizationState * {
  return parent_->ionization_state();
}

auto StageData::mass_fractions(const std::string &name) const
    -> AthelasArray3D<double> {
  return parent_->mass_fractions(name, stage_);
}

[[nodiscard]] auto StageData::eos() const -> const eos::EOS & {
  return parent_->eos();
}

[[nodiscard]] auto StageData::opac() const -> const Opacity & {
  return parent_->opac();
}

[[nodiscard]] auto StageData::fluid_basis() const -> const basis::ModalBasis & {
  return parent_->fluid_basis();
}

[[nodiscard]] auto StageData::rad_basis() const -> const basis::ModalBasis & {
  return parent_->rad_basis();
}

// --- MeshState ---

MeshState::MeshState(const ProblemIn *const pin, const int nstages)
    : params_(std::make_unique<Params>()), nstages_(nstages) {

  const bool composition_enabled =
      pin->param()->get<bool>("physics.composition_enabled");
  const bool ionization_enabled =
      pin->param()->get<bool>("physics.ionization_enabled");
  // NOTE: This will need to be extended when mixing is added.
  const bool composition_evolved =
      pin->param()->get<bool>("physics.heating.nickel.enabled");
  const bool nickel_evolved =
      pin->param()->get<bool>("physics.heating.nickel.enabled");
  const int porder = pin->param()->get<int>("fluid.porder");

  params_->add("p_order", porder);
  params_->add("n_stages", nstages);
  params_->add("composition_enabled", composition_enabled);
  params_->add("ionization_enabled", ionization_enabled);
  params_->add("composition_evolved", composition_evolved);
  params_->add("nickel_evolved", nickel_evolved);

  // microphysics
  eos_ = std::make_unique<eos::EOS>(eos::initialize_eos(pin));
  opac_ = std::make_unique<Opacity>(initialize_opacity(pin));
}

[[nodiscard]] auto MeshState::comps() const -> atom::CompositionData * {
  return comps_.get();
}

[[nodiscard]] auto MeshState::eos() const -> const eos::EOS & {
  if (!eos_) {
    throw_athelas_error("EOS not initialized!");
  }
  return *eos_;
}

[[nodiscard]] auto MeshState::eos() -> eos::EOS & {
  if (!eos_) {
    throw_athelas_error("EOS not initialized!");
  }
  return *eos_;
}

[[nodiscard]] auto MeshState::opac() const -> const Opacity & {
  if (!opac_) {
    throw_athelas_error("Opacity not initialized!");
  }
  return *opac_;
}

[[nodiscard]] auto MeshState::ionization_state() const
    -> atom::IonizationState * {
  return ionization_state_.get();
}
// Variable index access
[[nodiscard]] auto MeshState::var_index(const std::string &field,
                                        const std::string &var_name) const
    -> int {
  const auto &meta = get_metadata(field);
  if (!meta.var_map) {
    throw std::runtime_error("Field " + field + " has no variable mapping");
  }
  return meta.var_map->index(var_name);
}

[[nodiscard]] auto MeshState::var_name(const std::string &field,
                                       int index) const -> std::string {
  const auto &meta = get_metadata(field);
  if (!meta.var_map) {
    throw std::runtime_error("Field " + field + " has no variable mapping");
  }
  return meta.var_map->name(index);
}

/**
 * @brief returns number of variables in a field
 */
[[nodiscard]] auto MeshState::nvars(const std::string &field) const -> int {
  auto it = arrays_.find(field);
  if (it == arrays_.end()) {
    throw_athelas_error("Field not allocated: " + field);
  }

  // Use visitor to get the last extent
  return std::visit(
      [](auto &&arr) -> int {
        using ArrayType = std::decay_t<decltype(arr)>;
        constexpr int rank = ArrayType::rank;
        return static_cast<int>(arr.extent(rank - 1)); // Last dimension
      },
      it->second);
}

[[nodiscard]] auto MeshState::has_field(const std::string &field) const
    -> bool {
  return metadata_.contains(field);
}

[[nodiscard]] auto MeshState::is_allocated(const std::string &field) const
    -> bool {
  auto it = metadata_.find(field);
  return it != metadata_.end() && it->second.allocated;
}

[[nodiscard]] auto MeshState::is_staged(const std::string &field) const
    -> bool {
  return get_metadata(field).policy == DataPolicy::Staged;
}

[[nodiscard]] auto MeshState::is_onecopy(const std::string &field) const
    -> bool {
  return get_metadata(field).policy == DataPolicy::OneCopy;
}

[[nodiscard]] auto
MeshState::get_comp_start_index(const std::string &field_name) const -> int {
  const auto &metadata = get_metadata(field_name);
  if (!metadata.var_map) {
    return -1;
  }

  if (!metadata.var_map->has("comps_0")) {
    return -1;
  }

  return metadata.var_map->index("comps_0");
}

[[nodiscard]] auto MeshState::mass_fractions(const std::string &field_name,
                                             int stage) const
    -> AthelasArray3D<double> {
  int comp_start = get_comp_start_index(field_name);
  if (comp_start < 0) {
    throw_athelas_error("Field " + field_name + " has no composition data!");
  }

  const auto &meta = get_metadata(field_name);
  int total_vars = nvars(field_name);

  if (meta.policy == DataPolicy::Staged) {
    auto field = get_field<AthelasArray4D<double>>(field_name);
    return Kokkos::subview(field, stage, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::make_pair(comp_start, total_vars));
  }
  auto field = get_field<AthelasArray3D<double>>(field_name);
  return Kokkos::subview(field, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::make_pair(comp_start, total_vars));
}

[[nodiscard]] auto MeshState::field_info() const -> std::string {
  std::string info = "# --- Registered Fields ---\n";

  for (const auto &[name, arr_variant] : arrays_) {
    const auto &metadata = get_metadata(name);

    // Field name and policy
    info += "\n" + metadata.name;
    info += " [" +
            std::string(metadata.policy == DataPolicy::Staged ? "Staged"
                                                              : "OneCopy") +
            "]";

    info += ":\n";
    info += "#  Description: " + metadata.description + "\n";

    // Variable names if available
    if (metadata.var_map) {
      info += "#  Variables: ";
      const auto &vars = metadata.var_map->list();
      for (size_t i = 0; i < vars.size(); ++i) {
        if (i > 0) {
          info += ", ";
        }
        info += vars[i];
      }
      info += "\n";
    }
  }

  return info;
}

[[nodiscard]] auto MeshState::has_rad_basis() const noexcept -> bool {
  return rad_basis_ != nullptr;
}

[[nodiscard]] auto MeshState::fluid_basis() const -> const basis::ModalBasis & {
  if (!fluid_basis_) {
    throw_athelas_error("Fluid basis not initialized!");
  }
  return *fluid_basis_;
}

[[nodiscard]] auto MeshState::rad_basis() const -> const basis::ModalBasis & {
  if (!rad_basis_) {
    throw_athelas_error("Radiation basis not initialized!");
  }
  return *rad_basis_;
}
} // namespace athelas
