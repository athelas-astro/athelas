#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "composition/compdata.hpp"
#include "interface/params.hpp"
#include "kokkos_types.hpp"
#include "pgen/problem_in.hpp"

namespace athelas {

enum class DataPolicy {
  Staged, // per-stage storage
  OneCopy // Shared across all stages
};

// Variable index mapping
class VariableMap {
 public:
  void add(std::string name, int index) {
    name_to_index_[std::move(name)] = index;
    index_to_name_[index] = name;
  }

  [[nodiscard]] auto index(const std::string &name) const -> int {
    auto it = name_to_index_.find(name);
    if (it == name_to_index_.end()) {
      THROW_ATHELAS_ERROR("Variable not found: " + name);
    }
    return it->second;
  }

  [[nodiscard]] auto name(int index) const -> std::string {
    auto it = index_to_name_.find(index);
    if (it == index_to_name_.end()) {
      THROW_ATHELAS_ERROR("Invalid variable index: " + std::to_string(index));
    }
    return it->second;
  }

  [[nodiscard]] auto has(const std::string &name) const -> bool {
    return name_to_index_.contains(name);
  }

  [[nodiscard]] auto size() const -> size_t { return name_to_index_.size(); }

  [[nodiscard]] auto list() const -> std::vector<std::string> {
    std::vector<std::string> names;
    names.reserve(name_to_index_.size());
    for (const auto &[name, _] : name_to_index_) {
      names.push_back(name);
    }
    return names;
  }

 private:
  std::unordered_map<std::string, int> name_to_index_;
  std::unordered_map<int, std::string> index_to_name_;
};

/**
 * @struct FieldMetadata
 * @brief holds metadata for a registered Field
 */
struct FieldMetadata {
  std::string name;
  DataPolicy policy;
  int rank;
  std::string description;
  bool allocated;

  // Optional variable mapping for fields with named variables
  std::shared_ptr<VariableMap> var_map;

  FieldMetadata() : policy(DataPolicy::OneCopy), rank(0), allocated(false) {}

  FieldMetadata(std::string n, DataPolicy p, int r, std::string desc)
      : name(std::move(n)), policy(p), rank(r), description(std::move(desc)),
        allocated(false) {}
};

// Forward declaration
class MeshState;

/**
 * @brief Lightweight view of simulation data at a specific RK stage.
 *
 * Provides unified access to both staged (per-RK-stage) and shared (OneCopy)
 * data. Staged fields return the slice for this stage; shared fields are
 * pass-through to the parent MeshState.
 */
class StageData {
 public:
  StageData(int stage, const MeshState *parent)
      : stage_(stage), parent_(parent) {}

  // Generic field access
  [[nodiscard]] auto get_field(const std::string &name) const
      -> AthelasArray3D<double>;

  // Access with variable name instead of index
  [[nodiscard]] auto get_var(const std::string &field,
                             const std::string &var_name, int i, int q) const
      -> double;

  [[nodiscard]] auto comps() const -> atom::CompositionData *;
  [[nodiscard]] auto ionization_state() const -> atom::IonizationState *;
  [[nodiscard]] auto nvars(const std::string &field) const -> int;

  [[nodiscard]] auto ionization_enabled() const noexcept -> bool;
  [[nodiscard]] auto composition_enabled() const noexcept -> bool;
  [[nodiscard]] auto stage() const noexcept -> int { return stage_; }
  [[nodiscard]] auto mass_fractions(const std::string &field_name) const
      -> AthelasArray3D<double>;

 private:
  int stage_;
  const MeshState *parent_;
};

/**
 * @brief Container for all simulation state data and metadata.
 *
 * Manages fluid variables, composition data, and auxiliary fields with support
 * for multi-stage time integration. Fields can be registered as either Staged
 * (one copy per RK stage) or OneCopy (shared across stages). Provides a
 * runtime registry for dynamic field allocation and metadata queries.
 *
 * Access patterns:
 * - Stage-specific: `auto stage_data = mesh_state(stage)`
 *   - Then: `auto u = stage_data.get_field("u_cf")`
 * - Direct access: `mesh_state.get_field("u_pf")`
 * - Metadata: `constexpr int rho_idx = mesh_state.var_index("u_pf", "density")`
 */
class MeshState {
 public:
  MeshState(const ProblemIn *pin, int nstages);

  // MeshState is move only.
  MeshState(const MeshState &) = delete;
  auto operator=(const MeshState &) -> MeshState & = delete;
  MeshState(MeshState &&) noexcept = default;
  auto operator=(MeshState &&) noexcept -> MeshState & = default;
  ~MeshState() = default;

  [[nodiscard]] auto operator()(int stage) const -> StageData {
    return {stage, this};
  }

  [[nodiscard]] auto stage(int stage) const -> StageData {
    return {stage, this};
  }

  [[nodiscard]] auto n_stages() const noexcept -> int { return nstages_; }
  [[nodiscard]] auto p_order() const noexcept -> int {
    return params_->get<int>("p_order");
  }

  // --- Feature Flags ---
  [[nodiscard]] auto composition_enabled() const noexcept -> bool {
    return comps_ != nullptr;
  }

  [[nodiscard]] auto ionization_enabled() const noexcept -> bool {
    return ionization_state_ != nullptr;
  }

  [[nodiscard]] auto composition_evolved() const noexcept -> bool {
    return has_field("mass_fractions");
  }

  void setup_composition(std::shared_ptr<atom::CompositionData> comps) {
    comps_ = std::move(comps);
  }

  void setup_ionization(std::shared_ptr<atom::IonizationState> ion) {
    ionization_state_ = std::move(ion);
  }

  [[nodiscard]] auto params() noexcept -> Params * { return params_.get(); }
  // auto mesh() noexcept -> Mesh* { return mesh_.get(); }

  template <typename T>
  [[nodiscard]] auto get_field(const std::string &name) const -> T {
    auto it = arrays_.find(name);
    if (it == arrays_.end()) {
      THROW_ATHELAS_ERROR("Field does not exist: " + name);
    }

    return std::get<T>(it->second);
  }

  // Get staged field at specific stage (returns 3D subview)
  [[nodiscard]] auto get_field_at_stage(const std::string &name,
                                        int stage) const
      -> AthelasArray3D<double> {
    auto arr_4d = get_field<AthelasArray4D<double>>(name);
    return Kokkos::subview(arr_4d, stage, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  // Variable index access
  [[nodiscard]] auto var_index(const std::string &field,
                               const std::string &var_name) const -> int;

  [[nodiscard]] auto var_name(const std::string &field, int index) const
      -> std::string;

  [[nodiscard]] auto nvars(const std::string &field) const -> int;

  [[nodiscard]] auto comps() const -> atom::CompositionData *;
  [[nodiscard]] auto ionization_state() const -> atom::IonizationState *;

  [[nodiscard]] auto get_metadata(const std::string &field) const
      -> const FieldMetadata & {
    auto it = metadata_.find(field);
    if (it == metadata_.end()) {
      THROW_ATHELAS_ERROR("Field not found: " + field);
    }
    return it->second;
  }

  [[nodiscard]] auto has_field(const std::string &field) const -> bool;
  [[nodiscard]] auto is_allocated(const std::string &field) const -> bool;
  [[nodiscard]] auto is_staged(const std::string &field) const -> bool;
  [[nodiscard]] auto is_onecopy(const std::string &field) const -> bool;
  [[nodiscard]] auto get_comp_start_index(const std::string &field_name) const
      -> int;
  [[nodiscard]] auto mass_fractions(const std::string &field_name,
                                    int stage = 0) const
      -> AthelasArray3D<double>;
  [[nodiscard]] auto field_info() const -> std::string;

  friend class StageData;

  // Register and allocate a field
  template <typename... Dims>
  void register_field(std::string name, DataPolicy policy,
                      std::string description, Dims... dims) {
    assert(!metadata_.contains(name) && "Field already registered!");

    // Create metadata
    metadata_[name] =
        FieldMetadata{name, policy, sizeof...(dims), std::move(description)};
    metadata_[name].allocated = true;

    // Allocate the array. Kind of awful.
    constexpr auto rank = sizeof...(dims);
    if (policy == DataPolicy::Staged) {
      switch (rank) {
      case 1:
        arrays_[name] =
            allocate_nd<AthelasArray2D<double>>(name, nstages_, dims...);
        break;
      case 2:
        arrays_[name] =
            allocate_nd<AthelasArray3D<double>>(name, nstages_, dims...);
        break;
      case 3:
        arrays_[name] =
            allocate_nd<AthelasArray4D<double>>(name, nstages_, dims...);
        break;
      case 4:
        arrays_[name] =
            allocate_nd<AthelasArray5D<double>>(name, nstages_, dims...);
        break;
      default:
        THROW_ATHELAS_ERROR("MeshState: Field registration of rank " +
                            std::to_string(rank) + " is not supported!");
        break;
      }
    } else {
      switch (rank) {
      case 1:
        arrays_[name] = allocate_nd<AthelasArray1D<double>>(name, dims...);
        break;
      case 2:
        arrays_[name] = allocate_nd<AthelasArray2D<double>>(name, dims...);
        break;
      case 3:
        arrays_[name] = allocate_nd<AthelasArray3D<double>>(name, dims...);
        break;
      case 4:
        arrays_[name] = allocate_nd<AthelasArray4D<double>>(name, dims...);
        break;
      default:
        THROW_ATHELAS_ERROR("MeshState: Field registration of rank " +
                            std::to_string(rank) + " is not supported!");
        break;
      }
    }
  }

  // Register field with variable names
  template <typename... Dims>
  void register_field(std::string name, DataPolicy policy,
                      std::string description,
                      std::vector<std::string> var_names, Dims... dims) {
    std::array<std::size_t, sizeof...(dims)> checker{
        static_cast<std::size_t>(dims)...};

    if (checker.back() != var_names.size()) {
      THROW_ATHELAS_ERROR("MeshState::register_field: Last input dimension "
                          "must match var_names.size()!");
    }
    register_field(name, policy, description, dims...);

    auto var_map = std::make_shared<VariableMap>();
    for (std::size_t i = 0; i < var_names.size(); ++i) {
      var_map->add(std::move(var_names[i]), static_cast<int>(i));
    }
    metadata_[name].var_map = std::move(var_map);
  }

 private:
  template <typename T, typename... Dims>
  auto allocate_nd(const std::string &name, Dims... dims) -> T {
    return T(name, dims...);
  }

  // TODO(astrobarker) [MeshState] fold in mesh
  std::unique_ptr<Params> params_;
  std::unordered_map<std::string, FieldMetadata> metadata_;

  using array_type =
      std::variant<AthelasArray1D<double>, AthelasArray2D<double>,
                   AthelasArray3D<double>, AthelasArray4D<double>,
                   AthelasArray5D<double>>;
  std::unordered_map<std::string, array_type> arrays_;

  int nstages_;

  // misc
  std::shared_ptr<atom::CompositionData> comps_;
  std::shared_ptr<atom::IonizationState> ionization_state_;
};

} // namespace athelas
