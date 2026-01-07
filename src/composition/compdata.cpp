#include <memory>

#include "composition/compdata.hpp"
#include "utils/error.hpp"

namespace athelas::atom {

// NOTE: if nodes exceeds nnodes we have a problem here.
CompositionData::CompositionData(const int nX, const int nnodes,
                                 const int n_species)
    : nX_(nX), nnodes_(nnodes), n_species_(n_species),
      species_indexer_(std::make_unique<Params>()) {

  ye_ = AthelasArray2D<double>("ye", nX, nnodes + 2);
  abar_ = AthelasArray2D<double>("abar", nX, nnodes + 2);
  number_density_ = AthelasArray2D<double>("ye", nX, nnodes + 2);
  electron_number_density_ = AthelasArray2D<double>("ye", nX, nnodes + 2);
  charge_ = AthelasArray1D<int>("charge", n_species);
  neutron_number_ = AthelasArray1D<int>("neutron_number", n_species);
  inverse_atomic_mass_ =
      AthelasArray1D<double>("inverse atomic mass", n_species);

  if (n_species <= 0) {
    throw_athelas_error("CompositionData :: n_species must be > 0!");
  }
}

[[nodiscard]] auto CompositionData::charge() const noexcept
    -> AthelasArray1D<int> {
  return charge_;
}
[[nodiscard]] auto CompositionData::neutron_number() const noexcept
    -> AthelasArray1D<int> {
  return neutron_number_;
}
[[nodiscard]] auto CompositionData::inverse_atomic_mass() const noexcept
    -> AthelasArray1D<double> {
  return inverse_atomic_mass_;
}
[[nodiscard]] auto CompositionData::ye() const noexcept
    -> AthelasArray2D<double> {
  return ye_;
}
[[nodiscard]] auto CompositionData::abar() const noexcept
    -> AthelasArray2D<double> {
  return abar_;
}
[[nodiscard]] auto CompositionData::number_density() const noexcept
    -> AthelasArray2D<double> {
  return number_density_;
}
[[nodiscard]] auto CompositionData::electron_number_density() const noexcept
    -> AthelasArray2D<double> {
  return electron_number_density_;
}
[[nodiscard]] auto CompositionData::species_indexer() noexcept -> Params * {
  return species_indexer_.get();
}
[[nodiscard]] auto CompositionData::species_indexer() const noexcept
    -> Params * {
  return species_indexer_.get();
}

// --- end CompositionData ---

/**
 * @brief IonizationState constructor
 * @note: Something awful here: need an estimate of the max number of
 * ionization states when constructing the comps object
 * We allocate in general more data than necessary
 *
 * TODO(astrobarker): flatten last two dimensions to reduce memory footprint.
 * See atom.hpp
 * TODO(astrobarker): We can now just construct this with state.
 */
IonizationState::IonizationState(const int nX, const int nNodes,
                                 const int n_species, const int n_states,
                                 const int saha_ncomps,
                                 const std::string &fn_ionization,
                                 const std::string &fn_degeneracy)
    : saha_ncomps_(saha_ncomps),
      ionization_fractions_("ionization_fractions", nX, nNodes + 2, n_species,
                            n_states),
      atomic_data_(std::make_unique<AtomicData>(fn_ionization, fn_degeneracy)),
      zbar_("zbar", nX, nNodes + 2, n_species), ybar_("ybar", nX, nNodes + 2),
      e_ion_corr_("e_ion_corr", nX, nNodes + 2), saha_f_("saha_f", n_states),
      sigma1_("sigma1", nX, nNodes + 2), sigma2_("sigma2", nX, nNodes + 2),
      sigma3_("sigma3", nX, nNodes + 2) {
  if (n_species <= 0) {
    throw_athelas_error("IonizationState :: n_species must be > 0!");
  }
  if (n_states <= 0) {
    throw_athelas_error("IonizationState :: n_states must be > 0!");
  }
}

[[nodiscard]] auto IonizationState::ncomps() const noexcept -> int {
  return saha_ncomps_;
}

[[nodiscard]] auto IonizationState::ionization_fractions() const noexcept
    -> AthelasArray4D<double> {
  return ionization_fractions_;
}

[[nodiscard]] auto IonizationState::atomic_data() const noexcept
    -> AtomicData * {
  return atomic_data_.get();
}

[[nodiscard]] auto IonizationState::zbar() const noexcept
    -> AthelasArray3D<double> {
  return zbar_;
}

[[nodiscard]] auto IonizationState::ybar() const noexcept
    -> AthelasArray2D<double> {
  return ybar_;
}

[[nodiscard]] auto IonizationState::e_ion_corr() const noexcept
    -> AthelasArray2D<double> {
  return e_ion_corr_;
}

[[nodiscard]] auto IonizationState::saha_factor() const noexcept
    -> AthelasArray1D<double> {
  return saha_f_;
}

[[nodiscard]] auto IonizationState::sigma1() const noexcept
    -> AthelasArray2D<double> {
  return sigma1_;
}

[[nodiscard]] auto IonizationState::sigma2() const noexcept
    -> AthelasArray2D<double> {
  return sigma2_;
}

[[nodiscard]] auto IonizationState::sigma3() const noexcept
    -> AthelasArray2D<double> {
  return sigma3_;
}

} // namespace athelas::atom
