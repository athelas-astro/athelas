#pragma once

#include <memory>

#include "atom/atom.hpp"
#include "interface/params.hpp"
#include "kokkos_types.hpp"

namespace athelas::atom {

/**
 * @class IonizationState
 * @brief class for holding ionization state. We store here the ionization
 * fractions of species, an AtomicData object (see atom.hpp), the mean
 * ionizaiton state ybar, and several quantities needed in the
 * Paczynski eos for ionization corrections.
 */
class IonizationState {
 public:
  IonizationState(int nX, int nNodes, int n_species, int n_states,
                  const std::string &fn_ionization,
                  const std::string &fn_degeneracy);

  [[nodiscard]] auto ionization_fractions() const noexcept
      -> AthelasArray4D<double>;
  [[nodiscard]] auto atomic_data() const noexcept -> AtomicData *;
  [[nodiscard]] auto ybar() const noexcept -> AthelasArray2D<double>;
  [[nodiscard]] auto e_ion_corr() const noexcept -> AthelasArray2D<double>;
  [[nodiscard]] auto sigma1() const noexcept -> AthelasArray2D<double>;
  [[nodiscard]] auto sigma2() const noexcept -> AthelasArray2D<double>;
  [[nodiscard]] auto sigma3() const noexcept -> AthelasArray2D<double>;

 private:
  AthelasArray4D<double>
      ionization_fractions_; // [nX][nNodes][n_species][max_charge+1]
  std::unique_ptr<AtomicData> atomic_data_;

  // Derived quantities for Paczynski, stored nodally
  AthelasArray2D<double> ybar_; // mean ionization fraction
  AthelasArray2D<double>
      e_ion_corr_; // ionization correction to internal energy
  AthelasArray2D<double> sigma1_;
  AthelasArray2D<double> sigma2_;
  AthelasArray2D<double> sigma3_;
};

/**
 * @class CompositionData
 * TODO(astrobarker): probably moving mass fractions into ucf soon.
 */
class CompositionData {
 public:
  CompositionData(int nX, int order, int n_species);

  [[nodiscard]] auto charge() const noexcept -> AthelasArray1D<int>;
  [[nodiscard]] auto neutron_number() const noexcept -> AthelasArray1D<int>;
  [[nodiscard]] auto ye() const noexcept -> AthelasArray2D<double>;
  [[nodiscard]] auto number_density() const noexcept -> AthelasArray2D<double>;
  [[nodiscard]] auto species_indexer() noexcept -> Params *;
  [[nodiscard]] auto species_indexer() const noexcept -> Params *;

  [[nodiscard]] auto n_species() const noexcept -> size_t {
    return charge_.size();
  }

 private:
  int nX_, order_, n_species_;

  // This params object holds indices of species of interest.
  // For example, for nickel heating, I store indices "ni56" -> int etc.
  // Note: As I am currently using this it corresponds to the index in
  // ucons, i.e., it will never be 0, 1, 2, .. This means that if you pull
  // out the mass_fractions view from State that these will not work!
  // Of course you can define them however you like... but note it!
  // Put whatever you like here.
  std::unique_ptr<Params> species_indexer_;

  AthelasArray2D<double> number_density_; // [nX][order] number per unit mass
  AthelasArray2D<double> ye_; // [nx][nnodes]
  AthelasArray1D<int> charge_; // n_species
  AthelasArray1D<int> neutron_number_;
}; // class CompositionData

} // namespace athelas::atom
