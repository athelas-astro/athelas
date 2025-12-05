/*
 * @file atom.hpp
 * --------------
 *
 * @brief Atomic data structures
 * @note Used to hold NIST atomic data
 *   Designed for an array of structures pattern,
 *   where algorithms should loop over species, then cells.
 *
 * We expect to read ionization potentials in eV and then convert to erg.
 */

#pragma once

#include <vector>

#include "io/parser.hpp"
#include "kokkos_types.hpp"
#include "print"
#include "utils/constants.hpp"

namespace athelas::atom {

/**
 * @struct IonLevel
 * @brief Holds ionization energy and degeneracy weights for ionization level
 * Note sure if this layout is optimal or ideal. It will have some
 * data duplication.
 * Chi is the ionization potential to go form ionization state n -> n+1
 * g_lower and g_upper are the degeneracy weights of states n, n+1.
 */
struct IonLevel {
  double chi; // ionization potential (n-1)
  double g_lower; // degeneracy of lower state (n-1)
  double g_upper; // degeneracy of upper state (n)
};

/**
 * @class AtomicData
 * @brief Class for holding atomic data -- ionization potentials and
 *   degeneracy factors for many species.
 * As it stands, the constructor assumes that atomic species are
 * contiguous. If that assumption is broken then this will be as well.
 */
class AtomicData {
 private:
  AthelasArray1D<IonLevel> ion_data_;
  AthelasArray1D<int> offsets_;
  AthelasArray1D<int> atomic_numbers_;
  AthelasArray1D<double> sum_pot_; // sum of ionization potentials, flattened.
  int num_species_;

 public:
  AtomicData() = default;

  AtomicData(const std::string &fn_ionization,
             const std::string &fn_degeneracy) {

    // --- load atomic data from file ---
    auto ionization_data = io::Parser::parse_file(fn_ionization);
    auto degeneracy_data = io::Parser::parse_file(fn_degeneracy);

    // -- extract columns ---
    auto [atomic_numbers, ion_charges, ionization_energies] =
        get_columns_by_indices<int, int, double>(*ionization_data, 0, 1, 2);
    auto [atomic_numbers_2, ion_charges_, weights] =
        get_columns_by_indices<int, int, double>(*degeneracy_data, 0, 1, 2);

    // NOTE: This explicitly assumes that the data in the files are ordered
    // and contiguous. If this assumption is broken, then this is incorrect.
    // TODO(astrobarker): Sorting the data after extracting would harden it.
    num_species_ = static_cast<int>(atomic_numbers.back());

    int total_levels = 0;
    std::vector<int> offs(num_species_);

    for (int s = 0; s < num_species_; ++s) {
      const int Z = s + 1;
      offs[s] = total_levels;
      total_levels += Z; // Z ionization levels
    }

    ion_data_ =
        AthelasArray1D<IonLevel>("ion_data", total_levels + num_species_ + 2);
    atomic_numbers_ = AthelasArray1D<int>("atomic_number", num_species_ + 0);
    offsets_ = AthelasArray1D<int>("offsets", num_species_);
    sum_pot_ =
        AthelasArray1D<double>("sum_pot", total_levels + num_species_ + 2);

    auto offs_host = Kokkos::create_mirror_view(offsets_);
    auto ion_data_host = Kokkos::create_mirror_view(ion_data_);
    auto atomic_numbers_host = Kokkos::create_mirror_view(atomic_numbers_);
    auto sum_pot_h = Kokkos::create_mirror_view(sum_pot_);
    for (int s = 0; s < num_species_; ++s) {
      offs_host(s) = offs[s];
    }

    // load atomic numbers. kind of horrible
    int ind = 0;
    for (auto &z : atomic_numbers) {
      if (ind != 0) {
        if (atomic_numbers_host(ind - 1) == z) { // deal with repeats
          continue;
        }
      }
      atomic_numbers_host(ind) = z;
      ind++;
    }

    // process ionization and degerneracy data
    int chi_idx = 0; // Index into ionization potentials vector
    int g_idx = 0; // Index into degeneracy factors vector
    int level_idx = 0; // Index into our ion_data_ array

    for (int s = 0; s < num_species_; ++s) {
      const int Z = atomic_numbers_host(s);
      sum_pot_h(level_idx) =
          ionization_energies[chi_idx] * constants::ev_to_erg;

      // For species s, we have:
      // - Z ionization potentials: chi[0] to chi[Z-1]
      // - Z+1 degeneracy factors: g[0] to g[Z]
      // - Z ionization levels in our data structure

      for (int n = 0; n < Z; ++n) { // n goes from 0 to Z-1
        // Level n represents ionization from state n to state n+1
        ion_data_host(level_idx).chi =
            ionization_energies[chi_idx + n] * constants::ev_to_erg;
        sum_pot_h(level_idx + 1) =
            sum_pot_h(level_idx) +
            ionization_energies[chi_idx + n + 1] * constants::ev_to_erg;
        ion_data_host(level_idx).g_lower = weights[g_idx + n]; // g[n]
        ion_data_host(level_idx).g_upper = weights[g_idx + n + 1]; // g[n+1]

        level_idx++;
      }

      // Advance indices for next species
      chi_idx += Z;
      g_idx += Z + 1;
    }

    Kokkos::deep_copy(offsets_, offs_host);
    Kokkos::deep_copy(ion_data_, ion_data_host);
    Kokkos::deep_copy(atomic_numbers_, atomic_numbers_host);
    Kokkos::deep_copy(sum_pot_, sum_pot_h);
  }

  [[nodiscard]] auto offsets() const noexcept { return offsets_; }
  [[nodiscard]] auto ion_data() const noexcept { return ion_data_; }
  [[nodiscard]] auto sum_pots() const noexcept { return sum_pot_; }
};

[[nodiscard]] KOKKOS_INLINE_FUNCTION auto
species_data(const AthelasArray1D<IonLevel> ion_data,
             const AthelasArray1D<int> offsets, const size_t species) {
  const size_t num_species = offsets.size();
  const size_t offset = offsets(species - 1);
  const size_t next_offset =
      (species < num_species) ? offsets(species) + 1 : ion_data.extent(0);
  return Kokkos::subview(ion_data, Kokkos::make_pair(offset, next_offset));
}

[[nodiscard]] KOKKOS_INLINE_FUNCTION auto
species_data(const AthelasArray1D<double> sum_pots,
             const AthelasArray1D<int> offsets, const size_t species) {
  const size_t num_species = offsets.size();
  const size_t offset = offsets(species - 1);
  const size_t next_offset =
      (species < num_species) ? offsets(species) + 1 : sum_pots.extent(0);
  return Kokkos::subview(sum_pots, Kokkos::make_pair(offset, next_offset));
}

} // namespace athelas::atom
