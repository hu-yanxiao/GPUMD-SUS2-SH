#pragma once

#include <cmath>
#include <string>
#include <vector>

#if defined(__CUDACC__)
#define SUS2_ZBL_HD __host__ __device__ __forceinline__
#else
#define SUS2_ZBL_HD inline
#endif

struct SUS2ZBLPairValue {
  double energy;
  double dEdr;
};

SUS2_ZBL_HD double sus2_zbl_default_inner_cutoff()
{
  return 0.7;
}

SUS2_ZBL_HD double sus2_zbl_default_outer_cutoff()
{
  return 1.4;
}

SUS2_ZBL_HD double sus2_zbl_default_typewise_cutoff_factor()
{
  return 0.7;
}

inline double sus2_zbl_covalent_radius(int atomic_number)
{
  static const double radius[94] = {
    0.426667, 0.613333, 1.6,     1.25333, 1.02667, 1.0,     0.946667, 0.84,
    0.853333, 0.893333, 1.86667, 1.66667, 1.50667, 1.38667, 1.46667, 1.36,
    1.32,     1.28,    2.34667,  2.05333, 1.77333, 1.62667, 1.61333, 1.46667,
    1.42667,  1.38667, 1.33333,  1.32,    1.34667, 1.45333, 1.49333, 1.45333,
    1.53333,  1.46667, 1.52,     1.56,    2.52,    2.22667, 1.96,    1.85333,
    1.76,     1.65333, 1.53333,  1.50667, 1.50667, 1.44,    1.53333, 1.64,
    1.70667,  1.68,    1.68,     1.64,    1.76,    1.74667, 2.78667, 2.34667,
    2.16,     1.96,    2.10667,  2.09333, 2.08,    2.06667, 2.01333, 2.02667,
    2.01333,  2.0,     1.98667,  1.98667, 1.97333, 2.04,    1.94667, 1.82667,
    1.74667,  1.64,    1.57333,  1.54667, 1.48,    1.49333, 1.50667, 1.76,
    1.73333,  1.73333, 1.81333,  1.74667, 1.84,    1.89333, 2.68,    2.41333,
    2.22667,  2.10667, 2.02667,  2.04,    2.05333, 2.06667};
  if (atomic_number <= 0 || atomic_number > 94) {
    return 0.0;
  }
  return radius[atomic_number - 1];
}

inline int sus2_zbl_atomic_number_from_symbol(const std::string& symbol)
{
  static const char* symbols[94] = {
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg",
    "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
    "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};
  for (int i = 0; i < 94; ++i) {
    if (symbol == symbols[i]) {
      return i + 1;
    }
  }
  return 0;
}

inline double sus2_zbl_typewise_outer_cutoff(
  int atomic_number_i,
  int atomic_number_j,
  double global_outer_cutoff,
  double typewise_cutoff_factor)
{
  const double ri = sus2_zbl_covalent_radius(atomic_number_i);
  const double rj = sus2_zbl_covalent_radius(atomic_number_j);
  if (ri <= 0.0 || rj <= 0.0 || global_outer_cutoff <= 0.0 ||
      typewise_cutoff_factor < 0.5) {
    return 0.0;
  }
  const double pair_outer = typewise_cutoff_factor * (ri + rj);
  return pair_outer < global_outer_cutoff ? pair_outer : global_outer_cutoff;
}

inline bool sus2_zbl_fill_pair_cutoff_tables(
  const std::vector<int>& atomic_numbers,
  double global_inner_cutoff,
  double global_outer_cutoff,
  bool typewise_cutoff_enabled,
  double typewise_cutoff_factor,
  std::vector<double>& pair_inner_cutoffs,
  std::vector<double>& pair_outer_cutoffs,
  std::vector<double>& pair_outer_sq,
  std::string* error_message)
{
  const int species_count = static_cast<int>(atomic_numbers.size());
  if (species_count <= 0) {
    if (error_message != nullptr) {
      *error_message = "ZBL requires one atomic number per model species.";
    }
    return false;
  }
  if (global_inner_cutoff < 0.0) {
    if (error_message != nullptr) {
      *error_message = "ZBL inner cutoff should be non-negative.";
    }
    return false;
  }
  if (global_outer_cutoff <= 0.0) {
    if (error_message != nullptr) {
      *error_message = "ZBL outer cutoff should be positive.";
    }
    return false;
  }
  if (typewise_cutoff_enabled) {
    if (typewise_cutoff_factor < 0.5) {
      if (error_message != nullptr) {
        *error_message = "ZBL typewise cutoff factor should be at least 0.5.";
      }
      return false;
    }
  } else if (global_outer_cutoff <= global_inner_cutoff) {
    if (error_message != nullptr) {
      *error_message = "ZBL cutoffs should satisfy 0 <= inner < outer.";
    }
    return false;
  }

  pair_inner_cutoffs.resize(static_cast<size_t>(species_count) * species_count);
  pair_outer_cutoffs.resize(static_cast<size_t>(species_count) * species_count);
  pair_outer_sq.resize(static_cast<size_t>(species_count) * species_count);
  for (int i = 0; i < species_count; ++i) {
    const int Zi = atomic_numbers[i];
    if (Zi <= 0 || (typewise_cutoff_enabled && sus2_zbl_covalent_radius(Zi) <= 0.0)) {
      if (error_message != nullptr) {
        *error_message = "ZBL atomic numbers should be in [1, 94].";
      }
      return false;
    }
    for (int j = 0; j < species_count; ++j) {
      const int Zj = atomic_numbers[j];
      if (Zj <= 0 || (typewise_cutoff_enabled && sus2_zbl_covalent_radius(Zj) <= 0.0)) {
        if (error_message != nullptr) {
          *error_message = "ZBL atomic numbers should be in [1, 94].";
        }
        return false;
      }
      const double inner = typewise_cutoff_enabled ? 0.0 : global_inner_cutoff;
      const double outer = typewise_cutoff_enabled ?
        sus2_zbl_typewise_outer_cutoff(Zi, Zj, global_outer_cutoff, typewise_cutoff_factor) :
        global_outer_cutoff;
      if (outer <= inner) {
        if (error_message != nullptr) {
          *error_message = "ZBL pair cutoffs should satisfy inner < outer.";
        }
        return false;
      }
      const size_t pair_index = static_cast<size_t>(i) * species_count + j;
      pair_inner_cutoffs[pair_index] = inner;
      pair_outer_cutoffs[pair_index] = outer;
      pair_outer_sq[pair_index] = outer * outer;
    }
  }
  return true;
}

SUS2_ZBL_HD SUS2ZBLPairValue sus2_zbl_pair(
  int atomic_number_i,
  int atomic_number_j,
  double distance,
  double inner_cutoff,
  double outer_cutoff)
{
  if (atomic_number_i <= 0 || atomic_number_j <= 0 || distance <= 0.0 ||
      outer_cutoff <= 0.0 || distance >= outer_cutoff) {
    return SUS2ZBLPairValue{0.0, 0.0};
  }

  const double ev_angstrom_per_e2 = 14.3996454784255;
  const double screening_inv =
    2.134563 * (pow(static_cast<double>(atomic_number_i), 0.23) +
                pow(static_cast<double>(atomic_number_j), 0.23));
  const double x = screening_inv * distance;

  double phi = 0.0;
  double dphi_dr = 0.0;
  double exp_value = exp(-3.1998 * x);
  phi += 0.18175 * exp_value;
  dphi_dr -= 0.18175 * 3.1998 * screening_inv * exp_value;
  exp_value = exp(-0.94229 * x);
  phi += 0.50986 * exp_value;
  dphi_dr -= 0.50986 * 0.94229 * screening_inv * exp_value;
  exp_value = exp(-0.4029 * x);
  phi += 0.28022 * exp_value;
  dphi_dr -= 0.28022 * 0.4029 * screening_inv * exp_value;
  exp_value = exp(-0.20162 * x);
  phi += 0.02817 * exp_value;
  dphi_dr -= 0.02817 * 0.20162 * screening_inv * exp_value;

  const double prefactor =
    ev_angstrom_per_e2 * static_cast<double>(atomic_number_i) *
    static_cast<double>(atomic_number_j);
  const double base_energy = prefactor * phi / distance;
  const double base_dEdr =
    prefactor * (dphi_dr / distance - phi / (distance * distance));

  double switch_value = 1.0;
  double switch_derivative = 0.0;
  if (distance > inner_cutoff) {
    const double pi_factor = 3.141592653589793238462643383279502884 /
      (outer_cutoff - inner_cutoff);
    switch_value = 0.5 * cos(pi_factor * (distance - inner_cutoff)) + 0.5;
    switch_derivative =
      -0.5 * pi_factor * sin(pi_factor * (distance - inner_cutoff));
  }

  SUS2ZBLPairValue value;
  value.energy = switch_value * base_energy;
  value.dEdr = switch_value * base_dEdr + switch_derivative * base_energy;
  return value;
}

#undef SUS2_ZBL_HD
