#include "src/force/sus2_zbl_common.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

void check_close(double got, double expected, double tol, const char* label)
{
  const double diff = std::fabs(got - expected);
  if (diff > tol) {
    std::fprintf(
      stderr, "%s mismatch: got %.17g expected %.17g diff %.3g tol %.3g\n",
      label, got, expected, diff, tol);
    std::exit(1);
  }
}

SUS2ZBLPairValue reference_zbl_pair(
  int Zi, int Zj, double r, double inner, double outer)
{
  if (r >= outer) {
    return SUS2ZBLPairValue{0.0, 0.0};
  }
  const double coeff[4] = {0.18175, 0.50986, 0.28022, 0.02817};
  const double expo[4] = {3.1998, 0.94229, 0.4029, 0.20162};
  const double prefactor = 14.3996454784255 * Zi * Zj;
  const double screening_inv =
    2.134563 * (std::pow(static_cast<double>(Zi), 0.23) +
                std::pow(static_cast<double>(Zj), 0.23));
  const double x = screening_inv * r;
  double phi = 0.0;
  double dphi = 0.0;
  for (int i = 0; i < 4; ++i) {
    const double e = std::exp(-expo[i] * x);
    phi += coeff[i] * e;
    dphi -= coeff[i] * expo[i] * screening_inv * e;
  }
  const double base_e = prefactor * phi / r;
  const double base_de = prefactor * (dphi / r - phi / (r * r));
  double sw = 1.0;
  double dsw = 0.0;
  if (r > inner) {
    const double pi_factor = std::acos(-1.0) / (outer - inner);
    sw = 0.5 * std::cos(pi_factor * (r - inner)) + 0.5;
    dsw = -0.5 * pi_factor * std::sin(pi_factor * (r - inner));
  }
  return SUS2ZBLPairValue{sw * base_e, sw * base_de + dsw * base_e};
}

} // namespace

int main()
{
  check_close(sus2_zbl_default_inner_cutoff(), 0.7, 0.0, "default inner");
  check_close(sus2_zbl_default_outer_cutoff(), 1.4, 0.0, "default outer");
  check_close(sus2_zbl_default_typewise_cutoff_factor(), 0.7, 0.0, "default factor");

  if (sus2_zbl_atomic_number_from_symbol("H") != 1 ||
      sus2_zbl_atomic_number_from_symbol("C") != 6 ||
      sus2_zbl_atomic_number_from_symbol("Cl") != 17 ||
      sus2_zbl_atomic_number_from_symbol("Co") != 27) {
    std::fprintf(stderr, "atomic symbol mapping mismatch\n");
    return 1;
  }

  check_close(sus2_zbl_covalent_radius(1), 0.426667, 0.0, "H covalent radius");
  check_close(sus2_zbl_covalent_radius(6), 1.0, 0.0, "C covalent radius");

  const double hc_outer = sus2_zbl_typewise_outer_cutoff(1, 6, 1.4, 0.7);
  check_close(hc_outer, 0.7 * (0.426667 + 1.0), 1.0e-14, "H-C typewise outer");

  std::vector<int> atomic_numbers = {1, 6};
  std::vector<double> inner;
  std::vector<double> outer;
  std::vector<double> outer_sq;
  std::string error;
  if (!sus2_zbl_fill_pair_cutoff_tables(
        atomic_numbers, 0.7, 1.4, true, 0.7, inner, outer, outer_sq, &error)) {
    std::fprintf(stderr, "fill pair cutoffs failed: %s\n", error.c_str());
    return 1;
  }
  check_close(inner[1], 0.0, 0.0, "typewise inner");
  check_close(outer[1], hc_outer, 1.0e-14, "cached H-C outer");
  check_close(outer_sq[1], hc_outer * hc_outer, 1.0e-14, "cached H-C outer sq");

  const SUS2ZBLPairValue got = sus2_zbl_pair(1, 6, 0.9, inner[1], outer[1]);
  const SUS2ZBLPairValue ref = reference_zbl_pair(1, 6, 0.9, inner[1], outer[1]);
  check_close(got.energy, ref.energy, 1.0e-12, "ZBL pair energy");
  check_close(got.dEdr, ref.dEdr, 1.0e-12, "ZBL pair dEdr");
  const SUS2ZBLPairValue outside = sus2_zbl_pair(1, 6, outer[1], inner[1], outer[1]);
  check_close(outside.energy, 0.0, 0.0, "outside energy");
  check_close(outside.dEdr, 0.0, 0.0, "outside dEdr");

  std::puts("GPUMD-SH ZBL common checks passed.");
  return 0;
}
