#include "sus2_v11.cuh"

#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr int kBlockSize = 128;
constexpr int kJacobiMaxIndexedBlock = 5;
constexpr int kSus2MaxConstAlphaTimes = 6000;
constexpr double kLaguerreMinRho = 1.0e-8;
constexpr double kLaguerrePositiveParamFloor = 1.0e-6;

__constant__ unsigned short c_sus2_alpha_times_u16[kSus2MaxConstAlphaTimes * 4];

enum class RadialBasisKind {
  JacobiSSS,
  JacobiSSSNoWeight,
  ChebyshevSSS,
  LaguerreLog1p,
  LaguerreLog1pNoEnv,
  LaguerreLog1pPositive
};

struct JacobiBlockSpec {
  int alpha;
  int beta;
  double linear_const;
  double linear_x;
};

struct SUS2HostModel {
  int species_count = 0;
  int angular_channels = 0;
  int scaling_block_count = 0;
  int radial_funcs_count = 0;
  int rb_size = 0;
  int alpha_basic_count = 0;
  int alpha_times_count = 0;
  int alpha_moments_count = 0;
  int alpha_scalar_moments = 0;
  int original_alpha_times_count = 0;
  int original_alpha_moments_count = 0;
  int max_rank = 0;
  double scaling = 1.0;
  double max_dist = 0.0;
  std::string radial_basis_type;
  std::string scaling_map;
  RadialBasisKind radial_basis_kind = RadialBasisKind::JacobiSSS;
  std::vector<double> shift_coeffs;
  std::vector<double> scal_coeffs;
  std::vector<double> radial_coeffs;
  std::vector<double> radial_type_coeffs;
  std::vector<double> species_coeffs;
  std::vector<double> moment_coeffs;
  std::vector<int> alpha_basic;
  std::vector<int> alpha_times;
  std::vector<int> alpha_moment_mapping;
  std::vector<int> mu_to_scaling_block;
  std::vector<int> mu_to_jacobi_block;
};

struct SUS2DeviceModel {
  int species_count;
  int radial_funcs_count;
  int alpha_basic_count;
  int alpha_times_count;
  int alpha_moments_count;
  int alpha_scalar_moments;
  int max_rank;
  int lut_size;
  double max_dist;
  double lut_inv_dr;
  const double* shift_coeffs;
  const double* species_coeffs;
  const double* moment_coeffs;
  const int* alpha_basic;
  const int* alpha_times;
  const int* alpha_moment_mapping;
  const float* lut_vals;
  const float* lut_ders;
  bool use_l3k3_basic_fastpath;
  bool use_const_alpha_times;
};

[[noreturn]] void sus2_input_error(const std::string& message)
{
  std::cout << message << std::endl;
  std::exit(1);
}

std::string read_text_file(const std::string& path)
{
  std::ifstream ifs(path);
  if (!ifs) {
    sus2_input_error("Failed to open SUS2 v1.1 model file: " + path);
  }
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

std::string trim(const std::string& text)
{
  size_t begin = 0;
  while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
    ++begin;
  }
  size_t end = text.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
    --end;
  }
  return text.substr(begin, end - begin);
}

size_t find_required(const std::string& text, const std::string& token, size_t from = 0)
{
  const size_t pos = text.find(token, from);
  if (pos == std::string::npos) {
    sus2_input_error("Missing token in SUS2 v1.1 model file: " + token);
  }
  return pos;
}

std::string parse_string_after(const std::string& text, const std::string& token)
{
  const size_t pos = find_required(text, token) + token.size();
  const size_t end = text.find('\n', pos);
  return trim(text.substr(pos, end == std::string::npos ? std::string::npos : end - pos));
}

double parse_double_after(const std::string& text, const std::string& token)
{
  return std::stod(parse_string_after(text, token));
}

int parse_int_after(const std::string& text, const std::string& token)
{
  return std::stoi(parse_string_after(text, token));
}

RadialBasisKind radial_basis_kind_from_string(const std::string& type)
{
  if (type == "RBJacobi_sss" || type == "RBJacobi_sss_lmp") {
    return RadialBasisKind::JacobiSSS;
  }
  if (type == "RBJacobi_sss_noweight" || type == "RBJacobi_sss_noweight_lmp") {
    return RadialBasisKind::JacobiSSSNoWeight;
  }
  if (type == "RBChebyshev_sss" || type == "RBChebyshev_sss_lmp") {
    return RadialBasisKind::ChebyshevSSS;
  }
  if (type == "RBLaguerre_log1p" || type == "RBLaguerre_log1p_lmp") {
    return RadialBasisKind::LaguerreLog1p;
  }
  if (type == "RBLaguerre_log1p_noenv" || type == "RBLaguerre_log1p_noenv_lmp") {
    return RadialBasisKind::LaguerreLog1pNoEnv;
  }
  if (type == "RBLaguerre_log1p_pos" || type == "RBLaguerre_log1p_pos_lmp") {
    return RadialBasisKind::LaguerreLog1pPositive;
  }
  sus2_input_error(
    "Unsupported SUS2 v1.1 radial_basis_type in GPUMD: " + type +
    ". Supported now: RBJacobi_sss[_lmp], RBJacobi_sss_noweight[_lmp], "
    "RBChebyshev_sss[_lmp], RBLaguerre_log1p[_lmp], RBLaguerre_log1p_noenv[_lmp], "
    "RBLaguerre_log1p_pos[_lmp].");
}

std::string extract_braced_after(const std::string& text, const std::string& token)
{
  size_t pos = find_required(text, token);
  pos = find_required(text, "{", pos);
  int depth = 0;
  size_t start = pos;
  for (size_t i = pos; i < text.size(); ++i) {
    if (text[i] == '{') {
      if (depth == 0) {
        start = i;
      }
      ++depth;
    } else if (text[i] == '}') {
      --depth;
      if (depth == 0) {
        return text.substr(start + 1, i - start - 1);
      }
    }
  }
  sus2_input_error("Unbalanced braces after token: " + token);
}

std::vector<std::string> extract_all_brace_groups(const std::string& text)
{
  std::vector<std::string> groups;
  int depth = 0;
  size_t start = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '{') {
      if (depth == 0) {
        start = i;
      }
      ++depth;
    } else if (text[i] == '}') {
      --depth;
      if (depth == 0) {
        groups.emplace_back(text.substr(start + 1, i - start - 1));
      }
    }
  }
  return groups;
}

template <typename T>
std::vector<T> parse_numbers(const std::string& text);

template <>
std::vector<double> parse_numbers<double>(const std::string& text)
{
  std::vector<double> values;
  const char* cur = text.c_str();
  char* end = nullptr;
  while (*cur != '\0') {
    const double value = std::strtod(cur, &end);
    if (end != cur) {
      values.push_back(value);
      cur = end;
    } else {
      ++cur;
    }
  }
  return values;
}

template <>
std::vector<int> parse_numbers<int>(const std::string& text)
{
  std::vector<int> values;
  const char* cur = text.c_str();
  char* end = nullptr;
  while (*cur != '\0') {
    const long value = std::strtol(cur, &end, 10);
    if (end != cur) {
      values.push_back(static_cast<int>(value));
      cur = end;
    } else {
      ++cur;
    }
  }
  return values;
}

const JacobiBlockSpec& jacobi_block_spec_for_index(int k)
{
  static const std::array<JacobiBlockSpec, kJacobiMaxIndexedBlock + 1> table = {{
    {0, 0, 0.0, 1.0},
    {1, 0, 0.5, 1.5},
    {1, 1, 0.0, 2.0},
    {2, 0, 1.0, 2.0},
    {2, 1, 0.5, 2.5},
    {2, 2, 0.0, 3.0},
  }};
  if (k < 0 || k > kJacobiMaxIndexedBlock) {
    sus2_input_error("RBJacobi_sss_lmp supports only indexed Jacobi blocks k=0..5.");
  }
  return table[static_cast<size_t>(k)];
}

void jacobi_coefficients_for_order(
  int block,
  int order,
  double& coeff_const,
  double& coeff_x,
  double& prev_coeff)
{
  const JacobiBlockSpec& spec = jacobi_block_spec_for_index(block);
  const double alpha = static_cast<double>(spec.alpha);
  const double beta = static_cast<double>(spec.beta);
  const double n = static_cast<double>(order);
  const double denom = 2.0 * n * (n + alpha + beta) * (2.0 * n + alpha + beta - 2.0);
  const double b = 2.0 * n + alpha + beta - 1.0;
  const double c = (2.0 * n + alpha + beta) * (2.0 * n + alpha + beta - 2.0);
  const double d = alpha * alpha - beta * beta;
  const double e = 2.0 * (n + alpha - 1.0) * (n + beta - 1.0) * (2.0 * n + alpha + beta);
  coeff_const = b * d / denom;
  coeff_x = b * c / denom;
  prev_coeff = e / denom;
}

void jacobi_weight_terms(
  const JacobiBlockSpec& spec,
  double x,
  double& sqrt_weight,
  double& log_weight_x,
  double& log_weight_xx)
{
  constexpr double eps = 1.0e-12;
  sqrt_weight = 1.0;
  log_weight_x = 0.0;
  log_weight_xx = 0.0;

  const double one_minus_x = std::max(eps, 1.0 - x);
  const double one_plus_x = std::max(eps, 1.0 + x);
  if (spec.alpha == 1) {
    sqrt_weight *= std::sqrt(one_minus_x);
  } else if (spec.alpha == 2) {
    sqrt_weight *= one_minus_x;
  }
  if (spec.beta == 1) {
    sqrt_weight *= std::sqrt(one_plus_x);
  } else if (spec.beta == 2) {
    sqrt_weight *= one_plus_x;
  }
  if (spec.alpha != 0) {
    const double inv = 1.0 / one_minus_x;
    log_weight_x -= 0.5 * static_cast<double>(spec.alpha) * inv;
    log_weight_xx -= 0.5 * static_cast<double>(spec.alpha) * inv * inv;
  }
  if (spec.beta != 0) {
    const double inv = 1.0 / one_plus_x;
    log_weight_x += 0.5 * static_cast<double>(spec.beta) * inv;
    log_weight_xx -= 0.5 * static_cast<double>(spec.beta) * inv * inv;
  }
}

void jacobi_sss_calc_host(
  int rb_size,
  double basis_scaling,
  double max_dist,
  double r,
  double scal,
  double s,
  int block,
  bool apply_weight,
  double* vals,
  double* ders)
{
  constexpr double eps = 1.0e-12;
  const JacobiBlockSpec& spec = jacobi_block_spec_for_index(block);
  const double z = 0.5 * scal * (r - s);
  double x = std::tanh(z);
  x = std::max(-1.0 + eps, std::min(1.0 - eps, x));
  const double sech_sq = 1.0 - x * x;
  const double x_r = 0.5 * scal * sech_sq;

  double sqrt_weight = 1.0;
  double log_weight_x = 0.0;
  double log_weight_xx = 0.0;
  if (apply_weight) {
    jacobi_weight_terms(spec, x, sqrt_weight, log_weight_x, log_weight_xx);
  }

  double y_prev = 0.0;
  double y_prev_x = 0.0;
  double y_curr = sqrt_weight;
  double y_curr_x = sqrt_weight * log_weight_x;

  const double dr = r - max_dist;
  const double cutoff_f = dr * dr;
  const double cutoff_der = 2.0 * dr;
  const double scaled_cutoff_f = basis_scaling * cutoff_f;
  const double scaled_cutoff_der = basis_scaling * cutoff_der;

  auto store_basis = [&](int index, double y, double y_x) {
    vals[index] = scaled_cutoff_f * y;
    ders[index] = scaled_cutoff_der * y + scaled_cutoff_f * y_x * x_r;
  };

  store_basis(0, y_curr, y_curr_x);
  if (rb_size == 1) {
    return;
  }

  const double linear = spec.linear_const + spec.linear_x * x;
  const double linear_x = spec.linear_x;
  double y_next = linear * y_curr;
  double y_next_x = linear_x * y_curr + linear * y_curr_x;
  store_basis(1, y_next, y_next_x);

  y_prev = y_curr;
  y_prev_x = y_curr_x;
  y_curr = y_next;
  y_curr_x = y_next_x;
  for (int order = 2; order < rb_size; ++order) {
    double coeff_const = 0.0;
    double coeff_x = 0.0;
    double prev_coeff = 0.0;
    jacobi_coefficients_for_order(block, order, coeff_const, coeff_x, prev_coeff);
    const double coeff = coeff_const + coeff_x * x;
    y_next = coeff * y_curr - prev_coeff * y_prev;
    y_next_x = coeff_x * y_curr + coeff * y_curr_x - prev_coeff * y_prev_x;
    store_basis(order, y_next, y_next_x);
    y_prev = y_curr;
    y_prev_x = y_curr_x;
    y_curr = y_next;
    y_curr_x = y_next_x;
  }
}

double stable_softplus(double x)
{
  if (x > 40.0) {
    return x;
  }
  if (x < -40.0) {
    return std::exp(x);
  }
  return std::log1p(std::exp(x));
}

void laguerre_log1p_calc_host(
  int rb_size,
  double basis_scaling,
  double max_dist,
  double r,
  double scal_raw,
  double s_raw,
  bool apply_exponential_envelope,
  bool positive_params,
  double* vals,
  double* ders)
{
  double scal = scal_raw;
  double rho = s_raw;
  if (positive_params) {
    scal = kLaguerrePositiveParamFloor + stable_softplus(scal_raw);
    rho = kLaguerrePositiveParamFloor + stable_softplus(s_raw);
  }

  const bool rho_is_active = rho > kLaguerreMinRho;
  rho = rho_is_active ? rho : kLaguerreMinRho;
  const double log_term = std::log1p(r / rho);
  const double u = scal * log_term;
  const double u_r = scal / (rho + r);

  const double dr = r - max_dist;
  const double cutoff_f = dr * dr;
  const double cutoff_der = 2.0 * dr;
  const double exp_factor = apply_exponential_envelope ? std::exp(-0.5 * u) : 1.0;

  double phi_prev = 0.0;
  double dphi_prev = 0.0;
  double phi_curr = basis_scaling * cutoff_f * exp_factor;
  double dphi_curr = basis_scaling * cutoff_der * exp_factor;

  if (apply_exponential_envelope) {
    dphi_curr -= 0.5 * u_r * phi_curr;
  }

  vals[0] = phi_curr;
  ders[0] = dphi_curr;

  for (int n = 0; n < rb_size - 1; ++n) {
    const double inv_np1 = 1.0 / (static_cast<double>(n) + 1.0);
    const double coeff = (2.0 * static_cast<double>(n) + 1.0 - u) * inv_np1;
    const double prev_coeff = static_cast<double>(n) * inv_np1;
    const double phi_next = coeff * phi_curr - prev_coeff * phi_prev;
    const double dphi_next =
      -u_r * inv_np1 * phi_curr + coeff * dphi_curr - prev_coeff * dphi_prev;

    vals[n + 1] = phi_next;
    ders[n + 1] = dphi_next;

    phi_prev = phi_curr;
    dphi_prev = dphi_curr;
    phi_curr = phi_next;
    dphi_curr = dphi_next;
  }
}

void chebyshev_sss_calc_host(
  int rb_size,
  double basis_scaling,
  double max_dist,
  double r,
  double scal,
  double s,
  double* vals,
  double* ders)
{
  const double x = 0.5 * scal * (r - s);
  const double ksi = std::tanh(x);
  const double der = 1.0 - ksi * ksi;
  const double mult = 0.5 * scal * der;
  const double dr = r - max_dist;
  const double cutoff_f = dr * dr;
  const double cutoff_der = 2.0 * dr;

  vals[0] = basis_scaling * cutoff_f;
  ders[0] = basis_scaling * cutoff_der;
  if (rb_size == 1) {
    return;
  }

  vals[1] = basis_scaling * ksi * cutoff_f;
  ders[1] = basis_scaling * (mult * cutoff_f + cutoff_der * ksi);
  for (int i = 2; i < rb_size; ++i) {
    vals[i] = 2.0 * ksi * vals[i - 1] - vals[i - 2];
    ders[i] = 2.0 * (mult * vals[i - 1] + ksi * ders[i - 1]) - ders[i - 2];
  }
}

void compress_active_moment_dag(SUS2HostModel& model)
{
  model.original_alpha_moments_count = model.alpha_moments_count;
  model.original_alpha_times_count = model.alpha_times_count;

  std::vector<unsigned char> needed(model.alpha_moments_count, 0);
  auto require_moment = [&](int id, const char* section) {
    if (id < 0 || id >= model.alpha_moments_count) {
      sus2_input_error(std::string("Invalid moment id in ") + section + ".");
    }
    needed[id] = 1;
  };

  // Keep all basic moments contiguous. This avoids changing the neighbor-to-basic
  // moment kernel while still allowing unused product moments to be removed.
  for (int basic = 0; basic < model.alpha_basic_count; ++basic) {
    require_moment(basic, "alpha_index_basic");
  }
  for (int id : model.alpha_moment_mapping) {
    require_moment(id, "alpha_moment_mapping");
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (int t = model.alpha_times_count - 1; t >= 0; --t) {
      const int src0 = model.alpha_times[t * 4 + 0];
      const int src1 = model.alpha_times[t * 4 + 1];
      const int dst = model.alpha_times[t * 4 + 3];
      if (src0 < 0 || src0 >= model.alpha_moments_count || src1 < 0 ||
          src1 >= model.alpha_moments_count || dst < 0 || dst >= model.alpha_moments_count) {
        sus2_input_error("Invalid moment id in alpha_index_times.");
      }
      if (!needed[dst]) {
        continue;
      }
      if (!needed[src0]) {
        needed[src0] = 1;
        changed = true;
      }
      if (!needed[src1]) {
        needed[src1] = 1;
        changed = true;
      }
    }
  }

  std::vector<int> old_to_new(model.alpha_moments_count, -1);
  int active_count = 0;
  for (int old_id = 0; old_id < model.alpha_moments_count; ++old_id) {
    if (needed[old_id]) {
      old_to_new[old_id] = active_count++;
    }
  }
  if (active_count == model.alpha_moments_count) {
    return;
  }

  for (int basic = 0; basic < model.alpha_basic_count; ++basic) {
    if (old_to_new[basic] != basic) {
      sus2_input_error("Internal SUS2 active DAG remapping expected contiguous basic moments.");
    }
  }

  std::vector<int> new_alpha_times;
  new_alpha_times.reserve(model.alpha_times.size());
  for (int t = 0; t < model.alpha_times_count; ++t) {
    const int src0 = model.alpha_times[t * 4 + 0];
    const int src1 = model.alpha_times[t * 4 + 1];
    const int mult = model.alpha_times[t * 4 + 2];
    const int dst = model.alpha_times[t * 4 + 3];
    if (!needed[dst]) {
      continue;
    }
    new_alpha_times.push_back(old_to_new[src0]);
    new_alpha_times.push_back(old_to_new[src1]);
    new_alpha_times.push_back(mult);
    new_alpha_times.push_back(old_to_new[dst]);
  }

  for (int& id : model.alpha_moment_mapping) {
    id = old_to_new[id];
  }

  model.alpha_times.swap(new_alpha_times);
  model.alpha_times_count = static_cast<int>(model.alpha_times.size() / 4);
  model.alpha_moments_count = active_count;
}

bool has_l3k3_alpha_basic_layout(const SUS2HostModel& model)
{
  if (model.radial_funcs_count != 12 || model.alpha_basic_count != 60 || model.max_rank != 3) {
    return false;
  }

  int basic = 0;
  for (int group = 0; group < 3; ++group) {
    for (int rank = 0; rank <= 3; ++rank) {
      const int mu = group * 4 + rank;
      for (int a = rank; a >= 0; --a) {
        for (int b = rank - a; b >= 0; --b) {
          const int c = rank - a - b;
          const int offset = basic * 4;
          if (model.alpha_basic[offset + 0] != mu || model.alpha_basic[offset + 1] != a ||
              model.alpha_basic[offset + 2] != b || model.alpha_basic[offset + 3] != c) {
            return false;
          }
          ++basic;
        }
      }
    }
  }
  return basic == model.alpha_basic_count;
}

bool can_pack_alpha_times_u16(const SUS2HostModel& model)
{
  if (model.alpha_times_count > kSus2MaxConstAlphaTimes || model.alpha_moments_count > 65535) {
    return false;
  }
  for (int value : model.alpha_times) {
    if (value < 0 || value > 65535) {
      return false;
    }
  }
  return true;
}

SUS2HostModel load_model(const std::string& path)
{
  const std::string text = read_text_file(path);
  SUS2HostModel model;

  const std::string version = parse_string_after(text, "version =");
  if (version != "1.1.0") {
    sus2_input_error("GPUMD SUS2_V11 currently supports only SUS2 model version = 1.1.0.");
  }

  model.scaling = parse_double_after(text, "scaling =");
  model.angular_channels = parse_int_after(text, "L =") + 1;
  model.scaling_map = parse_string_after(text, "scaling_map =");
  model.species_count = parse_int_after(text, "species_count =");
  model.radial_basis_type = parse_string_after(text, "radial_basis_type =");
  model.radial_basis_kind = radial_basis_kind_from_string(model.radial_basis_type);
  model.max_dist = parse_double_after(text, "max_dist =");
  model.rb_size = parse_int_after(text, "radial_basis_size =");
  model.radial_funcs_count = parse_int_after(text, "radial_funcs_count =");
  model.alpha_moments_count = parse_int_after(text, "alpha_moments_count =");
  model.alpha_basic_count = parse_int_after(text, "alpha_index_basic_count =");
  model.alpha_times_count = parse_int_after(text, "alpha_index_times_count =");
  model.alpha_scalar_moments = parse_int_after(text, "alpha_scalar_moments =");

  if (model.scaling_map == "K") {
    model.scaling_block_count = model.radial_funcs_count / model.angular_channels;
  } else if (model.scaling_map == "L") {
    model.scaling_block_count = model.angular_channels;
  } else if (model.scaling_map == "LK") {
    model.scaling_block_count = model.radial_funcs_count;
  } else {
    sus2_input_error("Unsupported SUS2 scaling_map: " + model.scaling_map);
  }

  model.shift_coeffs = parse_numbers<double>(extract_braced_after(text, "shift_coeffs ="));
  model.scal_coeffs = parse_numbers<double>(extract_braced_after(text, "scal_coeffs ="));
  model.alpha_basic = parse_numbers<int>(extract_braced_after(text, "alpha_index_basic ="));
  model.alpha_times = parse_numbers<int>(extract_braced_after(text, "alpha_index_times ="));
  model.alpha_moment_mapping =
    parse_numbers<int>(extract_braced_after(text, "alpha_moment_mapping ="));
  model.species_coeffs = parse_numbers<double>(extract_braced_after(text, "species_coeffs ="));
  model.moment_coeffs = parse_numbers<double>(extract_braced_after(text, "moment_coeffs ="));

  if (static_cast<int>(model.shift_coeffs.size()) != model.species_count ||
      static_cast<int>(model.species_coeffs.size()) != model.species_count ||
      static_cast<int>(model.scal_coeffs.size()) !=
        2 * model.species_count * model.species_count * model.scaling_block_count ||
      static_cast<int>(model.alpha_basic.size()) != model.alpha_basic_count * 4 ||
      static_cast<int>(model.alpha_times.size()) != model.alpha_times_count * 4 ||
      static_cast<int>(model.alpha_moment_mapping.size()) != model.alpha_scalar_moments ||
      static_cast<int>(model.moment_coeffs.size()) != model.alpha_scalar_moments) {
    sus2_input_error("Unexpected SUS2 v1.1 model dimensions while parsing coefficients.");
  }

  model.radial_coeffs.resize(model.radial_funcs_count * model.rb_size, 0.0);
  model.radial_type_coeffs.resize(model.species_count, 1.0);
  const size_t radial_start = find_required(text, "radial_coeffs");
  const size_t radial_end = find_required(text, "alpha_moments_count", radial_start);
  const auto radial_groups =
    extract_all_brace_groups(text.substr(radial_start, radial_end - radial_start));
  if (static_cast<int>(radial_groups.size()) < model.radial_funcs_count) {
    sus2_input_error("Unexpected radial_coeffs section in SUS2 v1.1 model file.");
  }
  for (int mu = 0; mu < model.radial_funcs_count; ++mu) {
    const auto values = parse_numbers<double>(radial_groups[mu]);
    if (static_cast<int>(values.size()) != model.rb_size + model.species_count) {
      sus2_input_error("Unexpected radial_coeffs row size in SUS2 v1.1 model file.");
    }
    for (int xi = 0; xi < model.rb_size; ++xi) {
      model.radial_coeffs[mu * model.rb_size + xi] = values[xi];
    }
    if (mu == 0) {
      for (int t = 0; t < model.species_count; ++t) {
        model.radial_type_coeffs[t] = values[model.rb_size + t];
      }
    }
  }

  model.mu_to_scaling_block.resize(model.radial_funcs_count);
  model.mu_to_jacobi_block.resize(model.radial_funcs_count);
  for (int mu = 0; mu < model.radial_funcs_count; ++mu) {
    if (model.scaling_map == "K") {
      model.mu_to_scaling_block[mu] = mu / model.angular_channels;
    } else if (model.scaling_map == "L") {
      model.mu_to_scaling_block[mu] = mu % model.angular_channels;
    } else {
      model.mu_to_scaling_block[mu] = mu;
    }
    const int radial_block = mu / model.angular_channels;
    if ((model.radial_basis_kind == RadialBasisKind::JacobiSSS ||
         model.radial_basis_kind == RadialBasisKind::JacobiSSSNoWeight) &&
        radial_block > kJacobiMaxIndexedBlock) {
      sus2_input_error("RBJacobi_sss supports at most six Jacobi blocks.");
    }
    model.mu_to_jacobi_block[mu] = radial_block;
  }

  model.max_rank = 0;
  for (int basic = 0; basic < model.alpha_basic_count; ++basic) {
    const int rank = model.alpha_basic[basic * 4 + 1] + model.alpha_basic[basic * 4 + 2] +
                     model.alpha_basic[basic * 4 + 3];
    model.max_rank = std::max(model.max_rank, rank);
  }
  compress_active_moment_dag(model);
  return model;
}

void build_lut(const SUS2HostModel& model, int lut_size, double lut_inv_dr, std::vector<double>& vals, std::vector<double>& ders)
{
  vals.assign(
    static_cast<size_t>(model.species_count) * model.species_count * lut_size *
      model.radial_funcs_count,
    0.0);
  ders.assign(vals.size(), 0.0);

  std::vector<double> rb_vals(model.rb_size);
  std::vector<double> rb_ders(model.rb_size);
  for (int zi = 0; zi < model.species_count; ++zi) {
    for (int zj = 0; zj < model.species_count; ++zj) {
      const int pair = zi * model.species_count + zj;
      for (int idx = 0; idx < lut_size; ++idx) {
        const double r = std::min(static_cast<double>(idx) / lut_inv_dr, model.max_dist);
        for (int mu = 0; mu < model.radial_funcs_count; ++mu) {
          const int scaling_block = model.mu_to_scaling_block[mu];
          const int shift = model.species_count * zi + zj;
          const int scal_offset =
            2 * scaling_block * model.species_count * model.species_count + shift;
          const double scal = model.scal_coeffs[scal_offset];
          const double s =
            model.scal_coeffs[scal_offset + model.species_count * model.species_count];
          if (model.radial_basis_kind == RadialBasisKind::ChebyshevSSS) {
            chebyshev_sss_calc_host(
              model.rb_size, 1.0, model.max_dist, r, scal, s, rb_vals.data(), rb_ders.data());
          } else if (
            model.radial_basis_kind == RadialBasisKind::JacobiSSS ||
            model.radial_basis_kind == RadialBasisKind::JacobiSSSNoWeight) {
            jacobi_sss_calc_host(
              model.rb_size,
              1.0,
              model.max_dist,
              r,
              scal,
              s,
              model.mu_to_jacobi_block[mu],
              model.radial_basis_kind == RadialBasisKind::JacobiSSS,
              rb_vals.data(),
              rb_ders.data());
          } else {
            laguerre_log1p_calc_host(
              model.rb_size,
              1.0,
              model.max_dist,
              r,
              scal,
              s,
              model.radial_basis_kind == RadialBasisKind::LaguerreLog1p ||
                model.radial_basis_kind == RadialBasisKind::LaguerreLog1pPositive,
              model.radial_basis_kind == RadialBasisKind::LaguerreLog1pPositive,
              rb_vals.data(),
              rb_ders.data());
          }
          double acc_val = 0.0;
          double acc_der = 0.0;
          for (int xi = 0; xi < model.rb_size; ++xi) {
            const double coeff = model.radial_coeffs[mu * model.rb_size + xi];
            acc_val += coeff * rb_vals[xi];
            acc_der += coeff * rb_ders[xi];
          }
          const double type_scale = model.radial_type_coeffs[zi] * model.radial_type_coeffs[zj];
          const size_t out =
            ((static_cast<size_t>(pair) * lut_size + idx) * model.radial_funcs_count) + mu;
          vals[out] = acc_val * model.scaling * type_scale;
          ders[out] = acc_der * model.scaling * type_scale;
        }
      }
    }
  }
}

bool starts_with(const std::string& text, const std::string& prefix)
{
  return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

int parse_lut_span(
  const SUS2HostModel& model,
  int num_potential_options,
  const char** potential_options)
{
  constexpr double default_lut_dr = 1.0e-4;
  int lut_span = static_cast<int>(std::ceil(model.max_dist / default_lut_dr));
  const char* env_span = std::getenv("SUS2_GPUMD_LUT_SPAN");
  if (env_span != nullptr && std::atoi(env_span) > 0) {
    lut_span = std::atoi(env_span);
  }
  const char* env_dr = std::getenv("SUS2_GPUMD_LUT_DR");
  if (env_dr != nullptr && std::atof(env_dr) > 0.0) {
    lut_span = static_cast<int>(std::ceil(model.max_dist / std::atof(env_dr)));
  }

  const int option_begin = std::min(num_potential_options, model.species_count);
  for (int i = option_begin; i < num_potential_options; ++i) {
    const std::string option = potential_options[i] == nullptr ? "" : potential_options[i];
    if (starts_with(option, "sus2_lut_span=") || starts_with(option, "lut_span=")) {
      const size_t eq = option.find('=');
      lut_span = std::stoi(option.substr(eq + 1));
    } else if (starts_with(option, "sus2_lut_dr=") || starts_with(option, "lut_dr=")) {
      const size_t eq = option.find('=');
      const double dr = std::stod(option.substr(eq + 1));
      if (dr <= 0.0) {
        sus2_input_error("SUS2 GPUMD lut_dr must be positive.");
      }
      lut_span = static_cast<int>(std::ceil(model.max_dist / dr));
    }
  }

  if (lut_span < 8) {
    sus2_input_error("SUS2 GPUMD LUT span is too small; use at least 8 intervals.");
  }
  if (lut_span > 5000000) {
    sus2_input_error("SUS2 GPUMD LUT span is too large; refusing more than 5000000 intervals.");
  }
  return lut_span;
}

bool parse_bool_value(const std::string& value, const std::string& option_name)
{
  std::string lower = value;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (lower == "1" || lower == "true" || lower == "yes" || lower == "on") {
    return true;
  }
  if (lower == "0" || lower == "false" || lower == "no" || lower == "off") {
    return false;
  }
  sus2_input_error("SUS2 GPUMD boolean option " + option_name + " must be one of 0/1/true/false/on/off.");
}

bool parse_float_moment_grads(
  const SUS2HostModel& model,
  int num_potential_options,
  const char** potential_options)
{
  bool use_float = false;
  const char* env = std::getenv("SUS2_GPUMD_GRAD_FLOAT");
  if (env != nullptr) {
    use_float = parse_bool_value(env, "SUS2_GPUMD_GRAD_FLOAT");
  }

  const int option_begin = std::min(num_potential_options, model.species_count);
  for (int i = option_begin; i < num_potential_options; ++i) {
    const std::string option = potential_options[i] == nullptr ? "" : potential_options[i];
    if (starts_with(option, "sus2_grad_float=") || starts_with(option, "grad_float=") ||
        starts_with(option, "sus2_moment_grad_float=")) {
      const size_t eq = option.find('=');
      use_float = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_float;
}

int periodic_image_range(int pbc, double cutoff, double thickness)
{
  if (pbc != 1) {
    return 0;
  }
  if (thickness <= 0.0) {
    return 1;
  }
  return std::max(1, static_cast<int>(std::ceil(cutoff / thickness)));
}

static __global__ void gpu_count_neighbors_images_on2(
  int N,
  const Box box,
  int sx_min,
  int sx_max,
  int sy_min,
  int sy_max,
  int sz_min,
  int sz_max,
  double cutoff_square,
  const double* x,
  const double* y,
  const double* z,
  int* counts)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }
  const double xi = x[i];
  const double yi = y[i];
  const double zi = z[i];
  int count = 0;
  for (int j = 0; j < N; ++j) {
    const double dx0 = x[j] - xi;
    const double dy0 = y[j] - yi;
    const double dz0 = z[j] - zi;
    for (int sx = sx_min; sx <= sx_max; ++sx) {
      for (int sy = sy_min; sy <= sy_max; ++sy) {
        for (int sz = sz_min; sz <= sz_max; ++sz) {
          if (j == i && sx == 0 && sy == 0 && sz == 0) {
            continue;
          }
          const double dx = dx0 + sx * box.cpu_h[0] + sy * box.cpu_h[1] + sz * box.cpu_h[2];
          const double dy = dy0 + sx * box.cpu_h[3] + sy * box.cpu_h[4] + sz * box.cpu_h[5];
          const double dz = dz0 + sx * box.cpu_h[6] + sy * box.cpu_h[7] + sz * box.cpu_h[8];
          const double d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < cutoff_square) {
            ++count;
          }
        }
      }
    }
  }
  counts[i] = count;
}

static __global__ void gpu_fill_neighbors_images_on2(
  int N,
  const Box box,
  int sx_min,
  int sx_max,
  int sy_min,
  int sy_max,
  int sz_min,
  int sz_max,
  double cutoff_square,
  const double* x,
  const double* y,
  const double* z,
  int* neighbor_atoms,
  double* neighbor_dx,
  double* neighbor_dy,
  double* neighbor_dz)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }
  const double xi = x[i];
  const double yi = y[i];
  const double zi = z[i];
  int count = 0;
  for (int j = 0; j < N; ++j) {
    const double dx0 = x[j] - xi;
    const double dy0 = y[j] - yi;
    const double dz0 = z[j] - zi;
    for (int sx = sx_min; sx <= sx_max; ++sx) {
      for (int sy = sy_min; sy <= sy_max; ++sy) {
        for (int sz = sz_min; sz <= sz_max; ++sz) {
          if (j == i && sx == 0 && sy == 0 && sz == 0) {
            continue;
          }
          const double dx = dx0 + sx * box.cpu_h[0] + sy * box.cpu_h[1] + sz * box.cpu_h[2];
          const double dy = dy0 + sx * box.cpu_h[3] + sy * box.cpu_h[4] + sz * box.cpu_h[5];
          const double dz = dz0 + sx * box.cpu_h[6] + sy * box.cpu_h[7] + sz * box.cpu_h[8];
          const double d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < cutoff_square) {
            const size_t out = static_cast<size_t>(count) * N + i;
            neighbor_atoms[out] = j;
            neighbor_dx[out] = dx;
            neighbor_dy[out] = dy;
            neighbor_dz[out] = dz;
            ++count;
          }
        }
      }
    }
  }
}

__device__ __forceinline__ void interp_radial_vals_ders(
  const SUS2DeviceModel& model,
  int pair,
  double r,
  double* vals,
  double* ders)
{
  const double scaled_r = r * model.lut_inv_dr;
  int lut_idx = static_cast<int>(floor(scaled_r));
  if (lut_idx < 0) {
    lut_idx = 0;
  }
  if (lut_idx > model.lut_size - 2) {
    lut_idx = model.lut_size - 2;
  }
  const int lut_next = lut_idx + 1;
  const double t = scaled_r - static_cast<double>(lut_idx);
  const size_t base0 =
    (static_cast<size_t>(pair) * model.lut_size + lut_idx) * model.radial_funcs_count;
  const size_t base1 =
    (static_cast<size_t>(pair) * model.lut_size + lut_next) * model.radial_funcs_count;
  for (int mu = 0; mu < model.radial_funcs_count; ++mu) {
    const double v0 = static_cast<double>(model.lut_vals[base0 + mu]);
    const double v1 = static_cast<double>(model.lut_vals[base1 + mu]);
    vals[mu] = v0 + t * (v1 - v0);
    if (ders != nullptr) {
      const double d0 = static_cast<double>(model.lut_ders[base0 + mu]);
      const double d1 = static_cast<double>(model.lut_ders[base1 + mu]);
      ders[mu] = d0 + t * (d1 - d0);
    }
  }
}

__device__ __forceinline__ void load_sus2_edge_displacement(
  bool use_cached_displacements,
  int N,
  Box box,
  size_t edge,
  int i,
  int j,
  const double* neighbor_dx,
  const double* neighbor_dy,
  const double* neighbor_dz,
  const double* x,
  const double* y,
  const double* z,
  double& dx,
  double& dy,
  double& dz)
{
  if (use_cached_displacements) {
    dx = neighbor_dx[edge];
    dy = neighbor_dy[edge];
    dz = neighbor_dz[edge];
    return;
  }

  dx = x[j] - x[i];
  dy = y[j] - y[i];
  dz = z[j] - z[i];
  apply_mic(box, dx, dy, dz);
}

__device__ __forceinline__ void add_l3k3_basic_moments(
  int N,
  const SUS2DeviceModel& model,
  int atom,
  int pair,
  double dx,
  double dy,
  double dz,
  double r,
  double* moments)
{
  double mu_val[32];
  interp_radial_vals_ders(model, pair, r, mu_val, nullptr);

  const double inv_r = 1.0 / r;
  const double inv_r2 = inv_r * inv_r;
  const double inv_r3 = inv_r2 * inv_r;
  const double x2 = dx * dx;
  const double y2 = dy * dy;
  const double z2 = dz * dz;
  const double xy = dx * dy;
  const double xz = dx * dz;
  const double yz = dy * dz;

#define SUS2_ADD_L3K3_MOMENT(BASIC, SCALE, GEOM) \
  moments[static_cast<size_t>(BASIC) * N + atom] += (SCALE) * (GEOM)

  for (int group = 0; group < 3; ++group) {
    const int base = group * 20;
    const int mu = group * 4;
    const double s0 = mu_val[mu + 0];
    const double s1 = mu_val[mu + 1] * inv_r;
    const double s2 = mu_val[mu + 2] * inv_r2;
    const double s3 = mu_val[mu + 3] * inv_r3;

    SUS2_ADD_L3K3_MOMENT(base + 0, s0, 1.0);
    SUS2_ADD_L3K3_MOMENT(base + 1, s1, dx);
    SUS2_ADD_L3K3_MOMENT(base + 2, s1, dy);
    SUS2_ADD_L3K3_MOMENT(base + 3, s1, dz);
    SUS2_ADD_L3K3_MOMENT(base + 4, s2, x2);
    SUS2_ADD_L3K3_MOMENT(base + 5, s2, xy);
    SUS2_ADD_L3K3_MOMENT(base + 6, s2, xz);
    SUS2_ADD_L3K3_MOMENT(base + 7, s2, y2);
    SUS2_ADD_L3K3_MOMENT(base + 8, s2, yz);
    SUS2_ADD_L3K3_MOMENT(base + 9, s2, z2);
    SUS2_ADD_L3K3_MOMENT(base + 10, s3, x2 * dx);
    SUS2_ADD_L3K3_MOMENT(base + 11, s3, x2 * dy);
    SUS2_ADD_L3K3_MOMENT(base + 12, s3, x2 * dz);
    SUS2_ADD_L3K3_MOMENT(base + 13, s3, dx * y2);
    SUS2_ADD_L3K3_MOMENT(base + 14, s3, xy * dz);
    SUS2_ADD_L3K3_MOMENT(base + 15, s3, dx * z2);
    SUS2_ADD_L3K3_MOMENT(base + 16, s3, y2 * dy);
    SUS2_ADD_L3K3_MOMENT(base + 17, s3, y2 * dz);
    SUS2_ADD_L3K3_MOMENT(base + 18, s3, dy * z2);
    SUS2_ADD_L3K3_MOMENT(base + 19, s3, z2 * dz);
  }

#undef SUS2_ADD_L3K3_MOMENT
}

template <typename GradT>
__device__ __forceinline__ double load_sus2_grad(const GradT* grads, int N, int moment, int atom)
{
  return static_cast<double>(grads[static_cast<size_t>(moment) * N + atom]);
}

template <typename GradT>
__device__ __forceinline__ void add_sus2_grad(GradT* grads, size_t index, double value)
{
  grads[index] = static_cast<GradT>(static_cast<double>(grads[index]) + value);
}

template <typename GradT>
__device__ __forceinline__ void compute_sus2_edge_derivative_l3k3(
  int N,
  const SUS2DeviceModel& model,
  int center_atom,
  int center_type,
  int neighbor_type,
  double dx,
  double dy,
  double dz,
  double r,
  const GradT* grads,
  double& dEx,
  double& dEy,
  double& dEz)
{
  const int pair = center_type * model.species_count + neighbor_type;
  const double center_coeff = model.species_coeffs[center_type];

  double mu_val[32];
  double mu_der[32];
  interp_radial_vals_ders(model, pair, r, mu_val, mu_der);

  const double inv_r = 1.0 / r;
  const double inv_r2 = inv_r * inv_r;
  const double inv_r3 = inv_r2 * inv_r;
  const double x2 = dx * dx;
  const double y2 = dy * dy;
  const double z2 = dz * dz;
  const double xy = dx * dy;
  const double xz = dx * dz;
  const double yz = dy * dz;

  dEx = 0.0;
  dEy = 0.0;
  dEz = 0.0;

#define SUS2_ACCUM_L3K3_DERIV(BASIC, GEOM, DGX, DGY, DGZ, INV_SCALED, RAD_COMMON) \
  do { \
    const double basic_grad = \
      load_sus2_grad(grads, N, BASIC, center_atom) * center_coeff; \
    const double common = (GEOM) * (RAD_COMMON); \
    dEx += basic_grad * (common * dx + (INV_SCALED) * (DGX)); \
    dEy += basic_grad * (common * dy + (INV_SCALED) * (DGY)); \
    dEz += basic_grad * (common * dz + (INV_SCALED) * (DGZ)); \
  } while (0)

  for (int group = 0; group < 3; ++group) {
    const int base = group * 20;
    const int mu = group * 4;

    const double inv0 = mu_val[mu + 0];
    const double rc0 = mu_der[mu + 0] * inv_r;
    SUS2_ACCUM_L3K3_DERIV(base + 0, 1.0, 0.0, 0.0, 0.0, inv0, rc0);

    const double inv1 = mu_val[mu + 1] * inv_r;
    const double rc1 = (mu_der[mu + 1] * inv_r - inv1 * inv_r) * inv_r;
    SUS2_ACCUM_L3K3_DERIV(base + 1, dx, 1.0, 0.0, 0.0, inv1, rc1);
    SUS2_ACCUM_L3K3_DERIV(base + 2, dy, 0.0, 1.0, 0.0, inv1, rc1);
    SUS2_ACCUM_L3K3_DERIV(base + 3, dz, 0.0, 0.0, 1.0, inv1, rc1);

    const double inv2 = mu_val[mu + 2] * inv_r2;
    const double rc2 = (mu_der[mu + 2] * inv_r2 - 2.0 * inv2 * inv_r) * inv_r;
    SUS2_ACCUM_L3K3_DERIV(base + 4, x2, 2.0 * dx, 0.0, 0.0, inv2, rc2);
    SUS2_ACCUM_L3K3_DERIV(base + 5, xy, dy, dx, 0.0, inv2, rc2);
    SUS2_ACCUM_L3K3_DERIV(base + 6, xz, dz, 0.0, dx, inv2, rc2);
    SUS2_ACCUM_L3K3_DERIV(base + 7, y2, 0.0, 2.0 * dy, 0.0, inv2, rc2);
    SUS2_ACCUM_L3K3_DERIV(base + 8, yz, 0.0, dz, dy, inv2, rc2);
    SUS2_ACCUM_L3K3_DERIV(base + 9, z2, 0.0, 0.0, 2.0 * dz, inv2, rc2);

    const double inv3 = mu_val[mu + 3] * inv_r3;
    const double rc3 = (mu_der[mu + 3] * inv_r3 - 3.0 * inv3 * inv_r) * inv_r;
    SUS2_ACCUM_L3K3_DERIV(base + 10, x2 * dx, 3.0 * x2, 0.0, 0.0, inv3, rc3);
    SUS2_ACCUM_L3K3_DERIV(base + 11, x2 * dy, 2.0 * xy, x2, 0.0, inv3, rc3);
    SUS2_ACCUM_L3K3_DERIV(base + 12, x2 * dz, 2.0 * xz, 0.0, x2, inv3, rc3);
    SUS2_ACCUM_L3K3_DERIV(base + 13, dx * y2, y2, 2.0 * xy, 0.0, inv3, rc3);
    SUS2_ACCUM_L3K3_DERIV(base + 14, xy * dz, yz, xz, xy, inv3, rc3);
    SUS2_ACCUM_L3K3_DERIV(base + 15, dx * z2, z2, 0.0, 2.0 * xz, inv3, rc3);
    SUS2_ACCUM_L3K3_DERIV(base + 16, y2 * dy, 0.0, 3.0 * y2, 0.0, inv3, rc3);
    SUS2_ACCUM_L3K3_DERIV(base + 17, y2 * dz, 0.0, 2.0 * yz, y2, inv3, rc3);
    SUS2_ACCUM_L3K3_DERIV(base + 18, dy * z2, 0.0, z2, 2.0 * yz, inv3, rc3);
    SUS2_ACCUM_L3K3_DERIV(base + 19, z2 * dz, 0.0, 0.0, 3.0 * z2, inv3, rc3);
  }

#undef SUS2_ACCUM_L3K3_DERIV
}

template <typename GradT>
__device__ __forceinline__ void compute_sus2_edge_derivative(
  int N,
  const SUS2DeviceModel& model,
  int center_atom,
  int center_type,
  int neighbor_type,
  double dx,
  double dy,
  double dz,
  double r,
  const GradT* grads,
  double& dEx,
  double& dEy,
  double& dEz)
{
  if (model.use_l3k3_basic_fastpath) {
    compute_sus2_edge_derivative_l3k3<GradT>(
      N, model, center_atom, center_type, neighbor_type, dx, dy, dz, r, grads, dEx, dEy, dEz);
    return;
  }

  const int pair = center_type * model.species_count + neighbor_type;
  const double center_coeff = model.species_coeffs[center_type];

  double mu_val[32];
  double mu_der[32];
  interp_radial_vals_ders(model, pair, r, mu_val, mu_der);

  double dist_pow[8];
  double x_pow[8];
  double y_pow[8];
  double z_pow[8];
  dist_pow[0] = 1.0;
  x_pow[0] = 1.0;
  y_pow[0] = 1.0;
  z_pow[0] = 1.0;
  for (int k = 1; k <= model.max_rank; ++k) {
    dist_pow[k] = dist_pow[k - 1] * r;
    x_pow[k] = x_pow[k - 1] * dx;
    y_pow[k] = y_pow[k - 1] * dy;
    z_pow[k] = z_pow[k - 1] * dz;
  }

  dEx = 0.0;
  dEy = 0.0;
  dEz = 0.0;
  for (int basic = 0; basic < model.alpha_basic_count; ++basic) {
    const int mu = model.alpha_basic[basic * 4 + 0];
    const int a = model.alpha_basic[basic * 4 + 1];
    const int b = model.alpha_basic[basic * 4 + 2];
    const int c = model.alpha_basic[basic * 4 + 3];
    const int rank = a + b + c;
    const double inv_dist_pow = mu_val[mu] / dist_pow[rank];
    const double geom = x_pow[a] * y_pow[b] * z_pow[c];
    const double radial_der =
      mu_der[mu] / dist_pow[rank] - static_cast<double>(rank) * inv_dist_pow / r;
    const double common = geom * radial_der / r;
    double jac_x = common * dx;
    double jac_y = common * dy;
    double jac_z = common * dz;
    if (a != 0) {
      jac_x += inv_dist_pow * static_cast<double>(a) * x_pow[a - 1] * y_pow[b] * z_pow[c];
    }
    if (b != 0) {
      jac_y += inv_dist_pow * static_cast<double>(b) * x_pow[a] * y_pow[b - 1] * z_pow[c];
    }
    if (c != 0) {
      jac_z += inv_dist_pow * static_cast<double>(c) * x_pow[a] * y_pow[b] * z_pow[c - 1];
    }
    const double basic_grad = load_sus2_grad(grads, N, basic, center_atom) * center_coeff;
    dEx += basic_grad * jac_x;
    dEy += basic_grad * jac_y;
    dEz += basic_grad * jac_z;
  }
}

static __global__ void gpu_compute_basic_moments(
  int N,
  Box box,
  double cutoff_square,
  bool use_cached_displacements,
  SUS2DeviceModel model,
  const int* type,
  const int* neighbor_count,
  const int* neighbor_atoms,
  const double* neighbor_dx,
  const double* neighbor_dy,
  const double* neighbor_dz,
  const double* x,
  const double* y,
  const double* z,
  double* moments)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  const int type_i = type[i];
  const int count = neighbor_count[i];

  for (int nbr = 0; nbr < count; ++nbr) {
    const size_t edge = static_cast<size_t>(nbr) * N + i;
    const int j = neighbor_atoms[edge];
    double dx;
    double dy;
    double dz;
    load_sus2_edge_displacement(
      use_cached_displacements, N, box, edge, i, j, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, dx, dy, dz);
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= cutoff_square) {
      continue;
    }
    const double r = sqrt(r2);
    const int type_j = type[j];
    const int pair = type_i * model.species_count + type_j;

    if (model.use_l3k3_basic_fastpath) {
      add_l3k3_basic_moments(N, model, i, pair, dx, dy, dz, r, moments);
      continue;
    }

    double mu_val[32];
    interp_radial_vals_ders(model, pair, r, mu_val, nullptr);

    double dist_pow[8];
    double x_pow[8];
    double y_pow[8];
    double z_pow[8];
    dist_pow[0] = 1.0;
    x_pow[0] = 1.0;
    y_pow[0] = 1.0;
    z_pow[0] = 1.0;
    for (int k = 1; k <= model.max_rank; ++k) {
      dist_pow[k] = dist_pow[k - 1] * r;
      x_pow[k] = x_pow[k - 1] * dx;
      y_pow[k] = y_pow[k - 1] * dy;
      z_pow[k] = z_pow[k - 1] * dz;
    }

    for (int basic = 0; basic < model.alpha_basic_count; ++basic) {
      const int mu = model.alpha_basic[basic * 4 + 0];
      const int a = model.alpha_basic[basic * 4 + 1];
      const int b = model.alpha_basic[basic * 4 + 2];
      const int c = model.alpha_basic[basic * 4 + 3];
      const int rank = a + b + c;
      const double geom = x_pow[a] * y_pow[b] * z_pow[c];
      moments[static_cast<size_t>(basic) * N + i] += (mu_val[mu] / dist_pow[rank]) * geom;
    }
  }
}

static __global__ void gpu_compute_basic_moments_l3k3_accum(
  int N,
  Box box,
  double cutoff_square,
  bool use_cached_displacements,
  SUS2DeviceModel model,
  const int* type,
  const int* neighbor_count,
  const int* neighbor_atoms,
  const double* neighbor_dx,
  const double* neighbor_dy,
  const double* neighbor_dz,
  const double* x,
  const double* y,
  const double* z,
  double* moments)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  double acc[60];
#pragma unroll
  for (int k = 0; k < 60; ++k) {
    acc[k] = 0.0;
  }

  const int type_i = type[i];
  const int count = neighbor_count[i];

  for (int nbr = 0; nbr < count; ++nbr) {
    const size_t edge = static_cast<size_t>(nbr) * N + i;
    const int j = neighbor_atoms[edge];
    double dx;
    double dy;
    double dz;
    load_sus2_edge_displacement(
      use_cached_displacements, N, box, edge, i, j, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, dx, dy, dz);
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= cutoff_square) {
      continue;
    }
    const double r = sqrt(r2);
    const int pair = type_i * model.species_count + type[j];

    double mu_val[32];
    interp_radial_vals_ders(model, pair, r, mu_val, nullptr);

    const double inv_r = 1.0 / r;
    const double inv_r2 = inv_r * inv_r;
    const double inv_r3 = inv_r2 * inv_r;
    const double x2 = dx * dx;
    const double y2 = dy * dy;
    const double z2 = dz * dz;
    const double xy = dx * dy;
    const double xz = dx * dz;
    const double yz = dy * dz;

#pragma unroll
    for (int group = 0; group < 3; ++group) {
      const int base = group * 20;
      const int mu = group * 4;
      const double s0 = mu_val[mu + 0];
      const double s1 = mu_val[mu + 1] * inv_r;
      const double s2 = mu_val[mu + 2] * inv_r2;
      const double s3 = mu_val[mu + 3] * inv_r3;

      acc[base + 0] += s0;
      acc[base + 1] += s1 * dx;
      acc[base + 2] += s1 * dy;
      acc[base + 3] += s1 * dz;
      acc[base + 4] += s2 * x2;
      acc[base + 5] += s2 * xy;
      acc[base + 6] += s2 * xz;
      acc[base + 7] += s2 * y2;
      acc[base + 8] += s2 * yz;
      acc[base + 9] += s2 * z2;
      acc[base + 10] += s3 * x2 * dx;
      acc[base + 11] += s3 * x2 * dy;
      acc[base + 12] += s3 * x2 * dz;
      acc[base + 13] += s3 * dx * y2;
      acc[base + 14] += s3 * xy * dz;
      acc[base + 15] += s3 * dx * z2;
      acc[base + 16] += s3 * y2 * dy;
      acc[base + 17] += s3 * y2 * dz;
      acc[base + 18] += s3 * dy * z2;
      acc[base + 19] += s3 * z2 * dz;
    }
  }

#pragma unroll
  for (int basic = 0; basic < 60; ++basic) {
    moments[static_cast<size_t>(basic) * N + i] = acc[basic];
  }
}

static __global__ void gpu_forward_times(int N, SUS2DeviceModel model, double* moments)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }
  for (int t = 0; t < model.alpha_times_count; ++t) {
    int src0;
    int src1;
    int mult;
    int dst;
    if (model.use_const_alpha_times) {
      const int offset = t * 4;
      src0 = static_cast<int>(c_sus2_alpha_times_u16[offset + 0]);
      src1 = static_cast<int>(c_sus2_alpha_times_u16[offset + 1]);
      mult = static_cast<int>(c_sus2_alpha_times_u16[offset + 2]);
      dst = static_cast<int>(c_sus2_alpha_times_u16[offset + 3]);
    } else {
      src0 = model.alpha_times[t * 4 + 0];
      src1 = model.alpha_times[t * 4 + 1];
      mult = model.alpha_times[t * 4 + 2];
      dst = model.alpha_times[t * 4 + 3];
    }
    moments[static_cast<size_t>(dst) * N + i] +=
      static_cast<double>(mult) * moments[static_cast<size_t>(src0) * N + i] *
      moments[static_cast<size_t>(src1) * N + i];
  }
}

static __global__ void gpu_forward_times_const_u16(int N, SUS2DeviceModel model, double* moments)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }
  for (int t = 0; t < model.alpha_times_count; ++t) {
    const int offset = t * 4;
    const int src0 = static_cast<int>(c_sus2_alpha_times_u16[offset + 0]);
    const int src1 = static_cast<int>(c_sus2_alpha_times_u16[offset + 1]);
    const int mult = static_cast<int>(c_sus2_alpha_times_u16[offset + 2]);
    const int dst = static_cast<int>(c_sus2_alpha_times_u16[offset + 3]);
    moments[static_cast<size_t>(dst) * N + i] +=
      static_cast<double>(mult) * moments[static_cast<size_t>(src0) * N + i] *
      moments[static_cast<size_t>(src1) * N + i];
  }
}

template <typename GradT>
static __global__ void gpu_site_energy_init_grad(
  int N,
  SUS2DeviceModel model,
  const int* type,
  const double* moments,
  GradT* grads,
  double* potential)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }
  const int type_i = type[i];
  double site_energy =
    model.shift_coeffs[type_i] + model.species_coeffs[type_i];
  for (int idx = 0; idx < model.alpha_scalar_moments; ++idx) {
    const int moment_id = model.alpha_moment_mapping[idx];
    const double coeff = model.moment_coeffs[idx];
    site_energy += coeff * moments[static_cast<size_t>(moment_id) * N + i] *
                   model.species_coeffs[type_i];
    add_sus2_grad(grads, static_cast<size_t>(moment_id) * N + i, coeff);
  }
  potential[i] += site_energy;
}

template <typename GradT>
static __global__ void gpu_backward_times(int N, SUS2DeviceModel model, const double* moments, GradT* grads)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }
  for (int t = model.alpha_times_count - 1; t >= 0; --t) {
    int src0;
    int src1;
    int mult;
    int dst;
    if (model.use_const_alpha_times) {
      const int offset = t * 4;
      src0 = static_cast<int>(c_sus2_alpha_times_u16[offset + 0]);
      src1 = static_cast<int>(c_sus2_alpha_times_u16[offset + 1]);
      mult = static_cast<int>(c_sus2_alpha_times_u16[offset + 2]);
      dst = static_cast<int>(c_sus2_alpha_times_u16[offset + 3]);
    } else {
      src0 = model.alpha_times[t * 4 + 0];
      src1 = model.alpha_times[t * 4 + 1];
      mult = model.alpha_times[t * 4 + 2];
      dst = model.alpha_times[t * 4 + 3];
    }
    const double gdst = load_sus2_grad(grads, N, dst, i) * static_cast<double>(mult);
    add_sus2_grad(grads, static_cast<size_t>(src1) * N + i, gdst * moments[static_cast<size_t>(src0) * N + i]);
    add_sus2_grad(grads, static_cast<size_t>(src0) * N + i, gdst * moments[static_cast<size_t>(src1) * N + i]);
  }
}

template <typename GradT>
static __global__ void gpu_backward_times_const_u16(
  int N,
  SUS2DeviceModel model,
  const double* moments,
  GradT* grads)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }
  for (int t = model.alpha_times_count - 1; t >= 0; --t) {
    const int offset = t * 4;
    const int src0 = static_cast<int>(c_sus2_alpha_times_u16[offset + 0]);
    const int src1 = static_cast<int>(c_sus2_alpha_times_u16[offset + 1]);
    const int mult = static_cast<int>(c_sus2_alpha_times_u16[offset + 2]);
    const int dst = static_cast<int>(c_sus2_alpha_times_u16[offset + 3]);
    const double gdst = load_sus2_grad(grads, N, dst, i) * static_cast<double>(mult);
    add_sus2_grad(grads, static_cast<size_t>(src1) * N + i, gdst * moments[static_cast<size_t>(src0) * N + i]);
    add_sus2_grad(grads, static_cast<size_t>(src0) * N + i, gdst * moments[static_cast<size_t>(src1) * N + i]);
  }
}

template <typename GradT>
static __global__ void gpu_compute_forces(
  int N,
  Box box,
  double cutoff_square,
  bool use_cached_displacements,
  SUS2DeviceModel model,
  const int* type,
  const int* neighbor_count,
  const int* neighbor_atoms,
  const double* neighbor_dx,
  const double* neighbor_dy,
  const double* neighbor_dz,
  const double* x,
  const double* y,
  const double* z,
  const GradT* grads,
  float* force_tmp,
  float* virial_tmp)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  const int type_i = type[i];
  const int count = neighbor_count[i];

  double fx_self = 0.0;
  double fy_self = 0.0;
  double fz_self = 0.0;
  double s_xx = 0.0;
  double s_yy = 0.0;
  double s_zz = 0.0;
  double s_xy = 0.0;
  double s_xz = 0.0;
  double s_yz = 0.0;
  double s_yx = 0.0;
  double s_zx = 0.0;
  double s_zy = 0.0;

  for (int nbr = 0; nbr < count; ++nbr) {
    const size_t edge = static_cast<size_t>(nbr) * N + i;
    const int j = neighbor_atoms[edge];
    double dx;
    double dy;
    double dz;
    load_sus2_edge_displacement(
      use_cached_displacements, N, box, edge, i, j, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, dx, dy, dz);
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= cutoff_square) {
      continue;
    }
    const double r = sqrt(r2);
    const int type_j = type[j];

    double dEx = 0.0;
    double dEy = 0.0;
    double dEz = 0.0;
    compute_sus2_edge_derivative<GradT>(N, model, i, type_i, type_j, dx, dy, dz, r, grads, dEx, dEy, dEz);

    fx_self += dEx;
    fy_self += dEy;
    fz_self += dEz;

    atomicAdd(force_tmp + j, static_cast<float>(-dEx));
    atomicAdd(force_tmp + j + N, static_cast<float>(-dEy));
    atomicAdd(force_tmp + j + 2 * N, static_cast<float>(-dEz));

    s_xx -= dEx * dx;
    s_yy -= dEy * dy;
    s_zz -= dEz * dz;
    s_xy -= dEx * dy;
    s_xz -= dEx * dz;
    s_yz -= dEy * dz;
    s_yx -= dEy * dx;
    s_zx -= dEz * dx;
    s_zy -= dEz * dy;
  }

  atomicAdd(force_tmp + i, static_cast<float>(fx_self));
  atomicAdd(force_tmp + i + N, static_cast<float>(fy_self));
  atomicAdd(force_tmp + i + 2 * N, static_cast<float>(fz_self));

  virial_tmp[i + 0 * N] += static_cast<float>(s_xx);
  virial_tmp[i + 1 * N] += static_cast<float>(s_yy);
  virial_tmp[i + 2 * N] += static_cast<float>(s_zz);
  virial_tmp[i + 3 * N] += static_cast<float>(s_xy);
  virial_tmp[i + 4 * N] += static_cast<float>(s_xz);
  virial_tmp[i + 5 * N] += static_cast<float>(s_yz);
  virial_tmp[i + 6 * N] += static_cast<float>(s_yx);
  virial_tmp[i + 7 * N] += static_cast<float>(s_zx);
  virial_tmp[i + 8 * N] += static_cast<float>(s_zy);
}

template <typename GradT>
static __global__ void gpu_compute_forces_pairwise_no_atomic(
  int N,
  Box box,
  double cutoff_square,
  SUS2DeviceModel model,
  const int* type,
  const int* neighbor_count,
  const int* neighbor_atoms,
  const double* x,
  const double* y,
  const double* z,
  const GradT* grads,
  float* force_tmp,
  float* virial_tmp)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  const int type_i = type[i];
  const int count = neighbor_count[i];

  double fx = 0.0;
  double fy = 0.0;
  double fz = 0.0;
  double s_xx = 0.0;
  double s_yy = 0.0;
  double s_zz = 0.0;
  double s_xy = 0.0;
  double s_xz = 0.0;
  double s_yz = 0.0;
  double s_yx = 0.0;
  double s_zx = 0.0;
  double s_zy = 0.0;

  for (int nbr = 0; nbr < count; ++nbr) {
    const size_t edge = static_cast<size_t>(nbr) * N + i;
    const int j = neighbor_atoms[edge];
    double dx = x[j] - x[i];
    double dy = y[j] - y[i];
    double dz = z[j] - z[i];
    apply_mic(box, dx, dy, dz);
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= cutoff_square) {
      continue;
    }

    const double r = sqrt(r2);
    const int type_j = type[j];

    double dFix;
    double dFiy;
    double dFiz;
    compute_sus2_edge_derivative<GradT>(N, model, i, type_i, type_j, dx, dy, dz, r, grads, dFix, dFiy, dFiz);

    double dFjx;
    double dFjy;
    double dFjz;
    compute_sus2_edge_derivative<GradT>(N, model, j, type_j, type_i, -dx, -dy, -dz, r, grads, dFjx, dFjy, dFjz);

    fx += dFix - dFjx;
    fy += dFiy - dFjy;
    fz += dFiz - dFjz;

    s_xx -= dFix * dx;
    s_yy -= dFiy * dy;
    s_zz -= dFiz * dz;
    s_xy -= dFix * dy;
    s_xz -= dFix * dz;
    s_yz -= dFiy * dz;
    s_yx -= dFiy * dx;
    s_zx -= dFiz * dx;
    s_zy -= dFiz * dy;
  }

  force_tmp[i] = static_cast<float>(fx);
  force_tmp[i + N] = static_cast<float>(fy);
  force_tmp[i + 2 * N] = static_cast<float>(fz);

  virial_tmp[i + 0 * N] = static_cast<float>(s_xx);
  virial_tmp[i + 1 * N] = static_cast<float>(s_yy);
  virial_tmp[i + 2 * N] = static_cast<float>(s_zz);
  virial_tmp[i + 3 * N] = static_cast<float>(s_xy);
  virial_tmp[i + 4 * N] = static_cast<float>(s_xz);
  virial_tmp[i + 5 * N] = static_cast<float>(s_yz);
  virial_tmp[i + 6 * N] = static_cast<float>(s_yx);
  virial_tmp[i + 7 * N] = static_cast<float>(s_zx);
  virial_tmp[i + 8 * N] = static_cast<float>(s_zy);
}

static __global__ void gpu_accumulate_float_to_double(
  int N,
  const float* force_tmp,
  const float* virial_tmp,
  double* force,
  double* virial)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }
  force[i] += static_cast<double>(force_tmp[i]);
  force[i + N] += static_cast<double>(force_tmp[i + N]);
  force[i + 2 * N] += static_cast<double>(force_tmp[i + 2 * N]);
  for (int k = 0; k < 9; ++k) {
    virial[i + k * N] += static_cast<double>(virial_tmp[i + k * N]);
  }
}

} // namespace

SUS2_V11::SUS2_V11(
  const char* file_potential,
  int num_atoms,
  int num_potential_options,
  const char** potential_options)
{
  const SUS2HostModel host_model = load_model(file_potential);
  species_count_ = host_model.species_count;
  radial_funcs_count_ = host_model.radial_funcs_count;
  rb_size_ = host_model.rb_size;
  alpha_basic_count_ = host_model.alpha_basic_count;
  alpha_times_count_ = host_model.alpha_times_count;
  alpha_moments_count_ = host_model.alpha_moments_count;
  alpha_scalar_moments_ = host_model.alpha_scalar_moments;
  max_rank_ = host_model.max_rank;
  rc = host_model.max_dist;
  use_l3k3_basic_fastpath_ = has_l3k3_alpha_basic_layout(host_model);
  use_const_alpha_times_ = can_pack_alpha_times_u16(host_model);

  if (radial_funcs_count_ > 32 || max_rank_ > 7) {
    sus2_input_error("SUS2_V11 GPU scratch limits exceeded: radial_funcs_count<=32 and rank<=7.");
  }

  const char* profile_env = std::getenv("SUS2_GPUMD_PROFILE");
  profile_enabled_ = profile_env != nullptr && std::atoi(profile_env) != 0;
  const char* profile_interval_env = std::getenv("SUS2_GPUMD_PROFILE_INTERVAL");
  if (profile_interval_env != nullptr && std::atoi(profile_interval_env) > 0) {
    profile_interval_ = std::atoi(profile_interval_env);
  }
  const char* no_atomic_force_env = std::getenv("SUS2_GPUMD_PAIRWISE_NO_ATOMIC_FORCE");
  use_pairwise_no_atomic_force_ =
    no_atomic_force_env != nullptr && std::atoi(no_atomic_force_env) != 0;
  use_float_moment_grads_ = parse_float_moment_grads(host_model, num_potential_options, potential_options);

  shift_coeffs_.resize(host_model.shift_coeffs.size());
  shift_coeffs_.copy_from_host(host_model.shift_coeffs.data());
  species_coeffs_.resize(host_model.species_coeffs.size());
  species_coeffs_.copy_from_host(host_model.species_coeffs.data());
  moment_coeffs_.resize(host_model.moment_coeffs.size());
  moment_coeffs_.copy_from_host(host_model.moment_coeffs.data());
  alpha_basic_.resize(host_model.alpha_basic.size());
  alpha_basic_.copy_from_host(host_model.alpha_basic.data());
  alpha_times_.resize(host_model.alpha_times.size());
  alpha_times_.copy_from_host(host_model.alpha_times.data());
  if (use_const_alpha_times_) {
    std::vector<unsigned short> packed_alpha_times(host_model.alpha_times.size());
    for (size_t i = 0; i < host_model.alpha_times.size(); ++i) {
      packed_alpha_times[i] = static_cast<unsigned short>(host_model.alpha_times[i]);
    }
    CHECK(gpuMemcpyToSymbol(
      c_sus2_alpha_times_u16,
      packed_alpha_times.data(),
      packed_alpha_times.size() * sizeof(unsigned short)));
  }
  alpha_moment_mapping_.resize(host_model.alpha_moment_mapping.size());
  alpha_moment_mapping_.copy_from_host(host_model.alpha_moment_mapping.data());

  const int lut_span = parse_lut_span(host_model, num_potential_options, potential_options);
  lut_size_ = lut_span + 2;
  lut_inv_dr_ = static_cast<double>(lut_span) / rc;
  std::vector<double> host_lut_vals_double;
  std::vector<double> host_lut_ders_double;
  build_lut(host_model, lut_size_, lut_inv_dr_, host_lut_vals_double, host_lut_ders_double);
  std::vector<float> host_lut_vals(host_lut_vals_double.begin(), host_lut_vals_double.end());
  std::vector<float> host_lut_ders(host_lut_ders_double.begin(), host_lut_ders_double.end());
  lut_vals_.resize(host_lut_vals.size());
  lut_ders_.resize(host_lut_ders.size());
  lut_vals_.copy_from_host(host_lut_vals.data());
  lut_ders_.copy_from_host(host_lut_ders.data());

  neighbor_count_.resize(num_atoms);
  cell_contents_.resize(num_atoms);
  neighbor_cache_.initialize(rc, num_atoms, 512);
  resize_work_buffers(num_atoms);

  printf(
    "Use SUS2 v1.1 GPUMD potential: radial_type=%s, species=%d, radial=%d, basics=%d, moments=%d, scalars=%d, cutoff=%g A, LUT=%d (dr=%g A).\n",
    host_model.radial_basis_type.c_str(),
    species_count_,
    radial_funcs_count_,
    alpha_basic_count_,
    alpha_moments_count_,
    alpha_scalar_moments_,
    rc,
    lut_size_,
    1.0 / lut_inv_dr_);
  if (host_model.original_alpha_moments_count != alpha_moments_count_ ||
      host_model.original_alpha_times_count != alpha_times_count_) {
    printf(
      "SUS2 active DAG compression: moments %d -> %d, product rules %d -> %d.\n",
      host_model.original_alpha_moments_count,
      alpha_moments_count_,
      host_model.original_alpha_times_count,
      alpha_times_count_);
  } else {
    printf(
      "SUS2 active DAG compression: no inactive moments found (%d moments, %d product rules).\n",
      alpha_moments_count_,
      alpha_times_count_);
  }
  if (profile_enabled_) {
    printf("SUS2 v1.1 GPUMD profiling is enabled, report interval = %d calls.\n", profile_interval_);
  }
  if (use_pairwise_no_atomic_force_) {
    printf("SUS2 v1.1 GPUMD force mode: pairwise no-atomic for large-box neighbor lists.\n");
  }
  if (use_l3k3_basic_fastpath_) {
    printf("SUS2 v1.1 GPUMD basic/force fast path: l3k3 alpha_index_basic layout.\n");
  }
  if (use_const_alpha_times_) {
    printf("SUS2 v1.1 GPUMD product-rule table: constant-memory uint16 alpha_index_times.\n");
  }
  printf(
    "SUS2 v1.1 GPUMD moment-gradient workspace: %s.\n",
    use_float_moment_grads_ ? "float" : "double");
}

SUS2_V11::~SUS2_V11(void) {}

void SUS2_V11::maybe_print_profile()
{
  if (!profile_enabled_) {
    return;
  }
  ++profile_calls_;
  if (profile_calls_ % profile_interval_ != 0) {
    return;
  }

  const double inv = 1.0 / static_cast<double>(profile_interval_);
  const double total =
    profile_ms_[profile_neighbor] + profile_ms_[profile_zero] + profile_ms_[profile_basic] +
    profile_ms_[profile_forward] + profile_ms_[profile_energy_grad] + profile_ms_[profile_backward] +
    profile_ms_[profile_force] + profile_ms_[profile_accumulate];

  printf(
    "SUS2_PROFILE calls=%lld avg_ms: neighbor=%.6f neighbor_global=%.6f neighbor_local=%.6f zero=%.6f basic=%.6f forward=%.6f energy_grad=%.6f backward=%.6f force=%.6f accumulate=%.6f measured_total=%.6f\n",
    profile_calls_,
    profile_ms_[profile_neighbor] * inv,
    profile_ms_[profile_neighbor_global] * inv,
    profile_ms_[profile_neighbor_local] * inv,
    profile_ms_[profile_zero] * inv,
    profile_ms_[profile_basic] * inv,
    profile_ms_[profile_forward] * inv,
    profile_ms_[profile_energy_grad] * inv,
    profile_ms_[profile_backward] * inv,
    profile_ms_[profile_force] * inv,
    profile_ms_[profile_accumulate] * inv,
    total * inv);

  for (int i = 0; i < profile_count; ++i) {
    profile_ms_[i] = 0.0;
  }
}

void SUS2_V11::resize_work_buffers(int num_atoms)
{
  const size_t moment_size = static_cast<size_t>(alpha_moments_count_) * num_atoms;
  if (moment_vals_.size() != moment_size) {
    moment_vals_.resize(moment_size);
  }
  if (!use_float_moment_grads_ && moment_grads_.size() != moment_size) {
    moment_grads_.resize(moment_size);
  }
  if (use_float_moment_grads_ && moment_grads_float_.size() != moment_size) {
    moment_grads_float_.resize(moment_size);
  }
  const size_t force_size = static_cast<size_t>(num_atoms) * 3;
  const size_t virial_size = static_cast<size_t>(num_atoms) * 9;
  if (force_tmp_.size() != force_size) {
    force_tmp_.resize(force_size);
  }
  if (virial_tmp_.size() != virial_size) {
    virial_tmp_.resize(virial_size);
  }
}

void SUS2_V11::build_neighbor_list(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  int num_atoms)
{
  if (neighbor_count_.size() != static_cast<size_t>(num_atoms)) {
    neighbor_count_.resize(num_atoms);
  }
  if (cell_contents_.size() != static_cast<size_t>(num_atoms)) {
    cell_contents_.resize(num_atoms);
  }

  const int grid_size = (num_atoms - 1) / kBlockSize + 1;
  const double* x = position.data();
  const double* y = position.data() + num_atoms;
  const double* z = position.data() + num_atoms * 2;
  const double cutoff_square = rc * rc;
  const double volume = box.get_volume();
  box.thickness_x = volume / box.get_area(0);
  box.thickness_y = volume / box.get_area(1);
  box.thickness_z = volume / box.get_area(2);
  const int sx_range = periodic_image_range(box.pbc_x, rc, box.thickness_x);
  const int sy_range = periodic_image_range(box.pbc_y, rc, box.thickness_y);
  const int sz_range = periodic_image_range(box.pbc_z, rc, box.thickness_z);
  const bool needs_multi_image =
    (box.pbc_x && box.thickness_x < 2.0 * rc) || (box.pbc_y && box.thickness_y < 2.0 * rc) ||
    (box.pbc_z && box.thickness_z < 2.0 * rc);

  if (!needs_multi_image) {
    use_cached_neighbor_displacements_ = false;
    const size_t edge_capacity = static_cast<size_t>(num_atoms) * neighbor_capacity_;
    if (neighbor_atom_.size() != edge_capacity) {
      neighbor_atom_.resize(edge_capacity);
    }

    if (profile_enabled_) {
      CHECK(gpuDeviceSynchronize());
    }
    const auto global_start = std::chrono::high_resolution_clock::now();
    neighbor_cache_.find_neighbor_global(rc, box, type, position);
    if (profile_enabled_) {
      CHECK(gpuDeviceSynchronize());
      const auto global_stop = std::chrono::high_resolution_clock::now();
      profile_ms_[profile_neighbor_global] +=
        std::chrono::duration<double, std::milli>(global_stop - global_start).count();
    }

    if (profile_enabled_) {
      CHECK(gpuDeviceSynchronize());
    }
    const auto local_start = std::chrono::high_resolution_clock::now();
    neighbor_cache_.find_local_neighbor_from_global(rc, box, position, neighbor_count_, neighbor_atom_);
    if (profile_enabled_) {
      CHECK(gpuDeviceSynchronize());
      const auto local_stop = std::chrono::high_resolution_clock::now();
      profile_ms_[profile_neighbor_local] +=
        std::chrono::duration<double, std::milli>(local_stop - local_start).count();
    }
    return;
  }

  use_cached_neighbor_displacements_ = true;
  gpu_count_neighbors_images_on2<<<grid_size, kBlockSize>>>(
    num_atoms,
    box,
    -sx_range,
    sx_range,
    -sy_range,
    sy_range,
    -sz_range,
    sz_range,
    cutoff_square,
    x,
    y,
    z,
    neighbor_count_.data());
  GPU_CHECK_KERNEL

  int max_neighbors = 0;
  int* max_ptr = thrust::max_element(
    thrust::device, neighbor_count_.data(), neighbor_count_.data() + num_atoms);
  CHECK(gpuMemcpy(&max_neighbors, max_ptr, sizeof(int), gpuMemcpyDeviceToHost));
  const int alloc_neighbors = std::max(max_neighbors, 1);
  if (neighbor_atom_.size() != static_cast<size_t>(num_atoms) * alloc_neighbors) {
    neighbor_atom_.resize(static_cast<size_t>(num_atoms) * alloc_neighbors);
  }
  const size_t edge_capacity = static_cast<size_t>(num_atoms) * alloc_neighbors;
  if (neighbor_dx_.size() != edge_capacity) {
    neighbor_dx_.resize(edge_capacity);
  }
  if (neighbor_dy_.size() != edge_capacity) {
    neighbor_dy_.resize(edge_capacity);
  }
  if (neighbor_dz_.size() != edge_capacity) {
    neighbor_dz_.resize(edge_capacity);
  }
  gpu_fill_neighbors_images_on2<<<grid_size, kBlockSize>>>(
    num_atoms,
    box,
    -sx_range,
    sx_range,
    -sy_range,
    sy_range,
    -sz_range,
    sz_range,
    cutoff_square,
    x,
    y,
    z,
    neighbor_atom_.data(),
    neighbor_dx_.data(),
    neighbor_dy_.data(),
    neighbor_dz_.data());
  GPU_CHECK_KERNEL
}

void SUS2_V11::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial)
{
  const int num_atoms = static_cast<int>(type.size());
  resize_work_buffers(num_atoms);

  using ProfileClock = std::chrono::high_resolution_clock;
  auto profile_start = [&]() {
    if (profile_enabled_) {
      CHECK(gpuDeviceSynchronize());
    }
    return ProfileClock::now();
  };
  auto profile_stop = [&](ProfileSlot slot, const ProfileClock::time_point& start) {
    if (profile_enabled_) {
      CHECK(gpuDeviceSynchronize());
      const auto stop = ProfileClock::now();
      profile_ms_[slot] +=
        std::chrono::duration<double, std::milli>(stop - start).count();
    }
  };

  auto profile_t = profile_start();
  build_neighbor_list(box, type, position, num_atoms);
  profile_stop(profile_neighbor, profile_t);

  const int grid_size = (num_atoms - 1) / kBlockSize + 1;
  const size_t moment_size = static_cast<size_t>(alpha_moments_count_) * num_atoms;
  const bool use_pairwise_no_atomic_force =
    !use_cached_neighbor_displacements_ && use_pairwise_no_atomic_force_;

  profile_t = profile_start();
  CHECK(gpuMemset(moment_vals_.data(), 0, moment_size * sizeof(double)));
  if (use_float_moment_grads_) {
    CHECK(gpuMemset(moment_grads_float_.data(), 0, moment_size * sizeof(float)));
  } else {
    CHECK(gpuMemset(moment_grads_.data(), 0, moment_size * sizeof(double)));
  }
  if (!use_pairwise_no_atomic_force) {
    CHECK(gpuMemset(force_tmp_.data(), 0, static_cast<size_t>(num_atoms) * 3 * sizeof(float)));
    CHECK(gpuMemset(virial_tmp_.data(), 0, static_cast<size_t>(num_atoms) * 9 * sizeof(float)));
  }
  profile_stop(profile_zero, profile_t);

  SUS2DeviceModel model{
    species_count_,
    radial_funcs_count_,
    alpha_basic_count_,
    alpha_times_count_,
    alpha_moments_count_,
    alpha_scalar_moments_,
    max_rank_,
    lut_size_,
    rc,
    lut_inv_dr_,
    shift_coeffs_.data(),
    species_coeffs_.data(),
    moment_coeffs_.data(),
    alpha_basic_.data(),
    alpha_times_.data(),
    alpha_moment_mapping_.data(),
    lut_vals_.data(),
    lut_ders_.data(),
    use_l3k3_basic_fastpath_,
    use_const_alpha_times_};

  profile_t = profile_start();
  if (use_l3k3_basic_fastpath_) {
    gpu_compute_basic_moments_l3k3_accum<<<grid_size, kBlockSize>>>(
      num_atoms,
      box,
      rc * rc,
      use_cached_neighbor_displacements_,
      model,
      type.data(),
      neighbor_count_.data(),
      neighbor_atom_.data(),
      neighbor_dx_.data(),
      neighbor_dy_.data(),
      neighbor_dz_.data(),
      position.data(),
      position.data() + num_atoms,
      position.data() + 2 * num_atoms,
      moment_vals_.data());
  } else {
    gpu_compute_basic_moments<<<grid_size, kBlockSize>>>(
      num_atoms,
      box,
      rc * rc,
      use_cached_neighbor_displacements_,
      model,
      type.data(),
      neighbor_count_.data(),
      neighbor_atom_.data(),
      neighbor_dx_.data(),
      neighbor_dy_.data(),
      neighbor_dz_.data(),
      position.data(),
      position.data() + num_atoms,
      position.data() + 2 * num_atoms,
      moment_vals_.data());
  }
  GPU_CHECK_KERNEL
  profile_stop(profile_basic, profile_t);

  profile_t = profile_start();
  if (use_const_alpha_times_) {
    gpu_forward_times_const_u16<<<grid_size, kBlockSize>>>(num_atoms, model, moment_vals_.data());
  } else {
    gpu_forward_times<<<grid_size, kBlockSize>>>(num_atoms, model, moment_vals_.data());
  }
  GPU_CHECK_KERNEL
  profile_stop(profile_forward, profile_t);

  profile_t = profile_start();
  if (use_float_moment_grads_) {
    gpu_site_energy_init_grad<float><<<grid_size, kBlockSize>>>(
      num_atoms, model, type.data(), moment_vals_.data(), moment_grads_float_.data(), potential.data());
  } else {
    gpu_site_energy_init_grad<double><<<grid_size, kBlockSize>>>(
      num_atoms, model, type.data(), moment_vals_.data(), moment_grads_.data(), potential.data());
  }
  GPU_CHECK_KERNEL
  profile_stop(profile_energy_grad, profile_t);

  profile_t = profile_start();
  if (use_float_moment_grads_) {
    if (use_const_alpha_times_) {
      gpu_backward_times_const_u16<float><<<grid_size, kBlockSize>>>(
        num_atoms, model, moment_vals_.data(), moment_grads_float_.data());
    } else {
      gpu_backward_times<float><<<grid_size, kBlockSize>>>(
        num_atoms, model, moment_vals_.data(), moment_grads_float_.data());
    }
  } else {
    if (use_const_alpha_times_) {
      gpu_backward_times_const_u16<double><<<grid_size, kBlockSize>>>(
        num_atoms, model, moment_vals_.data(), moment_grads_.data());
    } else {
      gpu_backward_times<double><<<grid_size, kBlockSize>>>(
        num_atoms, model, moment_vals_.data(), moment_grads_.data());
    }
  }
  GPU_CHECK_KERNEL
  profile_stop(profile_backward, profile_t);

  profile_t = profile_start();
  if (!use_pairwise_no_atomic_force) {
    if (use_float_moment_grads_) {
      gpu_compute_forces<float><<<grid_size, kBlockSize>>>(
        num_atoms,
        box,
        rc * rc,
        use_cached_neighbor_displacements_,
        model,
        type.data(),
        neighbor_count_.data(),
        neighbor_atom_.data(),
        neighbor_dx_.data(),
        neighbor_dy_.data(),
        neighbor_dz_.data(),
        position.data(),
        position.data() + num_atoms,
        position.data() + 2 * num_atoms,
        moment_grads_float_.data(),
        force_tmp_.data(),
        virial_tmp_.data());
    } else {
      gpu_compute_forces<double><<<grid_size, kBlockSize>>>(
        num_atoms,
        box,
        rc * rc,
        use_cached_neighbor_displacements_,
        model,
        type.data(),
        neighbor_count_.data(),
        neighbor_atom_.data(),
        neighbor_dx_.data(),
        neighbor_dy_.data(),
        neighbor_dz_.data(),
        position.data(),
        position.data() + num_atoms,
        position.data() + 2 * num_atoms,
        moment_grads_.data(),
        force_tmp_.data(),
        virial_tmp_.data());
    }
    GPU_CHECK_KERNEL
  } else {
    if (use_float_moment_grads_) {
      gpu_compute_forces_pairwise_no_atomic<float><<<grid_size, kBlockSize>>>(
        num_atoms,
        box,
        rc * rc,
        model,
        type.data(),
        neighbor_count_.data(),
        neighbor_atom_.data(),
        position.data(),
        position.data() + num_atoms,
        position.data() + 2 * num_atoms,
        moment_grads_float_.data(),
        force_tmp_.data(),
        virial_tmp_.data());
    } else {
      gpu_compute_forces_pairwise_no_atomic<double><<<grid_size, kBlockSize>>>(
        num_atoms,
        box,
        rc * rc,
        model,
        type.data(),
        neighbor_count_.data(),
        neighbor_atom_.data(),
        position.data(),
        position.data() + num_atoms,
        position.data() + 2 * num_atoms,
        moment_grads_.data(),
        force_tmp_.data(),
        virial_tmp_.data());
    }
    GPU_CHECK_KERNEL
  }
  profile_stop(profile_force, profile_t);

  profile_t = profile_start();
  gpu_accumulate_float_to_double<<<grid_size, kBlockSize>>>(
    num_atoms, force_tmp_.data(), virial_tmp_.data(), force.data(), virial.data());
  GPU_CHECK_KERNEL
  profile_stop(profile_accumulate, profile_t);

  maybe_print_profile();
}
