#include "sus2_sh.cuh"

#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
constexpr int kBlockSize = 128;
constexpr int kMaxSHL = 4;
constexpr int kMaxSHComponents = (kMaxSHL + 1) * (kMaxSHL + 1);
constexpr int kMaxSHRadialFuncs = 32;
constexpr int kMaxSHRbSize = 16;
constexpr int kMaxSHBasics = 256;
constexpr int kSHForceGradCache64 = 64;
constexpr double kPi = 3.141592653589793238462643383279502884;

enum SHProfileStage {
  sh_profile_neighbor = 0,
  sh_profile_memset = 1,
  sh_profile_basic = 2,
  sh_profile_product = 3,
  sh_profile_force = 4,
  sh_profile_accumulate = 5,
  sh_profile_count = 6
};

struct SHProductHost {
  int left = 0;
  int right = 0;
  int target = 0;
  double coeff = 0.0;
};

struct SHHostModel {
  int species_count = 0;
  int sh_l_max = 0;
  int sh_k_max = 0;
  int sh_body_order = 0;
  int radial_funcs_count = 0;
  int rb_size = 0;
  int alpha_basic_count = 0;
  int alpha_moments_count = 0;
  int alpha_scalar_moments = 0;
  double scaling = 1.0;
  double max_dist = 0.0;
  std::string potential_tag;
  std::string radial_basis_type;
  std::string scaling_map;
  std::vector<double> shift_coeffs;
  std::vector<double> scal_coeffs;
  std::vector<double> radial_coeffs;
  std::vector<double> radial_type_coeffs;
  std::vector<double> species_coeffs;
  std::vector<double> moment_coeffs;
  std::vector<int> alpha_basic;
  std::vector<SHProductHost> products;
  std::vector<int> alpha_moment_mapping;
};

[[noreturn]] void sh_input_error(const std::string& message)
{
  std::cout << message << std::endl;
  std::exit(1);
}

std::string read_text_file(const std::string& path)
{
  std::ifstream ifs(path);
  if (!ifs) {
    sh_input_error("Failed to open SUS2-SH model file: " + path);
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
    sh_input_error("Missing token in SUS2-SH model file: " + token);
  }
  return pos;
}

bool has_token(const std::string& text, const std::string& token)
{
  return text.find(token) != std::string::npos;
}

std::string parse_string_after(const std::string& text, const std::string& token)
{
  const size_t pos = find_required(text, token) + token.size();
  const size_t end = text.find('\n', pos);
  return trim(text.substr(pos, end == std::string::npos ? std::string::npos : end - pos));
}

int parse_int_after(const std::string& text, const std::string& token)
{
  return std::stoi(parse_string_after(text, token));
}

double parse_double_after(const std::string& text, const std::string& token)
{
  return std::stod(parse_string_after(text, token));
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
        return text.substr(start, i - start + 1);
      }
    }
  }
  sh_input_error("Unterminated braced section in SUS2-SH model file: " + token);
}

std::vector<std::string> extract_top_level_groups(const std::string& text)
{
  std::vector<std::string> groups;
  int depth = 0;
  size_t start = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '{') {
      if (depth == 1) {
        start = i;
      }
      ++depth;
    } else if (text[i] == '}') {
      if (depth == 2) {
        groups.push_back(text.substr(start, i - start + 1));
      }
      --depth;
    }
  }
  return groups;
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
        groups.push_back(text.substr(start, i - start + 1));
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
  std::string cleaned = text;
  for (char& c : cleaned) {
    if (c == '{' || c == '}' || c == ',') {
      c = ' ';
    }
  }
  std::istringstream iss(cleaned);
  std::vector<double> values;
  double value = 0.0;
  while (iss >> value) {
    values.push_back(value);
  }
  return values;
}

template <>
std::vector<int> parse_numbers<int>(const std::string& text)
{
  std::string cleaned = text;
  for (char& c : cleaned) {
    if (c == '{' || c == '}' || c == ',') {
      c = ' ';
    }
  }
  std::istringstream iss(cleaned);
  std::vector<int> values;
  int value = 0;
  while (iss >> value) {
    values.push_back(value);
  }
  return values;
}

bool starts_with(const std::string& text, const std::string& prefix)
{
  return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool parse_bool_value(const std::string& value, const std::string& option_name)
{
  if (value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on") {
    return true;
  }
  if (value == "0" || value == "false" || value == "FALSE" || value == "no" || value == "off") {
    return false;
  }
  sh_input_error("Invalid boolean value for " + option_name + ": " + value);
}

bool parse_sh_float(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_float = true;
  const char* env = std::getenv("SUS2_GPUMD_FLOAT");
  if (env != nullptr) {
    use_float = parse_bool_value(env, "SUS2_GPUMD_FLOAT");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_float=") || starts_with(option, "sus2_sh_float=")) {
      const size_t eq = option.find('=');
      use_float = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_float;
}

bool parse_sh_radial_direct(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_direct = true;
  const char* env = std::getenv("SUS2_GPUMD_RADIAL_DIRECT");
  if (env != nullptr) {
    use_direct = parse_bool_value(env, "SUS2_GPUMD_RADIAL_DIRECT");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_radial_direct=") ||
        starts_with(option, "sus2_sh_radial_direct=") ||
        starts_with(option, "radial_direct=")) {
      const size_t eq = option.find('=');
      use_direct = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_direct;
}

bool parse_sh_force_self_buffer(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_buffer = true;
  const char* env = std::getenv("SUS2_GPUMD_FORCE_SELF_BUFFER");
  if (env != nullptr) {
    use_buffer = parse_bool_value(env, "SUS2_GPUMD_FORCE_SELF_BUFFER");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_force_self_buffer=") ||
        starts_with(option, "sus2_sh_force_self_buffer=") ||
        starts_with(option, "force_self_buffer=")) {
      const size_t eq = option.find('=');
      use_buffer = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_buffer;
}

bool parse_sh_force_grad_cache(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_cache = false;
  const char* env = std::getenv("SUS2_SH_GPUMD_FORCE_GRAD_CACHE");
  if (env != nullptr) {
    use_cache = parse_bool_value(env, "SUS2_SH_GPUMD_FORCE_GRAD_CACHE");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_force_grad_cache=") ||
        starts_with(option, "sus2_force_grad_cache=")) {
      const size_t eq = option.find('=');
      use_cache = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_cache;
}

bool parse_sh_profile_enabled(const SHHostModel& model, int nopts, const char** opts)
{
  bool enabled = false;
  const char* env = std::getenv("SUS2_SH_GPUMD_PROFILE");
  if (env == nullptr) {
    env = std::getenv("SUS2_GPUMD_PROFILE");
  }
  if (env != nullptr) {
    enabled = parse_bool_value(env, "SUS2_SH_GPUMD_PROFILE");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_profile=") || starts_with(option, "sus2_profile=")) {
      const size_t eq = option.find('=');
      enabled = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return enabled;
}

int parse_sh_profile_interval(const SHHostModel& model, int nopts, const char** opts)
{
  int interval = 50;
  const char* env = std::getenv("SUS2_SH_GPUMD_PROFILE_INTERVAL");
  if (env == nullptr) {
    env = std::getenv("SUS2_GPUMD_PROFILE_INTERVAL");
  }
  if (env != nullptr) {
    interval = std::max(1, std::atoi(env));
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_profile_interval=") ||
        starts_with(option, "sus2_profile_interval=")) {
      const size_t eq = option.find('=');
      interval = std::max(1, std::atoi(option.substr(eq + 1).c_str()));
    }
  }
  return interval;
}

SHHostModel load_sh_model(const std::string& path)
{
  const std::string text = read_text_file(path);
  SHHostModel model;
  const std::string version = parse_string_after(text, "version =");
  if (version != "1.1.0") {
    sh_input_error("SUS2_SH currently supports only version = 1.1.0.");
  }
  model.potential_tag = parse_string_after(text, "potential_tag =");
  if (model.potential_tag != "SUS2-SH") {
    sh_input_error("SUS2_SH selected for a non-SUS2-SH model.");
  }
  model.scaling = parse_double_after(text, "scaling =");
  model.scaling_map = parse_string_after(text, "scaling_map =");
  model.species_count = parse_int_after(text, "species_count =");
  model.sh_l_max = parse_int_after(text, "sh_l_max =");
  model.sh_k_max = parse_int_after(text, "sh_k_max =");
  model.sh_body_order = parse_int_after(text, "sh_body_order =");
  model.radial_basis_type = parse_string_after(text, "radial_basis_type =");
  model.max_dist = parse_double_after(text, "max_dist =");
  model.rb_size = parse_int_after(text, "radial_basis_size =");
  model.radial_funcs_count = parse_int_after(text, "radial_funcs_count =");
  model.alpha_moments_count = parse_int_after(text, "alpha_moments_count =");
  model.alpha_basic_count = parse_int_after(text, "alpha_index_basic_count =");
  model.alpha_scalar_moments = parse_int_after(text, "alpha_scalar_moments =");

  if (model.scaling_map != "LK") {
    sh_input_error("SUS2_SH requires scaling_map = LK.");
  }
  if (model.radial_basis_type != "RBChebyshev_sss") {
    sh_input_error("SUS2_SH first GPUMD backend currently supports RBChebyshev_sss only.");
  }
  if (model.sh_l_max < 0 || model.sh_l_max > kMaxSHL) {
    sh_input_error("SUS2_SH supports sh_l_max in [0,4].");
  }
  if (model.sh_k_max <= 0 || model.radial_funcs_count != model.sh_k_max * (model.sh_l_max + 1)) {
    sh_input_error("SUS2_SH inconsistent sh_k_max/radial_funcs_count.");
  }
  if (model.rb_size <= 0 || model.rb_size > kMaxSHRbSize ||
      model.radial_funcs_count > kMaxSHRadialFuncs) {
    sh_input_error("SUS2_SH GPU scratch limit exceeded: rb_size<=16, radial_funcs_count<=32.");
  }
  if (model.alpha_basic_count <= 0 || model.alpha_basic_count > kMaxSHBasics) {
    sh_input_error("SUS2_SH GPU scratch limit exceeded: alpha_index_basic_count<=256.");
  }

  model.shift_coeffs = parse_numbers<double>(extract_braced_after(text, "shift_coeffs ="));
  model.scal_coeffs = parse_numbers<double>(extract_braced_after(text, "scal_coeffs ="));
  model.species_coeffs = parse_numbers<double>(extract_braced_after(text, "species_coeffs ="));
  model.moment_coeffs = parse_numbers<double>(extract_braced_after(text, "moment_coeffs ="));
  model.alpha_moment_mapping =
    parse_numbers<int>(extract_braced_after(text, "alpha_moment_mapping ="));

  const std::vector<int> basic_raw =
    parse_numbers<int>(extract_braced_after(text, "alpha_index_basic ="));
  if (static_cast<int>(basic_raw.size()) != model.alpha_basic_count * 3) {
    sh_input_error("Unexpected SUS2_SH alpha_index_basic size.");
  }
  model.alpha_basic.resize(model.alpha_basic_count * 3);
  for (int i = 0; i < model.alpha_basic_count; ++i) {
    const int k = basic_raw[i * 3 + 0];
    const int l = basic_raw[i * 3 + 1];
    const int m = basic_raw[i * 3 + 2];
    if (k < 0 || k >= model.sh_k_max || l < 0 || l > model.sh_l_max || std::abs(m) > l) {
      sh_input_error("Invalid SUS2_SH alpha_index_basic entry.");
    }
    model.alpha_basic[i * 3 + 0] = k * (model.sh_l_max + 1) + l;
    model.alpha_basic[i * 3 + 1] = l;
    model.alpha_basic[i * 3 + 2] = m;
  }

  const int sh_product_count = parse_int_after(text, "sh_product_count =");
  const auto product_groups = extract_top_level_groups(extract_braced_after(text, "sh_products ="));
  if (static_cast<int>(product_groups.size()) != sh_product_count) {
    sh_input_error("Unexpected SUS2_SH sh_products size.");
  }
  model.products.resize(sh_product_count);
  for (int p = 0; p < sh_product_count; ++p) {
    const auto values = parse_numbers<double>(product_groups[p]);
    if (values.size() != 4) {
      sh_input_error("Invalid SUS2_SH product entry.");
    }
    model.products[p].left = static_cast<int>(values[0]);
    model.products[p].right = static_cast<int>(values[1]);
    model.products[p].target = static_cast<int>(values[2]);
    model.products[p].coeff = values[3];
    if (model.products[p].left < 0 || model.products[p].left >= model.alpha_moments_count ||
        model.products[p].right < 0 || model.products[p].right >= model.alpha_moments_count ||
        model.products[p].target < 0 || model.products[p].target >= model.alpha_moments_count) {
      sh_input_error("SUS2_SH product index out of range.");
    }
  }

  model.radial_coeffs.resize(model.radial_funcs_count * model.rb_size, 0.0);
  model.radial_type_coeffs.assign(model.species_count, 1.0);
  const size_t radial_start = find_required(text, "radial_coeffs");
  const size_t radial_end = find_required(text, "alpha_moments_count", radial_start);
  const auto radial_groups =
    extract_all_brace_groups(text.substr(radial_start, radial_end - radial_start));
  if (static_cast<int>(radial_groups.size()) < model.radial_funcs_count) {
    sh_input_error("Unexpected SUS2_SH radial_coeffs section.");
  }
  for (int mu = 0; mu < model.radial_funcs_count; ++mu) {
    const auto values = parse_numbers<double>(radial_groups[mu]);
    if (static_cast<int>(values.size()) != model.rb_size + model.species_count) {
      sh_input_error("Unexpected SUS2_SH radial_coeffs row size.");
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

  if (static_cast<int>(model.shift_coeffs.size()) != model.species_count ||
      static_cast<int>(model.species_coeffs.size()) != model.species_count ||
      static_cast<int>(model.scal_coeffs.size()) !=
        2 * model.species_count * model.species_count * model.radial_funcs_count ||
      static_cast<int>(model.alpha_moment_mapping.size()) != model.alpha_scalar_moments ||
      static_cast<int>(model.moment_coeffs.size()) != model.alpha_scalar_moments) {
    sh_input_error("Unexpected SUS2_SH coefficient dimensions.");
  }
  return model;
}

void build_direct_radial_tables(
  const SHHostModel& model,
  std::vector<float>& coeffs,
  std::vector<float>& scal_s)
{
  const size_t pair_count = static_cast<size_t>(model.species_count) * model.species_count;
  coeffs.assign(pair_count * model.radial_funcs_count * model.rb_size, 0.0f);
  scal_s.assign(pair_count * model.radial_funcs_count * 2, 0.0f);
  for (int zi = 0; zi < model.species_count; ++zi) {
    for (int zj = 0; zj < model.species_count; ++zj) {
      const int pair = zi * model.species_count + zj;
      const int shift = model.species_count * zi + zj;
      const float type_scale = static_cast<float>(
        model.scaling * model.radial_type_coeffs[zi] * model.radial_type_coeffs[zj]);
      for (int mu = 0; mu < model.radial_funcs_count; ++mu) {
        const int scal_offset =
          2 * mu * model.species_count * model.species_count + shift;
        const size_t scal_base = (static_cast<size_t>(pair) * model.radial_funcs_count + mu) * 2;
        scal_s[scal_base + 0] = static_cast<float>(model.scal_coeffs[scal_offset]);
        scal_s[scal_base + 1] =
          static_cast<float>(model.scal_coeffs[scal_offset + model.species_count * model.species_count]);
        const size_t coeff_base =
          (static_cast<size_t>(pair) * model.radial_funcs_count + mu) * model.rb_size;
        for (int xi = 0; xi < model.rb_size; ++xi) {
          coeffs[coeff_base + xi] =
            static_cast<float>(model.radial_coeffs[mu * model.rb_size + xi]) * type_scale;
        }
      }
    }
  }
}

struct SHDeviceModel {
  int species_count;
  int sh_l_max;
  int radial_funcs_count;
  int rb_size;
  int alpha_basic_count;
  int sh_product_count;
  int alpha_moments_count;
  int alpha_scalar_moments;
  double max_dist;
  const double* shift_coeffs;
  const double* species_coeffs;
  const double* moment_coeffs;
  const float* shift_coeffs_float;
  const float* species_coeffs_float;
  const float* moment_coeffs_float;
  const int* alpha_basic;
  const int* alpha_basic_mu_yidx;
  const int* sh_products_int;
  const double* sh_products_coeff;
  const float* sh_products_coeff_float;
  const int* alpha_moment_mapping;
  const float* radial_direct_coeffs;
  const float* radial_direct_scal_s;
  bool use_float_model_params;
};

template <typename RealT>
__device__ __forceinline__ RealT sh_shift_coeff(const SHDeviceModel& model, int type)
{
  return model.use_float_model_params ? static_cast<RealT>(model.shift_coeffs_float[type])
                                      : static_cast<RealT>(model.shift_coeffs[type]);
}

template <typename RealT>
__device__ __forceinline__ RealT sh_species_coeff(const SHDeviceModel& model, int type)
{
  return model.use_float_model_params ? static_cast<RealT>(model.species_coeffs_float[type])
                                      : static_cast<RealT>(model.species_coeffs[type]);
}

template <typename RealT>
__device__ __forceinline__ RealT sh_moment_coeff(const SHDeviceModel& model, int idx)
{
  return model.use_float_model_params ? static_cast<RealT>(model.moment_coeffs_float[idx])
                                      : static_cast<RealT>(model.moment_coeffs[idx]);
}

template <typename RealT>
__device__ __forceinline__ RealT sh_product_coeff(const SHDeviceModel& model, int idx)
{
  return model.use_float_model_params ? static_cast<RealT>(model.sh_products_coeff_float[idx])
                                      : static_cast<RealT>(model.sh_products_coeff[idx]);
}

__device__ __forceinline__ int sh_flat_index(int l, int m)
{
  return l * l + (m + l);
}

template <typename RealT>
__device__ __forceinline__ RealT sh_inv_power(int l, RealT r)
{
  const RealT inv_r = static_cast<RealT>(1.0) / r;
  const RealT inv_r2 = inv_r * inv_r;
  if (l == 0) {
    return static_cast<RealT>(1.0);
  }
  if (l == 1) {
    return inv_r;
  }
  if (l == 2) {
    return inv_r2;
  }
  if (l == 3) {
    return inv_r2 * inv_r;
  }
  return inv_r2 * inv_r2;
}

template <typename RealT>
__device__ __forceinline__ void sh_add_real(
  int l,
  int m,
  RealT coeff,
  RealT poly,
  RealT dpx,
  RealT dpy,
  RealT dpz,
  RealT x,
  RealT y,
  RealT z,
  RealT r,
  RealT* vals,
  RealT* ders)
{
  const int idx = sh_flat_index(l, m);
  const RealT inv_r = static_cast<RealT>(1.0) / r;
  const RealT inv_pow = sh_inv_power(l, r);
  const RealT inv_pow_der =
    l == 0 ? static_cast<RealT>(0.0) : -static_cast<RealT>(l) * inv_pow * inv_r * inv_r;
  vals[idx] = coeff * poly * inv_pow;
  if (ders != nullptr) {
    ders[3 * idx + 0] = coeff * (dpx * inv_pow + poly * inv_pow_der * x);
    ders[3 * idx + 1] = coeff * (dpy * inv_pow + poly * inv_pow_der * y);
    ders[3 * idx + 2] = coeff * (dpz * inv_pow + poly * inv_pow_der * z);
  }
}

template <typename RealT>
__device__ __forceinline__ void eval_real_sh(
  RealT x,
  RealT y,
  RealT z,
  RealT r,
  int lmax,
  RealT* vals,
  RealT* ders)
{
  for (int i = 0; i < kMaxSHComponents; ++i) {
    vals[i] = static_cast<RealT>(0.0);
    if (ders != nullptr) {
      ders[3 * i + 0] = ders[3 * i + 1] = ders[3 * i + 2] = static_cast<RealT>(0.0);
    }
  }
  const RealT x2 = x * x;
  const RealT y2 = y * y;
  const RealT z2 = z * z;
  const RealT rho2 = x2 + y2;

  sh_add_real<RealT>(0, 0, static_cast<RealT>(0.5 / sqrt(kPi)), static_cast<RealT>(1.0),
              static_cast<RealT>(0.0), static_cast<RealT>(0.0), static_cast<RealT>(0.0),
              x, y, z, r, vals, ders);
  if (lmax == 0) {
    return;
  }

  const RealT c1 = static_cast<RealT>(0.5 * sqrt(3.0 / kPi));
  sh_add_real<RealT>(1, -1, c1, y, 0, 1, 0, x, y, z, r, vals, ders);
  sh_add_real<RealT>(1, 0, c1, z, 0, 0, 1, x, y, z, r, vals, ders);
  sh_add_real<RealT>(1, 1, c1, x, 1, 0, 0, x, y, z, r, vals, ders);
  if (lmax == 1) {
    return;
  }

  const RealT c2a = static_cast<RealT>(0.5 * sqrt(15.0 / kPi));
  const RealT c20 = static_cast<RealT>(0.25 * sqrt(5.0 / kPi));
  const RealT c22 = static_cast<RealT>(0.25 * sqrt(15.0 / kPi));
  sh_add_real<RealT>(2, -2, c2a, x * y, y, x, 0, x, y, z, r, vals, ders);
  sh_add_real<RealT>(2, -1, c2a, y * z, 0, z, y, x, y, z, r, vals, ders);
  sh_add_real<RealT>(2, 0, c20, static_cast<RealT>(2.0) * z2 - x2 - y2,
              -static_cast<RealT>(2.0) * x, -static_cast<RealT>(2.0) * y,
              static_cast<RealT>(4.0) * z, x, y, z, r, vals, ders);
  sh_add_real<RealT>(2, 1, c2a, x * z, z, 0, x, x, y, z, r, vals, ders);
  sh_add_real<RealT>(2, 2, c22, x2 - y2, static_cast<RealT>(2.0) * x,
              -static_cast<RealT>(2.0) * y, 0, x, y, z, r, vals, ders);
  if (lmax == 2) {
    return;
  }

  const RealT c33 = static_cast<RealT>(0.125 * sqrt(70.0 / kPi));
  const RealT c32 = static_cast<RealT>(0.5 * sqrt(105.0 / kPi));
  const RealT c31 = static_cast<RealT>(0.125 * sqrt(42.0 / kPi));
  const RealT c30 = static_cast<RealT>(0.25 * sqrt(7.0 / kPi));
  const RealT c3p2 = static_cast<RealT>(0.25 * sqrt(105.0 / kPi));
  const RealT p3m3 = y * (static_cast<RealT>(3.0) * x2 - y2);
  const RealT a31 = static_cast<RealT>(4.0) * z2 - x2 - y2;
  const RealT p30 = z * (static_cast<RealT>(2.0) * z2 - static_cast<RealT>(3.0) * x2 -
                         static_cast<RealT>(3.0) * y2);
  const RealT p32 = z * (x2 - y2);
  const RealT p33 = x * (x2 - static_cast<RealT>(3.0) * y2);
  sh_add_real<RealT>(3, -3, c33, p3m3, static_cast<RealT>(6.0) * x * y,
              static_cast<RealT>(3.0) * x2 - static_cast<RealT>(3.0) * y2, 0,
              x, y, z, r, vals, ders);
  sh_add_real<RealT>(3, -2, c32, x * y * z, y * z, x * z, x * y, x, y, z, r, vals, ders);
  sh_add_real<RealT>(3, -1, c31, y * a31, -static_cast<RealT>(2.0) * x * y,
              a31 - static_cast<RealT>(2.0) * y2, static_cast<RealT>(8.0) * y * z,
              x, y, z, r, vals, ders);
  sh_add_real<RealT>(3, 0, c30, p30, -static_cast<RealT>(6.0) * x * z,
              -static_cast<RealT>(6.0) * y * z,
              static_cast<RealT>(6.0) * z2 - static_cast<RealT>(3.0) * x2 -
                static_cast<RealT>(3.0) * y2,
              x, y, z, r, vals, ders);
  sh_add_real<RealT>(3, 1, c31, x * a31, a31 - static_cast<RealT>(2.0) * x2,
              -static_cast<RealT>(2.0) * x * y, static_cast<RealT>(8.0) * x * z,
              x, y, z, r, vals, ders);
  sh_add_real<RealT>(3, 2, c3p2, p32, static_cast<RealT>(2.0) * x * z,
              -static_cast<RealT>(2.0) * y * z, x2 - y2, x, y, z, r, vals, ders);
  sh_add_real<RealT>(3, 3, c33, p33, static_cast<RealT>(3.0) * x2 - static_cast<RealT>(3.0) * y2,
              -static_cast<RealT>(6.0) * x * y, 0, x, y, z, r, vals, ders);
  if (lmax == 3) {
    return;
  }

  const RealT c44m = static_cast<RealT>(0.75 * sqrt(35.0 / kPi));
  const RealT c43 = static_cast<RealT>(0.375 * sqrt(70.0 / kPi));
  const RealT c42m = static_cast<RealT>(0.75 * sqrt(5.0 / kPi));
  const RealT c41 = static_cast<RealT>(0.375 * sqrt(10.0 / kPi));
  const RealT c40 = static_cast<RealT>(0.1875 / sqrt(kPi));
  const RealT c42 = static_cast<RealT>(0.375 * sqrt(5.0 / kPi));
  const RealT c44 = static_cast<RealT>(0.1875 * sqrt(35.0 / kPi));
  const RealT p44base = x2 - y2;
  const RealT p4m4 = x * y * p44base;
  const RealT a42 = static_cast<RealT>(6.0) * z2 - rho2;
  const RealT a41 = static_cast<RealT>(4.0) * z2 - static_cast<RealT>(3.0) * rho2;
  const RealT p40 = static_cast<RealT>(8.0) * z2 * z2 -
                    static_cast<RealT>(24.0) * z2 * rho2 +
                    static_cast<RealT>(3.0) * rho2 * rho2;
  const RealT p44 = x2 * x2 - static_cast<RealT>(6.0) * x2 * y2 + y2 * y2;
  sh_add_real<RealT>(4, -4, c44m, p4m4, y * (static_cast<RealT>(3.0) * x2 - y2),
              x * (x2 - static_cast<RealT>(3.0) * y2), 0, x, y, z, r, vals, ders);
  sh_add_real<RealT>(4, -3, c43, z * p3m3, static_cast<RealT>(6.0) * x * y * z,
              z * (static_cast<RealT>(3.0) * x2 - static_cast<RealT>(3.0) * y2),
              p3m3, x, y, z, r, vals, ders);
  sh_add_real<RealT>(4, -2, c42m, x * y * a42, y * a42 - static_cast<RealT>(2.0) * x2 * y,
              x * a42 - static_cast<RealT>(2.0) * x * y2,
              static_cast<RealT>(12.0) * x * y * z, x, y, z, r, vals, ders);
  sh_add_real<RealT>(4, -1, c41, y * z * a41, -static_cast<RealT>(6.0) * x * y * z,
              z * (a41 - static_cast<RealT>(6.0) * y2),
              y * (static_cast<RealT>(12.0) * z2 - static_cast<RealT>(3.0) * rho2),
              x, y, z, r, vals, ders);
  sh_add_real<RealT>(4, 0, c40, p40, static_cast<RealT>(12.0) * x * (rho2 - static_cast<RealT>(4.0) * z2),
              static_cast<RealT>(12.0) * y * (rho2 - static_cast<RealT>(4.0) * z2),
              static_cast<RealT>(16.0) * z * (static_cast<RealT>(2.0) * z2 -
                                              static_cast<RealT>(3.0) * rho2),
              x, y, z, r, vals, ders);
  sh_add_real<RealT>(4, 1, c41, x * z * a41, z * (a41 - static_cast<RealT>(6.0) * x2),
              -static_cast<RealT>(6.0) * x * y * z,
              x * (static_cast<RealT>(12.0) * z2 - static_cast<RealT>(3.0) * rho2),
              x, y, z, r, vals, ders);
  sh_add_real<RealT>(4, 2, c42, p44base * a42,
              static_cast<RealT>(2.0) * x * a42 - static_cast<RealT>(2.0) * x * p44base,
              -static_cast<RealT>(2.0) * y * a42 - static_cast<RealT>(2.0) * y * p44base,
              static_cast<RealT>(12.0) * z * p44base, x, y, z, r, vals, ders);
  sh_add_real<RealT>(4, 3, c43, z * p33,
              z * (static_cast<RealT>(3.0) * x2 - static_cast<RealT>(3.0) * y2),
              -static_cast<RealT>(6.0) * x * y * z, p33, x, y, z, r, vals, ders);
  sh_add_real<RealT>(4, 4, c44, p44, static_cast<RealT>(4.0) * x * x2 -
              static_cast<RealT>(12.0) * x * y2,
              -static_cast<RealT>(12.0) * x2 * y + static_cast<RealT>(4.0) * y * y2,
              0, x, y, z, r, vals, ders);
}

template <typename RealT>
__device__ __forceinline__ void sh_direct_radial_vals_ders(
  const SHDeviceModel& model,
  int pair,
  RealT r,
  RealT* vals,
  RealT* ders)
{
  const RealT dr = r - static_cast<RealT>(model.max_dist);
  const RealT cutoff_f = dr * dr;
  const RealT cutoff_der = static_cast<RealT>(2.0) * dr;
  for (int mu = 0; mu < model.radial_funcs_count; ++mu) {
    const size_t scal_base = (static_cast<size_t>(pair) * model.radial_funcs_count + mu) * 2;
    const RealT scal = static_cast<RealT>(model.radial_direct_scal_s[scal_base + 0]);
    const RealT shift = static_cast<RealT>(model.radial_direct_scal_s[scal_base + 1]);
    const RealT z = static_cast<RealT>(0.5) * scal * (r - shift);
    const RealT ksi = tanh(z);
    const RealT mult = static_cast<RealT>(0.5) * scal * (static_cast<RealT>(1.0) - ksi * ksi);
    const size_t coeff_base = (static_cast<size_t>(pair) * model.radial_funcs_count + mu) * model.rb_size;

    RealT prev = static_cast<RealT>(1.0);
    RealT prev_x = static_cast<RealT>(0.0);
    RealT acc_s = static_cast<RealT>(model.radial_direct_coeffs[coeff_base]);
    RealT acc_sx = static_cast<RealT>(0.0);
    if (model.rb_size > 1) {
      RealT curr = ksi;
      RealT curr_x = static_cast<RealT>(1.0);
      RealT coeff = static_cast<RealT>(model.radial_direct_coeffs[coeff_base + 1]);
      acc_s += coeff * curr;
      acc_sx += coeff * curr_x;
      for (int xi = 2; xi < model.rb_size; ++xi) {
        const RealT next = static_cast<RealT>(2.0) * ksi * curr - prev;
        const RealT next_x = static_cast<RealT>(2.0) * (curr + ksi * curr_x) - prev_x;
        coeff = static_cast<RealT>(model.radial_direct_coeffs[coeff_base + xi]);
        acc_s += coeff * next;
        acc_sx += coeff * next_x;
        prev = curr;
        prev_x = curr_x;
        curr = next;
        curr_x = next_x;
      }
    }
    vals[mu] = cutoff_f * acc_s;
    if (ders != nullptr) {
      ders[mu] = cutoff_der * acc_s + cutoff_f * mult * acc_sx;
    }
  }
}

template <typename RealT>
__device__ __forceinline__ void load_sh_edge_displacement(
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
  RealT& dx,
  RealT& dy,
  RealT& dz)
{
  if (use_cached_displacements) {
    dx = static_cast<RealT>(neighbor_dx[edge]);
    dy = static_cast<RealT>(neighbor_dy[edge]);
    dz = static_cast<RealT>(neighbor_dz[edge]);
    return;
  }
  dx = static_cast<RealT>(x[j] - x[i]);
  dy = static_cast<RealT>(y[j] - y[i]);
  dz = static_cast<RealT>(z[j] - z[i]);
  apply_mic(box, dx, dy, dz);
}

template <typename RealT>
static __global__ void gpu_sh_compute_basic(
  int N,
  Box box,
  double cutoff_square,
  bool use_cached_displacements,
  SHDeviceModel model,
  const int* type,
  const int* neighbor_count,
  const int* neighbor_atoms,
  const double* neighbor_dx,
  const double* neighbor_dy,
  const double* neighbor_dz,
  const double* x,
  const double* y,
  const double* z,
  RealT* moments)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  RealT basic[kMaxSHBasics];
  for (int b = 0; b < model.alpha_basic_count; ++b) {
    basic[b] = static_cast<RealT>(0.0);
  }

  const int type_i = type[i];
  const int count = neighbor_count[i];
  for (int nbr = 0; nbr < count; ++nbr) {
    const size_t edge = static_cast<size_t>(nbr) * N + i;
    const int j = neighbor_atoms[edge];
    RealT dx;
    RealT dy;
    RealT dz;
    load_sh_edge_displacement(
      use_cached_displacements, N, box, edge, i, j, neighbor_dx, neighbor_dy, neighbor_dz,
      x, y, z, dx, dy, dz);
    const RealT r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= static_cast<RealT>(cutoff_square)) {
      continue;
    }
    const RealT r = sqrt(r2);
    const int type_j = type[j];
    const int pair = type_i * model.species_count + type_j;
    RealT radial_vals[kMaxSHRadialFuncs];
    RealT sh_vals[kMaxSHComponents];
    sh_direct_radial_vals_ders(model, pair, r, radial_vals, static_cast<RealT*>(nullptr));
    eval_real_sh(dx, dy, dz, r, model.sh_l_max, sh_vals, static_cast<RealT*>(nullptr));
    for (int b = 0; b < model.alpha_basic_count; ++b) {
      const int mu = model.alpha_basic_mu_yidx[b * 2 + 0];
      const int yidx = model.alpha_basic_mu_yidx[b * 2 + 1];
      basic[b] += radial_vals[mu] * sh_vals[yidx];
    }
  }

  for (int b = 0; b < model.alpha_basic_count; ++b) {
    moments[static_cast<size_t>(b) * N + i] = basic[b];
  }
}

template <typename RealT, typename GradT>
static __global__ void gpu_sh_forward_energy_backward(
  int N,
  SHDeviceModel model,
  const int* type,
  RealT* moments,
  GradT* grads,
  double* potential)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  for (int p = 0; p < model.sh_product_count; ++p) {
    const int left = model.sh_products_int[p * 3 + 0];
    const int right = model.sh_products_int[p * 3 + 1];
    const int target = model.sh_products_int[p * 3 + 2];
    const RealT coeff = sh_product_coeff<RealT>(model, p);
    moments[static_cast<size_t>(target) * N + i] +=
      coeff * moments[static_cast<size_t>(left) * N + i] *
      moments[static_cast<size_t>(right) * N + i];
  }

  const int type_i = type[i];
  const RealT center_coeff = sh_species_coeff<RealT>(model, type_i);
  RealT site_energy = sh_shift_coeff<RealT>(model, type_i) + center_coeff;
  for (int s = 0; s < model.alpha_scalar_moments; ++s) {
    const int moment_id = model.alpha_moment_mapping[s];
    const RealT coeff = sh_moment_coeff<RealT>(model, s);
    site_energy += center_coeff * coeff * moments[static_cast<size_t>(moment_id) * N + i];
    grads[static_cast<size_t>(moment_id) * N + i] += static_cast<GradT>(center_coeff * coeff);
  }
  potential[i] += static_cast<double>(site_energy);

  for (int p = model.sh_product_count - 1; p >= 0; --p) {
    const int left = model.sh_products_int[p * 3 + 0];
    const int right = model.sh_products_int[p * 3 + 1];
    const int target = model.sh_products_int[p * 3 + 2];
    const RealT coeff = sh_product_coeff<RealT>(model, p);
    const RealT gtarget =
      static_cast<RealT>(grads[static_cast<size_t>(target) * N + i]) * coeff;
    grads[static_cast<size_t>(left) * N + i] +=
      static_cast<GradT>(gtarget * moments[static_cast<size_t>(right) * N + i]);
    grads[static_cast<size_t>(right) * N + i] +=
      static_cast<GradT>(gtarget * moments[static_cast<size_t>(left) * N + i]);
  }
}

template <typename GradT, typename RealT>
__device__ __forceinline__ void compute_sh_edge_derivative(
  int N,
  SHDeviceModel model,
  int center_atom,
  int center_type,
  int neighbor_type,
  RealT dx,
  RealT dy,
  RealT dz,
  RealT r,
  const GradT* grads,
  RealT& dEx,
  RealT& dEy,
  RealT& dEz)
{
  const int pair = center_type * model.species_count + neighbor_type;
  RealT radial_vals[kMaxSHRadialFuncs];
  RealT radial_ders[kMaxSHRadialFuncs];
  RealT sh_vals[kMaxSHComponents];
  RealT sh_ders[3 * kMaxSHComponents];
  sh_direct_radial_vals_ders(model, pair, r, radial_vals, radial_ders);
  eval_real_sh(dx, dy, dz, r, model.sh_l_max, sh_vals, sh_ders);
  const RealT inv_r = static_cast<RealT>(1.0) / r;
  dEx = dEy = dEz = static_cast<RealT>(0.0);
  for (int b = 0; b < model.alpha_basic_count; ++b) {
    const RealT adj = static_cast<RealT>(grads[static_cast<size_t>(b) * N + center_atom]);
    if (adj == static_cast<RealT>(0.0)) {
      continue;
    }
    const int mu = model.alpha_basic_mu_yidx[b * 2 + 0];
    const int yidx = model.alpha_basic_mu_yidx[b * 2 + 1];
    const RealT radial = radial_vals[mu];
    const RealT radial_der = radial_ders[mu];
    const RealT ylm = sh_vals[yidx];
    dEx += adj * (radial_der * dx * inv_r * ylm + radial * sh_ders[3 * yidx + 0]);
    dEy += adj * (radial_der * dy * inv_r * ylm + radial * sh_ders[3 * yidx + 1]);
    dEz += adj * (radial_der * dz * inv_r * ylm + radial * sh_ders[3 * yidx + 2]);
  }
}

template <typename GradT, typename RealT>
__device__ __forceinline__ void compute_sh_edge_derivative_cached_grads(
  SHDeviceModel model,
  int center_type,
  int neighbor_type,
  RealT dx,
  RealT dy,
  RealT dz,
  RealT r,
  const GradT* grad_cache,
  RealT& dEx,
  RealT& dEy,
  RealT& dEz)
{
  const int pair = center_type * model.species_count + neighbor_type;
  RealT radial_vals[kMaxSHRadialFuncs];
  RealT radial_ders[kMaxSHRadialFuncs];
  RealT sh_vals[kMaxSHComponents];
  RealT sh_ders[3 * kMaxSHComponents];
  sh_direct_radial_vals_ders(model, pair, r, radial_vals, radial_ders);
  eval_real_sh(dx, dy, dz, r, model.sh_l_max, sh_vals, sh_ders);
  const RealT inv_r = static_cast<RealT>(1.0) / r;
  dEx = dEy = dEz = static_cast<RealT>(0.0);
  for (int b = 0; b < model.alpha_basic_count; ++b) {
    const RealT adj = static_cast<RealT>(grad_cache[b]);
    if (adj == static_cast<RealT>(0.0)) {
      continue;
    }
    const int mu = model.alpha_basic_mu_yidx[b * 2 + 0];
    const int yidx = model.alpha_basic_mu_yidx[b * 2 + 1];
    const RealT radial = radial_vals[mu];
    const RealT radial_der = radial_ders[mu];
    const RealT ylm = sh_vals[yidx];
    dEx += adj * (radial_der * dx * inv_r * ylm + radial * sh_ders[3 * yidx + 0]);
    dEy += adj * (radial_der * dy * inv_r * ylm + radial * sh_ders[3 * yidx + 1]);
    dEz += adj * (radial_der * dz * inv_r * ylm + radial * sh_ders[3 * yidx + 2]);
  }
}

template <typename GradT, typename RealT>
static __global__ void gpu_sh_compute_forces(
  int N,
  Box box,
  double cutoff_square,
  bool use_cached_displacements,
  SHDeviceModel model,
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
  float* force_self_tmp,
  float* virial_tmp)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  const int type_i = type[i];
  const int count = neighbor_count[i];
  RealT fx_self = static_cast<RealT>(0.0);
  RealT fy_self = static_cast<RealT>(0.0);
  RealT fz_self = static_cast<RealT>(0.0);
  RealT s_xx = static_cast<RealT>(0.0);
  RealT s_yy = static_cast<RealT>(0.0);
  RealT s_zz = static_cast<RealT>(0.0);
  RealT s_xy = static_cast<RealT>(0.0);
  RealT s_xz = static_cast<RealT>(0.0);
  RealT s_yz = static_cast<RealT>(0.0);
  RealT s_yx = static_cast<RealT>(0.0);
  RealT s_zx = static_cast<RealT>(0.0);
  RealT s_zy = static_cast<RealT>(0.0);

  for (int nbr = 0; nbr < count; ++nbr) {
    const size_t edge = static_cast<size_t>(nbr) * N + i;
    const int j = neighbor_atoms[edge];
    RealT dx;
    RealT dy;
    RealT dz;
    load_sh_edge_displacement(
      use_cached_displacements, N, box, edge, i, j, neighbor_dx, neighbor_dy, neighbor_dz,
      x, y, z, dx, dy, dz);
    const RealT r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= static_cast<RealT>(cutoff_square)) {
      continue;
    }
    const RealT r = sqrt(r2);
    const int type_j = type[j];
    RealT dEx;
    RealT dEy;
    RealT dEz;
    compute_sh_edge_derivative<GradT, RealT>(
      N, model, i, type_i, type_j, dx, dy, dz, r, grads, dEx, dEy, dEz);

    fx_self += dEx;
    fy_self += dEy;
    fz_self += dEz;
    atomicAdd(force_tmp + j, static_cast<float>(-dEx));
    atomicAdd(force_tmp + j + N, static_cast<float>(-dEy));
    atomicAdd(force_tmp + j + 2 * N, static_cast<float>(-dEz));

    s_xx -= dEx * dx;
    s_yy -= dEy * dy;
    s_zz -= dEz * dz;
    s_xy -= dx * dEy;
    s_xz -= dx * dEz;
    s_yz -= dy * dEz;
    s_yx -= dy * dEx;
    s_zx -= dz * dEx;
    s_zy -= dz * dEy;
  }

  if (force_self_tmp != nullptr) {
    force_self_tmp[i] = static_cast<float>(fx_self);
    force_self_tmp[i + N] = static_cast<float>(fy_self);
    force_self_tmp[i + 2 * N] = static_cast<float>(fz_self);
  } else {
    atomicAdd(force_tmp + i, static_cast<float>(fx_self));
    atomicAdd(force_tmp + i + N, static_cast<float>(fy_self));
    atomicAdd(force_tmp + i + 2 * N, static_cast<float>(fz_self));
  }

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

template <typename GradT, typename RealT, int CacheBasics>
static __global__ void gpu_sh_compute_forces_cached_grads(
  int N,
  Box box,
  double cutoff_square,
  bool use_cached_displacements,
  SHDeviceModel model,
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
  float* force_self_tmp,
  float* virial_tmp)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  GradT grad_cache[CacheBasics];
  for (int b = 0; b < model.alpha_basic_count; ++b) {
    grad_cache[b] = grads[static_cast<size_t>(b) * N + i];
  }

  const int type_i = type[i];
  const int count = neighbor_count[i];
  RealT fx_self = static_cast<RealT>(0.0);
  RealT fy_self = static_cast<RealT>(0.0);
  RealT fz_self = static_cast<RealT>(0.0);
  RealT s_xx = static_cast<RealT>(0.0);
  RealT s_yy = static_cast<RealT>(0.0);
  RealT s_zz = static_cast<RealT>(0.0);
  RealT s_xy = static_cast<RealT>(0.0);
  RealT s_xz = static_cast<RealT>(0.0);
  RealT s_yz = static_cast<RealT>(0.0);
  RealT s_yx = static_cast<RealT>(0.0);
  RealT s_zx = static_cast<RealT>(0.0);
  RealT s_zy = static_cast<RealT>(0.0);

  for (int nbr = 0; nbr < count; ++nbr) {
    const size_t edge = static_cast<size_t>(nbr) * N + i;
    const int j = neighbor_atoms[edge];
    RealT dx;
    RealT dy;
    RealT dz;
    load_sh_edge_displacement(
      use_cached_displacements, N, box, edge, i, j, neighbor_dx, neighbor_dy, neighbor_dz,
      x, y, z, dx, dy, dz);
    const RealT r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= static_cast<RealT>(cutoff_square)) {
      continue;
    }
    const RealT r = sqrt(r2);
    const int type_j = type[j];
    RealT dEx;
    RealT dEy;
    RealT dEz;
    compute_sh_edge_derivative_cached_grads<GradT, RealT>(
      model, type_i, type_j, dx, dy, dz, r, grad_cache, dEx, dEy, dEz);

    fx_self += dEx;
    fy_self += dEy;
    fz_self += dEz;
    atomicAdd(force_tmp + j, static_cast<float>(-dEx));
    atomicAdd(force_tmp + j + N, static_cast<float>(-dEy));
    atomicAdd(force_tmp + j + 2 * N, static_cast<float>(-dEz));

    s_xx -= dEx * dx;
    s_yy -= dEy * dy;
    s_zz -= dEz * dz;
    s_xy -= dx * dEy;
    s_xz -= dx * dEz;
    s_yz -= dy * dEz;
    s_yx -= dy * dEx;
    s_zx -= dz * dEx;
    s_zy -= dz * dEy;
  }

  if (force_self_tmp != nullptr) {
    force_self_tmp[i] = static_cast<float>(fx_self);
    force_self_tmp[i + N] = static_cast<float>(fy_self);
    force_self_tmp[i + 2 * N] = static_cast<float>(fz_self);
  } else {
    atomicAdd(force_tmp + i, static_cast<float>(fx_self));
    atomicAdd(force_tmp + i + N, static_cast<float>(fy_self));
    atomicAdd(force_tmp + i + 2 * N, static_cast<float>(fz_self));
  }

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
  const float* force_self_tmp,
  const float* virial_tmp,
  double* force,
  double* virial)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }
  double fx = static_cast<double>(force_tmp[i]);
  double fy = static_cast<double>(force_tmp[i + N]);
  double fz = static_cast<double>(force_tmp[i + 2 * N]);
  if (force_self_tmp != nullptr) {
    fx += static_cast<double>(force_self_tmp[i]);
    fy += static_cast<double>(force_self_tmp[i + N]);
    fz += static_cast<double>(force_self_tmp[i + 2 * N]);
  }
  force[i] += fx;
  force[i + N] += fy;
  force[i + 2 * N] += fz;
  for (int k = 0; k < 9; ++k) {
    virial[i + k * N] += static_cast<double>(virial_tmp[i + k * N]);
  }
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

} // namespace

bool is_sus2_sh_potential_file(const char* file_potential)
{
  std::ifstream ifs(file_potential);
  if (!ifs) {
    return false;
  }
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return has_token(oss.str(), "potential_tag = SUS2-SH");
}

SUS2_SH::SUS2_SH(
  const char* file_potential,
  int num_atoms,
  int num_potential_options,
  const char** potential_options)
{
  for (int i = 0; i < sh_profile_count; ++i) {
    profile_ms_[i] = 0.0;
  }
  const SHHostModel host_model = load_sh_model(file_potential);
  species_count_ = host_model.species_count;
  sh_l_max_ = host_model.sh_l_max;
  sh_k_max_ = host_model.sh_k_max;
  radial_funcs_count_ = host_model.radial_funcs_count;
  rb_size_ = host_model.rb_size;
  alpha_basic_count_ = host_model.alpha_basic_count;
  sh_product_count_ = static_cast<int>(host_model.products.size());
  alpha_moments_count_ = host_model.alpha_moments_count;
  alpha_scalar_moments_ = host_model.alpha_scalar_moments;
  rc = host_model.max_dist;
  use_float_moments_ = parse_sh_float(host_model, num_potential_options, potential_options);
  use_radial_direct_ =
    parse_sh_radial_direct(host_model, num_potential_options, potential_options);
  use_force_self_buffer_ =
    parse_sh_force_self_buffer(host_model, num_potential_options, potential_options);
  use_force_grad_cache_ =
    parse_sh_force_grad_cache(host_model, num_potential_options, potential_options) &&
    alpha_basic_count_ <= kSHForceGradCache64;
  profile_enabled_ = parse_sh_profile_enabled(host_model, num_potential_options, potential_options);
  profile_interval_ =
    parse_sh_profile_interval(host_model, num_potential_options, potential_options);
  if (!use_radial_direct_) {
    sh_input_error("SUS2_SH first backend currently requires direct radial evaluation.");
  }

  shift_coeffs_.resize(host_model.shift_coeffs.size());
  shift_coeffs_.copy_from_host(host_model.shift_coeffs.data());
  species_coeffs_.resize(host_model.species_coeffs.size());
  species_coeffs_.copy_from_host(host_model.species_coeffs.data());
  moment_coeffs_.resize(host_model.moment_coeffs.size());
  moment_coeffs_.copy_from_host(host_model.moment_coeffs.data());
  if (use_float_moments_) {
    std::vector<float> shift_f(host_model.shift_coeffs.begin(), host_model.shift_coeffs.end());
    std::vector<float> species_f(host_model.species_coeffs.begin(), host_model.species_coeffs.end());
    std::vector<float> moment_f(host_model.moment_coeffs.begin(), host_model.moment_coeffs.end());
    shift_coeffs_float_.resize(shift_f.size());
    species_coeffs_float_.resize(species_f.size());
    moment_coeffs_float_.resize(moment_f.size());
    shift_coeffs_float_.copy_from_host(shift_f.data());
    species_coeffs_float_.copy_from_host(species_f.data());
    moment_coeffs_float_.copy_from_host(moment_f.data());
  }

  alpha_basic_.resize(host_model.alpha_basic.size());
  alpha_basic_.copy_from_host(host_model.alpha_basic.data());
  std::vector<int> basic_mu_yidx(static_cast<size_t>(alpha_basic_count_) * 2);
  for (int b = 0; b < alpha_basic_count_; ++b) {
    const int mu = host_model.alpha_basic[b * 3 + 0];
    const int l = host_model.alpha_basic[b * 3 + 1];
    const int m = host_model.alpha_basic[b * 3 + 2];
    basic_mu_yidx[b * 2 + 0] = mu;
    basic_mu_yidx[b * 2 + 1] = l * l + (m + l);
  }
  alpha_basic_mu_yidx_.resize(basic_mu_yidx.size());
  alpha_basic_mu_yidx_.copy_from_host(basic_mu_yidx.data());
  std::vector<int> product_ints(static_cast<size_t>(sh_product_count_) * 3);
  std::vector<double> product_coeffs(sh_product_count_);
  for (int p = 0; p < sh_product_count_; ++p) {
    product_ints[p * 3 + 0] = host_model.products[p].left;
    product_ints[p * 3 + 1] = host_model.products[p].right;
    product_ints[p * 3 + 2] = host_model.products[p].target;
    product_coeffs[p] = host_model.products[p].coeff;
  }
  sh_products_int_.resize(product_ints.size());
  sh_products_int_.copy_from_host(product_ints.data());
  sh_products_coeff_.resize(product_coeffs.size());
  sh_products_coeff_.copy_from_host(product_coeffs.data());
  if (use_float_moments_) {
    std::vector<float> product_coeffs_f(product_coeffs.begin(), product_coeffs.end());
    sh_products_coeff_float_.resize(product_coeffs_f.size());
    sh_products_coeff_float_.copy_from_host(product_coeffs_f.data());
  }
  alpha_moment_mapping_.resize(host_model.alpha_moment_mapping.size());
  alpha_moment_mapping_.copy_from_host(host_model.alpha_moment_mapping.data());

  std::vector<float> radial_coeffs;
  std::vector<float> radial_scal_s;
  build_direct_radial_tables(host_model, radial_coeffs, radial_scal_s);
  radial_direct_coeffs_.resize(radial_coeffs.size());
  radial_direct_scal_s_.resize(radial_scal_s.size());
  radial_direct_coeffs_.copy_from_host(radial_coeffs.data());
  radial_direct_scal_s_.copy_from_host(radial_scal_s.data());

  neighbor_count_.resize(num_atoms);
  cell_contents_.resize(num_atoms);
  neighbor_cache_.initialize(rc, num_atoms, 512);
  resize_work_buffers(num_atoms);

  printf(
    "Use SUS2-SH GPUMD potential: radial_type=%s, species=%d, sh_l_max=%d, sh_k_max=%d, basics=%d, products=%d, moments=%d, scalars=%d, cutoff=%g A, radial_eval=direct basis recurrence.\n",
    host_model.radial_basis_type.c_str(),
    species_count_,
    sh_l_max_,
    sh_k_max_,
    alpha_basic_count_,
    sh_product_count_,
    alpha_moments_count_,
    alpha_scalar_moments_,
    rc);
  printf(
    "SUS2-SH GPUMD precision mode: %s; force self-buffer: %s; force basic-grad cache: %s.\n",
    use_float_moments_ ? "NEP-like float moments/gradients/local arithmetic" : "double moments/local arithmetic",
    use_force_self_buffer_ ? "on" : "off",
    use_force_grad_cache_ ? "on" : "off");
  if (profile_enabled_) {
    printf("SUS2-SH GPUMD profile enabled; interval=%d steps.\n", profile_interval_);
  }
}

SUS2_SH::~SUS2_SH(void) {}

void SUS2_SH::resize_work_buffers(int num_atoms)
{
  const size_t moment_size = static_cast<size_t>(alpha_moments_count_) * num_atoms;
  const size_t force_size = static_cast<size_t>(num_atoms) * 3;
  const size_t virial_size = static_cast<size_t>(num_atoms) * 9;
  if (use_float_moments_) {
    if (moment_vals_float_.size() != moment_size) {
      moment_vals_float_.resize(moment_size);
    }
    if (moment_grads_float_.size() != moment_size) {
      moment_grads_float_.resize(moment_size);
    }
  } else {
    if (moment_vals_.size() != moment_size) {
      moment_vals_.resize(moment_size);
    }
    if (moment_grads_.size() != moment_size) {
      moment_grads_.resize(moment_size);
    }
  }
  if (force_tmp_.size() != force_size) {
    force_tmp_.resize(force_size);
  }
  if (use_force_self_buffer_ && force_self_tmp_.size() != force_size) {
    force_self_tmp_.resize(force_size);
  }
  if (virial_tmp_.size() != virial_size) {
    virial_tmp_.resize(virial_size);
  }
}

void SUS2_SH::build_neighbor_list(
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
    neighbor_cache_.find_neighbor_global(rc, box, type, position);
    neighbor_cache_.find_local_neighbor_from_global(rc, box, position, neighbor_count_, neighbor_atom_);
    return;
  }

  use_cached_neighbor_displacements_ = true;
  gpu_count_neighbors_images_on2<<<grid_size, kBlockSize>>>(
    num_atoms, box, -sx_range, sx_range, -sy_range, sy_range, -sz_range, sz_range,
    cutoff_square, x, y, z, neighbor_count_.data());
  GPU_CHECK_KERNEL

  int max_neighbors = 0;
  int* max_ptr = thrust::max_element(
    thrust::device, neighbor_count_.data(), neighbor_count_.data() + num_atoms);
  CHECK(gpuMemcpy(&max_neighbors, max_ptr, sizeof(int), gpuMemcpyDeviceToHost));
  const int alloc_neighbors = std::max(max_neighbors, 1);
  const size_t edge_capacity = static_cast<size_t>(num_atoms) * alloc_neighbors;
  if (neighbor_atom_.size() != edge_capacity) {
    neighbor_atom_.resize(edge_capacity);
  }
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
    num_atoms, box, -sx_range, sx_range, -sy_range, sy_range, -sz_range, sz_range,
    cutoff_square, x, y, z, neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
    neighbor_dz_.data());
  GPU_CHECK_KERNEL
}

void SUS2_SH::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial)
{
  const int num_atoms = static_cast<int>(type.size());
  resize_work_buffers(num_atoms);

  using Clock = std::chrono::high_resolution_clock;
  auto profile_start = [&]() {
    if (profile_enabled_) {
      CHECK(gpuDeviceSynchronize());
    }
    return Clock::now();
  };
  auto profile_stop = [&](int stage, Clock::time_point start) {
    if (profile_enabled_) {
      CHECK(gpuDeviceSynchronize());
      const auto stop = Clock::now();
      profile_ms_[stage] +=
        std::chrono::duration<double, std::milli>(stop - start).count();
    }
  };

  auto stage_start = profile_start();
  build_neighbor_list(box, type, position, num_atoms);
  profile_stop(sh_profile_neighbor, stage_start);

  const int grid_size = (num_atoms - 1) / kBlockSize + 1;
  const size_t moment_size = static_cast<size_t>(alpha_moments_count_) * num_atoms;
  const size_t force_size = static_cast<size_t>(num_atoms) * 3;
  const size_t virial_size = static_cast<size_t>(num_atoms) * 9;
  stage_start = profile_start();
  if (use_float_moments_) {
    CHECK(gpuMemset(moment_vals_float_.data(), 0, moment_size * sizeof(float)));
    CHECK(gpuMemset(moment_grads_float_.data(), 0, moment_size * sizeof(float)));
  } else {
    CHECK(gpuMemset(moment_vals_.data(), 0, moment_size * sizeof(double)));
    CHECK(gpuMemset(moment_grads_.data(), 0, moment_size * sizeof(double)));
  }
  CHECK(gpuMemset(force_tmp_.data(), 0, force_size * sizeof(float)));
  if (use_force_self_buffer_) {
    CHECK(gpuMemset(force_self_tmp_.data(), 0, force_size * sizeof(float)));
  }
  CHECK(gpuMemset(virial_tmp_.data(), 0, virial_size * sizeof(float)));
  profile_stop(sh_profile_memset, stage_start);

  SHDeviceModel model{
    species_count_,
    sh_l_max_,
    radial_funcs_count_,
    rb_size_,
    alpha_basic_count_,
    sh_product_count_,
    alpha_moments_count_,
    alpha_scalar_moments_,
    rc,
    shift_coeffs_.data(),
    species_coeffs_.data(),
    moment_coeffs_.data(),
    use_float_moments_ ? shift_coeffs_float_.data() : nullptr,
    use_float_moments_ ? species_coeffs_float_.data() : nullptr,
    use_float_moments_ ? moment_coeffs_float_.data() : nullptr,
    alpha_basic_.data(),
    alpha_basic_mu_yidx_.data(),
    sh_products_int_.data(),
    sh_products_coeff_.data(),
    use_float_moments_ ? sh_products_coeff_float_.data() : nullptr,
    alpha_moment_mapping_.data(),
    radial_direct_coeffs_.data(),
    radial_direct_scal_s_.data(),
    use_float_moments_};

  float* force_self_tmp_ptr = use_force_self_buffer_ ? force_self_tmp_.data() : nullptr;
  if (use_float_moments_) {
    stage_start = profile_start();
    gpu_sh_compute_basic<float><<<grid_size, kBlockSize>>>(
      num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
      neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
      neighbor_dz_.data(), position.data(), position.data() + num_atoms,
      position.data() + 2 * num_atoms, moment_vals_float_.data());
    GPU_CHECK_KERNEL
    profile_stop(sh_profile_basic, stage_start);
    stage_start = profile_start();
    gpu_sh_forward_energy_backward<float, float><<<grid_size, kBlockSize>>>(
      num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
      potential.data());
    GPU_CHECK_KERNEL
    profile_stop(sh_profile_product, stage_start);
    stage_start = profile_start();
    if (use_force_grad_cache_) {
      gpu_sh_compute_forces_cached_grads<float, float, kSHForceGradCache64><<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, moment_grads_float_.data(), force_tmp_.data(),
        force_self_tmp_ptr, virial_tmp_.data());
    } else {
      gpu_sh_compute_forces<float, float><<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, moment_grads_float_.data(), force_tmp_.data(),
        force_self_tmp_ptr, virial_tmp_.data());
    }
    GPU_CHECK_KERNEL
    profile_stop(sh_profile_force, stage_start);
  } else {
    stage_start = profile_start();
    gpu_sh_compute_basic<double><<<grid_size, kBlockSize>>>(
      num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
      neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
      neighbor_dz_.data(), position.data(), position.data() + num_atoms,
      position.data() + 2 * num_atoms, moment_vals_.data());
    GPU_CHECK_KERNEL
    profile_stop(sh_profile_basic, stage_start);
    stage_start = profile_start();
    gpu_sh_forward_energy_backward<double, double><<<grid_size, kBlockSize>>>(
      num_atoms, model, type.data(), moment_vals_.data(), moment_grads_.data(),
      potential.data());
    GPU_CHECK_KERNEL
    profile_stop(sh_profile_product, stage_start);
    stage_start = profile_start();
    if (use_force_grad_cache_) {
      gpu_sh_compute_forces_cached_grads<double, double, kSHForceGradCache64><<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, moment_grads_.data(), force_tmp_.data(),
        force_self_tmp_ptr, virial_tmp_.data());
    } else {
      gpu_sh_compute_forces<double, double><<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, moment_grads_.data(), force_tmp_.data(),
        force_self_tmp_ptr, virial_tmp_.data());
    }
    GPU_CHECK_KERNEL
    profile_stop(sh_profile_force, stage_start);
  }

  stage_start = profile_start();
  gpu_accumulate_float_to_double<<<grid_size, kBlockSize>>>(
    num_atoms, force_tmp_.data(), force_self_tmp_ptr, virial_tmp_.data(), force.data(), virial.data());
  GPU_CHECK_KERNEL
  profile_stop(sh_profile_accumulate, stage_start);

  if (profile_enabled_) {
    ++profile_steps_;
    if (profile_steps_ >= profile_interval_) {
      const double inv = 1.0 / static_cast<double>(profile_steps_);
      const double neighbor_ms = profile_ms_[sh_profile_neighbor] * inv;
      const double memset_ms = profile_ms_[sh_profile_memset] * inv;
      const double basic_ms = profile_ms_[sh_profile_basic] * inv;
      const double product_ms = profile_ms_[sh_profile_product] * inv;
      const double force_ms = profile_ms_[sh_profile_force] * inv;
      const double accumulate_ms = profile_ms_[sh_profile_accumulate] * inv;
      const double total_ms =
        neighbor_ms + memset_ms + basic_ms + product_ms + force_ms + accumulate_ms;
      printf(
        "SUS2-SH profile avg over %d steps: neighbor=%.3f ms, memset=%.3f ms, basic=%.3f ms, product=%.3f ms, force=%.3f ms, accumulate=%.3f ms, total=%.3f ms.\n",
        profile_steps_,
        neighbor_ms,
        memset_ms,
        basic_ms,
        product_ms,
        force_ms,
        accumulate_ms,
        total_ms);
      profile_steps_ = 0;
      for (int i = 0; i < sh_profile_count; ++i) {
        profile_ms_[i] = 0.0;
      }
    }
  }
}
