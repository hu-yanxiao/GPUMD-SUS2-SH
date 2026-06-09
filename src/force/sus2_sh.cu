#include "sus2_sh.cuh"

#include "sus2_zbl_common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace
{
constexpr int kBlockSize = 128;
constexpr int kProductBlockSize = 128;
constexpr int kMaxSHL = 6;
constexpr int kMaxSHComponents = (kMaxSHL + 1) * (kMaxSHL + 1);
constexpr int kMaxSHRadialFuncs = 48;
constexpr int kMaxSHRbSize = 16;
constexpr int kMaxSHBasics = 320;
constexpr int kSHProductBasicCache = 64;
constexpr int kSHForceGradCache64 = 64;
constexpr int kSHForceGradCache128 = 128;
constexpr int kSHForceGradCache256 = 256;
constexpr int kSHMaxConstForwardU32 = 16384;
constexpr int kSHTerminalDotGroupRowTermThreshold = 1000;
constexpr int kSHTerminalDotGroupMaxCount = 17;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kLaguerreMinRho = 1.0e-8;

__constant__ unsigned int c_sh_forward_u32[kSHMaxConstForwardU32];

enum class SHRadialBasisKind {
  ChebyshevSSS = 0,
  LaguerreLog1p = 1
};

enum class SHFactorPruningMode {
  Legacy = 0,
  QTotal = 1
};

enum SHProfileStage {
  sh_profile_neighbor = 0,
  sh_profile_memset = 1,
  sh_profile_basic = 2,
  sh_profile_product = 3,
  sh_profile_force = 4,
  sh_profile_accumulate = 5,
  sh_profile_product_forward = 6,
  sh_profile_product_back = 7,
  sh_profile_count = 8
};

struct SHProductHost {
  int left = 0;
  int right = 0;
  int target = 0;
  double coeff = 0.0;
};

struct SHBasicIndexHost {
  int k = 0;
  int l = 0;
  int m = 0;
};

struct SHQIndexHost {
  int l = 0;
  int k = 0;
  int mu = 0;
  std::string key;
};

struct SHScalarSpecHost {
  int body_order = 0;
  int q0 = -1;
  int q1 = -1;
  int q2 = -1;
  int q3 = -1;
  int q4 = -1;
  int intermediate_l = 0;
};

struct SHTensorHost {
  std::string key;
  int l = 0;
  int block = -1;
  int layer = 0;
  std::vector<int> node;
  bool zero = false;
};

struct SHLocalProductHost {
  int left = 0;
  int right = 0;
  int target_component = 0;
  double coeff = 0.0;
};

struct SHCGTermHost {
  int left_component = 0;
  int right_component = 0;
  int target_component = 0;
  double coeff = 0.0;
};

struct SHCGRowTermHost {
  int left_component = 0;
  int right_component = 0;
  double coeff = 0.0;
};

struct SHCGRowHost {
  int layer = 0;
  int left_base = 0;
  int right_base = 0;
  int target = 0;
  int term_begin = 0;
  int term_count = 0;
};

struct SHCGBackTermHost {
  int target = 0;
  int other = 0;
  double coeff = 0.0;
};

struct SHCGBackRowHost {
  int layer = 0;
  int source = 0;
  int term_begin = 0;
  int term_count = 0;
};

struct SHTensorBlockHost {
  int kind = 0;
  int layer = 0;
  int base = 0;
  int l = 0;
  int k = -1;
  int mu = -1;
  int left_block = -1;
  int right_block = -1;
};

struct SHCGBlockHost {
  int layer = 0;
  int left_block = -1;
  int right_block = -1;
  int target_block = -1;
  int left_base = 0;
  int right_base = 0;
  int target_base = 0;
  int l1 = 0;
  int l2 = 0;
  int L = 0;
  int term_begin = 0;
  int term_count = 0;
};

struct SHStandardGraphHost {
  int node_count = 0;
  int max_layer = 0;
  std::vector<SHBasicIndexHost> basic;
  std::vector<SHProductHost> products;
  std::vector<int> scalars;
  std::vector<SHTensorBlockHost> tensor_blocks;
  std::vector<SHCGBlockHost> cg_blocks;
  std::vector<SHCGTermHost> cg_terms;
};

struct SHHostModel {
  int species_count = 0;
  int sh_l_max = 0;
  int sh_k_max = 0;
  int sh_body_order = 0;
  int sh_standard_tensor_blocks = 0;
  int sh_standard_cg_blocks = 0;
  int sh_standard_cg_terms = 0;
  int sh_standard_cg_layers = 0;
  bool sh_standard_graph_matched = false;
  int radial_funcs_count = 0;
  int rb_size = 0;
  int alpha_basic_count = 0;
  int alpha_moments_count = 0;
  int alpha_scalar_moments = 0;
  double scaling = 1.0;
  double max_dist = 0.0;
  bool zbl_enabled = false;
  double zbl_inner = 0.7;
  double zbl_outer = 1.4;
  bool zbl_typewise_cutoff_enabled = false;
  double zbl_typewise_cutoff_factor = 0.7;
  bool two_layer_gate_enabled = false;
  double two_layer_gate_tanh_amplitude = 0.8;
  int two_layer_gate_weight_count = 0;
  int two_layer_gate_product_limit = 0;
  SHRadialBasisKind radial_basis_kind = SHRadialBasisKind::ChebyshevSSS;
  std::string potential_tag;
  std::string radial_basis_type;
  std::string scaling_map;
  std::vector<double> shift_coeffs;
  std::vector<double> scal_coeffs;
  std::vector<double> radial_coeffs;
  std::vector<double> two_layer_gate_radial_coeffs;
  std::vector<double> two_layer_gate_additive_coeffs;
  std::vector<double> two_layer_gate_weights;
  std::vector<double> radial_type_coeffs;
  std::vector<double> species_coeffs;
  std::vector<double> moment_coeffs;
  std::vector<int> two_layer_gate_scalar_indices;
  std::vector<int> two_layer_gate_moment_indices;
  std::vector<int> two_layer_gate_needed_moment_flags;
  std::vector<float> two_layer_gate_moment_weights_float;
  std::vector<int> zbl_atomic_numbers;
  std::vector<double> zbl_pair_inner_cutoffs;
  std::vector<double> zbl_pair_outer_cutoffs;
  std::vector<double> zbl_pair_outer_sq;
  std::vector<int> sh_body_l_max;
  std::vector<int> alpha_basic;
  std::vector<SHProductHost> products;
  std::vector<int> alpha_moment_mapping;
  std::vector<SHCGBlockHost> cg_blocks;
  std::vector<SHCGTermHost> cg_terms;
  std::vector<SHCGRowHost> cg_rows;
  std::vector<SHCGRowTermHost> cg_row_terms;
  std::vector<int> cg_layer_offsets;
  std::vector<SHCGBackRowHost> cg_back_rows;
  std::vector<SHCGBackTermHost> cg_back_terms;
  std::vector<int> cg_back_layer_offsets;
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

int parse_optional_int_after(const std::string& text, const std::string& token, int default_value)
{
  return has_token(text, token) ? parse_int_after(text, token) : default_value;
}

double parse_optional_double_after(
  const std::string& text,
  const std::string& token,
  double default_value)
{
  return has_token(text, token) ? parse_double_after(text, token) : default_value;
}

bool parse_optional_bool_after(
  const std::string& text,
  const std::string& token,
  bool default_value)
{
  if (!has_token(text, token)) {
    return default_value;
  }
  const std::string value = parse_string_after(text, token);
  if (value == "1" || value == "true" || value == "TRUE" ||
      value == "yes" || value == "on") {
    return true;
  }
  if (value == "0" || value == "false" || value == "FALSE" ||
      value == "no" || value == "off") {
    return false;
  }
  sh_input_error("Invalid boolean value in SUS2-SH model file for " + token + ": " + value);
}

SHRadialBasisKind sh_radial_basis_kind_from_string(const std::string& type)
{
  if (type == "RBChebyshev_sss" || type == "RBChebyshev_sss_lmp") {
    return SHRadialBasisKind::ChebyshevSSS;
  }
  if (type == "RBLaguerre_log1p" || type == "RBLaguerre_log1p_lmp") {
    return SHRadialBasisKind::LaguerreLog1p;
  }
  sh_input_error(
    "Unsupported SUS2_SH radial_basis_type in GPUMD: " + type +
    ". Supported now: RBChebyshev_sss[_lmp], RBLaguerre_log1p[_lmp].");
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

SHFactorPruningMode parse_sh_factor_pruning(const SHHostModel& model, int nopts, const char** opts)
{
  std::string mode = "legacy";
  const char* env = std::getenv("SUS2_SH_GPUMD_FACTOR_PRUNING");
  if (env == nullptr) {
    env = std::getenv("SUS2_GPUMD_FACTOR_PRUNING");
  }
  if (env != nullptr) {
    mode = env;
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_factor_pruning=") ||
        starts_with(option, "sus2_factor_pruning=") ||
        starts_with(option, "sh_factor_pruning=")) {
      const size_t eq = option.find('=');
      mode = option.substr(eq + 1);
    }
  }
  if (mode == "legacy" || mode == "Legacy" || mode == "LEGACY") {
    return SHFactorPruningMode::Legacy;
  }
  if (mode == "q-total" || mode == "q_total" || mode == "total" ||
      mode == "QTotal" || mode == "QTOTAL") {
    return SHFactorPruningMode::QTotal;
  }
  sh_input_error(
    "Invalid SUS2_SH factor pruning mode: " + mode + " (expected legacy or q-total).");
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
  bool use_cache = model.alpha_basic_count <= kSHForceGradCache64;
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

bool parse_sh_cg_block_forward(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_cg = false;
  const char* env = std::getenv("SUS2_SH_GPUMD_CG_BLOCK_FORWARD");
  if (env != nullptr) {
    use_cg = parse_bool_value(env, "SUS2_SH_GPUMD_CG_BLOCK_FORWARD");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_cg_block_forward=") ||
        starts_with(option, "sus2_cg_block_forward=")) {
      const size_t eq = option.find('=');
      use_cg = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_cg;
}

bool parse_sh_tensor_product_parallel(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_tensor = false;
  const char* env = std::getenv("SUS2_SH_GPUMD_TENSOR_PRODUCT_PARALLEL");
  if (env != nullptr) {
    use_tensor = parse_bool_value(env, "SUS2_SH_GPUMD_TENSOR_PRODUCT_PARALLEL");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_tensor_product_parallel=") ||
        starts_with(option, "sus2_tensor_product_parallel=")) {
      const size_t eq = option.find('=');
      use_tensor = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_tensor;
}

bool parse_sh_compact_serial_product(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_compact = true;
  const char* env = std::getenv("SUS2_SH_GPUMD_COMPACT_SERIAL_PRODUCT");
  if (env != nullptr) {
    use_compact = parse_bool_value(env, "SUS2_SH_GPUMD_COMPACT_SERIAL_PRODUCT");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_compact_serial_product=") ||
        starts_with(option, "sus2_compact_serial_product=")) {
      const size_t eq = option.find('=');
      use_compact = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_compact;
}

bool parse_sh_const_forward_rows(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_const = true;
  const char* env = std::getenv("SUS2_SH_GPUMD_CONST_FORWARD");
  if (env != nullptr) {
    use_const = parse_bool_value(env, "SUS2_SH_GPUMD_CONST_FORWARD");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_const_forward=") ||
        starts_with(option, "sus2_const_forward=")) {
      const size_t eq = option.find('=');
      use_const = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_const;
}

bool parse_sh_product_pattern_rows(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_pattern = model.cg_row_terms.size() >= 2500;
  const char* env = std::getenv("SUS2_SH_GPUMD_PRODUCT_PATTERN_ROWS");
  if (env != nullptr) {
    use_pattern = parse_bool_value(env, "SUS2_SH_GPUMD_PRODUCT_PATTERN_ROWS");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_product_pattern_rows=") ||
        starts_with(option, "sus2_product_pattern_rows=")) {
      const size_t eq = option.find('=');
      use_pattern = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_pattern;
}

bool parse_sh_parallel_back_rows(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_parallel = model.cg_back_terms.size() >= 4096;
  const char* env = std::getenv("SUS2_SH_GPUMD_PARALLEL_BACK_ROWS");
  if (env != nullptr) {
    use_parallel = parse_bool_value(env, "SUS2_SH_GPUMD_PARALLEL_BACK_ROWS");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_parallel_back_rows=") ||
        starts_with(option, "sus2_parallel_back_rows=")) {
      const size_t eq = option.find('=');
      use_parallel = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_parallel;
}

bool parse_sh_packed_back_rows(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_packed = true;
  const char* env = std::getenv("SUS2_SH_GPUMD_PACKED_BACK_ROWS");
  if (env != nullptr) {
    use_packed = parse_bool_value(env, "SUS2_SH_GPUMD_PACKED_BACK_ROWS");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_packed_back_rows=") ||
        starts_with(option, "sus2_packed_back_rows=")) {
      const size_t eq = option.find('=');
      use_packed = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_packed;
}

bool parse_sh_const_back_rows(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_const = true;
  const char* env = std::getenv("SUS2_SH_GPUMD_CONST_BACK");
  if (env != nullptr) {
    use_const = parse_bool_value(env, "SUS2_SH_GPUMD_CONST_BACK");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_const_back=") ||
        starts_with(option, "sus2_const_back=")) {
      const size_t eq = option.find('=');
      use_const = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_const;
}

bool parse_sh_static_basic(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_static = true;
  const char* env = std::getenv("SUS2_SH_GPUMD_STATIC_BASIC");
  if (env != nullptr) {
    use_static = parse_bool_value(env, "SUS2_SH_GPUMD_STATIC_BASIC");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_static_basic=") ||
        starts_with(option, "sus2_static_basic=")) {
      const size_t eq = option.find('=');
      use_static = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_static;
}

bool parse_sh_static_force(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_static = true;
  const char* env = std::getenv("SUS2_SH_GPUMD_STATIC_FORCE");
  if (env != nullptr) {
    use_static = parse_bool_value(env, "SUS2_SH_GPUMD_STATIC_FORCE");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_static_force=") ||
        starts_with(option, "sus2_static_force=")) {
      const size_t eq = option.find('=');
      use_static = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_static;
}

bool parse_sh_terminal_scalar_fusion(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_fusion = true;
  const char* env = std::getenv("SUS2_SH_GPUMD_TERMINAL_SCALAR_FUSION");
  if (env != nullptr) {
    use_fusion = parse_bool_value(env, "SUS2_SH_GPUMD_TERMINAL_SCALAR_FUSION");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_terminal_scalar_fusion=") ||
        starts_with(option, "sus2_terminal_scalar_fusion=")) {
      const size_t eq = option.find('=');
      use_fusion = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_fusion;
}

bool parse_sh_row_scalar_fusion(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_fusion = false;
  const char* env = std::getenv("SUS2_SH_GPUMD_ROW_SCALAR_FUSION");
  if (env != nullptr) {
    use_fusion = parse_bool_value(env, "SUS2_SH_GPUMD_ROW_SCALAR_FUSION");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_row_scalar_fusion=") ||
        starts_with(option, "sus2_row_scalar_fusion=")) {
      const size_t eq = option.find('=');
      use_fusion = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_fusion;
}

bool parse_sh_terminal_dot_rows(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_dot = true;
  const char* env = std::getenv("SUS2_SH_GPUMD_TERMINAL_DOT");
  if (env != nullptr) {
    use_dot = parse_bool_value(env, "SUS2_SH_GPUMD_TERMINAL_DOT");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_terminal_dot=") ||
        starts_with(option, "sus2_terminal_dot=")) {
      const size_t eq = option.find('=');
      use_dot = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_dot;
}

bool parse_sh_terminal_dot_groups(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_groups =
    static_cast<int>(model.cg_row_terms.size()) >= kSHTerminalDotGroupRowTermThreshold;
  const char* env = std::getenv("SUS2_SH_GPUMD_TERMINAL_DOT_GROUPS");
  if (env != nullptr) {
    use_groups = parse_bool_value(env, "SUS2_SH_GPUMD_TERMINAL_DOT_GROUPS");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_terminal_dot_groups=") ||
        starts_with(option, "sus2_terminal_dot_groups=")) {
      const size_t eq = option.find('=');
      use_groups = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_groups;
}

bool parse_sh_terminal_dot_premul(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_premul = true;
  const char* env = std::getenv("SUS2_SH_GPUMD_TERMINAL_DOT_PREMUL");
  if (env != nullptr) {
    use_premul = parse_bool_value(env, "SUS2_SH_GPUMD_TERMINAL_DOT_PREMUL");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_terminal_dot_premul=") ||
        starts_with(option, "sus2_terminal_dot_premul=")) {
      const size_t eq = option.find('=');
      use_premul = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_premul;
}

bool parse_sh_terminal_dot_row_list(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_list = true;
  const char* env = std::getenv("SUS2_SH_GPUMD_TERMINAL_DOT_ROW_LIST");
  if (env != nullptr) {
    use_list = parse_bool_value(env, "SUS2_SH_GPUMD_TERMINAL_DOT_ROW_LIST");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_terminal_dot_row_list=") ||
        starts_with(option, "sus2_terminal_dot_row_list=")) {
      const size_t eq = option.find('=');
      use_list = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_list;
}

bool parse_sh_fused_terminal_dot(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_fused = false;
  const char* env = std::getenv("SUS2_SH_GPUMD_FUSED_TERMINAL_DOT");
  if (env != nullptr) {
    use_fused = parse_bool_value(env, "SUS2_SH_GPUMD_FUSED_TERMINAL_DOT");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_fused_terminal_dot=") ||
        starts_with(option, "sus2_fused_terminal_dot=")) {
      const size_t eq = option.find('=');
      use_fused = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_fused;
}

bool parse_sh_selective_grad_zero(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_selective = true;
  const char* env = std::getenv("SUS2_SH_GPUMD_SELECTIVE_GRAD_ZERO");
  if (env != nullptr) {
    use_selective = parse_bool_value(env, "SUS2_SH_GPUMD_SELECTIVE_GRAD_ZERO");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_selective_grad_zero=") ||
        starts_with(option, "sus2_selective_grad_zero=")) {
      const size_t eq = option.find('=');
      use_selective = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_selective;
}

bool parse_sh_product_basic_cache(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_cache = false;
  const char* env = std::getenv("SUS2_SH_GPUMD_PRODUCT_BASIC_CACHE");
  if (env != nullptr) {
    use_cache = parse_bool_value(env, "SUS2_SH_GPUMD_PRODUCT_BASIC_CACHE");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_product_basic_cache=") ||
        starts_with(option, "sus2_product_basic_cache=")) {
      const size_t eq = option.find('=');
      use_cache = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_cache;
}

bool parse_sh_merge_back_duplicates(const SHHostModel& model, int nopts, const char** opts)
{
  bool use_merge = true;
  const char* env = std::getenv("SUS2_SH_GPUMD_MERGE_BACK_DUPLICATES");
  if (env != nullptr) {
    use_merge = parse_bool_value(env, "SUS2_SH_GPUMD_MERGE_BACK_DUPLICATES");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_merge_back_duplicates=") ||
        starts_with(option, "sus2_merge_back_duplicates=")) {
      const size_t eq = option.find('=');
      use_merge = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_merge;
}

int parse_sh_tensor_product_grid_cap(const SHHostModel& model, int nopts, const char** opts)
{
  int cap = 8192;
  const char* env = std::getenv("SUS2_SH_GPUMD_TENSOR_GRID_CAP");
  if (env != nullptr) {
    cap = std::max(1, std::atoi(env));
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_tensor_grid_cap=") ||
        starts_with(option, "sus2_tensor_grid_cap=")) {
      const size_t eq = option.find('=');
      cap = std::max(1, std::atoi(option.substr(eq + 1).c_str()));
    }
  }
  return cap;
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

bool parse_sh_profile_product_detail(const SHHostModel& model, int nopts, const char** opts)
{
  bool enabled = false;
  const char* env = std::getenv("SUS2_SH_GPUMD_PROFILE_PRODUCT_DETAIL");
  if (env != nullptr) {
    enabled = parse_bool_value(env, "SUS2_SH_GPUMD_PROFILE_PRODUCT_DETAIL");
  }
  const int begin = std::min(nopts, model.species_count);
  for (int i = begin; i < nopts; ++i) {
    const std::string option = opts[i] == nullptr ? "" : opts[i];
    if (starts_with(option, "sus2_sh_profile_product_detail=") ||
        starts_with(option, "sus2_profile_product_detail=")) {
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

double sh_host_fact(int n)
{
  if (n < 0) {
    return 0.0;
  }
  double value = 1.0;
  for (int i = 2; i <= n; ++i) {
    value *= static_cast<double>(i);
  }
  return value;
}

bool sh_host_triangle(int l1, int l2, int l3)
{
  return std::abs(l1 - l2) <= l3 && l3 <= l1 + l2;
}

double sh_host_clebsch_gordan(int j1, int m1, int j2, int m2, int j, int m)
{
  if (m != m1 + m2) {
    return 0.0;
  }
  if (!sh_host_triangle(j1, j2, j)) {
    return 0.0;
  }
  if (std::abs(m1) > j1 || std::abs(m2) > j2 || std::abs(m) > j) {
    return 0.0;
  }

  const double pref1 = std::sqrt(
    (2.0 * j + 1.0) * sh_host_fact(j1 + j2 - j) * sh_host_fact(j1 - j2 + j) *
    sh_host_fact(-j1 + j2 + j) / sh_host_fact(j1 + j2 + j + 1));
  const double pref2 = std::sqrt(
    sh_host_fact(j1 + m1) * sh_host_fact(j1 - m1) * sh_host_fact(j2 + m2) *
    sh_host_fact(j2 - m2) * sh_host_fact(j + m) * sh_host_fact(j - m));

  double sum = 0.0;
  for (int k = 0; k <= 64; ++k) {
    const int a = j1 + j2 - j - k;
    const int b = j1 - m1 - k;
    const int c = j2 + m2 - k;
    const int d = j - j2 + m1 + k;
    const int e = j - j1 - m2 + k;
    if (a < 0 || b < 0 || c < 0 || d < 0 || e < 0) {
      continue;
    }
    const double term = ((k % 2) ? -1.0 : 1.0) /
      (sh_host_fact(k) * sh_host_fact(a) * sh_host_fact(b) * sh_host_fact(c) *
       sh_host_fact(d) * sh_host_fact(e));
    sum += term;
  }
  return pref1 * pref2 * sum;
}

double sh_host_parity_sign(int m)
{
  return (std::abs(m) % 2) == 0 ? 1.0 : -1.0;
}

std::complex<double> sh_host_real_from_complex_coeff(int real_m, int complex_m)
{
  const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  if (real_m == 0) {
    return complex_m == 0 ? std::complex<double>(1.0, 0.0) :
                            std::complex<double>(0.0, 0.0);
  }
  const int a = std::abs(real_m);
  if (std::abs(complex_m) != a) {
    return std::complex<double>(0.0, 0.0);
  }
  if (real_m > 0) {
    if (complex_m == a) {
      return inv_sqrt2 * sh_host_parity_sign(a);
    }
    return inv_sqrt2;
  }
  if (complex_m == a) {
    return std::complex<double>(0.0, -inv_sqrt2 * sh_host_parity_sign(a));
  }
  return std::complex<double>(0.0, inv_sqrt2);
}

std::complex<double> sh_host_real_coupling_phase(int l1, int l2, int L)
{
  switch ((l1 + l2 - L) & 3) {
  case 0:
    return std::complex<double>(1.0, 0.0);
  case 1:
    return std::complex<double>(0.0, 1.0);
  case 2:
    return std::complex<double>(-1.0, 0.0);
  default:
    return std::complex<double>(0.0, -1.0);
  }
}

double sh_host_real_cg_coeff(int l1, int rm1, int l2, int rm2, int L, int rM)
{
  std::complex<double> sum(0.0, 0.0);
  for (int M = -L; M <= L; ++M) {
    const std::complex<double> out_u =
      std::conj(sh_host_real_from_complex_coeff(rM, M));
    if (std::abs(out_u) == 0.0) {
      continue;
    }
    for (int m1 = -l1; m1 <= l1; ++m1) {
      const std::complex<double> in1 = sh_host_real_from_complex_coeff(rm1, m1);
      if (std::abs(in1) == 0.0) {
        continue;
      }
      for (int m2 = -l2; m2 <= l2; ++m2) {
        const std::complex<double> in2 = sh_host_real_from_complex_coeff(rm2, m2);
        if (std::abs(in2) == 0.0) {
          continue;
        }
        const double cg = sh_host_clebsch_gordan(l1, m1, l2, m2, L, M);
        if (std::abs(cg) < 1.0e-14) {
          continue;
        }
        sum += out_u * cg * in1 * in2;
      }
    }
  }
  sum *= sh_host_real_coupling_phase(l1, l2, L);
  if (std::abs(sum.imag()) > 1.0e-10) {
    sh_input_error("SUS2_SH standard graph produced a non-real CG coefficient.");
  }
  return sum.real();
}

std::string sh_host_tensor_key(const std::string& left, const std::string& right, int L)
{
  std::ostringstream oss;
  oss << "(" << left << "x" << right << ")->" << L;
  return oss.str();
}

class SHStandardGraphBuilderHost {
public:
  SHStandardGraphBuilderHost(int lmax, int kmax) : lmax_(lmax), kmax_(kmax) {}

  SHStandardGraphHost BuildGraph(int body_order, const std::vector<int>& body_lmax)
  {
    body_lmax_ = body_lmax;
    BuildQ();
    AddScalars(body_order);

    SHStandardGraphHost graph;
    graph.node_count = node_count_;
    graph.basic = basic_;
    graph.products = products_;
    graph.scalars = scalars_;
    graph.tensor_blocks = tensor_blocks_;
    graph.cg_blocks = cg_blocks_;
    graph.cg_terms = cg_terms_;
    for (size_t i = 0; i < tensor_blocks_.size(); ++i) {
      graph.max_layer = std::max(graph.max_layer, tensor_blocks_[i].layer);
    }
    return graph;
  }

private:
  const SHTensorHost& BasicTensor(int q_index)
  {
    const SHQIndexHost& q = q_[q_index];
    const std::map<std::string, int>::const_iterator found = tensor_lookup_.find(q.key);
    if (found != tensor_lookup_.end()) {
      return tensors_[found->second];
    }

    SHTensorHost tensor;
    tensor.key = q.key;
    tensor.l = q.l;
    tensor.layer = 0;
    tensor.zero = false;
    tensor.node.resize(2 * q.l + 1);
    for (int m = -q.l; m <= q.l; ++m) {
      const int node = AddNode();
      tensor.node[m + q.l] = node;
      SHBasicIndexHost basic;
      basic.k = q.k;
      basic.l = q.l;
      basic.m = m;
      basic_.push_back(basic);
    }

    SHTensorBlockHost block;
    block.kind = 0;
    block.layer = 0;
    block.base = tensor.node.empty() ? -1 : tensor.node[0];
    block.l = q.l;
    block.k = q.k;
    block.mu = q.mu;
    tensor.block = static_cast<int>(tensor_blocks_.size());
    tensor_blocks_.push_back(block);

    const int index = static_cast<int>(tensors_.size());
    tensors_.push_back(tensor);
    tensor_lookup_[q.key] = index;
    return tensors_.back();
  }

  const SHTensorHost& Couple(const SHTensorHost& left, const SHTensorHost& right, int L)
  {
    const SHTensorHost left_copy = left;
    const SHTensorHost right_copy = right;
    const std::string key = sh_host_tensor_key(left_copy.key, right_copy.key, L);
    const std::map<std::string, int>::const_iterator found = tensor_lookup_.find(key);
    if (found != tensor_lookup_.end()) {
      return tensors_[found->second];
    }

    SHTensorHost out;
    out.key = key;
    out.l = L;
    out.layer = 1 + std::max(left_copy.layer, right_copy.layer);
    if (left_copy.zero || right_copy.zero) {
      out.zero = true;
      const int index = static_cast<int>(tensors_.size());
      tensors_.push_back(out);
      tensor_lookup_[key] = index;
      return tensors_.back();
    }

    std::vector<SHLocalProductHost> local_products;
    std::vector<SHCGTermHost> logical_terms;
    std::map<std::string, int> local_lookup;
    std::map<std::string, int> logical_lookup;

    auto add_logical_term = [&](int left_component, int right_component, int target_component,
                                double coeff) {
      if (std::abs(coeff) < 1.0e-12) {
        return;
      }
      std::ostringstream local_key;
      local_key << left_component << ',' << right_component << ',' << target_component;
      const std::string key_text = local_key.str();
      std::map<std::string, int>::iterator logical_found = logical_lookup.find(key_text);
      if (logical_found != logical_lookup.end()) {
        logical_terms[logical_found->second].coeff += coeff;
        return;
      }
      SHCGTermHost term;
      term.left_component = left_component;
      term.right_component = right_component;
      term.target_component = target_component;
      term.coeff = coeff;
      logical_lookup[key_text] = static_cast<int>(logical_terms.size());
      logical_terms.push_back(term);
    };

    auto add_local_product = [&](int left_node, int right_node, int target_component,
                                 double coeff) {
      if (std::abs(coeff) < 1.0e-12) {
        return;
      }
      if (right_node < left_node) {
        std::swap(left_node, right_node);
      }
      std::ostringstream local_key;
      local_key << left_node << ',' << right_node << ',' << target_component;
      const std::string key_text = local_key.str();
      std::map<std::string, int>::iterator local_found = local_lookup.find(key_text);
      if (local_found != local_lookup.end()) {
        local_products[local_found->second].coeff += coeff;
        return;
      }
      SHLocalProductHost product;
      product.left = left_node;
      product.right = right_node;
      product.target_component = target_component;
      product.coeff = coeff;
      local_lookup[key_text] = static_cast<int>(local_products.size());
      local_products.push_back(product);
    };

    for (int rm1 = -left_copy.l; rm1 <= left_copy.l; ++rm1) {
      for (int rm2 = -right_copy.l; rm2 <= right_copy.l; ++rm2) {
        for (int rM = -L; rM <= L; ++rM) {
          const double coeff =
            sh_host_real_cg_coeff(left_copy.l, rm1, right_copy.l, rm2, L, rM);
          if (std::abs(coeff) < 1.0e-12) {
            continue;
          }
          const int left_component = rm1 + left_copy.l;
          const int right_component = rm2 + right_copy.l;
          const int target_component = rM + L;
          add_logical_term(left_component, right_component, target_component, coeff);
          add_local_product(left_copy.node[left_component], right_copy.node[right_component],
                            target_component, coeff);
        }
      }
    }

    std::vector<SHLocalProductHost> compact_local;
    compact_local.reserve(local_products.size());
    for (size_t i = 0; i < local_products.size(); ++i) {
      if (std::abs(local_products[i].coeff) >= 1.0e-12) {
        compact_local.push_back(local_products[i]);
      }
    }
    std::vector<SHCGTermHost> compact_terms;
    compact_terms.reserve(logical_terms.size());
    for (size_t i = 0; i < logical_terms.size(); ++i) {
      if (std::abs(logical_terms[i].coeff) >= 1.0e-12) {
        compact_terms.push_back(logical_terms[i]);
      }
    }
    if (compact_local.empty()) {
      out.zero = true;
      const int index = static_cast<int>(tensors_.size());
      tensors_.push_back(out);
      tensor_lookup_[key] = index;
      return tensors_.back();
    }

    out.zero = false;
    out.node.resize(2 * L + 1);
    for (int M = -L; M <= L; ++M) {
      out.node[M + L] = AddNode();
    }

    SHTensorBlockHost tensor_block;
    tensor_block.kind = 1;
    tensor_block.layer = out.layer;
    tensor_block.base = out.node[0];
    tensor_block.l = L;
    tensor_block.left_block = left_copy.block;
    tensor_block.right_block = right_copy.block;
    out.block = static_cast<int>(tensor_blocks_.size());
    tensor_blocks_.push_back(tensor_block);

    const int term_begin = static_cast<int>(cg_terms_.size());
    cg_terms_.insert(cg_terms_.end(), compact_terms.begin(), compact_terms.end());
    SHCGBlockHost cg_block;
    cg_block.layer = out.layer;
    cg_block.left_block = left_copy.block;
    cg_block.right_block = right_copy.block;
    cg_block.target_block = out.block;
    cg_block.left_base = left_copy.node.empty() ? -1 : left_copy.node[0];
    cg_block.right_base = right_copy.node.empty() ? -1 : right_copy.node[0];
    cg_block.target_base = out.node[0];
    cg_block.l1 = left_copy.l;
    cg_block.l2 = right_copy.l;
    cg_block.L = L;
    cg_block.term_begin = term_begin;
    cg_block.term_count = static_cast<int>(compact_terms.size());
    cg_blocks_.push_back(cg_block);

    const int index = static_cast<int>(tensors_.size());
    tensors_.push_back(out);
    tensor_lookup_[key] = index;
    SHTensorHost& stored = tensors_.back();

    for (size_t i = 0; i < compact_local.size(); ++i) {
      AddProduct(compact_local[i].left, compact_local[i].right,
                 stored.node[compact_local[i].target_component], compact_local[i].coeff);
    }
    return stored;
  }

  void BuildQ()
  {
    for (int l = lmax_; l >= 0; --l) {
      for (int k = kmax_ - 1; k >= 0; --k) {
        SHQIndexHost q;
        q.l = l;
        q.k = k;
        q.mu = k * (lmax_ + 1) + l;
        std::ostringstream key;
        key << "q" << q.mu << "_l" << l << "_k" << k;
        q.key = key.str();
        q_.push_back(q);
      }
    }
  }

  bool Allowed(int q_index, int body_order) const
  {
    return q_[q_index].l <= body_lmax_[body_order];
  }

  void EnumerateFactorTuples(
    int factor_count,
    int body_order,
    const std::function<void(const std::vector<int>&)>& callback)
  {
    std::vector<int> tuple;
    std::function<void(int, int, int)> rec = [&](int pos, int max_l, int max_k) {
      if (pos == factor_count) {
        callback(tuple);
        return;
      }
      for (int q_index = 0; q_index < static_cast<int>(q_.size()); ++q_index) {
        const SHQIndexHost& q = q_[q_index];
        if (!Allowed(q_index, body_order) || q.l > max_l || q.k > max_k) {
          continue;
        }
        tuple.push_back(q_index);
        rec(pos + 1, q.l, q.k);
        tuple.pop_back();
      }
    };
    rec(0, lmax_, kmax_ - 1);
  }

  void AddScalarSpec(int body_order, int q0, int q1, int q2, int q3, int q4,
                     int intermediate_l)
  {
    SHScalarSpecHost spec;
    spec.body_order = body_order;
    spec.q0 = q0;
    spec.q1 = q1;
    spec.q2 = q2;
    spec.q3 = q3;
    spec.q4 = q4;
    spec.intermediate_l = intermediate_l;
    scalar_specs_.push_back(spec);
    required_q_[q0] = true;
    if (body_order >= 3) {
      required_q_[q1] = true;
    }
    if (body_order >= 4) {
      required_q_[q2] = true;
    }
    if (body_order >= 5) {
      required_q_[q3] = true;
    }
    if (body_order >= 6) {
      required_q_[q4] = true;
    }
  }

  void CollectScalarSpecs(int body_order)
  {
    required_q_.assign(q_.size(), false);
    scalar_specs_.clear();

    if (body_order >= 2) {
      EnumerateFactorTuples(1, 2, [&](const std::vector<int>& qids) {
        if (q_[qids[0]].l == 0) {
          AddScalarSpec(2, qids[0], -1, -1, -1, -1, 0);
        }
      });
    }
    if (body_order >= 3) {
      EnumerateFactorTuples(2, 3, [&](const std::vector<int>& qids) {
        if (q_[qids[0]].l == q_[qids[1]].l) {
          AddScalarSpec(3, qids[0], qids[1], -1, -1, -1, 0);
        }
      });
    }
    if (body_order >= 4) {
      EnumerateFactorTuples(3, 4, [&](const std::vector<int>& qids) {
        const int l0 = q_[qids[0]].l;
        const int l1 = q_[qids[1]].l;
        const int l2 = q_[qids[2]].l;
        if (((l0 + l1 + l2) % 2) == 0 && sh_host_triangle(l0, l1, l2)) {
          AddScalarSpec(4, qids[0], qids[1], qids[2], -1, -1, l2);
        }
      });
    }
    if (body_order >= 5) {
      EnumerateFactorTuples(4, 5, [&](const std::vector<int>& qids) {
        const int l0 = q_[qids[0]].l;
        const int l1 = q_[qids[1]].l;
        const int l2 = q_[qids[2]].l;
        const int l3 = q_[qids[3]].l;
        if (((l0 + l1 + l2 + l3) % 2) != 0) {
          return;
        }
        const int lo = std::max(std::abs(l0 - l1), std::abs(l2 - l3));
        const int hi = std::min(l0 + l1, l2 + l3);
        for (int L = lo; L <= hi; ++L) {
          AddScalarSpec(5, qids[0], qids[1], qids[2], qids[3], -1, L);
        }
      });
    }
    if (body_order >= 6) {
      EnumerateFactorTuples(5, 6, [&](const std::vector<int>& qids) {
        const int l0 = q_[qids[0]].l;
        const int l1 = q_[qids[1]].l;
        const int l2 = q_[qids[2]].l;
        const int l3 = q_[qids[3]].l;
        const int l4 = q_[qids[4]].l;
        if (((l0 + l1 + l2 + l3 + l4) % 2) != 0) {
          return;
        }
        const int lo = std::max(std::abs(l0 - l1), std::abs(l2 - l3));
        const int hi = std::min(l0 + l1, l2 + l3);
        for (int L = lo; L <= hi; ++L) {
          if (sh_host_triangle(L, l4, L)) {
            AddScalarSpec(6, qids[0], qids[1], qids[2], qids[3], qids[4], L);
          }
        }
      });
    }
  }

  int BuildScalar(const SHScalarSpecHost& spec)
  {
    const SHTensorHost t0 = BasicTensor(spec.q0);
    if (spec.body_order == 2) {
      return t0.node[0];
    }
    const SHTensorHost t1 = BasicTensor(spec.q1);
    if (spec.body_order == 3) {
      const SHTensorHost scalar = Couple(t0, t1, 0);
      return scalar.zero ? -1 : scalar.node[0];
    }
    const SHTensorHost t2 = BasicTensor(spec.q2);
    if (spec.body_order == 4) {
      const SHTensorHost pair = Couple(t0, t1, spec.intermediate_l);
      if (pair.zero) {
        return -1;
      }
      const SHTensorHost scalar = Couple(pair, t2, 0);
      return scalar.zero ? -1 : scalar.node[0];
    }
    const SHTensorHost t3 = BasicTensor(spec.q3);
    const SHTensorHost left = Couple(t0, t1, spec.intermediate_l);
    const SHTensorHost right_pair = Couple(t2, t3, spec.intermediate_l);
    if (spec.body_order == 6) {
      const SHTensorHost t4 = BasicTensor(spec.q4);
      if (left.zero || right_pair.zero) {
        return -1;
      }
      const SHTensorHost right = Couple(right_pair, t4, spec.intermediate_l);
      if (right.zero) {
        return -1;
      }
      const SHTensorHost scalar = Couple(left, right, 0);
      return scalar.zero ? -1 : scalar.node[0];
    }
    if (left.zero || right_pair.zero) {
      return -1;
    }
    const SHTensorHost scalar = Couple(left, right_pair, 0);
    return scalar.zero ? -1 : scalar.node[0];
  }

  void CompressProducts()
  {
    std::vector<SHProductHost> compact;
    compact.reserve(products_.size());
    for (size_t i = 0; i < products_.size(); ++i) {
      if (std::abs(products_[i].coeff) >= 1.0e-12) {
        compact.push_back(products_[i]);
      }
    }
    products_.swap(compact);
  }

  void AddScalars(int body_order)
  {
    CollectScalarSpecs(body_order);
    for (int q_index = 0; q_index < static_cast<int>(q_.size()); ++q_index) {
      if (required_q_[q_index]) {
        BasicTensor(q_index);
      }
    }
    for (size_t i = 0; i < scalar_specs_.size(); ++i) {
      const int scalar = BuildScalar(scalar_specs_[i]);
      if (scalar >= 0) {
        scalars_.push_back(scalar);
      }
    }
    CompressProducts();
  }

  int AddNode()
  {
    return node_count_++;
  }

  void AddProduct(int left, int right, int target, double coeff)
  {
    if (std::abs(coeff) < 1.0e-12) {
      return;
    }
    if (right < left) {
      std::swap(left, right);
    }
    std::ostringstream key;
    key << left << ',' << right << ',' << target;
    const std::string key_text = key.str();
    std::map<std::string, int>::iterator found = product_lookup_.find(key_text);
    if (found != product_lookup_.end()) {
      products_[found->second].coeff += coeff;
      return;
    }
    SHProductHost product;
    product.left = left;
    product.right = right;
    product.target = target;
    product.coeff = coeff;
    product_lookup_[key_text] = static_cast<int>(products_.size());
    products_.push_back(product);
  }

  int lmax_ = 0;
  int kmax_ = 0;
  int node_count_ = 0;
  std::vector<int> body_lmax_;
  std::vector<SHQIndexHost> q_;
  std::vector<SHTensorHost> tensors_;
  std::map<std::string, int> tensor_lookup_;
  std::map<std::string, int> product_lookup_;
  std::vector<char> required_q_;
  std::vector<SHScalarSpecHost> scalar_specs_;
  std::vector<SHBasicIndexHost> basic_;
  std::vector<SHProductHost> products_;
  std::vector<int> scalars_;
  std::vector<SHTensorBlockHost> tensor_blocks_;
  std::vector<SHCGBlockHost> cg_blocks_;
  std::vector<SHCGTermHost> cg_terms_;
};

void validate_standard_sh_graph(SHHostModel& model)
{
  SHStandardGraphBuilderHost builder(model.sh_l_max, model.sh_k_max);
  const SHStandardGraphHost graph =
    builder.BuildGraph(model.sh_body_order, model.sh_body_l_max);

  if (graph.node_count != model.alpha_moments_count) {
    std::ostringstream oss;
    oss << "SUS2_SH standard graph node count mismatch: model="
        << model.alpha_moments_count << " generated=" << graph.node_count;
    sh_input_error(oss.str());
  }
  if (static_cast<int>(graph.basic.size()) != model.alpha_basic_count) {
    std::ostringstream oss;
    oss << "SUS2_SH standard graph basic count mismatch: model="
        << model.alpha_basic_count << " generated=" << graph.basic.size();
    sh_input_error(oss.str());
  }
  for (int i = 0; i < model.alpha_basic_count; ++i) {
    const SHBasicIndexHost& basic = graph.basic[i];
    const int mu = basic.k * (model.sh_l_max + 1) + basic.l;
    if (model.alpha_basic[i * 3 + 0] != mu ||
        model.alpha_basic[i * 3 + 1] != basic.l ||
        model.alpha_basic[i * 3 + 2] != basic.m) {
      std::ostringstream oss;
      oss << "SUS2_SH standard graph alpha_index_basic mismatch at " << i
          << ": model={" << model.alpha_basic[i * 3 + 0] << ", "
          << model.alpha_basic[i * 3 + 1] << ", " << model.alpha_basic[i * 3 + 2]
          << "} generated={" << mu << ", " << basic.l << ", " << basic.m << "}";
      sh_input_error(oss.str());
    }
  }

  if (graph.products.size() != model.products.size()) {
    std::ostringstream oss;
    oss << "SUS2_SH standard graph product count mismatch: model="
        << model.products.size() << " generated=" << graph.products.size();
    sh_input_error(oss.str());
  }
  const double coeff_tol = 2.0e-10;
  for (size_t p = 0; p < model.products.size(); ++p) {
    const SHProductHost& expected = graph.products[p];
    const SHProductHost& actual = model.products[p];
    const double tol = coeff_tol * (1.0 + std::abs(expected.coeff));
    if (actual.left != expected.left || actual.right != expected.right ||
        actual.target != expected.target || std::abs(actual.coeff - expected.coeff) > tol) {
      std::ostringstream oss;
      oss << "SUS2_SH standard graph product mismatch at " << p
          << ": model={" << actual.left << ", " << actual.right << ", "
          << actual.target << ", " << actual.coeff << "} generated={"
          << expected.left << ", " << expected.right << ", " << expected.target
          << ", " << expected.coeff << "}";
      sh_input_error(oss.str());
    }
  }

  if (graph.scalars.size() != model.alpha_moment_mapping.size()) {
    std::ostringstream oss;
    oss << "SUS2_SH standard graph scalar count mismatch: model="
        << model.alpha_moment_mapping.size() << " generated=" << graph.scalars.size();
    sh_input_error(oss.str());
  }
  for (size_t i = 0; i < graph.scalars.size(); ++i) {
    if (graph.scalars[i] != model.alpha_moment_mapping[i]) {
      std::ostringstream oss;
      oss << "SUS2_SH standard graph scalar mapping mismatch at " << i
          << ": model=" << model.alpha_moment_mapping[i]
          << " generated=" << graph.scalars[i];
      sh_input_error(oss.str());
    }
  }

  model.sh_standard_tensor_blocks = static_cast<int>(graph.tensor_blocks.size());
  model.sh_standard_cg_blocks = static_cast<int>(graph.cg_blocks.size());
  model.sh_standard_cg_terms = static_cast<int>(graph.cg_terms.size());
  model.sh_standard_cg_layers = graph.max_layer;
  model.cg_blocks = graph.cg_blocks;
  model.cg_terms = graph.cg_terms;

  std::vector<int> node_layer(graph.node_count, 0);
  std::vector<std::map<int, std::vector<SHCGRowTermHost>>> rows_by_target(
    graph.max_layer + 1);
  for (size_t b = 0; b < graph.tensor_blocks.size(); ++b) {
    const SHTensorBlockHost& block = graph.tensor_blocks[b];
    if (block.layer <= 0 || block.base < 0) {
      continue;
    }
    const int dim = 2 * block.l + 1;
    for (int c = 0; c < dim; ++c) {
      const int node = block.base + c;
      if (node >= 0 && node < graph.node_count) {
        node_layer[node] = block.layer;
        rows_by_target[block.layer][node];
      }
    }
  }
  for (size_t p = 0; p < graph.products.size(); ++p) {
    const SHProductHost& product = graph.products[p];
    const int layer = node_layer[product.target];
    if (layer <= 0 || layer > graph.max_layer) {
      sh_input_error("SUS2_SH compact tensor row has invalid target layer.");
    }
    SHCGRowTermHost term;
    term.left_component = product.left;
    term.right_component = product.right;
    term.coeff = product.coeff;
    rows_by_target[layer][product.target].push_back(term);
  }
  model.cg_layer_offsets.assign(graph.max_layer + 2, 0);
  for (int layer = 1; layer <= graph.max_layer; ++layer) {
    model.cg_layer_offsets[layer] = static_cast<int>(model.cg_rows.size());
    for (std::map<int, std::vector<SHCGRowTermHost>>::const_iterator it =
           rows_by_target[layer].begin();
         it != rows_by_target[layer].end();
         ++it) {
      SHCGRowHost row;
      row.layer = layer;
      row.left_base = 0;
      row.right_base = 0;
      row.target = it->first;
      row.term_begin = static_cast<int>(model.cg_row_terms.size());
      row.term_count = static_cast<int>(it->second.size());
      model.cg_rows.push_back(row);
      model.cg_row_terms.insert(model.cg_row_terms.end(), it->second.begin(), it->second.end());
    }
  }
  model.cg_layer_offsets[graph.max_layer + 1] = static_cast<int>(model.cg_rows.size());

  std::vector<std::map<int, std::vector<SHCGBackTermHost>>> back_by_layer(graph.max_layer + 1);
  for (size_t p = 0; p < graph.products.size(); ++p) {
    const SHProductHost& product = graph.products[p];
    const int layer = node_layer[product.target];
    SHCGBackTermHost left_term;
    left_term.target = product.target;
    left_term.other = product.right;
    left_term.coeff = product.coeff;
    back_by_layer[layer][product.left].push_back(left_term);

    SHCGBackTermHost right_term;
    right_term.target = product.target;
    right_term.other = product.left;
    right_term.coeff = product.coeff;
    back_by_layer[layer][product.right].push_back(right_term);
  }
  model.cg_back_layer_offsets.assign(graph.max_layer + 2, 0);
  for (int layer = 1; layer <= graph.max_layer; ++layer) {
    model.cg_back_layer_offsets[layer] = static_cast<int>(model.cg_back_rows.size());
    for (std::map<int, std::vector<SHCGBackTermHost>>::const_iterator it =
           back_by_layer[layer].begin();
         it != back_by_layer[layer].end();
         ++it) {
      SHCGBackRowHost row;
      row.layer = layer;
      row.source = it->first;
      row.term_begin = static_cast<int>(model.cg_back_terms.size());
      row.term_count = static_cast<int>(it->second.size());
      model.cg_back_rows.push_back(row);
      model.cg_back_terms.insert(model.cg_back_terms.end(), it->second.begin(), it->second.end());
    }
  }
  model.cg_back_layer_offsets[graph.max_layer + 1] =
    static_cast<int>(model.cg_back_rows.size());
}

bool standard_sh_graph_matches(const SHHostModel& model, SHStandardGraphHost& graph)
{
  SHStandardGraphBuilderHost builder(model.sh_l_max, model.sh_k_max);
  graph = builder.BuildGraph(model.sh_body_order, model.sh_body_l_max);

  if (graph.node_count != model.alpha_moments_count ||
      static_cast<int>(graph.basic.size()) != model.alpha_basic_count ||
      graph.products.size() != model.products.size() ||
      graph.scalars.size() != model.alpha_moment_mapping.size()) {
    return false;
  }

  for (int i = 0; i < model.alpha_basic_count; ++i) {
    const SHBasicIndexHost& basic = graph.basic[i];
    const int mu = basic.k * (model.sh_l_max + 1) + basic.l;
    if (model.alpha_basic[i * 3 + 0] != mu ||
        model.alpha_basic[i * 3 + 1] != basic.l ||
        model.alpha_basic[i * 3 + 2] != basic.m) {
      return false;
    }
  }

  const double coeff_tol = 2.0e-10;
  for (size_t p = 0; p < model.products.size(); ++p) {
    const SHProductHost& expected = graph.products[p];
    const SHProductHost& actual = model.products[p];
    const double tol = coeff_tol * (1.0 + std::abs(expected.coeff));
    if (actual.left != expected.left || actual.right != expected.right ||
        actual.target != expected.target || std::abs(actual.coeff - expected.coeff) > tol) {
      return false;
    }
  }

  for (size_t i = 0; i < graph.scalars.size(); ++i) {
    if (graph.scalars[i] != model.alpha_moment_mapping[i]) {
      return false;
    }
  }
  return true;
}

void build_explicit_sh_graph_metadata(
  SHHostModel& model,
  const SHStandardGraphHost* standard_graph)
{
  const int node_count = model.alpha_moments_count;
  if (node_count <= 0 || model.alpha_basic_count <= 0 ||
      model.alpha_basic_count > node_count) {
    sh_input_error("SUS2_SH invalid explicit product graph dimensions.");
  }

  model.cg_blocks.clear();
  model.cg_terms.clear();
  model.cg_rows.clear();
  model.cg_row_terms.clear();
  model.cg_layer_offsets.clear();
  model.cg_back_rows.clear();
  model.cg_back_terms.clear();
  model.cg_back_layer_offsets.clear();

  if (standard_graph != nullptr) {
    model.sh_standard_tensor_blocks =
      static_cast<int>(standard_graph->tensor_blocks.size());
    model.sh_standard_cg_blocks = static_cast<int>(standard_graph->cg_blocks.size());
    model.sh_standard_cg_terms = static_cast<int>(standard_graph->cg_terms.size());
    model.cg_blocks = standard_graph->cg_blocks;
    model.cg_terms = standard_graph->cg_terms;
    model.sh_standard_graph_matched = true;
  } else {
    model.sh_standard_tensor_blocks = 0;
    model.sh_standard_cg_blocks = 0;
    model.sh_standard_cg_terms = 0;
    model.sh_standard_graph_matched = false;
  }

  std::vector<int> last_definition(node_count, -1);
  std::vector<std::vector<SHCGRowTermHost>> terms_by_target(node_count);
  for (size_t p = 0; p < model.products.size(); ++p) {
    const SHProductHost& product = model.products[p];
    if (product.left < 0 || product.left >= node_count ||
        product.right < 0 || product.right >= node_count ||
        product.target < 0 || product.target >= node_count) {
      sh_input_error("SUS2_SH explicit product graph index out of range.");
    }
    if (product.target < model.alpha_basic_count) {
      sh_input_error("SUS2_SH explicit product graph writes into a basic moment.");
    }
    SHCGRowTermHost term;
    term.left_component = product.left;
    term.right_component = product.right;
    term.coeff = product.coeff;
    terms_by_target[product.target].push_back(term);
    last_definition[product.target] = static_cast<int>(p);
  }
  for (size_t p = 0; p < model.products.size(); ++p) {
    const SHProductHost& product = model.products[p];
    if ((product.left >= model.alpha_basic_count &&
         last_definition[product.left] < 0) ||
        (product.right >= model.alpha_basic_count &&
         last_definition[product.right] < 0)) {
      sh_input_error("SUS2_SH explicit product graph references an undefined moment.");
    }
    if ((product.left >= model.alpha_basic_count &&
         last_definition[product.left] >= static_cast<int>(p)) ||
        (product.right >= model.alpha_basic_count &&
         last_definition[product.right] >= static_cast<int>(p))) {
      sh_input_error("SUS2_SH explicit product graph is not topologically ordered.");
    }
  }

  std::vector<int> targets;
  targets.reserve(model.products.size());
  std::vector<unsigned char> defined(node_count, 0);
  for (int i = 0; i < model.alpha_basic_count; ++i) {
    defined[i] = 1;
  }
  for (int target = model.alpha_basic_count; target < node_count; ++target) {
    if (!terms_by_target[target].empty()) {
      targets.push_back(target);
      defined[target] = 1;
    }
  }
  std::sort(targets.begin(), targets.end(), [&](int a, int b) {
    if (last_definition[a] != last_definition[b]) {
      return last_definition[a] < last_definition[b];
    }
    return a < b;
  });

  for (int scalar = 0; scalar < model.alpha_scalar_moments; ++scalar) {
    const int moment = model.alpha_moment_mapping[scalar];
    if (moment < 0 || moment >= node_count || !defined[moment]) {
      sh_input_error("SUS2_SH alpha_moment_mapping references an undefined moment.");
    }
  }

  std::vector<int> node_layer(node_count, 0);
  std::vector<int> target_layer(node_count, 0);
  int max_layer = 0;
  for (int target : targets) {
    int source_layer = 0;
    const std::vector<SHCGRowTermHost>& terms = terms_by_target[target];
    for (size_t t = 0; t < terms.size(); ++t) {
      source_layer = std::max(source_layer, node_layer[terms[t].left_component]);
      source_layer = std::max(source_layer, node_layer[terms[t].right_component]);
    }
    const int layer = source_layer + 1;
    node_layer[target] = layer;
    target_layer[target] = layer;
    max_layer = std::max(max_layer, layer);
  }

  model.sh_standard_cg_layers = max_layer;
  std::vector<std::vector<int>> targets_by_layer(max_layer + 1);
  for (int target : targets) {
    targets_by_layer[target_layer[target]].push_back(target);
  }
  model.cg_layer_offsets.assign(max_layer + 2, 0);
  for (int layer = 1; layer <= max_layer; ++layer) {
    model.cg_layer_offsets[layer] = static_cast<int>(model.cg_rows.size());
    std::sort(targets_by_layer[layer].begin(), targets_by_layer[layer].end());
    for (int target : targets_by_layer[layer]) {
      SHCGRowHost row;
      row.layer = layer;
      row.left_base = 0;
      row.right_base = 0;
      row.target = target;
      row.term_begin = static_cast<int>(model.cg_row_terms.size());
      row.term_count = static_cast<int>(terms_by_target[target].size());
      model.cg_rows.push_back(row);
      model.cg_row_terms.insert(
        model.cg_row_terms.end(),
        terms_by_target[target].begin(),
        terms_by_target[target].end());
    }
  }
  model.cg_layer_offsets[max_layer + 1] = static_cast<int>(model.cg_rows.size());

  std::vector<std::map<int, std::vector<SHCGBackTermHost>>> back_by_layer(max_layer + 1);
  for (size_t row_index = 0; row_index < model.cg_rows.size(); ++row_index) {
    const SHCGRowHost& row = model.cg_rows[row_index];
    for (int t = 0; t < row.term_count; ++t) {
      const SHCGRowTermHost& term = model.cg_row_terms[row.term_begin + t];
      const int left = row.left_base + term.left_component;
      const int right = row.right_base + term.right_component;

      SHCGBackTermHost left_term;
      left_term.target = row.target;
      left_term.other = right;
      left_term.coeff = term.coeff;
      back_by_layer[row.layer][left].push_back(left_term);

      SHCGBackTermHost right_term;
      right_term.target = row.target;
      right_term.other = left;
      right_term.coeff = term.coeff;
      back_by_layer[row.layer][right].push_back(right_term);
    }
  }

  model.cg_back_layer_offsets.assign(max_layer + 2, 0);
  for (int layer = 1; layer <= max_layer; ++layer) {
    model.cg_back_layer_offsets[layer] = static_cast<int>(model.cg_back_rows.size());
    for (std::map<int, std::vector<SHCGBackTermHost>>::const_iterator it =
           back_by_layer[layer].begin();
         it != back_by_layer[layer].end();
         ++it) {
      SHCGBackRowHost row;
      row.layer = layer;
      row.source = it->first;
      row.term_begin = static_cast<int>(model.cg_back_terms.size());
      row.term_count = static_cast<int>(it->second.size());
      model.cg_back_rows.push_back(row);
      model.cg_back_terms.insert(
        model.cg_back_terms.end(),
        it->second.begin(),
        it->second.end());
    }
  }
  model.cg_back_layer_offsets[max_layer + 1] =
    static_cast<int>(model.cg_back_rows.size());
}

void prepare_sh_graph_metadata(SHHostModel& model)
{
  SHStandardGraphHost standard_graph;
  if (standard_sh_graph_matches(model, standard_graph)) {
    build_explicit_sh_graph_metadata(model, &standard_graph);
  } else {
    build_explicit_sh_graph_metadata(model, nullptr);
  }
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
  if (has_token(text, "sh_parity =")) {
    const std::string parity = parse_string_after(text, "sh_parity =");
    if (parity != "even") {
      sh_input_error("SUS2_SH GPUMD backend currently supports even parity models only.");
    }
  }
  model.radial_basis_type = parse_string_after(text, "radial_basis_type =");
  model.radial_basis_kind = sh_radial_basis_kind_from_string(model.radial_basis_type);
  model.max_dist = parse_double_after(text, "max_dist =");
  model.zbl_enabled = parse_optional_bool_after(text, "zbl_enabled =", false);
  if (model.zbl_enabled) {
    model.zbl_inner = parse_optional_double_after(
      text, "zbl_inner =", sus2_zbl_default_inner_cutoff());
    model.zbl_outer = parse_optional_double_after(
      text, "zbl_outer =", sus2_zbl_default_outer_cutoff());
    model.zbl_typewise_cutoff_enabled = has_token(text, "zbl_typewise_cutoff_enabled =")
      ? parse_optional_bool_after(text, "zbl_typewise_cutoff_enabled =", false)
      : has_token(text, "zbl_typewise_cutoff_factor =");
    model.zbl_typewise_cutoff_factor = parse_optional_double_after(
      text, "zbl_typewise_cutoff_factor =", sus2_zbl_default_typewise_cutoff_factor());
    if (has_token(text, "zbl_atomic_numbers =")) {
      model.zbl_atomic_numbers =
        parse_numbers<int>(extract_braced_after(text, "zbl_atomic_numbers ="));
    }
  }
  model.rb_size = parse_int_after(text, "radial_basis_size =");
  model.radial_funcs_count = parse_int_after(text, "radial_funcs_count =");
  model.alpha_moments_count = parse_int_after(text, "alpha_moments_count =");
  model.alpha_basic_count = parse_int_after(text, "alpha_index_basic_count =");
  model.alpha_scalar_moments = parse_int_after(text, "alpha_scalar_moments =");
  model.two_layer_gate_enabled =
    parse_optional_bool_after(text, "two_layer_gate_enabled =", false);
  if (model.two_layer_gate_enabled) {
    if (parse_optional_bool_after(text, "two_layer_gate_include_one_body =", false)) {
      sh_input_error("GPUMD SUS2_SH gate backend requires two_layer_gate_include_one_body = false.");
    }
    if (has_token(text, "two_layer_gate_scale_mode =")) {
      const std::string scale_mode = parse_string_after(text, "two_layer_gate_scale_mode =");
      if (scale_mode != "legacy") {
        sh_input_error("GPUMD SUS2_SH gate backend supports the current tanh additive gate only.");
      }
    }
    model.two_layer_gate_tanh_amplitude =
      parse_optional_double_after(text, "two_layer_gate_tanh_amplitude =", 0.8);
    if (!std::isfinite(model.two_layer_gate_tanh_amplitude) ||
        model.two_layer_gate_tanh_amplitude < 0.0 ||
        model.two_layer_gate_tanh_amplitude > 1.0) {
      sh_input_error("SUS2_SH two_layer_gate_tanh_amplitude should be finite and in [0,1].");
    }
    const std::string radial_mode =
      has_token(text, "two_layer_gate_radial_mode =")
        ? parse_string_after(text, "two_layer_gate_radial_mode =")
        : std::string();
    if (radial_mode != "shared-radial") {
      sh_input_error("GPUMD SUS2_SH gate backend currently requires shared-radial gate mode.");
    }
    const int gate_radial_count =
      parse_int_after(text, "two_layer_gate_radial_coeff_count =");
    const int expected_gate_radial_count = model.radial_funcs_count * model.rb_size;
    if (gate_radial_count != expected_gate_radial_count) {
      sh_input_error("SUS2_SH two-layer gate radial coefficient count is inconsistent.");
    }
    model.two_layer_gate_radial_coeffs =
      parse_numbers<double>(extract_braced_after(text, "two_layer_gate_radial_coeffs ="));
    if (static_cast<int>(model.two_layer_gate_radial_coeffs.size()) != gate_radial_count) {
      sh_input_error("SUS2_SH two-layer gate radial coeff list has wrong size.");
    }
    if (has_token(text, "two_layer_gate_additive_coeff_count =")) {
      const int additive_count =
        parse_int_after(text, "two_layer_gate_additive_coeff_count =");
      const int expected_additive_count = model.species_count * model.radial_funcs_count;
      if (additive_count != expected_additive_count) {
        sh_input_error("SUS2_SH two-layer gate additive coefficient count is inconsistent.");
      }
      model.two_layer_gate_additive_coeffs =
        parse_numbers<double>(extract_braced_after(text, "two_layer_gate_additive_coeffs ="));
      if (static_cast<int>(model.two_layer_gate_additive_coeffs.size()) != additive_count) {
        sh_input_error("SUS2_SH two-layer gate additive coeff list has wrong size.");
      }
    } else {
      model.two_layer_gate_additive_coeffs.assign(
        model.species_count * model.radial_funcs_count, 1.0);
    }
    model.two_layer_gate_weight_count =
      parse_int_after(text, "two_layer_gate_weight_count =");
    if (model.two_layer_gate_weight_count <= 0) {
      sh_input_error("SUS2_SH two-layer gate should contain at least one scalar weight.");
    }
    model.two_layer_gate_scalar_indices =
      parse_numbers<int>(extract_braced_after(text, "two_layer_gate_scalar_indices ="));
    model.two_layer_gate_weights =
      parse_numbers<double>(extract_braced_after(text, "two_layer_gate_weights ="));
    if (static_cast<int>(model.two_layer_gate_scalar_indices.size()) !=
          model.two_layer_gate_weight_count ||
        static_cast<int>(model.two_layer_gate_weights.size()) !=
          model.two_layer_gate_weight_count) {
      sh_input_error("SUS2_SH two-layer gate scalar index/weight list has wrong size.");
    }
  }

  if (model.scaling_map != "LK") {
    sh_input_error("SUS2_SH requires scaling_map = LK.");
  }
  if (model.sh_l_max < 0 || model.sh_l_max > kMaxSHL) {
    sh_input_error("SUS2_SH supports sh_l_max in [0,6].");
  }
  if (model.sh_body_order < 2 || model.sh_body_order > 6) {
    sh_input_error("SUS2_SH supports sh_body_order in [2,6].");
  }
  if (model.sh_k_max <= 0 || model.radial_funcs_count != model.sh_k_max * (model.sh_l_max + 1)) {
    sh_input_error("SUS2_SH inconsistent sh_k_max/radial_funcs_count.");
  }
  if (model.rb_size <= 0 || model.rb_size > kMaxSHRbSize ||
      model.radial_funcs_count > kMaxSHRadialFuncs) {
    sh_input_error("SUS2_SH GPU scratch limit exceeded: rb_size<=16, radial_funcs_count<=48.");
  }
  if (model.alpha_basic_count <= 0 || model.alpha_basic_count > kMaxSHBasics) {
    sh_input_error("SUS2_SH GPU scratch limit exceeded: alpha_index_basic_count<=320.");
  }
  model.sh_body_l_max.assign(7, model.sh_l_max);
  if (has_token(text, "sh_body_l_max =")) {
    const std::vector<int> body_values =
      parse_numbers<int>(extract_braced_after(text, "sh_body_l_max ="));
    const int expected = model.sh_body_order >= 6 ? 5 : 4;
    if (static_cast<int>(body_values.size()) != expected) {
      sh_input_error("Unexpected SUS2_SH sh_body_l_max size.");
    }
    for (int body = 2; body <= 1 + expected; ++body) {
      model.sh_body_l_max[body] = body_values[body - 2];
    }
  }
  for (int body = 2; body <= model.sh_body_order; ++body) {
    if (model.sh_body_l_max[body] < 0 || model.sh_body_l_max[body] > model.sh_l_max) {
      sh_input_error("SUS2_SH sh_body_l_max entry is out of range.");
    }
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
  if (model.two_layer_gate_enabled) {
    model.two_layer_gate_moment_indices.resize(model.two_layer_gate_weight_count);
    model.two_layer_gate_moment_weights_float.assign(model.alpha_moments_count, 0.0f);
    std::vector<unsigned char> needed(model.alpha_moments_count, 0);
    for (int q = 0; q < model.two_layer_gate_weight_count; ++q) {
      const int scalar_index = model.two_layer_gate_scalar_indices[q];
      if (scalar_index < 0 || scalar_index >= model.alpha_scalar_moments) {
        sh_input_error("SUS2_SH two-layer gate scalar index is out of range.");
      }
      const int moment_index = model.alpha_moment_mapping[scalar_index];
      if (moment_index < 0 || moment_index >= model.alpha_moments_count) {
        sh_input_error("SUS2_SH two-layer gate mapped moment index is out of range.");
      }
      model.two_layer_gate_moment_indices[q] = moment_index;
      model.two_layer_gate_moment_weights_float[moment_index] +=
        static_cast<float>(model.two_layer_gate_weights[q]);
      if (model.two_layer_gate_weights[q] != 0.0) {
        needed[moment_index] = 1;
      }
    }
    model.two_layer_gate_product_limit = 0;
    for (int p = sh_product_count - 1; p >= 0; --p) {
      const SHProductHost& product = model.products[p];
      if (!needed[product.target]) {
        continue;
      }
      needed[product.left] = 1;
      needed[product.right] = 1;
      if (model.two_layer_gate_product_limit == 0) {
        model.two_layer_gate_product_limit = p + 1;
      }
    }
    model.two_layer_gate_needed_moment_flags.assign(model.alpha_moments_count, 0);
    for (int m = 0; m < model.alpha_moments_count; ++m) {
      model.two_layer_gate_needed_moment_flags[m] = needed[m] ? 1 : 0;
    }
  }
  prepare_sh_graph_metadata(model);
  return model;
}

void build_direct_radial_tables(
  const SHHostModel& model,
  std::vector<float>& coeffs,
  std::vector<float>& scal_s,
  const std::vector<double>* radial_coeff_override = nullptr)
{
  const std::vector<double>& radial_coeffs =
    radial_coeff_override == nullptr ? model.radial_coeffs : *radial_coeff_override;
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
            static_cast<float>(radial_coeffs[mu * model.rb_size + xi]) * type_scale;
        }
      }
    }
  }
}

bool has_static_full_sh_basic_layout(const SHHostModel& model)
{
  if (model.sh_l_max < 0 || model.sh_l_max > kMaxSHL ||
      model.sh_k_max <= 0 || model.sh_k_max > 6 ||
      model.rb_size != 10) {
    return false;
  }
  const int expected_count =
    model.sh_k_max * (model.sh_l_max + 1) * (model.sh_l_max + 1);
  const int expected_radial_funcs = model.sh_k_max * (model.sh_l_max + 1);
  if (model.alpha_basic_count != expected_count ||
      model.radial_funcs_count != expected_radial_funcs ||
      static_cast<int>(model.alpha_basic.size()) != expected_count * 3) {
    return false;
  }

  int index = 0;
  for (int l = model.sh_l_max; l >= 0; --l) {
    for (int k = model.sh_k_max - 1; k >= 0; --k) {
      const int mu = k * (model.sh_l_max + 1) + l;
      for (int m = -l; m <= l; ++m) {
        const int base = index * 3;
        if (model.alpha_basic[base + 0] != mu ||
            model.alpha_basic[base + 1] != l ||
            model.alpha_basic[base + 2] != m) {
          return false;
        }
        ++index;
      }
    }
  }
  return true;
}

struct SHDeviceModel {
  int species_count;
  int sh_l_max;
  int radial_basis_kind;
  int radial_funcs_count;
  int rb_size;
  int alpha_basic_count;
  int sh_product_count;
  int sh_cg_block_count;
  int sh_cg_term_count;
  int sh_cg_row_count;
  int sh_cg_row_term_count;
  int sh_cg_row_pattern_count;
  int sh_cg_row_pattern_term_count;
  int sh_cg_back_row_count;
  int sh_cg_back_term_count;
  int sh_cg_layer_count;
  int sh_terminal_dot_group_count;
  int sh_terminal_dot_group_entry_count;
  int sh_terminal_dot_nondot_row_count;
  int sh_fused_terminal_dot_group_count;
  int sh_fused_terminal_dot_group_entry_count;
  int sh_fused_terminal_dot_producer_count;
  int sh_fused_terminal_dot_component_count;
  int sh_fused_terminal_dot_term_count;
  int alpha_moments_count;
  int alpha_scalar_moments;
  int active_scalar_moments;
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
  const int* sh_cg_blocks_int;
  const int* sh_cg_terms_int;
  const double* sh_cg_terms_coeff;
  const float* sh_cg_terms_coeff_float;
  const int* sh_cg_rows_int;
  const int* sh_cg_row_terms_int;
  const double* sh_cg_row_terms_coeff;
  const float* sh_cg_row_terms_coeff_float;
  const int* sh_cg_row_scalar_index;
  const int* sh_terminal_moment_flags;
  const unsigned int* sh_cg_row_dot_u32;
  const unsigned int* sh_cg_row_pattern_u32;
  const unsigned int* sh_terminal_dot_group_u32;
  const int* sh_terminal_dot_group_layer_offsets;
  const int* sh_terminal_dot_nondot_rows;
  const int* sh_terminal_dot_nondot_layer_offsets;
  const unsigned int* sh_fused_terminal_dot_u32;
  const int* sh_cg_back_rows_int;
  const int* sh_cg_back_terms_int;
  const double* sh_cg_back_terms_coeff;
  const float* sh_cg_back_terms_coeff_float;
  const int* sh_cg_back_layer_offsets;
  const unsigned int* sh_cg_back_packed_u32;
  const int* active_scalar_moment;
  const double* active_scalar_coeff;
  const float* active_scalar_coeff_float;
  const int* sh_cg_layer_offsets;
  const int* alpha_moment_mapping;
  const float* radial_direct_coeffs;
  const float* radial_direct_scal_s;
  bool two_layer_gate_enabled;
  int two_layer_gate_weight_count;
  int two_layer_gate_product_limit;
  double two_layer_gate_tanh_amplitude;
  const float* two_layer_gate_radial_direct_coeffs;
  const int* two_layer_gate_moment_indices;
  const float* two_layer_gate_weights_float;
  const int* two_layer_gate_needed_moment_flags;
  const float* two_layer_gate_moment_weights_float;
  const float* two_layer_gate_additive_coeffs_float;
  bool zbl_enabled;
  const int* zbl_atomic_numbers;
  const double* zbl_pair_inner_cutoffs;
  const double* zbl_pair_outer_cutoffs;
  const double* zbl_pair_outer_sq;
  bool use_float_model_params;
  bool use_const_forward_rows;
  bool use_product_pattern_rows;
  bool use_const_pattern_rows;
  bool use_terminal_scalar_fusion;
  bool use_packed_back_rows;
  bool use_const_back_rows;
  bool use_terminal_dot_rows;
  bool use_terminal_dot_groups;
  bool use_terminal_dot_premul;
  bool use_product_basic_cache;
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
__device__ __forceinline__ RealT sh_active_scalar_coeff(const SHDeviceModel& model, int idx)
{
  return model.use_float_model_params ? static_cast<RealT>(model.active_scalar_coeff_float[idx])
                                      : static_cast<RealT>(model.active_scalar_coeff[idx]);
}

template <typename RealT>
__device__ __forceinline__ RealT sh_product_coeff(const SHDeviceModel& model, int idx)
{
  return model.use_float_model_params ? static_cast<RealT>(model.sh_products_coeff_float[idx])
                                      : static_cast<RealT>(model.sh_products_coeff[idx]);
}

template <typename RealT>
__device__ __forceinline__ RealT sh_cg_term_coeff(const SHDeviceModel& model, int idx)
{
  return model.use_float_model_params ? static_cast<RealT>(model.sh_cg_terms_coeff_float[idx])
                                      : static_cast<RealT>(model.sh_cg_terms_coeff[idx]);
}

template <typename RealT>
__device__ __forceinline__ RealT sh_cg_row_term_coeff(const SHDeviceModel& model, int idx)
{
  return model.use_float_model_params ? static_cast<RealT>(model.sh_cg_row_terms_coeff_float[idx])
                                      : static_cast<RealT>(model.sh_cg_row_terms_coeff[idx]);
}

template <typename RealT>
__device__ __forceinline__ RealT sh_cg_back_term_coeff(const SHDeviceModel& model, int idx)
{
  return model.use_float_model_params ? static_cast<RealT>(model.sh_cg_back_terms_coeff_float[idx])
                                      : static_cast<RealT>(model.sh_cg_back_terms_coeff[idx]);
}

__device__ __forceinline__ int sh_const_back_u32_offset(const SHDeviceModel& model)
{
  if (model.use_const_pattern_rows) {
    return model.sh_cg_row_count * 2 + model.sh_cg_row_pattern_count * 2 +
           model.sh_cg_row_pattern_term_count * 2;
  }
  if (model.use_const_forward_rows) {
    return model.sh_cg_row_count * 3 + model.sh_cg_row_term_count * 2;
  }
  return 0;
}

template <typename RealT>
__device__ __forceinline__ RealT sh_const_forward_coeff(unsigned int bits)
{
  return static_cast<RealT>(__uint_as_float(bits));
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
  const RealT inv_r4 = inv_r2 * inv_r2;
  if (l == 4) {
    return inv_r4;
  }
  if (l == 5) {
    return inv_r4 * inv_r;
  }
  return inv_r4 * inv_r2;
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
struct SHDeviceComplex {
  RealT re;
  RealT im;
};

template <typename RealT>
__device__ __forceinline__ SHDeviceComplex<RealT> sh_cmake(RealT re, RealT im)
{
  SHDeviceComplex<RealT> value;
  value.re = re;
  value.im = im;
  return value;
}

template <typename RealT>
__device__ __forceinline__ SHDeviceComplex<RealT> sh_cadd(
  SHDeviceComplex<RealT> a,
  SHDeviceComplex<RealT> b)
{
  return sh_cmake(a.re + b.re, a.im + b.im);
}

template <typename RealT>
__device__ __forceinline__ SHDeviceComplex<RealT> sh_csub(
  SHDeviceComplex<RealT> a,
  SHDeviceComplex<RealT> b)
{
  return sh_cmake(a.re - b.re, a.im - b.im);
}

template <typename RealT>
__device__ __forceinline__ SHDeviceComplex<RealT> sh_cmul(
  SHDeviceComplex<RealT> a,
  SHDeviceComplex<RealT> b)
{
  return sh_cmake(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

template <typename RealT>
__device__ __forceinline__ SHDeviceComplex<RealT> sh_cscale(
  SHDeviceComplex<RealT> a,
  RealT scale)
{
  return sh_cmake(a.re * scale, a.im * scale);
}

template <typename RealT>
__device__ __forceinline__ RealT sh_factorial(int n)
{
  RealT value = static_cast<RealT>(1.0);
  for (int i = 2; i <= n; ++i) {
    value *= static_cast<RealT>(i);
  }
  return value;
}

template <typename RealT>
__device__ __forceinline__ RealT sh_complex_norm(int l, int m)
{
  return sqrt(
    (static_cast<RealT>(2 * l + 1) / static_cast<RealT>(4.0 * kPi)) *
    sh_factorial<RealT>(l - m) / sh_factorial<RealT>(l + m));
}

__device__ __forceinline__ int sh_parity_sign_int(int m)
{
  return (m & 1) == 0 ? 1 : -1;
}

template <typename RealT>
__device__ __forceinline__ void eval_real_sh_from_solid(
  RealT x,
  RealT y,
  RealT z,
  RealT r,
  int lmax,
  RealT* vals,
  RealT* ders)
{
  const int count = (lmax + 1) * (lmax + 1);
  for (int i = 0; i < count; ++i) {
    vals[i] = static_cast<RealT>(0.0);
    if (ders != nullptr) {
      ders[3 * i + 0] = ders[3 * i + 1] = ders[3 * i + 2] = static_cast<RealT>(0.0);
    }
  }

  SHDeviceComplex<RealT> solid[kMaxSHComponents];
  SHDeviceComplex<RealT> solid_ders[3 * kMaxSHComponents];
  for (int i = 0; i < kMaxSHComponents; ++i) {
    solid[i] = sh_cmake<RealT>(static_cast<RealT>(0.0), static_cast<RealT>(0.0));
    if (ders != nullptr) {
      solid_ders[3 * i + 0] = solid[i];
      solid_ders[3 * i + 1] = solid[i];
      solid_ders[3 * i + 2] = solid[i];
    }
  }

  const RealT r2 = r * r;
  const SHDeviceComplex<RealT> u = sh_cmake(x, y);
  const SHDeviceComplex<RealT> zero =
    sh_cmake<RealT>(static_cast<RealT>(0.0), static_cast<RealT>(0.0));
  const SHDeviceComplex<RealT> du[3] = {
    sh_cmake<RealT>(static_cast<RealT>(1.0), static_cast<RealT>(0.0)),
    sh_cmake<RealT>(static_cast<RealT>(0.0), static_cast<RealT>(1.0)),
    zero
  };

  solid[sh_flat_index(0, 0)] =
    sh_cmake<RealT>(static_cast<RealT>(1.0), static_cast<RealT>(0.0));
  for (int m = 1; m <= lmax; ++m) {
    const int prev = sh_flat_index(m - 1, m - 1);
    const int idx = sh_flat_index(m, m);
    const RealT coeff = -static_cast<RealT>(2 * m - 1);
    solid[idx] = sh_cscale(sh_cmul(u, solid[prev]), coeff);
    if (ders != nullptr) {
      for (int a = 0; a < 3; ++a) {
        solid_ders[3 * idx + a] =
          sh_cscale(
            sh_cadd(sh_cmul(du[a], solid[prev]), sh_cmul(u, solid_ders[3 * prev + a])),
            coeff);
      }
    }
  }

  for (int m = 0; m <= lmax; ++m) {
    const int diag = sh_flat_index(m, m);
    if (m + 1 <= lmax) {
      const int idx = sh_flat_index(m + 1, m);
      const RealT coeff = static_cast<RealT>(2 * m + 1);
      solid[idx] = sh_cscale(solid[diag], coeff * z);
      if (ders != nullptr) {
        for (int a = 0; a < 3; ++a) {
          const SHDeviceComplex<RealT> dz_term = (a == 2) ? solid[diag] : zero;
          solid_ders[3 * idx + a] =
            sh_cscale(
              sh_cadd(dz_term, sh_cscale(solid_ders[3 * diag + a], z)),
              coeff);
        }
      }
    }
    for (int l = m + 2; l <= lmax; ++l) {
      const int idx = sh_flat_index(l, m);
      const int prev1 = sh_flat_index(l - 1, m);
      const int prev2 = sh_flat_index(l - 2, m);
      const RealT acoef = static_cast<RealT>(2 * l - 1);
      const RealT bcoef = static_cast<RealT>(l + m - 1);
      const RealT inv_denom = static_cast<RealT>(1.0) / static_cast<RealT>(l - m);
      solid[idx] =
        sh_cscale(
          sh_csub(sh_cscale(solid[prev1], acoef * z),
                  sh_cscale(solid[prev2], bcoef * r2)),
          inv_denom);
      if (ders != nullptr) {
        for (int a = 0; a < 3; ++a) {
          const RealT dr2 = static_cast<RealT>(2.0) *
            (a == 0 ? x : (a == 1 ? y : z));
          const SHDeviceComplex<RealT> dz_term = (a == 2) ? solid[prev1] : zero;
          const SHDeviceComplex<RealT> first =
            sh_cscale(
              sh_cadd(dz_term, sh_cscale(solid_ders[3 * prev1 + a], z)),
              acoef);
          const SHDeviceComplex<RealT> second =
            sh_cscale(
              sh_cadd(sh_cscale(solid[prev2], dr2),
                      sh_cscale(solid_ders[3 * prev2 + a], r2)),
              bcoef);
          solid_ders[3 * idx + a] = sh_cscale(sh_csub(first, second), inv_denom);
        }
      }
    }
  }

  const RealT sqrt2 = sqrt(static_cast<RealT>(2.0));
  for (int l = 0; l <= lmax; ++l) {
    const RealT inv_pow = sh_inv_power(l, r);
    const RealT inv_pow_der =
      l == 0 ? static_cast<RealT>(0.0)
             : -static_cast<RealT>(l) * inv_pow / (r * r);
    for (int m = 0; m <= l; ++m) {
      const int cidx = sh_flat_index(l, m);
      const RealT norm = sh_complex_norm<RealT>(l, m);
      const SHDeviceComplex<RealT> y_complex =
        sh_cscale(solid[cidx], norm * inv_pow);
      if (m == 0) {
        const int ridx = sh_flat_index(l, 0);
        vals[ridx] = y_complex.re;
        if (ders != nullptr) {
          for (int a = 0; a < 3; ++a) {
            const RealT coord = a == 0 ? x : (a == 1 ? y : z);
            const SHDeviceComplex<RealT> dy_complex =
              sh_cscale(
                sh_cadd(sh_cscale(solid_ders[3 * cidx + a], inv_pow),
                        sh_cscale(solid[cidx], inv_pow_der * coord)),
                norm);
            ders[3 * ridx + a] = dy_complex.re;
          }
        }
      } else {
        const RealT factor = sqrt2 * static_cast<RealT>(sh_parity_sign_int(m));
        const int pidx = sh_flat_index(l, m);
        const int nidx = sh_flat_index(l, -m);
        vals[pidx] = factor * y_complex.re;
        vals[nidx] = factor * y_complex.im;
        if (ders != nullptr) {
          for (int a = 0; a < 3; ++a) {
            const RealT coord = a == 0 ? x : (a == 1 ? y : z);
            const SHDeviceComplex<RealT> dy_complex =
              sh_cscale(
                sh_cadd(sh_cscale(solid_ders[3 * cidx + a], inv_pow),
                        sh_cscale(solid[cidx], inv_pow_der * coord)),
                norm);
            ders[3 * pidx + a] = factor * dy_complex.re;
            ders[3 * nidx + a] = factor * dy_complex.im;
          }
        }
      }
    }
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
  if (lmax > 4) {
    eval_real_sh_from_solid(x, y, z, r, lmax, vals, ders);
    return;
  }
  const int clear_count = ders == nullptr ? (lmax + 1) * (lmax + 1) : kMaxSHComponents;
  for (int i = 0; i < clear_count; ++i) {
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
__device__ __forceinline__ void sh_direct_radial_vals_ders_dynamic(
  const SHDeviceModel& model,
  const float* radial_coeffs,
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
    const size_t coeff_base =
      (static_cast<size_t>(pair) * model.radial_funcs_count + mu) * model.rb_size;

    RealT prev = static_cast<RealT>(1.0);
    RealT prev_x = static_cast<RealT>(0.0);
    RealT acc_s = static_cast<RealT>(radial_coeffs[coeff_base]);
    RealT acc_sx = static_cast<RealT>(0.0);
    if (model.rb_size > 1) {
      RealT curr = ksi;
      RealT curr_x = static_cast<RealT>(1.0);
      RealT coeff = static_cast<RealT>(radial_coeffs[coeff_base + 1]);
	      acc_s += coeff * curr;
	      acc_sx += coeff * curr_x;
	      for (int xi = 2; xi < model.rb_size; ++xi) {
	        const RealT next = static_cast<RealT>(2.0) * ksi * curr - prev;
	        const RealT next_x = static_cast<RealT>(2.0) * (curr + ksi * curr_x) - prev_x;
	        coeff = static_cast<RealT>(radial_coeffs[coeff_base + xi]);
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

template <typename RealT, int RadialFuncs, int RbSize>
__device__ __forceinline__ void sh_laguerre_log1p_radial_vals_ders_static(
  const SHDeviceModel& model,
  const float* radial_coeffs,
  int pair,
  RealT r,
  RealT* vals,
  RealT* ders);

template <typename RealT, int RadialFuncs, int RbSize>
__device__ __forceinline__ void sh_direct_radial_vals_ders_static(
  const SHDeviceModel& model,
  const float* radial_coeffs,
  int pair,
  RealT r,
  RealT* vals,
  RealT* ders)
{
	  if (model.radial_basis_kind == static_cast<int>(SHRadialBasisKind::LaguerreLog1p)) {
	    sh_laguerre_log1p_radial_vals_ders_static<RealT, RadialFuncs, RbSize>(
	      model, radial_coeffs, pair, r, vals, ders);
	    return;
	  }

  const RealT dr = r - static_cast<RealT>(model.max_dist);
  const RealT cutoff_f = dr * dr;
  const RealT cutoff_der = static_cast<RealT>(2.0) * dr;
#pragma unroll
  for (int mu = 0; mu < RadialFuncs; ++mu) {
    const size_t scal_base = (static_cast<size_t>(pair) * RadialFuncs + mu) * 2;
    const RealT scal = static_cast<RealT>(model.radial_direct_scal_s[scal_base + 0]);
    const RealT shift = static_cast<RealT>(model.radial_direct_scal_s[scal_base + 1]);
    const RealT z = static_cast<RealT>(0.5) * scal * (r - shift);
    const RealT ksi = tanh(z);
    const RealT mult = static_cast<RealT>(0.5) * scal * (static_cast<RealT>(1.0) - ksi * ksi);
    const size_t coeff_base = (static_cast<size_t>(pair) * RadialFuncs + mu) * RbSize;

    RealT prev = static_cast<RealT>(1.0);
    RealT prev_x = static_cast<RealT>(0.0);
	    RealT acc_s = static_cast<RealT>(radial_coeffs[coeff_base]);
	    RealT acc_sx = static_cast<RealT>(0.0);
	    RealT curr = ksi;
	    RealT curr_x = static_cast<RealT>(1.0);
	    RealT coeff = static_cast<RealT>(radial_coeffs[coeff_base + 1]);
    acc_s += coeff * curr;
    acc_sx += coeff * curr_x;
#pragma unroll
    for (int xi = 2; xi < RbSize; ++xi) {
      const RealT next = static_cast<RealT>(2.0) * ksi * curr - prev;
      const RealT next_x = static_cast<RealT>(2.0) * (curr + ksi * curr_x) - prev_x;
	      coeff = static_cast<RealT>(radial_coeffs[coeff_base + xi]);
      acc_s += coeff * next;
      acc_sx += coeff * next_x;
      prev = curr;
      prev_x = curr_x;
      curr = next;
      curr_x = next_x;
    }
    vals[mu] = cutoff_f * acc_s;
    if (ders != nullptr) {
      ders[mu] = cutoff_der * acc_s + cutoff_f * mult * acc_sx;
    }
  }
}

template <typename RealT, int RadialFuncs, int RbSize>
__device__ __forceinline__ void sh_direct_radial_vals_ders_static(
  const SHDeviceModel& model,
  int pair,
  RealT r,
  RealT* vals,
  RealT* ders)
{
  sh_direct_radial_vals_ders_static<RealT, RadialFuncs, RbSize>(
    model, model.radial_direct_coeffs, pair, r, vals, ders);
}

template <typename RealT>
__device__ __forceinline__ void sh_laguerre_log1p_radial_vals_ders_dynamic(
  const SHDeviceModel& model,
  const float* radial_coeffs,
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
    RealT rho = static_cast<RealT>(model.radial_direct_scal_s[scal_base + 1]);
    rho = rho > static_cast<RealT>(kLaguerreMinRho) ? rho : static_cast<RealT>(kLaguerreMinRho);

    const RealT log_term = log1p(r / rho);
    const RealT u = scal * log_term;
    const RealT u_r = scal / (rho + r);
    const RealT exp_factor = exp(static_cast<RealT>(-0.5) * u);
    const size_t coeff_base = (static_cast<size_t>(pair) * model.radial_funcs_count + mu) * model.rb_size;

    RealT phi_prev = static_cast<RealT>(0.0);
    RealT dphi_prev = static_cast<RealT>(0.0);
    RealT phi_curr = cutoff_f * exp_factor;
    RealT dphi_curr = cutoff_der * exp_factor - static_cast<RealT>(0.5) * u_r * phi_curr;
	    RealT acc_s = static_cast<RealT>(radial_coeffs[coeff_base]) * phi_curr;
	    RealT acc_sr = static_cast<RealT>(radial_coeffs[coeff_base]) * dphi_curr;

    for (int n = 0; n < model.rb_size - 1; ++n) {
      const RealT inv_np1 =
        static_cast<RealT>(1.0) / (static_cast<RealT>(n) + static_cast<RealT>(1.0));
      const RealT recurrence_coeff =
        (static_cast<RealT>(2.0 * n + 1.0) - u) * inv_np1;
      const RealT prev_coeff = static_cast<RealT>(n) * inv_np1;
      const RealT phi_next = recurrence_coeff * phi_curr - prev_coeff * phi_prev;
      const RealT dphi_next =
        -u_r * inv_np1 * phi_curr + recurrence_coeff * dphi_curr - prev_coeff * dphi_prev;
	      const RealT radial_coeff = static_cast<RealT>(radial_coeffs[coeff_base + n + 1]);
      acc_s += radial_coeff * phi_next;
      acc_sr += radial_coeff * dphi_next;
      phi_prev = phi_curr;
      dphi_prev = dphi_curr;
      phi_curr = phi_next;
      dphi_curr = dphi_next;
    }

    vals[mu] = acc_s;
    if (ders != nullptr) {
      ders[mu] = acc_sr;
    }
  }
}

template <typename RealT, int RadialFuncs, int RbSize>
__device__ __forceinline__ void sh_laguerre_log1p_radial_vals_ders_static(
  const SHDeviceModel& model,
  const float* radial_coeffs,
  int pair,
  RealT r,
  RealT* vals,
  RealT* ders)
{
  const RealT dr = r - static_cast<RealT>(model.max_dist);
  const RealT cutoff_f = dr * dr;
  const RealT cutoff_der = static_cast<RealT>(2.0) * dr;
#pragma unroll
  for (int mu = 0; mu < RadialFuncs; ++mu) {
    const size_t scal_base = (static_cast<size_t>(pair) * RadialFuncs + mu) * 2;
    const RealT scal = static_cast<RealT>(model.radial_direct_scal_s[scal_base + 0]);
    RealT rho = static_cast<RealT>(model.radial_direct_scal_s[scal_base + 1]);
    rho = rho > static_cast<RealT>(kLaguerreMinRho) ? rho : static_cast<RealT>(kLaguerreMinRho);

    const RealT log_term = log1p(r / rho);
    const RealT u = scal * log_term;
    const RealT u_r = scal / (rho + r);
    const RealT exp_factor = exp(static_cast<RealT>(-0.5) * u);
    const size_t coeff_base = (static_cast<size_t>(pair) * RadialFuncs + mu) * RbSize;

    RealT phi_prev = static_cast<RealT>(0.0);
    RealT dphi_prev = static_cast<RealT>(0.0);
    RealT phi_curr = cutoff_f * exp_factor;
    RealT dphi_curr = cutoff_der * exp_factor - static_cast<RealT>(0.5) * u_r * phi_curr;
	    RealT acc_s = static_cast<RealT>(radial_coeffs[coeff_base]) * phi_curr;
	    RealT acc_sr = static_cast<RealT>(radial_coeffs[coeff_base]) * dphi_curr;

#pragma unroll
    for (int n = 0; n < RbSize - 1; ++n) {
      const RealT inv_np1 =
        static_cast<RealT>(1.0) / (static_cast<RealT>(n) + static_cast<RealT>(1.0));
      const RealT recurrence_coeff =
        (static_cast<RealT>(2.0 * n + 1.0) - u) * inv_np1;
      const RealT prev_coeff = static_cast<RealT>(n) * inv_np1;
      const RealT phi_next = recurrence_coeff * phi_curr - prev_coeff * phi_prev;
      const RealT dphi_next =
        -u_r * inv_np1 * phi_curr + recurrence_coeff * dphi_curr - prev_coeff * dphi_prev;
	      const RealT radial_coeff = static_cast<RealT>(radial_coeffs[coeff_base + n + 1]);
      acc_s += radial_coeff * phi_next;
      acc_sr += radial_coeff * dphi_next;
      phi_prev = phi_curr;
      dphi_prev = dphi_curr;
      phi_curr = phi_next;
      dphi_curr = dphi_next;
    }

    vals[mu] = acc_s;
    if (ders != nullptr) {
      ders[mu] = acc_sr;
    }
  }
}

template <typename RealT>
__device__ __forceinline__ void sh_direct_radial_vals_ders(
  const SHDeviceModel& model,
  const float* radial_coeffs,
  int pair,
  RealT r,
  RealT* vals,
  RealT* ders)
{
  if (model.radial_basis_kind == static_cast<int>(SHRadialBasisKind::LaguerreLog1p)) {
    if (model.radial_funcs_count == 12 && model.rb_size == 10) {
      sh_laguerre_log1p_radial_vals_ders_static<RealT, 12, 10>(
        model, radial_coeffs, pair, r, vals, ders);
      return;
    }
    if (model.radial_funcs_count == 20 && model.rb_size == 10) {
      sh_laguerre_log1p_radial_vals_ders_static<RealT, 20, 10>(
        model, radial_coeffs, pair, r, vals, ders);
      return;
    }
    sh_laguerre_log1p_radial_vals_ders_dynamic(model, radial_coeffs, pair, r, vals, ders);
    return;
  }
  if (model.radial_funcs_count == 12 && model.rb_size == 10) {
    sh_direct_radial_vals_ders_static<RealT, 12, 10>(
      model, radial_coeffs, pair, r, vals, ders);
    return;
  }
  sh_direct_radial_vals_ders_dynamic(model, radial_coeffs, pair, r, vals, ders);
}

template <typename RealT>
__device__ __forceinline__ void sh_direct_radial_vals_ders(
  const SHDeviceModel& model,
  int pair,
  RealT r,
  RealT* vals,
  RealT* ders)
{
  sh_direct_radial_vals_ders(
    model, model.radial_direct_coeffs, pair, r, vals, ders);
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

template <typename RealT, int BasicCapacity = kMaxSHBasics>
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

  if (model.alpha_basic_count > BasicCapacity) {
    return;
  }

  RealT basic[BasicCapacity];
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

template <typename RealT, int L>
__device__ __forceinline__ void eval_real_sh_static_values(
  RealT x,
  RealT y,
  RealT z,
  RealT r,
  RealT* sh)
{
  if (L > 4) {
    eval_real_sh_from_solid(x, y, z, r, L, sh, static_cast<RealT*>(nullptr));
    return;
  }
  const RealT inv_r = static_cast<RealT>(1.0) / r;
  const RealT inv_r2 = inv_r * inv_r;
  const RealT inv_r3 = inv_r2 * inv_r;
  const RealT x2 = x * x;
  const RealT y2 = y * y;
  const RealT z2 = z * z;
  const RealT xy = x * y;
  const RealT xz = x * z;
  const RealT yz = y * z;

  sh[0] = static_cast<RealT>(0.5 / sqrt(kPi));
  if (L == 0) {
    return;
  }

  const RealT c1 = static_cast<RealT>(0.5 * sqrt(3.0 / kPi));
  sh[1] = c1 * y * inv_r;
  sh[2] = c1 * z * inv_r;
  sh[3] = c1 * x * inv_r;
  if (L == 1) {
    return;
  }

  const RealT c2a = static_cast<RealT>(0.5 * sqrt(15.0 / kPi));
  const RealT c20 = static_cast<RealT>(0.25 * sqrt(5.0 / kPi));
  const RealT c22 = static_cast<RealT>(0.25 * sqrt(15.0 / kPi));
  sh[4] = c2a * xy * inv_r2;
  sh[5] = c2a * yz * inv_r2;
  sh[6] = c20 * (static_cast<RealT>(2.0) * z2 - x2 - y2) * inv_r2;
  sh[7] = c2a * xz * inv_r2;
  sh[8] = c22 * (x2 - y2) * inv_r2;
  if (L == 2) {
    return;
  }

  const RealT c33 = static_cast<RealT>(0.125 * sqrt(70.0 / kPi));
  const RealT c32 = static_cast<RealT>(0.5 * sqrt(105.0 / kPi));
  const RealT c31 = static_cast<RealT>(0.125 * sqrt(42.0 / kPi));
  const RealT c30 = static_cast<RealT>(0.25 * sqrt(7.0 / kPi));
  const RealT c3p2 = static_cast<RealT>(0.25 * sqrt(105.0 / kPi));
  const RealT p3m3 = y * (static_cast<RealT>(3.0) * x2 - y2);
  const RealT a31 = static_cast<RealT>(4.0) * z2 - x2 - y2;
  const RealT p30 = z * (static_cast<RealT>(2.0) * z2 -
                         static_cast<RealT>(3.0) * x2 -
                         static_cast<RealT>(3.0) * y2);
  const RealT p32 = z * (x2 - y2);
  const RealT p33 = x * (x2 - static_cast<RealT>(3.0) * y2);
  sh[9] = c33 * p3m3 * inv_r3;
  sh[10] = c32 * xy * z * inv_r3;
  sh[11] = c31 * y * a31 * inv_r3;
  sh[12] = c30 * p30 * inv_r3;
  sh[13] = c31 * x * a31 * inv_r3;
  sh[14] = c3p2 * p32 * inv_r3;
  sh[15] = c33 * p33 * inv_r3;
  if (L == 3) {
    return;
  }

  const RealT inv_r4 = inv_r2 * inv_r2;
  const RealT c44m = static_cast<RealT>(0.75 * sqrt(35.0 / kPi));
  const RealT c43 = static_cast<RealT>(0.375 * sqrt(70.0 / kPi));
  const RealT c42m = static_cast<RealT>(0.75 * sqrt(5.0 / kPi));
  const RealT c41 = static_cast<RealT>(0.375 * sqrt(10.0 / kPi));
  const RealT c40 = static_cast<RealT>(0.1875 / sqrt(kPi));
  const RealT c42 = static_cast<RealT>(0.375 * sqrt(5.0 / kPi));
  const RealT c44 = static_cast<RealT>(0.1875 * sqrt(35.0 / kPi));
  const RealT rho2 = x2 + y2;
  const RealT p44base = x2 - y2;
  const RealT p4m4 = x * y * p44base;
  const RealT a42 = static_cast<RealT>(6.0) * z2 - rho2;
  const RealT a41 = static_cast<RealT>(4.0) * z2 - static_cast<RealT>(3.0) * rho2;
  const RealT p40 = static_cast<RealT>(8.0) * z2 * z2 -
                    static_cast<RealT>(24.0) * z2 * rho2 +
                    static_cast<RealT>(3.0) * rho2 * rho2;
  const RealT p44 = x2 * x2 - static_cast<RealT>(6.0) * x2 * y2 + y2 * y2;
  sh[16] = c44m * p4m4 * inv_r4;
  sh[17] = c43 * z * p3m3 * inv_r4;
  sh[18] = c42m * x * y * a42 * inv_r4;
  sh[19] = c41 * y * z * a41 * inv_r4;
  sh[20] = c40 * p40 * inv_r4;
  sh[21] = c41 * x * z * a41 * inv_r4;
  sh[22] = c42 * p44base * a42 * inv_r4;
  sh[23] = c43 * z * p33 * inv_r4;
  sh[24] = c44 * p44 * inv_r4;
}

template <typename RealT, int L, int K, int RbSize>
static __global__ void gpu_sh_compute_basic_static(
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
  constexpr int Components = (L + 1) * (L + 1);
  constexpr int RadialFuncs = K * (L + 1);
  constexpr int BasicCount = K * Components;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || model.alpha_basic_count != BasicCount) {
    return;
  }

  RealT acc[BasicCount];
#pragma unroll
  for (int b = 0; b < BasicCount; ++b) {
    acc[b] = static_cast<RealT>(0.0);
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
    const int pair = type_i * model.species_count + type[j];
    RealT radial_vals[RadialFuncs];
    RealT sh[Components];
    sh_direct_radial_vals_ders_static<RealT, RadialFuncs, RbSize>(
      model, pair, r, radial_vals, static_cast<RealT*>(nullptr));
    eval_real_sh_static_values<RealT, L>(dx, dy, dz, r, sh);

    int b = 0;
#pragma unroll
    for (int l = L; l >= 0; --l) {
#pragma unroll
      for (int k = K - 1; k >= 0; --k) {
        const RealT radial = radial_vals[k * (L + 1) + l];
        const int ybase = l * l;
#pragma unroll
        for (int c = 0; c < 2 * l + 1; ++c) {
          acc[b++] += radial * sh[ybase + c];
        }
      }
    }
  }

#pragma unroll
	  for (int b = 0; b < BasicCount; ++b) {
	    moments[static_cast<size_t>(b) * N + i] = acc[b];
	  }
	}

template <typename RealT, int L, int K, int RbSize>
static __global__ void gpu_sh_compute_basic_with_radial_static(
  int N,
  Box box,
  double cutoff_square,
  bool use_cached_displacements,
  SHDeviceModel model,
  const float* radial_coeffs,
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
  constexpr int Components = (L + 1) * (L + 1);
  constexpr int RadialFuncs = K * (L + 1);
  constexpr int BasicCount = K * Components;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || model.alpha_basic_count != BasicCount ||
      model.radial_funcs_count != RadialFuncs || model.rb_size != RbSize) {
    return;
  }

  RealT acc[BasicCount];
#pragma unroll
  for (int b = 0; b < BasicCount; ++b) {
    acc[b] = static_cast<RealT>(0.0);
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
    const int pair = type_i * model.species_count + type[j];
    RealT radial_vals[RadialFuncs];
    RealT sh[Components];
    sh_direct_radial_vals_ders_static<RealT, RadialFuncs, RbSize>(
      model, radial_coeffs, pair, r, radial_vals, static_cast<RealT*>(nullptr));
    eval_real_sh_static_values<RealT, L>(dx, dy, dz, r, sh);

    int b = 0;
#pragma unroll
    for (int l = L; l >= 0; --l) {
#pragma unroll
      for (int k = K - 1; k >= 0; --k) {
        const RealT radial = radial_vals[k * (L + 1) + l];
        const int ybase = l * l;
#pragma unroll
        for (int c = 0; c < 2 * l + 1; ++c) {
          acc[b++] += radial * sh[ybase + c];
        }
      }
    }
  }

#pragma unroll
  for (int b = 0; b < BasicCount; ++b) {
    moments[static_cast<size_t>(b) * N + i] = acc[b];
  }
}

template <typename RealT, int L, int K, int RbSize>
static __global__ void gpu_sh_compute_basic_gate_main_static(
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
  const RealT* gate_values,
  RealT* moments)
{
  constexpr int Components = (L + 1) * (L + 1);
  constexpr int RadialFuncs = K * (L + 1);
  constexpr int BasicCount = K * Components;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || model.alpha_basic_count != BasicCount ||
      model.radial_funcs_count != RadialFuncs || model.rb_size != RbSize) {
    return;
  }

  RealT acc[BasicCount];
#pragma unroll
  for (int b = 0; b < BasicCount; ++b) {
    acc[b] = static_cast<RealT>(0.0);
  }

  const int type_i = type[i];
  const int count = neighbor_count[i];
  const RealT amplitude = static_cast<RealT>(model.two_layer_gate_tanh_amplitude);
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
    const RealT gate_residual = gate_values[j];
    RealT radial_vals[RadialFuncs];
    RealT sh[Components];
    sh_direct_radial_vals_ders_static<RealT, RadialFuncs, RbSize>(
      model, model.radial_direct_coeffs, pair, r, radial_vals, static_cast<RealT*>(nullptr));
    eval_real_sh_static_values<RealT, L>(dx, dy, dz, r, sh);

    int b = 0;
#pragma unroll
    for (int l = L; l >= 0; --l) {
#pragma unroll
      for (int k = K - 1; k >= 0; --k) {
        const int mu = k * (L + 1) + l;
        const RealT a = static_cast<RealT>(
          model.two_layer_gate_additive_coeffs_float[type_j * RadialFuncs + mu]);
        const RealT radial =
          radial_vals[mu] * (static_cast<RealT>(1.0) + amplitude * tanh(a * gate_residual));
        const int ybase = l * l;
#pragma unroll
        for (int c = 0; c < 2 * l + 1; ++c) {
          acc[b++] += radial * sh[ybase + c];
        }
      }
    }
  }

#pragma unroll
  for (int b = 0; b < BasicCount; ++b) {
    moments[static_cast<size_t>(b) * N + i] = acc[b];
  }
}

template <typename RealT>
bool launch_sh_compute_basic_static(
  int lmax,
  int kmax,
  int rb_size,
  int grid_size,
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
  if (rb_size != 10 || kmax < 1 || kmax > 6 || lmax < 0 || lmax > kMaxSHL) {
    return false;
  }

#define SUS2_SH_LAUNCH_STATIC_BASIC_FOR_L(LVAL)                                           \
  do {                                                                                    \
    switch (kmax) {                                                                       \
      case 1:                                                                             \
        gpu_sh_compute_basic_static<RealT, LVAL, 1, 10><<<grid_size, kBlockSize>>>(       \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, moments);        \
        return true;                                                                      \
      case 2:                                                                             \
        gpu_sh_compute_basic_static<RealT, LVAL, 2, 10><<<grid_size, kBlockSize>>>(       \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, moments);        \
        return true;                                                                      \
      case 3:                                                                             \
        gpu_sh_compute_basic_static<RealT, LVAL, 3, 10><<<grid_size, kBlockSize>>>(       \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, moments);        \
        return true;                                                                      \
      case 4:                                                                             \
        gpu_sh_compute_basic_static<RealT, LVAL, 4, 10><<<grid_size, kBlockSize>>>(       \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, moments);        \
        return true;                                                                      \
      case 5:                                                                             \
        gpu_sh_compute_basic_static<RealT, LVAL, 5, 10><<<grid_size, kBlockSize>>>(       \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, moments);        \
        return true;                                                                      \
      case 6:                                                                             \
        gpu_sh_compute_basic_static<RealT, LVAL, 6, 10><<<grid_size, kBlockSize>>>(       \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, moments);        \
        return true;                                                                      \
    }                                                                                     \
  } while (0)

  switch (lmax) {
    case 0:
      SUS2_SH_LAUNCH_STATIC_BASIC_FOR_L(0);
      break;
    case 1:
      SUS2_SH_LAUNCH_STATIC_BASIC_FOR_L(1);
      break;
    case 2:
      SUS2_SH_LAUNCH_STATIC_BASIC_FOR_L(2);
      break;
    case 3:
      SUS2_SH_LAUNCH_STATIC_BASIC_FOR_L(3);
      break;
    case 4:
      SUS2_SH_LAUNCH_STATIC_BASIC_FOR_L(4);
      break;
    case 5:
      SUS2_SH_LAUNCH_STATIC_BASIC_FOR_L(5);
      break;
    case 6:
      SUS2_SH_LAUNCH_STATIC_BASIC_FOR_L(6);
      break;
  }

#undef SUS2_SH_LAUNCH_STATIC_BASIC_FOR_L
  return false;
}

template <typename RealT>
bool launch_sh_compute_basic_with_radial_static(
  int lmax,
  int kmax,
  int rb_size,
  int grid_size,
  int N,
  Box box,
  double cutoff_square,
  bool use_cached_displacements,
  SHDeviceModel model,
  const float* radial_coeffs,
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
  if (rb_size != 10 || kmax < 1 || kmax > 6 || lmax < 0 || lmax > kMaxSHL) {
    return false;
  }

#define SUS2_SH_LAUNCH_STATIC_BASIC_RADIAL_FOR_L(LVAL)                                    \
  do {                                                                                    \
    switch (kmax) {                                                                       \
      case 1:                                                                             \
        gpu_sh_compute_basic_with_radial_static<RealT, LVAL, 1, 10>                       \
          <<<grid_size, kBlockSize>>>(                                                    \
          N, box, cutoff_square, use_cached_displacements, model, radial_coeffs, type,    \
          neighbor_count, neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz,           \
          x, y, z, moments);                                                              \
        return true;                                                                      \
      case 2:                                                                             \
        gpu_sh_compute_basic_with_radial_static<RealT, LVAL, 2, 10>                       \
          <<<grid_size, kBlockSize>>>(                                                    \
          N, box, cutoff_square, use_cached_displacements, model, radial_coeffs, type,    \
          neighbor_count, neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz,           \
          x, y, z, moments);                                                              \
        return true;                                                                      \
      case 3:                                                                             \
        gpu_sh_compute_basic_with_radial_static<RealT, LVAL, 3, 10>                       \
          <<<grid_size, kBlockSize>>>(                                                    \
          N, box, cutoff_square, use_cached_displacements, model, radial_coeffs, type,    \
          neighbor_count, neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz,           \
          x, y, z, moments);                                                              \
        return true;                                                                      \
      case 4:                                                                             \
        gpu_sh_compute_basic_with_radial_static<RealT, LVAL, 4, 10>                       \
          <<<grid_size, kBlockSize>>>(                                                    \
          N, box, cutoff_square, use_cached_displacements, model, radial_coeffs, type,    \
          neighbor_count, neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz,           \
          x, y, z, moments);                                                              \
        return true;                                                                      \
      case 5:                                                                             \
        gpu_sh_compute_basic_with_radial_static<RealT, LVAL, 5, 10>                       \
          <<<grid_size, kBlockSize>>>(                                                    \
          N, box, cutoff_square, use_cached_displacements, model, radial_coeffs, type,    \
          neighbor_count, neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz,           \
          x, y, z, moments);                                                              \
        return true;                                                                      \
      case 6:                                                                             \
        gpu_sh_compute_basic_with_radial_static<RealT, LVAL, 6, 10>                       \
          <<<grid_size, kBlockSize>>>(                                                    \
          N, box, cutoff_square, use_cached_displacements, model, radial_coeffs, type,    \
          neighbor_count, neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz,           \
          x, y, z, moments);                                                              \
        return true;                                                                      \
    }                                                                                     \
  } while (0)

  switch (lmax) {
    case 0:
      SUS2_SH_LAUNCH_STATIC_BASIC_RADIAL_FOR_L(0);
      break;
    case 1:
      SUS2_SH_LAUNCH_STATIC_BASIC_RADIAL_FOR_L(1);
      break;
    case 2:
      SUS2_SH_LAUNCH_STATIC_BASIC_RADIAL_FOR_L(2);
      break;
    case 3:
      SUS2_SH_LAUNCH_STATIC_BASIC_RADIAL_FOR_L(3);
      break;
    case 4:
      SUS2_SH_LAUNCH_STATIC_BASIC_RADIAL_FOR_L(4);
      break;
    case 5:
      SUS2_SH_LAUNCH_STATIC_BASIC_RADIAL_FOR_L(5);
      break;
    case 6:
      SUS2_SH_LAUNCH_STATIC_BASIC_RADIAL_FOR_L(6);
      break;
  }

#undef SUS2_SH_LAUNCH_STATIC_BASIC_RADIAL_FOR_L
  return false;
}

template <typename RealT>
bool launch_sh_compute_basic_gate_main_static(
  int lmax,
  int kmax,
  int rb_size,
  int grid_size,
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
  const RealT* gate_values,
  RealT* moments)
{
  if (rb_size != 10 || kmax < 1 || kmax > 6 || lmax < 0 || lmax > kMaxSHL) {
    return false;
  }

#define SUS2_SH_LAUNCH_STATIC_BASIC_GATE_MAIN_FOR_L(LVAL)                                \
  do {                                                                                    \
    switch (kmax) {                                                                       \
      case 1:                                                                             \
        gpu_sh_compute_basic_gate_main_static<RealT, LVAL, 1, 10>                         \
          <<<grid_size, kBlockSize>>>(                                                    \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_values,    \
          moments);                                                                       \
        return true;                                                                      \
      case 2:                                                                             \
        gpu_sh_compute_basic_gate_main_static<RealT, LVAL, 2, 10>                         \
          <<<grid_size, kBlockSize>>>(                                                    \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_values,    \
          moments);                                                                       \
        return true;                                                                      \
      case 3:                                                                             \
        gpu_sh_compute_basic_gate_main_static<RealT, LVAL, 3, 10>                         \
          <<<grid_size, kBlockSize>>>(                                                    \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_values,    \
          moments);                                                                       \
        return true;                                                                      \
      case 4:                                                                             \
        gpu_sh_compute_basic_gate_main_static<RealT, LVAL, 4, 10>                         \
          <<<grid_size, kBlockSize>>>(                                                    \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_values,    \
          moments);                                                                       \
        return true;                                                                      \
      case 5:                                                                             \
        gpu_sh_compute_basic_gate_main_static<RealT, LVAL, 5, 10>                         \
          <<<grid_size, kBlockSize>>>(                                                    \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_values,    \
          moments);                                                                       \
        return true;                                                                      \
      case 6:                                                                             \
        gpu_sh_compute_basic_gate_main_static<RealT, LVAL, 6, 10>                         \
          <<<grid_size, kBlockSize>>>(                                                    \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_values,    \
          moments);                                                                       \
        return true;                                                                      \
    }                                                                                     \
  } while (0)

  switch (lmax) {
    case 0:
      SUS2_SH_LAUNCH_STATIC_BASIC_GATE_MAIN_FOR_L(0);
      break;
    case 1:
      SUS2_SH_LAUNCH_STATIC_BASIC_GATE_MAIN_FOR_L(1);
      break;
    case 2:
      SUS2_SH_LAUNCH_STATIC_BASIC_GATE_MAIN_FOR_L(2);
      break;
    case 3:
      SUS2_SH_LAUNCH_STATIC_BASIC_GATE_MAIN_FOR_L(3);
      break;
    case 4:
      SUS2_SH_LAUNCH_STATIC_BASIC_GATE_MAIN_FOR_L(4);
      break;
    case 5:
      SUS2_SH_LAUNCH_STATIC_BASIC_GATE_MAIN_FOR_L(5);
      break;
    case 6:
      SUS2_SH_LAUNCH_STATIC_BASIC_GATE_MAIN_FOR_L(6);
      break;
  }

#undef SUS2_SH_LAUNCH_STATIC_BASIC_GATE_MAIN_FOR_L
  return false;
}

template <typename RealT, int BasicCapacity = kMaxSHBasics>
static __global__ void gpu_sh_compute_basic_with_radial(
  int N,
  Box box,
  double cutoff_square,
  bool use_cached_displacements,
  SHDeviceModel model,
  const float* radial_coeffs,
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
  if (i >= N || model.alpha_basic_count > BasicCapacity) {
    return;
  }

  RealT basic[BasicCapacity];
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
    sh_direct_radial_vals_ders(
      model, radial_coeffs, pair, r, radial_vals, static_cast<RealT*>(nullptr));
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

template <typename RealT, int BasicCapacity = kMaxSHBasics>
static __global__ void gpu_sh_compute_basic_gate_main(
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
  const RealT* gate_values,
  RealT* moments)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || model.alpha_basic_count > BasicCapacity) {
    return;
  }

  RealT basic[BasicCapacity];
  for (int b = 0; b < model.alpha_basic_count; ++b) {
    basic[b] = static_cast<RealT>(0.0);
  }

  const int type_i = type[i];
  const int count = neighbor_count[i];
  const RealT amplitude = static_cast<RealT>(model.two_layer_gate_tanh_amplitude);
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
    const RealT gate_residual = gate_values[j];
    RealT radial_vals[kMaxSHRadialFuncs];
    RealT sh_vals[kMaxSHComponents];
    sh_direct_radial_vals_ders(
      model, model.radial_direct_coeffs, pair, r, radial_vals, static_cast<RealT*>(nullptr));
    for (int mu = 0; mu < model.radial_funcs_count; ++mu) {
      const RealT a = static_cast<RealT>(
        model.two_layer_gate_additive_coeffs_float[type_j * model.radial_funcs_count + mu]);
      radial_vals[mu] *= static_cast<RealT>(1.0) + amplitude * tanh(a * gate_residual);
    }
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
static __global__ void gpu_sh_gate_forward_backward_products(
  int N,
  SHDeviceModel model,
  RealT* moments,
  GradT* grads,
  RealT* gate_values)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  for (int p = 0; p < model.two_layer_gate_product_limit; ++p) {
    const int left = model.sh_products_int[p * 3 + 0];
    const int right = model.sh_products_int[p * 3 + 1];
    const int target = model.sh_products_int[p * 3 + 2];
    const RealT coeff = sh_product_coeff<RealT>(model, p);
    moments[static_cast<size_t>(target) * N + i] +=
      coeff * moments[static_cast<size_t>(left) * N + i] *
      moments[static_cast<size_t>(right) * N + i];
  }

  RealT residual = static_cast<RealT>(0.0);
  for (int q = 0; q < model.two_layer_gate_weight_count; ++q) {
    const int moment_id = model.two_layer_gate_moment_indices[q];
    const RealT weight = static_cast<RealT>(model.two_layer_gate_weights_float[q]);
    residual += weight * moments[static_cast<size_t>(moment_id) * N + i];
    grads[static_cast<size_t>(moment_id) * N + i] += static_cast<GradT>(weight);
  }
  gate_values[i] = residual;

  for (int p = model.two_layer_gate_product_limit - 1; p >= 0; --p) {
    const int left = model.sh_products_int[p * 3 + 0];
    const int right = model.sh_products_int[p * 3 + 1];
    const int target = model.sh_products_int[p * 3 + 2];
    const RealT coeff = sh_product_coeff<RealT>(model, p);
    const RealT gtarget =
      static_cast<RealT>(grads[static_cast<size_t>(target) * N + i]) * coeff;
    if (gtarget == static_cast<RealT>(0.0)) {
      continue;
    }
    grads[static_cast<size_t>(left) * N + i] +=
      static_cast<GradT>(gtarget * moments[static_cast<size_t>(right) * N + i]);
    grads[static_cast<size_t>(right) * N + i] +=
      static_cast<GradT>(gtarget * moments[static_cast<size_t>(left) * N + i]);
  }
}

template <typename GradT>
static __global__ void gpu_sh_copy_basic_grads(
  int N,
  int basic_count,
  const GradT* grads,
  GradT* basic_grads)
{
  const size_t total = static_cast<size_t>(N) * basic_count;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += stride) {
    basic_grads[idx] = grads[idx];
  }
}

template <typename RealT, typename GradT, int PatternRows>
static __global__ void gpu_sh_gate_forward_backward_compact_rows(
  int N,
  SHDeviceModel model,
  RealT* moments,
  GradT* grads,
  RealT* gate_values)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= N) {
    return;
  }
  if (model.two_layer_gate_needed_moment_flags == nullptr ||
      model.two_layer_gate_moment_weights_float == nullptr) {
    return;
  }

  RealT residual = static_cast<RealT>(0.0);
  for (int layer = 1; layer <= model.sh_cg_layer_count; ++layer) {
    const int row_begin = model.sh_cg_layer_offsets[layer];
    const int row_end = model.sh_cg_layer_offsets[layer + 1];
    for (int row = row_begin; row < row_end; ++row) {
      int left_base;
      int right_base;
      int target;
      int term_begin;
      int term_count;
      if (PatternRows > 0) {
        const unsigned int* pattern_u32 =
          model.use_const_pattern_rows ? c_sh_forward_u32 : model.sh_cg_row_pattern_u32;
        const size_t pattern_row = static_cast<size_t>(row) * 2;
        const unsigned int row0 = pattern_u32[pattern_row + 0];
        const unsigned int row1 = pattern_u32[pattern_row + 1];
        left_base = static_cast<int>(row0 & 0xffffu);
        right_base = static_cast<int>(row0 >> 16);
        target = static_cast<int>(row1 & 0xffffu);
        const int pattern_id = static_cast<int>(row1 >> 16);
        const size_t header_base =
          static_cast<size_t>(model.sh_cg_row_count) * 2 +
          static_cast<size_t>(pattern_id) * 2;
        term_begin = static_cast<int>(pattern_u32[header_base + 0]);
        term_count = static_cast<int>(pattern_u32[header_base + 1]);
      } else {
        const int row_base = row * 5;
        left_base = model.sh_cg_rows_int[row_base + 0];
        right_base = model.sh_cg_rows_int[row_base + 1];
        target = model.sh_cg_rows_int[row_base + 2];
        term_begin = model.sh_cg_rows_int[row_base + 3];
        term_count = model.sh_cg_rows_int[row_base + 4];
      }
      if (target < 0 || target >= model.alpha_moments_count ||
          model.two_layer_gate_needed_moment_flags[target] == 0) {
        continue;
      }

      const RealT gate_weight =
        static_cast<RealT>(model.two_layer_gate_moment_weights_float[target]);
      const bool terminal_gate_row =
        gate_weight != static_cast<RealT>(0.0) &&
        model.sh_terminal_moment_flags != nullptr &&
        model.sh_terminal_moment_flags[target] != 0;
      RealT sum = static_cast<RealT>(0.0);
      for (int t = 0; t < term_count; ++t) {
        int left_component;
        int right_component;
        RealT coeff;
        if (PatternRows > 0) {
          const unsigned int* pattern_u32 =
            model.use_const_pattern_rows ? c_sh_forward_u32 : model.sh_cg_row_pattern_u32;
          const size_t pattern_term_base =
            static_cast<size_t>(model.sh_cg_row_count) * 2 +
            static_cast<size_t>(model.sh_cg_row_pattern_count) * 2 +
            static_cast<size_t>(term_begin + t) * 2;
          const unsigned int term_meta = pattern_u32[pattern_term_base + 0];
          left_component = static_cast<int>(term_meta & 0xffffu);
          right_component = static_cast<int>(term_meta >> 16);
          coeff = sh_const_forward_coeff<RealT>(pattern_u32[pattern_term_base + 1]);
        } else {
          const int term = term_begin + t;
          const int term_base = term * 2;
          left_component = model.sh_cg_row_terms_int[term_base + 0];
          right_component = model.sh_cg_row_terms_int[term_base + 1];
          coeff = sh_cg_row_term_coeff<RealT>(model, term);
        }
        const int left = left_base + left_component;
        const int right = right_base + right_component;
        const RealT left_value = moments[static_cast<size_t>(left) * N + atom];
        const RealT right_value = moments[static_cast<size_t>(right) * N + atom];
        sum += coeff * left_value * right_value;
        if (terminal_gate_row) {
          const RealT weighted = gate_weight * coeff;
          grads[static_cast<size_t>(left) * N + atom] +=
            static_cast<GradT>(weighted * right_value);
          grads[static_cast<size_t>(right) * N + atom] +=
            static_cast<GradT>(weighted * left_value);
        }
      }
      if (terminal_gate_row) {
        residual += gate_weight * sum;
      } else {
        moments[static_cast<size_t>(target) * N + atom] = sum;
      }
    }
  }

  for (int q = 0; q < model.two_layer_gate_weight_count; ++q) {
    const int moment_id = model.two_layer_gate_moment_indices[q];
    if (moment_id < 0 || moment_id >= model.alpha_moments_count) {
      continue;
    }
    const bool terminal_gate_moment =
      model.sh_terminal_moment_flags != nullptr &&
      model.sh_terminal_moment_flags[moment_id] != 0;
    if (terminal_gate_moment) {
      continue;
    }
    const RealT weight = static_cast<RealT>(model.two_layer_gate_weights_float[q]);
    residual += weight * moments[static_cast<size_t>(moment_id) * N + atom];
    grads[static_cast<size_t>(moment_id) * N + atom] += static_cast<GradT>(weight);
  }
  gate_values[atom] = residual;

  for (int layer = model.sh_cg_layer_count; layer >= 1; --layer) {
    const int row_begin = model.sh_cg_back_layer_offsets[layer];
    const int row_end = model.sh_cg_back_layer_offsets[layer + 1];
    for (int row = row_begin; row < row_end; ++row) {
      int source;
      int term_begin;
      int term_count;
      if (model.use_packed_back_rows) {
        const unsigned int* back_u32 =
          model.use_const_back_rows
            ? c_sh_forward_u32 + sh_const_back_u32_offset(model)
            : model.sh_cg_back_packed_u32;
        const unsigned int row0 = back_u32[static_cast<size_t>(row) * 2 + 0];
        source = static_cast<int>(row0 & 0xffffu);
        term_count = static_cast<int>(row0 >> 16);
        term_begin = static_cast<int>(back_u32[static_cast<size_t>(row) * 2 + 1]);
      } else {
        const int row_base = row * 3;
        source = model.sh_cg_back_rows_int[row_base + 0];
        term_begin = model.sh_cg_back_rows_int[row_base + 1];
        term_count = model.sh_cg_back_rows_int[row_base + 2];
      }
      if (source < 0 || source >= model.alpha_moments_count ||
          model.two_layer_gate_needed_moment_flags[source] == 0) {
        continue;
      }
      RealT sum = static_cast<RealT>(0.0);
      for (int t = 0; t < term_count; ++t) {
        const int term = term_begin + t;
        int target;
        int other;
        RealT coeff;
        if (model.use_packed_back_rows) {
          const unsigned int* back_u32 =
            model.use_const_back_rows
              ? c_sh_forward_u32 + sh_const_back_u32_offset(model)
              : model.sh_cg_back_packed_u32;
          const size_t packed_base =
            static_cast<size_t>(model.sh_cg_back_row_count) * 2 +
            static_cast<size_t>(term) * 2;
          const unsigned int meta = back_u32[packed_base + 0];
          target = static_cast<int>(meta & 0xffffu);
          other = static_cast<int>(meta >> 16);
          coeff = sh_const_forward_coeff<RealT>(back_u32[packed_base + 1]);
        } else {
          const int term_base = term * 2;
          target = model.sh_cg_back_terms_int[term_base + 0];
          other = model.sh_cg_back_terms_int[term_base + 1];
          coeff = sh_cg_back_term_coeff<RealT>(model, term);
        }
        if (target < 0 || target >= model.alpha_moments_count ||
            model.two_layer_gate_needed_moment_flags[target] == 0) {
          continue;
        }
        sum += coeff * static_cast<RealT>(grads[static_cast<size_t>(target) * N + atom]) *
               moments[static_cast<size_t>(other) * N + atom];
      }
      grads[static_cast<size_t>(source) * N + atom] += static_cast<GradT>(sum);
    }
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
  for (int s = 0; s < model.active_scalar_moments; ++s) {
    const int moment_id = model.active_scalar_moment[s];
    const RealT coeff = sh_active_scalar_coeff<RealT>(model, s);
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

template <typename RealT, typename GradT>
static __global__ void gpu_sh_forward_energy_backward_cg_blocks(
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

  for (int b = 0; b < model.sh_cg_block_count; ++b) {
    const int block_base = b * 6;
    const int left_base = model.sh_cg_blocks_int[block_base + 0];
    const int right_base = model.sh_cg_blocks_int[block_base + 1];
    const int target_base = model.sh_cg_blocks_int[block_base + 2];
    const int L = model.sh_cg_blocks_int[block_base + 3];
    const int term_begin = model.sh_cg_blocks_int[block_base + 4];
    const int term_count = model.sh_cg_blocks_int[block_base + 5];
    for (int c = 0; c < 2 * L + 1; ++c) {
      moments[static_cast<size_t>(target_base + c) * N + i] = static_cast<RealT>(0.0);
    }
    for (int t = 0; t < term_count; ++t) {
      const int term = term_begin + t;
      const int term_base = term * 3;
      const int left = left_base + model.sh_cg_terms_int[term_base + 0];
      const int right = right_base + model.sh_cg_terms_int[term_base + 1];
      const int target = target_base + model.sh_cg_terms_int[term_base + 2];
      const RealT coeff = sh_cg_term_coeff<RealT>(model, term);
      moments[static_cast<size_t>(target) * N + i] +=
        coeff * moments[static_cast<size_t>(left) * N + i] *
        moments[static_cast<size_t>(right) * N + i];
    }
  }

  const int type_i = type[i];
  const RealT center_coeff = sh_species_coeff<RealT>(model, type_i);
  RealT site_energy = sh_shift_coeff<RealT>(model, type_i) + center_coeff;
  for (int s = 0; s < model.active_scalar_moments; ++s) {
    const int moment_id = model.active_scalar_moment[s];
    const RealT coeff = sh_active_scalar_coeff<RealT>(model, s);
    site_energy += center_coeff * coeff * moments[static_cast<size_t>(moment_id) * N + i];
    grads[static_cast<size_t>(moment_id) * N + i] += static_cast<GradT>(center_coeff * coeff);
  }
  potential[i] += static_cast<double>(site_energy);

  for (int b = model.sh_cg_block_count - 1; b >= 0; --b) {
    const int block_base = b * 6;
    const int left_base = model.sh_cg_blocks_int[block_base + 0];
    const int right_base = model.sh_cg_blocks_int[block_base + 1];
    const int target_base = model.sh_cg_blocks_int[block_base + 2];
    const int term_begin = model.sh_cg_blocks_int[block_base + 4];
    const int term_count = model.sh_cg_blocks_int[block_base + 5];
    for (int t = term_count - 1; t >= 0; --t) {
      const int term = term_begin + t;
      const int term_base = term * 3;
      const int left = left_base + model.sh_cg_terms_int[term_base + 0];
      const int right = right_base + model.sh_cg_terms_int[term_base + 1];
      const int target = target_base + model.sh_cg_terms_int[term_base + 2];
      const RealT coeff = sh_cg_term_coeff<RealT>(model, term);
      const RealT gtarget =
        static_cast<RealT>(grads[static_cast<size_t>(target) * N + i]) * coeff;
      grads[static_cast<size_t>(left) * N + i] +=
        static_cast<GradT>(gtarget * moments[static_cast<size_t>(right) * N + i]);
      grads[static_cast<size_t>(right) * N + i] +=
        static_cast<GradT>(gtarget * moments[static_cast<size_t>(left) * N + i]);
    }
  }
}

template <typename RealT>
static __global__ void gpu_sh_tensor_product_rows_forward(
  int N,
  int layer,
  SHDeviceModel model,
  RealT* moments)
{
  const int row_begin = model.sh_cg_layer_offsets[layer];
  const int row_end = model.sh_cg_layer_offsets[layer + 1];
  const size_t row_count = static_cast<size_t>(row_end - row_begin);
  const size_t total = row_count * static_cast<size_t>(N);
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t task = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < total;
       task += stride) {
    const int atom = static_cast<int>(task % static_cast<size_t>(N));
    const int row = row_begin + static_cast<int>(task / static_cast<size_t>(N));
    const int row_base = row * 5;
    const int left_base = model.sh_cg_rows_int[row_base + 0];
    const int right_base = model.sh_cg_rows_int[row_base + 1];
    const int target = model.sh_cg_rows_int[row_base + 2];
    const int term_begin = model.sh_cg_rows_int[row_base + 3];
    const int term_count = model.sh_cg_rows_int[row_base + 4];
    RealT sum = static_cast<RealT>(0.0);
    for (int t = 0; t < term_count; ++t) {
      const int term = term_begin + t;
      const int term_base = term * 2;
      const int left = left_base + model.sh_cg_row_terms_int[term_base + 0];
      const int right = right_base + model.sh_cg_row_terms_int[term_base + 1];
      const RealT coeff = sh_cg_row_term_coeff<RealT>(model, term);
      sum += coeff * moments[static_cast<size_t>(left) * N + atom] *
             moments[static_cast<size_t>(right) * N + atom];
    }
    moments[static_cast<size_t>(target) * N + atom] = sum;
  }
}

template <typename RealT, typename GradT>
static __global__ void gpu_sh_energy_init_from_scalars(
  int N,
  SHDeviceModel model,
  const int* type,
  const RealT* moments,
  GradT* grads,
  double* potential)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }
  const int type_i = type[i];
  const RealT center_coeff = sh_species_coeff<RealT>(model, type_i);
  RealT site_energy = sh_shift_coeff<RealT>(model, type_i) + center_coeff;
  for (int s = 0; s < model.active_scalar_moments; ++s) {
    const int moment_id = model.active_scalar_moment[s];
    const RealT coeff = sh_active_scalar_coeff<RealT>(model, s);
    site_energy += center_coeff * coeff * moments[static_cast<size_t>(moment_id) * N + i];
    grads[static_cast<size_t>(moment_id) * N + i] += static_cast<GradT>(center_coeff * coeff);
  }
  potential[i] += static_cast<double>(site_energy);
}

template <typename RealT, typename GradT>
static __global__ void gpu_sh_tensor_product_rows_backward_atomic(
  int N,
  int layer,
  SHDeviceModel model,
  const RealT* moments,
  GradT* grads)
{
  const int row_begin = model.sh_cg_layer_offsets[layer];
  const int row_end = model.sh_cg_layer_offsets[layer + 1];
  const size_t row_count = static_cast<size_t>(row_end - row_begin);
  const size_t total = row_count * static_cast<size_t>(N);
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t task = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < total;
       task += stride) {
    const int atom = static_cast<int>(task % static_cast<size_t>(N));
    const int row = row_begin + static_cast<int>(task / static_cast<size_t>(N));
    const int row_base = row * 5;
    const int left_base = model.sh_cg_rows_int[row_base + 0];
    const int right_base = model.sh_cg_rows_int[row_base + 1];
    const int target = model.sh_cg_rows_int[row_base + 2];
    const int term_begin = model.sh_cg_rows_int[row_base + 3];
    const int term_count = model.sh_cg_rows_int[row_base + 4];
    const RealT gtarget = static_cast<RealT>(grads[static_cast<size_t>(target) * N + atom]);
    if (gtarget == static_cast<RealT>(0.0)) {
      continue;
    }
    for (int t = 0; t < term_count; ++t) {
      const int term = term_begin + t;
      const int term_base = term * 2;
      const int left = left_base + model.sh_cg_row_terms_int[term_base + 0];
      const int right = right_base + model.sh_cg_row_terms_int[term_base + 1];
      const RealT coeff = sh_cg_row_term_coeff<RealT>(model, term) * gtarget;
      atomicAdd(
        grads + static_cast<size_t>(left) * N + atom,
        static_cast<GradT>(coeff * moments[static_cast<size_t>(right) * N + atom]));
      atomicAdd(
        grads + static_cast<size_t>(right) * N + atom,
        static_cast<GradT>(coeff * moments[static_cast<size_t>(left) * N + atom]));
    }
  }
}

template <typename RealT, typename GradT>
static __global__ void gpu_sh_tensor_product_back_rows(
  int N,
  int layer,
  SHDeviceModel model,
  const RealT* moments,
  GradT* grads)
{
  const int row_begin = model.sh_cg_back_layer_offsets[layer];
  const int row_end = model.sh_cg_back_layer_offsets[layer + 1];
  const size_t row_count = static_cast<size_t>(row_end - row_begin);
  const size_t total = row_count * static_cast<size_t>(N);
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t task = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < total;
       task += stride) {
    const int atom = static_cast<int>(task % static_cast<size_t>(N));
    const int row = row_begin + static_cast<int>(task / static_cast<size_t>(N));
    int source;
    int term_begin;
    int term_count;
    if (model.use_packed_back_rows) {
      const unsigned int* back_u32 =
        model.use_const_back_rows
        ? c_sh_forward_u32 + sh_const_back_u32_offset(model)
        : model.sh_cg_back_packed_u32;
      const unsigned int row0 = back_u32[static_cast<size_t>(row) * 2 + 0];
      source = static_cast<int>(row0 & 0xffffu);
      term_count = static_cast<int>(row0 >> 16);
      term_begin = static_cast<int>(back_u32[static_cast<size_t>(row) * 2 + 1]);
    } else {
      const int row_base = row * 3;
      source = model.sh_cg_back_rows_int[row_base + 0];
      term_begin = model.sh_cg_back_rows_int[row_base + 1];
      term_count = model.sh_cg_back_rows_int[row_base + 2];
    }
    RealT sum = static_cast<RealT>(0.0);
    for (int t = 0; t < term_count; ++t) {
      const int term = term_begin + t;
      int target;
      int other;
      RealT coeff;
      if (model.use_packed_back_rows) {
        const unsigned int* back_u32 =
          model.use_const_back_rows
          ? c_sh_forward_u32 + sh_const_back_u32_offset(model)
          : model.sh_cg_back_packed_u32;
        const size_t packed_base =
          static_cast<size_t>(model.sh_cg_back_row_count) * 2 +
          static_cast<size_t>(term) * 2;
        const unsigned int meta = back_u32[packed_base + 0];
        target = static_cast<int>(meta & 0xffffu);
        other = static_cast<int>(meta >> 16);
        coeff = sh_const_forward_coeff<RealT>(back_u32[packed_base + 1]);
      } else {
        const int term_base = term * 2;
        target = model.sh_cg_back_terms_int[term_base + 0];
        other = model.sh_cg_back_terms_int[term_base + 1];
        coeff = sh_cg_back_term_coeff<RealT>(model, term);
      }
      sum += coeff * static_cast<RealT>(grads[static_cast<size_t>(target) * N + atom]) *
             moments[static_cast<size_t>(other) * N + atom];
    }
    grads[static_cast<size_t>(source) * N + atom] += static_cast<GradT>(sum);
  }
}

template <typename GradT>
static __global__ void gpu_sh_zero_selected_grads(
  int N,
  int moment_count,
  const int* moments,
  GradT* grads)
{
  const size_t total = static_cast<size_t>(moment_count) * static_cast<size_t>(N);
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t task = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < total;
       task += stride) {
    const int atom = static_cast<int>(task % static_cast<size_t>(N));
    const int moment = moments[task / static_cast<size_t>(N)];
    grads[static_cast<size_t>(moment) * N + atom] = static_cast<GradT>(0.0);
  }
}

template <typename RealT, typename GradT, int BasicCache, int DotCount>
__device__ __forceinline__ void sh_process_terminal_dot_group_fixed(
  int N,
  int atom,
  const SHDeviceModel& model,
  const RealT* moments,
  GradT* grads,
  const RealT* basic_cache,
  RealT center_coeff,
  int left0,
  int entry_begin,
  int entry_count,
  RealT& site_energy)
{
  RealT left_values[DotCount];
  RealT left_grads[DotCount];
#pragma unroll
  for (int c = 0; c < DotCount; ++c) {
    const int left = left0 + c;
    left_values[c] =
      (BasicCache > 0 && model.use_product_basic_cache && left < model.alpha_basic_count)
      ? basic_cache[left]
      : moments[static_cast<size_t>(left) * N + atom];
    left_grads[c] = static_cast<RealT>(0.0);
  }
  const size_t entry_offset =
    static_cast<size_t>(model.sh_terminal_dot_group_count) * 3;
  for (int e = 0; e < entry_count; ++e) {
    const size_t entry_base =
      entry_offset + static_cast<size_t>(entry_begin + e) * 2;
    const unsigned int entry0 = model.sh_terminal_dot_group_u32[entry_base + 0];
    const int right0 = static_cast<int>(entry0 & 0xffffu);
    const RealT coeff =
      sh_const_forward_coeff<RealT>(model.sh_terminal_dot_group_u32[entry_base + 1]);
    RealT weighted = coeff * center_coeff;
    if (!model.use_terminal_dot_premul) {
      const int scalar_index = static_cast<int>(entry0 >> 16);
      weighted *= sh_moment_coeff<RealT>(model, scalar_index);
    }
    RealT dot = static_cast<RealT>(0.0);
#pragma unroll
    for (int c = 0; c < DotCount; ++c) {
      const int right = right0 + c;
      const RealT right_value =
        (BasicCache > 0 && model.use_product_basic_cache && right < model.alpha_basic_count)
        ? basic_cache[right]
        : moments[static_cast<size_t>(right) * N + atom];
      dot += left_values[c] * right_value;
      left_grads[c] += weighted * right_value;
      grads[static_cast<size_t>(right) * N + atom] +=
        static_cast<GradT>(weighted * left_values[c]);
    }
    site_energy += weighted * dot;
  }
#pragma unroll
  for (int c = 0; c < DotCount; ++c) {
    grads[static_cast<size_t>(left0 + c) * N + atom] +=
      static_cast<GradT>(left_grads[c]);
  }
}

__device__ __forceinline__ size_t sh_fused_entry_u32_offset(const SHDeviceModel& model)
{
  return static_cast<size_t>(model.sh_fused_terminal_dot_group_count) * 3;
}

__device__ __forceinline__ size_t sh_fused_producer_u32_offset(const SHDeviceModel& model)
{
  return sh_fused_entry_u32_offset(model) +
         static_cast<size_t>(model.sh_fused_terminal_dot_group_entry_count) * 3;
}

__device__ __forceinline__ size_t sh_fused_component_u32_offset(const SHDeviceModel& model)
{
  return sh_fused_producer_u32_offset(model) +
         static_cast<size_t>(model.sh_fused_terminal_dot_producer_count) * 3;
}

__device__ __forceinline__ size_t sh_fused_term_u32_offset(const SHDeviceModel& model)
{
  return sh_fused_component_u32_offset(model) +
         static_cast<size_t>(model.sh_fused_terminal_dot_component_count) * 2;
}

template <typename RealT, int BasicCache>
__device__ __forceinline__ RealT sh_load_product_moment(
  int N,
  int atom,
  const SHDeviceModel& model,
  const RealT* moments,
  const RealT* basic_cache,
  int moment)
{
  return (BasicCache > 0 && model.use_product_basic_cache &&
          moment < model.alpha_basic_count)
    ? basic_cache[moment]
    : moments[static_cast<size_t>(moment) * N + atom];
}

template <typename RealT, int BasicCache>
__device__ __forceinline__ void sh_compute_fused_producer_values(
  int N,
  int atom,
  const SHDeviceModel& model,
  const RealT* moments,
  const RealT* basic_cache,
  int producer_id,
  int dot_count,
  RealT* values)
{
  const unsigned int* packed = model.sh_fused_terminal_dot_u32;
  const size_t producer_base =
    sh_fused_producer_u32_offset(model) + static_cast<size_t>(producer_id) * 3;
  const int component_begin = static_cast<int>(packed[producer_base + 1]);
  const size_t component_offset = sh_fused_component_u32_offset(model);
  const size_t term_offset = sh_fused_term_u32_offset(model);
  for (int c = 0; c < dot_count; ++c) {
    const size_t component_base =
      component_offset + static_cast<size_t>(component_begin + c) * 2;
    const int term_begin = static_cast<int>(packed[component_base + 0]);
    const int term_count = static_cast<int>(packed[component_base + 1]);
    RealT sum = static_cast<RealT>(0.0);
    for (int t = 0; t < term_count; ++t) {
      const size_t term_base = term_offset + static_cast<size_t>(term_begin + t) * 2;
      const unsigned int term0 = packed[term_base + 0];
      const int left = static_cast<int>(term0 & 0xffffu);
      const int right = static_cast<int>(term0 >> 16);
      const RealT coeff = sh_const_forward_coeff<RealT>(packed[term_base + 1]);
      const RealT left_value =
        sh_load_product_moment<RealT, BasicCache>(N, atom, model, moments, basic_cache, left);
      const RealT right_value =
        sh_load_product_moment<RealT, BasicCache>(N, atom, model, moments, basic_cache, right);
      sum += coeff * left_value * right_value;
    }
    values[c] = sum;
  }
}

template <typename RealT, typename GradT, int BasicCache>
__device__ __forceinline__ void sh_backprop_fused_producer_values(
  int N,
  int atom,
  const SHDeviceModel& model,
  const RealT* moments,
  GradT* grads,
  const RealT* basic_cache,
  int producer_id,
  int dot_count,
  const RealT* value_grads)
{
  const unsigned int* packed = model.sh_fused_terminal_dot_u32;
  const size_t producer_base =
    sh_fused_producer_u32_offset(model) + static_cast<size_t>(producer_id) * 3;
  const int component_begin = static_cast<int>(packed[producer_base + 1]);
  const size_t component_offset = sh_fused_component_u32_offset(model);
  const size_t term_offset = sh_fused_term_u32_offset(model);
  for (int c = 0; c < dot_count; ++c) {
    const RealT g = value_grads[c];
    if (g == static_cast<RealT>(0.0)) {
      continue;
    }
    const size_t component_base =
      component_offset + static_cast<size_t>(component_begin + c) * 2;
    const int term_begin = static_cast<int>(packed[component_base + 0]);
    const int term_count = static_cast<int>(packed[component_base + 1]);
    for (int t = 0; t < term_count; ++t) {
      const size_t term_base = term_offset + static_cast<size_t>(term_begin + t) * 2;
      const unsigned int term0 = packed[term_base + 0];
      const int left = static_cast<int>(term0 & 0xffffu);
      const int right = static_cast<int>(term0 >> 16);
      const RealT coeff = sh_const_forward_coeff<RealT>(packed[term_base + 1]);
      const RealT left_value =
        sh_load_product_moment<RealT, BasicCache>(N, atom, model, moments, basic_cache, left);
      const RealT right_value =
        sh_load_product_moment<RealT, BasicCache>(N, atom, model, moments, basic_cache, right);
      const RealT weighted = coeff * g;
      grads[static_cast<size_t>(left) * N + atom] += static_cast<GradT>(weighted * right_value);
      grads[static_cast<size_t>(right) * N + atom] += static_cast<GradT>(weighted * left_value);
    }
  }
}

template <typename RealT, typename GradT, int BasicCache>
__device__ __forceinline__ void sh_backprop_fused_producer_scaled(
  int N,
  int atom,
  const SHDeviceModel& model,
  const RealT* moments,
  GradT* grads,
  const RealT* basic_cache,
  int producer_id,
  int dot_count,
  const RealT* scale_values,
  RealT scale)
{
  const unsigned int* packed = model.sh_fused_terminal_dot_u32;
  const size_t producer_base =
    sh_fused_producer_u32_offset(model) + static_cast<size_t>(producer_id) * 3;
  const int component_begin = static_cast<int>(packed[producer_base + 1]);
  const size_t component_offset = sh_fused_component_u32_offset(model);
  const size_t term_offset = sh_fused_term_u32_offset(model);
  for (int c = 0; c < dot_count; ++c) {
    const RealT g = scale * scale_values[c];
    if (g == static_cast<RealT>(0.0)) {
      continue;
    }
    const size_t component_base =
      component_offset + static_cast<size_t>(component_begin + c) * 2;
    const int term_begin = static_cast<int>(packed[component_base + 0]);
    const int term_count = static_cast<int>(packed[component_base + 1]);
    for (int t = 0; t < term_count; ++t) {
      const size_t term_base = term_offset + static_cast<size_t>(term_begin + t) * 2;
      const unsigned int term0 = packed[term_base + 0];
      const int left = static_cast<int>(term0 & 0xffffu);
      const int right = static_cast<int>(term0 >> 16);
      const RealT coeff = sh_const_forward_coeff<RealT>(packed[term_base + 1]);
      const RealT left_value =
        sh_load_product_moment<RealT, BasicCache>(N, atom, model, moments, basic_cache, left);
      const RealT right_value =
        sh_load_product_moment<RealT, BasicCache>(N, atom, model, moments, basic_cache, right);
      const RealT weighted = coeff * g;
      grads[static_cast<size_t>(left) * N + atom] += static_cast<GradT>(weighted * right_value);
      grads[static_cast<size_t>(right) * N + atom] += static_cast<GradT>(weighted * left_value);
    }
  }
}

template <typename RealT, typename GradT, int BasicCache>
static __global__ void gpu_sh_fused_terminal_dot_groups_energy(
  int N,
  SHDeviceModel model,
  const int* type,
  const RealT* moments,
  GradT* grads,
  double* potential)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= N || model.sh_fused_terminal_dot_group_count <= 0 ||
      model.sh_fused_terminal_dot_u32 == nullptr) {
    return;
  }

  const int type_i = type[atom];
  const RealT center_coeff = sh_species_coeff<RealT>(model, type_i);
  RealT site_energy = static_cast<RealT>(0.0);
  RealT basic_cache[BasicCache > 0 ? BasicCache : 1];
  if (BasicCache > 0 && model.use_product_basic_cache) {
    for (int b = 0; b < model.alpha_basic_count; ++b) {
      basic_cache[b] = moments[static_cast<size_t>(b) * N + atom];
    }
  }

  const unsigned int* packed = model.sh_fused_terminal_dot_u32;
  const size_t entry_offset = sh_fused_entry_u32_offset(model);
  for (int group = 0; group < model.sh_fused_terminal_dot_group_count; ++group) {
    const size_t group_base = static_cast<size_t>(group) * 3;
    const unsigned int group0 = packed[group_base + 0];
    const int producer_id = static_cast<int>(group0 & 0xffffu);
    const int dot_count = static_cast<int>(group0 >> 16);
    if (dot_count <= 0 || dot_count > kSHTerminalDotGroupMaxCount) {
      continue;
    }
    RealT producer_values[kSHTerminalDotGroupMaxCount];
    RealT producer_grads[kSHTerminalDotGroupMaxCount];
    sh_compute_fused_producer_values<RealT, BasicCache>(
      N, atom, model, moments, basic_cache, producer_id, dot_count, producer_values);
    for (int c = 0; c < dot_count; ++c) {
      producer_grads[c] = static_cast<RealT>(0.0);
    }

    const int entry_begin = static_cast<int>(packed[group_base + 1]);
    const int entry_count = static_cast<int>(packed[group_base + 2]);
    for (int e = 0; e < entry_count; ++e) {
      const size_t entry_base = entry_offset + static_cast<size_t>(entry_begin + e) * 3;
      const unsigned int entry0 = packed[entry_base + 0];
      const int other0 = static_cast<int>(entry0 & 0xffffu);
      const int scalar_index = static_cast<int>(entry0 >> 16);
      RealT weighted =
        sh_const_forward_coeff<RealT>(packed[entry_base + 1]) * center_coeff;
      if (!model.use_terminal_dot_premul) {
        weighted *= sh_moment_coeff<RealT>(model, scalar_index);
      }
      const unsigned int other_producer_u32 = packed[entry_base + 2];
      const bool other_is_producer = other_producer_u32 != 0xffffffffu;
      RealT other_values[kSHTerminalDotGroupMaxCount];
      if (other_is_producer) {
        sh_compute_fused_producer_values<RealT, BasicCache>(
          N, atom, model, moments, basic_cache,
          static_cast<int>(other_producer_u32), dot_count, other_values);
      }
      RealT dot = static_cast<RealT>(0.0);
      for (int c = 0; c < dot_count; ++c) {
        const RealT other_value = other_is_producer
          ? other_values[c]
          : sh_load_product_moment<RealT, BasicCache>(
              N, atom, model, moments, basic_cache, other0 + c);
        dot += producer_values[c] * other_value;
        producer_grads[c] += weighted * other_value;
        if (other_is_producer) {
          continue;
        } else {
          grads[static_cast<size_t>(other0 + c) * N + atom] +=
            static_cast<GradT>(weighted * producer_values[c]);
        }
      }
      site_energy += weighted * dot;
      if (other_is_producer) {
        sh_backprop_fused_producer_scaled<RealT, GradT, BasicCache>(
          N, atom, model, moments, grads, basic_cache,
          static_cast<int>(other_producer_u32), dot_count, producer_values, weighted);
      }
    }
    sh_backprop_fused_producer_values<RealT, GradT, BasicCache>(
      N, atom, model, moments, grads, basic_cache, producer_id, dot_count, producer_grads);
  }
  potential[atom] += static_cast<double>(site_energy);
}

template <typename RealT, typename GradT, int BasicCache>
void launch_sh_fused_terminal_dot_groups_energy(
  int grid_size,
  int block_size,
  int N,
  SHDeviceModel model,
  const int* type,
  const RealT* moments,
  GradT* grads,
  double* potential)
{
  gpu_sh_fused_terminal_dot_groups_energy<RealT, GradT, BasicCache>
    <<<grid_size, block_size>>>(
      N, model, type, moments, grads, potential);
}

template <
  typename RealT,
  typename GradT,
  int BasicCache,
  int PatternRows,
  int DotGroups,
  int DoBackward>
static __global__ void gpu_sh_forward_energy_backward_compact_rows(
  int N,
  SHDeviceModel model,
  const int* type,
  RealT* moments,
  GradT* grads,
  double* potential)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= N) {
    return;
  }

  const int type_i = type[atom];
  const RealT center_coeff = sh_species_coeff<RealT>(model, type_i);
  RealT site_energy = sh_shift_coeff<RealT>(model, type_i) + center_coeff;
  RealT basic_cache[BasicCache > 0 ? BasicCache : 1];
  if (BasicCache > 0 && model.use_product_basic_cache) {
    for (int b = 0; b < model.alpha_basic_count; ++b) {
      basic_cache[b] = moments[static_cast<size_t>(b) * N + atom];
    }
  }

  for (int layer = 1; layer <= model.sh_cg_layer_count; ++layer) {
    const bool use_nondot_rows =
      DotGroups > 0 && model.use_terminal_dot_groups &&
      model.sh_terminal_dot_nondot_rows != nullptr &&
      model.sh_terminal_dot_nondot_layer_offsets != nullptr;
    const int row_begin = use_nondot_rows
      ? model.sh_terminal_dot_nondot_layer_offsets[layer]
      : model.sh_cg_layer_offsets[layer];
    const int row_end = use_nondot_rows
      ? model.sh_terminal_dot_nondot_layer_offsets[layer + 1]
      : model.sh_cg_layer_offsets[layer + 1];
    for (int row_pos = row_begin; row_pos < row_end; ++row_pos) {
      const int row = use_nondot_rows ? model.sh_terminal_dot_nondot_rows[row_pos] : row_pos;
      int left_base;
      int right_base;
      int target;
      int term_begin;
      int term_count;
      int pattern_id = -1;
      if (PatternRows > 0) {
        const unsigned int* pattern_u32 =
          model.use_const_pattern_rows ? c_sh_forward_u32 : model.sh_cg_row_pattern_u32;
        const size_t pattern_row = static_cast<size_t>(row) * 2;
        const unsigned int row0 = pattern_u32[pattern_row + 0];
        const unsigned int row1 = pattern_u32[pattern_row + 1];
        left_base = static_cast<int>(row0 & 0xffffu);
        right_base = static_cast<int>(row0 >> 16);
        target = static_cast<int>(row1 & 0xffffu);
        pattern_id = static_cast<int>(row1 >> 16);
        const size_t header_base =
          static_cast<size_t>(model.sh_cg_row_count) * 2 +
          static_cast<size_t>(pattern_id) * 2;
        term_begin = static_cast<int>(pattern_u32[header_base + 0]);
        term_count = static_cast<int>(pattern_u32[header_base + 1]);
      } else if (model.use_const_forward_rows) {
        const int const_row = row * 3;
        const unsigned int row0 = c_sh_forward_u32[const_row + 0];
        const unsigned int row1 = c_sh_forward_u32[const_row + 1];
        left_base = static_cast<int>(row0 & 0xffffu);
        right_base = static_cast<int>(row0 >> 16);
        target = static_cast<int>(row1 & 0xffffu);
        term_count = static_cast<int>(row1 >> 16);
        term_begin = static_cast<int>(c_sh_forward_u32[const_row + 2]);
      } else {
        const int row_base = row * 5;
        left_base = model.sh_cg_rows_int[row_base + 0];
        right_base = model.sh_cg_rows_int[row_base + 1];
        target = model.sh_cg_rows_int[row_base + 2];
        term_begin = model.sh_cg_rows_int[row_base + 3];
        term_count = model.sh_cg_rows_int[row_base + 4];
      }
      RealT sum = static_cast<RealT>(0.0);
      const int scalar_index = model.use_terminal_scalar_fusion
        ? model.sh_cg_row_scalar_index[row]
        : -1;
      const bool scalar_row = scalar_index >= 0;
      const bool terminal_scalar_row =
        scalar_row && model.sh_terminal_moment_flags[target] != 0;
      const RealT scalar_gtarget = scalar_row
        ? center_coeff * sh_moment_coeff<RealT>(model, scalar_index)
        : static_cast<RealT>(0.0);
      if (terminal_scalar_row && model.use_terminal_dot_rows) {
        const size_t dot_base = static_cast<size_t>(row) * 3;
        const unsigned int dot_count = model.sh_cg_row_dot_u32[dot_base + 1];
        if (dot_count != 0u) {
          if (DotGroups > 0) {
            continue;
          }
          const unsigned int dot0 = model.sh_cg_row_dot_u32[dot_base + 0];
          const int left0 = static_cast<int>(dot0 & 0xffffu);
          const int right0 = static_cast<int>(dot0 >> 16);
          const RealT coeff =
            sh_const_forward_coeff<RealT>(model.sh_cg_row_dot_u32[dot_base + 2]);
          const RealT weighted = coeff * scalar_gtarget;
          RealT dot = static_cast<RealT>(0.0);
          for (unsigned int c = 0; c < dot_count; ++c) {
            const int left = left0 + static_cast<int>(c);
            const int right = right0 + static_cast<int>(c);
            const RealT left_value =
              (BasicCache > 0 && model.use_product_basic_cache &&
               left < model.alpha_basic_count)
              ? basic_cache[left]
              : moments[static_cast<size_t>(left) * N + atom];
            const RealT right_value =
              (BasicCache > 0 && model.use_product_basic_cache &&
               right < model.alpha_basic_count)
              ? basic_cache[right]
              : moments[static_cast<size_t>(right) * N + atom];
            dot += left_value * right_value;
            grads[static_cast<size_t>(left) * N + atom] +=
              static_cast<GradT>(weighted * right_value);
            grads[static_cast<size_t>(right) * N + atom] +=
              static_cast<GradT>(weighted * left_value);
          }
          site_energy += weighted * dot;
          continue;
        }
      }
      for (int t = 0; t < term_count; ++t) {
        const int term = term_begin + t;
        int left_component;
        int right_component;
        RealT coeff;
        if (PatternRows > 0) {
          const unsigned int* pattern_u32 =
            model.use_const_pattern_rows ? c_sh_forward_u32 : model.sh_cg_row_pattern_u32;
          const size_t pattern_term_base =
            static_cast<size_t>(model.sh_cg_row_count) * 2 +
            static_cast<size_t>(model.sh_cg_row_pattern_count) * 2 +
            static_cast<size_t>(term_begin + t) * 2;
          const unsigned int term_meta = pattern_u32[pattern_term_base + 0];
          left_component = static_cast<int>(term_meta & 0xffffu);
          right_component = static_cast<int>(term_meta >> 16);
          coeff = sh_const_forward_coeff<RealT>(pattern_u32[pattern_term_base + 1]);
        } else if (model.use_const_forward_rows) {
          const int const_term = model.sh_cg_row_count * 3 + term * 2;
          const unsigned int term_meta = c_sh_forward_u32[const_term + 0];
          left_component = static_cast<int>(term_meta & 0xffffu);
          right_component = static_cast<int>(term_meta >> 16);
          coeff = sh_const_forward_coeff<RealT>(c_sh_forward_u32[const_term + 1]);
        } else {
          const int term_base = term * 2;
          left_component = model.sh_cg_row_terms_int[term_base + 0];
          right_component = model.sh_cg_row_terms_int[term_base + 1];
          coeff = sh_cg_row_term_coeff<RealT>(model, term);
        }
        const int left = left_base + left_component;
        const int right = right_base + right_component;
        const RealT left_value =
          (BasicCache > 0 && model.use_product_basic_cache &&
           left < model.alpha_basic_count)
          ? basic_cache[left]
          : moments[static_cast<size_t>(left) * N + atom];
        const RealT right_value =
          (BasicCache > 0 && model.use_product_basic_cache &&
           right < model.alpha_basic_count)
          ? basic_cache[right]
          : moments[static_cast<size_t>(right) * N + atom];
        sum += coeff * left_value * right_value;
        if (terminal_scalar_row) {
          const RealT weighted = coeff * scalar_gtarget;
          grads[static_cast<size_t>(left) * N + atom] +=
            static_cast<GradT>(weighted * right_value);
          grads[static_cast<size_t>(right) * N + atom] +=
            static_cast<GradT>(weighted * left_value);
        }
      }
      if (scalar_row) {
        site_energy += scalar_gtarget * sum;
      }
      if (terminal_scalar_row) {
        continue;
      }
      moments[static_cast<size_t>(target) * N + atom] = sum;
      if (scalar_row) {
        grads[static_cast<size_t>(target) * N + atom] += static_cast<GradT>(scalar_gtarget);
      }
    }
    if (DotGroups > 0) {
      const int group_begin = model.sh_terminal_dot_group_layer_offsets[layer];
      const int group_end = model.sh_terminal_dot_group_layer_offsets[layer + 1];
      for (int group = group_begin; group < group_end; ++group) {
        const size_t group_base = static_cast<size_t>(group) * 3;
        const unsigned int group0 = model.sh_terminal_dot_group_u32[group_base + 0];
        const int left0 = static_cast<int>(group0 & 0xffffu);
        const int dot_count = static_cast<int>(group0 >> 16);
        if (dot_count <= 0 || dot_count > kSHTerminalDotGroupMaxCount) {
          continue;
        }
        const int entry_begin =
          static_cast<int>(model.sh_terminal_dot_group_u32[group_base + 1]);
        const int entry_count =
          static_cast<int>(model.sh_terminal_dot_group_u32[group_base + 2]);
        switch (dot_count) {
        case 1:
          sh_process_terminal_dot_group_fixed<RealT, GradT, BasicCache, 1>(
            N, atom, model, moments, grads, basic_cache, center_coeff, left0,
            entry_begin, entry_count, site_energy);
          break;
        case 3:
          sh_process_terminal_dot_group_fixed<RealT, GradT, BasicCache, 3>(
            N, atom, model, moments, grads, basic_cache, center_coeff, left0,
            entry_begin, entry_count, site_energy);
          break;
        case 5:
          sh_process_terminal_dot_group_fixed<RealT, GradT, BasicCache, 5>(
            N, atom, model, moments, grads, basic_cache, center_coeff, left0,
            entry_begin, entry_count, site_energy);
          break;
        case 7:
          sh_process_terminal_dot_group_fixed<RealT, GradT, BasicCache, 7>(
            N, atom, model, moments, grads, basic_cache, center_coeff, left0,
            entry_begin, entry_count, site_energy);
          break;
        case 9:
          sh_process_terminal_dot_group_fixed<RealT, GradT, BasicCache, 9>(
            N, atom, model, moments, grads, basic_cache, center_coeff, left0,
            entry_begin, entry_count, site_energy);
          break;
        case 11:
          sh_process_terminal_dot_group_fixed<RealT, GradT, BasicCache, 11>(
            N, atom, model, moments, grads, basic_cache, center_coeff, left0,
            entry_begin, entry_count, site_energy);
          break;
        case 13:
          sh_process_terminal_dot_group_fixed<RealT, GradT, BasicCache, 13>(
            N, atom, model, moments, grads, basic_cache, center_coeff, left0,
            entry_begin, entry_count, site_energy);
          break;
        case 15:
          sh_process_terminal_dot_group_fixed<RealT, GradT, BasicCache, 15>(
            N, atom, model, moments, grads, basic_cache, center_coeff, left0,
            entry_begin, entry_count, site_energy);
          break;
        case 17:
          sh_process_terminal_dot_group_fixed<RealT, GradT, BasicCache, 17>(
            N, atom, model, moments, grads, basic_cache, center_coeff, left0,
            entry_begin, entry_count, site_energy);
          break;
        default: {
          RealT left_values[kSHTerminalDotGroupMaxCount];
          RealT left_grads[kSHTerminalDotGroupMaxCount];
          for (int c = 0; c < dot_count; ++c) {
            const int left = left0 + c;
            left_values[c] =
              (BasicCache > 0 && model.use_product_basic_cache &&
               left < model.alpha_basic_count)
              ? basic_cache[left]
              : moments[static_cast<size_t>(left) * N + atom];
            left_grads[c] = static_cast<RealT>(0.0);
          }
          const size_t entry_offset =
            static_cast<size_t>(model.sh_terminal_dot_group_count) * 3;
          for (int e = 0; e < entry_count; ++e) {
            const size_t entry_base =
              entry_offset + static_cast<size_t>(entry_begin + e) * 2;
            const unsigned int entry0 = model.sh_terminal_dot_group_u32[entry_base + 0];
            const int right0 = static_cast<int>(entry0 & 0xffffu);
            const RealT coeff =
              sh_const_forward_coeff<RealT>(model.sh_terminal_dot_group_u32[entry_base + 1]);
            RealT weighted = coeff * center_coeff;
            if (!model.use_terminal_dot_premul) {
              const int scalar_index = static_cast<int>(entry0 >> 16);
              weighted *= sh_moment_coeff<RealT>(model, scalar_index);
            }
            RealT dot = static_cast<RealT>(0.0);
            for (int c = 0; c < dot_count; ++c) {
              const int right = right0 + c;
              const RealT right_value =
                (BasicCache > 0 && model.use_product_basic_cache &&
                 right < model.alpha_basic_count)
                ? basic_cache[right]
                : moments[static_cast<size_t>(right) * N + atom];
              dot += left_values[c] * right_value;
              left_grads[c] += weighted * right_value;
              grads[static_cast<size_t>(right) * N + atom] +=
                static_cast<GradT>(weighted * left_values[c]);
            }
            site_energy += weighted * dot;
          }
          for (int c = 0; c < dot_count; ++c) {
            grads[static_cast<size_t>(left0 + c) * N + atom] +=
              static_cast<GradT>(left_grads[c]);
          }
          break;
        }
        }
      }
    }
  }

  for (int s = 0; s < model.active_scalar_moments; ++s) {
    const int moment_id = model.active_scalar_moment[s];
    const RealT coeff = sh_active_scalar_coeff<RealT>(model, s);
    const RealT moment_value =
      (BasicCache > 0 && model.use_product_basic_cache &&
       moment_id < model.alpha_basic_count)
      ? basic_cache[moment_id]
      : moments[static_cast<size_t>(moment_id) * N + atom];
    site_energy += center_coeff * coeff * moment_value;
    grads[static_cast<size_t>(moment_id) * N + atom] +=
      static_cast<GradT>(center_coeff * coeff);
  }
  potential[atom] += static_cast<double>(site_energy);

  if (DoBackward != 0) {
    for (int layer = model.sh_cg_layer_count; layer >= 1; --layer) {
      const int row_begin = model.sh_cg_back_layer_offsets[layer];
      const int row_end = model.sh_cg_back_layer_offsets[layer + 1];
      for (int row = row_begin; row < row_end; ++row) {
        int source;
        int term_begin;
        int term_count;
        if (model.use_packed_back_rows) {
          const unsigned int* back_u32 =
            model.use_const_back_rows
            ? c_sh_forward_u32 + sh_const_back_u32_offset(model)
            : model.sh_cg_back_packed_u32;
          const unsigned int row0 = back_u32[static_cast<size_t>(row) * 2 + 0];
          source = static_cast<int>(row0 & 0xffffu);
          term_count = static_cast<int>(row0 >> 16);
          term_begin = static_cast<int>(back_u32[static_cast<size_t>(row) * 2 + 1]);
        } else {
          const int row_base = row * 3;
          source = model.sh_cg_back_rows_int[row_base + 0];
          term_begin = model.sh_cg_back_rows_int[row_base + 1];
          term_count = model.sh_cg_back_rows_int[row_base + 2];
        }
        RealT sum = static_cast<RealT>(0.0);
        for (int t = 0; t < term_count; ++t) {
          const int term = term_begin + t;
          int target;
          int other;
          RealT coeff;
          if (model.use_packed_back_rows) {
            const unsigned int* back_u32 =
              model.use_const_back_rows
              ? c_sh_forward_u32 + sh_const_back_u32_offset(model)
              : model.sh_cg_back_packed_u32;
            const size_t packed_base =
              static_cast<size_t>(model.sh_cg_back_row_count) * 2 +
              static_cast<size_t>(term) * 2;
            const unsigned int meta = back_u32[packed_base + 0];
            target = static_cast<int>(meta & 0xffffu);
            other = static_cast<int>(meta >> 16);
            coeff = sh_const_forward_coeff<RealT>(back_u32[packed_base + 1]);
          } else {
            const int term_base = term * 2;
            target = model.sh_cg_back_terms_int[term_base + 0];
            other = model.sh_cg_back_terms_int[term_base + 1];
            coeff = sh_cg_back_term_coeff<RealT>(model, term);
          }
          sum += coeff * static_cast<RealT>(grads[static_cast<size_t>(target) * N + atom]) *
                 ((BasicCache > 0 && model.use_product_basic_cache &&
                   other < model.alpha_basic_count)
                  ? basic_cache[other]
                  : moments[static_cast<size_t>(other) * N + atom]);
        }
        grads[static_cast<size_t>(source) * N + atom] += static_cast<GradT>(sum);
      }
    }
  }
}

template <
  typename RealT,
  typename GradT,
  int BasicCache,
  int PatternRows,
  int DotGroups>
void launch_sh_forward_energy_backward_compact_rows(
  int grid_size,
  int block_size,
  int N,
  SHDeviceModel model,
  const int* type,
  RealT* moments,
  GradT* grads,
  double* potential,
  bool do_backward)
{
  if (do_backward) {
    gpu_sh_forward_energy_backward_compact_rows<
      RealT, GradT, BasicCache, PatternRows, DotGroups, 1>
      <<<grid_size, block_size>>>(
        N, model, type, moments, grads, potential);
  } else {
    gpu_sh_forward_energy_backward_compact_rows<
      RealT, GradT, BasicCache, PatternRows, DotGroups, 0>
      <<<grid_size, block_size>>>(
        N, model, type, moments, grads, potential);
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
__device__ __forceinline__ void compute_sh_edge_derivative_gate_main_cached_grads(
  SHDeviceModel model,
  int center_type,
  int neighbor_type,
  RealT gate_residual,
  RealT dx,
  RealT dy,
  RealT dz,
  RealT r,
  const GradT* grad_cache,
  RealT& dEx,
  RealT& dEy,
  RealT& dEz,
  RealT& gate_adjoint)
{
  const int pair = center_type * model.species_count + neighbor_type;
  RealT radial_vals[kMaxSHRadialFuncs];
  RealT radial_ders[kMaxSHRadialFuncs];
  RealT gate_multipliers[kMaxSHRadialFuncs];
  RealT gate_derivs[kMaxSHRadialFuncs];
  RealT sh_vals[kMaxSHComponents];
  RealT sh_ders[3 * kMaxSHComponents];
  sh_direct_radial_vals_ders(model, pair, r, radial_vals, radial_ders);
  const RealT amplitude = static_cast<RealT>(model.two_layer_gate_tanh_amplitude);
  for (int mu = 0; mu < model.radial_funcs_count; ++mu) {
    const RealT a = static_cast<RealT>(
      model.two_layer_gate_additive_coeffs_float[neighbor_type * model.radial_funcs_count + mu]);
    const RealT arg = a * gate_residual;
    const RealT tanh_arg = tanh(arg);
    gate_multipliers[mu] = static_cast<RealT>(1.0) + amplitude * tanh_arg;
    gate_derivs[mu] = amplitude * a * (static_cast<RealT>(1.0) - tanh_arg * tanh_arg);
  }
  eval_real_sh(dx, dy, dz, r, model.sh_l_max, sh_vals, sh_ders);
  const RealT inv_r = static_cast<RealT>(1.0) / r;
  dEx = dEy = dEz = gate_adjoint = static_cast<RealT>(0.0);
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
    const RealT mult = gate_multipliers[mu];
    dEx += adj * mult *
           (radial_der * dx * inv_r * ylm + radial * sh_ders[3 * yidx + 0]);
    dEy += adj * mult *
           (radial_der * dy * inv_r * ylm + radial * sh_ders[3 * yidx + 1]);
    dEz += adj * mult *
           (radial_der * dz * inv_r * ylm + radial * sh_ders[3 * yidx + 2]);
    gate_adjoint += adj * radial * ylm * gate_derivs[mu];
  }
}

template <typename GradT, typename RealT>
__device__ __forceinline__ void compute_sh_edge_derivative_with_radial_cached_grads(
  SHDeviceModel model,
  const float* radial_coeffs,
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
  sh_direct_radial_vals_ders(model, radial_coeffs, pair, r, radial_vals, radial_ders);
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

template <typename GradT, typename RealT, int L, int K, int RbSize>
__device__ __forceinline__ void compute_sh_edge_derivative_with_radial_static_layout(
  SHDeviceModel model,
  const float* radial_coeffs,
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
  constexpr int RadialFuncs = K * (L + 1);
  RealT radial_vals[RadialFuncs];
  RealT radial_ders[RadialFuncs];
  RealT sh_vals[kMaxSHComponents];
  RealT sh_ders[3 * kMaxSHComponents];
  const int pair = center_type * model.species_count + neighbor_type;
  sh_direct_radial_vals_ders_static<RealT, RadialFuncs, RbSize>(
    model, radial_coeffs, pair, r, radial_vals, radial_ders);
  eval_real_sh(dx, dy, dz, r, L, sh_vals, sh_ders);

  const RealT inv_r = static_cast<RealT>(1.0) / r;
  dEx = dEy = dEz = static_cast<RealT>(0.0);
  int basic = 0;
#pragma unroll
  for (int l_desc = 0; l_desc <= L; ++l_desc) {
    const int l = L - l_desc;
#pragma unroll
    for (int k_desc = 0; k_desc < K; ++k_desc) {
      const int k = K - 1 - k_desc;
      const int mu = k * (L + 1) + l;
      const RealT radial = radial_vals[mu];
      const RealT radial_der = radial_ders[mu];
      RealT projected_y = static_cast<RealT>(0.0);
      RealT projected_dx = static_cast<RealT>(0.0);
      RealT projected_dy = static_cast<RealT>(0.0);
      RealT projected_dz = static_cast<RealT>(0.0);
#pragma unroll
      for (int c = 0; c < 2 * l + 1; ++c) {
        const RealT adj = static_cast<RealT>(grad_cache[basic++]);
        const int yidx = l * l + c;
        projected_y += adj * sh_vals[yidx];
        projected_dx += adj * sh_ders[3 * yidx + 0];
        projected_dy += adj * sh_ders[3 * yidx + 1];
        projected_dz += adj * sh_ders[3 * yidx + 2];
      }
      const RealT radial_projected = radial_der * inv_r * projected_y;
      dEx += radial_projected * dx + radial * projected_dx;
      dEy += radial_projected * dy + radial * projected_dy;
      dEz += radial_projected * dz + radial * projected_dz;
    }
  }
}

template <typename GradT, typename RealT, int L, int K, int RbSize>
__device__ __forceinline__ void compute_sh_edge_derivative_gate_main_static_layout(
  SHDeviceModel model,
  int center_type,
  int neighbor_type,
  RealT gate_residual,
  RealT dx,
  RealT dy,
  RealT dz,
  RealT r,
  const GradT* grad_cache,
  RealT& dEx,
  RealT& dEy,
  RealT& dEz,
  RealT& gate_adjoint)
{
  constexpr int RadialFuncs = K * (L + 1);
  RealT radial_vals[RadialFuncs];
  RealT radial_ders[RadialFuncs];
  RealT gate_multipliers[RadialFuncs];
  RealT gate_derivs[RadialFuncs];
  RealT sh_vals[kMaxSHComponents];
  RealT sh_ders[3 * kMaxSHComponents];
  const int pair = center_type * model.species_count + neighbor_type;
  sh_direct_radial_vals_ders_static<RealT, RadialFuncs, RbSize>(
    model, model.radial_direct_coeffs, pair, r, radial_vals, radial_ders);
  const RealT amplitude = static_cast<RealT>(model.two_layer_gate_tanh_amplitude);
#pragma unroll
  for (int mu = 0; mu < RadialFuncs; ++mu) {
    const RealT a = static_cast<RealT>(
      model.two_layer_gate_additive_coeffs_float[neighbor_type * RadialFuncs + mu]);
    const RealT arg = a * gate_residual;
    const RealT tanh_arg = tanh(arg);
    gate_multipliers[mu] = static_cast<RealT>(1.0) + amplitude * tanh_arg;
    gate_derivs[mu] = amplitude * a * (static_cast<RealT>(1.0) - tanh_arg * tanh_arg);
  }
  eval_real_sh(dx, dy, dz, r, L, sh_vals, sh_ders);

  const RealT inv_r = static_cast<RealT>(1.0) / r;
  dEx = dEy = dEz = gate_adjoint = static_cast<RealT>(0.0);
  int basic = 0;
#pragma unroll
  for (int l_desc = 0; l_desc <= L; ++l_desc) {
    const int l = L - l_desc;
#pragma unroll
    for (int k_desc = 0; k_desc < K; ++k_desc) {
      const int k = K - 1 - k_desc;
      const int mu = k * (L + 1) + l;
      const RealT radial = radial_vals[mu];
      const RealT radial_der = radial_ders[mu];
      RealT projected_y = static_cast<RealT>(0.0);
      RealT projected_dx = static_cast<RealT>(0.0);
      RealT projected_dy = static_cast<RealT>(0.0);
      RealT projected_dz = static_cast<RealT>(0.0);
#pragma unroll
      for (int c = 0; c < 2 * l + 1; ++c) {
        const RealT adj = static_cast<RealT>(grad_cache[basic++]);
        const int yidx = l * l + c;
        projected_y += adj * sh_vals[yidx];
        projected_dx += adj * sh_ders[3 * yidx + 0];
        projected_dy += adj * sh_ders[3 * yidx + 1];
        projected_dz += adj * sh_ders[3 * yidx + 2];
      }
      const RealT mult = gate_multipliers[mu];
      const RealT radial_projected = mult * radial_der * inv_r * projected_y;
      dEx += radial_projected * dx + mult * radial * projected_dx;
      dEy += radial_projected * dy + mult * radial * projected_dy;
      dEz += radial_projected * dz + mult * radial * projected_dz;
      gate_adjoint += radial * gate_derivs[mu] * projected_y;
    }
  }
}

template <typename GradT, typename RealT, int L, int K, int RbSize>
__device__ __forceinline__ void compute_sh_edge_derivative_static_layout(
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
  constexpr int RadialFuncs = K * (L + 1);
  RealT radial_vals[RadialFuncs];
  RealT radial_ders[RadialFuncs];
  RealT sh_vals[kMaxSHComponents];
  RealT sh_ders[3 * kMaxSHComponents];
  const int pair = center_type * model.species_count + neighbor_type;
  sh_direct_radial_vals_ders_static<RealT, RadialFuncs, RbSize>(
    model, pair, r, radial_vals, radial_ders);
  eval_real_sh(dx, dy, dz, r, L, sh_vals, sh_ders);

  const RealT inv_r = static_cast<RealT>(1.0) / r;
  dEx = dEy = dEz = static_cast<RealT>(0.0);
  int basic = 0;
#pragma unroll
  for (int l_desc = 0; l_desc <= L; ++l_desc) {
    const int l = L - l_desc;
#pragma unroll
    for (int k_desc = 0; k_desc < K; ++k_desc) {
      const int k = K - 1 - k_desc;
      const int mu = k * (L + 1) + l;
      const RealT radial = radial_vals[mu];
      const RealT radial_der = radial_ders[mu];
      RealT projected_y = static_cast<RealT>(0.0);
      RealT projected_dx = static_cast<RealT>(0.0);
      RealT projected_dy = static_cast<RealT>(0.0);
      RealT projected_dz = static_cast<RealT>(0.0);
#pragma unroll
      for (int c = 0; c < 2 * l + 1; ++c) {
        const RealT adj = static_cast<RealT>(grad_cache[basic++]);
        const int yidx = l * l + c;
        projected_y += adj * sh_vals[yidx];
        projected_dx += adj * sh_ders[3 * yidx + 0];
        projected_dy += adj * sh_ders[3 * yidx + 1];
        projected_dz += adj * sh_ders[3 * yidx + 2];
      }
      const RealT radial_projected = radial_der * inv_r * projected_y;
      dEx += radial_projected * dx + radial * projected_dx;
      dEy += radial_projected * dy + radial * projected_dy;
      dEz += radial_projected * dz + radial * projected_dz;
    }
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

template <typename GradT, typename RealT, int CacheBasics>
static __global__ void gpu_sh_compute_forces_gate_main_cached_grads(
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
  const RealT* gate_values,
  const GradT* grads,
  RealT* gate_adjoints,
  float* force_tmp,
  float* force_self_tmp,
  float* virial_tmp)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || model.alpha_basic_count > CacheBasics) {
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
    RealT gate_adjoint;
	    compute_sh_edge_derivative_gate_main_cached_grads<GradT, RealT>(
	      model, type_i, type_j, gate_values[j], dx, dy, dz, r, grad_cache,
	      dEx, dEy, dEz, gate_adjoint);

    fx_self += dEx;
    fy_self += dEy;
    fz_self += dEz;
    atomicAdd(force_tmp + j, static_cast<float>(-dEx));
    atomicAdd(force_tmp + j + N, static_cast<float>(-dEy));
    atomicAdd(force_tmp + j + 2 * N, static_cast<float>(-dEz));
    atomicAdd(gate_adjoints + j, static_cast<RealT>(gate_adjoint));

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

template <typename GradT, typename RealT, int CacheBasics>
static __global__ void gpu_sh_compute_forces_gate_first_layer_cached_grads(
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
  const GradT* gate_basic_grads,
  const RealT* gate_adjoints,
  float* force_tmp,
  float* force_self_tmp,
  float* virial_tmp)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || model.alpha_basic_count > CacheBasics) {
    return;
  }

  const RealT center_adjoint = gate_adjoints[i];
  if (center_adjoint == static_cast<RealT>(0.0)) {
    return;
  }
  GradT grad_cache[CacheBasics];
  for (int b = 0; b < model.alpha_basic_count; ++b) {
    grad_cache[b] = static_cast<GradT>(
      center_adjoint * static_cast<RealT>(gate_basic_grads[static_cast<size_t>(b) * N + i]));
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
	    compute_sh_edge_derivative_with_radial_cached_grads<GradT, RealT>(
	      model, model.two_layer_gate_radial_direct_coeffs, type_i, type_j,
	      dx, dy, dz, r, grad_cache, dEx, dEy, dEz);

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
    force_self_tmp[i] += static_cast<float>(fx_self);
    force_self_tmp[i + N] += static_cast<float>(fy_self);
    force_self_tmp[i + 2 * N] += static_cast<float>(fz_self);
  } else {
    atomicAdd(force_tmp + i, static_cast<float>(fx_self));
    atomicAdd(force_tmp + i + N, static_cast<float>(fy_self));
    atomicAdd(force_tmp + i + 2 * N, static_cast<float>(fz_self));
  }

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

template <typename GradT, typename RealT, int L, int K, int RbSize>
static __global__ void gpu_sh_compute_forces_static_layout(
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
  constexpr int BasicCount = K * (L + 1) * (L + 1);
  constexpr int RadialFuncs = K * (L + 1);
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || model.alpha_basic_count != BasicCount ||
      model.radial_funcs_count != RadialFuncs || model.rb_size != RbSize) {
    return;
  }

  GradT grad_cache[BasicCount];
#pragma unroll
  for (int b = 0; b < BasicCount; ++b) {
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
    compute_sh_edge_derivative_static_layout<GradT, RealT, L, K, RbSize>(
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

template <typename GradT, typename RealT, int L, int K, int RbSize>
static __global__ void gpu_sh_compute_forces_gate_main_static_layout(
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
  const RealT* gate_values,
  const GradT* grads,
  RealT* gate_adjoints,
  float* force_tmp,
  float* force_self_tmp,
  float* virial_tmp)
{
  constexpr int BasicCount = K * (L + 1) * (L + 1);
  constexpr int RadialFuncs = K * (L + 1);
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || model.alpha_basic_count != BasicCount ||
      model.radial_funcs_count != RadialFuncs || model.rb_size != RbSize) {
    return;
  }

  GradT grad_cache[BasicCount];
#pragma unroll
  for (int b = 0; b < BasicCount; ++b) {
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
    RealT gate_adjoint;
    compute_sh_edge_derivative_gate_main_static_layout<GradT, RealT, L, K, RbSize>(
      model, type_i, type_j, gate_values[j], dx, dy, dz, r, grad_cache,
      dEx, dEy, dEz, gate_adjoint);

    fx_self += dEx;
    fy_self += dEy;
    fz_self += dEz;
    atomicAdd(force_tmp + j, static_cast<float>(-dEx));
    atomicAdd(force_tmp + j + N, static_cast<float>(-dEy));
    atomicAdd(force_tmp + j + 2 * N, static_cast<float>(-dEz));
    atomicAdd(gate_adjoints + j, static_cast<RealT>(gate_adjoint));

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

template <typename GradT, typename RealT, int L, int K, int RbSize>
static __global__ void gpu_sh_compute_forces_gate_first_layer_static_layout(
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
  const GradT* gate_basic_grads,
  const RealT* gate_adjoints,
  float* force_tmp,
  float* force_self_tmp,
  float* virial_tmp)
{
  constexpr int BasicCount = K * (L + 1) * (L + 1);
  constexpr int RadialFuncs = K * (L + 1);
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || model.alpha_basic_count != BasicCount ||
      model.radial_funcs_count != RadialFuncs || model.rb_size != RbSize) {
    return;
  }

  const RealT center_adjoint = gate_adjoints[i];
  if (center_adjoint == static_cast<RealT>(0.0)) {
    return;
  }
  GradT grad_cache[BasicCount];
#pragma unroll
  for (int b = 0; b < BasicCount; ++b) {
    grad_cache[b] = static_cast<GradT>(
      center_adjoint * static_cast<RealT>(gate_basic_grads[static_cast<size_t>(b) * N + i]));
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
    compute_sh_edge_derivative_with_radial_static_layout<GradT, RealT, L, K, RbSize>(
      model, model.two_layer_gate_radial_direct_coeffs, type_i, type_j,
      dx, dy, dz, r, grad_cache, dEx, dEy, dEz);

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
    force_self_tmp[i] += static_cast<float>(fx_self);
    force_self_tmp[i + N] += static_cast<float>(fy_self);
    force_self_tmp[i + 2 * N] += static_cast<float>(fz_self);
  } else {
    atomicAdd(force_tmp + i, static_cast<float>(fx_self));
    atomicAdd(force_tmp + i + N, static_cast<float>(fy_self));
    atomicAdd(force_tmp + i + 2 * N, static_cast<float>(fz_self));
  }

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

template <typename GradT, typename RealT>
bool launch_sh_compute_forces_static(
  int lmax,
  int kmax,
  int rb_size,
  int grid_size,
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
  if (rb_size != 10 || kmax < 1 || kmax > 6) {
    return false;
  }

#define SUS2_SH_LAUNCH_STATIC_FORCE_FOR_L(LVAL)                                           \
  do {                                                                                    \
    if (kmax == 1) {                                                                      \
      gpu_sh_compute_forces_static_layout<GradT, RealT, LVAL, 1, 10>                      \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, grads,           \
          force_tmp, force_self_tmp, virial_tmp);                                         \
      return true;                                                                        \
    }                                                                                     \
    if (kmax == 2) {                                                                      \
      gpu_sh_compute_forces_static_layout<GradT, RealT, LVAL, 2, 10>                      \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, grads,           \
          force_tmp, force_self_tmp, virial_tmp);                                         \
      return true;                                                                        \
    }                                                                                     \
    if (kmax == 3) {                                                                      \
      gpu_sh_compute_forces_static_layout<GradT, RealT, LVAL, 3, 10>                      \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, grads,           \
          force_tmp, force_self_tmp, virial_tmp);                                         \
      return true;                                                                        \
    }                                                                                     \
    if (kmax == 4) {                                                                      \
      gpu_sh_compute_forces_static_layout<GradT, RealT, LVAL, 4, 10>                      \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, grads,           \
          force_tmp, force_self_tmp, virial_tmp);                                         \
      return true;                                                                        \
    }                                                                                     \
    if (kmax == 5) {                                                                      \
      gpu_sh_compute_forces_static_layout<GradT, RealT, LVAL, 5, 10>                      \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, grads,           \
          force_tmp, force_self_tmp, virial_tmp);                                         \
      return true;                                                                        \
    }                                                                                     \
    if (kmax == 6) {                                                                      \
      gpu_sh_compute_forces_static_layout<GradT, RealT, LVAL, 6, 10>                      \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, grads,           \
          force_tmp, force_self_tmp, virial_tmp);                                         \
      return true;                                                                        \
    }                                                                                     \
  } while (0)

  switch (lmax) {
    case 0:
      SUS2_SH_LAUNCH_STATIC_FORCE_FOR_L(0);
      break;
    case 1:
      SUS2_SH_LAUNCH_STATIC_FORCE_FOR_L(1);
      break;
    case 2:
      SUS2_SH_LAUNCH_STATIC_FORCE_FOR_L(2);
      break;
    case 3:
      SUS2_SH_LAUNCH_STATIC_FORCE_FOR_L(3);
      break;
    case 4:
      SUS2_SH_LAUNCH_STATIC_FORCE_FOR_L(4);
      break;
    case 5:
      SUS2_SH_LAUNCH_STATIC_FORCE_FOR_L(5);
      break;
    case 6:
      SUS2_SH_LAUNCH_STATIC_FORCE_FOR_L(6);
      break;
  }

#undef SUS2_SH_LAUNCH_STATIC_FORCE_FOR_L
  return false;
}

template <typename GradT, typename RealT>
bool launch_sh_compute_forces_gate_main_static(
  int lmax,
  int kmax,
  int rb_size,
  int grid_size,
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
  const RealT* gate_values,
  const GradT* grads,
  RealT* gate_adjoints,
  float* force_tmp,
  float* force_self_tmp,
  float* virial_tmp)
{
  if (rb_size != 10 || kmax < 1 || kmax > 6 || lmax < 0 || lmax > kMaxSHL) {
    return false;
  }

#define SUS2_SH_LAUNCH_STATIC_GATE_MAIN_FORCE_FOR_L(LVAL)                                \
  do {                                                                                    \
    if (kmax == 1) {                                                                      \
      gpu_sh_compute_forces_gate_main_static_layout<GradT, RealT, LVAL, 1, 10>            \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_values,    \
          grads, gate_adjoints, force_tmp, force_self_tmp, virial_tmp);                   \
      return true;                                                                        \
    }                                                                                     \
    if (kmax == 2) {                                                                      \
      gpu_sh_compute_forces_gate_main_static_layout<GradT, RealT, LVAL, 2, 10>            \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_values,    \
          grads, gate_adjoints, force_tmp, force_self_tmp, virial_tmp);                   \
      return true;                                                                        \
    }                                                                                     \
    if (kmax == 3) {                                                                      \
      gpu_sh_compute_forces_gate_main_static_layout<GradT, RealT, LVAL, 3, 10>            \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_values,    \
          grads, gate_adjoints, force_tmp, force_self_tmp, virial_tmp);                   \
      return true;                                                                        \
    }                                                                                     \
    if (kmax == 4) {                                                                      \
      gpu_sh_compute_forces_gate_main_static_layout<GradT, RealT, LVAL, 4, 10>            \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_values,    \
          grads, gate_adjoints, force_tmp, force_self_tmp, virial_tmp);                   \
      return true;                                                                        \
    }                                                                                     \
    if (kmax == 5) {                                                                      \
      gpu_sh_compute_forces_gate_main_static_layout<GradT, RealT, LVAL, 5, 10>            \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_values,    \
          grads, gate_adjoints, force_tmp, force_self_tmp, virial_tmp);                   \
      return true;                                                                        \
    }                                                                                     \
    if (kmax == 6) {                                                                      \
      gpu_sh_compute_forces_gate_main_static_layout<GradT, RealT, LVAL, 6, 10>            \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_values,    \
          grads, gate_adjoints, force_tmp, force_self_tmp, virial_tmp);                   \
      return true;                                                                        \
    }                                                                                     \
  } while (0)

  switch (lmax) {
    case 0:
      SUS2_SH_LAUNCH_STATIC_GATE_MAIN_FORCE_FOR_L(0);
      break;
    case 1:
      SUS2_SH_LAUNCH_STATIC_GATE_MAIN_FORCE_FOR_L(1);
      break;
    case 2:
      SUS2_SH_LAUNCH_STATIC_GATE_MAIN_FORCE_FOR_L(2);
      break;
    case 3:
      SUS2_SH_LAUNCH_STATIC_GATE_MAIN_FORCE_FOR_L(3);
      break;
    case 4:
      SUS2_SH_LAUNCH_STATIC_GATE_MAIN_FORCE_FOR_L(4);
      break;
    case 5:
      SUS2_SH_LAUNCH_STATIC_GATE_MAIN_FORCE_FOR_L(5);
      break;
    case 6:
      SUS2_SH_LAUNCH_STATIC_GATE_MAIN_FORCE_FOR_L(6);
      break;
  }

#undef SUS2_SH_LAUNCH_STATIC_GATE_MAIN_FORCE_FOR_L
  return false;
}

template <typename GradT, typename RealT>
bool launch_sh_compute_forces_gate_first_layer_static(
  int lmax,
  int kmax,
  int rb_size,
  int grid_size,
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
  const GradT* gate_basic_grads,
  const RealT* gate_adjoints,
  float* force_tmp,
  float* force_self_tmp,
  float* virial_tmp)
{
  if (rb_size != 10 || kmax < 1 || kmax > 6 || lmax < 0 || lmax > kMaxSHL) {
    return false;
  }

#define SUS2_SH_LAUNCH_STATIC_GATE_FIRST_FORCE_FOR_L(LVAL)                               \
  do {                                                                                    \
    if (kmax == 1) {                                                                      \
      gpu_sh_compute_forces_gate_first_layer_static_layout<GradT, RealT, LVAL, 1, 10>     \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_basic_grads,\
          gate_adjoints, force_tmp, force_self_tmp, virial_tmp);                         \
      return true;                                                                        \
    }                                                                                     \
    if (kmax == 2) {                                                                      \
      gpu_sh_compute_forces_gate_first_layer_static_layout<GradT, RealT, LVAL, 2, 10>     \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_basic_grads,\
          gate_adjoints, force_tmp, force_self_tmp, virial_tmp);                         \
      return true;                                                                        \
    }                                                                                     \
    if (kmax == 3) {                                                                      \
      gpu_sh_compute_forces_gate_first_layer_static_layout<GradT, RealT, LVAL, 3, 10>     \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_basic_grads,\
          gate_adjoints, force_tmp, force_self_tmp, virial_tmp);                         \
      return true;                                                                        \
    }                                                                                     \
    if (kmax == 4) {                                                                      \
      gpu_sh_compute_forces_gate_first_layer_static_layout<GradT, RealT, LVAL, 4, 10>     \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_basic_grads,\
          gate_adjoints, force_tmp, force_self_tmp, virial_tmp);                         \
      return true;                                                                        \
    }                                                                                     \
    if (kmax == 5) {                                                                      \
      gpu_sh_compute_forces_gate_first_layer_static_layout<GradT, RealT, LVAL, 5, 10>     \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_basic_grads,\
          gate_adjoints, force_tmp, force_self_tmp, virial_tmp);                         \
      return true;                                                                        \
    }                                                                                     \
    if (kmax == 6) {                                                                      \
      gpu_sh_compute_forces_gate_first_layer_static_layout<GradT, RealT, LVAL, 6, 10>     \
        <<<grid_size, kBlockSize>>>(                                                      \
          N, box, cutoff_square, use_cached_displacements, model, type, neighbor_count,   \
          neighbor_atoms, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, gate_basic_grads,\
          gate_adjoints, force_tmp, force_self_tmp, virial_tmp);                         \
      return true;                                                                        \
    }                                                                                     \
  } while (0)

  switch (lmax) {
    case 0:
      SUS2_SH_LAUNCH_STATIC_GATE_FIRST_FORCE_FOR_L(0);
      break;
    case 1:
      SUS2_SH_LAUNCH_STATIC_GATE_FIRST_FORCE_FOR_L(1);
      break;
    case 2:
      SUS2_SH_LAUNCH_STATIC_GATE_FIRST_FORCE_FOR_L(2);
      break;
    case 3:
      SUS2_SH_LAUNCH_STATIC_GATE_FIRST_FORCE_FOR_L(3);
      break;
    case 4:
      SUS2_SH_LAUNCH_STATIC_GATE_FIRST_FORCE_FOR_L(4);
      break;
    case 5:
      SUS2_SH_LAUNCH_STATIC_GATE_FIRST_FORCE_FOR_L(5);
      break;
    case 6:
      SUS2_SH_LAUNCH_STATIC_GATE_FIRST_FORCE_FOR_L(6);
      break;
  }

#undef SUS2_SH_LAUNCH_STATIC_GATE_FIRST_FORCE_FOR_L
  return false;
}

static __global__ void gpu_sh_apply_zbl(
  int N,
  Box box,
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
  double* potential,
  float* force_tmp,
  float* virial_tmp)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || !model.zbl_enabled) {
    return;
  }

  const int type_i = type[i];
  if (type_i < 0 || type_i >= model.species_count) {
    return;
  }
  const int Zi = model.zbl_atomic_numbers[type_i];
  const int count = neighbor_count[i];
  double zbl_energy = 0.0;
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
    const int type_j = type[j];
    if (type_j < 0 || type_j >= model.species_count) {
      continue;
    }
    double dx;
    double dy;
    double dz;
    load_sh_edge_displacement(
      use_cached_displacements, N, box, edge, i, j, neighbor_dx, neighbor_dy, neighbor_dz,
      x, y, z, dx, dy, dz);
    const double r2 = dx * dx + dy * dy + dz * dz;
    const int pair = type_i * model.species_count + type_j;
    if (r2 <= 0.0 || r2 >= model.zbl_pair_outer_sq[pair]) {
      continue;
    }
    const double r = sqrt(r2);
    const int Zj = model.zbl_atomic_numbers[type_j];
    const SUS2ZBLPairValue zbl = sus2_zbl_pair(
      Zi, Zj, r, model.zbl_pair_inner_cutoffs[pair], model.zbl_pair_outer_cutoffs[pair]);
    if (zbl.energy == 0.0 && zbl.dEdr == 0.0) {
      continue;
    }

    const double scale = 0.5 * zbl.dEdr / r;
    const double dEx = scale * dx;
    const double dEy = scale * dy;
    const double dEz = scale * dz;
    zbl_energy += 0.5 * zbl.energy;
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

  potential[i] += zbl_energy;
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
  SHHostModel host_model = load_sh_model(file_potential);
  const SHFactorPruningMode factor_pruning_mode =
    parse_sh_factor_pruning(host_model, num_potential_options, potential_options);
  if (factor_pruning_mode == SHFactorPruningMode::QTotal) {
    build_explicit_sh_graph_metadata(host_model, nullptr);
  }
  species_count_ = host_model.species_count;
  sh_l_max_ = host_model.sh_l_max;
  sh_k_max_ = host_model.sh_k_max;
  radial_basis_kind_ = static_cast<int>(host_model.radial_basis_kind);
  radial_funcs_count_ = host_model.radial_funcs_count;
  rb_size_ = host_model.rb_size;
  alpha_basic_count_ = host_model.alpha_basic_count;
  sh_product_count_ = static_cast<int>(host_model.products.size());
  sh_cg_block_count_ = static_cast<int>(host_model.cg_blocks.size());
  sh_cg_term_count_ = static_cast<int>(host_model.cg_terms.size());
  sh_cg_row_count_ = static_cast<int>(host_model.cg_rows.size());
  sh_cg_row_term_count_ = static_cast<int>(host_model.cg_row_terms.size());
  sh_cg_back_row_count_ = static_cast<int>(host_model.cg_back_rows.size());
  sh_cg_back_term_count_ = static_cast<int>(host_model.cg_back_terms.size());
  sh_cg_layer_count_ = host_model.sh_standard_cg_layers;
  alpha_moments_count_ = host_model.alpha_moments_count;
  alpha_scalar_moments_ = host_model.alpha_scalar_moments;
	  rc = host_model.max_dist;
	  neighbor_cutoff_ = rc;
  two_layer_gate_enabled_ = host_model.two_layer_gate_enabled;
  two_layer_gate_tanh_amplitude_ = host_model.two_layer_gate_tanh_amplitude;
  two_layer_gate_weight_count_ = host_model.two_layer_gate_weight_count;
  two_layer_gate_product_limit_ = host_model.two_layer_gate_product_limit;
	  if (host_model.zbl_enabled) {
    if (host_model.zbl_atomic_numbers.empty()) {
      if (num_potential_options < host_model.species_count) {
        sh_input_error(
          "SUS2_SH ZBL model requires zbl_atomic_numbers metadata or element symbols after the model file.");
      }
      host_model.zbl_atomic_numbers.resize(host_model.species_count);
      for (int t = 0; t < host_model.species_count; ++t) {
        const std::string symbol = potential_options[t] == nullptr ? "" : potential_options[t];
        const int atomic_number = sus2_zbl_atomic_number_from_symbol(symbol);
        if (atomic_number <= 0) {
          sh_input_error("SUS2_SH ZBL cannot map element symbol to an atomic number: " + symbol);
        }
        host_model.zbl_atomic_numbers[t] = atomic_number;
      }
    }
    if (static_cast<int>(host_model.zbl_atomic_numbers.size()) != host_model.species_count) {
      sh_input_error("SUS2_SH ZBL metadata should provide one atomic number per species.");
    }
    std::string zbl_error;
    if (!sus2_zbl_fill_pair_cutoff_tables(
          host_model.zbl_atomic_numbers,
          host_model.zbl_inner,
          host_model.zbl_outer,
          host_model.zbl_typewise_cutoff_enabled,
          host_model.zbl_typewise_cutoff_enabled ? host_model.zbl_typewise_cutoff_factor : 0.0,
          host_model.zbl_pair_inner_cutoffs,
          host_model.zbl_pair_outer_cutoffs,
          host_model.zbl_pair_outer_sq,
          &zbl_error)) {
      sh_input_error("SUS2_SH ZBL metadata error: " + zbl_error);
    }
    zbl_outer_max_ = *std::max_element(
      host_model.zbl_pair_outer_cutoffs.begin(), host_model.zbl_pair_outer_cutoffs.end());
    zbl_enabled_ = true;
    neighbor_cutoff_ = std::max(rc, zbl_outer_max_);
	  }
	  use_float_moments_ = parse_sh_float(host_model, num_potential_options, potential_options);
  if (two_layer_gate_enabled_ && !use_float_moments_) {
    sh_input_error("GPUMD SUS2_SH two-layer gate currently requires float intermediate mode.");
  }
  if (two_layer_gate_enabled_ && alpha_basic_count_ > kSHForceGradCache256) {
    sh_input_error("GPUMD SUS2_SH two-layer gate requires alpha_index_basic_count <= 256.");
  }
	  use_radial_direct_ =
    parse_sh_radial_direct(host_model, num_potential_options, potential_options);
  use_force_self_buffer_ =
    parse_sh_force_self_buffer(host_model, num_potential_options, potential_options);
  use_force_grad_cache_ =
    parse_sh_force_grad_cache(host_model, num_potential_options, potential_options) &&
    alpha_basic_count_ <= kSHForceGradCache256;
  use_cg_block_forward_ =
    parse_sh_cg_block_forward(host_model, num_potential_options, potential_options);
  use_tensor_product_parallel_ =
    parse_sh_tensor_product_parallel(host_model, num_potential_options, potential_options);
  use_compact_serial_product_ =
    parse_sh_compact_serial_product(host_model, num_potential_options, potential_options);
  use_const_forward_rows_ =
    parse_sh_const_forward_rows(host_model, num_potential_options, potential_options);
  use_product_pattern_rows_ =
    parse_sh_product_pattern_rows(host_model, num_potential_options, potential_options);
  use_parallel_back_rows_ =
    parse_sh_parallel_back_rows(host_model, num_potential_options, potential_options);
  use_packed_back_rows_ =
    parse_sh_packed_back_rows(host_model, num_potential_options, potential_options) &&
    use_float_moments_ && use_parallel_back_rows_;
  use_const_back_rows_ =
    parse_sh_const_back_rows(host_model, num_potential_options, potential_options) &&
    use_packed_back_rows_;
  if (use_tensor_product_parallel_) {
    use_cg_block_forward_ = false;
    use_compact_serial_product_ = false;
  }
  if (!host_model.sh_standard_graph_matched) {
    use_cg_block_forward_ = false;
    use_tensor_product_parallel_ = false;
  }
  if (use_cg_block_forward_) {
    use_compact_serial_product_ = false;
  }
  use_parallel_back_rows_ = use_parallel_back_rows_ && use_compact_serial_product_;
  use_product_pattern_rows_ =
    use_product_pattern_rows_ && use_compact_serial_product_ && use_float_moments_;
  tensor_product_grid_cap_ =
    parse_sh_tensor_product_grid_cap(host_model, num_potential_options, potential_options);
  profile_enabled_ = parse_sh_profile_enabled(host_model, num_potential_options, potential_options);
  profile_product_detail_ =
    parse_sh_profile_product_detail(host_model, num_potential_options, potential_options);
  profile_interval_ =
    parse_sh_profile_interval(host_model, num_potential_options, potential_options);
  if (!use_radial_direct_) {
    sh_input_error("SUS2_SH first backend currently requires direct radial evaluation.");
  }
  use_static_basic_layout_ =
    parse_sh_static_basic(host_model, num_potential_options, potential_options) &&
    has_static_full_sh_basic_layout(host_model);
  use_static_force_layout_ =
    parse_sh_static_force(host_model, num_potential_options, potential_options) &&
    has_static_full_sh_basic_layout(host_model) && alpha_basic_count_ <= kMaxSHBasics;
  use_terminal_scalar_fusion_ =
    parse_sh_terminal_scalar_fusion(host_model, num_potential_options, potential_options) &&
    use_compact_serial_product_;
  use_row_scalar_fusion_ =
    parse_sh_row_scalar_fusion(host_model, num_potential_options, potential_options) &&
    use_terminal_scalar_fusion_;
  use_terminal_dot_rows_ =
    parse_sh_terminal_dot_rows(host_model, num_potential_options, potential_options) &&
    use_terminal_scalar_fusion_ && use_compact_serial_product_ && use_float_moments_;
  use_terminal_dot_groups_ =
    parse_sh_terminal_dot_groups(host_model, num_potential_options, potential_options) &&
    use_terminal_dot_rows_;
  use_terminal_dot_premul_ =
    parse_sh_terminal_dot_premul(host_model, num_potential_options, potential_options) &&
    use_terminal_dot_groups_ && use_float_moments_;
  use_terminal_dot_row_list_ =
    parse_sh_terminal_dot_row_list(host_model, num_potential_options, potential_options) &&
    use_terminal_dot_groups_;
  use_fused_terminal_dot_ =
    parse_sh_fused_terminal_dot(host_model, num_potential_options, potential_options) &&
    use_terminal_dot_groups_ && use_parallel_back_rows_ && use_float_moments_;
  use_selective_grad_zero_ =
    parse_sh_selective_grad_zero(host_model, num_potential_options, potential_options) &&
    use_compact_serial_product_;
  use_product_basic_cache_ =
    parse_sh_product_basic_cache(host_model, num_potential_options, potential_options) &&
    use_compact_serial_product_ && use_float_moments_ &&
    host_model.alpha_basic_count <= kSHProductBasicCache;
  use_merge_back_duplicates_ =
    parse_sh_merge_back_duplicates(host_model, num_potential_options, potential_options);

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
  std::vector<int> cg_block_ints(static_cast<size_t>(sh_cg_block_count_) * 6);
  for (int b = 0; b < sh_cg_block_count_; ++b) {
    const SHCGBlockHost& block = host_model.cg_blocks[b];
    cg_block_ints[b * 6 + 0] = block.left_base;
    cg_block_ints[b * 6 + 1] = block.right_base;
    cg_block_ints[b * 6 + 2] = block.target_base;
    cg_block_ints[b * 6 + 3] = block.L;
    cg_block_ints[b * 6 + 4] = block.term_begin;
    cg_block_ints[b * 6 + 5] = block.term_count;
  }
  std::vector<int> cg_term_ints(static_cast<size_t>(sh_cg_term_count_) * 3);
  std::vector<double> cg_term_coeffs(sh_cg_term_count_);
  for (int t = 0; t < sh_cg_term_count_; ++t) {
    const SHCGTermHost& term = host_model.cg_terms[t];
    cg_term_ints[t * 3 + 0] = term.left_component;
    cg_term_ints[t * 3 + 1] = term.right_component;
    cg_term_ints[t * 3 + 2] = term.target_component;
    cg_term_coeffs[t] = term.coeff;
  }
  sh_cg_blocks_int_.resize(cg_block_ints.size());
  if (!cg_block_ints.empty()) {
    sh_cg_blocks_int_.copy_from_host(cg_block_ints.data());
  }
  sh_cg_terms_int_.resize(cg_term_ints.size());
  if (!cg_term_ints.empty()) {
    sh_cg_terms_int_.copy_from_host(cg_term_ints.data());
  }
  sh_cg_terms_coeff_.resize(cg_term_coeffs.size());
  if (!cg_term_coeffs.empty()) {
    sh_cg_terms_coeff_.copy_from_host(cg_term_coeffs.data());
  }
  if (use_float_moments_) {
    std::vector<float> cg_term_coeffs_f(cg_term_coeffs.begin(), cg_term_coeffs.end());
    sh_cg_terms_coeff_float_.resize(cg_term_coeffs_f.size());
    if (!cg_term_coeffs_f.empty()) {
      sh_cg_terms_coeff_float_.copy_from_host(cg_term_coeffs_f.data());
    }
  }
  std::vector<int> cg_row_ints(static_cast<size_t>(sh_cg_row_count_) * 5);
  for (int r = 0; r < sh_cg_row_count_; ++r) {
    const SHCGRowHost& row = host_model.cg_rows[r];
    cg_row_ints[r * 5 + 0] = row.left_base;
    cg_row_ints[r * 5 + 1] = row.right_base;
    cg_row_ints[r * 5 + 2] = row.target;
    cg_row_ints[r * 5 + 3] = row.term_begin;
    cg_row_ints[r * 5 + 4] = row.term_count;
  }
  std::vector<int> cg_row_term_ints(static_cast<size_t>(sh_cg_row_term_count_) * 2);
  std::vector<double> cg_row_term_coeffs(sh_cg_row_term_count_);
  for (int t = 0; t < sh_cg_row_term_count_; ++t) {
    const SHCGRowTermHost& term = host_model.cg_row_terms[t];
    cg_row_term_ints[t * 2 + 0] = term.left_component;
    cg_row_term_ints[t * 2 + 1] = term.right_component;
    cg_row_term_coeffs[t] = term.coeff;
  }
  sh_cg_rows_int_.resize(cg_row_ints.size());
  if (!cg_row_ints.empty()) {
    sh_cg_rows_int_.copy_from_host(cg_row_ints.data());
  }
  sh_cg_row_terms_int_.resize(cg_row_term_ints.size());
  if (!cg_row_term_ints.empty()) {
    sh_cg_row_terms_int_.copy_from_host(cg_row_term_ints.data());
  }
  sh_cg_row_terms_coeff_.resize(cg_row_term_coeffs.size());
  if (!cg_row_term_coeffs.empty()) {
    sh_cg_row_terms_coeff_.copy_from_host(cg_row_term_coeffs.data());
  }
  if (use_float_moments_) {
    std::vector<float> cg_row_term_coeffs_f(
      cg_row_term_coeffs.begin(), cg_row_term_coeffs.end());
    sh_cg_row_terms_coeff_float_.resize(cg_row_term_coeffs_f.size());
    if (!cg_row_term_coeffs_f.empty()) {
      sh_cg_row_terms_coeff_float_.copy_from_host(cg_row_term_coeffs_f.data());
    }
  }

  std::vector<int> terminal_moment_flags(alpha_moments_count_, 0);
  std::vector<int> row_scalar_moment_flags(alpha_moments_count_, 0);
  std::vector<int> cg_row_scalar_index(sh_cg_row_count_, -1);
  std::vector<int> scalar_index_by_moment(alpha_moments_count_, -1);
  for (int s = 0; s < alpha_scalar_moments_; ++s) {
    const int moment = host_model.alpha_moment_mapping[s];
    if (moment >= 0 && moment < alpha_moments_count_) {
      scalar_index_by_moment[moment] = s;
    }
  }
  int terminal_scalar_count = 0;
  if (use_terminal_scalar_fusion_) {
    std::vector<unsigned char> used_as_source(alpha_moments_count_, 0);
    for (int row = 0; row < sh_cg_row_count_; ++row) {
      const SHCGRowHost& cg_row = host_model.cg_rows[row];
      for (int t = 0; t < cg_row.term_count; ++t) {
        const SHCGRowTermHost& term = host_model.cg_row_terms[cg_row.term_begin + t];
        const int left = cg_row.left_base + term.left_component;
        const int right = cg_row.right_base + term.right_component;
        if (left >= 0 && left < alpha_moments_count_) {
          used_as_source[left] = 1;
        }
        if (right >= 0 && right < alpha_moments_count_) {
          used_as_source[right] = 1;
        }
      }
    }
    for (int row = 0; row < sh_cg_row_count_; ++row) {
      const int target = host_model.cg_rows[row].target;
      if (target >= 0 && target < alpha_moments_count_ &&
          scalar_index_by_moment[target] >= 0) {
        if (use_row_scalar_fusion_) {
          row_scalar_moment_flags[target] = 1;
          cg_row_scalar_index[row] = scalar_index_by_moment[target];
        }
        if (!used_as_source[target]) {
          terminal_moment_flags[target] = 1;
          cg_row_scalar_index[row] = scalar_index_by_moment[target];
          ++terminal_scalar_count;
        }
      }
    }
    if (terminal_scalar_count == 0) {
      use_terminal_scalar_fusion_ = false;
      use_row_scalar_fusion_ = false;
      std::fill(cg_row_scalar_index.begin(), cg_row_scalar_index.end(), -1);
      std::fill(row_scalar_moment_flags.begin(), row_scalar_moment_flags.end(), 0);
    }
  }
  sh_cg_row_scalar_index_.resize(cg_row_scalar_index.size());
  if (!cg_row_scalar_index.empty()) {
    sh_cg_row_scalar_index_.copy_from_host(cg_row_scalar_index.data());
  }
  sh_terminal_moment_flags_.resize(terminal_moment_flags.size());
  if (!terminal_moment_flags.empty()) {
    sh_terminal_moment_flags_.copy_from_host(terminal_moment_flags.data());
  }

  int terminal_dot_row_count = 0;
  int terminal_dot_term_count = 0;
  bool terminal_dot_groups_swapped = false;
  int terminal_dot_group_cached_components = 0;
  std::vector<unsigned int> dot_rows;
  if (use_terminal_dot_rows_) {
    dot_rows.assign(static_cast<size_t>(sh_cg_row_count_) * 3, 0u);
    for (int row = 0; row < sh_cg_row_count_; ++row) {
      const SHCGRowHost& cg_row = host_model.cg_rows[row];
      const int target = cg_row.target;
      if (target < 0 || target >= alpha_moments_count_ ||
          terminal_moment_flags[target] == 0 || cg_row.term_count <= 0) {
        continue;
      }
      const SHCGRowTermHost& first = host_model.cg_row_terms[cg_row.term_begin];
      const int left0 = first.left_component;
      const int right0 = first.right_component;
      const double coeff0 = first.coeff;
      bool is_dot = left0 >= 0 && left0 <= 0xffff && right0 >= 0 && right0 <= 0xffff &&
                    cg_row.term_count <= 0xffff;
      for (int t = 0; t < cg_row.term_count && is_dot; ++t) {
        const SHCGRowTermHost& term = host_model.cg_row_terms[cg_row.term_begin + t];
        is_dot = term.left_component == left0 + t &&
                 term.right_component == right0 + t &&
                 std::abs(term.coeff - coeff0) <= 1.0e-12;
      }
      if (!is_dot) {
        continue;
      }
      const float coeff = static_cast<float>(coeff0);
      unsigned int coeff_bits = 0u;
      std::memcpy(&coeff_bits, &coeff, sizeof(coeff_bits));
      dot_rows[static_cast<size_t>(row) * 3 + 0] =
        static_cast<unsigned int>(left0) | (static_cast<unsigned int>(right0) << 16);
      dot_rows[static_cast<size_t>(row) * 3 + 1] =
        static_cast<unsigned int>(cg_row.term_count);
      dot_rows[static_cast<size_t>(row) * 3 + 2] = coeff_bits;
      ++terminal_dot_row_count;
      terminal_dot_term_count += cg_row.term_count;
    }
    if (terminal_dot_row_count == 0) {
      use_terminal_dot_rows_ = false;
      sh_cg_row_dot_u32_.resize(0);
    } else {
      sh_cg_row_dot_u32_.resize(dot_rows.size());
      sh_cg_row_dot_u32_.copy_from_host(dot_rows.data());
    }
  }
  if (!use_terminal_dot_rows_) {
    sh_cg_row_dot_u32_.resize(0);
    use_terminal_dot_groups_ = false;
  }
  use_terminal_dot_groups_ =
    use_terminal_dot_groups_ && use_terminal_dot_rows_ &&
    terminal_dot_row_count > 0;
  std::vector<unsigned char> fused_terminal_dot_row_flags(sh_cg_row_count_, 0);
  std::vector<unsigned char> fused_producer_moment_flags(alpha_moments_count_, 0);
  use_fused_terminal_dot_ = use_fused_terminal_dot_ && use_terminal_dot_groups_;
  if (use_fused_terminal_dot_) {
    std::vector<int> row_by_target(alpha_moments_count_, -1);
    std::vector<std::vector<int>> consumers_by_moment(alpha_moments_count_);
    for (int row = 0; row < sh_cg_row_count_; ++row) {
      const SHCGRowHost& cg_row = host_model.cg_rows[row];
      if (cg_row.target >= 0 && cg_row.target < alpha_moments_count_) {
        row_by_target[cg_row.target] = row;
      }
      for (int t = 0; t < cg_row.term_count; ++t) {
        const SHCGRowTermHost& term = host_model.cg_row_terms[cg_row.term_begin + t];
        const int left = cg_row.left_base + term.left_component;
        const int right = cg_row.right_base + term.right_component;
        if (left >= 0 && left < alpha_moments_count_) {
          consumers_by_moment[left].push_back(row);
        }
        if (right >= 0 && right < alpha_moments_count_) {
          consumers_by_moment[right].push_back(row);
        }
      }
    }

    std::vector<unsigned char> candidate_producer(alpha_moments_count_, 0);
    for (int row = 0; row < sh_cg_row_count_; ++row) {
      const SHCGRowHost& cg_row = host_model.cg_rows[row];
      const int target = cg_row.target;
      if (target < 0 || target >= alpha_moments_count_ ||
          scalar_index_by_moment[target] >= 0 ||
          terminal_moment_flags[target] != 0 ||
          consumers_by_moment[target].empty()) {
        continue;
      }
      bool only_terminal_dot = true;
      for (size_t c = 0; c < consumers_by_moment[target].size(); ++c) {
        const int consumer = consumers_by_moment[target][c];
        if (consumer < 0 || consumer >= sh_cg_row_count_ ||
            dot_rows[static_cast<size_t>(consumer) * 3 + 1] == 0u) {
          only_terminal_dot = false;
          break;
        }
      }
      if (only_terminal_dot) {
        candidate_producer[target] = 1;
      }
    }

    auto full_candidate_vector = [&](int base, int count) {
      if (base < 0 || count <= 0 || base + count > alpha_moments_count_) {
        return false;
      }
      for (int c = 0; c < count; ++c) {
        if (candidate_producer[base + c] == 0 ||
            row_by_target[base + c] < 0) {
          return false;
        }
      }
      return true;
    };

    struct FusedEntryHost {
      int other0 = 0;
      int scalar_index = 0;
      float coeff = 0.0f;
      int other_producer = -1;
    };
    typedef std::pair<int, int> ProducerKey;
    typedef std::pair<int, int> FusedGroupKey;
    std::map<ProducerKey, int> producer_lookup;
    std::vector<ProducerKey> producers;
    std::vector<std::vector<std::pair<int, float>>> producer_component_terms;
    auto producer_id_for = [&](int base, int dot_count) {
      const ProducerKey key(base, dot_count);
      std::map<ProducerKey, int>::const_iterator found = producer_lookup.find(key);
      if (found != producer_lookup.end()) {
        return found->second;
      }
      const int producer_id = static_cast<int>(producers.size());
      producer_lookup[key] = producer_id;
      producers.push_back(key);
      for (int c = 0; c < dot_count; ++c) {
        fused_producer_moment_flags[base + c] = 1;
        const int row = row_by_target[base + c];
        std::vector<std::pair<int, float>> terms;
        const SHCGRowHost& cg_row = host_model.cg_rows[row];
        terms.reserve(cg_row.term_count);
        for (int t = 0; t < cg_row.term_count; ++t) {
          const SHCGRowTermHost& term =
            host_model.cg_row_terms[cg_row.term_begin + t];
          const int left = cg_row.left_base + term.left_component;
          const int right = cg_row.right_base + term.right_component;
          if (left < 0 || left > 0xffff || right < 0 || right > 0xffff) {
            use_fused_terminal_dot_ = false;
            break;
          }
          const int packed_lr = left | (right << 16);
          terms.push_back(std::make_pair(packed_lr, static_cast<float>(term.coeff)));
        }
        producer_component_terms.push_back(terms);
      }
      return producer_id;
    };

    std::vector<std::map<FusedGroupKey, std::vector<FusedEntryHost>>> fused_by_layer(
      static_cast<size_t>(sh_cg_layer_count_) + 1);
    for (int row = 0; row < sh_cg_row_count_ && use_fused_terminal_dot_; ++row) {
      const unsigned int dot_count_u32 = dot_rows[static_cast<size_t>(row) * 3 + 1];
      if (dot_count_u32 == 0u) {
        continue;
      }
      const int dot_count = static_cast<int>(dot_count_u32);
      const int layer = host_model.cg_rows[row].layer;
      const int scalar_index = cg_row_scalar_index[row];
      const unsigned int dot0 = dot_rows[static_cast<size_t>(row) * 3 + 0];
      const int left0 = static_cast<int>(dot0 & 0xffffu);
      const int right0 = static_cast<int>(dot0 >> 16);
      if (layer <= 0 || layer > sh_cg_layer_count_ ||
          dot_count <= 0 || dot_count > kSHTerminalDotGroupMaxCount ||
          scalar_index < 0 || scalar_index > 0xffff) {
        use_fused_terminal_dot_ = false;
        break;
      }
      const bool left_full = full_candidate_vector(left0, dot_count);
      const bool right_full = full_candidate_vector(right0, dot_count);
      if (!left_full && !right_full) {
        continue;
      }
      const bool choose_left = left_full;
      const int producer0 = choose_left ? left0 : right0;
      const int other0 = choose_left ? right0 : left0;
      const bool other_full = choose_left ? right_full : left_full;
      const int producer_id = producer_id_for(producer0, dot_count);
      const int other_producer_id = other_full ? producer_id_for(other0, dot_count) : -1;
      FusedEntryHost entry;
      entry.other0 = other0;
      entry.scalar_index = scalar_index;
      float entry_coeff = 0.0f;
      std::memcpy(&entry_coeff, &dot_rows[static_cast<size_t>(row) * 3 + 2],
                  sizeof(entry_coeff));
      if (use_terminal_dot_premul_) {
        entry_coeff *= static_cast<float>(host_model.moment_coeffs[scalar_index]);
      }
      entry.coeff = entry_coeff;
      entry.other_producer = other_producer_id;
      fused_by_layer[static_cast<size_t>(layer)][FusedGroupKey(producer_id, dot_count)].push_back(
        entry);
      fused_terminal_dot_row_flags[row] = 1;
    }

    if (use_fused_terminal_dot_) {
      bool valid_fused = !producers.empty();
      for (int moment = 0; moment < alpha_moments_count_ && valid_fused; ++moment) {
        if (fused_producer_moment_flags[moment] == 0) {
          continue;
        }
        for (size_t c = 0; c < consumers_by_moment[moment].size(); ++c) {
          const int consumer = consumers_by_moment[moment][c];
          if (consumer < 0 || consumer >= sh_cg_row_count_ ||
              fused_terminal_dot_row_flags[consumer] == 0) {
            valid_fused = false;
            break;
          }
        }
      }
      if (valid_fused) {
        std::vector<unsigned int> group_headers;
        std::vector<unsigned int> entries;
        for (int layer = 1; layer <= sh_cg_layer_count_; ++layer) {
          const std::map<FusedGroupKey, std::vector<FusedEntryHost>>& groups =
            fused_by_layer[static_cast<size_t>(layer)];
          for (std::map<FusedGroupKey, std::vector<FusedEntryHost>>::const_iterator it =
                 groups.begin();
               it != groups.end();
               ++it) {
            const int producer_id = it->first.first;
            const int dot_count = it->first.second;
            const int entry_begin = static_cast<int>(entries.size() / 3);
            const int entry_count = static_cast<int>(it->second.size());
            if (producer_id < 0 || producer_id > 0xffff || dot_count <= 0 ||
                dot_count > 0xffff || entry_count <= 0) {
              valid_fused = false;
              break;
            }
            group_headers.push_back(
              static_cast<unsigned int>(producer_id) |
              (static_cast<unsigned int>(dot_count) << 16));
            group_headers.push_back(static_cast<unsigned int>(entry_begin));
            group_headers.push_back(static_cast<unsigned int>(entry_count));
            for (size_t e = 0; e < it->second.size(); ++e) {
              const FusedEntryHost& entry = it->second[e];
              if (entry.other0 < 0 || entry.other0 > 0xffff ||
                  entry.scalar_index < 0 || entry.scalar_index > 0xffff) {
                valid_fused = false;
                break;
              }
              unsigned int coeff_bits = 0u;
              std::memcpy(&coeff_bits, &entry.coeff, sizeof(coeff_bits));
              entries.push_back(
                static_cast<unsigned int>(entry.other0) |
                (static_cast<unsigned int>(entry.scalar_index) << 16));
              entries.push_back(coeff_bits);
              entries.push_back(
                entry.other_producer >= 0
                ? static_cast<unsigned int>(entry.other_producer)
                : 0xffffffffu);
            }
            if (!valid_fused) {
              break;
            }
          }
          if (!valid_fused) {
            break;
          }
        }
        std::vector<unsigned int> producer_headers;
        std::vector<unsigned int> component_headers;
        std::vector<unsigned int> producer_terms;
        int component_cursor = 0;
        int term_cursor = 0;
        for (size_t p = 0; p < producers.size() && valid_fused; ++p) {
          const int base = producers[p].first;
          const int dot_count = producers[p].second;
          if (base < 0 || base > 0xffff || dot_count <= 0 || dot_count > 0xffff) {
            valid_fused = false;
            break;
          }
          producer_headers.push_back(
            static_cast<unsigned int>(base) |
            (static_cast<unsigned int>(dot_count) << 16));
          producer_headers.push_back(static_cast<unsigned int>(component_cursor));
          producer_headers.push_back(0u);
          for (int c = 0; c < dot_count; ++c) {
            const std::vector<std::pair<int, float>>& terms =
              producer_component_terms[static_cast<size_t>(component_cursor)];
            component_headers.push_back(static_cast<unsigned int>(term_cursor));
            component_headers.push_back(static_cast<unsigned int>(terms.size()));
            for (size_t t = 0; t < terms.size(); ++t) {
              unsigned int coeff_bits = 0u;
              std::memcpy(&coeff_bits, &terms[t].second, sizeof(coeff_bits));
              producer_terms.push_back(static_cast<unsigned int>(terms[t].first));
              producer_terms.push_back(coeff_bits);
              ++term_cursor;
            }
            ++component_cursor;
          }
        }
        if (valid_fused) {
          sh_fused_terminal_dot_group_count_ =
            static_cast<int>(group_headers.size() / 3);
          sh_fused_terminal_dot_group_entry_count_ =
            static_cast<int>(entries.size() / 3);
          sh_fused_terminal_dot_producer_count_ =
            static_cast<int>(producer_headers.size() / 3);
          sh_fused_terminal_dot_component_count_ =
            static_cast<int>(component_headers.size() / 2);
          sh_fused_terminal_dot_term_count_ =
            static_cast<int>(producer_terms.size() / 2);
          std::vector<unsigned int> packed_fused;
          packed_fused.reserve(
            group_headers.size() + entries.size() + producer_headers.size() +
            component_headers.size() + producer_terms.size());
          packed_fused.insert(packed_fused.end(), group_headers.begin(), group_headers.end());
          packed_fused.insert(packed_fused.end(), entries.begin(), entries.end());
          packed_fused.insert(
            packed_fused.end(), producer_headers.begin(), producer_headers.end());
          packed_fused.insert(
            packed_fused.end(), component_headers.begin(), component_headers.end());
          packed_fused.insert(packed_fused.end(), producer_terms.begin(), producer_terms.end());
          sh_fused_terminal_dot_u32_.resize(packed_fused.size());
          if (!packed_fused.empty()) {
            sh_fused_terminal_dot_u32_.copy_from_host(packed_fused.data());
          }
        }
      } else {
        use_fused_terminal_dot_ = false;
      }
    }
  }
  if (!use_fused_terminal_dot_) {
    sh_fused_terminal_dot_group_count_ = 0;
    sh_fused_terminal_dot_group_entry_count_ = 0;
    sh_fused_terminal_dot_producer_count_ = 0;
    sh_fused_terminal_dot_component_count_ = 0;
    sh_fused_terminal_dot_term_count_ = 0;
    sh_fused_terminal_dot_u32_.resize(0);
    std::fill(fused_terminal_dot_row_flags.begin(), fused_terminal_dot_row_flags.end(), 0);
    std::fill(fused_producer_moment_flags.begin(), fused_producer_moment_flags.end(), 0);
  }
  if (use_terminal_dot_groups_) {
    typedef std::pair<int, int> DotGroupKey;
    typedef std::vector<std::map<DotGroupKey, std::vector<std::array<unsigned int, 2>>>>
      DotGroupsByLayer;
    DotGroupsByLayer by_left_layer(
      static_cast<size_t>(sh_cg_layer_count_) + 1);
    DotGroupsByLayer by_right_layer(
      static_cast<size_t>(sh_cg_layer_count_) + 1);
    bool valid_dot_groups = true;
    for (int row = 0; row < sh_cg_row_count_ && valid_dot_groups; ++row) {
      if (use_fused_terminal_dot_ && fused_terminal_dot_row_flags[row] != 0) {
        continue;
      }
      const unsigned int dot_count_u32 = dot_rows[static_cast<size_t>(row) * 3 + 1];
      if (dot_count_u32 == 0u) {
        continue;
      }
      const int dot_count = static_cast<int>(dot_count_u32);
      const int scalar_index = cg_row_scalar_index[row];
      const int layer = host_model.cg_rows[row].layer;
      const unsigned int dot0 = dot_rows[static_cast<size_t>(row) * 3 + 0];
      const int left0 = static_cast<int>(dot0 & 0xffffu);
      const int right0 = static_cast<int>(dot0 >> 16);
      if (layer <= 0 || layer > sh_cg_layer_count_ ||
          dot_count <= 0 || dot_count > kSHTerminalDotGroupMaxCount ||
          scalar_index < 0 || scalar_index > 0xffff ||
          left0 < 0 || left0 > 0xffff || right0 < 0 || right0 > 0xffff) {
        valid_dot_groups = false;
        break;
      }
      float entry_coeff = 0.0f;
      std::memcpy(
        &entry_coeff, &dot_rows[static_cast<size_t>(row) * 3 + 2], sizeof(entry_coeff));
      if (use_terminal_dot_premul_) {
        entry_coeff *= static_cast<float>(host_model.moment_coeffs[scalar_index]);
      }
      unsigned int entry_coeff_bits = 0u;
      std::memcpy(&entry_coeff_bits, &entry_coeff, sizeof(entry_coeff_bits));
      std::array<unsigned int, 2> left_entry;
      left_entry[0] = static_cast<unsigned int>(right0) |
        (static_cast<unsigned int>(scalar_index) << 16);
      left_entry[1] = entry_coeff_bits;
      by_left_layer[static_cast<size_t>(layer)][DotGroupKey(left0, dot_count)].push_back(
        left_entry);
      std::array<unsigned int, 2> right_entry;
      right_entry[0] = static_cast<unsigned int>(left0) |
        (static_cast<unsigned int>(scalar_index) << 16);
      right_entry[1] = entry_coeff_bits;
      by_right_layer[static_cast<size_t>(layer)][DotGroupKey(right0, dot_count)].push_back(
        right_entry);
    }

    if (valid_dot_groups) {
      int left_group_count = 0;
      int right_group_count = 0;
      int left_cached_components = 0;
      int right_cached_components = 0;
      for (int layer = 1; layer <= sh_cg_layer_count_; ++layer) {
        const std::map<DotGroupKey, std::vector<std::array<unsigned int, 2>>>& left_groups =
          by_left_layer[static_cast<size_t>(layer)];
        for (std::map<DotGroupKey, std::vector<std::array<unsigned int, 2>>>::const_iterator it =
               left_groups.begin();
             it != left_groups.end();
             ++it) {
          ++left_group_count;
          left_cached_components += it->first.second;
        }
        const std::map<DotGroupKey, std::vector<std::array<unsigned int, 2>>>& right_groups =
          by_right_layer[static_cast<size_t>(layer)];
        for (std::map<DotGroupKey, std::vector<std::array<unsigned int, 2>>>::const_iterator it =
               right_groups.begin();
             it != right_groups.end();
             ++it) {
          ++right_group_count;
          right_cached_components += it->first.second;
        }
      }
      terminal_dot_groups_swapped =
        right_cached_components < left_cached_components ||
        (right_cached_components == left_cached_components &&
         right_group_count < left_group_count);
      terminal_dot_group_cached_components = terminal_dot_groups_swapped
        ? right_cached_components
        : left_cached_components;
      const DotGroupsByLayer& by_layer =
        terminal_dot_groups_swapped ? by_right_layer : by_left_layer;
      std::vector<int> layer_offsets(static_cast<size_t>(sh_cg_layer_count_) + 2, 0);
      std::vector<unsigned int> group_headers;
      std::vector<unsigned int> group_entries;
      for (int layer = 1; layer <= sh_cg_layer_count_; ++layer) {
        layer_offsets[layer] = static_cast<int>(group_headers.size() / 3);
        const std::map<DotGroupKey, std::vector<std::array<unsigned int, 2>>>& groups =
          by_layer[static_cast<size_t>(layer)];
        for (std::map<DotGroupKey, std::vector<std::array<unsigned int, 2>>>::const_iterator it =
               groups.begin();
             it != groups.end();
             ++it) {
          const int left0 = it->first.first;
          const int dot_count = it->first.second;
          const int entry_begin = static_cast<int>(group_entries.size() / 2);
          const int entry_count = static_cast<int>(it->second.size());
          if (entry_count <= 0) {
            continue;
          }
          group_headers.push_back(
            static_cast<unsigned int>(left0) |
            (static_cast<unsigned int>(dot_count) << 16));
          group_headers.push_back(static_cast<unsigned int>(entry_begin));
          group_headers.push_back(static_cast<unsigned int>(entry_count));
          for (size_t e = 0; e < it->second.size(); ++e) {
            group_entries.push_back(it->second[e][0]);
            group_entries.push_back(it->second[e][1]);
          }
        }
      }
      layer_offsets[sh_cg_layer_count_ + 1] = static_cast<int>(group_headers.size() / 3);
      sh_terminal_dot_group_count_ = static_cast<int>(group_headers.size() / 3);
      sh_terminal_dot_group_entry_count_ = static_cast<int>(group_entries.size() / 2);
      std::vector<unsigned int> packed_groups;
      packed_groups.reserve(group_headers.size() + group_entries.size());
      packed_groups.insert(packed_groups.end(), group_headers.begin(), group_headers.end());
      packed_groups.insert(packed_groups.end(), group_entries.begin(), group_entries.end());
      sh_terminal_dot_group_u32_.resize(packed_groups.size());
      if (!packed_groups.empty()) {
        sh_terminal_dot_group_u32_.copy_from_host(packed_groups.data());
      }
      sh_terminal_dot_group_layer_offsets_.resize(layer_offsets.size());
      sh_terminal_dot_group_layer_offsets_.copy_from_host(layer_offsets.data());
    } else {
      use_terminal_dot_groups_ = false;
    }
  }
  if (!use_terminal_dot_groups_) {
    use_terminal_dot_row_list_ = false;
    use_terminal_dot_premul_ = false;
    use_fused_terminal_dot_ = false;
    sh_terminal_dot_group_count_ = 0;
    sh_terminal_dot_group_entry_count_ = 0;
    sh_terminal_dot_group_u32_.resize(0);
    sh_terminal_dot_group_layer_offsets_.resize(0);
    sh_terminal_dot_nondot_row_count_ = 0;
    sh_terminal_dot_nondot_rows_.resize(0);
    sh_terminal_dot_nondot_layer_offsets_.resize(0);
    sh_fused_terminal_dot_group_count_ = 0;
    sh_fused_terminal_dot_group_entry_count_ = 0;
    sh_fused_terminal_dot_producer_count_ = 0;
    sh_fused_terminal_dot_component_count_ = 0;
    sh_fused_terminal_dot_term_count_ = 0;
    sh_fused_terminal_dot_u32_.resize(0);
    std::fill(fused_terminal_dot_row_flags.begin(), fused_terminal_dot_row_flags.end(), 0);
    std::fill(fused_producer_moment_flags.begin(), fused_producer_moment_flags.end(), 0);
  } else if (use_terminal_dot_row_list_) {
    std::vector<int> nondot_layer_offsets(static_cast<size_t>(sh_cg_layer_count_) + 2, 0);
    std::vector<int> nondot_rows;
    nondot_rows.reserve(static_cast<size_t>(sh_cg_row_count_ - terminal_dot_row_count));
    for (int layer = 1; layer <= sh_cg_layer_count_; ++layer) {
      nondot_layer_offsets[layer] = static_cast<int>(nondot_rows.size());
      const int row_begin = host_model.cg_layer_offsets[layer];
      const int row_end = host_model.cg_layer_offsets[layer + 1];
      for (int row = row_begin; row < row_end; ++row) {
        const int target = host_model.cg_rows[row].target;
        const bool fused_producer =
          use_fused_terminal_dot_ && target >= 0 && target < alpha_moments_count_ &&
          fused_producer_moment_flags[target] != 0;
        if (dot_rows[static_cast<size_t>(row) * 3 + 1] == 0u && !fused_producer) {
          nondot_rows.push_back(row);
        }
      }
    }
    nondot_layer_offsets[sh_cg_layer_count_ + 1] = static_cast<int>(nondot_rows.size());
    sh_terminal_dot_nondot_row_count_ = static_cast<int>(nondot_rows.size());
    sh_terminal_dot_nondot_rows_.resize(nondot_rows.size());
    if (!nondot_rows.empty()) {
      sh_terminal_dot_nondot_rows_.copy_from_host(nondot_rows.data());
    }
    sh_terminal_dot_nondot_layer_offsets_.resize(nondot_layer_offsets.size());
    sh_terminal_dot_nondot_layer_offsets_.copy_from_host(nondot_layer_offsets.data());
  } else {
    sh_terminal_dot_nondot_row_count_ = 0;
    sh_terminal_dot_nondot_rows_.resize(0);
    sh_terminal_dot_nondot_layer_offsets_.resize(0);
  }
  std::vector<int> active_scalar_moments_host;
  std::vector<double> active_scalar_coeffs_host;
  active_scalar_moments_host.reserve(alpha_scalar_moments_);
  active_scalar_coeffs_host.reserve(alpha_scalar_moments_);
  for (int s = 0; s < alpha_scalar_moments_; ++s) {
    const int moment = host_model.alpha_moment_mapping[s];
    const bool skip_terminal =
      use_terminal_scalar_fusion_ && moment >= 0 && moment < alpha_moments_count_ &&
      terminal_moment_flags[moment];
    const bool skip_row_scalar =
      use_row_scalar_fusion_ && moment >= 0 && moment < alpha_moments_count_ &&
      row_scalar_moment_flags[moment];
    if (skip_terminal || skip_row_scalar) {
      continue;
    }
    active_scalar_moments_host.push_back(moment);
    active_scalar_coeffs_host.push_back(host_model.moment_coeffs[s]);
  }
  active_scalar_moments_ = static_cast<int>(active_scalar_moments_host.size());
  active_scalar_moment_.resize(active_scalar_moments_host.size());
  if (!active_scalar_moments_host.empty()) {
    active_scalar_moment_.copy_from_host(active_scalar_moments_host.data());
  }
  active_scalar_coeff_.resize(active_scalar_coeffs_host.size());
  if (!active_scalar_coeffs_host.empty()) {
    active_scalar_coeff_.copy_from_host(active_scalar_coeffs_host.data());
  }
  if (use_float_moments_) {
    std::vector<float> active_scalar_coeffs_f(
      active_scalar_coeffs_host.begin(), active_scalar_coeffs_host.end());
    active_scalar_coeff_float_.resize(active_scalar_coeffs_f.size());
    if (!active_scalar_coeffs_f.empty()) {
      active_scalar_coeff_float_.copy_from_host(active_scalar_coeffs_f.data());
    }
  }

  int product_pattern_u32_count = 0;
  if (use_product_pattern_rows_) {
    std::vector<unsigned int> pattern_rows(static_cast<size_t>(sh_cg_row_count_) * 2, 0u);
    std::vector<unsigned int> pattern_headers;
    std::vector<unsigned int> pattern_terms;
    std::map<std::vector<unsigned int>, int> pattern_lookup;
    bool valid_pattern_rows = true;

    for (int row = 0; row < sh_cg_row_count_ && valid_pattern_rows; ++row) {
      const int row_base = row * 5;
      const int row_left_base = cg_row_ints[row_base + 0];
      const int row_right_base = cg_row_ints[row_base + 1];
      const int target = cg_row_ints[row_base + 2];
      const int term_begin = cg_row_ints[row_base + 3];
      const int term_count = cg_row_ints[row_base + 4];
      int left_base = 0;
      int right_base = 0;
      if (target < 0 || target > 0xffff || term_count < 0) {
        valid_pattern_rows = false;
        break;
      }
      if (term_count > 0) {
        left_base = row_left_base + cg_row_term_ints[term_begin * 2 + 0];
        right_base = row_right_base + cg_row_term_ints[term_begin * 2 + 1];
        for (int t = 1; t < term_count; ++t) {
          const int term_base = (term_begin + t) * 2;
          left_base = std::min(left_base, row_left_base + cg_row_term_ints[term_base + 0]);
          right_base = std::min(right_base, row_right_base + cg_row_term_ints[term_base + 1]);
        }
      }
      if (left_base < 0 || left_base > 0xffff || right_base < 0 || right_base > 0xffff) {
        valid_pattern_rows = false;
        break;
      }

      std::vector<unsigned int> key;
      key.reserve(static_cast<size_t>(std::max(term_count, 0)) * 2);
      for (int t = 0; t < term_count; ++t) {
        const int term = term_begin + t;
        const int term_base = term * 2;
        const int left_offset = row_left_base + cg_row_term_ints[term_base + 0] - left_base;
        const int right_offset = row_right_base + cg_row_term_ints[term_base + 1] - right_base;
        if (left_offset < 0 || left_offset > 0xffff ||
            right_offset < 0 || right_offset > 0xffff) {
          valid_pattern_rows = false;
          break;
        }
        const float coeff = static_cast<float>(cg_row_term_coeffs[term]);
        unsigned int coeff_bits = 0u;
        std::memcpy(&coeff_bits, &coeff, sizeof(coeff_bits));
        key.push_back(
          static_cast<unsigned int>(left_offset) |
          (static_cast<unsigned int>(right_offset) << 16));
        key.push_back(coeff_bits);
      }
      if (!valid_pattern_rows) {
        break;
      }

      std::map<std::vector<unsigned int>, int>::const_iterator found =
        pattern_lookup.find(key);
      int pattern_id = -1;
      if (found == pattern_lookup.end()) {
        pattern_id = static_cast<int>(pattern_lookup.size());
        if (pattern_id > 0xffff) {
          valid_pattern_rows = false;
          break;
        }
        pattern_lookup[key] = pattern_id;
        pattern_headers.push_back(static_cast<unsigned int>(pattern_terms.size() / 2));
        pattern_headers.push_back(static_cast<unsigned int>(term_count));
        pattern_terms.insert(pattern_terms.end(), key.begin(), key.end());
      } else {
        pattern_id = found->second;
      }

      pattern_rows[static_cast<size_t>(row) * 2 + 0] =
        static_cast<unsigned int>(left_base) |
        (static_cast<unsigned int>(right_base) << 16);
      pattern_rows[static_cast<size_t>(row) * 2 + 1] =
        static_cast<unsigned int>(target) |
        (static_cast<unsigned int>(pattern_id) << 16);
    }

    if (valid_pattern_rows) {
      sh_cg_row_pattern_count_ = static_cast<int>(pattern_lookup.size());
      sh_cg_row_pattern_term_count_ = static_cast<int>(pattern_terms.size() / 2);
      std::vector<unsigned int> packed_pattern;
      packed_pattern.reserve(pattern_rows.size() + pattern_headers.size() + pattern_terms.size());
      packed_pattern.insert(packed_pattern.end(), pattern_rows.begin(), pattern_rows.end());
      packed_pattern.insert(packed_pattern.end(), pattern_headers.begin(), pattern_headers.end());
      packed_pattern.insert(packed_pattern.end(), pattern_terms.begin(), pattern_terms.end());
      product_pattern_u32_count = static_cast<int>(packed_pattern.size());
      use_const_pattern_rows_ = packed_pattern.size() <= kSHMaxConstForwardU32;
      if (use_const_pattern_rows_) {
        CHECK(cudaMemcpyToSymbol(
          c_sh_forward_u32,
          packed_pattern.data(),
          packed_pattern.size() * sizeof(unsigned int)));
        sh_cg_row_pattern_u32_.resize(0);
      } else {
        sh_cg_row_pattern_u32_.resize(packed_pattern.size());
        if (!packed_pattern.empty()) {
          sh_cg_row_pattern_u32_.copy_from_host(packed_pattern.data());
        }
      }
    } else {
      use_product_pattern_rows_ = false;
      use_const_pattern_rows_ = false;
      sh_cg_row_pattern_count_ = 0;
      sh_cg_row_pattern_term_count_ = 0;
      sh_cg_row_pattern_u32_.resize(0);
    }
  } else {
    sh_cg_row_pattern_u32_.resize(0);
  }

  const int const_forward_u32_count = sh_cg_row_count_ * 3 + sh_cg_row_term_count_ * 2;
  use_const_forward_rows_ =
    use_const_forward_rows_ && !use_product_pattern_rows_ &&
    use_float_moments_ && use_compact_serial_product_ &&
    const_forward_u32_count <= kSHMaxConstForwardU32;
  if (use_const_forward_rows_) {
    std::vector<unsigned int> packed_forward(static_cast<size_t>(const_forward_u32_count), 0u);
    for (int row = 0; row < sh_cg_row_count_; ++row) {
      const int row_base = row * 5;
      const int left_base = cg_row_ints[row_base + 0];
      const int right_base = cg_row_ints[row_base + 1];
      const int target = cg_row_ints[row_base + 2];
      const int term_begin = cg_row_ints[row_base + 3];
      const int term_count = cg_row_ints[row_base + 4];
      if (left_base < 0 || left_base > 0xffff || right_base < 0 || right_base > 0xffff ||
          target < 0 || target > 0xffff || term_count < 0 || term_count > 0xffff) {
        use_const_forward_rows_ = false;
        break;
      }
      packed_forward[static_cast<size_t>(row) * 3 + 0] =
        static_cast<unsigned int>(left_base) |
        (static_cast<unsigned int>(right_base) << 16);
      packed_forward[static_cast<size_t>(row) * 3 + 1] =
        static_cast<unsigned int>(target) |
        (static_cast<unsigned int>(term_count) << 16);
      packed_forward[static_cast<size_t>(row) * 3 + 2] =
        static_cast<unsigned int>(term_begin);
    }
    if (use_const_forward_rows_) {
      const int term_offset = sh_cg_row_count_ * 3;
      for (int term = 0; term < sh_cg_row_term_count_; ++term) {
        const int term_base = term * 2;
        const int left_component = cg_row_term_ints[term_base + 0];
        const int right_component = cg_row_term_ints[term_base + 1];
        if (left_component < 0 || left_component > 0xffff ||
            right_component < 0 || right_component > 0xffff) {
          use_const_forward_rows_ = false;
          break;
        }
        packed_forward[term_offset + static_cast<size_t>(term) * 2 + 0] =
          static_cast<unsigned int>(left_component) |
          (static_cast<unsigned int>(right_component) << 16);
        const float coeff = static_cast<float>(cg_row_term_coeffs[term]);
        unsigned int coeff_bits = 0u;
        std::memcpy(&coeff_bits, &coeff, sizeof(coeff_bits));
        packed_forward[term_offset + static_cast<size_t>(term) * 2 + 1] = coeff_bits;
      }
    }
    if (use_const_forward_rows_) {
      CHECK(cudaMemcpyToSymbol(
        c_sh_forward_u32,
        packed_forward.data(),
        packed_forward.size() * sizeof(unsigned int)));
    }
  }
  sh_cg_layer_offsets_.resize(host_model.cg_layer_offsets.size());
  if (!host_model.cg_layer_offsets.empty()) {
    sh_cg_layer_offsets_.copy_from_host(host_model.cg_layer_offsets.data());
  }
  std::vector<SHCGBackRowHost> device_back_rows;
  std::vector<SHCGBackTermHost> device_back_terms;
  std::vector<int> device_back_layer_offsets(host_model.cg_back_layer_offsets.size(), 0);
  int removed_terminal_back_terms = 0;
  int merged_duplicate_back_terms = 0;
  for (int layer = 1; layer <= sh_cg_layer_count_; ++layer) {
    device_back_layer_offsets[layer] = static_cast<int>(device_back_rows.size());
    const int row_begin = host_model.cg_back_layer_offsets[layer];
    const int row_end = host_model.cg_back_layer_offsets[layer + 1];
    for (int r = row_begin; r < row_end; ++r) {
      const SHCGBackRowHost& old_row = host_model.cg_back_rows[r];
      SHCGBackRowHost new_row;
      new_row.layer = old_row.layer;
      new_row.source = old_row.source;
      new_row.term_begin = static_cast<int>(device_back_terms.size());
      new_row.term_count = 0;
      std::vector<SHCGBackTermHost> combined_terms;
      combined_terms.reserve(old_row.term_count);
      for (int t = 0; t < old_row.term_count; ++t) {
        const SHCGBackTermHost& term =
          host_model.cg_back_terms[old_row.term_begin + t];
        const bool skip_terminal =
          use_terminal_scalar_fusion_ && term.target >= 0 &&
          term.target < alpha_moments_count_ && terminal_moment_flags[term.target];
        const bool skip_fused_producer =
          use_fused_terminal_dot_ && term.target >= 0 &&
          term.target < alpha_moments_count_ && fused_producer_moment_flags[term.target] != 0;
        if (skip_terminal || skip_fused_producer) {
          ++removed_terminal_back_terms;
          continue;
        }
        bool merged = false;
        if (use_merge_back_duplicates_) {
          for (size_t u = 0; u < combined_terms.size(); ++u) {
            if (combined_terms[u].target == term.target &&
                combined_terms[u].other == term.other) {
              combined_terms[u].coeff += term.coeff;
              ++merged_duplicate_back_terms;
              merged = true;
              break;
            }
          }
        }
        if (!merged) {
          combined_terms.push_back(term);
        }
      }
      for (size_t u = 0; u < combined_terms.size(); ++u) {
        if (std::abs(combined_terms[u].coeff) < 1.0e-12) {
          continue;
        }
        device_back_terms.push_back(combined_terms[u]);
        ++new_row.term_count;
      }
      if (new_row.term_count > 0) {
        device_back_rows.push_back(new_row);
      }
    }
  }
  if (!device_back_layer_offsets.empty()) {
    device_back_layer_offsets[sh_cg_layer_count_ + 1] =
      static_cast<int>(device_back_rows.size());
  }
  if (use_terminal_scalar_fusion_ && removed_terminal_back_terms == 0) {
    use_terminal_scalar_fusion_ = false;
  }
  sh_cg_back_row_count_ = static_cast<int>(device_back_rows.size());
  sh_cg_back_term_count_ = static_cast<int>(device_back_terms.size());

  std::vector<int> cg_back_row_ints(static_cast<size_t>(sh_cg_back_row_count_) * 3);
  for (int r = 0; r < sh_cg_back_row_count_; ++r) {
    const SHCGBackRowHost& row = device_back_rows[r];
    cg_back_row_ints[r * 3 + 0] = row.source;
    cg_back_row_ints[r * 3 + 1] = row.term_begin;
    cg_back_row_ints[r * 3 + 2] = row.term_count;
  }
  std::vector<int> cg_back_term_ints(static_cast<size_t>(sh_cg_back_term_count_) * 2);
  std::vector<double> cg_back_term_coeffs(sh_cg_back_term_count_);
  for (int t = 0; t < sh_cg_back_term_count_; ++t) {
    const SHCGBackTermHost& term = device_back_terms[t];
    cg_back_term_ints[t * 2 + 0] = term.target;
    cg_back_term_ints[t * 2 + 1] = term.other;
    cg_back_term_coeffs[t] = term.coeff;
  }
  sh_cg_back_rows_int_.resize(cg_back_row_ints.size());
  if (!cg_back_row_ints.empty()) {
    sh_cg_back_rows_int_.copy_from_host(cg_back_row_ints.data());
  }
  sh_cg_back_terms_int_.resize(cg_back_term_ints.size());
  if (!cg_back_term_ints.empty()) {
    sh_cg_back_terms_int_.copy_from_host(cg_back_term_ints.data());
  }
  sh_cg_back_terms_coeff_.resize(cg_back_term_coeffs.size());
  if (!cg_back_term_coeffs.empty()) {
    sh_cg_back_terms_coeff_.copy_from_host(cg_back_term_coeffs.data());
  }
  if (use_float_moments_) {
    std::vector<float> cg_back_term_coeffs_f(
      cg_back_term_coeffs.begin(), cg_back_term_coeffs.end());
    sh_cg_back_terms_coeff_float_.resize(cg_back_term_coeffs_f.size());
    if (!cg_back_term_coeffs_f.empty()) {
      sh_cg_back_terms_coeff_float_.copy_from_host(cg_back_term_coeffs_f.data());
    }
  }
  if (use_packed_back_rows_) {
    const size_t packed_count =
      static_cast<size_t>(sh_cg_back_row_count_) * 2 +
      static_cast<size_t>(sh_cg_back_term_count_) * 2;
    std::vector<unsigned int> packed_back(packed_count, 0u);
    for (int row = 0; row < sh_cg_back_row_count_ && use_packed_back_rows_; ++row) {
      const int row_base = row * 3;
      const int source = cg_back_row_ints[row_base + 0];
      const int term_begin = cg_back_row_ints[row_base + 1];
      const int term_count = cg_back_row_ints[row_base + 2];
      if (source < 0 || source > 0xffff || term_count < 0 || term_count > 0xffff) {
        use_packed_back_rows_ = false;
        break;
      }
      packed_back[static_cast<size_t>(row) * 2 + 0] =
        static_cast<unsigned int>(source) |
        (static_cast<unsigned int>(term_count) << 16);
      packed_back[static_cast<size_t>(row) * 2 + 1] = static_cast<unsigned int>(term_begin);
    }
    const size_t term_offset = static_cast<size_t>(sh_cg_back_row_count_) * 2;
    for (int term = 0; term < sh_cg_back_term_count_ && use_packed_back_rows_; ++term) {
      const int term_base = term * 2;
      const int target = cg_back_term_ints[term_base + 0];
      const int other = cg_back_term_ints[term_base + 1];
      if (target < 0 || target > 0xffff || other < 0 || other > 0xffff) {
        use_packed_back_rows_ = false;
        break;
      }
      packed_back[term_offset + static_cast<size_t>(term) * 2 + 0] =
        static_cast<unsigned int>(target) |
        (static_cast<unsigned int>(other) << 16);
      const float coeff = static_cast<float>(cg_back_term_coeffs[term]);
      unsigned int coeff_bits = 0u;
      std::memcpy(&coeff_bits, &coeff, sizeof(coeff_bits));
      packed_back[term_offset + static_cast<size_t>(term) * 2 + 1] = coeff_bits;
    }
    if (use_packed_back_rows_) {
      size_t const_back_offset = 0;
      if (use_const_pattern_rows_) {
        const_back_offset =
          static_cast<size_t>(sh_cg_row_count_) * 2 +
          static_cast<size_t>(sh_cg_row_pattern_count_) * 2 +
          static_cast<size_t>(sh_cg_row_pattern_term_count_) * 2;
      } else if (use_const_forward_rows_) {
        const_back_offset =
          static_cast<size_t>(sh_cg_row_count_) * 3 +
          static_cast<size_t>(sh_cg_row_term_count_) * 2;
      }
      if (use_const_back_rows_ &&
          const_back_offset + packed_back.size() <= kSHMaxConstForwardU32) {
        CHECK(cudaMemcpyToSymbol(
          c_sh_forward_u32,
          packed_back.data(),
          packed_back.size() * sizeof(unsigned int),
          const_back_offset * sizeof(unsigned int)));
        sh_cg_back_packed_u32_.resize(0);
      } else {
        use_const_back_rows_ = false;
        sh_cg_back_packed_u32_.resize(packed_back.size());
        if (!packed_back.empty()) {
          sh_cg_back_packed_u32_.copy_from_host(packed_back.data());
        }
      }
    }
  }
  if (!use_packed_back_rows_) {
    sh_cg_back_packed_u32_.resize(0);
    use_const_back_rows_ = false;
  }
  sh_cg_back_layer_offsets_.resize(device_back_layer_offsets.size());
  if (!device_back_layer_offsets.empty()) {
    sh_cg_back_layer_offsets_.copy_from_host(device_back_layer_offsets.data());
  }

  std::vector<int> grad_zero_moments;
  if (use_selective_grad_zero_) {
    std::vector<unsigned char> needs_zero(alpha_moments_count_, 0);
    auto mark_grad_zero = [&](int moment) {
      if (moment >= 0 && moment < alpha_moments_count_) {
        needs_zero[moment] = 1;
      }
    };
    for (int b = 0; b < alpha_basic_count_; ++b) {
      mark_grad_zero(b);
    }
    for (int moment : active_scalar_moments_host) {
      mark_grad_zero(moment);
    }
    if (use_terminal_scalar_fusion_) {
      for (int row = 0; row < sh_cg_row_count_; ++row) {
        const SHCGRowHost& cg_row = host_model.cg_rows[row];
        const bool scalar_row = cg_row_scalar_index[row] >= 0;
        if (!scalar_row) {
          continue;
        }
        const bool terminal_row =
          cg_row.target >= 0 && cg_row.target < alpha_moments_count_ &&
          terminal_moment_flags[cg_row.target] != 0;
        if (terminal_row) {
          for (int t = 0; t < cg_row.term_count; ++t) {
            const SHCGRowTermHost& term =
              host_model.cg_row_terms[cg_row.term_begin + t];
            mark_grad_zero(cg_row.left_base + term.left_component);
            mark_grad_zero(cg_row.right_base + term.right_component);
          }
        } else if (use_row_scalar_fusion_) {
          mark_grad_zero(cg_row.target);
        }
      }
    }
    for (const SHCGBackRowHost& row : device_back_rows) {
      mark_grad_zero(row.source);
    }
    for (const SHCGBackTermHost& term : device_back_terms) {
      mark_grad_zero(term.target);
    }
    grad_zero_moments.reserve(alpha_moments_count_);
    for (int moment = 0; moment < alpha_moments_count_; ++moment) {
      if (needs_zero[moment]) {
        grad_zero_moments.push_back(moment);
      }
    }
    sh_grad_zero_count_ = static_cast<int>(grad_zero_moments.size());
    if (sh_grad_zero_count_ <= 0 ||
        sh_grad_zero_count_ * 10 >= alpha_moments_count_ * 9) {
      use_selective_grad_zero_ = false;
      sh_grad_zero_count_ = 0;
      grad_zero_moments.clear();
    }
  }
  sh_grad_zero_moments_.resize(grad_zero_moments.size());
  if (!grad_zero_moments.empty()) {
    sh_grad_zero_moments_.copy_from_host(grad_zero_moments.data());
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
  if (two_layer_gate_enabled_) {
    std::vector<float> gate_radial_coeffs;
    std::vector<float> gate_radial_scal_s;
    build_direct_radial_tables(
      host_model, gate_radial_coeffs, gate_radial_scal_s,
      &host_model.two_layer_gate_radial_coeffs);
    if (gate_radial_scal_s.size() != radial_scal_s.size() ||
        !std::equal(gate_radial_scal_s.begin(), gate_radial_scal_s.end(), radial_scal_s.begin())) {
      sh_input_error("SUS2_SH gate radial scal/shift table unexpectedly differs from main table.");
    }
    two_layer_gate_radial_direct_coeffs_.resize(gate_radial_coeffs.size());
    two_layer_gate_radial_direct_coeffs_.copy_from_host(gate_radial_coeffs.data());
    two_layer_gate_moment_indices_.resize(host_model.two_layer_gate_moment_indices.size());
    two_layer_gate_moment_indices_.copy_from_host(host_model.two_layer_gate_moment_indices.data());
    std::vector<float> gate_weights(
      host_model.two_layer_gate_weights.begin(), host_model.two_layer_gate_weights.end());
    two_layer_gate_weights_float_.resize(gate_weights.size());
    two_layer_gate_weights_float_.copy_from_host(gate_weights.data());
    two_layer_gate_needed_moment_flags_.resize(
      host_model.two_layer_gate_needed_moment_flags.size());
    two_layer_gate_needed_moment_flags_.copy_from_host(
      host_model.two_layer_gate_needed_moment_flags.data());
    two_layer_gate_moment_weights_float_.resize(
      host_model.two_layer_gate_moment_weights_float.size());
    two_layer_gate_moment_weights_float_.copy_from_host(
      host_model.two_layer_gate_moment_weights_float.data());
    std::vector<float> additive_coeffs(
      host_model.two_layer_gate_additive_coeffs.begin(),
      host_model.two_layer_gate_additive_coeffs.end());
    two_layer_gate_additive_coeffs_float_.resize(additive_coeffs.size());
    two_layer_gate_additive_coeffs_float_.copy_from_host(additive_coeffs.data());
  }
	  if (zbl_enabled_) {
    zbl_atomic_numbers_.resize(host_model.zbl_atomic_numbers.size());
    zbl_pair_inner_cutoffs_.resize(host_model.zbl_pair_inner_cutoffs.size());
    zbl_pair_outer_cutoffs_.resize(host_model.zbl_pair_outer_cutoffs.size());
    zbl_pair_outer_sq_.resize(host_model.zbl_pair_outer_sq.size());
    zbl_atomic_numbers_.copy_from_host(host_model.zbl_atomic_numbers.data());
    zbl_pair_inner_cutoffs_.copy_from_host(host_model.zbl_pair_inner_cutoffs.data());
    zbl_pair_outer_cutoffs_.copy_from_host(host_model.zbl_pair_outer_cutoffs.data());
    zbl_pair_outer_sq_.copy_from_host(host_model.zbl_pair_outer_sq.data());
  }

  neighbor_count_.resize(num_atoms);
  cell_contents_.resize(num_atoms);
  neighbor_cache_.initialize(neighbor_cutoff_, num_atoms, 512);
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
	  if (zbl_enabled_) {
	    printf(
	      "SUS2-SH GPUMD ZBL enabled: inner=%g A, outer=%g A, typewise_cutoff=%s, typewise_factor=%g, neighbor_cutoff=%g A.\n",
      host_model.zbl_typewise_cutoff_enabled ? 0.0 : host_model.zbl_inner,
      host_model.zbl_outer,
      host_model.zbl_typewise_cutoff_enabled ? "on" : "off",
	      host_model.zbl_typewise_cutoff_factor,
	      neighbor_cutoff_);
	  }
  if (two_layer_gate_enabled_) {
    printf(
      "SUS2-SH GPUMD two-layer gate enabled: tanh_amplitude=%g, gate_weights=%d, gate_product_limit=%d/%d, additive_coeffs=%zu.\n",
      two_layer_gate_tanh_amplitude_,
      two_layer_gate_weight_count_,
      two_layer_gate_product_limit_,
      sh_product_count_,
      host_model.two_layer_gate_additive_coeffs.size());
  }
	  printf(
    "SUS2-SH GPUMD precision mode: %s; force self-buffer: %s; force basic-grad cache: %s; static basic: %s; static force: %s; terminal scalar fusion: %s; terminal dot rows: %s; terminal dot groups: %s; terminal dot premul: %s; terminal dot row-list: %s; fused terminal dot: %s; selective grad zero: %s; product basic cache: %s; row scalar fusion: %s; cg-block forward: %s; compact serial product: %s; product pattern rows: %s; const pattern rows: %s; parallel back rows: %s; packed back rows: %s; const forward rows: %s; const back rows: %s; tensor-product parallel: %s; tensor grid cap=%d.\n",
    use_float_moments_ ? "NEP-like float moments/gradients/local arithmetic" : "double moments/local arithmetic",
    use_force_self_buffer_ ? "on" : "off",
    use_force_grad_cache_ ? "on" : "off",
    use_static_basic_layout_ ? "on" : "off",
    use_static_force_layout_ ? "on" : "off",
    use_terminal_scalar_fusion_ ? "on" : "off",
    use_terminal_dot_rows_ ? "on" : "off",
    use_terminal_dot_groups_ ? "on" : "off",
    use_terminal_dot_premul_ ? "on" : "off",
    use_terminal_dot_row_list_ ? "on" : "off",
    use_fused_terminal_dot_ ? "on" : "off",
    use_selective_grad_zero_ ? "on" : "off",
    use_product_basic_cache_ ? "on" : "off",
    use_row_scalar_fusion_ ? "on" : "off",
    use_cg_block_forward_ ? "on" : "off",
    use_compact_serial_product_ ? "on" : "off",
    use_product_pattern_rows_ ? "on" : "off",
    use_const_pattern_rows_ ? "on" : "off",
    use_parallel_back_rows_ ? "on" : "off",
    use_packed_back_rows_ ? "on" : "off",
    use_const_forward_rows_ ? "on" : "off",
    use_const_back_rows_ ? "on" : "off",
    use_tensor_product_parallel_ ? "on" : "off",
    tensor_product_grid_cap_);
  printf(
    "SUS2-SH graph metadata: factor_pruning=%s, mode=%s, tensor_blocks=%d, cg_blocks=%d, cg_terms=%d, cg_rows=%d, cg_row_terms=%d, cg_back_rows=%d, cg_back_terms=%d, layers=%d.\n",
    factor_pruning_mode == SHFactorPruningMode::QTotal ? "q-total" : "legacy",
    host_model.sh_standard_graph_matched ? "standard" : "explicit",
    host_model.sh_standard_tensor_blocks,
    host_model.sh_standard_cg_blocks,
    host_model.sh_standard_cg_terms,
    sh_cg_row_count_,
    sh_cg_row_term_count_,
    sh_cg_back_row_count_,
    sh_cg_back_term_count_,
    host_model.sh_standard_cg_layers);
  if (use_product_pattern_rows_) {
    printf(
      "SUS2-SH product pattern rows: patterns=%d, pattern_terms=%d, packed_u32=%d.\n",
      sh_cg_row_pattern_count_,
      sh_cg_row_pattern_term_count_,
      product_pattern_u32_count);
  }
  if (use_terminal_scalar_fusion_) {
    printf(
      "SUS2-SH terminal scalar fusion: terminal_scalars=%d, active_scalar_seeds=%d, removed_back_terms=%d.\n",
      terminal_scalar_count,
      active_scalar_moments_,
      removed_terminal_back_terms);
  }
  if (merged_duplicate_back_terms > 0) {
    printf("SUS2-SH merged duplicate back terms: %d.\n", merged_duplicate_back_terms);
  }
  if (use_terminal_dot_rows_) {
    printf(
      "SUS2-SH terminal dot rows: rows=%d, terms=%d.\n",
      terminal_dot_row_count,
      terminal_dot_term_count);
  }
  if (use_terminal_dot_groups_) {
    printf(
      "SUS2-SH terminal dot groups: groups=%d, entries=%d, orientation=%s, cached_components=%d, row_list=%s, nondot_rows=%d.\n",
      sh_terminal_dot_group_count_,
      sh_terminal_dot_group_entry_count_,
      terminal_dot_groups_swapped ? "right" : "left",
      terminal_dot_group_cached_components,
      use_terminal_dot_row_list_ ? "on" : "off",
      sh_terminal_dot_nondot_row_count_);
  }
  if (use_fused_terminal_dot_) {
    printf(
      "SUS2-SH fused terminal dot: groups=%d, entries=%d, producers=%d, components=%d, producer_terms=%d.\n",
      sh_fused_terminal_dot_group_count_,
      sh_fused_terminal_dot_group_entry_count_,
      sh_fused_terminal_dot_producer_count_,
      sh_fused_terminal_dot_component_count_,
      sh_fused_terminal_dot_term_count_);
  }
  if (use_selective_grad_zero_) {
    printf(
      "SUS2-SH selective grad zero: moments=%d/%d.\n",
      sh_grad_zero_count_,
      alpha_moments_count_);
  }
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
    if (two_layer_gate_enabled_) {
      const size_t gate_basic_size = static_cast<size_t>(alpha_basic_count_) * num_atoms;
      if (two_layer_gate_basic_grads_float_.size() != gate_basic_size) {
        two_layer_gate_basic_grads_float_.resize(gate_basic_size);
      }
      if (two_layer_gate_values_float_.size() != static_cast<size_t>(num_atoms)) {
        two_layer_gate_values_float_.resize(num_atoms);
      }
      if (two_layer_gate_adjoints_float_.size() != static_cast<size_t>(num_atoms)) {
        two_layer_gate_adjoints_float_.resize(num_atoms);
      }
    }
	  } else {
	    if (moment_vals_.size() != moment_size) {
	      moment_vals_.resize(moment_size);
	    }
	    if (moment_grads_.size() != moment_size) {
	      moment_grads_.resize(moment_size);
	    }
    if (two_layer_gate_enabled_) {
      const size_t gate_basic_size = static_cast<size_t>(alpha_basic_count_) * num_atoms;
      if (two_layer_gate_basic_grads_.size() != gate_basic_size) {
        two_layer_gate_basic_grads_.resize(gate_basic_size);
      }
      if (two_layer_gate_values_.size() != static_cast<size_t>(num_atoms)) {
        two_layer_gate_values_.resize(num_atoms);
      }
      if (two_layer_gate_adjoints_.size() != static_cast<size_t>(num_atoms)) {
        two_layer_gate_adjoints_.resize(num_atoms);
      }
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
  const double cutoff_square = neighbor_cutoff_ * neighbor_cutoff_;
  const double volume = box.get_volume();
  box.thickness_x = volume / box.get_area(0);
  box.thickness_y = volume / box.get_area(1);
  box.thickness_z = volume / box.get_area(2);
  const int sx_range = periodic_image_range(box.pbc_x, neighbor_cutoff_, box.thickness_x);
  const int sy_range = periodic_image_range(box.pbc_y, neighbor_cutoff_, box.thickness_y);
  const int sz_range = periodic_image_range(box.pbc_z, neighbor_cutoff_, box.thickness_z);
  const bool needs_multi_image =
    (box.pbc_x && box.thickness_x < 2.0 * neighbor_cutoff_) ||
    (box.pbc_y && box.thickness_y < 2.0 * neighbor_cutoff_) ||
    (box.pbc_z && box.thickness_z < 2.0 * neighbor_cutoff_);

  if (!needs_multi_image) {
    use_cached_neighbor_displacements_ = false;
    const size_t edge_capacity = static_cast<size_t>(num_atoms) * neighbor_capacity_;
    if (neighbor_atom_.size() != edge_capacity) {
      neighbor_atom_.resize(edge_capacity);
    }
    neighbor_cache_.find_neighbor_global(neighbor_cutoff_, box, type, position);
    neighbor_cache_.find_local_neighbor_from_global(
      neighbor_cutoff_, box, position, neighbor_count_, neighbor_atom_);
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
  const int product_grid_size = (num_atoms - 1) / kProductBlockSize + 1;
  const int tensor_grid_size = std::max(
    1,
    std::min(
      tensor_product_grid_cap_,
      static_cast<int>(
        (static_cast<size_t>(num_atoms) * std::max(sh_cg_row_count_, 1) + kBlockSize - 1) /
        kBlockSize)));
  const size_t moment_size = static_cast<size_t>(alpha_moments_count_) * num_atoms;
  const size_t force_size = static_cast<size_t>(num_atoms) * 3;
  const size_t virial_size = static_cast<size_t>(num_atoms) * 9;
  stage_start = profile_start();
  if (use_float_moments_) {
    if (!use_cg_block_forward_ && !use_tensor_product_parallel_ &&
        !use_compact_serial_product_) {
      CHECK(gpuMemset(moment_vals_float_.data(), 0, moment_size * sizeof(float)));
    }
    if (use_selective_grad_zero_) {
      gpu_sh_zero_selected_grads<float><<<grid_size, kBlockSize>>>(
        num_atoms,
        sh_grad_zero_count_,
        sh_grad_zero_moments_.data(),
        moment_grads_float_.data());
      GPU_CHECK_KERNEL
    } else {
      CHECK(gpuMemset(moment_grads_float_.data(), 0, moment_size * sizeof(float)));
    }
  } else {
    if (!use_cg_block_forward_ && !use_tensor_product_parallel_ &&
        !use_compact_serial_product_) {
      CHECK(gpuMemset(moment_vals_.data(), 0, moment_size * sizeof(double)));
    }
    if (use_selective_grad_zero_) {
      gpu_sh_zero_selected_grads<double><<<grid_size, kBlockSize>>>(
        num_atoms,
        sh_grad_zero_count_,
        sh_grad_zero_moments_.data(),
        moment_grads_.data());
      GPU_CHECK_KERNEL
    } else {
      CHECK(gpuMemset(moment_grads_.data(), 0, moment_size * sizeof(double)));
    }
  }
  CHECK(gpuMemset(force_tmp_.data(), 0, force_size * sizeof(float)));
  profile_stop(sh_profile_memset, stage_start);

  SHDeviceModel model{
    species_count_,
    sh_l_max_,
    radial_basis_kind_,
    radial_funcs_count_,
    rb_size_,
    alpha_basic_count_,
    sh_product_count_,
    sh_cg_block_count_,
    sh_cg_term_count_,
    sh_cg_row_count_,
    sh_cg_row_term_count_,
    sh_cg_row_pattern_count_,
    sh_cg_row_pattern_term_count_,
    sh_cg_back_row_count_,
    sh_cg_back_term_count_,
    sh_cg_layer_count_,
    sh_terminal_dot_group_count_,
    sh_terminal_dot_group_entry_count_,
    sh_terminal_dot_nondot_row_count_,
    sh_fused_terminal_dot_group_count_,
    sh_fused_terminal_dot_group_entry_count_,
    sh_fused_terminal_dot_producer_count_,
    sh_fused_terminal_dot_component_count_,
    sh_fused_terminal_dot_term_count_,
    alpha_moments_count_,
    alpha_scalar_moments_,
    active_scalar_moments_,
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
    sh_cg_blocks_int_.data(),
    sh_cg_terms_int_.data(),
    sh_cg_terms_coeff_.data(),
    use_float_moments_ ? sh_cg_terms_coeff_float_.data() : nullptr,
    sh_cg_rows_int_.data(),
    sh_cg_row_terms_int_.data(),
    sh_cg_row_terms_coeff_.data(),
    use_float_moments_ ? sh_cg_row_terms_coeff_float_.data() : nullptr,
    sh_cg_row_scalar_index_.data(),
    sh_terminal_moment_flags_.data(),
    use_terminal_dot_rows_ ? sh_cg_row_dot_u32_.data() : nullptr,
    use_product_pattern_rows_ && !use_const_pattern_rows_ ? sh_cg_row_pattern_u32_.data() : nullptr,
    use_terminal_dot_groups_ ? sh_terminal_dot_group_u32_.data() : nullptr,
    use_terminal_dot_groups_ ? sh_terminal_dot_group_layer_offsets_.data() : nullptr,
    use_terminal_dot_row_list_ ? sh_terminal_dot_nondot_rows_.data() : nullptr,
    use_terminal_dot_row_list_ ? sh_terminal_dot_nondot_layer_offsets_.data() : nullptr,
    use_fused_terminal_dot_ ? sh_fused_terminal_dot_u32_.data() : nullptr,
    sh_cg_back_rows_int_.data(),
    sh_cg_back_terms_int_.data(),
    sh_cg_back_terms_coeff_.data(),
    use_float_moments_ ? sh_cg_back_terms_coeff_float_.data() : nullptr,
    sh_cg_back_layer_offsets_.data(),
    use_packed_back_rows_ ? sh_cg_back_packed_u32_.data() : nullptr,
    active_scalar_moment_.data(),
    active_scalar_coeff_.data(),
    use_float_moments_ ? active_scalar_coeff_float_.data() : nullptr,
    sh_cg_layer_offsets_.data(),
	    alpha_moment_mapping_.data(),
	    radial_direct_coeffs_.data(),
	    radial_direct_scal_s_.data(),
    two_layer_gate_enabled_,
    two_layer_gate_weight_count_,
    two_layer_gate_product_limit_,
    two_layer_gate_tanh_amplitude_,
    two_layer_gate_enabled_ ? two_layer_gate_radial_direct_coeffs_.data() : nullptr,
    two_layer_gate_enabled_ ? two_layer_gate_moment_indices_.data() : nullptr,
    two_layer_gate_enabled_ ? two_layer_gate_weights_float_.data() : nullptr,
    two_layer_gate_enabled_ ? two_layer_gate_needed_moment_flags_.data() : nullptr,
    two_layer_gate_enabled_ ? two_layer_gate_moment_weights_float_.data() : nullptr,
    two_layer_gate_enabled_ ? two_layer_gate_additive_coeffs_float_.data() : nullptr,
	    zbl_enabled_,
    zbl_enabled_ ? zbl_atomic_numbers_.data() : nullptr,
    zbl_enabled_ ? zbl_pair_inner_cutoffs_.data() : nullptr,
    zbl_enabled_ ? zbl_pair_outer_cutoffs_.data() : nullptr,
    zbl_enabled_ ? zbl_pair_outer_sq_.data() : nullptr,
    use_float_moments_,
    use_const_forward_rows_,
    use_product_pattern_rows_,
    use_const_pattern_rows_,
    use_terminal_scalar_fusion_,
    use_packed_back_rows_,
    use_const_back_rows_,
    use_terminal_dot_rows_,
    use_terminal_dot_groups_,
	    use_terminal_dot_premul_,
		    use_product_basic_cache_};
  float* force_self_tmp_ptr = use_force_self_buffer_ ? force_self_tmp_.data() : nullptr;

  if (two_layer_gate_enabled_) {
    stage_start = profile_start();
    CHECK(gpuMemset(moment_vals_float_.data(), 0, moment_size * sizeof(float)));
    CHECK(gpuMemset(moment_grads_float_.data(), 0, moment_size * sizeof(float)));
    const bool launched_gate_basic_static =
      use_static_basic_layout_ &&
      launch_sh_compute_basic_with_radial_static<float>(
        sh_l_max_, sh_k_max_, rb_size_, grid_size, num_atoms, box, rc * rc,
        use_cached_neighbor_displacements_, model, model.two_layer_gate_radial_direct_coeffs,
        type.data(), neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(),
        neighbor_dy_.data(), neighbor_dz_.data(), position.data(),
        position.data() + num_atoms, position.data() + 2 * num_atoms,
        moment_vals_float_.data());
    if (launched_gate_basic_static) {
      GPU_CHECK_KERNEL
    } else if (alpha_basic_count_ <= 64) {
      gpu_sh_compute_basic_with_radial<float, 64><<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model,
        model.two_layer_gate_radial_direct_coeffs, type.data(), neighbor_count_.data(),
        neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), neighbor_dz_.data(),
        position.data(), position.data() + num_atoms, position.data() + 2 * num_atoms,
        moment_vals_float_.data());
    } else if (alpha_basic_count_ <= 128) {
      gpu_sh_compute_basic_with_radial<float, 128><<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model,
        model.two_layer_gate_radial_direct_coeffs, type.data(), neighbor_count_.data(),
        neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), neighbor_dz_.data(),
        position.data(), position.data() + num_atoms, position.data() + 2 * num_atoms,
        moment_vals_float_.data());
    } else {
      gpu_sh_compute_basic_with_radial<float><<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model,
        model.two_layer_gate_radial_direct_coeffs, type.data(), neighbor_count_.data(),
        neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), neighbor_dz_.data(),
        position.data(), position.data() + num_atoms, position.data() + 2 * num_atoms,
        moment_vals_float_.data());
    }
    GPU_CHECK_KERNEL
    profile_stop(sh_profile_basic, stage_start);

	    stage_start = profile_start();
    if (use_compact_serial_product_) {
      if (use_product_pattern_rows_) {
        gpu_sh_gate_forward_backward_compact_rows<float, float, 1>
          <<<product_grid_size, kProductBlockSize>>>(
          num_atoms, model, moment_vals_float_.data(), moment_grads_float_.data(),
          two_layer_gate_values_float_.data());
      } else {
        gpu_sh_gate_forward_backward_compact_rows<float, float, 0>
          <<<product_grid_size, kProductBlockSize>>>(
          num_atoms, model, moment_vals_float_.data(), moment_grads_float_.data(),
          two_layer_gate_values_float_.data());
      }
    } else {
      gpu_sh_gate_forward_backward_products<float, float><<<product_grid_size, kProductBlockSize>>>(
        num_atoms, model, moment_vals_float_.data(), moment_grads_float_.data(),
        two_layer_gate_values_float_.data());
    }
    GPU_CHECK_KERNEL
    gpu_sh_copy_basic_grads<float><<<grid_size, kBlockSize>>>(
      num_atoms, alpha_basic_count_, moment_grads_float_.data(),
      two_layer_gate_basic_grads_float_.data());
    GPU_CHECK_KERNEL
    profile_stop(sh_profile_product, stage_start);

    stage_start = profile_start();
    CHECK(gpuMemset(moment_vals_float_.data(), 0, moment_size * sizeof(float)));
    CHECK(gpuMemset(moment_grads_float_.data(), 0, moment_size * sizeof(float)));
    CHECK(gpuMemset(two_layer_gate_adjoints_float_.data(), 0, num_atoms * sizeof(float)));
    CHECK(gpuMemset(virial_tmp_.data(), 0, virial_size * sizeof(float)));
    const bool launched_gate_main_basic_static =
      use_static_basic_layout_ &&
      launch_sh_compute_basic_gate_main_static<float>(
        sh_l_max_, sh_k_max_, rb_size_, grid_size, num_atoms, box, rc * rc,
        use_cached_neighbor_displacements_, model, type.data(), neighbor_count_.data(),
        neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), neighbor_dz_.data(),
        position.data(), position.data() + num_atoms, position.data() + 2 * num_atoms,
        two_layer_gate_values_float_.data(), moment_vals_float_.data());
    if (launched_gate_main_basic_static) {
      GPU_CHECK_KERNEL
    } else if (alpha_basic_count_ <= 64) {
      gpu_sh_compute_basic_gate_main<float, 64><<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, two_layer_gate_values_float_.data(),
        moment_vals_float_.data());
    } else if (alpha_basic_count_ <= 128) {
      gpu_sh_compute_basic_gate_main<float, 128><<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, two_layer_gate_values_float_.data(),
        moment_vals_float_.data());
    } else {
      gpu_sh_compute_basic_gate_main<float><<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, two_layer_gate_values_float_.data(),
        moment_vals_float_.data());
    }
    GPU_CHECK_KERNEL
    profile_stop(sh_profile_basic, stage_start);

    stage_start = profile_start();
    if (use_compact_serial_product_) {
      if (use_product_basic_cache_) {
        if (use_product_pattern_rows_) {
          if (use_terminal_dot_groups_) {
            launch_sh_forward_energy_backward_compact_rows<float, float, kSHProductBasicCache, 1, 1>(
              product_grid_size, kProductBlockSize,
              num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
              potential.data(), !use_parallel_back_rows_);
          } else {
            launch_sh_forward_energy_backward_compact_rows<float, float, kSHProductBasicCache, 1, 0>(
              product_grid_size, kProductBlockSize,
              num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
              potential.data(), !use_parallel_back_rows_);
          }
        } else {
          if (use_terminal_dot_groups_) {
            launch_sh_forward_energy_backward_compact_rows<float, float, kSHProductBasicCache, 0, 1>(
              product_grid_size, kProductBlockSize,
              num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
              potential.data(), !use_parallel_back_rows_);
          } else {
            launch_sh_forward_energy_backward_compact_rows<float, float, kSHProductBasicCache, 0, 0>(
              product_grid_size, kProductBlockSize,
              num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
              potential.data(), !use_parallel_back_rows_);
          }
        }
      } else if (use_product_pattern_rows_) {
        if (use_terminal_dot_groups_) {
          launch_sh_forward_energy_backward_compact_rows<float, float, 0, 1, 1>(
            product_grid_size, kProductBlockSize,
            num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
            potential.data(), !use_parallel_back_rows_);
        } else {
          launch_sh_forward_energy_backward_compact_rows<float, float, 0, 1, 0>(
            product_grid_size, kProductBlockSize,
            num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
            potential.data(), !use_parallel_back_rows_);
        }
      } else {
        if (use_terminal_dot_groups_) {
          launch_sh_forward_energy_backward_compact_rows<float, float, 0, 0, 1>(
            product_grid_size, kProductBlockSize,
            num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
            potential.data(), !use_parallel_back_rows_);
        } else {
          launch_sh_forward_energy_backward_compact_rows<float, float, 0, 0, 0>(
            product_grid_size, kProductBlockSize,
            num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
            potential.data(), !use_parallel_back_rows_);
        }
      }
      GPU_CHECK_KERNEL
      if (use_fused_terminal_dot_) {
        if (use_product_basic_cache_) {
          launch_sh_fused_terminal_dot_groups_energy<float, float, kSHProductBasicCache>(
            product_grid_size, kProductBlockSize,
            num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
            potential.data());
        } else {
          launch_sh_fused_terminal_dot_groups_energy<float, float, 0>(
            product_grid_size, kProductBlockSize,
            num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
            potential.data());
        }
        GPU_CHECK_KERNEL
      }
      if (use_parallel_back_rows_) {
        for (int layer = sh_cg_layer_count_; layer >= 1; --layer) {
          gpu_sh_tensor_product_back_rows<float, float><<<tensor_grid_size, kBlockSize>>>(
            num_atoms, layer, model, moment_vals_float_.data(), moment_grads_float_.data());
          GPU_CHECK_KERNEL
        }
      }
    } else if (use_tensor_product_parallel_) {
      for (int layer = 1; layer <= sh_cg_layer_count_; ++layer) {
        gpu_sh_tensor_product_rows_forward<float><<<tensor_grid_size, kBlockSize>>>(
          num_atoms, layer, model, moment_vals_float_.data());
        GPU_CHECK_KERNEL
      }
      gpu_sh_energy_init_from_scalars<float, float><<<grid_size, kBlockSize>>>(
        num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
        potential.data());
      GPU_CHECK_KERNEL
      for (int layer = sh_cg_layer_count_; layer >= 1; --layer) {
        gpu_sh_tensor_product_back_rows<float, float><<<tensor_grid_size, kBlockSize>>>(
          num_atoms, layer, model, moment_vals_float_.data(), moment_grads_float_.data());
        GPU_CHECK_KERNEL
      }
    } else if (use_cg_block_forward_) {
      gpu_sh_forward_energy_backward_cg_blocks<float, float>
        <<<product_grid_size, kProductBlockSize>>>(
        num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
        potential.data());
      GPU_CHECK_KERNEL
    } else {
      gpu_sh_forward_energy_backward<float, float>
        <<<product_grid_size, kProductBlockSize>>>(
        num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
        potential.data());
      GPU_CHECK_KERNEL
    }
    profile_stop(sh_profile_product, stage_start);

	    stage_start = profile_start();
    const bool launched_gate_main_force_static =
      use_static_force_layout_ &&
      launch_sh_compute_forces_gate_main_static<float, float>(
        sh_l_max_, sh_k_max_, rb_size_, grid_size, num_atoms, box, rc * rc,
        use_cached_neighbor_displacements_, model, type.data(), neighbor_count_.data(),
        neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), neighbor_dz_.data(),
        position.data(), position.data() + num_atoms, position.data() + 2 * num_atoms,
        two_layer_gate_values_float_.data(), moment_grads_float_.data(),
        two_layer_gate_adjoints_float_.data(), force_tmp_.data(), force_self_tmp_ptr,
        virial_tmp_.data());
    if (launched_gate_main_force_static) {
      GPU_CHECK_KERNEL
    } else if (alpha_basic_count_ <= 48) {
      gpu_sh_compute_forces_gate_main_cached_grads<float, float, 48><<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, two_layer_gate_values_float_.data(),
        moment_grads_float_.data(), two_layer_gate_adjoints_float_.data(), force_tmp_.data(),
        force_self_tmp_ptr, virial_tmp_.data());
    } else if (alpha_basic_count_ <= kSHForceGradCache64) {
      gpu_sh_compute_forces_gate_main_cached_grads<float, float, kSHForceGradCache64>
        <<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, two_layer_gate_values_float_.data(),
        moment_grads_float_.data(), two_layer_gate_adjoints_float_.data(), force_tmp_.data(),
        force_self_tmp_ptr, virial_tmp_.data());
    } else if (alpha_basic_count_ <= kSHForceGradCache128) {
      gpu_sh_compute_forces_gate_main_cached_grads<float, float, kSHForceGradCache128>
        <<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, two_layer_gate_values_float_.data(),
        moment_grads_float_.data(), two_layer_gate_adjoints_float_.data(), force_tmp_.data(),
        force_self_tmp_ptr, virial_tmp_.data());
    } else {
      gpu_sh_compute_forces_gate_main_cached_grads<float, float, kSHForceGradCache256>
        <<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, two_layer_gate_values_float_.data(),
        moment_grads_float_.data(), two_layer_gate_adjoints_float_.data(), force_tmp_.data(),
        force_self_tmp_ptr, virial_tmp_.data());
	    }
	    GPU_CHECK_KERNEL
    const bool launched_gate_first_force_static =
      use_static_force_layout_ &&
      launch_sh_compute_forces_gate_first_layer_static<float, float>(
        sh_l_max_, sh_k_max_, rb_size_, grid_size, num_atoms, box, rc * rc,
        use_cached_neighbor_displacements_, model, type.data(), neighbor_count_.data(),
        neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), neighbor_dz_.data(),
        position.data(), position.data() + num_atoms, position.data() + 2 * num_atoms,
        two_layer_gate_basic_grads_float_.data(), two_layer_gate_adjoints_float_.data(),
        force_tmp_.data(), force_self_tmp_ptr, virial_tmp_.data());
    if (launched_gate_first_force_static) {
      GPU_CHECK_KERNEL
    } else if (alpha_basic_count_ <= 48) {
      gpu_sh_compute_forces_gate_first_layer_cached_grads<float, float, 48>
        <<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, two_layer_gate_basic_grads_float_.data(),
        two_layer_gate_adjoints_float_.data(), force_tmp_.data(), force_self_tmp_ptr,
        virial_tmp_.data());
    } else if (alpha_basic_count_ <= kSHForceGradCache64) {
      gpu_sh_compute_forces_gate_first_layer_cached_grads<float, float, kSHForceGradCache64>
        <<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, two_layer_gate_basic_grads_float_.data(),
        two_layer_gate_adjoints_float_.data(), force_tmp_.data(), force_self_tmp_ptr,
        virial_tmp_.data());
    } else if (alpha_basic_count_ <= kSHForceGradCache128) {
      gpu_sh_compute_forces_gate_first_layer_cached_grads<float, float, kSHForceGradCache128>
        <<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, two_layer_gate_basic_grads_float_.data(),
        two_layer_gate_adjoints_float_.data(), force_tmp_.data(), force_self_tmp_ptr,
        virial_tmp_.data());
    } else {
      gpu_sh_compute_forces_gate_first_layer_cached_grads<float, float, kSHForceGradCache256>
        <<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, two_layer_gate_basic_grads_float_.data(),
        two_layer_gate_adjoints_float_.data(), force_tmp_.data(), force_self_tmp_ptr,
        virial_tmp_.data());
    }
    GPU_CHECK_KERNEL
    profile_stop(sh_profile_force, stage_start);

    if (zbl_enabled_) {
      stage_start = profile_start();
      gpu_sh_apply_zbl<<<grid_size, kBlockSize>>>(
        num_atoms, box, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, potential.data(), force_tmp_.data(), virial_tmp_.data());
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
          profile_steps_, neighbor_ms, memset_ms, basic_ms, product_ms, force_ms,
          accumulate_ms, total_ms);
        profile_steps_ = 0;
        for (int i = 0; i < sh_profile_count; ++i) {
          profile_ms_[i] = 0.0;
        }
      }
    }
    return;
  }

  if (use_float_moments_) {
    stage_start = profile_start();
    const bool launched_static_basic =
      use_static_basic_layout_ &&
      launch_sh_compute_basic_static<float>(
        sh_l_max_, sh_k_max_, rb_size_, grid_size, num_atoms, box, rc * rc,
        use_cached_neighbor_displacements_, model, type.data(), neighbor_count_.data(),
        neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), neighbor_dz_.data(),
        position.data(), position.data() + num_atoms, position.data() + 2 * num_atoms,
        moment_vals_float_.data());
    if (launched_static_basic) {
      GPU_CHECK_KERNEL
    } else if (alpha_basic_count_ <= 64) {
      gpu_sh_compute_basic<float, 64><<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, moment_vals_float_.data());
    } else if (alpha_basic_count_ <= 128) {
      gpu_sh_compute_basic<float, 128><<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, moment_vals_float_.data());
    } else {
      gpu_sh_compute_basic<float><<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, moment_vals_float_.data());
    }
    GPU_CHECK_KERNEL
    profile_stop(sh_profile_basic, stage_start);
    stage_start = profile_start();
    const bool product_detail = profile_enabled_ && profile_product_detail_;
    const Clock::time_point product_forward_start = stage_start;
    if (use_tensor_product_parallel_) {
      for (int layer = 1; layer <= sh_cg_layer_count_; ++layer) {
        gpu_sh_tensor_product_rows_forward<float><<<tensor_grid_size, kBlockSize>>>(
          num_atoms, layer, model, moment_vals_float_.data());
        GPU_CHECK_KERNEL
      }
      gpu_sh_energy_init_from_scalars<float, float><<<grid_size, kBlockSize>>>(
        num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
        potential.data());
      GPU_CHECK_KERNEL
      Clock::time_point product_back_start = product_forward_start;
      if (product_detail) {
        profile_stop(sh_profile_product_forward, product_forward_start);
        product_back_start = profile_start();
      }
      for (int layer = sh_cg_layer_count_; layer >= 1; --layer) {
        gpu_sh_tensor_product_back_rows<float, float><<<tensor_grid_size, kBlockSize>>>(
          num_atoms, layer, model, moment_vals_float_.data(), moment_grads_float_.data());
        GPU_CHECK_KERNEL
      }
      if (product_detail) {
        profile_stop(sh_profile_product_back, product_back_start);
      }
    } else if (use_compact_serial_product_) {
      if (use_product_basic_cache_) {
        if (use_product_pattern_rows_) {
          if (use_terminal_dot_groups_) {
            launch_sh_forward_energy_backward_compact_rows<float, float, kSHProductBasicCache, 1, 1>(
              product_grid_size, kProductBlockSize,
              num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
              potential.data(), !use_parallel_back_rows_);
          } else {
            launch_sh_forward_energy_backward_compact_rows<float, float, kSHProductBasicCache, 1, 0>(
              product_grid_size, kProductBlockSize,
              num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
              potential.data(), !use_parallel_back_rows_);
          }
        } else {
          if (use_terminal_dot_groups_) {
            launch_sh_forward_energy_backward_compact_rows<float, float, kSHProductBasicCache, 0, 1>(
              product_grid_size, kProductBlockSize,
              num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
              potential.data(), !use_parallel_back_rows_);
          } else {
            launch_sh_forward_energy_backward_compact_rows<float, float, kSHProductBasicCache, 0, 0>(
              product_grid_size, kProductBlockSize,
              num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
              potential.data(), !use_parallel_back_rows_);
          }
        }
      } else {
        if (use_product_pattern_rows_) {
          if (use_terminal_dot_groups_) {
            launch_sh_forward_energy_backward_compact_rows<float, float, 0, 1, 1>(
              product_grid_size, kProductBlockSize,
              num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
              potential.data(), !use_parallel_back_rows_);
          } else {
            launch_sh_forward_energy_backward_compact_rows<float, float, 0, 1, 0>(
              product_grid_size, kProductBlockSize,
              num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
              potential.data(), !use_parallel_back_rows_);
          }
        } else {
          if (use_terminal_dot_groups_) {
            launch_sh_forward_energy_backward_compact_rows<float, float, 0, 0, 1>(
              product_grid_size, kProductBlockSize,
              num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
              potential.data(), !use_parallel_back_rows_);
          } else {
            launch_sh_forward_energy_backward_compact_rows<float, float, 0, 0, 0>(
              product_grid_size, kProductBlockSize,
              num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
              potential.data(), !use_parallel_back_rows_);
          }
        }
      }
      GPU_CHECK_KERNEL
      if (use_fused_terminal_dot_) {
        if (use_product_basic_cache_) {
          launch_sh_fused_terminal_dot_groups_energy<float, float, kSHProductBasicCache>(
            product_grid_size, kProductBlockSize,
            num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
            potential.data());
        } else {
          launch_sh_fused_terminal_dot_groups_energy<float, float, 0>(
            product_grid_size, kProductBlockSize,
            num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
            potential.data());
        }
        GPU_CHECK_KERNEL
      }
      if (product_detail) {
        profile_stop(sh_profile_product_forward, product_forward_start);
      }
      if (use_parallel_back_rows_) {
        const Clock::time_point product_back_start = product_detail ? profile_start() : stage_start;
        for (int layer = sh_cg_layer_count_; layer >= 1; --layer) {
          gpu_sh_tensor_product_back_rows<float, float><<<tensor_grid_size, kBlockSize>>>(
            num_atoms, layer, model, moment_vals_float_.data(), moment_grads_float_.data());
          GPU_CHECK_KERNEL
        }
        if (product_detail) {
          profile_stop(sh_profile_product_back, product_back_start);
        }
      }
    } else if (use_cg_block_forward_) {
      gpu_sh_forward_energy_backward_cg_blocks<float, float>
        <<<product_grid_size, kProductBlockSize>>>(
        num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
        potential.data());
      if (product_detail) {
        profile_stop(sh_profile_product_forward, product_forward_start);
      }
    } else {
      gpu_sh_forward_energy_backward<float, float>
        <<<product_grid_size, kProductBlockSize>>>(
        num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(),
        potential.data());
      if (product_detail) {
        profile_stop(sh_profile_product_forward, product_forward_start);
      }
    }
    GPU_CHECK_KERNEL
    profile_stop(sh_profile_product, stage_start);
    stage_start = profile_start();
    const bool launched_static_force =
      use_static_force_layout_ &&
      launch_sh_compute_forces_static<float, float>(
        sh_l_max_, sh_k_max_, rb_size_, grid_size, num_atoms, box, rc * rc,
        use_cached_neighbor_displacements_, model, type.data(), neighbor_count_.data(),
        neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), neighbor_dz_.data(),
        position.data(), position.data() + num_atoms, position.data() + 2 * num_atoms,
        moment_grads_float_.data(), force_tmp_.data(), force_self_tmp_ptr, virial_tmp_.data());
    if (launched_static_force) {
      GPU_CHECK_KERNEL
    } else if (use_force_grad_cache_) {
      if (alpha_basic_count_ <= 48) {
        gpu_sh_compute_forces_cached_grads<float, float, 48><<<grid_size, kBlockSize>>>(
          num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
          neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
          neighbor_dz_.data(), position.data(), position.data() + num_atoms,
          position.data() + 2 * num_atoms, moment_grads_float_.data(), force_tmp_.data(),
          force_self_tmp_ptr, virial_tmp_.data());
      } else if (alpha_basic_count_ <= kSHForceGradCache64) {
        gpu_sh_compute_forces_cached_grads<float, float, kSHForceGradCache64><<<grid_size, kBlockSize>>>(
          num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
          neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
          neighbor_dz_.data(), position.data(), position.data() + num_atoms,
          position.data() + 2 * num_atoms, moment_grads_float_.data(), force_tmp_.data(),
          force_self_tmp_ptr, virial_tmp_.data());
      } else if (alpha_basic_count_ <= kSHForceGradCache128) {
        gpu_sh_compute_forces_cached_grads<float, float, kSHForceGradCache128><<<grid_size, kBlockSize>>>(
          num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
          neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
          neighbor_dz_.data(), position.data(), position.data() + num_atoms,
          position.data() + 2 * num_atoms, moment_grads_float_.data(), force_tmp_.data(),
          force_self_tmp_ptr, virial_tmp_.data());
      } else {
        gpu_sh_compute_forces_cached_grads<float, float, kSHForceGradCache256><<<grid_size, kBlockSize>>>(
          num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
          neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
          neighbor_dz_.data(), position.data(), position.data() + num_atoms,
          position.data() + 2 * num_atoms, moment_grads_float_.data(), force_tmp_.data(),
          force_self_tmp_ptr, virial_tmp_.data());
      }
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
    const bool launched_static_basic =
      use_static_basic_layout_ &&
      launch_sh_compute_basic_static<double>(
        sh_l_max_, sh_k_max_, rb_size_, grid_size, num_atoms, box, rc * rc,
        use_cached_neighbor_displacements_, model, type.data(), neighbor_count_.data(),
        neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), neighbor_dz_.data(),
        position.data(), position.data() + num_atoms, position.data() + 2 * num_atoms,
        moment_vals_.data());
    if (launched_static_basic) {
      GPU_CHECK_KERNEL
    } else if (alpha_basic_count_ <= 64) {
      gpu_sh_compute_basic<double, 64><<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, moment_vals_.data());
    } else if (alpha_basic_count_ <= 128) {
      gpu_sh_compute_basic<double, 128><<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, moment_vals_.data());
    } else {
      gpu_sh_compute_basic<double><<<grid_size, kBlockSize>>>(
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
        neighbor_dz_.data(), position.data(), position.data() + num_atoms,
        position.data() + 2 * num_atoms, moment_vals_.data());
    }
    GPU_CHECK_KERNEL
    profile_stop(sh_profile_basic, stage_start);
    stage_start = profile_start();
    const bool product_detail = profile_enabled_ && profile_product_detail_;
    const Clock::time_point product_forward_start = stage_start;
    if (use_tensor_product_parallel_) {
      for (int layer = 1; layer <= sh_cg_layer_count_; ++layer) {
        gpu_sh_tensor_product_rows_forward<double><<<tensor_grid_size, kBlockSize>>>(
          num_atoms, layer, model, moment_vals_.data());
        GPU_CHECK_KERNEL
      }
      gpu_sh_energy_init_from_scalars<double, double><<<grid_size, kBlockSize>>>(
        num_atoms, model, type.data(), moment_vals_.data(), moment_grads_.data(),
        potential.data());
      GPU_CHECK_KERNEL
      Clock::time_point product_back_start = product_forward_start;
      if (product_detail) {
        profile_stop(sh_profile_product_forward, product_forward_start);
        product_back_start = profile_start();
      }
      for (int layer = sh_cg_layer_count_; layer >= 1; --layer) {
        gpu_sh_tensor_product_back_rows<double, double><<<tensor_grid_size, kBlockSize>>>(
          num_atoms, layer, model, moment_vals_.data(), moment_grads_.data());
        GPU_CHECK_KERNEL
      }
      if (product_detail) {
        profile_stop(sh_profile_product_back, product_back_start);
      }
    } else if (use_compact_serial_product_) {
      launch_sh_forward_energy_backward_compact_rows<double, double, 0, 0, 0>(
        product_grid_size, kProductBlockSize,
        num_atoms, model, type.data(), moment_vals_.data(), moment_grads_.data(),
        potential.data(), !use_parallel_back_rows_);
      GPU_CHECK_KERNEL
      if (product_detail) {
        profile_stop(sh_profile_product_forward, product_forward_start);
      }
      if (use_parallel_back_rows_) {
        const Clock::time_point product_back_start = product_detail ? profile_start() : stage_start;
        for (int layer = sh_cg_layer_count_; layer >= 1; --layer) {
          gpu_sh_tensor_product_back_rows<double, double><<<tensor_grid_size, kBlockSize>>>(
            num_atoms, layer, model, moment_vals_.data(), moment_grads_.data());
          GPU_CHECK_KERNEL
        }
        if (product_detail) {
          profile_stop(sh_profile_product_back, product_back_start);
        }
      }
    } else if (use_cg_block_forward_) {
      gpu_sh_forward_energy_backward_cg_blocks<double, double>
        <<<product_grid_size, kProductBlockSize>>>(
        num_atoms, model, type.data(), moment_vals_.data(), moment_grads_.data(),
        potential.data());
      if (product_detail) {
        profile_stop(sh_profile_product_forward, product_forward_start);
      }
    } else {
      gpu_sh_forward_energy_backward<double, double>
        <<<product_grid_size, kProductBlockSize>>>(
        num_atoms, model, type.data(), moment_vals_.data(), moment_grads_.data(),
        potential.data());
      if (product_detail) {
        profile_stop(sh_profile_product_forward, product_forward_start);
      }
    }
    GPU_CHECK_KERNEL
    profile_stop(sh_profile_product, stage_start);
    stage_start = profile_start();
    const bool launched_static_force =
      use_static_force_layout_ &&
      launch_sh_compute_forces_static<double, double>(
        sh_l_max_, sh_k_max_, rb_size_, grid_size, num_atoms, box, rc * rc,
        use_cached_neighbor_displacements_, model, type.data(), neighbor_count_.data(),
        neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), neighbor_dz_.data(),
        position.data(), position.data() + num_atoms, position.data() + 2 * num_atoms,
        moment_grads_.data(), force_tmp_.data(), force_self_tmp_ptr, virial_tmp_.data());
    if (launched_static_force) {
      GPU_CHECK_KERNEL
    } else if (use_force_grad_cache_) {
      if (alpha_basic_count_ <= 48) {
        gpu_sh_compute_forces_cached_grads<double, double, 48><<<grid_size, kBlockSize>>>(
          num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
          neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
          neighbor_dz_.data(), position.data(), position.data() + num_atoms,
          position.data() + 2 * num_atoms, moment_grads_.data(), force_tmp_.data(),
          force_self_tmp_ptr, virial_tmp_.data());
      } else if (alpha_basic_count_ <= kSHForceGradCache64) {
        gpu_sh_compute_forces_cached_grads<double, double, kSHForceGradCache64><<<grid_size, kBlockSize>>>(
          num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
          neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
          neighbor_dz_.data(), position.data(), position.data() + num_atoms,
          position.data() + 2 * num_atoms, moment_grads_.data(), force_tmp_.data(),
          force_self_tmp_ptr, virial_tmp_.data());
      } else if (alpha_basic_count_ <= kSHForceGradCache128) {
        gpu_sh_compute_forces_cached_grads<double, double, kSHForceGradCache128><<<grid_size, kBlockSize>>>(
          num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
          neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
          neighbor_dz_.data(), position.data(), position.data() + num_atoms,
          position.data() + 2 * num_atoms, moment_grads_.data(), force_tmp_.data(),
          force_self_tmp_ptr, virial_tmp_.data());
      } else {
        gpu_sh_compute_forces_cached_grads<double, double, kSHForceGradCache256><<<grid_size, kBlockSize>>>(
          num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(),
          neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
          neighbor_dz_.data(), position.data(), position.data() + num_atoms,
          position.data() + 2 * num_atoms, moment_grads_.data(), force_tmp_.data(),
          force_self_tmp_ptr, virial_tmp_.data());
      }
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

  if (zbl_enabled_) {
    stage_start = profile_start();
    gpu_sh_apply_zbl<<<grid_size, kBlockSize>>>(
      num_atoms, box, use_cached_neighbor_displacements_, model, type.data(),
      neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(),
      neighbor_dz_.data(), position.data(), position.data() + num_atoms,
      position.data() + 2 * num_atoms, potential.data(), force_tmp_.data(), virial_tmp_.data());
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
      if (profile_product_detail_) {
        const double product_forward_ms = profile_ms_[sh_profile_product_forward] * inv;
        const double product_back_ms = profile_ms_[sh_profile_product_back] * inv;
        printf(
          "SUS2-SH product detail avg over %d steps: forward=%.3f ms, backward=%.3f ms.\n",
          profile_steps_,
          product_forward_ms,
          product_back_ms);
      }
      profile_steps_ = 0;
      for (int i = 0; i < sh_profile_count; ++i) {
        profile_ms_[i] = 0.0;
      }
    }
  }
}
