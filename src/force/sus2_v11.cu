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
constexpr int kSus2MaxDirectRbSize = 16;
constexpr int kSus2MaxConstAlphaTimes = 6000;
constexpr int kSus2MaxConstScalarMoments = 2048;
constexpr int kSus2MaxConstSpecies = 128;
constexpr int kSus2ProductGroupPairWords = 2;
constexpr int kSus2LocalGraphMaxMoments = 640;
constexpr int kSus2MaxTensorRank = 4;
constexpr int kSus2MaxTensorGroups = 4;
constexpr int kSus2MaxTensorBasic = 140;
constexpr int kSus2L3K3TensorScalarBasic = 60;
constexpr int kSus2TensorScalarMaxDegree = 5;
constexpr int kSus2TensorScalarPackedInts = kSus2TensorScalarMaxDegree + 1;
constexpr int kSus2TensorScalarBlockSize = 128;
constexpr int kSus2TensorBlockOpInts = 6;
constexpr int kSus2TensorBlockOpRowInts = 2;
constexpr int kSus2TensorBlockRowWords = 2;
constexpr int kSus2TensorBlockRowMaxU16 = 65535;
constexpr int kSus2TensorBlockGeneric = 0;
constexpr int kSus2TensorBlockScalarScalar = 1;
constexpr int kSus2TensorBlockScalarTensor = 2;
constexpr int kSus2TensorBlockTensorScalar = 3;
constexpr int kSus2TensorBlockDot11 = 4;
constexpr int kSus2TensorBlockDot22 = 5;
constexpr int kSus2TensorBlockDot33 = 6;
constexpr int kSus2TensorBlockVecSym2ToVec = 7;
constexpr int kSus2TensorBlockSym2VecToVec = 8;
constexpr int kSus2TensorBlockVecSym3ToSym2 = 9;
constexpr int kSus2TensorBlockSym3VecToSym2 = 10;
constexpr int kSus2TensorBlockSym2Sym3ToVec = 11;
constexpr int kSus2TensorBlockSym3Sym2ToVec = 12;
constexpr int kSus2TensorBlockSym2Sym2ToSym2 = 13;
constexpr int kSus2TensorBlockSym2Sym2ToMatAB = 14;
constexpr int kSus2TensorBlockSym2Sym2ToMatBA = 15;
constexpr int kSus2TensorBlockVecVecOuterAB = 16;
constexpr int kSus2TensorBlockVecVecOuterBA = 17;
constexpr int kSus2TensorBlockVecVecToSym2 = 18;
constexpr int kSus2TensorBlockSym2MatScalar = 19;
constexpr int kSus2TensorBlockMatSym2Scalar = 20;
constexpr int kSus2TensorBlockMatMatSameScalar = 21;
constexpr int kSus2TensorBlockMatMatTransScalar = 22;
constexpr int kSus2TensorBlockStructured = 23;
constexpr int kSus2TensorBlockMaxKind = kSus2TensorBlockStructured;
constexpr int kSus2TensorBlockMaxGroups = 4;
constexpr int kSus2TensorBlockMaxLabels = 8;
constexpr int kSus2TensorBlockMetaGroupACount = 0;
constexpr int kSus2TensorBlockMetaGroupsA = 1;
constexpr int kSus2TensorBlockMetaGroupBCount = 5;
constexpr int kSus2TensorBlockMetaGroupsB = 6;
constexpr int kSus2TensorBlockMetaOutGroupCount = 10;
constexpr int kSus2TensorBlockMetaOutGroups = 11;
constexpr int kSus2TensorBlockMetaMatrix = 15;
constexpr int kSus2TensorBlockMetaLabelCount = 31;
constexpr int kSus2TensorBlockMetaLabelGroups = 32;
constexpr int kSus2TensorBlockMetaLabels = 40;
constexpr int kSus2TensorBlockMetaInts = 64;
constexpr double kLaguerreMinRho = 1.0e-8;
constexpr double kLaguerrePositiveParamFloor = 1.0e-6;

template <int L>
struct Sus2TensorStaticLayout {
  static constexpr int basic_per_group = (L + 1) * (L + 2) * (L + 3) / 6;
};

__constant__ unsigned short c_sus2_alpha_times_u16[kSus2MaxConstAlphaTimes * 4];
__constant__ unsigned short c_sus2_alpha_moment_mapping_u16[kSus2MaxConstScalarMoments];
__constant__ float c_sus2_shift_coeffs_float[kSus2MaxConstSpecies];
__constant__ float c_sus2_species_coeffs_float[kSus2MaxConstSpecies];
__constant__ float c_sus2_moment_coeffs_float[kSus2MaxConstScalarMoments];
__constant__ float c_sus2_jacobi_coeff_const
  [(kJacobiMaxIndexedBlock + 1) * (kSus2MaxDirectRbSize + 1)] = {
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.111111111f, 0.05f, 0.0285714286f, 0.0185185185f, 0.012987013f, 0.00961538462f, 0.00740740741f, 0.00588235294f, 0.004784689f, 0.00396825397f, 0.00334448161f, 0.00285714286f, 0.0024691358f, 0.00215517241f, 0.00189753321f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.3125f, 0.155555556f, 0.09375f, 0.0628571429f, 0.0451388889f, 0.0340136054f, 0.0265625f, 0.0213243547f, 0.0175f, 0.0146217419f, 0.0124007937f, 0.0106508876f, 0.00924744898f, 0.00810457516f, 0.00716145833f,
  0.0f, 0.0f, 0.18f, 0.0952380952f, 0.0595238095f, 0.0409090909f, 0.0299145299f, 0.0228571429f, 0.0180481283f, 0.014619883f, 0.0120879121f, 0.0101637493f, 0.00866666667f, 0.00747863248f, 0.00651984932f, 0.00573476703f, 0.00508373206f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
};
__constant__ float c_sus2_jacobi_coeff_x
  [(kJacobiMaxIndexedBlock + 1) * (kSus2MaxDirectRbSize + 1)] = {
  0.0f, 0.0f, 1.5f, 1.66666667f, 1.75f, 1.8f, 1.83333333f, 1.85714286f, 1.875f, 1.88888889f, 1.9f, 1.90909091f, 1.91666667f, 1.92307692f, 1.92857143f, 1.93333333f, 1.9375f,
  0.0f, 0.0f, 1.66666667f, 1.75f, 1.8f, 1.83333333f, 1.85714286f, 1.875f, 1.88888889f, 1.9f, 1.90909091f, 1.91666667f, 1.92307692f, 1.92857143f, 1.93333333f, 1.9375f, 1.94117647f,
  0.0f, 0.0f, 1.875f, 1.86666667f, 1.875f, 1.88571429f, 1.89583333f, 1.9047619f, 1.9125f, 1.91919192f, 1.925f, 1.93006993f, 1.93452381f, 1.93846154f, 1.94196429f, 1.94509804f, 1.94791667f,
  0.0f, 0.0f, 1.875f, 1.86666667f, 1.875f, 1.88571429f, 1.89583333f, 1.9047619f, 1.9125f, 1.91919192f, 1.925f, 1.93006993f, 1.93452381f, 1.93846154f, 1.94196429f, 1.94509804f, 1.94791667f,
  0.0f, 0.0f, 2.1f, 2.0f, 1.96428571f, 1.95f, 1.94444444f, 1.94285714f, 1.94318182f, 1.94444444f, 1.94615385f, 1.94805195f, 1.95f, 1.95192308f, 1.95378151f, 1.95555556f, 1.95723684f,
  0.0f, 0.0f, 2.33333333f, 2.14285714f, 2.0625f, 2.02222222f, 2.0f, 1.98701299f, 1.97916667f, 1.97435897f, 1.97142857f, 1.96969697f, 1.96875f, 1.96832579f, 1.96825397f, 1.96842105f, 1.96875f,
};
__constant__ float c_sus2_jacobi_prev_coeff
  [(kJacobiMaxIndexedBlock + 1) * (kSus2MaxDirectRbSize + 1)] = {
  0.0f, 0.0f, 0.5f, 0.666666667f, 0.75f, 0.8f, 0.833333333f, 0.857142857f, 0.875f, 0.888888889f, 0.9f, 0.909090909f, 0.916666667f, 0.923076923f, 0.928571429f, 0.933333333f, 0.9375f,
  0.0f, 0.0f, 0.555555556f, 0.7f, 0.771428571f, 0.814814815f, 0.844155844f, 0.865384615f, 0.881481481f, 0.894117647f, 0.90430622f, 0.912698413f, 0.919732441f, 0.925714286f, 0.930864198f, 0.935344828f, 0.939278937f,
  0.0f, 0.0f, 0.75f, 0.8f, 0.833333333f, 0.857142857f, 0.875f, 0.888888889f, 0.9f, 0.909090909f, 0.916666667f, 0.923076923f, 0.928571429f, 0.933333333f, 0.9375f, 0.941176471f, 0.944444444f,
  0.0f, 0.0f, 0.5625f, 0.711111111f, 0.78125f, 0.822857143f, 0.850694444f, 0.870748299f, 0.8859375f, 0.897867565f, 0.9075f, 0.915448188f, 0.922123016f, 0.927810651f, 0.932716837f, 0.936993464f, 0.940755208f,
  0.0f, 0.0f, 0.84f, 0.857142857f, 0.873015873f, 0.886363636f, 0.897435897f, 0.906666667f, 0.914438503f, 0.921052632f, 0.926739927f, 0.931677019f, 0.936f, 0.939814815f, 0.943204868f, 0.946236559f, 0.948963317f,
  0.0f, 0.0f, 1.0f, 0.952380952f, 0.9375f, 0.933333333f, 0.933333333f, 0.935064935f, 0.9375f, 0.94017094f, 0.942857143f, 0.945454545f, 0.947916667f, 0.950226244f, 0.952380952f, 0.954385965f, 0.95625f,
};

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
  int angular_channels;
  int radial_basis_kind;
  int radial_funcs_count;
  int alpha_basic_count;
  int alpha_times_count;
  int alpha_time_group_count;
  int alpha_moments_count;
  int alpha_scalar_moments;
  int max_rank;
  int rb_size;
  int lut_size;
  double max_dist;
  double lut_inv_dr;
  const double* shift_coeffs;
  const double* species_coeffs;
  const double* moment_coeffs;
  const float* shift_coeffs_float;
  const float* species_coeffs_float;
  const float* moment_coeffs_float;
  const int* alpha_basic;
  const int* alpha_times;
  const int* alpha_time_groups;
  const unsigned int* alpha_time_group_pairs;
  const int* alpha_moment_mapping;
  int l3k3_tensor_scalar_term_count;
  const int* l3k3_tensor_scalar_terms;
  const double* l3k3_tensor_scalar_coeffs;
  const float* l3k3_tensor_scalar_coeffs_float;
  int l3k3_tensor_block_op_count;
  const int* l3k3_tensor_block_ops;
  const int* l3k3_tensor_block_op_rows;
  const int* l3k3_tensor_block_metas;
  int l3k3_tensor_block_row_count;
  const unsigned int* l3k3_tensor_block_rows;
  const float* lut_vals;
  const float* lut_ders;
  const float* radial_direct_coeffs;
  const float* radial_direct_scal_s;
  bool use_tensor_basic_fastpath;
  int tensor_l;
  int tensor_k;
  int tensor_basic_per_group;
  bool use_const_alpha_times;
  bool use_const_scalar_moments;
  bool use_const_float_coeffs;
  bool use_float_model_params;
  bool use_radial_direct;
  bool use_l3k3_tensor_scalar;
  bool use_l3k3_tensor_block;
  bool use_l3k3_tensor_block_fast_forward;
  bool use_l3k3_tensor_block_fast_backward;
};

struct TensorBasicLayout {
  bool enabled = false;
  int l = 0;
  int k = 0;
  int basic_per_group = 0;
};

struct L3K3TensorScalarPlan {
  bool enabled = false;
  int max_degree = 0;
  std::vector<int> terms;
  std::vector<double> coeffs;
};

struct L3K3TensorBlockPlan {
  bool enabled = false;
  int op_count = 0;
  int component_group_count = 0;
  int fast_op_count = 0;
  int generic_op_count = 0;
  int candidate_count = 0;
  int matched_candidate_count = 0;
  int generic_row_count = 0;
  int row_count = 0;
  int selected_cost_units = 0;
  std::array<int, kSus2TensorBlockMaxKind + 1> op_kind_counts{};
  std::vector<int> ops;
  std::vector<int> op_rows;
  std::vector<int> metas;
  std::vector<unsigned int> rows;
};

struct TensorAutoDecision {
  bool use_tensor_block = false;
  std::string reason;
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

int tensor_basic_count_per_group(int l)
{
  return (l + 1) * (l + 2) * (l + 3) / 6;
}

int tensor_symmetric_component_count(int rank)
{
  return (rank + 1) * (rank + 2) / 2;
}

TensorBasicLayout detect_tensor_alpha_basic_layout(const SUS2HostModel& model)
{
  TensorBasicLayout layout;
  const int l = model.max_rank;
  if (l < 0 || l > kSus2MaxTensorRank || model.radial_funcs_count % (l + 1) != 0) {
    return layout;
  }

  const int k = model.radial_funcs_count / (l + 1);
  const int basic_per_group = tensor_basic_count_per_group(l);
  if (k <= 0 || k > kSus2MaxTensorGroups ||
      model.alpha_basic_count != k * basic_per_group) {
    return layout;
  }

  int basic = 0;
  for (int group = 0; group < k; ++group) {
    for (int rank = 0; rank <= l; ++rank) {
      const int mu = group * (l + 1) + rank;
      for (int a = rank; a >= 0; --a) {
        for (int b = rank - a; b >= 0; --b) {
          const int c = rank - a - b;
          const int offset = basic * 4;
          if (model.alpha_basic[offset + 0] != mu || model.alpha_basic[offset + 1] != a ||
              model.alpha_basic[offset + 2] != b || model.alpha_basic[offset + 3] != c) {
            return layout;
          }
          ++basic;
        }
      }
    }
  }
  if (basic == model.alpha_basic_count) {
    layout.enabled = true;
    layout.l = l;
    layout.k = k;
    layout.basic_per_group = basic_per_group;
  }
  return layout;
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

bool can_pack_scalar_moments_u16(const SUS2HostModel& model)
{
  if (model.alpha_scalar_moments > kSus2MaxConstScalarMoments ||
      model.alpha_moments_count > 65535) {
    return false;
  }
  for (int value : model.alpha_moment_mapping) {
    if (value < 0 || value > 65535) {
      return false;
    }
  }
  return true;
}

bool can_pack_alpha_time_groups_u16(const std::vector<int>& groups)
{
  if (groups.size() % 3 != 0) {
    return false;
  }
  for (int value : groups) {
    if (value < 0 || value > 65535) {
      return false;
    }
  }
  return true;
}

std::vector<unsigned int> pack_alpha_time_group_pairs(const std::vector<int>& groups)
{
  std::vector<unsigned int> pairs;
  pairs.reserve((groups.size() / 3) * kSus2ProductGroupPairWords);
  for (size_t group = 0; group < groups.size() / 3; ++group) {
    const unsigned int begin = static_cast<unsigned int>(groups[group * 3 + 0]);
    const unsigned int len = static_cast<unsigned int>(groups[group * 3 + 1]);
    const unsigned int dst = static_cast<unsigned int>(groups[group * 3 + 2]);
    pairs.push_back(begin | (len << 16));
    pairs.push_back(dst);
  }
  return pairs;
}

bool supports_product_assign(const SUS2HostModel& model)
{
  std::vector<unsigned char> dst_seen(model.alpha_moments_count, 0);
  int t = 0;
  while (t < model.alpha_times_count) {
    const int begin = t;
    const int dst = model.alpha_times[t * 4 + 3];
    if (dst < model.alpha_basic_count || dst >= model.alpha_moments_count || dst_seen[dst]) {
      return false;
    }
    dst_seen[dst] = 1;
    do {
      const int src0 = model.alpha_times[t * 4 + 0];
      const int src1 = model.alpha_times[t * 4 + 1];
      const int current_dst = model.alpha_times[t * 4 + 3];
      if (current_dst != dst || src0 == dst || src1 == dst) {
        return false;
      }
      ++t;
    } while (t < model.alpha_times_count && model.alpha_times[t * 4 + 3] == dst);
    if (t == begin) {
      return false;
    }
  }

  for (int moment = model.alpha_basic_count; moment < model.alpha_moments_count; ++moment) {
    if (!dst_seen[moment]) {
      return false;
    }
  }
  return true;
}

std::array<int, kSus2TensorScalarPackedInts> make_l3k3_tensor_scalar_key(
  const std::array<int, kSus2TensorScalarMaxDegree>& ids,
  int degree)
{
  std::array<int, kSus2TensorScalarPackedInts> key;
  key.fill(-1);
  key[0] = degree;
  for (int i = 0; i < degree; ++i) {
    key[i + 1] = ids[i];
  }
  return key;
}

L3K3TensorScalarPlan build_l3k3_tensor_scalar_plan(const SUS2HostModel& model)
{
  L3K3TensorScalarPlan plan;
  const TensorBasicLayout layout = detect_tensor_alpha_basic_layout(model);
  if (!layout.enabled || layout.l != 3 || layout.k != 3 ||
      model.alpha_basic_count != kSus2L3K3TensorScalarBasic) {
    return plan;
  }

  using MonoKey = std::array<int, kSus2TensorScalarPackedInts>;
  using Polynomial = std::map<MonoKey, double>;
  std::vector<Polynomial> polys(model.alpha_moments_count);
  std::vector<unsigned char> built(model.alpha_moments_count, 0);

  for (int basic = 0; basic < model.alpha_basic_count; ++basic) {
    std::array<int, kSus2TensorScalarMaxDegree> ids;
    ids.fill(-1);
    ids[0] = basic;
    polys[basic][make_l3k3_tensor_scalar_key(ids, 1)] = 1.0;
    built[basic] = 1;
  }

  int max_degree = 1;
  for (int t = 0; t < model.alpha_times_count; ++t) {
    const int src0 = model.alpha_times[t * 4 + 0];
    const int src1 = model.alpha_times[t * 4 + 1];
    const int mult = model.alpha_times[t * 4 + 2];
    const int dst = model.alpha_times[t * 4 + 3];
    if (src0 < 0 || src0 >= model.alpha_moments_count || src1 < 0 ||
        src1 >= model.alpha_moments_count || dst < 0 || dst >= model.alpha_moments_count ||
        !built[src0] || !built[src1]) {
      return L3K3TensorScalarPlan{};
    }

    Polynomial product;
    for (const auto& lhs : polys[src0]) {
      const int lhs_degree = lhs.first[0];
      for (const auto& rhs : polys[src1]) {
        const int rhs_degree = rhs.first[0];
        const int degree = lhs_degree + rhs_degree;
        if (degree > kSus2TensorScalarMaxDegree) {
          return L3K3TensorScalarPlan{};
        }
        std::array<int, kSus2TensorScalarMaxDegree> ids;
        ids.fill(-1);
        int cursor = 0;
        for (int k = 0; k < lhs_degree; ++k) {
          ids[cursor++] = lhs.first[k + 1];
        }
        for (int k = 0; k < rhs_degree; ++k) {
          ids[cursor++] = rhs.first[k + 1];
        }
        std::sort(ids.begin(), ids.begin() + degree);
        const MonoKey key = make_l3k3_tensor_scalar_key(ids, degree);
        product[key] += static_cast<double>(mult) * lhs.second * rhs.second;
        max_degree = std::max(max_degree, degree);
      }
    }
    for (const auto& term : product) {
      polys[dst][term.first] += term.second;
      if (polys[dst][term.first] == 0.0) {
        polys[dst].erase(term.first);
      }
    }
    built[dst] = 1;
  }

  Polynomial merged;
  for (int idx = 0; idx < model.alpha_scalar_moments; ++idx) {
    const int moment_id = model.alpha_moment_mapping[idx];
    if (moment_id < 0 || moment_id >= model.alpha_moments_count || !built[moment_id]) {
      return L3K3TensorScalarPlan{};
    }
    const double coeff = model.moment_coeffs[idx];
    if (coeff == 0.0) {
      continue;
    }
    for (const auto& term : polys[moment_id]) {
      merged[term.first] += coeff * term.second;
      if (merged[term.first] == 0.0) {
        merged.erase(term.first);
      }
    }
  }

  plan.enabled = true;
  plan.max_degree = max_degree;
  plan.terms.reserve(merged.size() * kSus2TensorScalarPackedInts);
  plan.coeffs.reserve(merged.size());
  for (const auto& term : merged) {
    const int degree = term.first[0];
    if (degree <= 0 || degree > kSus2TensorScalarMaxDegree) {
      return L3K3TensorScalarPlan{};
    }
    plan.terms.push_back(degree);
    for (int k = 1; k < kSus2TensorScalarPackedInts; ++k) {
      plan.terms.push_back(term.first[k]);
    }
    plan.coeffs.push_back(term.second);
  }
  return plan;
}

using L3K3Component = std::vector<std::vector<int>>;

struct L3K3TensorBlockHostBlock {
  int id = -1;
  int start = 0;
  std::vector<int> ids;
  std::vector<int> groups;
  std::vector<L3K3Component> components;

  int id_for(L3K3Component comp) const
  {
    for (auto& part : comp) {
      std::sort(part.begin(), part.end());
    }
    for (size_t i = 0; i < components.size(); ++i) {
      if (components[i] == comp) {
        return ids[i];
      }
    }
    return -1;
  }
};

struct L3K3TensorBlockLabel {
  int side = 0;
  int index = 0;
  int size = 0;
};

struct L3K3TensorBlockCandidate {
  int component_count = 0;
  std::vector<std::vector<int>> matrix;
  std::vector<int> out_groups;
  std::vector<std::vector<L3K3TensorBlockLabel>> grouped_labels;
};

std::vector<std::vector<int>> l3k3_sym_tuples(int rank)
{
  std::vector<std::vector<int>> out;
  if (rank == 0) {
    out.push_back(std::vector<int>());
    return out;
  }
  for (int a = rank; a >= 0; --a) {
    for (int b = rank - a; b >= 0; --b) {
      const int c = rank - a - b;
      std::vector<int> part;
      part.insert(part.end(), a, 0);
      part.insert(part.end(), b, 1);
      part.insert(part.end(), c, 2);
      out.push_back(part);
    }
  }
  return out;
}

void l3k3_components_rec(
  const std::vector<int>& groups,
  int index,
  L3K3Component& current,
  std::vector<L3K3Component>& out)
{
  if (index == static_cast<int>(groups.size())) {
    out.push_back(current);
    return;
  }
  const std::vector<std::vector<int>> parts = l3k3_sym_tuples(groups[index]);
  for (const auto& part : parts) {
    current.push_back(part);
    l3k3_components_rec(groups, index + 1, current, out);
    current.pop_back();
  }
}

std::vector<L3K3Component> l3k3_components(const std::vector<int>& groups)
{
  std::vector<L3K3Component> out;
  if (groups.empty()) {
    out.push_back(L3K3Component());
    return out;
  }
  L3K3Component current;
  l3k3_components_rec(groups, 0, current, out);
  return out;
}

L3K3TensorBlockHostBlock make_l3k3_tensor_block(
  int id,
  int start,
  int count,
  const std::vector<int>& groups)
{
  L3K3TensorBlockHostBlock block;
  block.id = id;
  block.start = start;
  block.groups = groups;
  block.ids.reserve(count);
  for (int k = 0; k < count; ++k) {
    block.ids.push_back(start + k);
  }
  block.components = l3k3_components(groups);
  if (static_cast<int>(block.components.size()) != count) {
    return L3K3TensorBlockHostBlock{};
  }
  return block;
}

void l3k3_coarsen_labels_rec(
  const std::vector<L3K3TensorBlockLabel>& labels,
  int pos,
  std::vector<int>& sizes,
  std::vector<std::vector<L3K3TensorBlockLabel>>& grouped,
  std::vector<L3K3TensorBlockCandidate>& out,
  const std::vector<std::vector<int>>& matrix)
{
  if (pos == static_cast<int>(labels.size())) {
    L3K3TensorBlockCandidate candidate;
    candidate.matrix = matrix;
    candidate.out_groups = sizes;
    candidate.grouped_labels = grouped;
    candidate.component_count = static_cast<int>(l3k3_components(sizes).size());
    out.push_back(candidate);
    return;
  }
  int total = 0;
  std::vector<L3K3TensorBlockLabel> label_group;
  for (int end = pos; end < static_cast<int>(labels.size()); ++end) {
    total += labels[end].size;
    label_group.push_back(labels[end]);
    sizes.push_back(total);
    grouped.push_back(label_group);
    l3k3_coarsen_labels_rec(labels, end + 1, sizes, grouped, out, matrix);
    sizes.pop_back();
    grouped.pop_back();
  }
}

void l3k3_add_candidate_coarsenings(
  const std::vector<L3K3TensorBlockLabel>& labels,
  const std::vector<std::vector<int>>& matrix,
  std::vector<L3K3TensorBlockCandidate>& out)
{
  if (labels.empty()) {
    L3K3TensorBlockCandidate candidate;
    candidate.matrix = matrix;
    candidate.component_count = 1;
    out.push_back(candidate);
    return;
  }
  std::vector<int> order(labels.size());
  for (size_t i = 0; i < order.size(); ++i) {
    order[i] = static_cast<int>(i);
  }
  do {
    std::vector<L3K3TensorBlockLabel> permuted;
    permuted.reserve(labels.size());
    for (int id : order) {
      permuted.push_back(labels[id]);
    }
    std::vector<int> sizes;
    std::vector<std::vector<L3K3TensorBlockLabel>> grouped;
    l3k3_coarsen_labels_rec(permuted, 0, sizes, grouped, out, matrix);
  } while (std::next_permutation(order.begin(), order.end()));
}

void l3k3_contraction_matrices_rec(
  const std::vector<int>& groups_a,
  const std::vector<int>& groups_b,
  int cell,
  std::vector<int>& rem_a,
  std::vector<int>& rem_b,
  std::vector<std::vector<int>>& matrix,
  std::vector<std::vector<std::vector<int>>>& out)
{
  const int rows = static_cast<int>(groups_a.size());
  const int cols = static_cast<int>(groups_b.size());
  if (cell == rows * cols) {
    out.push_back(matrix);
    return;
  }
  const int i = cell / cols;
  const int j = cell % cols;
  const int max_value = std::min(rem_a[i], rem_b[j]);
  for (int value = 0; value <= max_value; ++value) {
    matrix[i][j] = value;
    rem_a[i] -= value;
    rem_b[j] -= value;
    l3k3_contraction_matrices_rec(groups_a, groups_b, cell + 1, rem_a, rem_b, matrix, out);
    rem_a[i] += value;
    rem_b[j] += value;
  }
  matrix[i][j] = 0;
}

std::vector<std::vector<std::vector<int>>> l3k3_contraction_matrices(
  const std::vector<int>& groups_a,
  const std::vector<int>& groups_b)
{
  if (groups_a.empty() || groups_b.empty()) {
    return std::vector<std::vector<std::vector<int>>>{std::vector<std::vector<int>>()};
  }
  std::vector<std::vector<std::vector<int>>> out;
  std::vector<int> rem_a = groups_a;
  std::vector<int> rem_b = groups_b;
  std::vector<std::vector<int>> matrix(groups_a.size(), std::vector<int>(groups_b.size(), 0));
  l3k3_contraction_matrices_rec(groups_a, groups_b, 0, rem_a, rem_b, matrix, out);
  return out;
}

int l3k3_matrix_value(const std::vector<std::vector<int>>& matrix, int i, int j)
{
  return matrix.empty() ? 0 : matrix[i][j];
}

bool l3k3_split_output_component(
  const L3K3Component& out_component,
  const std::vector<std::vector<L3K3TensorBlockLabel>>& grouped_labels,
  std::map<std::pair<int, int>, std::vector<int>>& assignments)
{
  assignments.clear();
  if (out_component.size() != grouped_labels.size()) {
    return false;
  }
  for (size_t g = 0; g < grouped_labels.size(); ++g) {
    int offset = 0;
    for (const auto& label : grouped_labels[g]) {
      if (offset + label.size > static_cast<int>(out_component[g].size())) {
        return false;
      }
      assignments[std::make_pair(label.side, label.index)] = std::vector<int>(
        out_component[g].begin() + offset, out_component[g].begin() + offset + label.size);
      offset += label.size;
    }
    if (offset != static_cast<int>(out_component[g].size())) {
      return false;
    }
  }
  return true;
}

void l3k3_generated_rows_rec(
  int pair_index,
  const std::vector<std::pair<int, int>>& pairs,
  const L3K3TensorBlockHostBlock& block_a,
  const L3K3TensorBlockHostBlock& block_b,
  std::vector<std::vector<int>>& comp_a,
  std::vector<std::vector<int>>& comp_b,
  std::map<std::pair<int, int>, int>& rows,
  bool& ok)
{
  if (!ok) {
    return;
  }
  if (pair_index == static_cast<int>(pairs.size())) {
    const int id_a = block_a.id_for(comp_a);
    const int id_b = block_b.id_for(comp_b);
    if (id_a < 0 || id_b < 0) {
      ok = false;
      return;
    }
    rows[std::make_pair(id_a, id_b)] += 1;
    return;
  }
  const int ai = pairs[pair_index].first;
  const int bi = pairs[pair_index].second;
  for (int value = 0; value < 3; ++value) {
    comp_a[ai].push_back(value);
    comp_b[bi].push_back(value);
    l3k3_generated_rows_rec(pair_index + 1, pairs, block_a, block_b, comp_a, comp_b, rows, ok);
    comp_b[bi].pop_back();
    comp_a[ai].pop_back();
  }
}

bool l3k3_generated_rows(
  const L3K3TensorBlockHostBlock& block_a,
  const L3K3TensorBlockHostBlock& block_b,
  const std::vector<std::vector<int>>& matrix,
  const std::vector<std::vector<L3K3TensorBlockLabel>>& grouped_labels,
  const L3K3Component& out_component,
  std::vector<std::array<int, 3>>& out_rows)
{
  std::map<std::pair<int, int>, std::vector<int>> assignments;
  if (!l3k3_split_output_component(out_component, grouped_labels, assignments)) {
    return false;
  }
  std::vector<std::vector<int>> comp_a(block_a.groups.size());
  std::vector<std::vector<int>> comp_b(block_b.groups.size());
  for (size_t i = 0; i < block_a.groups.size(); ++i) {
    const auto found = assignments.find(std::make_pair(0, static_cast<int>(i)));
    if (found != assignments.end()) {
      comp_a[i] = found->second;
    }
  }
  for (size_t j = 0; j < block_b.groups.size(); ++j) {
    const auto found = assignments.find(std::make_pair(1, static_cast<int>(j)));
    if (found != assignments.end()) {
      comp_b[j] = found->second;
    }
  }
  std::vector<std::pair<int, int>> pairs;
  for (size_t i = 0; i < block_a.groups.size(); ++i) {
    for (size_t j = 0; j < block_b.groups.size(); ++j) {
      const int count = l3k3_matrix_value(matrix, static_cast<int>(i), static_cast<int>(j));
      for (int k = 0; k < count; ++k) {
        pairs.push_back(std::make_pair(static_cast<int>(i), static_cast<int>(j)));
      }
    }
  }
  std::map<std::pair<int, int>, int> row_map;
  bool ok = true;
  l3k3_generated_rows_rec(0, pairs, block_a, block_b, comp_a, comp_b, row_map, ok);
  if (!ok) {
    return false;
  }
  out_rows.clear();
  for (const auto& row : row_map) {
    out_rows.push_back(std::array<int, 3>{{row.first.first, row.first.second, row.second}});
  }
  return true;
}

bool l3k3_candidate_matches(
  const SUS2HostModel& model,
  const std::vector<int>& group_begin,
  const std::vector<int>& group_len,
  int group_index,
  const L3K3TensorBlockHostBlock& block_a,
  const L3K3TensorBlockHostBlock& block_b,
  const L3K3TensorBlockCandidate& candidate)
{
  const std::vector<L3K3Component> out_components = l3k3_components(candidate.out_groups);
  if (group_index + static_cast<int>(out_components.size()) > static_cast<int>(group_begin.size())) {
    return false;
  }
  for (size_t component = 0; component < out_components.size(); ++component) {
    const int g = group_index + static_cast<int>(component);
    std::vector<std::array<int, 3>> expected;
    if (!l3k3_generated_rows(
          block_a,
          block_b,
          candidate.matrix,
          candidate.grouped_labels,
          out_components[component],
          expected)) {
      return false;
    }
    std::vector<std::array<int, 3>> actual;
    actual.reserve(group_len[g]);
    for (int k = 0; k < group_len[g]; ++k) {
      const int t = group_begin[g] + k;
      actual.push_back(std::array<int, 3>{{model.alpha_times[t * 4 + 0], model.alpha_times[t * 4 + 1], model.alpha_times[t * 4 + 2]}});
    }
    std::sort(actual.begin(), actual.end());
    if (actual != expected) {
      return false;
    }
  }
  return true;
}

unsigned int pack_l3k3_tensor_block_row_pair(int low, int high)
{
  return static_cast<unsigned int>(low) |
         (static_cast<unsigned int>(high) << 16);
}

bool append_l3k3_tensor_block_rows(
  const SUS2HostModel& model,
  const std::vector<int>& group_begin,
  const std::vector<int>& group_len,
  const std::vector<int>& group_dst,
  int group_index,
  int component_count,
  const L3K3TensorBlockHostBlock& block_a,
  const L3K3TensorBlockHostBlock& block_b,
  int dst_start,
  std::vector<unsigned int>& rows)
{
  for (int component = 0; component < component_count; ++component) {
    const int g = group_index + component;
    if (g < 0 || g >= static_cast<int>(group_begin.size())) {
      return false;
    }
    const int dst_rel = group_dst[g] - dst_start;
    if (dst_rel < 0 || dst_rel >= component_count) {
      return false;
    }
    for (int k = 0; k < group_len[g]; ++k) {
      const int t = group_begin[g] + k;
      const int src0 = model.alpha_times[t * 4 + 0];
      const int src1 = model.alpha_times[t * 4 + 1];
      const int mult = model.alpha_times[t * 4 + 2];
      const int dst = model.alpha_times[t * 4 + 3];
      const int src0_rel = src0 - block_a.start;
      const int src1_rel = src1 - block_b.start;
      if (dst != group_dst[g] || src0_rel < 0 ||
          src0_rel >= static_cast<int>(block_a.ids.size()) || src1_rel < 0 ||
          src1_rel >= static_cast<int>(block_b.ids.size())) {
        return false;
      }
      if (dst_rel > kSus2TensorBlockRowMaxU16 || src0_rel > kSus2TensorBlockRowMaxU16 ||
          src1_rel > kSus2TensorBlockRowMaxU16 || mult < 0 ||
          mult > kSus2TensorBlockRowMaxU16) {
        return false;
      }
      rows.push_back(pack_l3k3_tensor_block_row_pair(dst_rel, src0_rel));
      rows.push_back(pack_l3k3_tensor_block_row_pair(src1_rel, mult));
    }
  }
  return true;
}

int l3k3_int_pow(int base, int exp)
{
  int value = 1;
  for (int i = 0; i < exp; ++i) {
    value *= base;
  }
  return value;
}

int l3k3_tensor_block_contracted_rank(const L3K3TensorBlockCandidate& candidate)
{
  int contracted_rank = 0;
  for (const auto& row : candidate.matrix) {
    for (int value : row) {
      contracted_rank += value;
    }
  }
  return contracted_rank;
}

int l3k3_tensor_block_structured_work_count(const L3K3TensorBlockCandidate& candidate)
{
  return candidate.component_count *
         l3k3_int_pow(3, l3k3_tensor_block_contracted_rank(candidate));
}

bool l3k3_tensor_block_structured_meta_supported(
  const L3K3TensorBlockHostBlock& block_a,
  const L3K3TensorBlockHostBlock& block_b,
  const L3K3TensorBlockCandidate& candidate)
{
  if (block_a.groups.size() > kSus2TensorBlockMaxGroups ||
      block_b.groups.size() > kSus2TensorBlockMaxGroups ||
      candidate.out_groups.size() > kSus2TensorBlockMaxGroups ||
      candidate.grouped_labels.size() != candidate.out_groups.size() ||
      l3k3_tensor_block_contracted_rank(candidate) > kSus2MaxTensorRank) {
    return false;
  }
  int label_count = 0;
  for (const auto& group : candidate.grouped_labels) {
    label_count += static_cast<int>(group.size());
  }
  return label_count <= kSus2TensorBlockMaxLabels;
}

bool append_l3k3_tensor_block_meta(
  const L3K3TensorBlockHostBlock& block_a,
  const L3K3TensorBlockHostBlock& block_b,
  const L3K3TensorBlockCandidate& candidate,
  std::vector<int>& metas)
{
  if (!l3k3_tensor_block_structured_meta_supported(block_a, block_b, candidate)) {
    return false;
  }
  std::array<int, kSus2TensorBlockMetaInts> meta{};
  meta[kSus2TensorBlockMetaGroupACount] = static_cast<int>(block_a.groups.size());
  for (size_t i = 0; i < block_a.groups.size(); ++i) {
    meta[kSus2TensorBlockMetaGroupsA + static_cast<int>(i)] = block_a.groups[i];
  }
  meta[kSus2TensorBlockMetaGroupBCount] = static_cast<int>(block_b.groups.size());
  for (size_t i = 0; i < block_b.groups.size(); ++i) {
    meta[kSus2TensorBlockMetaGroupsB + static_cast<int>(i)] = block_b.groups[i];
  }
  meta[kSus2TensorBlockMetaOutGroupCount] = static_cast<int>(candidate.out_groups.size());
  for (size_t i = 0; i < candidate.out_groups.size(); ++i) {
    meta[kSus2TensorBlockMetaOutGroups + static_cast<int>(i)] = candidate.out_groups[i];
  }
  for (size_t i = 0; i < candidate.matrix.size(); ++i) {
    if (candidate.matrix[i].size() > kSus2TensorBlockMaxGroups) {
      return false;
    }
    for (size_t j = 0; j < candidate.matrix[i].size(); ++j) {
      meta[kSus2TensorBlockMetaMatrix + static_cast<int>(i) * kSus2TensorBlockMaxGroups +
           static_cast<int>(j)] = candidate.matrix[i][j];
    }
  }

  int label_cursor = 0;
  for (size_t group = 0; group < candidate.grouped_labels.size(); ++group) {
    meta[kSus2TensorBlockMetaLabelGroups + static_cast<int>(group) * 2 + 0] = label_cursor;
    meta[kSus2TensorBlockMetaLabelGroups + static_cast<int>(group) * 2 + 1] =
      static_cast<int>(candidate.grouped_labels[group].size());
    for (const auto& label : candidate.grouped_labels[group]) {
      if (label_cursor >= kSus2TensorBlockMaxLabels) {
        return false;
      }
      const int offset = kSus2TensorBlockMetaLabels + label_cursor * 3;
      meta[offset + 0] = label.side;
      meta[offset + 1] = label.index;
      meta[offset + 2] = label.size;
      ++label_cursor;
    }
  }
  meta[kSus2TensorBlockMetaLabelCount] = label_cursor;
  metas.insert(metas.end(), meta.begin(), meta.end());
  return true;
}

bool l3k3_matrix_1x1(const std::vector<std::vector<int>>& matrix, int value)
{
  return matrix.size() == 1 && matrix[0].size() == 1 && matrix[0][0] == value;
}

bool l3k3_matrix_1x2(const std::vector<std::vector<int>>& matrix, int value0, int value1)
{
  return matrix.size() == 1 && matrix[0].size() == 2 &&
         matrix[0][0] == value0 && matrix[0][1] == value1;
}

bool l3k3_matrix_2x1(const std::vector<std::vector<int>>& matrix, int value0, int value1)
{
  return matrix.size() == 2 && matrix[0].size() == 1 && matrix[1].size() == 1 &&
         matrix[0][0] == value0 && matrix[1][0] == value1;
}

bool l3k3_matrix_2x2(
  const std::vector<std::vector<int>>& matrix,
  int value00,
  int value01,
  int value10,
  int value11)
{
  return matrix.size() == 2 && matrix[0].size() == 2 && matrix[1].size() == 2 &&
         matrix[0][0] == value00 && matrix[0][1] == value01 &&
         matrix[1][0] == value10 && matrix[1][1] == value11;
}

bool l3k3_label_matches(
  const L3K3TensorBlockLabel& label,
  int side,
  int index,
  int size)
{
  return label.side == side && label.index == index && label.size == size;
}

bool l3k3_labels_single_group(
  const std::vector<std::vector<L3K3TensorBlockLabel>>& labels,
  int side,
  int index,
  int size)
{
  return labels.size() == 1 && labels[0].size() == 1 &&
         l3k3_label_matches(labels[0][0], side, index, size);
}

bool l3k3_labels_two_groups(
  const std::vector<std::vector<L3K3TensorBlockLabel>>& labels,
  int side0,
  int index0,
  int size0,
  int side1,
  int index1,
  int size1)
{
  return labels.size() == 2 && labels[0].size() == 1 && labels[1].size() == 1 &&
         l3k3_label_matches(labels[0][0], side0, index0, size0) &&
         l3k3_label_matches(labels[1][0], side1, index1, size1);
}

bool l3k3_labels_coarsened_two(
  const std::vector<std::vector<L3K3TensorBlockLabel>>& labels,
  int side0,
  int index0,
  int size0,
  int side1,
  int index1,
  int size1)
{
  return labels.size() == 1 && labels[0].size() == 2 &&
         l3k3_label_matches(labels[0][0], side0, index0, size0) &&
         l3k3_label_matches(labels[0][1], side1, index1, size1);
}

bool l3k3_labels_side_groups(
  const std::vector<std::vector<L3K3TensorBlockLabel>>& labels,
  int side,
  const std::vector<int>& groups)
{
  if (labels.size() != groups.size()) {
    return false;
  }
  for (size_t i = 0; i < groups.size(); ++i) {
    if (labels[i].size() != 1 ||
        !l3k3_label_matches(labels[i][0], side, static_cast<int>(i), groups[i])) {
      return false;
    }
  }
  return true;
}

bool l3k3_groups_equal(const std::vector<int>& groups, std::initializer_list<int> expected)
{
  return groups == std::vector<int>(expected);
}

int classify_l3k3_tensor_block_op(
  const L3K3TensorBlockHostBlock& block_a,
  const L3K3TensorBlockHostBlock& block_b,
  const L3K3TensorBlockCandidate& candidate)
{
  const std::vector<int>& groups_a = block_a.groups;
  const std::vector<int>& groups_b = block_b.groups;
  const std::vector<int>& out_groups = candidate.out_groups;
  const auto& matrix = candidate.matrix;
  const auto& labels = candidate.grouped_labels;

  if (groups_a.empty() && groups_b.empty() && matrix.empty() && out_groups.empty()) {
    return kSus2TensorBlockScalarScalar;
  }
  if (groups_a.empty() && matrix.empty() && out_groups == groups_b &&
      l3k3_labels_side_groups(labels, 1, groups_b)) {
    return kSus2TensorBlockScalarTensor;
  }
  if (groups_b.empty() && matrix.empty() && out_groups == groups_a &&
      l3k3_labels_side_groups(labels, 0, groups_a)) {
    return kSus2TensorBlockTensorScalar;
  }
  if (l3k3_groups_equal(groups_a, {1}) && l3k3_groups_equal(groups_b, {1}) &&
      l3k3_matrix_1x1(matrix, 1) && out_groups.empty()) {
    return kSus2TensorBlockDot11;
  }
  if (l3k3_groups_equal(groups_a, {2}) && l3k3_groups_equal(groups_b, {2}) &&
      l3k3_matrix_1x1(matrix, 2) && out_groups.empty()) {
    return kSus2TensorBlockDot22;
  }
  if (l3k3_groups_equal(groups_a, {3}) && l3k3_groups_equal(groups_b, {3}) &&
      l3k3_matrix_1x1(matrix, 3) && out_groups.empty()) {
    return kSus2TensorBlockDot33;
  }
  if (l3k3_groups_equal(groups_a, {1}) && l3k3_groups_equal(groups_b, {2}) &&
      l3k3_matrix_1x1(matrix, 1) && l3k3_groups_equal(out_groups, {1}) &&
      l3k3_labels_single_group(labels, 1, 0, 1)) {
    return kSus2TensorBlockVecSym2ToVec;
  }
  if (l3k3_groups_equal(groups_a, {2}) && l3k3_groups_equal(groups_b, {1}) &&
      l3k3_matrix_1x1(matrix, 1) && l3k3_groups_equal(out_groups, {1}) &&
      l3k3_labels_single_group(labels, 0, 0, 1)) {
    return kSus2TensorBlockSym2VecToVec;
  }
  if (l3k3_groups_equal(groups_a, {1}) && l3k3_groups_equal(groups_b, {3}) &&
      l3k3_matrix_1x1(matrix, 1) && l3k3_groups_equal(out_groups, {2}) &&
      l3k3_labels_single_group(labels, 1, 0, 2)) {
    return kSus2TensorBlockVecSym3ToSym2;
  }
  if (l3k3_groups_equal(groups_a, {3}) && l3k3_groups_equal(groups_b, {1}) &&
      l3k3_matrix_1x1(matrix, 1) && l3k3_groups_equal(out_groups, {2}) &&
      l3k3_labels_single_group(labels, 0, 0, 2)) {
    return kSus2TensorBlockSym3VecToSym2;
  }
  if (l3k3_groups_equal(groups_a, {2}) && l3k3_groups_equal(groups_b, {3}) &&
      l3k3_matrix_1x1(matrix, 2) && l3k3_groups_equal(out_groups, {1}) &&
      l3k3_labels_single_group(labels, 1, 0, 1)) {
    return kSus2TensorBlockSym2Sym3ToVec;
  }
  if (l3k3_groups_equal(groups_a, {3}) && l3k3_groups_equal(groups_b, {2}) &&
      l3k3_matrix_1x1(matrix, 2) && l3k3_groups_equal(out_groups, {1}) &&
      l3k3_labels_single_group(labels, 0, 0, 1)) {
    return kSus2TensorBlockSym3Sym2ToVec;
  }
  if (l3k3_groups_equal(groups_a, {2}) && l3k3_groups_equal(groups_b, {2}) &&
      l3k3_matrix_1x1(matrix, 1) && l3k3_groups_equal(out_groups, {2}) &&
      l3k3_labels_coarsened_two(labels, 0, 0, 1, 1, 0, 1)) {
    return kSus2TensorBlockSym2Sym2ToSym2;
  }
  if (l3k3_groups_equal(groups_a, {2}) && l3k3_groups_equal(groups_b, {2}) &&
      l3k3_matrix_1x1(matrix, 1) && l3k3_groups_equal(out_groups, {1, 1}) &&
      l3k3_labels_two_groups(labels, 0, 0, 1, 1, 0, 1)) {
    return kSus2TensorBlockSym2Sym2ToMatAB;
  }
  if (l3k3_groups_equal(groups_a, {2}) && l3k3_groups_equal(groups_b, {2}) &&
      l3k3_matrix_1x1(matrix, 1) && l3k3_groups_equal(out_groups, {1, 1}) &&
      l3k3_labels_two_groups(labels, 1, 0, 1, 0, 0, 1)) {
    return kSus2TensorBlockSym2Sym2ToMatBA;
  }
  if (l3k3_groups_equal(groups_a, {1}) && l3k3_groups_equal(groups_b, {1}) &&
      l3k3_matrix_1x1(matrix, 0) && l3k3_groups_equal(out_groups, {1, 1}) &&
      l3k3_labels_two_groups(labels, 0, 0, 1, 1, 0, 1)) {
    return kSus2TensorBlockVecVecOuterAB;
  }
  if (l3k3_groups_equal(groups_a, {1}) && l3k3_groups_equal(groups_b, {1}) &&
      l3k3_matrix_1x1(matrix, 0) && l3k3_groups_equal(out_groups, {1, 1}) &&
      l3k3_labels_two_groups(labels, 1, 0, 1, 0, 0, 1)) {
    return kSus2TensorBlockVecVecOuterBA;
  }
  if (l3k3_groups_equal(groups_a, {1}) && l3k3_groups_equal(groups_b, {1}) &&
      l3k3_matrix_1x1(matrix, 0) && l3k3_groups_equal(out_groups, {2}) &&
      l3k3_labels_coarsened_two(labels, 0, 0, 1, 1, 0, 1)) {
    return kSus2TensorBlockVecVecToSym2;
  }
  if (l3k3_groups_equal(groups_a, {2}) && l3k3_groups_equal(groups_b, {1, 1}) &&
      l3k3_matrix_1x2(matrix, 1, 1) && out_groups.empty()) {
    return kSus2TensorBlockSym2MatScalar;
  }
  if (l3k3_groups_equal(groups_a, {1, 1}) && l3k3_groups_equal(groups_b, {2}) &&
      l3k3_matrix_2x1(matrix, 1, 1) && out_groups.empty()) {
    return kSus2TensorBlockMatSym2Scalar;
  }
  if (l3k3_groups_equal(groups_a, {1, 1}) && l3k3_groups_equal(groups_b, {1, 1}) &&
      l3k3_matrix_2x2(matrix, 1, 0, 0, 1) && out_groups.empty()) {
    return kSus2TensorBlockMatMatSameScalar;
  }
  if (l3k3_groups_equal(groups_a, {1, 1}) && l3k3_groups_equal(groups_b, {1, 1}) &&
      l3k3_matrix_2x2(matrix, 0, 1, 1, 0) && out_groups.empty()) {
    return kSus2TensorBlockMatMatTransScalar;
  }
  return kSus2TensorBlockGeneric;
}

const char* l3k3_tensor_block_kind_name(int kind)
{
  switch (kind) {
    case kSus2TensorBlockGeneric:
      return "generic";
    case kSus2TensorBlockScalarScalar:
      return "scalar_scalar";
    case kSus2TensorBlockScalarTensor:
      return "scalar_tensor";
    case kSus2TensorBlockTensorScalar:
      return "tensor_scalar";
    case kSus2TensorBlockDot11:
      return "dot11";
    case kSus2TensorBlockDot22:
      return "dot22";
    case kSus2TensorBlockDot33:
      return "dot33";
    case kSus2TensorBlockVecSym2ToVec:
      return "vec_sym2_vec";
    case kSus2TensorBlockSym2VecToVec:
      return "sym2_vec_vec";
    case kSus2TensorBlockVecSym3ToSym2:
      return "vec_sym3_sym2";
    case kSus2TensorBlockSym3VecToSym2:
      return "sym3_vec_sym2";
    case kSus2TensorBlockSym2Sym3ToVec:
      return "sym2_sym3_vec";
    case kSus2TensorBlockSym3Sym2ToVec:
      return "sym3_sym2_vec";
    case kSus2TensorBlockSym2Sym2ToSym2:
      return "sym2_sym2_sym2";
    case kSus2TensorBlockSym2Sym2ToMatAB:
      return "sym2_sym2_mat_ab";
    case kSus2TensorBlockSym2Sym2ToMatBA:
      return "sym2_sym2_mat_ba";
    case kSus2TensorBlockVecVecOuterAB:
      return "vec_vec_outer_ab";
    case kSus2TensorBlockVecVecOuterBA:
      return "vec_vec_outer_ba";
    case kSus2TensorBlockVecVecToSym2:
      return "vec_vec_sym2";
    case kSus2TensorBlockSym2MatScalar:
      return "sym2_mat_scalar";
    case kSus2TensorBlockMatSym2Scalar:
      return "mat_sym2_scalar";
    case kSus2TensorBlockMatMatSameScalar:
      return "mat_mat_same_scalar";
    case kSus2TensorBlockMatMatTransScalar:
      return "mat_mat_trans_scalar";
    case kSus2TensorBlockStructured:
      return "structured";
    default:
      return "unknown";
  }
}

int l3k3_tensor_block_fast_row_count(int kind, int component_count, int generic_rows)
{
  switch (kind) {
    case kSus2TensorBlockScalarScalar:
      return 1;
    case kSus2TensorBlockScalarTensor:
    case kSus2TensorBlockTensorScalar:
      return component_count;
    case kSus2TensorBlockDot11:
      return 3;
    case kSus2TensorBlockDot22:
      return 6;
    case kSus2TensorBlockDot33:
      return 10;
    case kSus2TensorBlockVecSym2ToVec:
    case kSus2TensorBlockSym2VecToVec:
      return 9;
    case kSus2TensorBlockVecSym3ToSym2:
    case kSus2TensorBlockSym3VecToSym2:
    case kSus2TensorBlockSym2Sym3ToVec:
    case kSus2TensorBlockSym3Sym2ToVec:
    case kSus2TensorBlockSym2Sym2ToSym2:
      return 18;
    case kSus2TensorBlockSym2Sym2ToMatAB:
    case kSus2TensorBlockSym2Sym2ToMatBA:
      return 27;
    case kSus2TensorBlockVecVecOuterAB:
    case kSus2TensorBlockVecVecOuterBA:
    case kSus2TensorBlockSym2MatScalar:
    case kSus2TensorBlockMatSym2Scalar:
    case kSus2TensorBlockMatMatSameScalar:
    case kSus2TensorBlockMatMatTransScalar:
      return 9;
    case kSus2TensorBlockVecVecToSym2:
      return 6;
    case kSus2TensorBlockStructured:
      return generic_rows;
    default:
      return generic_rows;
  }
}

std::string format_l3k3_tensor_block_histogram(const L3K3TensorBlockPlan& plan)
{
  std::ostringstream out;
  bool first = true;
  for (int kind = 0; kind < static_cast<int>(plan.op_kind_counts.size()); ++kind) {
    const int count = plan.op_kind_counts[kind];
    if (count <= 0) {
      continue;
    }
    if (!first) {
      out << ",";
    }
    out << l3k3_tensor_block_kind_name(kind) << ":" << count;
    first = false;
  }
  return first ? "none" : out.str();
}

L3K3TensorBlockPlan build_l3k3_tensor_block_plan(const SUS2HostModel& model)
{
  L3K3TensorBlockPlan plan;
  const TensorBasicLayout layout = detect_tensor_alpha_basic_layout(model);
  if (!layout.enabled || layout.l < 1 || layout.l > 4 || layout.k < 1 || layout.k > 4 ||
      !supports_product_assign(model)) {
    return plan;
  }

  std::vector<int> group_begin;
  std::vector<int> group_len;
  std::vector<int> group_dst;
  std::vector<int> dst_to_group(model.alpha_moments_count, -1);
  for (int t = 0; t < model.alpha_times_count;) {
    const int begin = t;
    const int dst = model.alpha_times[t * 4 + 3];
    do {
      ++t;
    } while (t < model.alpha_times_count && model.alpha_times[t * 4 + 3] == dst);
    group_begin.push_back(begin);
    group_len.push_back(t - begin);
    group_dst.push_back(dst);
    if (dst < 0 || dst >= model.alpha_moments_count || dst_to_group[dst] >= 0) {
      return L3K3TensorBlockPlan{};
    }
    dst_to_group[dst] = static_cast<int>(group_begin.size()) - 1;
  }

  std::vector<L3K3TensorBlockHostBlock> blocks;
  std::vector<int> moment_to_block(model.alpha_moments_count, -1);
  std::vector<int> starts(layout.l + 1, 0);
  std::vector<int> counts(layout.l + 1, 0);
  for (int rank = 0; rank <= layout.l; ++rank) {
    counts[rank] = tensor_symmetric_component_count(rank);
    starts[rank] = rank == 0 ? 0 : starts[rank - 1] + counts[rank - 1];
  }
  for (int group = 0; group < layout.k; ++group) {
    for (int rank = 0; rank <= layout.l; ++rank) {
      std::vector<int> tensor_groups;
      if (rank > 0) {
        tensor_groups.push_back(rank);
      }
      L3K3TensorBlockHostBlock block =
        make_l3k3_tensor_block(
          static_cast<int>(blocks.size()),
          group * layout.basic_per_group + starts[rank],
          counts[rank],
          tensor_groups);
      if (block.id < 0) {
        return L3K3TensorBlockPlan{};
      }
      blocks.push_back(block);
      for (int id : block.ids) {
        moment_to_block[id] = block.id;
      }
    }
  }

  for (int cursor = model.alpha_basic_count; cursor < model.alpha_moments_count;) {
    if (moment_to_block[cursor] >= 0) {
      ++cursor;
      continue;
    }
    const int group_index = dst_to_group[cursor];
    if (group_index < 0) {
      return L3K3TensorBlockPlan{};
    }
    int src_block_a = -1;
    int src_block_b = -1;
    for (int k = 0; k < group_len[group_index]; ++k) {
      const int t = group_begin[group_index] + k;
      const int src0 = model.alpha_times[t * 4 + 0];
      const int src1 = model.alpha_times[t * 4 + 1];
      if (src0 < 0 || src0 >= model.alpha_moments_count || src1 < 0 ||
          src1 >= model.alpha_moments_count || moment_to_block[src0] < 0 ||
          moment_to_block[src1] < 0) {
        return L3K3TensorBlockPlan{};
      }
      if (k == 0) {
        src_block_a = moment_to_block[src0];
        src_block_b = moment_to_block[src1];
      } else if (src_block_a != moment_to_block[src0] || src_block_b != moment_to_block[src1]) {
        return L3K3TensorBlockPlan{};
      }
    }
    const L3K3TensorBlockHostBlock& block_a = blocks[src_block_a];
    const L3K3TensorBlockHostBlock& block_b = blocks[src_block_b];

    std::vector<L3K3TensorBlockCandidate> candidates;
    const auto matrices = l3k3_contraction_matrices(block_a.groups, block_b.groups);
    for (const auto& matrix : matrices) {
      std::vector<L3K3TensorBlockLabel> labels;
      bool valid = true;
      int free_rank = 0;
      for (size_t i = 0; i < block_a.groups.size(); ++i) {
        int used = 0;
        for (size_t j = 0; j < block_b.groups.size(); ++j) {
          used += l3k3_matrix_value(matrix, static_cast<int>(i), static_cast<int>(j));
        }
        const int remaining = block_a.groups[i] - used;
        if (remaining < 0) {
          valid = false;
        } else if (remaining > 0) {
          L3K3TensorBlockLabel label;
          label.side = 0;
          label.index = static_cast<int>(i);
          label.size = remaining;
          labels.push_back(label);
          free_rank += remaining;
        }
      }
      for (size_t j = 0; j < block_b.groups.size(); ++j) {
        int used = 0;
        for (size_t i = 0; i < block_a.groups.size(); ++i) {
          used += l3k3_matrix_value(matrix, static_cast<int>(i), static_cast<int>(j));
        }
        const int remaining = block_b.groups[j] - used;
        if (remaining < 0) {
          valid = false;
        } else if (remaining > 0) {
          L3K3TensorBlockLabel label;
          label.side = 1;
          label.index = static_cast<int>(j);
          label.size = remaining;
          labels.push_back(label);
          free_rank += remaining;
        }
      }
      if (!valid || free_rank > layout.l) {
        continue;
      }
      l3k3_add_candidate_coarsenings(labels, matrix, candidates);
    }
    plan.candidate_count += static_cast<int>(candidates.size());

    const L3K3TensorBlockCandidate* matched = nullptr;
    int selected_op_kind = kSus2TensorBlockGeneric;
    int selected_generic_rows = 0;
    int selected_cost = 0;
    for (const auto& candidate : candidates) {
      if (candidate.component_count <= 0 || cursor + candidate.component_count > model.alpha_moments_count) {
        continue;
      }
      bool contiguous_dst = true;
      for (int k = 0; k < candidate.component_count; ++k) {
        const int g = group_index + k;
        if (g >= static_cast<int>(group_dst.size()) || group_dst[g] != cursor + k) {
          contiguous_dst = false;
          break;
        }
      }
      if (contiguous_dst &&
          l3k3_candidate_matches(model, group_begin, group_len, group_index, block_a, block_b, candidate)) {
        int op_kind = classify_l3k3_tensor_block_op(block_a, block_b, candidate);
        int generic_rows = 0;
        for (int component = 0; component < candidate.component_count; ++component) {
          generic_rows += group_len[group_index + component];
        }
        const int structured_work = l3k3_tensor_block_structured_work_count(candidate);
        if (op_kind == kSus2TensorBlockGeneric &&
            l3k3_tensor_block_structured_meta_supported(block_a, block_b, candidate) &&
            structured_work * 2 < generic_rows) {
          op_kind = kSus2TensorBlockStructured;
        }
        const int fast_rows = op_kind == kSus2TensorBlockStructured
          ? structured_work
          : l3k3_tensor_block_fast_row_count(op_kind, candidate.component_count, generic_rows);
        const int cost = 2 * fast_rows;
        ++plan.matched_candidate_count;
        if (matched == nullptr ||
            candidate.component_count > matched->component_count ||
            (candidate.component_count == matched->component_count &&
             (cost < selected_cost ||
              (cost == selected_cost && op_kind != kSus2TensorBlockGeneric &&
               selected_op_kind == kSus2TensorBlockGeneric)))) {
          matched = &candidate;
          selected_op_kind = op_kind;
          selected_generic_rows = generic_rows;
          selected_cost = cost;
        }
      }
    }
    if (matched == nullptr) {
      return L3K3TensorBlockPlan{};
    }

    L3K3TensorBlockHostBlock block = make_l3k3_tensor_block(
      static_cast<int>(blocks.size()), cursor, matched->component_count, matched->out_groups);
    if (block.id < 0) {
      return L3K3TensorBlockPlan{};
    }
    const int selected_row_begin =
      static_cast<int>(plan.rows.size() / kSus2TensorBlockRowWords);
    if (!append_l3k3_tensor_block_rows(
          model,
          group_begin,
          group_len,
          group_dst,
          group_index,
          matched->component_count,
          block_a,
          block_b,
          cursor,
          plan.rows)) {
      return L3K3TensorBlockPlan{};
    }
    const int selected_row_count =
      static_cast<int>(plan.rows.size() / kSus2TensorBlockRowWords) - selected_row_begin;
    if (!append_l3k3_tensor_block_meta(block_a, block_b, *matched, plan.metas)) {
      return L3K3TensorBlockPlan{};
    }
    plan.ops.push_back(group_index);
    plan.ops.push_back(matched->component_count);
    plan.ops.push_back(selected_op_kind);
    plan.ops.push_back(block_a.start);
    plan.ops.push_back(block_b.start);
    plan.ops.push_back(cursor);
    plan.op_rows.push_back(selected_row_begin);
    plan.op_rows.push_back(selected_row_count);
    if (selected_op_kind != kSus2TensorBlockGeneric) {
      ++plan.fast_op_count;
    } else {
      ++plan.generic_op_count;
      plan.generic_row_count += selected_generic_rows;
    }
    if (selected_op_kind >= 0 &&
        selected_op_kind < static_cast<int>(plan.op_kind_counts.size())) {
      ++plan.op_kind_counts[selected_op_kind];
    }
    plan.component_group_count += matched->component_count;
    plan.selected_cost_units += selected_cost;
    blocks.push_back(block);
    for (int id : block.ids) {
      moment_to_block[id] = block.id;
    }
    cursor += matched->component_count;
  }

  for (int moment : model.alpha_moment_mapping) {
    if (moment < 0 || moment >= model.alpha_moments_count || moment_to_block[moment] < 0) {
      return L3K3TensorBlockPlan{};
    }
  }
  plan.enabled = true;
  plan.op_count = static_cast<int>(plan.ops.size() / kSus2TensorBlockOpInts);
  plan.row_count = static_cast<int>(plan.rows.size() / kSus2TensorBlockRowWords);
  return plan;
}

TensorAutoDecision choose_tensor_auto_plan(
  const SUS2HostModel& model,
  const L3K3TensorBlockPlan& block_plan)
{
  TensorAutoDecision decision;
  const TensorBasicLayout layout = detect_tensor_alpha_basic_layout(model);
  std::ostringstream reason;
  if (!layout.enabled) {
    decision.reason = "no tensor alpha_index_basic layout; product graph selected";
    return decision;
  }
  if (!block_plan.enabled || block_plan.op_count <= 0) {
    reason << "tensor-block planner unsupported for l" << layout.l << "k" << layout.k
           << "; product graph selected";
    decision.reason = reason.str();
    return decision;
  }

  const double fast_fraction =
    block_plan.op_count > 0 ? static_cast<double>(block_plan.fast_op_count) /
                                static_cast<double>(block_plan.op_count)
                            : 0.0;
  if (block_plan.generic_op_count > 0) {
    reason << "l" << layout.l << "k" << layout.k
           << ", scalars=" << model.alpha_scalar_moments
           << ", moments=" << model.alpha_moments_count
           << ", product_rules=" << model.alpha_times_count
           << ", graph_specific=yes"
           << ", tensor_ops=" << block_plan.op_count
           << ", fast_ops=" << block_plan.fast_op_count
           << ", generic_ops=" << block_plan.generic_op_count
           << ", fast_fraction=" << std::fixed << std::setprecision(3) << fast_fraction
           << ", component_groups=" << block_plan.component_group_count
           << ", candidate_matches=" << block_plan.matched_candidate_count << "/"
           << block_plan.candidate_count
           << ", generic_rows=" << block_plan.generic_row_count
           << ", specific_rows=" << block_plan.row_count
           << ", cost_units=" << block_plan.selected_cost_units
           << ", op_histogram=" << format_l3k3_tensor_block_histogram(block_plan)
           << "; product graph selected because tensor-block still needs packed-row fallback";
    decision.reason = reason.str();
    return decision;
  }

  reason << "l" << layout.l << "k" << layout.k
         << ", scalars=" << model.alpha_scalar_moments
         << ", moments=" << model.alpha_moments_count
         << ", product_rules=" << model.alpha_times_count
         << ", graph_specific=yes"
         << ", tensor_ops=" << block_plan.op_count
         << ", fast_ops=" << block_plan.fast_op_count
         << ", generic_ops=" << block_plan.generic_op_count
         << ", fast_fraction=" << std::fixed << std::setprecision(3) << fast_fraction
         << ", component_groups=" << block_plan.component_group_count
         << ", candidate_matches=" << block_plan.matched_candidate_count << "/"
         << block_plan.candidate_count
         << ", generic_rows=" << block_plan.generic_row_count
         << ", specific_rows=" << block_plan.row_count
         << ", cost_units=" << block_plan.selected_cost_units
         << ", op_histogram=" << format_l3k3_tensor_block_histogram(block_plan)
         << "; tensor-block selected";
  decision.use_tensor_block = true;
  decision.reason = reason.str();
  return decision;
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

void build_direct_radial_tables(
  const SUS2HostModel& model,
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
        const int scaling_block = model.mu_to_scaling_block[mu];
        const int scal_offset =
          2 * scaling_block * model.species_count * model.species_count + shift;
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

bool parse_radial_direct(
  const SUS2HostModel& model,
  int num_potential_options,
  const char** potential_options)
{
  bool use_direct = false;
  const char* env = std::getenv("SUS2_GPUMD_RADIAL_DIRECT");
  if (env != nullptr) {
    use_direct = parse_bool_value(env, "SUS2_GPUMD_RADIAL_DIRECT");
  }

  const int option_begin = std::min(num_potential_options, model.species_count);
  for (int i = option_begin; i < num_potential_options; ++i) {
    const std::string option = potential_options[i] == nullptr ? "" : potential_options[i];
    if (starts_with(option, "sus2_radial_direct=") || starts_with(option, "radial_direct=") ||
        starts_with(option, "sus2_no_lut=")) {
      const size_t eq = option.find('=');
      use_direct = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }

  if (use_direct && model.rb_size > 16) {
    sus2_input_error("SUS2 GPUMD radial_direct currently supports radial_basis_size <= 16.");
  }
  return use_direct;
}

bool parse_float_moments(
  const SUS2HostModel& model,
  int num_potential_options,
  const char** potential_options)
{
  bool use_float = false;
  const char* env = std::getenv("SUS2_GPUMD_FLOAT");
  if (env != nullptr) {
    use_float = parse_bool_value(env, "SUS2_GPUMD_FLOAT");
  }
  const char* env_nep = std::getenv("SUS2_GPUMD_NEPLIKE_FLOAT");
  if (env_nep != nullptr) {
    use_float = parse_bool_value(env_nep, "SUS2_GPUMD_NEPLIKE_FLOAT");
  }

  const int option_begin = std::min(num_potential_options, model.species_count);
  for (int i = option_begin; i < num_potential_options; ++i) {
    const std::string option = potential_options[i] == nullptr ? "" : potential_options[i];
    if (starts_with(option, "sus2_float=") || starts_with(option, "sus2_nep_float=") ||
        starts_with(option, "sus2_moment_float=") || starts_with(option, "moment_float=")) {
      const size_t eq = option.find('=');
      use_float = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_float;
}

bool parse_fused_energy_backward(
  const SUS2HostModel& model,
  int num_potential_options,
  const char** potential_options)
{
  bool use_fused = true;
  const char* env = std::getenv("SUS2_GPUMD_FUSED_BACKWARD");
  if (env != nullptr) {
    use_fused = parse_bool_value(env, "SUS2_GPUMD_FUSED_BACKWARD");
  }

  const int option_begin = std::min(num_potential_options, model.species_count);
  for (int i = option_begin; i < num_potential_options; ++i) {
    const std::string option = potential_options[i] == nullptr ? "" : potential_options[i];
    if (starts_with(option, "sus2_fused_backward=") ||
        starts_with(option, "fused_backward=")) {
      const size_t eq = option.find('=');
      use_fused = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_fused;
}

bool parse_tensor_force_grad_cache(
  const SUS2HostModel& model,
  int num_potential_options,
  const char** potential_options)
{
  bool use_cache = true;
  const char* env = std::getenv("SUS2_GPUMD_TENSOR_FORCE_GRAD_CACHE");
  if (env != nullptr) {
    use_cache = parse_bool_value(env, "SUS2_GPUMD_TENSOR_FORCE_GRAD_CACHE");
  }
  const char* old_env = std::getenv("SUS2_GPUMD_L3K3_FORCE_GRAD_CACHE");
  if (old_env != nullptr) {
    use_cache = parse_bool_value(old_env, "SUS2_GPUMD_L3K3_FORCE_GRAD_CACHE");
  }

  const int option_begin = std::min(num_potential_options, model.species_count);
  for (int i = option_begin; i < num_potential_options; ++i) {
    const std::string option = potential_options[i] == nullptr ? "" : potential_options[i];
    if (starts_with(option, "sus2_tensor_force_grad_cache=") ||
        starts_with(option, "tensor_force_grad_cache=") ||
        starts_with(option, "sus2_l3k3_force_grad_cache=") ||
        starts_with(option, "l3k3_force_grad_cache=")) {
      const size_t eq = option.find('=');
      use_cache = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_cache;
}

bool parse_l3k3_tensor_scalar(
  const SUS2HostModel& model,
  int num_potential_options,
  const char** potential_options)
{
  bool use_tensor_scalar = false;
  const char* env = std::getenv("SUS2_GPUMD_L3K3_TENSOR_SCALAR");
  if (env != nullptr) {
    use_tensor_scalar = parse_bool_value(env, "SUS2_GPUMD_L3K3_TENSOR_SCALAR");
  }
  const char* generic_env = std::getenv("SUS2_GPUMD_TENSOR_SCALAR");
  if (generic_env != nullptr) {
    use_tensor_scalar = parse_bool_value(generic_env, "SUS2_GPUMD_TENSOR_SCALAR");
  }

  const int option_begin = std::min(num_potential_options, model.species_count);
  for (int i = option_begin; i < num_potential_options; ++i) {
    const std::string option = potential_options[i] == nullptr ? "" : potential_options[i];
    if (starts_with(option, "sus2_l3k3_tensor_scalar=") ||
        starts_with(option, "l3k3_tensor_scalar=") ||
        starts_with(option, "sus2_tensor_scalar=") ||
        starts_with(option, "tensor_scalar=")) {
      const size_t eq = option.find('=');
      use_tensor_scalar = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_tensor_scalar;
}

bool parse_l3k3_tensor_block(
  const SUS2HostModel& model,
  int num_potential_options,
  const char** potential_options)
{
  bool use_tensor_block = false;
  const char* env = std::getenv("SUS2_GPUMD_L3K3_TENSOR_BLOCK");
  if (env != nullptr) {
    use_tensor_block = parse_bool_value(env, "SUS2_GPUMD_L3K3_TENSOR_BLOCK");
  }
  const char* generic_env = std::getenv("SUS2_GPUMD_TENSOR_BLOCK");
  if (generic_env != nullptr) {
    use_tensor_block = parse_bool_value(generic_env, "SUS2_GPUMD_TENSOR_BLOCK");
  }

  const int option_begin = std::min(num_potential_options, model.species_count);
  for (int i = option_begin; i < num_potential_options; ++i) {
    const std::string option = potential_options[i] == nullptr ? "" : potential_options[i];
    if (starts_with(option, "sus2_l3k3_tensor_block=") ||
        starts_with(option, "l3k3_tensor_block=") ||
        starts_with(option, "sus2_tensor_block=") ||
        starts_with(option, "tensor_block=")) {
      const size_t eq = option.find('=');
      use_tensor_block = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_tensor_block;
}

bool parse_tensor_auto(
  const SUS2HostModel& model,
  int num_potential_options,
  const char** potential_options)
{
  bool use_tensor_auto = false;
  const char* env = std::getenv("SUS2_GPUMD_TENSOR_AUTO");
  if (env != nullptr) {
    use_tensor_auto = parse_bool_value(env, "SUS2_GPUMD_TENSOR_AUTO");
  }
  const char* generic_env = std::getenv("SUS2_GPUMD_TENSOR");
  if (generic_env != nullptr) {
    use_tensor_auto = parse_bool_value(generic_env, "SUS2_GPUMD_TENSOR");
  }

  const int option_begin = std::min(num_potential_options, model.species_count);
  for (int i = option_begin; i < num_potential_options; ++i) {
    const std::string option = potential_options[i] == nullptr ? "" : potential_options[i];
    if (starts_with(option, "sus2_tensor_auto=") ||
        starts_with(option, "tensor_auto=") ||
        starts_with(option, "sus2_tensor=") ||
        starts_with(option, "tensor=")) {
      const size_t eq = option.find('=');
      use_tensor_auto = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_tensor_auto;
}

bool parse_l3k3_tensor_block_fast_forward(
  const SUS2HostModel& model,
  int num_potential_options,
  const char** potential_options)
{
  bool use_fast = true;
  const char* env = std::getenv("SUS2_GPUMD_L3K3_TENSOR_BLOCK_FAST_FORWARD");
  if (env != nullptr) {
    use_fast = parse_bool_value(env, "SUS2_GPUMD_L3K3_TENSOR_BLOCK_FAST_FORWARD");
  }

  const int option_begin = std::min(num_potential_options, model.species_count);
  for (int i = option_begin; i < num_potential_options; ++i) {
    const std::string option = potential_options[i] == nullptr ? "" : potential_options[i];
    if (starts_with(option, "sus2_l3k3_tensor_block_fast_forward=") ||
        starts_with(option, "tensor_block_fast_forward=")) {
      const size_t eq = option.find('=');
      use_fast = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_fast;
}

bool parse_l3k3_tensor_block_fast_backward(
  const SUS2HostModel& model,
  int num_potential_options,
  const char** potential_options)
{
  bool use_fast = true;
  const char* env = std::getenv("SUS2_GPUMD_L3K3_TENSOR_BLOCK_FAST_BACKWARD");
  if (env != nullptr) {
    use_fast = parse_bool_value(env, "SUS2_GPUMD_L3K3_TENSOR_BLOCK_FAST_BACKWARD");
  }

  const int option_begin = std::min(num_potential_options, model.species_count);
  for (int i = option_begin; i < num_potential_options; ++i) {
    const std::string option = potential_options[i] == nullptr ? "" : potential_options[i];
    if (starts_with(option, "sus2_l3k3_tensor_block_fast_backward=") ||
        starts_with(option, "tensor_block_fast_backward=")) {
      const size_t eq = option.find('=');
      use_fast = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_fast;
}

bool parse_fused_graph(
  const SUS2HostModel& model,
  int num_potential_options,
  const char** potential_options)
{
  bool use_fused = true;
  const char* env = std::getenv("SUS2_GPUMD_FUSED_GRAPH");
  if (env != nullptr) {
    use_fused = parse_bool_value(env, "SUS2_GPUMD_FUSED_GRAPH");
  }

  const int option_begin = std::min(num_potential_options, model.species_count);
  for (int i = option_begin; i < num_potential_options; ++i) {
    const std::string option = potential_options[i] == nullptr ? "" : potential_options[i];
    if (starts_with(option, "sus2_fused_graph=") ||
        starts_with(option, "fused_graph=")) {
      const size_t eq = option.find('=');
      use_fused = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_fused;
}

bool parse_product_assign(
  const SUS2HostModel& model,
  int num_potential_options,
  const char** potential_options)
{
  bool use_assign = true;
  const char* env = std::getenv("SUS2_GPUMD_PRODUCT_ASSIGN");
  if (env != nullptr) {
    use_assign = parse_bool_value(env, "SUS2_GPUMD_PRODUCT_ASSIGN");
  }

  const int option_begin = std::min(num_potential_options, model.species_count);
  for (int i = option_begin; i < num_potential_options; ++i) {
    const std::string option = potential_options[i] == nullptr ? "" : potential_options[i];
    if (starts_with(option, "sus2_product_assign=") ||
        starts_with(option, "product_assign=")) {
      const size_t eq = option.find('=');
      use_assign = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_assign;
}

bool parse_graph_specific_product(
  const SUS2HostModel& model,
  int num_potential_options,
  const char** potential_options)
{
  bool use_specific = false;
  const char* env = std::getenv("SUS2_GPUMD_GRAPH_SPECIFIC_PRODUCT");
  if (env != nullptr) {
    use_specific = parse_bool_value(env, "SUS2_GPUMD_GRAPH_SPECIFIC_PRODUCT");
  }

  const int option_begin = std::min(num_potential_options, model.species_count);
  for (int i = option_begin; i < num_potential_options; ++i) {
    const std::string option = potential_options[i] == nullptr ? "" : potential_options[i];
    if (starts_with(option, "sus2_graph_specific=") ||
        starts_with(option, "sus2_product_graph_specific=") ||
        starts_with(option, "product_graph_specific=")) {
      const size_t eq = option.find('=');
      use_specific = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_specific;
}

bool parse_force_self_buffer(
  const SUS2HostModel& model,
  int num_potential_options,
  const char** potential_options)
{
  bool use_buffer = false;
  const char* env = std::getenv("SUS2_GPUMD_FORCE_SELF_BUFFER");
  if (env != nullptr) {
    use_buffer = parse_bool_value(env, "SUS2_GPUMD_FORCE_SELF_BUFFER");
  }

  const int option_begin = std::min(num_potential_options, model.species_count);
  for (int i = option_begin; i < num_potential_options; ++i) {
    const std::string option = potential_options[i] == nullptr ? "" : potential_options[i];
    if (starts_with(option, "sus2_force_self_buffer=") ||
        starts_with(option, "sus2_force_self=") ||
        starts_with(option, "force_self_buffer=")) {
      const size_t eq = option.find('=');
      use_buffer = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_buffer;
}

bool parse_local_product_graph(
  const SUS2HostModel& model,
  int num_potential_options,
  const char** potential_options)
{
  bool use_local = false;
  const char* env = std::getenv("SUS2_GPUMD_LOCAL_PRODUCT_GRAPH");
  if (env != nullptr) {
    use_local = parse_bool_value(env, "SUS2_GPUMD_LOCAL_PRODUCT_GRAPH");
  }

  const int option_begin = std::min(num_potential_options, model.species_count);
  for (int i = option_begin; i < num_potential_options; ++i) {
    const std::string option = potential_options[i] == nullptr ? "" : potential_options[i];
    if (starts_with(option, "sus2_local_product_graph=") ||
        starts_with(option, "local_product_graph=")) {
      const size_t eq = option.find('=');
      use_local = parse_bool_value(option.substr(eq + 1), option.substr(0, eq));
    }
  }
  return use_local;
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

template <typename RealT>
__device__ __forceinline__ RealT sus2_shift_coeff(const SUS2DeviceModel& model, int type)
{
  if (model.use_float_model_params && model.use_const_float_coeffs) {
    return static_cast<RealT>(c_sus2_shift_coeffs_float[type]);
  }
  return model.use_float_model_params ? static_cast<RealT>(model.shift_coeffs_float[type])
                                      : static_cast<RealT>(model.shift_coeffs[type]);
}

template <typename RealT>
__device__ __forceinline__ RealT sus2_species_coeff(const SUS2DeviceModel& model, int type)
{
  if (model.use_float_model_params && model.use_const_float_coeffs) {
    return static_cast<RealT>(c_sus2_species_coeffs_float[type]);
  }
  return model.use_float_model_params ? static_cast<RealT>(model.species_coeffs_float[type])
                                      : static_cast<RealT>(model.species_coeffs[type]);
}

template <typename RealT>
__device__ __forceinline__ RealT sus2_moment_coeff(const SUS2DeviceModel& model, int idx)
{
  if (model.use_float_model_params && model.use_const_float_coeffs) {
    return static_cast<RealT>(c_sus2_moment_coeffs_float[idx]);
  }
  return model.use_float_model_params ? static_cast<RealT>(model.moment_coeffs_float[idx])
                                      : static_cast<RealT>(model.moment_coeffs[idx]);
}

template <typename RealT>
__device__ __forceinline__ RealT sus2_l3k3_tensor_scalar_coeff(const SUS2DeviceModel& model, int idx)
{
  return model.use_float_model_params && model.l3k3_tensor_scalar_coeffs_float != nullptr
    ? static_cast<RealT>(model.l3k3_tensor_scalar_coeffs_float[idx])
    : static_cast<RealT>(model.l3k3_tensor_scalar_coeffs[idx]);
}

__device__ __forceinline__ int sus2_scalar_moment_id(const SUS2DeviceModel& model, int idx)
{
  return model.use_const_scalar_moments
    ? static_cast<int>(c_sus2_alpha_moment_mapping_u16[idx])
    : model.alpha_moment_mapping[idx];
}

template <typename RealT>
__device__ __forceinline__ RealT sus2_device_softplus(RealT x)
{
  if (x > static_cast<RealT>(40.0)) {
    return x;
  }
  if (x < static_cast<RealT>(-40.0)) {
    return exp(x);
  }
  return log1p(exp(x));
}

template <typename RealT>
__device__ __forceinline__ void jacobi_spec_device(
  int block,
  int& alpha,
  int& beta,
  RealT& linear_const,
  RealT& linear_x)
{
  switch (block) {
    case 0:
      alpha = 0;
      beta = 0;
      linear_const = static_cast<RealT>(0.0);
      linear_x = static_cast<RealT>(1.0);
      break;
    case 1:
      alpha = 1;
      beta = 0;
      linear_const = static_cast<RealT>(0.5);
      linear_x = static_cast<RealT>(1.5);
      break;
    case 2:
      alpha = 1;
      beta = 1;
      linear_const = static_cast<RealT>(0.0);
      linear_x = static_cast<RealT>(2.0);
      break;
    case 3:
      alpha = 2;
      beta = 0;
      linear_const = static_cast<RealT>(1.0);
      linear_x = static_cast<RealT>(2.0);
      break;
    case 4:
      alpha = 2;
      beta = 1;
      linear_const = static_cast<RealT>(0.5);
      linear_x = static_cast<RealT>(2.5);
      break;
    default:
      alpha = 2;
      beta = 2;
      linear_const = static_cast<RealT>(0.0);
      linear_x = static_cast<RealT>(3.0);
      break;
  }
}

template <typename RealT>
__device__ __forceinline__ void jacobi_weight_terms_device(
  int alpha,
  int beta,
  RealT x,
  RealT& sqrt_weight,
  RealT& log_weight_x)
{
  constexpr double kEpsDouble = 1.0e-12;
  const RealT eps = static_cast<RealT>(kEpsDouble);
  const RealT one_minus_x_raw = static_cast<RealT>(1.0) - x;
  const RealT one_plus_x_raw = static_cast<RealT>(1.0) + x;
  const RealT one_minus_x = one_minus_x_raw > eps ? one_minus_x_raw : eps;
  const RealT one_plus_x = one_plus_x_raw > eps ? one_plus_x_raw : eps;

  sqrt_weight = static_cast<RealT>(1.0);
  log_weight_x = static_cast<RealT>(0.0);
  if (alpha == 1) {
    sqrt_weight *= sqrt(one_minus_x);
  } else if (alpha == 2) {
    sqrt_weight *= one_minus_x;
  }
  if (beta == 1) {
    sqrt_weight *= sqrt(one_plus_x);
  } else if (beta == 2) {
    sqrt_weight *= one_plus_x;
  }
  if (alpha != 0) {
    log_weight_x -= static_cast<RealT>(0.5 * alpha) / one_minus_x;
  }
  if (beta != 0) {
    log_weight_x += static_cast<RealT>(0.5 * beta) / one_plus_x;
  }
}

template <typename RealT>
__device__ __forceinline__ void jacobi_coefficients_device(
  int block,
  int order,
  RealT& coeff_const,
  RealT& coeff_x,
  RealT& prev_coeff)
{
  if (block >= 0 && block <= kJacobiMaxIndexedBlock && order >= 0 &&
      order <= kSus2MaxDirectRbSize) {
    const int idx = block * (kSus2MaxDirectRbSize + 1) + order;
    coeff_const = static_cast<RealT>(c_sus2_jacobi_coeff_const[idx]);
    coeff_x = static_cast<RealT>(c_sus2_jacobi_coeff_x[idx]);
    prev_coeff = static_cast<RealT>(c_sus2_jacobi_prev_coeff[idx]);
    return;
  }

  int alpha_i = 0;
  int beta_i = 0;
  RealT linear_const = static_cast<RealT>(0.0);
  RealT linear_x = static_cast<RealT>(0.0);
  jacobi_spec_device(block, alpha_i, beta_i, linear_const, linear_x);
  const RealT alpha = static_cast<RealT>(alpha_i);
  const RealT beta = static_cast<RealT>(beta_i);
  const RealT n = static_cast<RealT>(order);
  const RealT denom =
    static_cast<RealT>(2.0) * n * (n + alpha + beta) *
    (static_cast<RealT>(2.0) * n + alpha + beta - static_cast<RealT>(2.0));
  const RealT b = static_cast<RealT>(2.0) * n + alpha + beta - static_cast<RealT>(1.0);
  const RealT c = (static_cast<RealT>(2.0) * n + alpha + beta) *
                  (static_cast<RealT>(2.0) * n + alpha + beta - static_cast<RealT>(2.0));
  const RealT d = alpha * alpha - beta * beta;
  const RealT e = static_cast<RealT>(2.0) * (n + alpha - static_cast<RealT>(1.0)) *
                  (n + beta - static_cast<RealT>(1.0)) *
                  (static_cast<RealT>(2.0) * n + alpha + beta);
  coeff_const = b * d / denom;
  coeff_x = b * c / denom;
  prev_coeff = e / denom;
}

template <typename RealT, int RadialFuncs, int RbSize, int AngularChannels, bool StaticShape>
__device__ __forceinline__ void direct_chebyshev_vals_ders_impl(
  const SUS2DeviceModel& model,
  int pair,
  RealT r,
  RealT* vals,
  RealT* ders)
{
  const RealT dr = r - static_cast<RealT>(model.max_dist);
  const RealT cutoff_f = dr * dr;
  const RealT cutoff_der = static_cast<RealT>(2.0) * dr;
  const int radial_funcs = StaticShape ? RadialFuncs : model.radial_funcs_count;
  const int rb_size = StaticShape ? RbSize : model.rb_size;

  for (int mu = 0; mu < RadialFuncs; ++mu) {
    if (!StaticShape && mu >= radial_funcs) {
      break;
    }
    const size_t scal_base = (static_cast<size_t>(pair) * radial_funcs + mu) * 2;
    const RealT scal = static_cast<RealT>(model.radial_direct_scal_s[scal_base + 0]);
    const RealT shift = static_cast<RealT>(model.radial_direct_scal_s[scal_base + 1]);
    const RealT z = static_cast<RealT>(0.5) * scal * (r - shift);
    const RealT ksi = tanh(z);
    const RealT mult = static_cast<RealT>(0.5) * scal * (static_cast<RealT>(1.0) - ksi * ksi);
    const size_t coeff_base = (static_cast<size_t>(pair) * radial_funcs + mu) * rb_size;

    RealT prev = static_cast<RealT>(1.0);
    RealT prev_x = static_cast<RealT>(0.0);
    RealT acc_s = static_cast<RealT>(model.radial_direct_coeffs[coeff_base]);
    RealT acc_sx = static_cast<RealT>(0.0);

    if (rb_size > 1) {
      RealT curr = ksi;
      RealT curr_x = static_cast<RealT>(1.0);
      RealT coeff = static_cast<RealT>(model.radial_direct_coeffs[coeff_base + 1]);
      acc_s += coeff * curr;
      acc_sx += coeff * curr_x;

      for (int xi = 2; xi < RbSize; ++xi) {
        if (!StaticShape && xi >= rb_size) {
          break;
        }
        const RealT next = static_cast<RealT>(2.0) * ksi * curr - prev;
        const RealT next_x =
          static_cast<RealT>(2.0) * (curr + ksi * curr_x) - prev_x;
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

template <
  typename RealT,
  int RadialFuncs,
  int RbSize,
  int AngularChannels,
  bool StaticShape,
  int BasisKind = -1>
__device__ __forceinline__ void direct_jacobi_vals_ders_impl(
  const SUS2DeviceModel& model,
  int pair,
  RealT r,
  RealT* vals,
  RealT* ders)
{
  const RealT dr = r - static_cast<RealT>(model.max_dist);
  const RealT cutoff_f = dr * dr;
  const RealT cutoff_der = static_cast<RealT>(2.0) * dr;
  const int radial_funcs = StaticShape ? RadialFuncs : model.radial_funcs_count;
  const int rb_size = StaticShape ? RbSize : model.rb_size;
  const int angular_channels = StaticShape ? AngularChannels : model.angular_channels;
  const bool apply_weight = BasisKind >= 0
    ? BasisKind == static_cast<int>(RadialBasisKind::JacobiSSS)
    : model.radial_basis_kind == static_cast<int>(RadialBasisKind::JacobiSSS);

  for (int mu = 0; mu < RadialFuncs; ++mu) {
    if (!StaticShape && mu >= radial_funcs) {
      break;
    }
    const size_t scal_base = (static_cast<size_t>(pair) * radial_funcs + mu) * 2;
    const RealT scal = static_cast<RealT>(model.radial_direct_scal_s[scal_base + 0]);
    const RealT shift = static_cast<RealT>(model.radial_direct_scal_s[scal_base + 1]);
    const RealT z = static_cast<RealT>(0.5) * scal * (r - shift);
    RealT x = tanh(z);
    const RealT eps = static_cast<RealT>(1.0e-12);
    const RealT x_min = -static_cast<RealT>(1.0) + eps;
    const RealT x_max = static_cast<RealT>(1.0) - eps;
    x = x < x_min ? x_min : (x > x_max ? x_max : x);
    const RealT x_r = static_cast<RealT>(0.5) * scal * (static_cast<RealT>(1.0) - x * x);
    const size_t coeff_base = (static_cast<size_t>(pair) * radial_funcs + mu) * rb_size;
    const int block = mu / angular_channels;

    int alpha = 0;
    int beta = 0;
    RealT linear_const = static_cast<RealT>(0.0);
    RealT linear_x = static_cast<RealT>(0.0);
    jacobi_spec_device(block, alpha, beta, linear_const, linear_x);

    RealT sqrt_weight = static_cast<RealT>(1.0);
    RealT log_weight_x = static_cast<RealT>(0.0);
    if (apply_weight) {
      jacobi_weight_terms_device(alpha, beta, x, sqrt_weight, log_weight_x);
    }

    RealT y_prev = static_cast<RealT>(0.0);
    RealT y_prev_x = static_cast<RealT>(0.0);
    RealT y_curr = sqrt_weight;
    RealT y_curr_x = sqrt_weight * log_weight_x;
    RealT acc_s = static_cast<RealT>(model.radial_direct_coeffs[coeff_base]) * y_curr;
    RealT acc_sx = static_cast<RealT>(model.radial_direct_coeffs[coeff_base]) * y_curr_x;

    if (rb_size > 1) {
      const RealT linear = linear_const + linear_x * x;
      RealT y_next = linear * y_curr;
      RealT y_next_x = linear_x * y_curr + linear * y_curr_x;
      RealT radial_coeff = static_cast<RealT>(model.radial_direct_coeffs[coeff_base + 1]);
      acc_s += radial_coeff * y_next;
      acc_sx += radial_coeff * y_next_x;

      y_prev = y_curr;
      y_prev_x = y_curr_x;
      y_curr = y_next;
      y_curr_x = y_next_x;
      for (int order = 2; order < RbSize; ++order) {
        if (!StaticShape && order >= rb_size) {
          break;
        }
        RealT coeff_const = static_cast<RealT>(0.0);
        RealT coeff_x = static_cast<RealT>(0.0);
        RealT prev_coeff = static_cast<RealT>(0.0);
        jacobi_coefficients_device(block, order, coeff_const, coeff_x, prev_coeff);
        const RealT recurrence_coeff = coeff_const + coeff_x * x;
        y_next = recurrence_coeff * y_curr - prev_coeff * y_prev;
        y_next_x =
          coeff_x * y_curr + recurrence_coeff * y_curr_x - prev_coeff * y_prev_x;
        radial_coeff = static_cast<RealT>(model.radial_direct_coeffs[coeff_base + order]);
        acc_s += radial_coeff * y_next;
        acc_sx += radial_coeff * y_next_x;
        y_prev = y_curr;
        y_prev_x = y_curr_x;
        y_curr = y_next;
        y_curr_x = y_next_x;
      }
    }

    vals[mu] = cutoff_f * acc_s;
    if (ders != nullptr) {
      ders[mu] = cutoff_der * acc_s + cutoff_f * acc_sx * x_r;
    }
  }
}

template <typename RealT, int RadialFuncs, int RbSize, bool StaticShape, int BasisKind = -1>
__device__ __forceinline__ void direct_laguerre_vals_ders_impl(
  const SUS2DeviceModel& model,
  int pair,
  RealT r,
  RealT* vals,
  RealT* ders)
{
  const bool apply_exponential_envelope = BasisKind >= 0
    ? BasisKind == static_cast<int>(RadialBasisKind::LaguerreLog1p) ||
        BasisKind == static_cast<int>(RadialBasisKind::LaguerreLog1pPositive)
    : model.radial_basis_kind == static_cast<int>(RadialBasisKind::LaguerreLog1p) ||
        model.radial_basis_kind == static_cast<int>(RadialBasisKind::LaguerreLog1pPositive);
  const bool positive_params = BasisKind >= 0
    ? BasisKind == static_cast<int>(RadialBasisKind::LaguerreLog1pPositive)
    : model.radial_basis_kind == static_cast<int>(RadialBasisKind::LaguerreLog1pPositive);
  const int radial_funcs = StaticShape ? RadialFuncs : model.radial_funcs_count;
  const int rb_size = StaticShape ? RbSize : model.rb_size;
  const RealT dr = r - static_cast<RealT>(model.max_dist);
  const RealT cutoff_f = dr * dr;
  const RealT cutoff_der = static_cast<RealT>(2.0) * dr;

  for (int mu = 0; mu < RadialFuncs; ++mu) {
    if (!StaticShape && mu >= radial_funcs) {
      break;
    }
    const size_t scal_base = (static_cast<size_t>(pair) * radial_funcs + mu) * 2;
    RealT scal = static_cast<RealT>(model.radial_direct_scal_s[scal_base + 0]);
    RealT rho = static_cast<RealT>(model.radial_direct_scal_s[scal_base + 1]);
    if (positive_params) {
      scal = static_cast<RealT>(kLaguerrePositiveParamFloor) + sus2_device_softplus(scal);
      rho = static_cast<RealT>(kLaguerrePositiveParamFloor) + sus2_device_softplus(rho);
    }
    rho = rho > static_cast<RealT>(kLaguerreMinRho) ? rho : static_cast<RealT>(kLaguerreMinRho);

    const RealT log_term = log1p(r / rho);
    const RealT u = scal * log_term;
    const RealT u_r = scal / (rho + r);
    const RealT exp_factor =
      apply_exponential_envelope ? exp(static_cast<RealT>(-0.5) * u) : static_cast<RealT>(1.0);
    const size_t coeff_base = (static_cast<size_t>(pair) * radial_funcs + mu) * rb_size;

    RealT phi_prev = static_cast<RealT>(0.0);
    RealT dphi_prev = static_cast<RealT>(0.0);
    RealT phi_curr = cutoff_f * exp_factor;
    RealT dphi_curr = cutoff_der * exp_factor;
    if (apply_exponential_envelope) {
      dphi_curr -= static_cast<RealT>(0.5) * u_r * phi_curr;
    }

    RealT acc_s = static_cast<RealT>(model.radial_direct_coeffs[coeff_base]) * phi_curr;
    RealT acc_sr = static_cast<RealT>(model.radial_direct_coeffs[coeff_base]) * dphi_curr;

    for (int n = 0; n < RbSize - 1; ++n) {
      if (!StaticShape && n >= rb_size - 1) {
        break;
      }
      const RealT inv_np1 = static_cast<RealT>(1.0) / (static_cast<RealT>(n) + static_cast<RealT>(1.0));
      const RealT recurrence_coeff =
        (static_cast<RealT>(2.0 * n + 1.0) - u) * inv_np1;
      const RealT prev_coeff = static_cast<RealT>(n) * inv_np1;
      const RealT phi_next = recurrence_coeff * phi_curr - prev_coeff * phi_prev;
      const RealT dphi_next =
        -u_r * inv_np1 * phi_curr + recurrence_coeff * dphi_curr - prev_coeff * dphi_prev;
      const RealT radial_coeff = static_cast<RealT>(model.radial_direct_coeffs[coeff_base + n + 1]);
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

template <typename RealT, int RadialFuncs, int RbSize, int AngularChannels, bool StaticShape>
__device__ __forceinline__ void direct_radial_vals_ders_impl(
  const SUS2DeviceModel& model,
  int pair,
  RealT r,
  RealT* vals,
  RealT* ders)
{
  if (model.radial_basis_kind == static_cast<int>(RadialBasisKind::ChebyshevSSS)) {
    direct_chebyshev_vals_ders_impl<RealT, RadialFuncs, RbSize, AngularChannels, StaticShape>(
      model, pair, r, vals, ders);
  } else if (
    model.radial_basis_kind == static_cast<int>(RadialBasisKind::JacobiSSS) ||
    model.radial_basis_kind == static_cast<int>(RadialBasisKind::JacobiSSSNoWeight)) {
    direct_jacobi_vals_ders_impl<RealT, RadialFuncs, RbSize, AngularChannels, StaticShape>(
      model, pair, r, vals, ders);
  } else {
    direct_laguerre_vals_ders_impl<RealT, RadialFuncs, RbSize, StaticShape>(
      model, pair, r, vals, ders);
  }
}

template <typename RealT>
__device__ __noinline__ void direct_radial_vals_ders_dynamic(
  const SUS2DeviceModel& model,
  int pair,
  RealT r,
  RealT* vals,
  RealT* ders)
{
  direct_radial_vals_ders_impl<RealT, 32, 16, 8, false>(model, pair, r, vals, ders);
}

template <typename RealT, int L, int K, int RbSize>
__device__ __forceinline__ bool direct_radial_vals_ders_lkrb_static(
  const SUS2DeviceModel& model,
  int pair,
  RealT r,
  RealT* vals,
  RealT* ders)
{
  constexpr int RadialFuncs = K * (L + 1);
  constexpr int AngularChannels = L + 1;
  if (model.radial_funcs_count != RadialFuncs || model.rb_size != RbSize) {
    return false;
  }

  if (model.radial_basis_kind == static_cast<int>(RadialBasisKind::ChebyshevSSS)) {
    direct_chebyshev_vals_ders_impl<RealT, RadialFuncs, RbSize, AngularChannels, true>(
      model, pair, r, vals, ders);
    return true;
  }
  if (model.radial_basis_kind == static_cast<int>(RadialBasisKind::JacobiSSS)) {
    direct_jacobi_vals_ders_impl<
      RealT,
      RadialFuncs,
      RbSize,
      AngularChannels,
      true,
      static_cast<int>(RadialBasisKind::JacobiSSS)>(
      model, pair, r, vals, ders);
    return true;
  }
  if (model.radial_basis_kind == static_cast<int>(RadialBasisKind::JacobiSSSNoWeight)) {
    direct_jacobi_vals_ders_impl<
      RealT,
      RadialFuncs,
      RbSize,
      AngularChannels,
      true,
      static_cast<int>(RadialBasisKind::JacobiSSSNoWeight)>(
      model, pair, r, vals, ders);
    return true;
  }
  if (model.radial_basis_kind == static_cast<int>(RadialBasisKind::LaguerreLog1p)) {
    direct_laguerre_vals_ders_impl<
      RealT,
      RadialFuncs,
      RbSize,
      true,
      static_cast<int>(RadialBasisKind::LaguerreLog1p)>(
      model, pair, r, vals, ders);
    return true;
  }
  if (model.radial_basis_kind == static_cast<int>(RadialBasisKind::LaguerreLog1pNoEnv)) {
    direct_laguerre_vals_ders_impl<
      RealT,
      RadialFuncs,
      RbSize,
      true,
      static_cast<int>(RadialBasisKind::LaguerreLog1pNoEnv)>(
      model, pair, r, vals, ders);
    return true;
  }
  if (model.radial_basis_kind == static_cast<int>(RadialBasisKind::LaguerreLog1pPositive)) {
    direct_laguerre_vals_ders_impl<
      RealT,
      RadialFuncs,
      RbSize,
      true,
      static_cast<int>(RadialBasisKind::LaguerreLog1pPositive)>(
      model, pair, r, vals, ders);
    return true;
  }
  return false;
}

template <typename RealT, int L, int K>
__device__ __forceinline__ bool direct_radial_vals_ders_lk_static(
  const SUS2DeviceModel& model,
  int pair,
  RealT r,
  RealT* vals,
  RealT* ders)
{
  constexpr int RadialFuncs = K * (L + 1);
  if (model.radial_funcs_count != RadialFuncs) {
    return false;
  }

  switch (model.rb_size) {
    case 1:
      return direct_radial_vals_ders_lkrb_static<RealT, L, K, 1>(model, pair, r, vals, ders);
    case 2:
      return direct_radial_vals_ders_lkrb_static<RealT, L, K, 2>(model, pair, r, vals, ders);
    case 3:
      return direct_radial_vals_ders_lkrb_static<RealT, L, K, 3>(model, pair, r, vals, ders);
    case 4:
      return direct_radial_vals_ders_lkrb_static<RealT, L, K, 4>(model, pair, r, vals, ders);
    case 5:
      return direct_radial_vals_ders_lkrb_static<RealT, L, K, 5>(model, pair, r, vals, ders);
    case 6:
      return direct_radial_vals_ders_lkrb_static<RealT, L, K, 6>(model, pair, r, vals, ders);
    case 7:
      return direct_radial_vals_ders_lkrb_static<RealT, L, K, 7>(model, pair, r, vals, ders);
    case 8:
      return direct_radial_vals_ders_lkrb_static<RealT, L, K, 8>(model, pair, r, vals, ders);
    case 9:
      return direct_radial_vals_ders_lkrb_static<RealT, L, K, 9>(model, pair, r, vals, ders);
    case 10:
      return direct_radial_vals_ders_lkrb_static<RealT, L, K, 10>(model, pair, r, vals, ders);
    default:
      return false;
  }
}

template <typename RealT>
__device__ __forceinline__ void interp_radial_vals_ders(
  const SUS2DeviceModel& model,
  int pair,
  RealT r,
  RealT* vals,
  RealT* ders)
{
  if (model.use_radial_direct) {
    direct_radial_vals_ders_dynamic(model, pair, r, vals, ders);
    return;
  }

  const RealT scaled_r = r * static_cast<RealT>(model.lut_inv_dr);
  int lut_idx = static_cast<int>(floor(scaled_r));
  if (lut_idx < 0) {
    lut_idx = 0;
  }
  if (lut_idx > model.lut_size - 2) {
    lut_idx = model.lut_size - 2;
  }
  const int lut_next = lut_idx + 1;
  const RealT t = scaled_r - static_cast<RealT>(lut_idx);
  const size_t base0 =
    (static_cast<size_t>(pair) * model.lut_size + lut_idx) * model.radial_funcs_count;
  const size_t base1 =
    (static_cast<size_t>(pair) * model.lut_size + lut_next) * model.radial_funcs_count;
  for (int mu = 0; mu < model.radial_funcs_count; ++mu) {
    const RealT v0 = static_cast<RealT>(model.lut_vals[base0 + mu]);
    const RealT v1 = static_cast<RealT>(model.lut_vals[base1 + mu]);
    vals[mu] = v0 + t * (v1 - v0);
    if (ders != nullptr) {
      const RealT d0 = static_cast<RealT>(model.lut_ders[base0 + mu]);
      const RealT d1 = static_cast<RealT>(model.lut_ders[base1 + mu]);
      ders[mu] = d0 + t * (d1 - d0);
    }
  }
}

template <int L, int K>
struct Sus2DirectRadialStaticDispatch {
  template <typename RealT>
  __device__ __forceinline__ static void eval(
    const SUS2DeviceModel& model,
    int pair,
    RealT r,
    RealT* vals,
    RealT* ders)
  {
    interp_radial_vals_ders(model, pair, r, vals, ders);
  }
};

#define SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH(LVAL, KVAL) \
  template <> \
  struct Sus2DirectRadialStaticDispatch<LVAL, KVAL> { \
    template <typename RealT> \
    __device__ __forceinline__ static void eval( \
      const SUS2DeviceModel& model, \
      int pair, \
      RealT r, \
      RealT* vals, \
      RealT* ders) \
    { \
      if (model.use_radial_direct && \
          direct_radial_vals_ders_lk_static<RealT, LVAL, KVAL>(model, pair, r, vals, ders)) { \
        return; \
      } \
      interp_radial_vals_ders(model, pair, r, vals, ders); \
    } \
  }

SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH(2, 3);
SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH(3, 3);
SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH(1, 1);
SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH(1, 2);
SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH(1, 3);
SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH(1, 4);
SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH(2, 1);
SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH(2, 2);
SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH(2, 4);
SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH(3, 1);
SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH(3, 2);
SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH(3, 4);
SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH(4, 1);
SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH(4, 2);
SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH(4, 3);
SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH(4, 4);

#undef SUS2_DEFINE_DIRECT_RADIAL_STATIC_DISPATCH

template <typename RealT, int L, int K>
__device__ __forceinline__ void interp_radial_vals_ders_lk_static(
  const SUS2DeviceModel& model,
  int pair,
  RealT r,
  RealT* vals,
  RealT* ders)
{
  Sus2DirectRadialStaticDispatch<L, K>::template eval<RealT>(model, pair, r, vals, ders);
}

template <typename RealT>
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
__device__ __forceinline__ void add_l3k3_basic_moments(
  int N,
  const SUS2DeviceModel& model,
  int atom,
  int pair,
  RealT dx,
  RealT dy,
  RealT dz,
  RealT r,
  RealT* moments)
{
  RealT mu_val[32];
  interp_radial_vals_ders_lk_static<RealT, 3, 3>(
    model, pair, r, mu_val, static_cast<RealT*>(nullptr));

  const RealT inv_r = static_cast<RealT>(1.0) / r;
  const RealT inv_r2 = inv_r * inv_r;
  const RealT inv_r3 = inv_r2 * inv_r;
  const RealT x2 = dx * dx;
  const RealT y2 = dy * dy;
  const RealT z2 = dz * dz;
  const RealT xy = dx * dy;
  const RealT xz = dx * dz;
  const RealT yz = dy * dz;

#define SUS2_ADD_L3K3_MOMENT(BASIC, SCALE, GEOM) \
  moments[static_cast<size_t>(BASIC) * N + atom] += (SCALE) * (GEOM)

  for (int group = 0; group < 3; ++group) {
    const int base = group * 20;
    const int mu = group * 4;
    const RealT s0 = mu_val[mu + 0];
    const RealT s1 = mu_val[mu + 1] * inv_r;
    const RealT s2 = mu_val[mu + 2] * inv_r2;
    const RealT s3 = mu_val[mu + 3] * inv_r3;

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

template <typename RealT>
__device__ __forceinline__ void store_sus2_self_force(
  int N,
  int atom,
  RealT fx,
  RealT fy,
  RealT fz,
  float* force_tmp,
  float* force_self_tmp)
{
  if (force_self_tmp != nullptr) {
    force_self_tmp[atom] = static_cast<float>(fx);
    force_self_tmp[atom + N] = static_cast<float>(fy);
    force_self_tmp[atom + 2 * N] = static_cast<float>(fz);
  } else {
    atomicAdd(force_tmp + atom, static_cast<float>(fx));
    atomicAdd(force_tmp + atom + N, static_cast<float>(fy));
    atomicAdd(force_tmp + atom + 2 * N, static_cast<float>(fz));
  }
}

template <typename GradT, typename RealT>
__device__ __forceinline__ void compute_sus2_edge_derivative_l3k3(
  int N,
  const SUS2DeviceModel& model,
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
  const RealT center_coeff = sus2_species_coeff<RealT>(model, center_type);

  RealT mu_val[32];
  RealT mu_der[32];
  interp_radial_vals_ders_lk_static<RealT, 3, 3>(model, pair, r, mu_val, mu_der);

  const RealT inv_r = static_cast<RealT>(1.0) / r;
  const RealT inv_r2 = inv_r * inv_r;
  const RealT inv_r3 = inv_r2 * inv_r;
  const RealT x2 = dx * dx;
  const RealT y2 = dy * dy;
  const RealT z2 = dz * dz;
  const RealT xy = dx * dy;
  const RealT xz = dx * dz;
  const RealT yz = dy * dz;

  dEx = 0.0;
  dEy = 0.0;
  dEz = 0.0;

#define SUS2_ACCUM_L3K3_DERIV(BASIC, GEOM, DGX, DGY, DGZ, INV_SCALED, RAD_COMMON) \
  do { \
    const RealT basic_grad = \
      static_cast<RealT>(load_sus2_grad(grads, N, BASIC, center_atom)) * center_coeff; \
    const RealT common = (GEOM) * (RAD_COMMON); \
    dEx += basic_grad * (common * dx + (INV_SCALED) * (DGX)); \
    dEy += basic_grad * (common * dy + (INV_SCALED) * (DGY)); \
    dEz += basic_grad * (common * dz + (INV_SCALED) * (DGZ)); \
  } while (0)

  for (int group = 0; group < 3; ++group) {
    const int base = group * 20;
    const int mu = group * 4;

    const RealT inv0 = mu_val[mu + 0];
    const RealT rc0 = mu_der[mu + 0] * inv_r;
    SUS2_ACCUM_L3K3_DERIV(base + 0, 1.0, 0.0, 0.0, 0.0, inv0, rc0);

    const RealT inv1 = mu_val[mu + 1] * inv_r;
    const RealT rc1 = (mu_der[mu + 1] * inv_r - inv1 * inv_r) * inv_r;
    SUS2_ACCUM_L3K3_DERIV(base + 1, dx, 1.0, 0.0, 0.0, inv1, rc1);
    SUS2_ACCUM_L3K3_DERIV(base + 2, dy, 0.0, 1.0, 0.0, inv1, rc1);
    SUS2_ACCUM_L3K3_DERIV(base + 3, dz, 0.0, 0.0, 1.0, inv1, rc1);

    const RealT inv2 = mu_val[mu + 2] * inv_r2;
    const RealT rc2 = (mu_der[mu + 2] * inv_r2 - static_cast<RealT>(2.0) * inv2 * inv_r) * inv_r;
    SUS2_ACCUM_L3K3_DERIV(base + 4, x2, 2.0 * dx, 0.0, 0.0, inv2, rc2);
    SUS2_ACCUM_L3K3_DERIV(base + 5, xy, dy, dx, 0.0, inv2, rc2);
    SUS2_ACCUM_L3K3_DERIV(base + 6, xz, dz, 0.0, dx, inv2, rc2);
    SUS2_ACCUM_L3K3_DERIV(base + 7, y2, 0.0, 2.0 * dy, 0.0, inv2, rc2);
    SUS2_ACCUM_L3K3_DERIV(base + 8, yz, 0.0, dz, dy, inv2, rc2);
    SUS2_ACCUM_L3K3_DERIV(base + 9, z2, 0.0, 0.0, 2.0 * dz, inv2, rc2);

    const RealT inv3 = mu_val[mu + 3] * inv_r3;
    const RealT rc3 = (mu_der[mu + 3] * inv_r3 - static_cast<RealT>(3.0) * inv3 * inv_r) * inv_r;
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

template <typename RealT>
__device__ __forceinline__ void compute_sus2_edge_derivative_l3k3_cached(
  const SUS2DeviceModel& model,
  int center_type,
  int neighbor_type,
  RealT dx,
  RealT dy,
  RealT dz,
  RealT r,
  const RealT* basic_grads,
  RealT& dEx,
  RealT& dEy,
  RealT& dEz)
{
  const int pair = center_type * model.species_count + neighbor_type;

  RealT mu_val[32];
  RealT mu_der[32];
  interp_radial_vals_ders_lk_static<RealT, 3, 3>(model, pair, r, mu_val, mu_der);

  const RealT inv_r = static_cast<RealT>(1.0) / r;
  const RealT inv_r2 = inv_r * inv_r;
  const RealT inv_r3 = inv_r2 * inv_r;
  const RealT x2 = dx * dx;
  const RealT y2 = dy * dy;
  const RealT z2 = dz * dz;
  const RealT xy = dx * dy;
  const RealT xz = dx * dz;
  const RealT yz = dy * dz;

  dEx = 0.0;
  dEy = 0.0;
  dEz = 0.0;

  for (int group = 0; group < 3; ++group) {
    const int base = group * 20;
    const int mu = group * 4;

    const RealT inv0 = mu_val[mu + 0];
    const RealT rc0 = mu_der[mu + 0] * inv_r;
    const RealT c0 = basic_grads[base + 0] * rc0;
    dEx += c0 * dx;
    dEy += c0 * dy;
    dEz += c0 * dz;

    const RealT inv1 = mu_val[mu + 1] * inv_r;
    const RealT rc1 = (mu_der[mu + 1] * inv_r - inv1 * inv_r) * inv_r;
    const RealT g1x = basic_grads[base + 1];
    const RealT g1y = basic_grads[base + 2];
    const RealT g1z = basic_grads[base + 3];
    const RealT p1 = g1x * dx + g1y * dy + g1z * dz;
    const RealT c1 = rc1 * p1;
    dEx += c1 * dx + inv1 * g1x;
    dEy += c1 * dy + inv1 * g1y;
    dEz += c1 * dz + inv1 * g1z;

    const RealT inv2 = mu_val[mu + 2] * inv_r2;
    const RealT rc2 = (mu_der[mu + 2] * inv_r2 - static_cast<RealT>(2.0) * inv2 * inv_r) * inv_r;
    const RealT g2xx = basic_grads[base + 4];
    const RealT g2xy = basic_grads[base + 5];
    const RealT g2xz = basic_grads[base + 6];
    const RealT g2yy = basic_grads[base + 7];
    const RealT g2yz = basic_grads[base + 8];
    const RealT g2zz = basic_grads[base + 9];
    const RealT p2 = g2xx * x2 + g2xy * xy + g2xz * xz + g2yy * y2 + g2yz * yz + g2zz * z2;
    const RealT p2x = static_cast<RealT>(2.0) * g2xx * dx + g2xy * dy + g2xz * dz;
    const RealT p2y = g2xy * dx + static_cast<RealT>(2.0) * g2yy * dy + g2yz * dz;
    const RealT p2z = g2xz * dx + g2yz * dy + static_cast<RealT>(2.0) * g2zz * dz;
    const RealT c2 = rc2 * p2;
    dEx += c2 * dx + inv2 * p2x;
    dEy += c2 * dy + inv2 * p2y;
    dEz += c2 * dz + inv2 * p2z;

    const RealT inv3 = mu_val[mu + 3] * inv_r3;
    const RealT rc3 = (mu_der[mu + 3] * inv_r3 - static_cast<RealT>(3.0) * inv3 * inv_r) * inv_r;
    const RealT g3xxx = basic_grads[base + 10];
    const RealT g3xxy = basic_grads[base + 11];
    const RealT g3xxz = basic_grads[base + 12];
    const RealT g3xyy = basic_grads[base + 13];
    const RealT g3xyz = basic_grads[base + 14];
    const RealT g3xzz = basic_grads[base + 15];
    const RealT g3yyy = basic_grads[base + 16];
    const RealT g3yyz = basic_grads[base + 17];
    const RealT g3yzz = basic_grads[base + 18];
    const RealT g3zzz = basic_grads[base + 19];
    const RealT p3 =
      g3xxx * x2 * dx + g3xxy * x2 * dy + g3xxz * x2 * dz + g3xyy * dx * y2 +
      g3xyz * xy * dz + g3xzz * dx * z2 + g3yyy * y2 * dy + g3yyz * y2 * dz +
      g3yzz * dy * z2 + g3zzz * z2 * dz;
    const RealT p3x =
      static_cast<RealT>(3.0) * g3xxx * x2 + static_cast<RealT>(2.0) * g3xxy * xy +
      static_cast<RealT>(2.0) * g3xxz * xz + g3xyy * y2 + g3xyz * yz + g3xzz * z2;
    const RealT p3y =
      g3xxy * x2 + static_cast<RealT>(2.0) * g3xyy * xy + g3xyz * xz +
      static_cast<RealT>(3.0) * g3yyy * y2 + static_cast<RealT>(2.0) * g3yyz * yz +
      g3yzz * z2;
    const RealT p3z =
      g3xxz * x2 + g3xyz * xy + static_cast<RealT>(2.0) * g3xzz * xz + g3yyz * y2 +
      static_cast<RealT>(2.0) * g3yzz * yz + static_cast<RealT>(3.0) * g3zzz * z2;
    const RealT c3 = rc3 * p3;
    dEx += c3 * dx + inv3 * p3x;
    dEy += c3 * dy + inv3 * p3y;
    dEz += c3 * dz + inv3 * p3z;
  }
}

template <typename RealT, int MaxBasic>
__device__ __forceinline__ void compute_sus2_edge_derivative_tensor_cached(
  const SUS2DeviceModel& model,
  int center_type,
  int neighbor_type,
  RealT dx,
  RealT dy,
  RealT dz,
  RealT r,
  const RealT* basic_grads,
  RealT& dEx,
  RealT& dEy,
  RealT& dEz)
{
  if (model.alpha_basic_count > MaxBasic) {
    dEx = static_cast<RealT>(0.0);
    dEy = static_cast<RealT>(0.0);
    dEz = static_cast<RealT>(0.0);
    return;
  }

  const int pair = center_type * model.species_count + neighbor_type;

  RealT mu_val[32];
  RealT mu_der[32];
  interp_radial_vals_ders(model, pair, r, mu_val, mu_der);

  const RealT inv_r = static_cast<RealT>(1.0) / r;
  const RealT inv_r2 = inv_r * inv_r;
  const RealT inv_r3 = inv_r2 * inv_r;
  const RealT inv_r4 = inv_r2 * inv_r2;
  const RealT x2 = dx * dx;
  const RealT y2 = dy * dy;
  const RealT z2 = dz * dz;
  const RealT xy = dx * dy;
  const RealT xz = dx * dz;
  const RealT yz = dy * dz;

  dEx = static_cast<RealT>(0.0);
  dEy = static_cast<RealT>(0.0);
  dEz = static_cast<RealT>(0.0);

  for (int group = 0; group < model.tensor_k; ++group) {
    const int base = group * model.tensor_basic_per_group;
    const int mu_base = group * (model.tensor_l + 1);

    const RealT rc0 = mu_der[mu_base + 0] * inv_r;
    const RealT c0 = basic_grads[base + 0] * rc0;
    dEx += c0 * dx;
    dEy += c0 * dy;
    dEz += c0 * dz;

    if (model.tensor_l >= 1) {
      const RealT inv1 = mu_val[mu_base + 1] * inv_r;
      const RealT rc1 = (mu_der[mu_base + 1] * inv_r - inv1 * inv_r) * inv_r;
      const RealT g1x = basic_grads[base + 1];
      const RealT g1y = basic_grads[base + 2];
      const RealT g1z = basic_grads[base + 3];
      const RealT p1 = g1x * dx + g1y * dy + g1z * dz;
      const RealT c1 = rc1 * p1;
      dEx += c1 * dx + inv1 * g1x;
      dEy += c1 * dy + inv1 * g1y;
      dEz += c1 * dz + inv1 * g1z;
    }

    if (model.tensor_l >= 2) {
      const RealT inv2 = mu_val[mu_base + 2] * inv_r2;
      const RealT rc2 =
        (mu_der[mu_base + 2] * inv_r2 - static_cast<RealT>(2.0) * inv2 * inv_r) * inv_r;
      const RealT g2xx = basic_grads[base + 4];
      const RealT g2xy = basic_grads[base + 5];
      const RealT g2xz = basic_grads[base + 6];
      const RealT g2yy = basic_grads[base + 7];
      const RealT g2yz = basic_grads[base + 8];
      const RealT g2zz = basic_grads[base + 9];
      const RealT p2 =
        g2xx * x2 + g2xy * xy + g2xz * xz + g2yy * y2 + g2yz * yz + g2zz * z2;
      const RealT p2x = static_cast<RealT>(2.0) * g2xx * dx + g2xy * dy + g2xz * dz;
      const RealT p2y = g2xy * dx + static_cast<RealT>(2.0) * g2yy * dy + g2yz * dz;
      const RealT p2z = g2xz * dx + g2yz * dy + static_cast<RealT>(2.0) * g2zz * dz;
      const RealT c2 = rc2 * p2;
      dEx += c2 * dx + inv2 * p2x;
      dEy += c2 * dy + inv2 * p2y;
      dEz += c2 * dz + inv2 * p2z;
    }

    if (model.tensor_l >= 3) {
      const RealT inv3 = mu_val[mu_base + 3] * inv_r3;
      const RealT rc3 =
        (mu_der[mu_base + 3] * inv_r3 - static_cast<RealT>(3.0) * inv3 * inv_r) * inv_r;
      const RealT g3xxx = basic_grads[base + 10];
      const RealT g3xxy = basic_grads[base + 11];
      const RealT g3xxz = basic_grads[base + 12];
      const RealT g3xyy = basic_grads[base + 13];
      const RealT g3xyz = basic_grads[base + 14];
      const RealT g3xzz = basic_grads[base + 15];
      const RealT g3yyy = basic_grads[base + 16];
      const RealT g3yyz = basic_grads[base + 17];
      const RealT g3yzz = basic_grads[base + 18];
      const RealT g3zzz = basic_grads[base + 19];
      const RealT p3 =
        g3xxx * x2 * dx + g3xxy * x2 * dy + g3xxz * x2 * dz + g3xyy * dx * y2 +
        g3xyz * xy * dz + g3xzz * dx * z2 + g3yyy * y2 * dy + g3yyz * y2 * dz +
        g3yzz * dy * z2 + g3zzz * z2 * dz;
      const RealT p3x =
        static_cast<RealT>(3.0) * g3xxx * x2 + static_cast<RealT>(2.0) * g3xxy * xy +
        static_cast<RealT>(2.0) * g3xxz * xz + g3xyy * y2 + g3xyz * yz + g3xzz * z2;
      const RealT p3y =
        g3xxy * x2 + static_cast<RealT>(2.0) * g3xyy * xy + g3xyz * xz +
        static_cast<RealT>(3.0) * g3yyy * y2 + static_cast<RealT>(2.0) * g3yyz * yz +
        g3yzz * z2;
      const RealT p3z =
        g3xxz * x2 + g3xyz * xy + static_cast<RealT>(2.0) * g3xzz * xz + g3yyz * y2 +
        static_cast<RealT>(2.0) * g3yzz * yz + static_cast<RealT>(3.0) * g3zzz * z2;
      const RealT c3 = rc3 * p3;
      dEx += c3 * dx + inv3 * p3x;
      dEy += c3 * dy + inv3 * p3y;
      dEz += c3 * dz + inv3 * p3z;
    }

    if (model.tensor_l >= 4) {
      const RealT inv4 = mu_val[mu_base + 4] * inv_r4;
      const RealT rc4 =
        (mu_der[mu_base + 4] * inv_r4 - static_cast<RealT>(4.0) * inv4 * inv_r) * inv_r;
      const RealT g4xxxx = basic_grads[base + 20];
      const RealT g4xxxy = basic_grads[base + 21];
      const RealT g4xxxz = basic_grads[base + 22];
      const RealT g4xxyy = basic_grads[base + 23];
      const RealT g4xxyz = basic_grads[base + 24];
      const RealT g4xxzz = basic_grads[base + 25];
      const RealT g4xyyy = basic_grads[base + 26];
      const RealT g4xyyz = basic_grads[base + 27];
      const RealT g4xyzz = basic_grads[base + 28];
      const RealT g4xzzz = basic_grads[base + 29];
      const RealT g4yyyy = basic_grads[base + 30];
      const RealT g4yyyz = basic_grads[base + 31];
      const RealT g4yyzz = basic_grads[base + 32];
      const RealT g4yzzz = basic_grads[base + 33];
      const RealT g4zzzz = basic_grads[base + 34];
      const RealT p4 =
        g4xxxx * x2 * x2 + g4xxxy * x2 * dx * dy + g4xxxz * x2 * dx * dz +
        g4xxyy * x2 * y2 + g4xxyz * x2 * yz + g4xxzz * x2 * z2 +
        g4xyyy * dx * y2 * dy + g4xyyz * dx * y2 * dz + g4xyzz * dx * dy * z2 +
        g4xzzz * dx * z2 * dz + g4yyyy * y2 * y2 + g4yyyz * y2 * dy * dz +
        g4yyzz * y2 * z2 + g4yzzz * dy * z2 * dz + g4zzzz * z2 * z2;
      const RealT p4x =
        static_cast<RealT>(4.0) * g4xxxx * x2 * dx +
        static_cast<RealT>(3.0) * g4xxxy * x2 * dy +
        static_cast<RealT>(3.0) * g4xxxz * x2 * dz +
        static_cast<RealT>(2.0) * g4xxyy * dx * y2 +
        static_cast<RealT>(2.0) * g4xxyz * dx * yz +
        static_cast<RealT>(2.0) * g4xxzz * dx * z2 +
        g4xyyy * y2 * dy + g4xyyz * y2 * dz + g4xyzz * dy * z2 +
        g4xzzz * z2 * dz;
      const RealT p4y =
        g4xxxy * x2 * dx + static_cast<RealT>(2.0) * g4xxyy * x2 * dy +
        g4xxyz * x2 * dz + static_cast<RealT>(3.0) * g4xyyy * dx * y2 +
        static_cast<RealT>(2.0) * g4xyyz * dx * yz + g4xyzz * dx * z2 +
        static_cast<RealT>(4.0) * g4yyyy * y2 * dy +
        static_cast<RealT>(3.0) * g4yyyz * y2 * dz +
        static_cast<RealT>(2.0) * g4yyzz * dy * z2 + g4yzzz * z2 * dz;
      const RealT p4z =
        g4xxxz * x2 * dx + g4xxyz * x2 * dy +
        static_cast<RealT>(2.0) * g4xxzz * x2 * dz + g4xyyz * dx * y2 +
        static_cast<RealT>(2.0) * g4xyzz * dx * yz +
        static_cast<RealT>(3.0) * g4xzzz * dx * z2 + g4yyyz * y2 * dy +
        static_cast<RealT>(2.0) * g4yyzz * y2 * dz +
        static_cast<RealT>(3.0) * g4yzzz * dy * z2 +
        static_cast<RealT>(4.0) * g4zzzz * z2 * dz;
      const RealT c4 = rc4 * p4;
      dEx += c4 * dx + inv4 * p4x;
      dEy += c4 * dy + inv4 * p4y;
      dEz += c4 * dz + inv4 * p4z;
    }
  }
}

template <typename RealT, int L, int K>
__device__ __forceinline__ void compute_sus2_edge_derivative_tensor_cached_static(
  const SUS2DeviceModel& model,
  int center_type,
  int neighbor_type,
  RealT dx,
  RealT dy,
  RealT dz,
  RealT r,
  const RealT* basic_grads,
  RealT& dEx,
  RealT& dEy,
  RealT& dEz)
{
  constexpr int BasicPerGroup = Sus2TensorStaticLayout<L>::basic_per_group;
  constexpr int BasicCount = K * BasicPerGroup;
  if (model.alpha_basic_count != BasicCount) {
    dEx = static_cast<RealT>(0.0);
    dEy = static_cast<RealT>(0.0);
    dEz = static_cast<RealT>(0.0);
    return;
  }

  const int pair = center_type * model.species_count + neighbor_type;

  RealT mu_val[32];
  RealT mu_der[32];
  interp_radial_vals_ders_lk_static<RealT, L, K>(model, pair, r, mu_val, mu_der);

  const RealT inv_r = static_cast<RealT>(1.0) / r;
  const RealT inv_r2 = inv_r * inv_r;
  const RealT inv_r3 = inv_r2 * inv_r;
  const RealT inv_r4 = inv_r2 * inv_r2;
  const RealT x2 = dx * dx;
  const RealT y2 = dy * dy;
  const RealT z2 = dz * dz;
  const RealT xy = dx * dy;
  const RealT xz = dx * dz;
  const RealT yz = dy * dz;

  dEx = static_cast<RealT>(0.0);
  dEy = static_cast<RealT>(0.0);
  dEz = static_cast<RealT>(0.0);

#pragma unroll
  for (int group = 0; group < K; ++group) {
    const int base = group * BasicPerGroup;
    const int mu_base = group * (L + 1);

    const RealT rc0 = mu_der[mu_base + 0] * inv_r;
    const RealT c0 = basic_grads[base + 0] * rc0;
    dEx += c0 * dx;
    dEy += c0 * dy;
    dEz += c0 * dz;

    if (L >= 1) {
      const RealT inv1 = mu_val[mu_base + 1] * inv_r;
      const RealT rc1 = (mu_der[mu_base + 1] * inv_r - inv1 * inv_r) * inv_r;
      const RealT g1x = basic_grads[base + 1];
      const RealT g1y = basic_grads[base + 2];
      const RealT g1z = basic_grads[base + 3];
      const RealT p1 = g1x * dx + g1y * dy + g1z * dz;
      const RealT c1 = rc1 * p1;
      dEx += c1 * dx + inv1 * g1x;
      dEy += c1 * dy + inv1 * g1y;
      dEz += c1 * dz + inv1 * g1z;
    }

    if (L >= 2) {
      const RealT inv2 = mu_val[mu_base + 2] * inv_r2;
      const RealT rc2 =
        (mu_der[mu_base + 2] * inv_r2 - static_cast<RealT>(2.0) * inv2 * inv_r) * inv_r;
      const RealT g2xx = basic_grads[base + 4];
      const RealT g2xy = basic_grads[base + 5];
      const RealT g2xz = basic_grads[base + 6];
      const RealT g2yy = basic_grads[base + 7];
      const RealT g2yz = basic_grads[base + 8];
      const RealT g2zz = basic_grads[base + 9];
      const RealT p2 =
        g2xx * x2 + g2xy * xy + g2xz * xz + g2yy * y2 + g2yz * yz + g2zz * z2;
      const RealT p2x = static_cast<RealT>(2.0) * g2xx * dx + g2xy * dy + g2xz * dz;
      const RealT p2y = g2xy * dx + static_cast<RealT>(2.0) * g2yy * dy + g2yz * dz;
      const RealT p2z = g2xz * dx + g2yz * dy + static_cast<RealT>(2.0) * g2zz * dz;
      const RealT c2 = rc2 * p2;
      dEx += c2 * dx + inv2 * p2x;
      dEy += c2 * dy + inv2 * p2y;
      dEz += c2 * dz + inv2 * p2z;
    }

    if (L >= 3) {
      const RealT inv3 = mu_val[mu_base + 3] * inv_r3;
      const RealT rc3 =
        (mu_der[mu_base + 3] * inv_r3 - static_cast<RealT>(3.0) * inv3 * inv_r) * inv_r;
      const RealT g3xxx = basic_grads[base + 10];
      const RealT g3xxy = basic_grads[base + 11];
      const RealT g3xxz = basic_grads[base + 12];
      const RealT g3xyy = basic_grads[base + 13];
      const RealT g3xyz = basic_grads[base + 14];
      const RealT g3xzz = basic_grads[base + 15];
      const RealT g3yyy = basic_grads[base + 16];
      const RealT g3yyz = basic_grads[base + 17];
      const RealT g3yzz = basic_grads[base + 18];
      const RealT g3zzz = basic_grads[base + 19];
      const RealT p3 =
        g3xxx * x2 * dx + g3xxy * x2 * dy + g3xxz * x2 * dz + g3xyy * dx * y2 +
        g3xyz * xy * dz + g3xzz * dx * z2 + g3yyy * y2 * dy + g3yyz * y2 * dz +
        g3yzz * dy * z2 + g3zzz * z2 * dz;
      const RealT p3x =
        static_cast<RealT>(3.0) * g3xxx * x2 + static_cast<RealT>(2.0) * g3xxy * xy +
        static_cast<RealT>(2.0) * g3xxz * xz + g3xyy * y2 + g3xyz * yz + g3xzz * z2;
      const RealT p3y =
        g3xxy * x2 + static_cast<RealT>(2.0) * g3xyy * xy + g3xyz * xz +
        static_cast<RealT>(3.0) * g3yyy * y2 + static_cast<RealT>(2.0) * g3yyz * yz +
        g3yzz * z2;
      const RealT p3z =
        g3xxz * x2 + g3xyz * xy + static_cast<RealT>(2.0) * g3xzz * xz + g3yyz * y2 +
        static_cast<RealT>(2.0) * g3yzz * yz + static_cast<RealT>(3.0) * g3zzz * z2;
      const RealT c3 = rc3 * p3;
      dEx += c3 * dx + inv3 * p3x;
      dEy += c3 * dy + inv3 * p3y;
      dEz += c3 * dz + inv3 * p3z;
    }

    if (L >= 4) {
      const RealT inv4 = mu_val[mu_base + 4] * inv_r4;
      const RealT rc4 =
        (mu_der[mu_base + 4] * inv_r4 - static_cast<RealT>(4.0) * inv4 * inv_r) * inv_r;
      const RealT g4xxxx = basic_grads[base + 20];
      const RealT g4xxxy = basic_grads[base + 21];
      const RealT g4xxxz = basic_grads[base + 22];
      const RealT g4xxyy = basic_grads[base + 23];
      const RealT g4xxyz = basic_grads[base + 24];
      const RealT g4xxzz = basic_grads[base + 25];
      const RealT g4xyyy = basic_grads[base + 26];
      const RealT g4xyyz = basic_grads[base + 27];
      const RealT g4xyzz = basic_grads[base + 28];
      const RealT g4xzzz = basic_grads[base + 29];
      const RealT g4yyyy = basic_grads[base + 30];
      const RealT g4yyyz = basic_grads[base + 31];
      const RealT g4yyzz = basic_grads[base + 32];
      const RealT g4yzzz = basic_grads[base + 33];
      const RealT g4zzzz = basic_grads[base + 34];
      const RealT p4 =
        g4xxxx * x2 * x2 + g4xxxy * x2 * dx * dy + g4xxxz * x2 * dx * dz +
        g4xxyy * x2 * y2 + g4xxyz * x2 * yz + g4xxzz * x2 * z2 +
        g4xyyy * dx * y2 * dy + g4xyyz * dx * y2 * dz + g4xyzz * dx * dy * z2 +
        g4xzzz * dx * z2 * dz + g4yyyy * y2 * y2 + g4yyyz * y2 * dy * dz +
        g4yyzz * y2 * z2 + g4yzzz * dy * z2 * dz + g4zzzz * z2 * z2;
      const RealT p4x =
        static_cast<RealT>(4.0) * g4xxxx * x2 * dx +
        static_cast<RealT>(3.0) * g4xxxy * x2 * dy +
        static_cast<RealT>(3.0) * g4xxxz * x2 * dz +
        static_cast<RealT>(2.0) * g4xxyy * dx * y2 +
        static_cast<RealT>(2.0) * g4xxyz * dx * yz +
        static_cast<RealT>(2.0) * g4xxzz * dx * z2 +
        g4xyyy * y2 * dy + g4xyyz * y2 * dz + g4xyzz * dy * z2 +
        g4xzzz * z2 * dz;
      const RealT p4y =
        g4xxxy * x2 * dx + static_cast<RealT>(2.0) * g4xxyy * x2 * dy +
        g4xxyz * x2 * dz + static_cast<RealT>(3.0) * g4xyyy * dx * y2 +
        static_cast<RealT>(2.0) * g4xyyz * dx * yz + g4xyzz * dx * z2 +
        static_cast<RealT>(4.0) * g4yyyy * y2 * dy +
        static_cast<RealT>(3.0) * g4yyyz * y2 * dz +
        static_cast<RealT>(2.0) * g4yyzz * dy * z2 + g4yzzz * z2 * dz;
      const RealT p4z =
        g4xxxz * x2 * dx + g4xxyz * x2 * dy +
        static_cast<RealT>(2.0) * g4xxzz * x2 * dz + g4xyyz * dx * y2 +
        static_cast<RealT>(2.0) * g4xyzz * dx * yz +
        static_cast<RealT>(3.0) * g4xzzz * dx * z2 + g4yyyz * y2 * dy +
        static_cast<RealT>(2.0) * g4yyzz * y2 * dz +
        static_cast<RealT>(3.0) * g4yzzz * dy * z2 +
        static_cast<RealT>(4.0) * g4zzzz * z2 * dz;
      const RealT c4 = rc4 * p4;
      dEx += c4 * dx + inv4 * p4x;
      dEy += c4 * dy + inv4 * p4y;
      dEz += c4 * dz + inv4 * p4z;
    }
  }
}

template <typename GradT, typename RealT>
__device__ __forceinline__ void compute_sus2_edge_derivative(
  int N,
  const SUS2DeviceModel& model,
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
  if (model.use_tensor_basic_fastpath && model.tensor_l == 3 && model.tensor_k == 3) {
    compute_sus2_edge_derivative_l3k3<GradT, RealT>(
      N, model, center_atom, center_type, neighbor_type, dx, dy, dz, r, grads, dEx, dEy, dEz);
    return;
  }

  const int pair = center_type * model.species_count + neighbor_type;
  const RealT center_coeff = sus2_species_coeff<RealT>(model, center_type);

  RealT mu_val[32];
  RealT mu_der[32];
  interp_radial_vals_ders(model, pair, r, mu_val, mu_der);

  RealT dist_pow[8];
  RealT x_pow[8];
  RealT y_pow[8];
  RealT z_pow[8];
  dist_pow[0] = static_cast<RealT>(1.0);
  x_pow[0] = static_cast<RealT>(1.0);
  y_pow[0] = static_cast<RealT>(1.0);
  z_pow[0] = static_cast<RealT>(1.0);
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
    const RealT inv_dist_pow = mu_val[mu] / dist_pow[rank];
    const RealT geom = x_pow[a] * y_pow[b] * z_pow[c];
    const RealT radial_der =
      mu_der[mu] / dist_pow[rank] - static_cast<RealT>(rank) * inv_dist_pow / r;
    const RealT common = geom * radial_der / r;
    RealT jac_x = common * dx;
    RealT jac_y = common * dy;
    RealT jac_z = common * dz;
    if (a != 0) {
      jac_x += inv_dist_pow * static_cast<RealT>(a) * x_pow[a - 1] * y_pow[b] * z_pow[c];
    }
    if (b != 0) {
      jac_y += inv_dist_pow * static_cast<RealT>(b) * x_pow[a] * y_pow[b - 1] * z_pow[c];
    }
    if (c != 0) {
      jac_z += inv_dist_pow * static_cast<RealT>(c) * x_pow[a] * y_pow[b] * z_pow[c - 1];
    }
    const RealT basic_grad =
      static_cast<RealT>(load_sus2_grad(grads, N, basic, center_atom)) * center_coeff;
    dEx += basic_grad * jac_x;
    dEy += basic_grad * jac_y;
    dEz += basic_grad * jac_z;
  }
}

template <typename RealT>
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
  RealT* moments)
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
    RealT dx;
    RealT dy;
    RealT dz;
    load_sus2_edge_displacement(
      use_cached_displacements, N, box, edge, i, j, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, dx, dy, dz);
    const RealT r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= static_cast<RealT>(cutoff_square)) {
      continue;
    }
    const RealT r = sqrt(r2);
    const int type_j = type[j];
    const int pair = type_i * model.species_count + type_j;

    if (model.use_tensor_basic_fastpath && model.tensor_l == 3 && model.tensor_k == 3) {
      add_l3k3_basic_moments(N, model, i, pair, dx, dy, dz, r, moments);
      continue;
    }

    RealT mu_val[32];
    interp_radial_vals_ders_lk_static<RealT, 3, 3>(
      model, pair, r, mu_val, static_cast<RealT*>(nullptr));

    RealT dist_pow[8];
    RealT x_pow[8];
    RealT y_pow[8];
    RealT z_pow[8];
    dist_pow[0] = static_cast<RealT>(1.0);
    x_pow[0] = static_cast<RealT>(1.0);
    y_pow[0] = static_cast<RealT>(1.0);
    z_pow[0] = static_cast<RealT>(1.0);
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
      const RealT geom = x_pow[a] * y_pow[b] * z_pow[c];
      moments[static_cast<size_t>(basic) * N + i] += (mu_val[mu] / dist_pow[rank]) * geom;
    }
  }
}

template <typename RealT>
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
  RealT* moments)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  RealT acc[60];
#pragma unroll
  for (int k = 0; k < 60; ++k) {
    acc[k] = static_cast<RealT>(0.0);
  }

  const int type_i = type[i];
  const int count = neighbor_count[i];

  for (int nbr = 0; nbr < count; ++nbr) {
    const size_t edge = static_cast<size_t>(nbr) * N + i;
    const int j = neighbor_atoms[edge];
    RealT dx;
    RealT dy;
    RealT dz;
    load_sus2_edge_displacement(
      use_cached_displacements, N, box, edge, i, j, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, dx, dy, dz);
    const RealT r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= static_cast<RealT>(cutoff_square)) {
      continue;
    }
    const RealT r = sqrt(r2);
    const int pair = type_i * model.species_count + type[j];

    RealT mu_val[32];
    interp_radial_vals_ders_lk_static<RealT, 3, 3>(
      model, pair, r, mu_val, static_cast<RealT*>(nullptr));

    const RealT inv_r = static_cast<RealT>(1.0) / r;
    const RealT inv_r2 = inv_r * inv_r;
    const RealT inv_r3 = inv_r2 * inv_r;
    const RealT x2 = dx * dx;
    const RealT y2 = dy * dy;
    const RealT z2 = dz * dz;
    const RealT xy = dx * dy;
    const RealT xz = dx * dz;
    const RealT yz = dy * dz;

#pragma unroll
    for (int group = 0; group < 3; ++group) {
      const int base = group * 20;
      const int mu = group * 4;
      const RealT s0 = mu_val[mu + 0];
      const RealT s1 = mu_val[mu + 1] * inv_r;
      const RealT s2 = mu_val[mu + 2] * inv_r2;
      const RealT s3 = mu_val[mu + 3] * inv_r3;

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

template <typename RealT, int MaxBasic>
static __global__ void gpu_compute_basic_moments_tensor_accum(
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
  RealT* moments)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || model.alpha_basic_count > MaxBasic) {
    return;
  }

  RealT acc[MaxBasic];
  for (int basic = 0; basic < MaxBasic; ++basic) {
    acc[basic] = static_cast<RealT>(0.0);
  }

  const int type_i = type[i];
  const int count = neighbor_count[i];

  for (int nbr = 0; nbr < count; ++nbr) {
    const size_t edge = static_cast<size_t>(nbr) * N + i;
    const int j = neighbor_atoms[edge];
    RealT dx;
    RealT dy;
    RealT dz;
    load_sus2_edge_displacement(
      use_cached_displacements, N, box, edge, i, j, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, dx, dy, dz);
    const RealT r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= static_cast<RealT>(cutoff_square)) {
      continue;
    }
    const RealT r = sqrt(r2);
    const int pair = type_i * model.species_count + type[j];

    RealT mu_val[32];
    interp_radial_vals_ders(model, pair, r, mu_val, static_cast<RealT*>(nullptr));

    const RealT inv_r = static_cast<RealT>(1.0) / r;
    const RealT inv_r2 = inv_r * inv_r;
    const RealT inv_r3 = inv_r2 * inv_r;
    const RealT inv_r4 = inv_r2 * inv_r2;
    const RealT x2 = dx * dx;
    const RealT y2 = dy * dy;
    const RealT z2 = dz * dz;
    const RealT xy = dx * dy;
    const RealT xz = dx * dz;
    const RealT yz = dy * dz;

    for (int group = 0; group < model.tensor_k; ++group) {
      const int base = group * model.tensor_basic_per_group;
      const int mu_base = group * (model.tensor_l + 1);
      acc[base + 0] += mu_val[mu_base + 0];

      if (model.tensor_l >= 1) {
        const RealT s1 = mu_val[mu_base + 1] * inv_r;
        acc[base + 1] += s1 * dx;
        acc[base + 2] += s1 * dy;
        acc[base + 3] += s1 * dz;
      }

      if (model.tensor_l >= 2) {
        const RealT s2 = mu_val[mu_base + 2] * inv_r2;
        acc[base + 4] += s2 * x2;
        acc[base + 5] += s2 * xy;
        acc[base + 6] += s2 * xz;
        acc[base + 7] += s2 * y2;
        acc[base + 8] += s2 * yz;
        acc[base + 9] += s2 * z2;
      }

      if (model.tensor_l >= 3) {
        const RealT s3 = mu_val[mu_base + 3] * inv_r3;
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

      if (model.tensor_l >= 4) {
        const RealT s4 = mu_val[mu_base + 4] * inv_r4;
        acc[base + 20] += s4 * x2 * x2;
        acc[base + 21] += s4 * x2 * dx * dy;
        acc[base + 22] += s4 * x2 * dx * dz;
        acc[base + 23] += s4 * x2 * y2;
        acc[base + 24] += s4 * x2 * yz;
        acc[base + 25] += s4 * x2 * z2;
        acc[base + 26] += s4 * dx * y2 * dy;
        acc[base + 27] += s4 * dx * y2 * dz;
        acc[base + 28] += s4 * dx * dy * z2;
        acc[base + 29] += s4 * dx * z2 * dz;
        acc[base + 30] += s4 * y2 * y2;
        acc[base + 31] += s4 * y2 * dy * dz;
        acc[base + 32] += s4 * y2 * z2;
        acc[base + 33] += s4 * dy * z2 * dz;
        acc[base + 34] += s4 * z2 * z2;
      }
    }
  }

  for (int basic = 0; basic < model.alpha_basic_count; ++basic) {
    moments[static_cast<size_t>(basic) * N + i] = acc[basic];
  }
}

template <typename RealT, int L, int K>
static __global__ void gpu_compute_basic_moments_tensor_accum_static(
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
  RealT* moments)
{
  constexpr int BasicPerGroup = Sus2TensorStaticLayout<L>::basic_per_group;
  constexpr int BasicCount = K * BasicPerGroup;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || model.alpha_basic_count != BasicCount) {
    return;
  }

  RealT acc[BasicCount];
#pragma unroll
  for (int basic = 0; basic < BasicCount; ++basic) {
    acc[basic] = static_cast<RealT>(0.0);
  }

  const int type_i = type[i];
  const int count = neighbor_count[i];

  for (int nbr = 0; nbr < count; ++nbr) {
    const size_t edge = static_cast<size_t>(nbr) * N + i;
    const int j = neighbor_atoms[edge];
    RealT dx;
    RealT dy;
    RealT dz;
    load_sus2_edge_displacement(
      use_cached_displacements, N, box, edge, i, j, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, dx, dy, dz);
    const RealT r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= static_cast<RealT>(cutoff_square)) {
      continue;
    }
    const RealT r = sqrt(r2);
    const int pair = type_i * model.species_count + type[j];

    RealT mu_val[32];
    interp_radial_vals_ders_lk_static<RealT, L, K>(
      model, pair, r, mu_val, static_cast<RealT*>(nullptr));

    const RealT inv_r = static_cast<RealT>(1.0) / r;
    const RealT inv_r2 = inv_r * inv_r;
    const RealT inv_r3 = inv_r2 * inv_r;
    const RealT inv_r4 = inv_r2 * inv_r2;
    const RealT x2 = dx * dx;
    const RealT y2 = dy * dy;
    const RealT z2 = dz * dz;
    const RealT xy = dx * dy;
    const RealT yz = dy * dz;

#pragma unroll
    for (int group = 0; group < K; ++group) {
      const int base = group * BasicPerGroup;
      const int mu_base = group * (L + 1);
      acc[base + 0] += mu_val[mu_base + 0];

      if (L >= 1) {
        const RealT s1 = mu_val[mu_base + 1] * inv_r;
        acc[base + 1] += s1 * dx;
        acc[base + 2] += s1 * dy;
        acc[base + 3] += s1 * dz;
      }

      if (L >= 2) {
        const RealT s2 = mu_val[mu_base + 2] * inv_r2;
        acc[base + 4] += s2 * x2;
        acc[base + 5] += s2 * xy;
        acc[base + 6] += s2 * dx * dz;
        acc[base + 7] += s2 * y2;
        acc[base + 8] += s2 * yz;
        acc[base + 9] += s2 * z2;
      }

      if (L >= 3) {
        const RealT s3 = mu_val[mu_base + 3] * inv_r3;
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

      if (L >= 4) {
        const RealT s4 = mu_val[mu_base + 4] * inv_r4;
        acc[base + 20] += s4 * x2 * x2;
        acc[base + 21] += s4 * x2 * dx * dy;
        acc[base + 22] += s4 * x2 * dx * dz;
        acc[base + 23] += s4 * x2 * y2;
        acc[base + 24] += s4 * x2 * yz;
        acc[base + 25] += s4 * x2 * z2;
        acc[base + 26] += s4 * dx * y2 * dy;
        acc[base + 27] += s4 * dx * y2 * dz;
        acc[base + 28] += s4 * dx * dy * z2;
        acc[base + 29] += s4 * dx * z2 * dz;
        acc[base + 30] += s4 * y2 * y2;
        acc[base + 31] += s4 * y2 * dy * dz;
        acc[base + 32] += s4 * y2 * z2;
        acc[base + 33] += s4 * dy * z2 * dz;
        acc[base + 34] += s4 * z2 * z2;
      }
    }
  }

#pragma unroll
  for (int basic = 0; basic < BasicCount; ++basic) {
    moments[static_cast<size_t>(basic) * N + i] = acc[basic];
  }
}

template <typename RealT>
static __global__ void gpu_forward_times(int N, SUS2DeviceModel model, RealT* moments)
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
      static_cast<RealT>(mult) * moments[static_cast<size_t>(src0) * N + i] *
      moments[static_cast<size_t>(src1) * N + i];
  }
}

template <typename RealT>
static __global__ void gpu_forward_times_const_u16(int N, SUS2DeviceModel model, RealT* moments)
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
      static_cast<RealT>(mult) * moments[static_cast<size_t>(src0) * N + i] *
      moments[static_cast<size_t>(src1) * N + i];
  }
}

template <typename RealT, typename GradT>
__device__ __forceinline__ void sus2_init_site_energy_and_scalar_grads(
  int N,
  SUS2DeviceModel model,
  int i,
  const int* type,
  const RealT* moments,
  GradT* grads,
  double* potential);

template <typename RealT, typename GradT>
static __global__ void gpu_forward_energy_backward(
  int N,
  SUS2DeviceModel model,
  const int* type,
  RealT* moments,
  GradT* grads,
  double* potential)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }
  for (int t = 0; t < model.alpha_times_count; ++t) {
    const int src0 = model.alpha_times[t * 4 + 0];
    const int src1 = model.alpha_times[t * 4 + 1];
    const int mult = model.alpha_times[t * 4 + 2];
    const int dst = model.alpha_times[t * 4 + 3];
    moments[static_cast<size_t>(dst) * N + i] +=
      static_cast<RealT>(mult) * moments[static_cast<size_t>(src0) * N + i] *
      moments[static_cast<size_t>(src1) * N + i];
  }
  sus2_init_site_energy_and_scalar_grads(N, model, i, type, moments, grads, potential);
  for (int t = model.alpha_times_count - 1; t >= 0; --t) {
    const int src0 = model.alpha_times[t * 4 + 0];
    const int src1 = model.alpha_times[t * 4 + 1];
    const int mult = model.alpha_times[t * 4 + 2];
    const int dst = model.alpha_times[t * 4 + 3];
    const RealT gdst =
      static_cast<RealT>(load_sus2_grad(grads, N, dst, i)) * static_cast<RealT>(mult);
    add_sus2_grad(
      grads, static_cast<size_t>(src1) * N + i, gdst * moments[static_cast<size_t>(src0) * N + i]);
    add_sus2_grad(
      grads, static_cast<size_t>(src0) * N + i, gdst * moments[static_cast<size_t>(src1) * N + i]);
  }
}

template <typename RealT, typename GradT>
static __global__ void gpu_forward_energy_backward_const_u16(
  int N,
  SUS2DeviceModel model,
  const int* type,
  RealT* moments,
  GradT* grads,
  double* potential)
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
      static_cast<RealT>(mult) * moments[static_cast<size_t>(src0) * N + i] *
      moments[static_cast<size_t>(src1) * N + i];
  }
  sus2_init_site_energy_and_scalar_grads(N, model, i, type, moments, grads, potential);
  for (int t = model.alpha_times_count - 1; t >= 0; --t) {
    const int offset = t * 4;
    const int src0 = static_cast<int>(c_sus2_alpha_times_u16[offset + 0]);
    const int src1 = static_cast<int>(c_sus2_alpha_times_u16[offset + 1]);
    const int mult = static_cast<int>(c_sus2_alpha_times_u16[offset + 2]);
    const int dst = static_cast<int>(c_sus2_alpha_times_u16[offset + 3]);
    const RealT gdst =
      static_cast<RealT>(load_sus2_grad(grads, N, dst, i)) * static_cast<RealT>(mult);
    add_sus2_grad(
      grads, static_cast<size_t>(src1) * N + i, gdst * moments[static_cast<size_t>(src0) * N + i]);
    add_sus2_grad(
      grads, static_cast<size_t>(src0) * N + i, gdst * moments[static_cast<size_t>(src1) * N + i]);
  }
}

template <typename RealT, typename GradT, int MaxMoments>
static __global__ void gpu_local_graph_energy_backward_to_basic(
  int N,
  SUS2DeviceModel model,
  const int* type,
  const RealT* basic_moments,
  GradT* basic_grads,
  double* potential)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || model.alpha_moments_count > MaxMoments) {
    return;
  }

  RealT vals[MaxMoments];
  GradT grads_local[MaxMoments];
  for (int m = 0; m < model.alpha_moments_count; ++m) {
    vals[m] = static_cast<RealT>(0.0);
    grads_local[m] = static_cast<GradT>(0.0);
  }
  for (int basic = 0; basic < model.alpha_basic_count; ++basic) {
    vals[basic] = basic_moments[static_cast<size_t>(basic) * N + i];
  }

  for (int t = 0; t < model.alpha_times_count; ++t) {
    const int offset = t * 4;
    int src0;
    int src1;
    int mult;
    int dst;
    if (model.use_const_alpha_times) {
      src0 = static_cast<int>(c_sus2_alpha_times_u16[offset + 0]);
      src1 = static_cast<int>(c_sus2_alpha_times_u16[offset + 1]);
      mult = static_cast<int>(c_sus2_alpha_times_u16[offset + 2]);
      dst = static_cast<int>(c_sus2_alpha_times_u16[offset + 3]);
    } else {
      src0 = model.alpha_times[offset + 0];
      src1 = model.alpha_times[offset + 1];
      mult = model.alpha_times[offset + 2];
      dst = model.alpha_times[offset + 3];
    }
    vals[dst] += static_cast<RealT>(mult) * vals[src0] * vals[src1];
  }

  const int type_i = type[i];
  const RealT species_coeff = sus2_species_coeff<RealT>(model, type_i);
  RealT site_energy = sus2_shift_coeff<RealT>(model, type_i) + species_coeff;
  for (int idx = 0; idx < model.alpha_scalar_moments; ++idx) {
    const int moment_id = sus2_scalar_moment_id(model, idx);
    const RealT coeff = sus2_moment_coeff<RealT>(model, idx);
    site_energy += coeff * vals[moment_id] * species_coeff;
    grads_local[moment_id] += static_cast<GradT>(coeff);
  }
  potential[i] += static_cast<double>(site_energy);

  for (int t = model.alpha_times_count - 1; t >= 0; --t) {
    const int offset = t * 4;
    int src0;
    int src1;
    int mult;
    int dst;
    if (model.use_const_alpha_times) {
      src0 = static_cast<int>(c_sus2_alpha_times_u16[offset + 0]);
      src1 = static_cast<int>(c_sus2_alpha_times_u16[offset + 1]);
      mult = static_cast<int>(c_sus2_alpha_times_u16[offset + 2]);
      dst = static_cast<int>(c_sus2_alpha_times_u16[offset + 3]);
    } else {
      src0 = model.alpha_times[offset + 0];
      src1 = model.alpha_times[offset + 1];
      mult = model.alpha_times[offset + 2];
      dst = model.alpha_times[offset + 3];
    }
    const RealT gdst = static_cast<RealT>(grads_local[dst]) * static_cast<RealT>(mult);
    grads_local[src1] += static_cast<GradT>(gdst * vals[src0]);
    grads_local[src0] += static_cast<GradT>(gdst * vals[src1]);
  }

  for (int basic = 0; basic < model.alpha_basic_count; ++basic) {
    basic_grads[static_cast<size_t>(basic) * N + i] = grads_local[basic];
  }
}

template <typename RealT, typename GradT>
static __global__ void gpu_l3k3_tensor_scalar_energy_backward(
  int N,
  SUS2DeviceModel model,
  const int* type,
  const RealT* basic_moments,
  GradT* basic_grads,
  double* potential)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || model.alpha_basic_count != kSus2L3K3TensorScalarBasic ||
      model.l3k3_tensor_scalar_term_count <= 0) {
    return;
  }

  RealT vals[kSus2L3K3TensorScalarBasic];
  RealT grads_local[kSus2L3K3TensorScalarBasic];
#pragma unroll
  for (int basic = 0; basic < kSus2L3K3TensorScalarBasic; ++basic) {
    vals[basic] = basic_moments[static_cast<size_t>(basic) * N + i];
    grads_local[basic] = static_cast<RealT>(0.0);
  }

  RealT scalar_sum = static_cast<RealT>(0.0);
  for (int term = 0; term < model.l3k3_tensor_scalar_term_count; ++term) {
    const int offset = term * kSus2TensorScalarPackedInts;
    const int degree = model.l3k3_tensor_scalar_terms[offset + 0];
    const int b0 = model.l3k3_tensor_scalar_terms[offset + 1];
    const int b1 = model.l3k3_tensor_scalar_terms[offset + 2];
    const int b2 = model.l3k3_tensor_scalar_terms[offset + 3];
    const int b3 = model.l3k3_tensor_scalar_terms[offset + 4];
    const int b4 = model.l3k3_tensor_scalar_terms[offset + 5];
    const RealT coeff = sus2_l3k3_tensor_scalar_coeff<RealT>(model, term);

    if (degree == 1) {
      scalar_sum += coeff * vals[b0];
      grads_local[b0] += coeff;
    } else if (degree == 2) {
      const RealT v0 = vals[b0];
      const RealT v1 = vals[b1];
      scalar_sum += coeff * v0 * v1;
      grads_local[b0] += coeff * v1;
      grads_local[b1] += coeff * v0;
    } else if (degree == 3) {
      const RealT v0 = vals[b0];
      const RealT v1 = vals[b1];
      const RealT v2 = vals[b2];
      scalar_sum += coeff * v0 * v1 * v2;
      grads_local[b0] += coeff * v1 * v2;
      grads_local[b1] += coeff * v0 * v2;
      grads_local[b2] += coeff * v0 * v1;
    } else if (degree == 4) {
      const RealT v0 = vals[b0];
      const RealT v1 = vals[b1];
      const RealT v2 = vals[b2];
      const RealT v3 = vals[b3];
      scalar_sum += coeff * v0 * v1 * v2 * v3;
      grads_local[b0] += coeff * v1 * v2 * v3;
      grads_local[b1] += coeff * v0 * v2 * v3;
      grads_local[b2] += coeff * v0 * v1 * v3;
      grads_local[b3] += coeff * v0 * v1 * v2;
    } else if (degree == 5) {
      const RealT v0 = vals[b0];
      const RealT v1 = vals[b1];
      const RealT v2 = vals[b2];
      const RealT v3 = vals[b3];
      const RealT v4 = vals[b4];
      scalar_sum += coeff * v0 * v1 * v2 * v3 * v4;
      grads_local[b0] += coeff * v1 * v2 * v3 * v4;
      grads_local[b1] += coeff * v0 * v2 * v3 * v4;
      grads_local[b2] += coeff * v0 * v1 * v3 * v4;
      grads_local[b3] += coeff * v0 * v1 * v2 * v4;
      grads_local[b4] += coeff * v0 * v1 * v2 * v3;
    }
  }

  const int type_i = type[i];
  const RealT species_coeff = sus2_species_coeff<RealT>(model, type_i);
  const RealT site_energy =
    sus2_shift_coeff<RealT>(model, type_i) + species_coeff + species_coeff * scalar_sum;
  potential[i] += static_cast<double>(site_energy);

#pragma unroll
  for (int basic = 0; basic < kSus2L3K3TensorScalarBasic; ++basic) {
    basic_grads[static_cast<size_t>(basic) * N + i] = static_cast<GradT>(grads_local[basic]);
  }
}

static __global__ void gpu_l3k3_tensor_scalar_energy_backward_float_parallel(
  int N,
  SUS2DeviceModel model,
  const int* type,
  const float* basic_moments,
  float* basic_grads,
  double* potential)
{
  const int i = blockIdx.x;
  const int tid = threadIdx.x;
  if (i >= N || model.alpha_basic_count != kSus2L3K3TensorScalarBasic ||
      model.l3k3_tensor_scalar_term_count <= 0) {
    return;
  }

  __shared__ float vals[kSus2L3K3TensorScalarBasic];
  __shared__ float grads_shared[kSus2L3K3TensorScalarBasic];
  __shared__ float scalar_shared[kSus2TensorScalarBlockSize];

  if (tid < kSus2L3K3TensorScalarBasic) {
    vals[tid] = basic_moments[static_cast<size_t>(tid) * N + i];
    grads_shared[tid] = 0.0f;
  }
  scalar_shared[tid] = 0.0f;
  __syncthreads();

  float scalar_local = 0.0f;
  for (int term = tid; term < model.l3k3_tensor_scalar_term_count; term += blockDim.x) {
    const int offset = term * kSus2TensorScalarPackedInts;
    const int degree = model.l3k3_tensor_scalar_terms[offset + 0];
    const int b0 = model.l3k3_tensor_scalar_terms[offset + 1];
    const int b1 = model.l3k3_tensor_scalar_terms[offset + 2];
    const int b2 = model.l3k3_tensor_scalar_terms[offset + 3];
    const int b3 = model.l3k3_tensor_scalar_terms[offset + 4];
    const int b4 = model.l3k3_tensor_scalar_terms[offset + 5];
    const float coeff = sus2_l3k3_tensor_scalar_coeff<float>(model, term);

    if (degree == 1) {
      scalar_local += coeff * vals[b0];
      atomicAdd(&grads_shared[b0], coeff);
    } else if (degree == 2) {
      const float v0 = vals[b0];
      const float v1 = vals[b1];
      scalar_local += coeff * v0 * v1;
      atomicAdd(&grads_shared[b0], coeff * v1);
      atomicAdd(&grads_shared[b1], coeff * v0);
    } else if (degree == 3) {
      const float v0 = vals[b0];
      const float v1 = vals[b1];
      const float v2 = vals[b2];
      scalar_local += coeff * v0 * v1 * v2;
      atomicAdd(&grads_shared[b0], coeff * v1 * v2);
      atomicAdd(&grads_shared[b1], coeff * v0 * v2);
      atomicAdd(&grads_shared[b2], coeff * v0 * v1);
    } else if (degree == 4) {
      const float v0 = vals[b0];
      const float v1 = vals[b1];
      const float v2 = vals[b2];
      const float v3 = vals[b3];
      scalar_local += coeff * v0 * v1 * v2 * v3;
      atomicAdd(&grads_shared[b0], coeff * v1 * v2 * v3);
      atomicAdd(&grads_shared[b1], coeff * v0 * v2 * v3);
      atomicAdd(&grads_shared[b2], coeff * v0 * v1 * v3);
      atomicAdd(&grads_shared[b3], coeff * v0 * v1 * v2);
    } else if (degree == 5) {
      const float v0 = vals[b0];
      const float v1 = vals[b1];
      const float v2 = vals[b2];
      const float v3 = vals[b3];
      const float v4 = vals[b4];
      scalar_local += coeff * v0 * v1 * v2 * v3 * v4;
      atomicAdd(&grads_shared[b0], coeff * v1 * v2 * v3 * v4);
      atomicAdd(&grads_shared[b1], coeff * v0 * v2 * v3 * v4);
      atomicAdd(&grads_shared[b2], coeff * v0 * v1 * v3 * v4);
      atomicAdd(&grads_shared[b3], coeff * v0 * v1 * v2 * v4);
      atomicAdd(&grads_shared[b4], coeff * v0 * v1 * v2 * v3);
    }
  }

  scalar_shared[tid] = scalar_local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scalar_shared[tid] += scalar_shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const int type_i = type[i];
    const float species_coeff = sus2_species_coeff<float>(model, type_i);
    const float site_energy =
      sus2_shift_coeff<float>(model, type_i) + species_coeff + species_coeff * scalar_shared[0];
    potential[i] += static_cast<double>(site_energy);
  }
  __syncthreads();

  if (tid < kSus2L3K3TensorScalarBasic) {
    basic_grads[static_cast<size_t>(tid) * N + i] = grads_shared[tid];
  }
}

template <typename RealT, typename GradT>
static __global__ void gpu_site_energy_init_grad(
  int N,
  SUS2DeviceModel model,
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
  const RealT species_coeff = sus2_species_coeff<RealT>(model, type_i);
  RealT site_energy = sus2_shift_coeff<RealT>(model, type_i) + species_coeff;
  for (int idx = 0; idx < model.alpha_scalar_moments; ++idx) {
    const int moment_id = sus2_scalar_moment_id(model, idx);
    const RealT coeff = sus2_moment_coeff<RealT>(model, idx);
    site_energy += coeff * moments[static_cast<size_t>(moment_id) * N + i] *
                   species_coeff;
    add_sus2_grad(grads, static_cast<size_t>(moment_id) * N + i, coeff);
  }
  potential[i] += static_cast<double>(site_energy);
}

template <typename RealT, typename GradT>
__device__ __forceinline__ void sus2_init_site_energy_and_scalar_grads(
  int N,
  SUS2DeviceModel model,
  int i,
  const int* type,
  const RealT* moments,
  GradT* grads,
  double* potential)
{
  const int type_i = type[i];
  const RealT species_coeff = sus2_species_coeff<RealT>(model, type_i);
  RealT site_energy = sus2_shift_coeff<RealT>(model, type_i) + species_coeff;
  for (int idx = 0; idx < model.alpha_scalar_moments; ++idx) {
    const int moment_id = sus2_scalar_moment_id(model, idx);
    const RealT coeff = sus2_moment_coeff<RealT>(model, idx);
    site_energy += coeff * moments[static_cast<size_t>(moment_id) * N + i] *
                   species_coeff;
    add_sus2_grad(grads, static_cast<size_t>(moment_id) * N + i, coeff);
  }
  potential[i] += static_cast<double>(site_energy);
}

template <typename RealT, typename GradT>
static __global__ void gpu_backward_times(int N, SUS2DeviceModel model, const RealT* moments, GradT* grads)
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
    const RealT gdst =
      static_cast<RealT>(load_sus2_grad(grads, N, dst, i)) * static_cast<RealT>(mult);
    add_sus2_grad(grads, static_cast<size_t>(src1) * N + i, gdst * moments[static_cast<size_t>(src0) * N + i]);
    add_sus2_grad(grads, static_cast<size_t>(src0) * N + i, gdst * moments[static_cast<size_t>(src1) * N + i]);
  }
}

template <typename RealT, typename GradT>
static __global__ void gpu_backward_times_const_u16(
  int N,
  SUS2DeviceModel model,
  const RealT* moments,
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
    const RealT gdst =
      static_cast<RealT>(load_sus2_grad(grads, N, dst, i)) * static_cast<RealT>(mult);
    add_sus2_grad(grads, static_cast<size_t>(src1) * N + i, gdst * moments[static_cast<size_t>(src0) * N + i]);
    add_sus2_grad(grads, static_cast<size_t>(src0) * N + i, gdst * moments[static_cast<size_t>(src1) * N + i]);
  }
}

template <typename RealT, typename GradT>
static __global__ void gpu_site_energy_init_grad_backward(
  int N,
  SUS2DeviceModel model,
  const int* type,
  const RealT* moments,
  GradT* grads,
  double* potential)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  sus2_init_site_energy_and_scalar_grads(N, model, i, type, moments, grads, potential);
  for (int t = model.alpha_times_count - 1; t >= 0; --t) {
    const int src0 = model.alpha_times[t * 4 + 0];
    const int src1 = model.alpha_times[t * 4 + 1];
    const int mult = model.alpha_times[t * 4 + 2];
    const int dst = model.alpha_times[t * 4 + 3];
    const RealT gdst =
      static_cast<RealT>(load_sus2_grad(grads, N, dst, i)) * static_cast<RealT>(mult);
    add_sus2_grad(
      grads, static_cast<size_t>(src1) * N + i, gdst * moments[static_cast<size_t>(src0) * N + i]);
    add_sus2_grad(
      grads, static_cast<size_t>(src0) * N + i, gdst * moments[static_cast<size_t>(src1) * N + i]);
  }
}

template <typename RealT, typename GradT>
static __global__ void gpu_site_energy_init_grad_backward_const_u16(
  int N,
  SUS2DeviceModel model,
  const int* type,
  const RealT* moments,
  GradT* grads,
  double* potential)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  sus2_init_site_energy_and_scalar_grads(N, model, i, type, moments, grads, potential);
  for (int t = model.alpha_times_count - 1; t >= 0; --t) {
    const int offset = t * 4;
    const int src0 = static_cast<int>(c_sus2_alpha_times_u16[offset + 0]);
    const int src1 = static_cast<int>(c_sus2_alpha_times_u16[offset + 1]);
    const int mult = static_cast<int>(c_sus2_alpha_times_u16[offset + 2]);
    const int dst = static_cast<int>(c_sus2_alpha_times_u16[offset + 3]);
    const RealT gdst =
      static_cast<RealT>(load_sus2_grad(grads, N, dst, i)) * static_cast<RealT>(mult);
    add_sus2_grad(
      grads, static_cast<size_t>(src1) * N + i, gdst * moments[static_cast<size_t>(src0) * N + i]);
    add_sus2_grad(
      grads, static_cast<size_t>(src0) * N + i, gdst * moments[static_cast<size_t>(src1) * N + i]);
  }
}

template <typename RealT, typename GradT>
static __global__ void gpu_forward_energy_backward_assign_group_table(
  int N,
  SUS2DeviceModel model,
  const int* type,
  RealT* moments,
  GradT* grads,
  double* potential)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  for (int g = 0; g < model.alpha_time_group_count; ++g) {
    const int group_offset = g * 3;
    const int begin = model.alpha_time_groups[group_offset + 0];
    const int len = model.alpha_time_groups[group_offset + 1];
    const int dst = model.alpha_time_groups[group_offset + 2];
    RealT dst_value = static_cast<RealT>(0.0);
    for (int k = 0; k < len; ++k) {
      const int offset = (begin + k) * 4;
      const int src0 = model.alpha_times[offset + 0];
      const int src1 = model.alpha_times[offset + 1];
      const int mult = model.alpha_times[offset + 2];
      dst_value += static_cast<RealT>(mult) * moments[static_cast<size_t>(src0) * N + i] *
                   moments[static_cast<size_t>(src1) * N + i];
    }
    moments[static_cast<size_t>(dst) * N + i] = dst_value;
  }

  sus2_init_site_energy_and_scalar_grads(N, model, i, type, moments, grads, potential);

  for (int g = model.alpha_time_group_count - 1; g >= 0; --g) {
    const int group_offset = g * 3;
    const int begin = model.alpha_time_groups[group_offset + 0];
    const int len = model.alpha_time_groups[group_offset + 1];
    const int dst = model.alpha_time_groups[group_offset + 2];
    const RealT dst_grad =
      static_cast<RealT>(load_sus2_grad(grads, N, dst, i));
    for (int k = len - 1; k >= 0; --k) {
      const int offset = (begin + k) * 4;
      const int src0 = model.alpha_times[offset + 0];
      const int src1 = model.alpha_times[offset + 1];
      const int mult = model.alpha_times[offset + 2];
      const RealT gdst = dst_grad * static_cast<RealT>(mult);
      add_sus2_grad(
        grads, static_cast<size_t>(src1) * N + i, gdst * moments[static_cast<size_t>(src0) * N + i]);
      add_sus2_grad(
        grads, static_cast<size_t>(src0) * N + i, gdst * moments[static_cast<size_t>(src1) * N + i]);
    }
  }
}

template <typename RealT, typename GradT>
static __global__ void gpu_forward_energy_backward_const_u16_assign_group_table(
  int N,
  SUS2DeviceModel model,
  const int* type,
  RealT* moments,
  GradT* grads,
  double* potential)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  for (int g = 0; g < model.alpha_time_group_count; ++g) {
    const int group_offset = g * 3;
    const int begin = model.alpha_time_groups[group_offset + 0];
    const int len = model.alpha_time_groups[group_offset + 1];
    const int dst = model.alpha_time_groups[group_offset + 2];
    RealT dst_value = static_cast<RealT>(0.0);
    for (int k = 0; k < len; ++k) {
      const int offset = (begin + k) * 4;
      const int src0 = static_cast<int>(c_sus2_alpha_times_u16[offset + 0]);
      const int src1 = static_cast<int>(c_sus2_alpha_times_u16[offset + 1]);
      const int mult = static_cast<int>(c_sus2_alpha_times_u16[offset + 2]);
      dst_value += static_cast<RealT>(mult) * moments[static_cast<size_t>(src0) * N + i] *
                   moments[static_cast<size_t>(src1) * N + i];
    }
    moments[static_cast<size_t>(dst) * N + i] = dst_value;
  }

  sus2_init_site_energy_and_scalar_grads(N, model, i, type, moments, grads, potential);

  for (int g = model.alpha_time_group_count - 1; g >= 0; --g) {
    const int group_offset = g * 3;
    const int begin = model.alpha_time_groups[group_offset + 0];
    const int len = model.alpha_time_groups[group_offset + 1];
    const int dst = model.alpha_time_groups[group_offset + 2];
    const RealT dst_grad =
      static_cast<RealT>(load_sus2_grad(grads, N, dst, i));
    for (int k = len - 1; k >= 0; --k) {
      const int offset = (begin + k) * 4;
      const int src0 = static_cast<int>(c_sus2_alpha_times_u16[offset + 0]);
      const int src1 = static_cast<int>(c_sus2_alpha_times_u16[offset + 1]);
      const int mult = static_cast<int>(c_sus2_alpha_times_u16[offset + 2]);
      const RealT gdst = dst_grad * static_cast<RealT>(mult);
      add_sus2_grad(
        grads, static_cast<size_t>(src1) * N + i, gdst * moments[static_cast<size_t>(src0) * N + i]);
      add_sus2_grad(
        grads, static_cast<size_t>(src0) * N + i, gdst * moments[static_cast<size_t>(src1) * N + i]);
    }
  }
}

__device__ __forceinline__ void sus2_product_group_pair(
  const SUS2DeviceModel& model,
  int group,
  int& begin,
  int& len,
  int& dst)
{
  const int offset = group * kSus2ProductGroupPairWords;
  const unsigned int word0 = __ldg(model.alpha_time_group_pairs + offset + 0);
  const unsigned int word1 = __ldg(model.alpha_time_group_pairs + offset + 1);
  begin = static_cast<int>(word0 & 0xffffu);
  len = static_cast<int>(word0 >> 16);
  dst = static_cast<int>(word1 & 0xffffu);
}

template <typename RealT, typename GradT>
static __global__ void gpu_forward_energy_backward_const_u16_group_pair_table(
  int N,
  SUS2DeviceModel model,
  const int* type,
  RealT* moments,
  GradT* grads,
  double* potential)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || model.alpha_time_group_pairs == nullptr) {
    return;
  }

  for (int g = 0; g < model.alpha_time_group_count; ++g) {
    int begin;
    int len;
    int dst;
    sus2_product_group_pair(model, g, begin, len, dst);
    RealT dst_value = static_cast<RealT>(0.0);
    for (int k = 0; k < len; ++k) {
      const int offset = (begin + k) * 4;
      const int src0 = static_cast<int>(c_sus2_alpha_times_u16[offset + 0]);
      const int src1 = static_cast<int>(c_sus2_alpha_times_u16[offset + 1]);
      const int mult = static_cast<int>(c_sus2_alpha_times_u16[offset + 2]);
      dst_value += static_cast<RealT>(mult) * moments[static_cast<size_t>(src0) * N + i] *
                   moments[static_cast<size_t>(src1) * N + i];
    }
    moments[static_cast<size_t>(dst) * N + i] = dst_value;
  }

  sus2_init_site_energy_and_scalar_grads(N, model, i, type, moments, grads, potential);

  for (int g = model.alpha_time_group_count - 1; g >= 0; --g) {
    int begin;
    int len;
    int dst;
    sus2_product_group_pair(model, g, begin, len, dst);
    const RealT dst_grad =
      static_cast<RealT>(load_sus2_grad(grads, N, dst, i));
    for (int k = len - 1; k >= 0; --k) {
      const int offset = (begin + k) * 4;
      const int src0 = static_cast<int>(c_sus2_alpha_times_u16[offset + 0]);
      const int src1 = static_cast<int>(c_sus2_alpha_times_u16[offset + 1]);
      const int mult = static_cast<int>(c_sus2_alpha_times_u16[offset + 2]);
      const RealT gdst = dst_grad * static_cast<RealT>(mult);
      add_sus2_grad(
        grads, static_cast<size_t>(src1) * N + i, gdst * moments[static_cast<size_t>(src0) * N + i]);
      add_sus2_grad(
        grads, static_cast<size_t>(src0) * N + i, gdst * moments[static_cast<size_t>(src1) * N + i]);
    }
  }
}

__device__ __forceinline__ int l3k3_sym2_offset(int a, int b)
{
  if (a > b) {
    const int tmp = a;
    a = b;
    b = tmp;
  }
  if (a == 0) {
    return b;
  }
  if (a == 1) {
    return b + 2;
  }
  return 5;
}

__device__ __forceinline__ int l3k3_sym3_offset(int a, int b, int c)
{
  if (a > b) {
    const int tmp = a;
    a = b;
    b = tmp;
  }
  if (b > c) {
    const int tmp = b;
    b = c;
    c = tmp;
  }
  if (a > b) {
    const int tmp = a;
    a = b;
    b = tmp;
  }
  if (a == 0) {
    if (b == 0) {
      return c;
    }
    return b == 1 ? c + 2 : 5;
  }
  if (a == 1) {
    return b == 1 ? c + 5 : 8;
  }
  return 9;
}

__device__ __forceinline__ int l3k3_mat_offset(int a, int b)
{
  return a * 3 + b;
}

__device__ __forceinline__ int l3k3_sym2_first(int offset)
{
  return offset < 3 ? 0 : (offset < 5 ? 1 : 2);
}

__device__ __forceinline__ int l3k3_sym2_second(int offset)
{
  return offset < 3 ? offset : (offset < 5 ? offset - 2 : 2);
}

__device__ __forceinline__ int l3k3_tensor_block_meta_value(
  const SUS2DeviceModel& model,
  int op,
  int offset)
{
  return __ldg(model.l3k3_tensor_block_metas + op * kSus2TensorBlockMetaInts + offset);
}

__device__ __forceinline__ int l3k3_tensor_block_sym_count(int rank)
{
  return (rank + 1) * (rank + 2) / 2;
}

__device__ __forceinline__ int l3k3_tensor_block_sym_offset_from_counts(
  int rank,
  int c0,
  int c1,
  int c2)
{
  if (c0 + c1 + c2 != rank) {
    return -1;
  }
  int offset = 0;
  for (int a = rank; a >= 0; --a) {
    for (int b = rank - a; b >= 0; --b) {
      const int c = rank - a - b;
      if (a == c0 && b == c1 && c == c2) {
        return offset;
      }
      ++offset;
    }
  }
  return -1;
}

__device__ __forceinline__ bool l3k3_tensor_block_sym_counts_from_offset(
  int rank,
  int target_offset,
  int& c0,
  int& c1,
  int& c2)
{
  int offset = 0;
  for (int a = rank; a >= 0; --a) {
    for (int b = rank - a; b >= 0; --b) {
      const int c = rank - a - b;
      if (offset == target_offset) {
        c0 = a;
        c1 = b;
        c2 = c;
        return true;
      }
      ++offset;
    }
  }
  return false;
}

__device__ __forceinline__ void l3k3_tensor_block_zero_counts(
  int counts[kSus2TensorBlockMaxGroups][3])
{
#pragma unroll
  for (int group = 0; group < kSus2TensorBlockMaxGroups; ++group) {
#pragma unroll
    for (int dim = 0; dim < 3; ++dim) {
      counts[group][dim] = 0;
    }
  }
}

__device__ __forceinline__ void l3k3_tensor_block_copy_counts(
  const int src[kSus2TensorBlockMaxGroups][3],
  int dst[kSus2TensorBlockMaxGroups][3])
{
#pragma unroll
  for (int group = 0; group < kSus2TensorBlockMaxGroups; ++group) {
#pragma unroll
    for (int dim = 0; dim < 3; ++dim) {
      dst[group][dim] = src[group][dim];
    }
  }
}

__device__ __forceinline__ int l3k3_tensor_block_component_offset_from_counts(
  const SUS2DeviceModel& model,
  int op,
  bool side_a,
  const int counts[kSus2TensorBlockMaxGroups][3])
{
  const int group_count_offset =
    side_a ? kSus2TensorBlockMetaGroupACount : kSus2TensorBlockMetaGroupBCount;
  const int groups_offset =
    side_a ? kSus2TensorBlockMetaGroupsA : kSus2TensorBlockMetaGroupsB;
  const int group_count = l3k3_tensor_block_meta_value(model, op, group_count_offset);
  int offset = 0;
  for (int group = 0; group < group_count; ++group) {
    const int rank = l3k3_tensor_block_meta_value(model, op, groups_offset + group);
    const int local_offset = l3k3_tensor_block_sym_offset_from_counts(
      rank, counts[group][0], counts[group][1], counts[group][2]);
    if (local_offset < 0) {
      return -1;
    }
    offset = offset * l3k3_tensor_block_sym_count(rank) + local_offset;
  }
  return offset;
}

__device__ __forceinline__ int l3k3_tensor_block_contract_pairs(
  const SUS2DeviceModel& model,
  int op,
  int pair_a[kSus2MaxTensorRank],
  int pair_b[kSus2MaxTensorRank])
{
  const int group_a_count =
    l3k3_tensor_block_meta_value(model, op, kSus2TensorBlockMetaGroupACount);
  const int group_b_count =
    l3k3_tensor_block_meta_value(model, op, kSus2TensorBlockMetaGroupBCount);
  int pair_count = 0;
  for (int a = 0; a < group_a_count; ++a) {
    for (int b = 0; b < group_b_count; ++b) {
      const int count = l3k3_tensor_block_meta_value(
        model, op, kSus2TensorBlockMetaMatrix + a * kSus2TensorBlockMaxGroups + b);
      for (int k = 0; k < count; ++k) {
        if (pair_count >= kSus2MaxTensorRank) {
          return -1;
        }
        pair_a[pair_count] = a;
        pair_b[pair_count] = b;
        ++pair_count;
      }
    }
  }
  return pair_count;
}

__device__ __forceinline__ bool l3k3_tensor_block_fill_free_counts(
  const SUS2DeviceModel& model,
  int op,
  int component,
  int counts_a[kSus2TensorBlockMaxGroups][3],
  int counts_b[kSus2TensorBlockMaxGroups][3])
{
  const int group_a_count =
    l3k3_tensor_block_meta_value(model, op, kSus2TensorBlockMetaGroupACount);
  const int group_b_count =
    l3k3_tensor_block_meta_value(model, op, kSus2TensorBlockMetaGroupBCount);
  const int out_group_count =
    l3k3_tensor_block_meta_value(model, op, kSus2TensorBlockMetaOutGroupCount);
  int group_offsets[kSus2TensorBlockMaxGroups] = {0, 0, 0, 0};
  int remainder = component;
  for (int group = out_group_count - 1; group >= 0; --group) {
    const int rank =
      l3k3_tensor_block_meta_value(model, op, kSus2TensorBlockMetaOutGroups + group);
    const int count = l3k3_tensor_block_sym_count(rank);
    if (count <= 0) {
      return false;
    }
    group_offsets[group] = remainder % count;
    remainder /= count;
  }
  if (remainder != 0) {
    return false;
  }

  for (int group = 0; group < out_group_count; ++group) {
    const int rank =
      l3k3_tensor_block_meta_value(model, op, kSus2TensorBlockMetaOutGroups + group);
    int remaining[3];
    if (!l3k3_tensor_block_sym_counts_from_offset(
          rank, group_offsets[group], remaining[0], remaining[1], remaining[2])) {
      return false;
    }
    const int label_begin = l3k3_tensor_block_meta_value(
      model, op, kSus2TensorBlockMetaLabelGroups + group * 2 + 0);
    const int label_count = l3k3_tensor_block_meta_value(
      model, op, kSus2TensorBlockMetaLabelGroups + group * 2 + 1);
    for (int label_index = 0; label_index < label_count; ++label_index) {
      const int label = label_begin + label_index;
      const int label_offset = kSus2TensorBlockMetaLabels + label * 3;
      const int side = l3k3_tensor_block_meta_value(model, op, label_offset + 0);
      const int index = l3k3_tensor_block_meta_value(model, op, label_offset + 1);
      const int size = l3k3_tensor_block_meta_value(model, op, label_offset + 2);
      if ((side == 0 && (index < 0 || index >= group_a_count)) ||
          (side == 1 && (index < 0 || index >= group_b_count)) ||
          (side != 0 && side != 1)) {
        return false;
      }
      for (int k = 0; k < size; ++k) {
        int dim = 0;
        while (dim < 3 && remaining[dim] == 0) {
          ++dim;
        }
        if (dim >= 3) {
          return false;
        }
        --remaining[dim];
        if (side == 0) {
          ++counts_a[index][dim];
        } else {
          ++counts_b[index][dim];
        }
      }
    }
    if (remaining[0] != 0 || remaining[1] != 0 || remaining[2] != 0) {
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ int l3k3_tensor_block_pow3(int exp)
{
  int value = 1;
  for (int i = 0; i < exp; ++i) {
    value *= 3;
  }
  return value;
}

template <typename RealT>
__device__ __forceinline__ RealT l3k3_tensor_block_moment(
  const RealT* moments,
  int N,
  int start,
  int component,
  int atom)
{
  return moments[static_cast<size_t>(start + component) * N + atom];
}

template <typename RealT>
__device__ __forceinline__ void l3k3_tensor_block_store(
  RealT* moments,
  int N,
  int start,
  int component,
  int atom,
  RealT value)
{
  moments[static_cast<size_t>(start + component) * N + atom] = value;
}

template <typename RealT, typename GradT>
__device__ __forceinline__ RealT l3k3_tensor_block_grad(
  const GradT* grads,
  int N,
  int start,
  int component,
  int atom)
{
  return static_cast<RealT>(load_sus2_grad(grads, N, start + component, atom));
}

template <typename GradT>
__device__ __forceinline__ void l3k3_tensor_block_add_grad(
  GradT* grads,
  int N,
  int start,
  int component,
  int atom,
  double value)
{
  add_sus2_grad(grads, static_cast<size_t>(start + component) * N + atom, value);
}

template <typename RealT, typename GradT>
__device__ __forceinline__ void l3k3_tensor_block_add_row_grad(
  const RealT* moments,
  GradT* grads,
  int N,
  int atom,
  int a,
  int a_offset,
  int b,
  int b_offset,
  RealT dst_grad,
  int mult)
{
  const RealT gdst = dst_grad * static_cast<RealT>(mult);
  const RealT av = l3k3_tensor_block_moment(moments, N, a, a_offset, atom);
  const RealT bv = l3k3_tensor_block_moment(moments, N, b, b_offset, atom);
  l3k3_tensor_block_add_grad(grads, N, b, b_offset, atom, static_cast<double>(gdst * av));
  l3k3_tensor_block_add_grad(grads, N, a, a_offset, atom, static_cast<double>(gdst * bv));
}

__device__ __forceinline__ unsigned int l3k3_tensor_block_row_word(
  const SUS2DeviceModel& model,
  int index)
{
  return __ldg(model.l3k3_tensor_block_rows + index);
}

template <typename RealT>
__device__ __noinline__ bool l3k3_tensor_block_forward_structured(
  int N,
  int atom,
  const SUS2DeviceModel& model,
  int op_offset,
  RealT* moments)
{
  if (model.l3k3_tensor_block_metas == nullptr) {
    return false;
  }
  const int op = op_offset / kSus2TensorBlockOpInts;
  const int component_count = model.l3k3_tensor_block_ops[op_offset + 1];
  const int a = model.l3k3_tensor_block_ops[op_offset + 3];
  const int b = model.l3k3_tensor_block_ops[op_offset + 4];
  const int d = model.l3k3_tensor_block_ops[op_offset + 5];
  int pair_a[kSus2MaxTensorRank] = {0, 0, 0, 0};
  int pair_b[kSus2MaxTensorRank] = {0, 0, 0, 0};
  const int pair_count = l3k3_tensor_block_contract_pairs(model, op, pair_a, pair_b);
  if (pair_count < 0) {
    return false;
  }
  const int combo_count = l3k3_tensor_block_pow3(pair_count);

  for (int component = 0; component < component_count; ++component) {
    int free_a[kSus2TensorBlockMaxGroups][3];
    int free_b[kSus2TensorBlockMaxGroups][3];
    l3k3_tensor_block_zero_counts(free_a);
    l3k3_tensor_block_zero_counts(free_b);
    if (!l3k3_tensor_block_fill_free_counts(model, op, component, free_a, free_b)) {
      return false;
    }
    RealT sum = static_cast<RealT>(0);
    for (int combo = 0; combo < combo_count; ++combo) {
      int counts_a[kSus2TensorBlockMaxGroups][3];
      int counts_b[kSus2TensorBlockMaxGroups][3];
      l3k3_tensor_block_copy_counts(free_a, counts_a);
      l3k3_tensor_block_copy_counts(free_b, counts_b);
      int key = combo;
      for (int pair = 0; pair < pair_count; ++pair) {
        const int dim = key % 3;
        key /= 3;
        ++counts_a[pair_a[pair]][dim];
        ++counts_b[pair_b[pair]][dim];
      }
      const int a_offset =
        l3k3_tensor_block_component_offset_from_counts(model, op, true, counts_a);
      const int b_offset =
        l3k3_tensor_block_component_offset_from_counts(model, op, false, counts_b);
      if (a_offset < 0 || b_offset < 0) {
        return false;
      }
      sum += l3k3_tensor_block_moment(moments, N, a, a_offset, atom) *
             l3k3_tensor_block_moment(moments, N, b, b_offset, atom);
    }
    l3k3_tensor_block_store(moments, N, d, component, atom, sum);
  }
  return true;
}

template <typename RealT, typename GradT>
__device__ __noinline__ bool l3k3_tensor_block_backward_structured(
  int N,
  int atom,
  const SUS2DeviceModel& model,
  int op_offset,
  const RealT* moments,
  GradT* grads)
{
  if (model.l3k3_tensor_block_metas == nullptr) {
    return false;
  }
  const int op = op_offset / kSus2TensorBlockOpInts;
  const int component_count = model.l3k3_tensor_block_ops[op_offset + 1];
  const int a = model.l3k3_tensor_block_ops[op_offset + 3];
  const int b = model.l3k3_tensor_block_ops[op_offset + 4];
  const int d = model.l3k3_tensor_block_ops[op_offset + 5];
  int pair_a[kSus2MaxTensorRank] = {0, 0, 0, 0};
  int pair_b[kSus2MaxTensorRank] = {0, 0, 0, 0};
  const int pair_count = l3k3_tensor_block_contract_pairs(model, op, pair_a, pair_b);
  if (pair_count < 0) {
    return false;
  }
  const int combo_count = l3k3_tensor_block_pow3(pair_count);

  for (int component = component_count - 1; component >= 0; --component) {
    int free_a[kSus2TensorBlockMaxGroups][3];
    int free_b[kSus2TensorBlockMaxGroups][3];
    l3k3_tensor_block_zero_counts(free_a);
    l3k3_tensor_block_zero_counts(free_b);
    if (!l3k3_tensor_block_fill_free_counts(model, op, component, free_a, free_b)) {
      return false;
    }
    const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, component, atom);
    for (int combo = combo_count - 1; combo >= 0; --combo) {
      int counts_a[kSus2TensorBlockMaxGroups][3];
      int counts_b[kSus2TensorBlockMaxGroups][3];
      l3k3_tensor_block_copy_counts(free_a, counts_a);
      l3k3_tensor_block_copy_counts(free_b, counts_b);
      int key = combo;
      for (int pair = 0; pair < pair_count; ++pair) {
        const int dim = key % 3;
        key /= 3;
        ++counts_a[pair_a[pair]][dim];
        ++counts_b[pair_b[pair]][dim];
      }
      const int a_offset =
        l3k3_tensor_block_component_offset_from_counts(model, op, true, counts_a);
      const int b_offset =
        l3k3_tensor_block_component_offset_from_counts(model, op, false, counts_b);
      if (a_offset < 0 || b_offset < 0) {
        return false;
      }
      l3k3_tensor_block_add_row_grad(
        moments, grads, N, atom, a, a_offset, b, b_offset, gd, 1);
    }
  }
  return true;
}

template <typename RealT>
__device__ __forceinline__ bool l3k3_tensor_block_forward_fast(
  int N,
  int atom,
  const SUS2DeviceModel& model,
  int op_offset,
  RealT* moments)
{
  const int component_count = model.l3k3_tensor_block_ops[op_offset + 1];
  const int kind = model.l3k3_tensor_block_ops[op_offset + 2];
  const int a = model.l3k3_tensor_block_ops[op_offset + 3];
  const int b = model.l3k3_tensor_block_ops[op_offset + 4];
  const int d = model.l3k3_tensor_block_ops[op_offset + 5];

  switch (kind) {
    case kSus2TensorBlockScalarScalar: {
      l3k3_tensor_block_store(
        moments, N, d, 0, atom,
        l3k3_tensor_block_moment(moments, N, a, 0, atom) *
          l3k3_tensor_block_moment(moments, N, b, 0, atom));
      return true;
    }
    case kSus2TensorBlockScalarTensor: {
      const RealT scalar = l3k3_tensor_block_moment(moments, N, a, 0, atom);
      for (int c = 0; c < component_count; ++c) {
        l3k3_tensor_block_store(
          moments, N, d, c, atom,
          scalar * l3k3_tensor_block_moment(moments, N, b, c, atom));
      }
      return true;
    }
    case kSus2TensorBlockTensorScalar: {
      const RealT scalar = l3k3_tensor_block_moment(moments, N, b, 0, atom);
      for (int c = 0; c < component_count; ++c) {
        l3k3_tensor_block_store(
          moments, N, d, c, atom,
          l3k3_tensor_block_moment(moments, N, a, c, atom) * scalar);
      }
      return true;
    }
    case kSus2TensorBlockDot11: {
      RealT sum = static_cast<RealT>(0);
      for (int x = 0; x < 3; ++x) {
        sum += l3k3_tensor_block_moment(moments, N, a, x, atom) *
               l3k3_tensor_block_moment(moments, N, b, x, atom);
      }
      l3k3_tensor_block_store(moments, N, d, 0, atom, sum);
      return true;
    }
    case kSus2TensorBlockDot22: {
      RealT sum = static_cast<RealT>(0);
      const int mult[6] = {1, 2, 2, 1, 2, 1};
      for (int offset = 0; offset < 6; ++offset) {
        sum += static_cast<RealT>(mult[offset]) *
               l3k3_tensor_block_moment(moments, N, a, offset, atom) *
               l3k3_tensor_block_moment(moments, N, b, offset, atom);
      }
      l3k3_tensor_block_store(moments, N, d, 0, atom, sum);
      return true;
    }
    case kSus2TensorBlockDot33: {
      RealT sum = static_cast<RealT>(0);
      const int mult[10] = {1, 3, 3, 3, 6, 3, 1, 3, 3, 1};
      for (int offset = 0; offset < 10; ++offset) {
        sum += static_cast<RealT>(mult[offset]) *
               l3k3_tensor_block_moment(moments, N, a, offset, atom) *
               l3k3_tensor_block_moment(moments, N, b, offset, atom);
      }
      l3k3_tensor_block_store(moments, N, d, 0, atom, sum);
      return true;
    }
    case kSus2TensorBlockVecSym2ToVec: {
      for (int y = 0; y < 3; ++y) {
        RealT sum = static_cast<RealT>(0);
        for (int x = 0; x < 3; ++x) {
          sum += l3k3_tensor_block_moment(moments, N, a, x, atom) *
                 l3k3_tensor_block_moment(moments, N, b, l3k3_sym2_offset(x, y), atom);
        }
        l3k3_tensor_block_store(moments, N, d, y, atom, sum);
      }
      return true;
    }
    case kSus2TensorBlockSym2VecToVec: {
      for (int x = 0; x < 3; ++x) {
        RealT sum = static_cast<RealT>(0);
        for (int y = 0; y < 3; ++y) {
          sum += l3k3_tensor_block_moment(moments, N, a, l3k3_sym2_offset(x, y), atom) *
                 l3k3_tensor_block_moment(moments, N, b, y, atom);
        }
        l3k3_tensor_block_store(moments, N, d, x, atom, sum);
      }
      return true;
    }
    case kSus2TensorBlockVecSym3ToSym2: {
      for (int y = 0; y < 3; ++y) {
        for (int z = y; z < 3; ++z) {
          RealT sum = static_cast<RealT>(0);
          for (int x = 0; x < 3; ++x) {
            sum += l3k3_tensor_block_moment(moments, N, a, x, atom) *
                   l3k3_tensor_block_moment(moments, N, b, l3k3_sym3_offset(x, y, z), atom);
          }
          l3k3_tensor_block_store(moments, N, d, l3k3_sym2_offset(y, z), atom, sum);
        }
      }
      return true;
    }
    case kSus2TensorBlockSym3VecToSym2: {
      for (int x = 0; x < 3; ++x) {
        for (int y = x; y < 3; ++y) {
          RealT sum = static_cast<RealT>(0);
          for (int z = 0; z < 3; ++z) {
            sum += l3k3_tensor_block_moment(moments, N, a, l3k3_sym3_offset(x, y, z), atom) *
                   l3k3_tensor_block_moment(moments, N, b, z, atom);
          }
          l3k3_tensor_block_store(moments, N, d, l3k3_sym2_offset(x, y), atom, sum);
        }
      }
      return true;
    }
    case kSus2TensorBlockSym2Sym3ToVec: {
      const int b_offsets[3][6] = {
        {0, 1, 2, 3, 4, 5},
        {1, 3, 4, 6, 7, 8},
        {2, 4, 5, 7, 8, 9}};
      const int mult[6] = {1, 2, 2, 1, 2, 1};
      for (int z = 0; z < 3; ++z) {
        RealT sum = static_cast<RealT>(0);
        for (int row = 0; row < 6; ++row) {
          sum += static_cast<RealT>(mult[row]) *
                 l3k3_tensor_block_moment(moments, N, a, row, atom) *
                 l3k3_tensor_block_moment(moments, N, b, b_offsets[z][row], atom);
        }
        l3k3_tensor_block_store(moments, N, d, z, atom, sum);
      }
      return true;
    }
    case kSus2TensorBlockSym3Sym2ToVec: {
      const int a_offsets[3][6] = {
        {0, 1, 2, 3, 4, 5},
        {1, 3, 4, 6, 7, 8},
        {2, 4, 5, 7, 8, 9}};
      const int mult[6] = {1, 2, 2, 1, 2, 1};
      for (int x = 0; x < 3; ++x) {
        RealT sum = static_cast<RealT>(0);
        for (int row = 0; row < 6; ++row) {
          sum += static_cast<RealT>(mult[row]) *
                 l3k3_tensor_block_moment(moments, N, a, a_offsets[x][row], atom) *
                 l3k3_tensor_block_moment(moments, N, b, row, atom);
        }
        l3k3_tensor_block_store(moments, N, d, x, atom, sum);
      }
      return true;
    }
    case kSus2TensorBlockSym2Sym2ToSym2: {
      for (int x = 0; x < 3; ++x) {
        for (int y = x; y < 3; ++y) {
          RealT sum = static_cast<RealT>(0);
          for (int z = 0; z < 3; ++z) {
            sum += l3k3_tensor_block_moment(moments, N, a, l3k3_sym2_offset(x, z), atom) *
                   l3k3_tensor_block_moment(moments, N, b, l3k3_sym2_offset(y, z), atom);
          }
          l3k3_tensor_block_store(moments, N, d, l3k3_sym2_offset(x, y), atom, sum);
        }
      }
      return true;
    }
    case kSus2TensorBlockSym2Sym2ToMatAB:
    case kSus2TensorBlockSym2Sym2ToMatBA: {
      for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 3; ++y) {
          RealT sum = static_cast<RealT>(0);
          for (int z = 0; z < 3; ++z) {
            const int a_offset = kind == kSus2TensorBlockSym2Sym2ToMatAB
              ? l3k3_sym2_offset(x, z)
              : l3k3_sym2_offset(y, z);
            const int b_offset = kind == kSus2TensorBlockSym2Sym2ToMatAB
              ? l3k3_sym2_offset(y, z)
              : l3k3_sym2_offset(x, z);
            sum += l3k3_tensor_block_moment(moments, N, a, a_offset, atom) *
                   l3k3_tensor_block_moment(moments, N, b, b_offset, atom);
          }
          l3k3_tensor_block_store(moments, N, d, l3k3_mat_offset(x, y), atom, sum);
        }
      }
      return true;
    }
    case kSus2TensorBlockVecVecOuterAB:
    case kSus2TensorBlockVecVecOuterBA: {
      for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 3; ++y) {
          const int a_offset = kind == kSus2TensorBlockVecVecOuterAB ? x : y;
          const int b_offset = kind == kSus2TensorBlockVecVecOuterAB ? y : x;
          l3k3_tensor_block_store(
            moments, N, d, l3k3_mat_offset(x, y), atom,
            l3k3_tensor_block_moment(moments, N, a, a_offset, atom) *
              l3k3_tensor_block_moment(moments, N, b, b_offset, atom));
        }
      }
      return true;
    }
    case kSus2TensorBlockVecVecToSym2: {
      for (int x = 0; x < 3; ++x) {
        for (int y = x; y < 3; ++y) {
          l3k3_tensor_block_store(
            moments, N, d, l3k3_sym2_offset(x, y), atom,
            l3k3_tensor_block_moment(moments, N, a, x, atom) *
              l3k3_tensor_block_moment(moments, N, b, y, atom));
        }
      }
      return true;
    }
    case kSus2TensorBlockSym2MatScalar: {
      RealT sum = static_cast<RealT>(0);
      const int a_offsets[9] = {0, 1, 1, 2, 2, 3, 4, 4, 5};
      const int b_offsets[9] = {0, 1, 3, 2, 6, 4, 5, 7, 8};
      for (int row = 0; row < 9; ++row) {
        sum += l3k3_tensor_block_moment(moments, N, a, a_offsets[row], atom) *
               l3k3_tensor_block_moment(moments, N, b, b_offsets[row], atom);
      }
      l3k3_tensor_block_store(moments, N, d, 0, atom, sum);
      return true;
    }
    case kSus2TensorBlockMatSym2Scalar: {
      RealT sum = static_cast<RealT>(0);
      for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 3; ++y) {
          sum += l3k3_tensor_block_moment(moments, N, a, l3k3_mat_offset(x, y), atom) *
                 l3k3_tensor_block_moment(moments, N, b, l3k3_sym2_offset(x, y), atom);
        }
      }
      l3k3_tensor_block_store(moments, N, d, 0, atom, sum);
      return true;
    }
    case kSus2TensorBlockMatMatSameScalar:
    case kSus2TensorBlockMatMatTransScalar: {
      RealT sum = static_cast<RealT>(0);
      for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 3; ++y) {
          const int b_offset = kind == kSus2TensorBlockMatMatSameScalar
            ? l3k3_mat_offset(x, y)
            : l3k3_mat_offset(y, x);
          sum += l3k3_tensor_block_moment(moments, N, a, l3k3_mat_offset(x, y), atom) *
                 l3k3_tensor_block_moment(moments, N, b, b_offset, atom);
        }
      }
      l3k3_tensor_block_store(moments, N, d, 0, atom, sum);
      return true;
    }
    case kSus2TensorBlockStructured:
      return l3k3_tensor_block_forward_structured(N, atom, model, op_offset, moments);
    default:
      return false;
  }
}

template <typename RealT, typename GradT>
__device__ __forceinline__ bool l3k3_tensor_block_backward_fast(
  int N,
  int atom,
  const SUS2DeviceModel& model,
  int op_offset,
  const RealT* moments,
  GradT* grads)
{
  const int component_count = model.l3k3_tensor_block_ops[op_offset + 1];
  const int kind = model.l3k3_tensor_block_ops[op_offset + 2];
  const int a = model.l3k3_tensor_block_ops[op_offset + 3];
  const int b = model.l3k3_tensor_block_ops[op_offset + 4];
  const int d = model.l3k3_tensor_block_ops[op_offset + 5];

  switch (kind) {
    case kSus2TensorBlockScalarScalar: {
      const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, 0, atom);
      l3k3_tensor_block_add_row_grad(moments, grads, N, atom, a, 0, b, 0, gd, 1);
      return true;
    }
    case kSus2TensorBlockScalarTensor: {
      for (int c = component_count - 1; c >= 0; --c) {
        const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, c, atom);
        l3k3_tensor_block_add_row_grad(moments, grads, N, atom, a, 0, b, c, gd, 1);
      }
      return true;
    }
    case kSus2TensorBlockTensorScalar: {
      for (int c = component_count - 1; c >= 0; --c) {
        const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, c, atom);
        l3k3_tensor_block_add_row_grad(moments, grads, N, atom, a, c, b, 0, gd, 1);
      }
      return true;
    }
    case kSus2TensorBlockDot11: {
      const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, 0, atom);
      for (int x = 2; x >= 0; --x) {
        l3k3_tensor_block_add_row_grad(moments, grads, N, atom, a, x, b, x, gd, 1);
      }
      return true;
    }
    case kSus2TensorBlockDot22: {
      const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, 0, atom);
      const int mult[6] = {1, 2, 2, 1, 2, 1};
      for (int offset = 5; offset >= 0; --offset) {
        l3k3_tensor_block_add_row_grad(
          moments, grads, N, atom, a, offset, b, offset, gd, mult[offset]);
      }
      return true;
    }
    case kSus2TensorBlockDot33: {
      const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, 0, atom);
      const int mult[10] = {1, 3, 3, 3, 6, 3, 1, 3, 3, 1};
      for (int offset = 9; offset >= 0; --offset) {
        l3k3_tensor_block_add_row_grad(
          moments, grads, N, atom, a, offset, b, offset, gd, mult[offset]);
      }
      return true;
    }
    case kSus2TensorBlockVecSym2ToVec: {
      for (int y = 2; y >= 0; --y) {
        const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, y, atom);
        for (int x = 2; x >= 0; --x) {
          l3k3_tensor_block_add_row_grad(
            moments, grads, N, atom, a, x, b, l3k3_sym2_offset(x, y), gd, 1);
        }
      }
      return true;
    }
    case kSus2TensorBlockSym2VecToVec: {
      for (int x = 2; x >= 0; --x) {
        const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, x, atom);
        for (int y = 2; y >= 0; --y) {
          l3k3_tensor_block_add_row_grad(
            moments, grads, N, atom, a, l3k3_sym2_offset(x, y), b, y, gd, 1);
        }
      }
      return true;
    }
    case kSus2TensorBlockVecSym3ToSym2: {
      for (int out = 5; out >= 0; --out) {
        const int y = l3k3_sym2_first(out);
        const int z = l3k3_sym2_second(out);
        const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, out, atom);
        for (int x = 2; x >= 0; --x) {
          l3k3_tensor_block_add_row_grad(
            moments, grads, N, atom, a, x, b, l3k3_sym3_offset(x, y, z), gd, 1);
        }
      }
      return true;
    }
    case kSus2TensorBlockSym3VecToSym2: {
      for (int out = 5; out >= 0; --out) {
        const int x = l3k3_sym2_first(out);
        const int y = l3k3_sym2_second(out);
        const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, out, atom);
        for (int z = 2; z >= 0; --z) {
          l3k3_tensor_block_add_row_grad(
            moments, grads, N, atom, a, l3k3_sym3_offset(x, y, z), b, z, gd, 1);
        }
      }
      return true;
    }
    case kSus2TensorBlockSym2Sym3ToVec: {
      const int b_offsets[3][6] = {
        {0, 1, 2, 3, 4, 5},
        {1, 3, 4, 6, 7, 8},
        {2, 4, 5, 7, 8, 9}};
      const int mult[6] = {1, 2, 2, 1, 2, 1};
      for (int z = 2; z >= 0; --z) {
        const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, z, atom);
        for (int row = 5; row >= 0; --row) {
          l3k3_tensor_block_add_row_grad(
            moments, grads, N, atom, a, row, b, b_offsets[z][row], gd, mult[row]);
        }
      }
      return true;
    }
    case kSus2TensorBlockSym3Sym2ToVec: {
      const int a_offsets[3][6] = {
        {0, 1, 2, 3, 4, 5},
        {1, 3, 4, 6, 7, 8},
        {2, 4, 5, 7, 8, 9}};
      const int mult[6] = {1, 2, 2, 1, 2, 1};
      for (int x = 2; x >= 0; --x) {
        const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, x, atom);
        for (int row = 5; row >= 0; --row) {
          l3k3_tensor_block_add_row_grad(
            moments, grads, N, atom, a, a_offsets[x][row], b, row, gd, mult[row]);
        }
      }
      return true;
    }
    case kSus2TensorBlockSym2Sym2ToSym2: {
      for (int out = 5; out >= 0; --out) {
        const int x = l3k3_sym2_first(out);
        const int y = l3k3_sym2_second(out);
        const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, out, atom);
        for (int z = 2; z >= 0; --z) {
          l3k3_tensor_block_add_row_grad(
            moments, grads, N, atom, a, l3k3_sym2_offset(x, z), b, l3k3_sym2_offset(y, z), gd, 1);
        }
      }
      return true;
    }
    case kSus2TensorBlockSym2Sym2ToMatAB:
    case kSus2TensorBlockSym2Sym2ToMatBA: {
      for (int out = 8; out >= 0; --out) {
        const int x = out / 3;
        const int y = out - x * 3;
        const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, out, atom);
        for (int z = 2; z >= 0; --z) {
          const int a_offset = kind == kSus2TensorBlockSym2Sym2ToMatAB
            ? l3k3_sym2_offset(x, z)
            : l3k3_sym2_offset(y, z);
          const int b_offset = kind == kSus2TensorBlockSym2Sym2ToMatAB
            ? l3k3_sym2_offset(y, z)
            : l3k3_sym2_offset(x, z);
          l3k3_tensor_block_add_row_grad(
            moments, grads, N, atom, a, a_offset, b, b_offset, gd, 1);
        }
      }
      return true;
    }
    case kSus2TensorBlockVecVecOuterAB:
    case kSus2TensorBlockVecVecOuterBA: {
      for (int out = 8; out >= 0; --out) {
        const int x = out / 3;
        const int y = out - x * 3;
        const int a_offset = kind == kSus2TensorBlockVecVecOuterAB ? x : y;
        const int b_offset = kind == kSus2TensorBlockVecVecOuterAB ? y : x;
        const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, out, atom);
        l3k3_tensor_block_add_row_grad(
          moments, grads, N, atom, a, a_offset, b, b_offset, gd, 1);
      }
      return true;
    }
    case kSus2TensorBlockVecVecToSym2: {
      for (int out = 5; out >= 0; --out) {
        const int x = l3k3_sym2_first(out);
        const int y = l3k3_sym2_second(out);
        const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, out, atom);
        l3k3_tensor_block_add_row_grad(moments, grads, N, atom, a, x, b, y, gd, 1);
      }
      return true;
    }
    case kSus2TensorBlockSym2MatScalar: {
      const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, 0, atom);
      const int a_offsets[9] = {0, 1, 1, 2, 2, 3, 4, 4, 5};
      const int b_offsets[9] = {0, 1, 3, 2, 6, 4, 5, 7, 8};
      for (int row = 8; row >= 0; --row) {
        l3k3_tensor_block_add_row_grad(
          moments, grads, N, atom, a, a_offsets[row], b, b_offsets[row], gd, 1);
      }
      return true;
    }
    case kSus2TensorBlockMatSym2Scalar: {
      const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, 0, atom);
      for (int row = 8; row >= 0; --row) {
        const int x = row / 3;
        const int y = row - x * 3;
        l3k3_tensor_block_add_row_grad(
          moments, grads, N, atom, a, row, b, l3k3_sym2_offset(x, y), gd, 1);
      }
      return true;
    }
    case kSus2TensorBlockMatMatSameScalar:
    case kSus2TensorBlockMatMatTransScalar: {
      const RealT gd = l3k3_tensor_block_grad<RealT>(grads, N, d, 0, atom);
      for (int row = 8; row >= 0; --row) {
        const int x = row / 3;
        const int y = row - x * 3;
        const int b_offset = kind == kSus2TensorBlockMatMatSameScalar
          ? row
          : l3k3_mat_offset(y, x);
        l3k3_tensor_block_add_row_grad(
          moments, grads, N, atom, a, row, b, b_offset, gd, 1);
      }
      return true;
    }
    case kSus2TensorBlockStructured:
      return l3k3_tensor_block_backward_structured<RealT, GradT>(
        N, atom, model, op_offset, moments, grads);
    default:
      return false;
  }
}

template <typename RealT, typename GradT>
static __global__ void gpu_l3k3_tensor_block_energy_backward(
  int N,
  SUS2DeviceModel model,
  const int* type,
  RealT* moments,
  GradT* grads,
  double* potential)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || model.l3k3_tensor_block_op_count <= 0 ||
      model.l3k3_tensor_block_ops == nullptr || model.l3k3_tensor_block_op_rows == nullptr ||
      model.l3k3_tensor_block_rows == nullptr) {
    return;
  }

  for (int op = 0; op < model.l3k3_tensor_block_op_count; ++op) {
    const int op_offset = op * kSus2TensorBlockOpInts;
    if (model.use_l3k3_tensor_block_fast_forward &&
        l3k3_tensor_block_forward_fast(N, i, model, op_offset, moments)) {
      continue;
    }
    const int component_count = model.l3k3_tensor_block_ops[op_offset + 1];
    const int a = model.l3k3_tensor_block_ops[op_offset + 3];
    const int b = model.l3k3_tensor_block_ops[op_offset + 4];
    const int d = model.l3k3_tensor_block_ops[op_offset + 5];
    const int row_op_offset = op * kSus2TensorBlockOpRowInts;
    const int row_begin = model.l3k3_tensor_block_op_rows[row_op_offset + 0];
    const int row_count = model.l3k3_tensor_block_op_rows[row_op_offset + 1];
    for (int component = 0; component < component_count; ++component) {
      moments[static_cast<size_t>(d + component) * N + i] = static_cast<RealT>(0.0);
    }
    for (int row = 0; row < row_count; ++row) {
      const int row_offset = (row_begin + row) * kSus2TensorBlockRowWords;
      const unsigned int word0 = l3k3_tensor_block_row_word(model, row_offset + 0);
      const unsigned int word1 = l3k3_tensor_block_row_word(model, row_offset + 1);
      const int dst = d + static_cast<int>(word0 & 0xffffu);
      const int src0 = a + static_cast<int>(word0 >> 16);
      const int src1 = b + static_cast<int>(word1 & 0xffffu);
      const int mult = static_cast<int>(word1 >> 16);
      moments[static_cast<size_t>(dst) * N + i] +=
        static_cast<RealT>(mult) * moments[static_cast<size_t>(src0) * N + i] *
        moments[static_cast<size_t>(src1) * N + i];
    }
  }

  sus2_init_site_energy_and_scalar_grads(N, model, i, type, moments, grads, potential);

  for (int op = model.l3k3_tensor_block_op_count - 1; op >= 0; --op) {
    const int op_offset = op * kSus2TensorBlockOpInts;
    if (model.use_l3k3_tensor_block_fast_backward &&
        l3k3_tensor_block_backward_fast<RealT, GradT>(N, i, model, op_offset, moments, grads)) {
      continue;
    }
    const int a = model.l3k3_tensor_block_ops[op_offset + 3];
    const int b = model.l3k3_tensor_block_ops[op_offset + 4];
    const int d = model.l3k3_tensor_block_ops[op_offset + 5];
    const int row_op_offset = op * kSus2TensorBlockOpRowInts;
    const int row_begin = model.l3k3_tensor_block_op_rows[row_op_offset + 0];
    const int row_count = model.l3k3_tensor_block_op_rows[row_op_offset + 1];
    for (int row = row_count - 1; row >= 0; --row) {
      const int row_offset = (row_begin + row) * kSus2TensorBlockRowWords;
      const unsigned int word0 = l3k3_tensor_block_row_word(model, row_offset + 0);
      const unsigned int word1 = l3k3_tensor_block_row_word(model, row_offset + 1);
      const int dst = d + static_cast<int>(word0 & 0xffffu);
      const int src0 = a + static_cast<int>(word0 >> 16);
      const int src1 = b + static_cast<int>(word1 & 0xffffu);
      const int mult = static_cast<int>(word1 >> 16);
      const RealT dst_grad =
        static_cast<RealT>(load_sus2_grad(grads, N, dst, i));
      const RealT gdst = dst_grad * static_cast<RealT>(mult);
      add_sus2_grad(
        grads, static_cast<size_t>(src1) * N + i, gdst * moments[static_cast<size_t>(src0) * N + i]);
      add_sus2_grad(
        grads, static_cast<size_t>(src0) * N + i, gdst * moments[static_cast<size_t>(src1) * N + i]);
    }
  }
}

template <typename GradT, typename RealT>
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
    load_sus2_edge_displacement(
      use_cached_displacements, N, box, edge, i, j, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, dx, dy, dz);
    const RealT r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= static_cast<RealT>(cutoff_square)) {
      continue;
    }
    const RealT r = sqrt(r2);
    const int type_j = type[j];

    RealT dEx = static_cast<RealT>(0.0);
    RealT dEy = static_cast<RealT>(0.0);
    RealT dEz = static_cast<RealT>(0.0);
    compute_sus2_edge_derivative<GradT, RealT>(
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

  store_sus2_self_force(N, i, fx_self, fy_self, fz_self, force_tmp, force_self_tmp);

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

template <typename GradT, typename RealT>
static __global__ void gpu_compute_forces_l3k3_cached_grads(
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
  float* force_self_tmp,
  float* virial_tmp)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  const int type_i = type[i];
  const RealT center_coeff = sus2_species_coeff<RealT>(model, type_i);
  RealT basic_grads[60];
#pragma unroll
  for (int basic = 0; basic < 60; ++basic) {
    basic_grads[basic] =
      static_cast<RealT>(load_sus2_grad(grads, N, basic, i)) * center_coeff;
  }

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
    load_sus2_edge_displacement(
      use_cached_displacements, N, box, edge, i, j, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, dx, dy, dz);
    const RealT r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= static_cast<RealT>(cutoff_square)) {
      continue;
    }
    const RealT r = sqrt(r2);
    const int type_j = type[j];

    RealT dEx = static_cast<RealT>(0.0);
    RealT dEy = static_cast<RealT>(0.0);
    RealT dEz = static_cast<RealT>(0.0);
    compute_sus2_edge_derivative_l3k3_cached<RealT>(
      model, type_i, type_j, dx, dy, dz, r, basic_grads, dEx, dEy, dEz);

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

  store_sus2_self_force(N, i, fx_self, fy_self, fz_self, force_tmp, force_self_tmp);

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

template <typename GradT, typename RealT, int MaxBasic>
static __global__ void gpu_compute_forces_tensor_cached_grads(
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
  float* force_self_tmp,
  float* virial_tmp)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || model.alpha_basic_count > MaxBasic) {
    return;
  }

  const int type_i = type[i];
  const RealT center_coeff = sus2_species_coeff<RealT>(model, type_i);
  RealT basic_grads[MaxBasic];
  for (int basic = 0; basic < model.alpha_basic_count; ++basic) {
    basic_grads[basic] =
      static_cast<RealT>(load_sus2_grad(grads, N, basic, i)) * center_coeff;
  }

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
    load_sus2_edge_displacement(
      use_cached_displacements, N, box, edge, i, j, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, dx, dy, dz);
    const RealT r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= static_cast<RealT>(cutoff_square)) {
      continue;
    }
    const RealT r = sqrt(r2);
    const int type_j = type[j];

    RealT dEx = static_cast<RealT>(0.0);
    RealT dEy = static_cast<RealT>(0.0);
    RealT dEz = static_cast<RealT>(0.0);
    compute_sus2_edge_derivative_tensor_cached<RealT, MaxBasic>(
      model, type_i, type_j, dx, dy, dz, r, basic_grads, dEx, dEy, dEz);

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

  store_sus2_self_force(N, i, fx_self, fy_self, fz_self, force_tmp, force_self_tmp);

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

template <typename GradT, typename RealT, int L, int K>
static __global__ void gpu_compute_forces_tensor_cached_grads_static(
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
  float* force_self_tmp,
  float* virial_tmp)
{
  constexpr int BasicPerGroup = Sus2TensorStaticLayout<L>::basic_per_group;
  constexpr int BasicCount = K * BasicPerGroup;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || model.alpha_basic_count != BasicCount) {
    return;
  }

  const int type_i = type[i];
  const RealT center_coeff = sus2_species_coeff<RealT>(model, type_i);
  RealT basic_grads[BasicCount];
#pragma unroll
  for (int basic = 0; basic < BasicCount; ++basic) {
    basic_grads[basic] =
      static_cast<RealT>(load_sus2_grad(grads, N, basic, i)) * center_coeff;
  }

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
    load_sus2_edge_displacement(
      use_cached_displacements, N, box, edge, i, j, neighbor_dx, neighbor_dy, neighbor_dz, x, y, z, dx, dy, dz);
    const RealT r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= static_cast<RealT>(cutoff_square)) {
      continue;
    }
    const RealT r = sqrt(r2);
    const int type_j = type[j];

    RealT dEx = static_cast<RealT>(0.0);
    RealT dEy = static_cast<RealT>(0.0);
    RealT dEz = static_cast<RealT>(0.0);
    compute_sus2_edge_derivative_tensor_cached_static<RealT, L, K>(
      model, type_i, type_j, dx, dy, dz, r, basic_grads, dEx, dEy, dEz);

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

  store_sus2_self_force(N, i, fx_self, fy_self, fz_self, force_tmp, force_self_tmp);

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

template <typename GradT, typename RealT>
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

  RealT fx = static_cast<RealT>(0.0);
  RealT fy = static_cast<RealT>(0.0);
  RealT fz = static_cast<RealT>(0.0);
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
    RealT dx = static_cast<RealT>(x[j] - x[i]);
    RealT dy = static_cast<RealT>(y[j] - y[i]);
    RealT dz = static_cast<RealT>(z[j] - z[i]);
    apply_mic(box, dx, dy, dz);
    const RealT r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= static_cast<RealT>(cutoff_square)) {
      continue;
    }

    const RealT r = sqrt(r2);
    const int type_j = type[j];

    RealT dFix;
    RealT dFiy;
    RealT dFiz;
    compute_sus2_edge_derivative<GradT, RealT>(
      N, model, i, type_i, type_j, dx, dy, dz, r, grads, dFix, dFiy, dFiz);

    RealT dFjx;
    RealT dFjy;
    RealT dFjz;
    compute_sus2_edge_derivative<GradT, RealT>(
      N, model, j, type_j, type_i, -dx, -dy, -dz, r, grads, dFjx, dFjy, dFjz);

    fx += dFix - dFjx;
    fy += dFiy - dFjy;
    fz += dFiz - dFjz;

    s_xx -= dFix * dx;
    s_yy -= dFiy * dy;
    s_zz -= dFiz * dz;
    s_xy -= dx * dFiy;
    s_xz -= dx * dFiz;
    s_yz -= dy * dFiz;
    s_yx -= dy * dFix;
    s_zx -= dz * dFix;
    s_zy -= dz * dFiy;
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

} // namespace

SUS2_V11::SUS2_V11(
  const char* file_potential,
  int num_atoms,
  int num_potential_options,
  const char** potential_options)
{
  const SUS2HostModel host_model = load_model(file_potential);
  species_count_ = host_model.species_count;
  angular_channels_ = host_model.angular_channels;
  radial_basis_kind_ = static_cast<int>(host_model.radial_basis_kind);
  radial_funcs_count_ = host_model.radial_funcs_count;
  rb_size_ = host_model.rb_size;
  alpha_basic_count_ = host_model.alpha_basic_count;
  alpha_times_count_ = host_model.alpha_times_count;
  alpha_moments_count_ = host_model.alpha_moments_count;
  alpha_scalar_moments_ = host_model.alpha_scalar_moments;
  max_rank_ = host_model.max_rank;
  rc = host_model.max_dist;
  const TensorBasicLayout tensor_layout = detect_tensor_alpha_basic_layout(host_model);
  use_tensor_basic_fastpath_ = tensor_layout.enabled;
  tensor_l_ = tensor_layout.l;
  tensor_k_ = tensor_layout.k;
  tensor_basic_per_group_ = tensor_layout.basic_per_group;
  use_const_alpha_times_ = can_pack_alpha_times_u16(host_model);
  use_const_scalar_moments_ = can_pack_scalar_moments_u16(host_model);

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
  use_force_self_buffer_ =
    parse_force_self_buffer(host_model, num_potential_options, potential_options);
  use_float_moments_ = parse_float_moments(host_model, num_potential_options, potential_options);
  use_float_moment_grads_ =
    use_float_moments_ || parse_float_moment_grads(host_model, num_potential_options, potential_options);
  use_fused_energy_backward_ =
    parse_fused_energy_backward(host_model, num_potential_options, potential_options);
  use_fused_graph_ = parse_fused_graph(host_model, num_potential_options, potential_options);
  use_product_assign_ =
    parse_product_assign(host_model, num_potential_options, potential_options);
  const bool request_graph_specific_product =
    parse_graph_specific_product(host_model, num_potential_options, potential_options);
  use_local_product_graph_ =
    parse_local_product_graph(host_model, num_potential_options, potential_options);
  use_tensor_force_grad_cache_ =
    parse_tensor_force_grad_cache(host_model, num_potential_options, potential_options);
  use_radial_direct_ = parse_radial_direct(host_model, num_potential_options, potential_options);
  const bool request_l3k3_tensor_scalar =
    parse_l3k3_tensor_scalar(host_model, num_potential_options, potential_options);
  const bool request_l3k3_tensor_block =
    parse_l3k3_tensor_block(host_model, num_potential_options, potential_options);
  const bool request_tensor_auto =
    parse_tensor_auto(host_model, num_potential_options, potential_options);
  use_l3k3_tensor_block_fast_forward_ =
    parse_l3k3_tensor_block_fast_forward(host_model, num_potential_options, potential_options);
  use_l3k3_tensor_block_fast_backward_ =
    parse_l3k3_tensor_block_fast_backward(host_model, num_potential_options, potential_options);
  L3K3TensorScalarPlan l3k3_tensor_scalar_plan;
  L3K3TensorBlockPlan l3k3_tensor_block_plan;
  TensorAutoDecision tensor_auto_decision;
  if ((request_l3k3_tensor_scalar && request_l3k3_tensor_block) ||
      (request_tensor_auto && (request_l3k3_tensor_scalar || request_l3k3_tensor_block))) {
    sus2_input_error(
      "SUS2 v1.1 tensor_auto, tensor_scalar, and tensor_block path requests are mutually exclusive.");
  }
  if (request_l3k3_tensor_block) {
    l3k3_tensor_block_plan = build_l3k3_tensor_block_plan(host_model);
    use_l3k3_tensor_block_ = l3k3_tensor_block_plan.enabled;
    l3k3_tensor_block_op_count_ = l3k3_tensor_block_plan.op_count;
  } else if (request_tensor_auto) {
    l3k3_tensor_block_plan = build_l3k3_tensor_block_plan(host_model);
    tensor_auto_decision = choose_tensor_auto_plan(host_model, l3k3_tensor_block_plan);
    use_l3k3_tensor_block_ = tensor_auto_decision.use_tensor_block;
    l3k3_tensor_block_op_count_ =
      use_l3k3_tensor_block_ ? l3k3_tensor_block_plan.op_count : 0;
  } else if (request_l3k3_tensor_scalar) {
    l3k3_tensor_scalar_plan = build_l3k3_tensor_scalar_plan(host_model);
    use_l3k3_tensor_scalar_ = l3k3_tensor_scalar_plan.enabled;
    l3k3_tensor_scalar_term_count_ =
      static_cast<int>(l3k3_tensor_scalar_plan.coeffs.size());
  }
  if (use_l3k3_tensor_scalar_ || use_l3k3_tensor_block_) {
    use_tensor_force_grad_cache_ = true;
    use_pairwise_no_atomic_force_ = false;
  }
  use_local_product_graph_ = use_local_product_graph_ && use_float_moments_ &&
                             alpha_moments_count_ <= kSus2LocalGraphMaxMoments &&
                             !use_l3k3_tensor_scalar_ && !use_l3k3_tensor_block_;
  product_assign_supported_ = supports_product_assign(host_model);
  use_product_assign_ = use_product_assign_ && product_assign_supported_ &&
                        use_fused_graph_ && !use_local_product_graph_ &&
                        use_tensor_basic_fastpath_ && !use_l3k3_tensor_scalar_ &&
                        !use_l3k3_tensor_block_;
  use_graph_specific_product_ =
    request_graph_specific_product && use_product_assign_ && use_const_alpha_times_;
  use_const_float_coeffs_ = use_float_moments_ &&
                            host_model.species_count <= kSus2MaxConstSpecies &&
                            host_model.alpha_scalar_moments <= kSus2MaxConstScalarMoments;

  shift_coeffs_.resize(host_model.shift_coeffs.size());
  shift_coeffs_.copy_from_host(host_model.shift_coeffs.data());
  species_coeffs_.resize(host_model.species_coeffs.size());
  species_coeffs_.copy_from_host(host_model.species_coeffs.data());
  moment_coeffs_.resize(host_model.moment_coeffs.size());
  moment_coeffs_.copy_from_host(host_model.moment_coeffs.data());
  if (use_float_moments_) {
    std::vector<float> host_shift_coeffs_float(host_model.shift_coeffs.begin(), host_model.shift_coeffs.end());
    std::vector<float> host_species_coeffs_float(host_model.species_coeffs.begin(), host_model.species_coeffs.end());
    std::vector<float> host_moment_coeffs_float(host_model.moment_coeffs.begin(), host_model.moment_coeffs.end());
    shift_coeffs_float_.resize(host_shift_coeffs_float.size());
    shift_coeffs_float_.copy_from_host(host_shift_coeffs_float.data());
    species_coeffs_float_.resize(host_species_coeffs_float.size());
    species_coeffs_float_.copy_from_host(host_species_coeffs_float.data());
    moment_coeffs_float_.resize(host_moment_coeffs_float.size());
    moment_coeffs_float_.copy_from_host(host_moment_coeffs_float.data());
    if (use_const_float_coeffs_) {
      CHECK(gpuMemcpyToSymbol(
        c_sus2_shift_coeffs_float,
        host_shift_coeffs_float.data(),
        host_shift_coeffs_float.size() * sizeof(float)));
      CHECK(gpuMemcpyToSymbol(
        c_sus2_species_coeffs_float,
        host_species_coeffs_float.data(),
        host_species_coeffs_float.size() * sizeof(float)));
      CHECK(gpuMemcpyToSymbol(
        c_sus2_moment_coeffs_float,
        host_moment_coeffs_float.data(),
        host_moment_coeffs_float.size() * sizeof(float)));
    }
  }
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
  std::vector<int> packed_alpha_time_groups;
  packed_alpha_time_groups.reserve(static_cast<size_t>(host_model.alpha_times_count) * 3);
  int alpha_time_cursor = 0;
  while (alpha_time_cursor < host_model.alpha_times_count) {
    const int begin = alpha_time_cursor;
    const int dst = host_model.alpha_times[alpha_time_cursor * 4 + 3];
    do {
      ++alpha_time_cursor;
    } while (alpha_time_cursor < host_model.alpha_times_count &&
             host_model.alpha_times[alpha_time_cursor * 4 + 3] == dst);
    const int len = alpha_time_cursor - begin;
    packed_alpha_time_groups.push_back(begin);
    packed_alpha_time_groups.push_back(len);
    packed_alpha_time_groups.push_back(dst);
  }
  alpha_time_group_count_ = static_cast<int>(packed_alpha_time_groups.size() / 3);
  alpha_time_groups_.resize(packed_alpha_time_groups.size());
  alpha_time_groups_.copy_from_host(packed_alpha_time_groups.data());
  if (use_graph_specific_product_) {
    if (can_pack_alpha_time_groups_u16(packed_alpha_time_groups)) {
      std::vector<unsigned int> group_pairs =
        pack_alpha_time_group_pairs(packed_alpha_time_groups);
      alpha_time_group_pairs_.resize(group_pairs.size());
      alpha_time_group_pairs_.copy_from_host(group_pairs.data());
    } else {
      use_graph_specific_product_ = false;
    }
  }
  alpha_moment_mapping_.resize(host_model.alpha_moment_mapping.size());
  alpha_moment_mapping_.copy_from_host(host_model.alpha_moment_mapping.data());
  if (use_const_scalar_moments_) {
    std::vector<unsigned short> packed_mapping(host_model.alpha_moment_mapping.size());
    for (size_t i = 0; i < host_model.alpha_moment_mapping.size(); ++i) {
      packed_mapping[i] = static_cast<unsigned short>(host_model.alpha_moment_mapping[i]);
    }
    CHECK(gpuMemcpyToSymbol(
      c_sus2_alpha_moment_mapping_u16,
      packed_mapping.data(),
      packed_mapping.size() * sizeof(unsigned short)));
  }
  if (use_l3k3_tensor_scalar_) {
    l3k3_tensor_scalar_terms_.resize(l3k3_tensor_scalar_plan.terms.size());
    l3k3_tensor_scalar_terms_.copy_from_host(l3k3_tensor_scalar_plan.terms.data());
    l3k3_tensor_scalar_coeffs_.resize(l3k3_tensor_scalar_plan.coeffs.size());
    l3k3_tensor_scalar_coeffs_.copy_from_host(l3k3_tensor_scalar_plan.coeffs.data());
    if (use_float_moments_) {
      std::vector<float> host_l3k3_tensor_scalar_coeffs_float(
        l3k3_tensor_scalar_plan.coeffs.begin(), l3k3_tensor_scalar_plan.coeffs.end());
      l3k3_tensor_scalar_coeffs_float_.resize(host_l3k3_tensor_scalar_coeffs_float.size());
      l3k3_tensor_scalar_coeffs_float_.copy_from_host(
        host_l3k3_tensor_scalar_coeffs_float.data());
    }
  }
  if (use_l3k3_tensor_block_) {
    l3k3_tensor_block_ops_.resize(l3k3_tensor_block_plan.ops.size());
    l3k3_tensor_block_ops_.copy_from_host(l3k3_tensor_block_plan.ops.data());
    l3k3_tensor_block_op_rows_.resize(l3k3_tensor_block_plan.op_rows.size());
    l3k3_tensor_block_op_rows_.copy_from_host(l3k3_tensor_block_plan.op_rows.data());
    l3k3_tensor_block_metas_.resize(l3k3_tensor_block_plan.metas.size());
    l3k3_tensor_block_metas_.copy_from_host(l3k3_tensor_block_plan.metas.data());
    l3k3_tensor_block_rows_.resize(l3k3_tensor_block_plan.rows.size());
    l3k3_tensor_block_rows_.copy_from_host(l3k3_tensor_block_plan.rows.data());
    l3k3_tensor_block_row_count_ = l3k3_tensor_block_plan.row_count;
  }

  if (use_radial_direct_) {
    std::vector<float> host_direct_coeffs;
    std::vector<float> host_direct_scal_s;
    build_direct_radial_tables(host_model, host_direct_coeffs, host_direct_scal_s);
    radial_direct_coeffs_.resize(host_direct_coeffs.size());
    radial_direct_scal_s_.resize(host_direct_scal_s.size());
    radial_direct_coeffs_.copy_from_host(host_direct_coeffs.data());
    radial_direct_scal_s_.copy_from_host(host_direct_scal_s.data());
    lut_size_ = 0;
    lut_inv_dr_ = 0.0;
  } else {
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
  }

  neighbor_count_.resize(num_atoms);
  cell_contents_.resize(num_atoms);
  neighbor_cache_.initialize(rc, num_atoms, 512);
  resize_work_buffers(num_atoms);

  if (use_radial_direct_) {
    printf(
      "Use SUS2 v1.1 GPUMD potential: radial_type=%s, species=%d, radial=%d, basics=%d, moments=%d, scalars=%d, cutoff=%g A, radial_eval=direct basis recurrence.\n",
      host_model.radial_basis_type.c_str(),
      species_count_,
      radial_funcs_count_,
      alpha_basic_count_,
      alpha_moments_count_,
      alpha_scalar_moments_,
      rc);
  } else {
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
  }
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
  if (use_force_self_buffer_ && !use_pairwise_no_atomic_force_) {
    printf("SUS2 v1.1 GPUMD force mode: self-force buffer avoids per-atom self atomic adds.\n");
  }
  if (use_tensor_basic_fastpath_) {
    printf(
      "SUS2 v1.1 GPUMD basic/force fast path: tensor l%dk%d alpha_index_basic layout.\n",
      tensor_l_,
      tensor_k_);
  }
  if (request_tensor_auto) {
    printf(
      "SUS2 v1.1 GPUMD tensor planner: auto requested, selected=%s, %s.\n",
      use_l3k3_tensor_block_ ? "tensor-block" : "product-graph",
      tensor_auto_decision.reason.c_str());
  }
  if (use_const_alpha_times_ && !use_l3k3_tensor_scalar_ && !use_l3k3_tensor_block_) {
    printf("SUS2 v1.1 GPUMD product-rule table: constant-memory uint16 alpha_index_times.\n");
  }
  if (use_const_scalar_moments_ && !use_l3k3_tensor_scalar_ && !use_l3k3_tensor_block_) {
    printf("SUS2 v1.1 GPUMD scalar mapping: constant-memory uint16 alpha_moment_mapping.\n");
  }
  if (use_const_float_coeffs_) {
    printf("SUS2 v1.1 GPUMD float coefficients: constant-memory shift/species/moment coeffs.\n");
  }
  if (use_fused_energy_backward_ && !use_l3k3_tensor_scalar_ && !use_l3k3_tensor_block_) {
    printf("SUS2 v1.1 GPUMD reverse path: fused site-energy gradient and product backward kernel.\n");
  }
  if (use_fused_graph_ && !use_local_product_graph_ && !use_l3k3_tensor_scalar_ &&
      !use_l3k3_tensor_block_) {
    printf("SUS2 v1.1 GPUMD product graph path: fused forward/site-gradient/backward kernel.\n");
  }
  if (use_product_assign_) {
    printf(
      "SUS2 v1.1 GPUMD product graph micro-optimization: product moments use assign-forward and skip moment-value memset.\n");
    if (use_graph_specific_product_) {
      printf(
        "SUS2 v1.1 GPUMD graph-specific product path: packed product-group plan, groups=%d, product_rules=%d.\n",
        alpha_time_group_count_,
        alpha_times_count_);
    }
  } else if (
    parse_product_assign(host_model, num_potential_options, potential_options) &&
    !use_l3k3_tensor_scalar_ && !use_l3k3_tensor_block_) {
    printf(
      "SUS2 v1.1 GPUMD product graph micro-optimization: product assign requested but unsupported for this path (supported=%s, fused=%s, local=%s, tensor_basic=%s).\n",
      product_assign_supported_ ? "yes" : "no",
      use_fused_graph_ ? "yes" : "no",
      use_local_product_graph_ ? "yes" : "no",
      use_tensor_basic_fastpath_ ? "yes" : "no");
  }
  if (request_graph_specific_product && !use_graph_specific_product_ &&
      !use_l3k3_tensor_scalar_ && !use_l3k3_tensor_block_) {
    printf(
      "SUS2 v1.1 GPUMD graph-specific product path: requested but unsupported; using mature product graph (product_assign=%s, const_u16_rules=%s).\n",
      use_product_assign_ ? "yes" : "no",
      use_const_alpha_times_ ? "yes" : "no");
  }
  if (use_local_product_graph_) {
    printf(
      "SUS2 v1.1 GPUMD product graph path: local per-atom graph workspace, write basic gradients only.\n");
  }
  if (use_l3k3_tensor_scalar_) {
    printf(
      "SUS2 v1.1 GPUMD tensor-scalar path: l3k3 compiled polynomial plan, terms=%d, max_degree=%d.\n",
      l3k3_tensor_scalar_term_count_,
      l3k3_tensor_scalar_plan.max_degree);
  } else if (request_l3k3_tensor_scalar) {
    printf(
      "SUS2 v1.1 GPUMD tensor-scalar path: requested but unsupported for this model; using product graph fallback.\n");
  }
  if (use_l3k3_tensor_block_) {
    printf(
      "SUS2 v1.1 GPUMD tensor-block path: graph-specific tensor l%dk%d block DAG, ops=%d, fast_ops=%d, generic_ops=%d, component_groups=%d, candidate_matches=%d/%d, generic_rows=%d, specific_rows=%d, cost_units=%d, fast_forward=%s, fast_backward=%s, op_histogram=%s.\n",
      tensor_l_,
      tensor_k_,
      l3k3_tensor_block_op_count_,
      l3k3_tensor_block_plan.fast_op_count,
      l3k3_tensor_block_plan.generic_op_count,
      l3k3_tensor_block_plan.component_group_count,
      l3k3_tensor_block_plan.matched_candidate_count,
      l3k3_tensor_block_plan.candidate_count,
      l3k3_tensor_block_plan.generic_row_count,
      l3k3_tensor_block_plan.row_count,
      l3k3_tensor_block_plan.selected_cost_units,
      use_l3k3_tensor_block_fast_forward_ ? "yes" : "no",
      use_l3k3_tensor_block_fast_backward_ ? "yes" : "no",
      format_l3k3_tensor_block_histogram(l3k3_tensor_block_plan).c_str());
  } else if (request_l3k3_tensor_block) {
    printf(
      "SUS2 v1.1 GPUMD tensor-block path: requested but unsupported for this model; using product graph fallback.\n");
  }
  if (use_tensor_basic_fastpath_ && use_tensor_force_grad_cache_) {
    printf("SUS2 v1.1 GPUMD force path: cached tensor center basic gradients.\n");
  }
  printf(
    "SUS2 v1.1 GPUMD precision mode: %s.\n",
    use_float_moments_ ? "NEP-like float moments/gradients/local arithmetic"
                       : "double moments/local arithmetic");
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
  if (!use_float_moments_ && moment_vals_.size() != moment_size) {
    moment_vals_.resize(moment_size);
  }
  if (use_float_moments_ && moment_vals_float_.size() != moment_size) {
    moment_vals_float_.resize(moment_size);
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
  if (use_force_self_buffer_ && force_self_tmp_.size() != force_size) {
    force_self_tmp_.resize(force_size);
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
  const size_t basic_moment_size = static_cast<size_t>(alpha_basic_count_) * num_atoms;
  const bool use_pairwise_no_atomic_force =
    !use_cached_neighbor_displacements_ && use_pairwise_no_atomic_force_;
  float* force_self_tmp_ptr =
    (use_force_self_buffer_ && !use_pairwise_no_atomic_force) ? force_self_tmp_.data() : nullptr;
  const bool use_local_product_graph = use_local_product_graph_;

  profile_t = profile_start();
  if (use_l3k3_tensor_scalar_) {
    // The l3k3 tensor-scalar path overwrites the 60 basic moments and their
    // gradients directly; product moments are not materialized.
  } else if (use_l3k3_tensor_block_) {
    if (use_float_moment_grads_) {
      CHECK(gpuMemset(moment_grads_float_.data(), 0, moment_size * sizeof(float)));
    } else {
      CHECK(gpuMemset(moment_grads_.data(), 0, moment_size * sizeof(double)));
    }
  } else if (use_local_product_graph) {
    if (!use_tensor_basic_fastpath_) {
      CHECK(gpuMemset(moment_vals_float_.data(), 0, basic_moment_size * sizeof(float)));
    }
  } else {
    if (!use_product_assign_) {
      if (use_float_moments_) {
        CHECK(gpuMemset(moment_vals_float_.data(), 0, moment_size * sizeof(float)));
      } else {
        CHECK(gpuMemset(moment_vals_.data(), 0, moment_size * sizeof(double)));
      }
    } else if (!use_tensor_basic_fastpath_) {
      if (use_float_moments_) {
        CHECK(gpuMemset(moment_vals_float_.data(), 0, basic_moment_size * sizeof(float)));
      } else {
        CHECK(gpuMemset(moment_vals_.data(), 0, basic_moment_size * sizeof(double)));
      }
    }
    if (use_float_moment_grads_) {
      CHECK(gpuMemset(moment_grads_float_.data(), 0, moment_size * sizeof(float)));
    } else {
      CHECK(gpuMemset(moment_grads_.data(), 0, moment_size * sizeof(double)));
    }
  }
  if (!use_pairwise_no_atomic_force) {
    CHECK(gpuMemset(force_tmp_.data(), 0, static_cast<size_t>(num_atoms) * 3 * sizeof(float)));
  }
  profile_stop(profile_zero, profile_t);

  SUS2DeviceModel model{
    species_count_,
    angular_channels_,
    radial_basis_kind_,
    radial_funcs_count_,
    alpha_basic_count_,
    alpha_times_count_,
    alpha_time_group_count_,
    alpha_moments_count_,
    alpha_scalar_moments_,
    max_rank_,
    rb_size_,
    lut_size_,
    rc,
    lut_inv_dr_,
    shift_coeffs_.data(),
    species_coeffs_.data(),
    moment_coeffs_.data(),
    use_float_moments_ ? shift_coeffs_float_.data() : nullptr,
    use_float_moments_ ? species_coeffs_float_.data() : nullptr,
    use_float_moments_ ? moment_coeffs_float_.data() : nullptr,
    alpha_basic_.data(),
    alpha_times_.data(),
    alpha_time_group_count_ > 0 ? alpha_time_groups_.data() : nullptr,
    use_graph_specific_product_ ? alpha_time_group_pairs_.data() : nullptr,
    alpha_moment_mapping_.data(),
    l3k3_tensor_scalar_term_count_,
    use_l3k3_tensor_scalar_ ? l3k3_tensor_scalar_terms_.data() : nullptr,
    use_l3k3_tensor_scalar_ ? l3k3_tensor_scalar_coeffs_.data() : nullptr,
    (use_l3k3_tensor_scalar_ && use_float_moments_) ? l3k3_tensor_scalar_coeffs_float_.data()
                                                     : nullptr,
    l3k3_tensor_block_op_count_,
    use_l3k3_tensor_block_ ? l3k3_tensor_block_ops_.data() : nullptr,
    use_l3k3_tensor_block_ ? l3k3_tensor_block_op_rows_.data() : nullptr,
    use_l3k3_tensor_block_ ? l3k3_tensor_block_metas_.data() : nullptr,
    l3k3_tensor_block_row_count_,
    use_l3k3_tensor_block_ ? l3k3_tensor_block_rows_.data() : nullptr,
    use_radial_direct_ ? nullptr : lut_vals_.data(),
    use_radial_direct_ ? nullptr : lut_ders_.data(),
    use_radial_direct_ ? radial_direct_coeffs_.data() : nullptr,
    use_radial_direct_ ? radial_direct_scal_s_.data() : nullptr,
    use_tensor_basic_fastpath_,
    tensor_l_,
    tensor_k_,
    tensor_basic_per_group_,
    use_const_alpha_times_,
    use_const_scalar_moments_,
    use_const_float_coeffs_,
    use_float_moments_,
    use_radial_direct_,
    use_l3k3_tensor_scalar_,
    use_l3k3_tensor_block_,
    use_l3k3_tensor_block_fast_forward_,
    use_l3k3_tensor_block_fast_backward_};

#define SUS2_LAUNCH_TENSOR_BASIC_DYNAMIC(REAL_T, OUT_PTR) \
  do { \
    if (alpha_basic_count_ <= 40) { \
      gpu_compute_basic_moments_tensor_accum<REAL_T, 40><<<grid_size, kBlockSize>>>( \
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(), \
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), \
        neighbor_dz_.data(), position.data(), position.data() + num_atoms, \
        position.data() + 2 * num_atoms, OUT_PTR); \
    } else if (alpha_basic_count_ <= 80) { \
      gpu_compute_basic_moments_tensor_accum<REAL_T, 80><<<grid_size, kBlockSize>>>( \
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(), \
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), \
        neighbor_dz_.data(), position.data(), position.data() + num_atoms, \
        position.data() + 2 * num_atoms, OUT_PTR); \
    } else { \
      gpu_compute_basic_moments_tensor_accum<REAL_T, kSus2MaxTensorBasic><<<grid_size, kBlockSize>>>( \
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(), \
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), \
        neighbor_dz_.data(), position.data(), position.data() + num_atoms, \
        position.data() + 2 * num_atoms, OUT_PTR); \
    } \
  } while (0)

#define SUS2_LAUNCH_TENSOR_BASIC_STATIC(REAL_T, LVAL, KVAL, OUT_PTR) \
  gpu_compute_basic_moments_tensor_accum_static<REAL_T, LVAL, KVAL><<<grid_size, kBlockSize>>>( \
    num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(), \
    neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), \
    neighbor_dz_.data(), position.data(), position.data() + num_atoms, \
    position.data() + 2 * num_atoms, OUT_PTR)

#define SUS2_LAUNCH_TENSOR_BASIC_SPECIALIZED(REAL_T, OUT_PTR) \
  do { \
    if (tensor_l_ == 1 && tensor_k_ == 1) { \
      SUS2_LAUNCH_TENSOR_BASIC_STATIC(REAL_T, 1, 1, OUT_PTR); \
    } else if (tensor_l_ == 1 && tensor_k_ == 2) { \
      SUS2_LAUNCH_TENSOR_BASIC_STATIC(REAL_T, 1, 2, OUT_PTR); \
    } else if (tensor_l_ == 1 && tensor_k_ == 3) { \
      SUS2_LAUNCH_TENSOR_BASIC_STATIC(REAL_T, 1, 3, OUT_PTR); \
    } else if (tensor_l_ == 1 && tensor_k_ == 4) { \
      SUS2_LAUNCH_TENSOR_BASIC_STATIC(REAL_T, 1, 4, OUT_PTR); \
    } else if (tensor_l_ == 2 && tensor_k_ == 1) { \
      SUS2_LAUNCH_TENSOR_BASIC_STATIC(REAL_T, 2, 1, OUT_PTR); \
    } else if (tensor_l_ == 2 && tensor_k_ == 2) { \
      SUS2_LAUNCH_TENSOR_BASIC_STATIC(REAL_T, 2, 2, OUT_PTR); \
    } else if (tensor_l_ == 2 && tensor_k_ == 3) { \
      SUS2_LAUNCH_TENSOR_BASIC_STATIC(REAL_T, 2, 3, OUT_PTR); \
    } else if (tensor_l_ == 2 && tensor_k_ == 4) { \
      SUS2_LAUNCH_TENSOR_BASIC_STATIC(REAL_T, 2, 4, OUT_PTR); \
    } else if (tensor_l_ == 3 && tensor_k_ == 1) { \
      SUS2_LAUNCH_TENSOR_BASIC_STATIC(REAL_T, 3, 1, OUT_PTR); \
    } else if (tensor_l_ == 3 && tensor_k_ == 2) { \
      SUS2_LAUNCH_TENSOR_BASIC_STATIC(REAL_T, 3, 2, OUT_PTR); \
    } else if (tensor_l_ == 3 && tensor_k_ == 3) { \
      SUS2_LAUNCH_TENSOR_BASIC_STATIC(REAL_T, 3, 3, OUT_PTR); \
    } else if (tensor_l_ == 3 && tensor_k_ == 4) { \
      SUS2_LAUNCH_TENSOR_BASIC_STATIC(REAL_T, 3, 4, OUT_PTR); \
    } else if (tensor_l_ == 4 && tensor_k_ == 1) { \
      SUS2_LAUNCH_TENSOR_BASIC_STATIC(REAL_T, 4, 1, OUT_PTR); \
    } else if (tensor_l_ == 4 && tensor_k_ == 2) { \
      SUS2_LAUNCH_TENSOR_BASIC_STATIC(REAL_T, 4, 2, OUT_PTR); \
    } else if (tensor_l_ == 4 && tensor_k_ == 3) { \
      SUS2_LAUNCH_TENSOR_BASIC_STATIC(REAL_T, 4, 3, OUT_PTR); \
    } else if (tensor_l_ == 4 && tensor_k_ == 4) { \
      SUS2_LAUNCH_TENSOR_BASIC_STATIC(REAL_T, 4, 4, OUT_PTR); \
    } else { \
      SUS2_LAUNCH_TENSOR_BASIC_DYNAMIC(REAL_T, OUT_PTR); \
    } \
  } while (0)

  profile_t = profile_start();
  const bool use_exact_l3k3_tensor = use_tensor_basic_fastpath_ && tensor_l_ == 3 && tensor_k_ == 3;
  const bool use_l3k3_tensor_scalar = use_l3k3_tensor_scalar_ && use_exact_l3k3_tensor;
  const bool use_l3k3_tensor_block = use_l3k3_tensor_block_ && use_tensor_basic_fastpath_;

  if (use_exact_l3k3_tensor) {
    if (use_float_moments_) {
      gpu_compute_basic_moments_l3k3_accum<float><<<grid_size, kBlockSize>>>(
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
        moment_vals_float_.data());
    } else {
      gpu_compute_basic_moments_l3k3_accum<double><<<grid_size, kBlockSize>>>(
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
  } else if (use_tensor_basic_fastpath_) {
    if (use_float_moments_) {
      SUS2_LAUNCH_TENSOR_BASIC_SPECIALIZED(float, moment_vals_float_.data());
    } else {
      SUS2_LAUNCH_TENSOR_BASIC_SPECIALIZED(double, moment_vals_.data());
    }
  } else {
    if (use_float_moments_) {
      gpu_compute_basic_moments<float><<<grid_size, kBlockSize>>>(
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
        moment_vals_float_.data());
    } else {
      gpu_compute_basic_moments<double><<<grid_size, kBlockSize>>>(
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
  }
  GPU_CHECK_KERNEL
  profile_stop(profile_basic, profile_t);

  if (use_l3k3_tensor_scalar) {
    profile_t = profile_start();
    if (use_float_moments_) {
      gpu_l3k3_tensor_scalar_energy_backward_float_parallel<<<num_atoms, kSus2TensorScalarBlockSize>>>(
        num_atoms,
        model,
        type.data(),
        moment_vals_float_.data(),
        moment_grads_float_.data(),
        potential.data());
    } else if (use_float_moment_grads_) {
      gpu_l3k3_tensor_scalar_energy_backward<double, float><<<grid_size, kBlockSize>>>(
        num_atoms,
        model,
        type.data(),
        moment_vals_.data(),
        moment_grads_float_.data(),
        potential.data());
    } else {
      gpu_l3k3_tensor_scalar_energy_backward<double, double><<<grid_size, kBlockSize>>>(
        num_atoms,
        model,
        type.data(),
        moment_vals_.data(),
        moment_grads_.data(),
        potential.data());
    }
    GPU_CHECK_KERNEL
    profile_stop(profile_forward, profile_t);
  } else if (use_l3k3_tensor_block) {
    profile_t = profile_start();
    if (use_float_moments_) {
      gpu_l3k3_tensor_block_energy_backward<float, float><<<grid_size, kBlockSize>>>(
        num_atoms,
        model,
        type.data(),
        moment_vals_float_.data(),
        moment_grads_float_.data(),
        potential.data());
    } else if (use_float_moment_grads_) {
      gpu_l3k3_tensor_block_energy_backward<double, float><<<grid_size, kBlockSize>>>(
        num_atoms,
        model,
        type.data(),
        moment_vals_.data(),
        moment_grads_float_.data(),
        potential.data());
    } else {
      gpu_l3k3_tensor_block_energy_backward<double, double><<<grid_size, kBlockSize>>>(
        num_atoms,
        model,
        type.data(),
        moment_vals_.data(),
        moment_grads_.data(),
        potential.data());
    }
    GPU_CHECK_KERNEL
    profile_stop(profile_forward, profile_t);
  } else if (use_local_product_graph) {
    profile_t = profile_start();
    gpu_local_graph_energy_backward_to_basic<float, float, kSus2LocalGraphMaxMoments><<<grid_size, kBlockSize>>>(
      num_atoms,
      model,
      type.data(),
      moment_vals_float_.data(),
      moment_grads_float_.data(),
      potential.data());
    GPU_CHECK_KERNEL
    profile_stop(profile_forward, profile_t);
  } else if (use_fused_graph_) {
    profile_t = profile_start();
    if (use_float_moments_) {
      if (use_const_alpha_times_) {
        if (use_product_assign_) {
          if (use_graph_specific_product_) {
            gpu_forward_energy_backward_const_u16_group_pair_table<float, float><<<grid_size, kBlockSize>>>(
              num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(), potential.data());
          } else {
            gpu_forward_energy_backward_const_u16_assign_group_table<float, float><<<grid_size, kBlockSize>>>(
              num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(), potential.data());
          }
        } else {
          gpu_forward_energy_backward_const_u16<float, float><<<grid_size, kBlockSize>>>(
            num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(), potential.data());
        }
      } else {
        if (use_product_assign_) {
          gpu_forward_energy_backward_assign_group_table<float, float><<<grid_size, kBlockSize>>>(
            num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(), potential.data());
        } else {
          gpu_forward_energy_backward<float, float><<<grid_size, kBlockSize>>>(
            num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(), potential.data());
        }
      }
    } else if (use_float_moment_grads_) {
      if (use_const_alpha_times_) {
        if (use_product_assign_) {
          if (use_graph_specific_product_) {
            gpu_forward_energy_backward_const_u16_group_pair_table<double, float><<<grid_size, kBlockSize>>>(
              num_atoms, model, type.data(), moment_vals_.data(), moment_grads_float_.data(), potential.data());
          } else {
            gpu_forward_energy_backward_const_u16_assign_group_table<double, float><<<grid_size, kBlockSize>>>(
              num_atoms, model, type.data(), moment_vals_.data(), moment_grads_float_.data(), potential.data());
          }
        } else {
          gpu_forward_energy_backward_const_u16<double, float><<<grid_size, kBlockSize>>>(
            num_atoms, model, type.data(), moment_vals_.data(), moment_grads_float_.data(), potential.data());
        }
      } else {
        if (use_product_assign_) {
          gpu_forward_energy_backward_assign_group_table<double, float><<<grid_size, kBlockSize>>>(
            num_atoms, model, type.data(), moment_vals_.data(), moment_grads_float_.data(), potential.data());
        } else {
          gpu_forward_energy_backward<double, float><<<grid_size, kBlockSize>>>(
            num_atoms, model, type.data(), moment_vals_.data(), moment_grads_float_.data(), potential.data());
        }
      }
    } else {
      if (use_const_alpha_times_) {
        if (use_product_assign_) {
          if (use_graph_specific_product_) {
            gpu_forward_energy_backward_const_u16_group_pair_table<double, double><<<grid_size, kBlockSize>>>(
              num_atoms, model, type.data(), moment_vals_.data(), moment_grads_.data(), potential.data());
          } else {
            gpu_forward_energy_backward_const_u16_assign_group_table<double, double><<<grid_size, kBlockSize>>>(
              num_atoms, model, type.data(), moment_vals_.data(), moment_grads_.data(), potential.data());
          }
        } else {
          gpu_forward_energy_backward_const_u16<double, double><<<grid_size, kBlockSize>>>(
            num_atoms, model, type.data(), moment_vals_.data(), moment_grads_.data(), potential.data());
        }
      } else {
        if (use_product_assign_) {
          gpu_forward_energy_backward_assign_group_table<double, double><<<grid_size, kBlockSize>>>(
            num_atoms, model, type.data(), moment_vals_.data(), moment_grads_.data(), potential.data());
        } else {
          gpu_forward_energy_backward<double, double><<<grid_size, kBlockSize>>>(
            num_atoms, model, type.data(), moment_vals_.data(), moment_grads_.data(), potential.data());
        }
      }
    }
    GPU_CHECK_KERNEL
    profile_stop(profile_forward, profile_t);
  } else {
    profile_t = profile_start();
    if (use_const_alpha_times_) {
      if (use_float_moments_) {
        gpu_forward_times_const_u16<float><<<grid_size, kBlockSize>>>(
          num_atoms, model, moment_vals_float_.data());
      } else {
        gpu_forward_times_const_u16<double><<<grid_size, kBlockSize>>>(
          num_atoms, model, moment_vals_.data());
      }
    } else {
      if (use_float_moments_) {
        gpu_forward_times<float><<<grid_size, kBlockSize>>>(num_atoms, model, moment_vals_float_.data());
      } else {
        gpu_forward_times<double><<<grid_size, kBlockSize>>>(num_atoms, model, moment_vals_.data());
      }
    }
    GPU_CHECK_KERNEL
    profile_stop(profile_forward, profile_t);

    if (use_fused_energy_backward_) {
      profile_t = profile_start();
      if (use_float_moments_) {
        if (use_const_alpha_times_) {
          gpu_site_energy_init_grad_backward_const_u16<float, float><<<grid_size, kBlockSize>>>(
            num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(), potential.data());
        } else {
          gpu_site_energy_init_grad_backward<float, float><<<grid_size, kBlockSize>>>(
            num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(), potential.data());
        }
      } else if (use_float_moment_grads_) {
        if (use_const_alpha_times_) {
          gpu_site_energy_init_grad_backward_const_u16<double, float><<<grid_size, kBlockSize>>>(
            num_atoms, model, type.data(), moment_vals_.data(), moment_grads_float_.data(), potential.data());
        } else {
          gpu_site_energy_init_grad_backward<double, float><<<grid_size, kBlockSize>>>(
            num_atoms, model, type.data(), moment_vals_.data(), moment_grads_float_.data(), potential.data());
        }
      } else {
        if (use_const_alpha_times_) {
          gpu_site_energy_init_grad_backward_const_u16<double, double><<<grid_size, kBlockSize>>>(
            num_atoms, model, type.data(), moment_vals_.data(), moment_grads_.data(), potential.data());
        } else {
          gpu_site_energy_init_grad_backward<double, double><<<grid_size, kBlockSize>>>(
            num_atoms, model, type.data(), moment_vals_.data(), moment_grads_.data(), potential.data());
        }
      }
      GPU_CHECK_KERNEL
      profile_stop(profile_energy_grad, profile_t);
    } else {
      profile_t = profile_start();
      if (use_float_moments_) {
        gpu_site_energy_init_grad<float, float><<<grid_size, kBlockSize>>>(
          num_atoms, model, type.data(), moment_vals_float_.data(), moment_grads_float_.data(), potential.data());
      } else if (use_float_moment_grads_) {
        gpu_site_energy_init_grad<double, float><<<grid_size, kBlockSize>>>(
          num_atoms, model, type.data(), moment_vals_.data(), moment_grads_float_.data(), potential.data());
      } else {
        gpu_site_energy_init_grad<double, double><<<grid_size, kBlockSize>>>(
          num_atoms, model, type.data(), moment_vals_.data(), moment_grads_.data(), potential.data());
      }
      GPU_CHECK_KERNEL
      profile_stop(profile_energy_grad, profile_t);

      profile_t = profile_start();
      if (use_float_moments_) {
        if (use_const_alpha_times_) {
          gpu_backward_times_const_u16<float, float><<<grid_size, kBlockSize>>>(
            num_atoms, model, moment_vals_float_.data(), moment_grads_float_.data());
        } else {
          gpu_backward_times<float, float><<<grid_size, kBlockSize>>>(
            num_atoms, model, moment_vals_float_.data(), moment_grads_float_.data());
        }
      } else if (use_float_moment_grads_) {
        if (use_const_alpha_times_) {
          gpu_backward_times_const_u16<double, float><<<grid_size, kBlockSize>>>(
            num_atoms, model, moment_vals_.data(), moment_grads_float_.data());
        } else {
          gpu_backward_times<double, float><<<grid_size, kBlockSize>>>(
            num_atoms, model, moment_vals_.data(), moment_grads_float_.data());
        }
      } else {
        if (use_const_alpha_times_) {
          gpu_backward_times_const_u16<double, double><<<grid_size, kBlockSize>>>(
            num_atoms, model, moment_vals_.data(), moment_grads_.data());
        } else {
          gpu_backward_times<double, double><<<grid_size, kBlockSize>>>(
            num_atoms, model, moment_vals_.data(), moment_grads_.data());
        }
      }
      GPU_CHECK_KERNEL
      profile_stop(profile_backward, profile_t);
    }
  }

#define SUS2_LAUNCH_TENSOR_CACHED_FORCE_DYNAMIC(GRAD_T, REAL_T, GRADS_PTR) \
  do { \
    if (alpha_basic_count_ <= 40) { \
      gpu_compute_forces_tensor_cached_grads<GRAD_T, REAL_T, 40><<<grid_size, kBlockSize>>>( \
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(), \
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), \
        neighbor_dz_.data(), position.data(), position.data() + num_atoms, \
        position.data() + 2 * num_atoms, GRADS_PTR, force_tmp_.data(), force_self_tmp_ptr, virial_tmp_.data()); \
    } else if (alpha_basic_count_ <= 80) { \
      gpu_compute_forces_tensor_cached_grads<GRAD_T, REAL_T, 80><<<grid_size, kBlockSize>>>( \
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(), \
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), \
        neighbor_dz_.data(), position.data(), position.data() + num_atoms, \
        position.data() + 2 * num_atoms, GRADS_PTR, force_tmp_.data(), force_self_tmp_ptr, virial_tmp_.data()); \
    } else { \
      gpu_compute_forces_tensor_cached_grads<GRAD_T, REAL_T, kSus2MaxTensorBasic><<<grid_size, kBlockSize>>>( \
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(), \
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), \
        neighbor_dz_.data(), position.data(), position.data() + num_atoms, \
        position.data() + 2 * num_atoms, GRADS_PTR, force_tmp_.data(), force_self_tmp_ptr, virial_tmp_.data()); \
    } \
  } while (0)

#define SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC(GRAD_T, REAL_T, LVAL, KVAL, GRADS_PTR) \
  gpu_compute_forces_tensor_cached_grads_static<GRAD_T, REAL_T, LVAL, KVAL><<<grid_size, kBlockSize>>>( \
    num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(), \
    neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), \
    neighbor_dz_.data(), position.data(), position.data() + num_atoms, \
    position.data() + 2 * num_atoms, GRADS_PTR, force_tmp_.data(), force_self_tmp_ptr, virial_tmp_.data())

#define SUS2_LAUNCH_TENSOR_CACHED_FORCE(GRAD_T, REAL_T, GRADS_PTR) \
  do { \
    if (use_exact_l3k3_tensor) { \
      gpu_compute_forces_l3k3_cached_grads<GRAD_T, REAL_T><<<grid_size, kBlockSize>>>( \
        num_atoms, box, rc * rc, use_cached_neighbor_displacements_, model, type.data(), \
        neighbor_count_.data(), neighbor_atom_.data(), neighbor_dx_.data(), neighbor_dy_.data(), \
        neighbor_dz_.data(), position.data(), position.data() + num_atoms, \
        position.data() + 2 * num_atoms, GRADS_PTR, force_tmp_.data(), force_self_tmp_ptr, virial_tmp_.data()); \
    } else if (tensor_l_ == 1 && tensor_k_ == 1) { \
      SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC(GRAD_T, REAL_T, 1, 1, GRADS_PTR); \
    } else if (tensor_l_ == 1 && tensor_k_ == 2) { \
      SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC(GRAD_T, REAL_T, 1, 2, GRADS_PTR); \
    } else if (tensor_l_ == 1 && tensor_k_ == 3) { \
      SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC(GRAD_T, REAL_T, 1, 3, GRADS_PTR); \
    } else if (tensor_l_ == 1 && tensor_k_ == 4) { \
      SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC(GRAD_T, REAL_T, 1, 4, GRADS_PTR); \
    } else if (tensor_l_ == 2 && tensor_k_ == 1) { \
      SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC(GRAD_T, REAL_T, 2, 1, GRADS_PTR); \
    } else if (tensor_l_ == 2 && tensor_k_ == 2) { \
      SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC(GRAD_T, REAL_T, 2, 2, GRADS_PTR); \
    } else if (tensor_l_ == 2 && tensor_k_ == 3) { \
      SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC(GRAD_T, REAL_T, 2, 3, GRADS_PTR); \
    } else if (tensor_l_ == 2 && tensor_k_ == 4) { \
      SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC(GRAD_T, REAL_T, 2, 4, GRADS_PTR); \
    } else if (tensor_l_ == 3 && tensor_k_ == 1) { \
      SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC(GRAD_T, REAL_T, 3, 1, GRADS_PTR); \
    } else if (tensor_l_ == 3 && tensor_k_ == 2) { \
      SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC(GRAD_T, REAL_T, 3, 2, GRADS_PTR); \
    } else if (tensor_l_ == 3 && tensor_k_ == 3) { \
      SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC(GRAD_T, REAL_T, 3, 3, GRADS_PTR); \
    } else if (tensor_l_ == 3 && tensor_k_ == 4) { \
      SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC(GRAD_T, REAL_T, 3, 4, GRADS_PTR); \
    } else if (tensor_l_ == 4 && tensor_k_ == 1) { \
      SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC(GRAD_T, REAL_T, 4, 1, GRADS_PTR); \
    } else if (tensor_l_ == 4 && tensor_k_ == 2) { \
      SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC(GRAD_T, REAL_T, 4, 2, GRADS_PTR); \
    } else if (tensor_l_ == 4 && tensor_k_ == 3) { \
      SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC(GRAD_T, REAL_T, 4, 3, GRADS_PTR); \
    } else if (tensor_l_ == 4 && tensor_k_ == 4) { \
      SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC(GRAD_T, REAL_T, 4, 4, GRADS_PTR); \
    } else { \
      SUS2_LAUNCH_TENSOR_CACHED_FORCE_DYNAMIC(GRAD_T, REAL_T, GRADS_PTR); \
    } \
  } while (0)

  profile_t = profile_start();
  if (!use_pairwise_no_atomic_force) {
    if (use_float_moments_) {
      if (use_tensor_basic_fastpath_ && use_tensor_force_grad_cache_) {
        SUS2_LAUNCH_TENSOR_CACHED_FORCE(float, float, moment_grads_float_.data());
      } else {
        gpu_compute_forces<float, float><<<grid_size, kBlockSize>>>(
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
          force_self_tmp_ptr,
          virial_tmp_.data());
      }
    } else if (use_float_moment_grads_) {
      if (use_tensor_basic_fastpath_ && use_tensor_force_grad_cache_) {
        SUS2_LAUNCH_TENSOR_CACHED_FORCE(float, double, moment_grads_float_.data());
      } else {
        gpu_compute_forces<float, double><<<grid_size, kBlockSize>>>(
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
          force_self_tmp_ptr,
          virial_tmp_.data());
      }
    } else {
      if (use_tensor_basic_fastpath_ && use_tensor_force_grad_cache_) {
        SUS2_LAUNCH_TENSOR_CACHED_FORCE(double, double, moment_grads_.data());
      } else {
        gpu_compute_forces<double, double><<<grid_size, kBlockSize>>>(
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
          force_self_tmp_ptr,
          virial_tmp_.data());
      }
    }
    GPU_CHECK_KERNEL
  } else {
    if (use_float_moments_) {
      gpu_compute_forces_pairwise_no_atomic<float, float><<<grid_size, kBlockSize>>>(
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
    } else if (use_float_moment_grads_) {
      gpu_compute_forces_pairwise_no_atomic<float, double><<<grid_size, kBlockSize>>>(
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
      gpu_compute_forces_pairwise_no_atomic<double, double><<<grid_size, kBlockSize>>>(
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
#undef SUS2_LAUNCH_TENSOR_CACHED_FORCE
#undef SUS2_LAUNCH_TENSOR_CACHED_FORCE_STATIC
#undef SUS2_LAUNCH_TENSOR_CACHED_FORCE_DYNAMIC
#undef SUS2_LAUNCH_TENSOR_BASIC_SPECIALIZED
#undef SUS2_LAUNCH_TENSOR_BASIC_STATIC
#undef SUS2_LAUNCH_TENSOR_BASIC_DYNAMIC
  profile_stop(profile_force, profile_t);

  profile_t = profile_start();
  gpu_accumulate_float_to_double<<<grid_size, kBlockSize>>>(
    num_atoms, force_tmp_.data(), force_self_tmp_ptr, virial_tmp_.data(), force.data(), virial.data());
  GPU_CHECK_KERNEL
  profile_stop(profile_accumulate, profile_t);

  maybe_print_profile();
}
