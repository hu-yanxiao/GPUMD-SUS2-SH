#pragma once

#include "neighbor.cuh"
#include "potential.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class SUS2_V11 : public Potential
{
public:
  using Potential::compute;

  SUS2_V11(
    const char* file_potential,
    int num_atoms,
    int num_potential_options = 0,
    const char** potential_options = nullptr);
  virtual ~SUS2_V11(void);

  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

private:
  enum ProfileSlot {
    profile_neighbor = 0,
    profile_neighbor_global,
    profile_neighbor_local,
    profile_zero,
    profile_basic,
    profile_forward,
    profile_energy_grad,
    profile_backward,
    profile_force,
    profile_accumulate,
    profile_count
  };

  void build_neighbor_list(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    int num_atoms);
  void resize_work_buffers(int num_atoms);
  void maybe_print_profile();

  int species_count_ = 0;
  int angular_channels_ = 0;
  int radial_basis_kind_ = 0;
  int radial_funcs_count_ = 0;
  int rb_size_ = 0;
  int alpha_basic_count_ = 0;
  int alpha_times_count_ = 0;
  int alpha_time_group_count_ = 0;
  int alpha_moments_count_ = 0;
  int alpha_scalar_moments_ = 0;
  int max_rank_ = 0;
  int tensor_l_ = 0;
  int tensor_k_ = 0;
  int tensor_basic_per_group_ = 0;
  int lut_size_ = 0;
  int neighbor_capacity_ = 512;
  double lut_inv_dr_ = 0.0;

  GPU_Vector<double> shift_coeffs_;
  GPU_Vector<double> species_coeffs_;
  GPU_Vector<double> moment_coeffs_;
  GPU_Vector<float> shift_coeffs_float_;
  GPU_Vector<float> species_coeffs_float_;
  GPU_Vector<float> moment_coeffs_float_;
  GPU_Vector<int> alpha_basic_;
  GPU_Vector<int> alpha_times_;
  GPU_Vector<int> alpha_time_groups_;
  GPU_Vector<int> alpha_moment_mapping_;
  GPU_Vector<int> l3k3_tensor_scalar_terms_;
  GPU_Vector<double> l3k3_tensor_scalar_coeffs_;
  GPU_Vector<float> l3k3_tensor_scalar_coeffs_float_;
  GPU_Vector<int> l3k3_tensor_block_ops_;
  GPU_Vector<float> lut_vals_;
  GPU_Vector<float> lut_ders_;
  GPU_Vector<float> radial_direct_coeffs_;
  GPU_Vector<float> radial_direct_scal_s_;

  GPU_Vector<int> neighbor_count_;
  GPU_Vector<int> neighbor_atom_;
  GPU_Vector<double> neighbor_dx_;
  GPU_Vector<double> neighbor_dy_;
  GPU_Vector<double> neighbor_dz_;
  GPU_Vector<int> cell_count_;
  GPU_Vector<int> cell_count_sum_;
  GPU_Vector<int> cell_contents_;
  Neighbor neighbor_cache_;
  bool use_cached_neighbor_displacements_ = false;
  bool use_pairwise_no_atomic_force_ = false;
  bool use_tensor_basic_fastpath_ = false;
  bool use_const_alpha_times_ = false;
  bool use_const_scalar_moments_ = false;
  bool use_const_float_coeffs_ = false;
  bool use_fused_energy_backward_ = true;
  bool use_fused_graph_ = true;
  bool use_local_product_graph_ = false;
  bool use_product_assign_ = true;
  bool product_assign_supported_ = false;
  bool use_tensor_force_grad_cache_ = true;
  bool use_l3k3_tensor_scalar_ = false;
  bool use_l3k3_tensor_block_ = false;
  bool use_float_moment_grads_ = false;
  bool use_float_moments_ = false;
  bool use_radial_direct_ = false;
  int l3k3_tensor_scalar_term_count_ = 0;
  int l3k3_tensor_block_op_count_ = 0;

  GPU_Vector<double> moment_vals_;
  GPU_Vector<float> moment_vals_float_;
  GPU_Vector<double> moment_grads_;
  GPU_Vector<float> moment_grads_float_;
  GPU_Vector<float> force_tmp_;
  GPU_Vector<float> virial_tmp_;

  bool profile_enabled_ = false;
  int profile_interval_ = 100;
  long long profile_calls_ = 0;
  double profile_ms_[profile_count] = {0.0};
};
