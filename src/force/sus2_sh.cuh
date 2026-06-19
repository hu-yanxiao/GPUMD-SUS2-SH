#pragma once

#include "neighbor.cuh"
#include "potential.cuh"
#include "utilities/gpu_vector.cuh"

class SUS2_SH : public Potential
{
public:
  using Potential::compute;

  SUS2_SH(
    const char* file_potential,
    int num_atoms,
    int num_potential_options = 0,
    const char** potential_options = nullptr);
  virtual ~SUS2_SH(void);

  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

private:
  void build_neighbor_list(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    int num_atoms);
  void resize_work_buffers(int num_atoms);

  int species_count_ = 0;
  int sh_l_max_ = 0;
  int sh_k_max_ = 0;
  int radial_basis_kind_ = 0;
  int radial_funcs_count_ = 0;
  int rb_size_ = 0;
  int alpha_basic_count_ = 0;
  int sh_product_count_ = 0;
  int sh_cg_block_count_ = 0;
  int sh_cg_term_count_ = 0;
  int sh_cg_row_count_ = 0;
  int sh_cg_row_term_count_ = 0;
  int sh_cg_row_pattern_count_ = 0;
  int sh_cg_row_pattern_term_count_ = 0;
  int two_layer_gate_cg_row_count_ = 0;
  int two_layer_gate_cg_row_term_count_ = 0;
  int sh_cg_back_row_count_ = 0;
  int sh_cg_back_term_count_ = 0;
  int two_layer_gate_cg_back_row_count_ = 0;
  int two_layer_gate_cg_back_term_count_ = 0;
  int sh_cg_layer_count_ = 0;
  int sh_terminal_dot_group_count_ = 0;
  int sh_terminal_dot_group_entry_count_ = 0;
  int sh_terminal_dot_nondot_row_count_ = 0;
  int sh_fused_terminal_dot_group_count_ = 0;
  int sh_fused_terminal_dot_group_entry_count_ = 0;
  int sh_fused_terminal_dot_producer_count_ = 0;
  int sh_fused_terminal_dot_component_count_ = 0;
  int sh_fused_terminal_dot_term_count_ = 0;
  int sh_grad_zero_count_ = 0;
  int alpha_moments_count_ = 0;
  int alpha_scalar_moments_ = 0;
  int active_scalar_moments_ = 0;
  int neighbor_capacity_ = 512;
  double rc = 0.0;
  double neighbor_cutoff_ = 0.0;
  bool zbl_enabled_ = false;
  double zbl_outer_max_ = 0.0;
  bool two_layer_gate_enabled_ = false;
  int two_layer_gate_mode_ = 0;
  int two_layer_gate_site_mode_ = 0;
  double two_layer_gate_tanh_amplitude_ = 0.8;
  int two_layer_gate_weight_count_ = 0;
  int two_layer_gate_scalar_count_ = 0;
  int two_layer_gate_body_order_count_ = 0;
  int two_layer_gate_product_limit_ = 0;
  bool use_float_moments_ = true;
  bool use_radial_direct_ = true;
  bool use_force_self_buffer_ = true;
  bool use_force_grad_cache_ = false;
  bool use_cg_block_forward_ = false;
  bool use_tensor_product_parallel_ = false;
  bool use_compact_serial_product_ = false;
  bool use_const_forward_rows_ = false;
  bool use_product_pattern_rows_ = false;
  bool use_const_pattern_rows_ = false;
  bool use_static_basic_layout_ = false;
  bool use_static_force_layout_ = false;
  bool use_parallel_back_rows_ = false;
  bool use_terminal_scalar_fusion_ = false;
  bool use_row_scalar_fusion_ = false;
  bool use_terminal_dot_rows_ = false;
  bool use_terminal_dot_groups_ = false;
  bool use_terminal_dot_premul_ = false;
  bool use_terminal_dot_row_list_ = false;
  bool use_fused_terminal_dot_ = false;
  bool use_selective_grad_zero_ = false;
  bool use_product_basic_cache_ = false;
  bool use_packed_back_rows_ = false;
  bool use_const_back_rows_ = false;
  bool use_merge_back_duplicates_ = true;
  bool use_cached_neighbor_displacements_ = false;
  bool profile_enabled_ = false;
  bool profile_product_detail_ = false;
  int profile_interval_ = 50;
  int tensor_product_grid_cap_ = 8192;
  int profile_steps_ = 0;
  double profile_ms_[11];

  GPU_Vector<double> shift_coeffs_;
  GPU_Vector<double> species_coeffs_;
  GPU_Vector<double> moment_coeffs_;
  GPU_Vector<float> shift_coeffs_float_;
  GPU_Vector<float> species_coeffs_float_;
  GPU_Vector<float> moment_coeffs_float_;
  GPU_Vector<int> alpha_basic_;
  GPU_Vector<int> alpha_basic_mu_yidx_;
  GPU_Vector<int> sh_products_int_;
  GPU_Vector<double> sh_products_coeff_;
  GPU_Vector<float> sh_products_coeff_float_;
  GPU_Vector<int> sh_cg_blocks_int_;
  GPU_Vector<int> sh_cg_terms_int_;
  GPU_Vector<double> sh_cg_terms_coeff_;
  GPU_Vector<float> sh_cg_terms_coeff_float_;
  GPU_Vector<int> sh_cg_rows_int_;
  GPU_Vector<int> sh_cg_row_terms_int_;
  GPU_Vector<double> sh_cg_row_terms_coeff_;
  GPU_Vector<float> sh_cg_row_terms_coeff_float_;
  GPU_Vector<int> sh_cg_row_scalar_index_;
  GPU_Vector<int> two_layer_gate_cg_rows_int_;
  GPU_Vector<int> two_layer_gate_cg_row_terms_int_;
  GPU_Vector<float> two_layer_gate_cg_row_terms_coeff_float_;
  GPU_Vector<int> two_layer_gate_cg_layer_offsets_;
  GPU_Vector<int> sh_terminal_moment_flags_;
  GPU_Vector<unsigned int> sh_cg_row_dot_u32_;
  GPU_Vector<unsigned int> sh_cg_row_pattern_u32_;
  GPU_Vector<unsigned int> sh_terminal_dot_group_u32_;
  GPU_Vector<int> sh_terminal_dot_group_layer_offsets_;
  GPU_Vector<int> sh_terminal_dot_nondot_rows_;
  GPU_Vector<int> sh_terminal_dot_nondot_layer_offsets_;
  GPU_Vector<unsigned int> sh_fused_terminal_dot_u32_;
  GPU_Vector<int> sh_cg_back_rows_int_;
  GPU_Vector<int> sh_cg_back_terms_int_;
  GPU_Vector<double> sh_cg_back_terms_coeff_;
  GPU_Vector<float> sh_cg_back_terms_coeff_float_;
  GPU_Vector<int> sh_cg_back_layer_offsets_;
  GPU_Vector<unsigned int> sh_cg_back_packed_u32_;
  GPU_Vector<int> two_layer_gate_cg_back_rows_int_;
  GPU_Vector<int> two_layer_gate_cg_back_terms_int_;
  GPU_Vector<float> two_layer_gate_cg_back_terms_coeff_float_;
  GPU_Vector<int> two_layer_gate_cg_back_layer_offsets_;
  GPU_Vector<int> sh_grad_zero_moments_;
  GPU_Vector<int> active_scalar_moment_;
  GPU_Vector<double> active_scalar_coeff_;
  GPU_Vector<float> active_scalar_coeff_float_;
  GPU_Vector<int> sh_cg_layer_offsets_;
  GPU_Vector<int> alpha_moment_mapping_;
  GPU_Vector<float> radial_direct_coeffs_;
  GPU_Vector<float> radial_direct_scal_s_;
  GPU_Vector<float> two_layer_gate_radial_direct_coeffs_;
  GPU_Vector<int> two_layer_gate_moment_indices_;
  GPU_Vector<int> two_layer_gate_scalar_body_ids_;
  GPU_Vector<float> two_layer_gate_weights_float_;
  GPU_Vector<float> two_layer_gate_weights_transposed_float_;
  GPU_Vector<float> two_layer_gate_body_mix_weights_float_;
  GPU_Vector<int> two_layer_gate_needed_moment_flags_;
  GPU_Vector<float> two_layer_gate_moment_weights_float_;
  GPU_Vector<float> two_layer_gate_additive_coeffs_float_;
  GPU_Vector<int> zbl_atomic_numbers_;
  GPU_Vector<double> zbl_pair_inner_cutoffs_;
  GPU_Vector<double> zbl_pair_outer_cutoffs_;
  GPU_Vector<double> zbl_pair_outer_sq_;

  GPU_Vector<int> neighbor_count_;
  GPU_Vector<int> neighbor_atom_;
  GPU_Vector<double> neighbor_dx_;
  GPU_Vector<double> neighbor_dy_;
  GPU_Vector<double> neighbor_dz_;
  GPU_Vector<int> cell_contents_;
  Neighbor neighbor_cache_;

  GPU_Vector<double> moment_vals_;
  GPU_Vector<double> moment_grads_;
  GPU_Vector<float> moment_vals_float_;
  GPU_Vector<float> moment_grads_float_;
  GPU_Vector<double> two_layer_gate_basic_grads_;
  GPU_Vector<double> two_layer_gate_values_;
  GPU_Vector<double> two_layer_gate_adjoints_;
  GPU_Vector<float> two_layer_gate_basic_grads_float_;
  GPU_Vector<float> two_layer_gate_moment_vals_float_;
  GPU_Vector<float> two_layer_gate_values_float_;
  GPU_Vector<float> two_layer_gate_multipliers_float_;
  GPU_Vector<float> two_layer_gate_derivs_float_;
  GPU_Vector<float> two_layer_gate_adjoints_float_;
  GPU_Vector<float> two_layer_gate_body_adjoints_float_;
  GPU_Vector<float> force_tmp_;
  GPU_Vector<float> force_self_tmp_;
  GPU_Vector<float> virial_tmp_;
};

bool is_sus2_sh_potential_file(const char* file_potential);
