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
  int radial_funcs_count_ = 0;
  int rb_size_ = 0;
  int alpha_basic_count_ = 0;
  int sh_product_count_ = 0;
  int sh_cg_block_count_ = 0;
  int sh_cg_term_count_ = 0;
  int sh_cg_row_count_ = 0;
  int sh_cg_row_term_count_ = 0;
  int sh_cg_back_row_count_ = 0;
  int sh_cg_back_term_count_ = 0;
  int sh_cg_layer_count_ = 0;
  int alpha_moments_count_ = 0;
  int alpha_scalar_moments_ = 0;
  int neighbor_capacity_ = 512;
  double rc = 0.0;
  bool use_float_moments_ = true;
  bool use_radial_direct_ = true;
  bool use_force_self_buffer_ = true;
  bool use_force_grad_cache_ = false;
  bool use_cg_block_forward_ = false;
  bool use_tensor_product_parallel_ = false;
  bool use_cached_neighbor_displacements_ = false;
  bool profile_enabled_ = false;
  int profile_interval_ = 50;
  int tensor_product_grid_cap_ = 8192;
  int profile_steps_ = 0;
  double profile_ms_[6];

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
  GPU_Vector<int> sh_cg_back_rows_int_;
  GPU_Vector<int> sh_cg_back_terms_int_;
  GPU_Vector<double> sh_cg_back_terms_coeff_;
  GPU_Vector<float> sh_cg_back_terms_coeff_float_;
  GPU_Vector<int> sh_cg_back_layer_offsets_;
  GPU_Vector<int> sh_cg_layer_offsets_;
  GPU_Vector<int> alpha_moment_mapping_;
  GPU_Vector<float> radial_direct_coeffs_;
  GPU_Vector<float> radial_direct_scal_s_;

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
  GPU_Vector<float> force_tmp_;
  GPU_Vector<float> force_self_tmp_;
  GPU_Vector<float> virial_tmp_;
};

bool is_sus2_sh_potential_file(const char* file_potential);
