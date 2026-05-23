#include <torch/extension.h>

#include <vector>

using TensorVec = std::vector<at::Tensor>;

void prepare_initial_accel_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    int64_t slot,
    int64_t active_idx,
    bool rollout_tail,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode);

void accel_to_sat_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    int64_t slot,
    int64_t active_idx,
    bool rollout_tail,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode);

void queue_aware_accel_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    int64_t active_idx,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode);

void cluster_center_accel_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    int64_t active_idx,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode);

void baseline_accel_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    int64_t active_idx,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode);

void queue_aware_sat_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode);

void baseline_sat_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode);

void queue_aware_bw_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode);

void baseline_bw_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode);

void sat_to_bw_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    int64_t slot,
    int64_t active_idx,
    bool rollout_tail,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode);

void finish_commit_prepare_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    int64_t slot,
    int64_t active_idx,
    bool rollout_tail,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode);

void apply_bw_macro_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    int64_t slot,
    int64_t active_idx,
    bool rollout_tail,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode);

void finish_commit_prepare_live_profiled_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    int64_t slot,
    int64_t active_idx,
    bool rollout_tail,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode,
    at::Tensor finish_profile_out);

void prepare_branch_replay_from_history_launcher(
    const TensorVec& source_float_tensors,
    const TensorVec& source_long_tensors,
    const TensorVec& source_bool_tensors,
    const TensorVec& source_int_tensors,
    const std::vector<int64_t>& source_int_params,
    const std::vector<double>& source_float_params,
    const TensorVec& target_float_tensors,
    const TensorVec& target_long_tensors,
    const TensorVec& target_bool_tensors,
    const TensorVec& target_int_tensors,
    const std::vector<int64_t>& target_int_params,
    const std::vector<double>& target_float_params,
    at::Tensor history_rows,
    int64_t stage_id);

void actor_accel_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    const TensorVec& actor_weights,
    const TensorVec& actor_int_tensors,
    const std::vector<int64_t>& actor_int_params,
    const std::vector<double>& actor_float_params,
    int64_t active_idx,
    bool deterministic,
    int64_t rng_step);

void actor_sat_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    const TensorVec& actor_weights,
    const TensorVec& actor_int_tensors,
    const std::vector<int64_t>& actor_int_params,
    const std::vector<double>& actor_float_params,
    bool deterministic,
    int64_t rng_step);

void actor_sat_live_fused_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    const TensorVec& actor_weights,
    const TensorVec& actor_int_tensors,
    const std::vector<int64_t>& actor_int_params,
    const std::vector<double>& actor_float_params,
    bool deterministic,
    int64_t rng_step);

void actor_accel_live_fused_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    const TensorVec& actor_weights,
    const TensorVec& actor_int_tensors,
    const std::vector<int64_t>& actor_int_params,
    const std::vector<double>& actor_float_params,
    int64_t active_idx,
    bool deterministic,
    int64_t rng_step);

void actor_bw_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    const TensorVec& actor_weights,
    const TensorVec& actor_int_tensors,
    const std::vector<int64_t>& actor_int_params,
    const std::vector<double>& actor_float_params,
    bool deterministic,
    int64_t rng_step,
    int64_t history_slot);

void actor_bw_live_fused_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    const TensorVec& actor_weights,
    const TensorVec& actor_int_tensors,
    const std::vector<int64_t>& actor_int_params,
    const std::vector<double>& actor_float_params,
    bool deterministic,
    int64_t rng_step,
    int64_t history_slot);

void stage_mc_gae_launcher(
    at::Tensor values,
    at::Tensor mc_returns,
    at::Tensor terminated,
    at::Tensor truncated,
    at::Tensor returns_out,
    at::Tensor advantages_out,
    int64_t num_steps,
    int64_t num_envs,
    double gamma,
    double gae_lambda);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare_initial_accel_live", &prepare_initial_accel_live_launcher, "prepare_initial_accel_live");
  m.def("accel_to_sat_live", &accel_to_sat_live_launcher, "accel_to_sat_live");
  m.def("queue_aware_accel_live", &queue_aware_accel_live_launcher, "queue_aware_accel_live");
  m.def("cluster_center_accel_live", &cluster_center_accel_live_launcher, "cluster_center_accel_live");
  m.def("baseline_accel_live", &baseline_accel_live_launcher, "baseline_accel_live");
  m.def("queue_aware_sat_live", &queue_aware_sat_live_launcher, "queue_aware_sat_live");
  m.def("baseline_sat_live", &baseline_sat_live_launcher, "baseline_sat_live");
  m.def("queue_aware_bw_live", &queue_aware_bw_live_launcher, "queue_aware_bw_live");
  m.def("baseline_bw_live", &baseline_bw_live_launcher, "baseline_bw_live");
  m.def("sat_to_bw_live", &sat_to_bw_live_launcher, "sat_to_bw_live");
  m.def("finish_commit_prepare_live", &finish_commit_prepare_live_launcher, "finish_commit_prepare_live");
  m.def("apply_bw_macro_live", &apply_bw_macro_live_launcher, "apply_bw_macro_live");
  m.def("finish_commit_prepare_live_profiled", &finish_commit_prepare_live_profiled_launcher, "finish_commit_prepare_live_profiled");
  m.def("prepare_branch_replay_from_history", &prepare_branch_replay_from_history_launcher, "prepare_branch_replay_from_history");
  m.def("actor_accel_live", &actor_accel_live_launcher, "actor_accel_live");
  m.def("actor_accel_live_fused", &actor_accel_live_fused_launcher, "actor_accel_live_fused");
  m.def("actor_sat_live", &actor_sat_live_launcher, "actor_sat_live");
  m.def("actor_sat_live_fused", &actor_sat_live_fused_launcher, "actor_sat_live_fused");
  m.def("actor_bw_live", &actor_bw_live_launcher, "actor_bw_live");
  m.def("actor_bw_live_fused", &actor_bw_live_fused_launcher, "actor_bw_live_fused");
  m.def("stage_mc_gae", &stage_mc_gae_launcher, "stage_mc_gae");
}
