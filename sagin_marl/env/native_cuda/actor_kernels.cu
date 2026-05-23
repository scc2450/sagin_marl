#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using TensorVec = std::vector<at::Tensor>;

namespace {

constexpr int kMaxRuntimeFloatTensors = 448;
constexpr int kMaxRuntimeLongTensors = 64;
constexpr int kMaxRuntimeBoolTensors = 64;
constexpr int kMaxRuntimeIntTensors = 32;
constexpr int kMaxRuntimeIntParams = 96;
constexpr int kMaxRuntimeFloatParams = 224;
constexpr int kMaxActorWeights = 1152;
constexpr int kMaxActorIntTensors = 16;
constexpr int kMaxActorIntParams = 128;
constexpr int kMaxActorFloatParams = 64;
constexpr int kMaxHidden = 512;
constexpr int kMaxEmbed = 256;
constexpr int kMaxItems = 1024;
constexpr int kMaxSubset = 4096;
constexpr int kMaxSelect = 8;
constexpr float kNegInf = -1.0e9f;
constexpr float kLog2Pi = 1.8378770664093453f;

enum ActorIntParamIndex : int {
  kActorHidden = 0,
  kActorEmbed = 1,
  kActorSatUseTokens = 2,
  kActorCompetitionHeads = 3,
  kActorCompetitionLayers = 4,
  kActorRngSeedLo = 5,
  kActorRngSeedHi = 6,
  kActorAccelEgoDim = 7,
  kActorAccelCellDim = 8,
  kActorAccelGuTokenDim = 9,
  kActorAccelPeerTokenDim = 10,
  kActorAccelSatTokenDim = 11,
  kActorAccelGuQueryCount = 12,
  kActorAccelPeerQueryCount = 13,
  kActorAccelSatQueryCount = 14,
  kActorBwDownQueryCount = 15,
  kActorSatAttentionHeads = 16,
  kActorSatCompetitionLayers = 17,
  kActorSatBlockWeightBase = 18,
  kActorSatBlockWeightStride = 19,
  kActorAccelEncoderMlpLayers = 20,
  kActorAccelContextMlpLayers = 21,
  kActorAccelInteractionLayers = 22,
  kActorAccelAttentionHeads = 23,
  kActorAccelBlockWeightBase = 24,
  kActorAccelBlockWeightStride = 25,
  kActorSatEncoderMlpLayers = 26,
  kActorSatContextMlpLayers = 27,
  kActorSatHeadMlpLayers = 28,
  kActorBwEncoderMlpLayers = 29,
  kActorBwContextMlpLayers = 30,
  kActorBwHeadMlpLayers = 31,
  kActorExtraMlpWeightBase = 32,
  kActorExtraMlpWeightStride = 33,
  kActorAccelHeadMlpLayers = 34,
  kActorAccelMuHeadMlpWeightBase = 35,
  kActorAccelHidden = 36,
  kActorAccelEmbed = 37,
  kActorSatHidden = 38,
  kActorSatEmbed = 39,
  kActorBwHidden = 40,
  kActorBwEmbed = 41,
  kActorBwTauKappaPackedWeightBase = 42,
  kActorBwDirichletDiagnosticMode = 43,
};

enum ActorFloatParamIndex : int {
  kActorAccelActionScale = 0,
  kActorBwTauMin = 1,
  kActorBwTauMax = 2,
  kActorBwKappaMin = 3,
  kActorBwKappaMax = 4,
  kActorBwFixedTau = 5,
  kActorBwFixedKappa = 6,
};

enum ActorWeightIndex : int {
  W_ACCEL_LOG_STD = 0,
  W_ACCEL_EGO_NORM_WEIGHT = 1,
  W_ACCEL_EGO_NORM_BIAS = 2,
  W_ACCEL_CELL_NORM_WEIGHT = 3,
  W_ACCEL_CELL_NORM_BIAS = 4,
  W_ACCEL_GU_NORM_WEIGHT = 5,
  W_ACCEL_GU_NORM_BIAS = 6,
  W_ACCEL_PEER_NORM_WEIGHT = 7,
  W_ACCEL_PEER_NORM_BIAS = 8,
  W_ACCEL_SAT_NORM_WEIGHT = 9,
  W_ACCEL_SAT_NORM_BIAS = 10,
  W_ACCEL_EGO_ENC0_WEIGHT = 11,
  W_ACCEL_EGO_ENC0_BIAS = 12,
  W_ACCEL_EGO_ENC2_WEIGHT = 13,
  W_ACCEL_EGO_ENC2_BIAS = 14,
  W_ACCEL_CELL_ENC0_WEIGHT = 15,
  W_ACCEL_CELL_ENC0_BIAS = 16,
  W_ACCEL_CELL_ENC2_WEIGHT = 17,
  W_ACCEL_CELL_ENC2_BIAS = 18,
  W_ACCEL_GU_ENC0_WEIGHT = 19,
  W_ACCEL_GU_ENC0_BIAS = 20,
  W_ACCEL_GU_ENC2_WEIGHT = 21,
  W_ACCEL_GU_ENC2_BIAS = 22,
  W_ACCEL_PEER_ENC0_WEIGHT = 23,
  W_ACCEL_PEER_ENC0_BIAS = 24,
  W_ACCEL_PEER_ENC2_WEIGHT = 25,
  W_ACCEL_PEER_ENC2_BIAS = 26,
  W_ACCEL_SAT_ENC0_WEIGHT = 27,
  W_ACCEL_SAT_ENC0_BIAS = 28,
  W_ACCEL_SAT_ENC2_WEIGHT = 29,
  W_ACCEL_SAT_ENC2_BIAS = 30,
  W_ACCEL_GU_QUERY_WEIGHT = 31,
  W_ACCEL_GU_QUERY_BIAS = 32,
  W_ACCEL_PEER_QUERY_WEIGHT = 33,
  W_ACCEL_PEER_QUERY_BIAS = 34,
  W_ACCEL_SAT_QUERY_WEIGHT = 35,
  W_ACCEL_SAT_QUERY_BIAS = 36,
  W_ACCEL_FUSION0_WEIGHT = 37,
  W_ACCEL_FUSION0_BIAS = 38,
  W_ACCEL_FUSION2_WEIGHT = 39,
  W_ACCEL_FUSION2_BIAS = 40,
  W_ACCEL_MU_WEIGHT = 41,
  W_ACCEL_MU_BIAS = 42,
  W_SAT_EGO_NORM_WEIGHT = 43,
  W_SAT_EGO_NORM_BIAS = 44,
  W_SAT_SUBSET_NORM_WEIGHT = 45,
  W_SAT_SUBSET_NORM_BIAS = 46,
  W_SAT_EGO_ENC0_WEIGHT = 47,
  W_SAT_EGO_ENC0_BIAS = 48,
  W_SAT_EGO_ENC2_WEIGHT = 49,
  W_SAT_EGO_ENC2_BIAS = 50,
  W_SAT_Q1_0_WEIGHT = 51,
  W_SAT_Q1_0_BIAS = 52,
  W_SAT_Q1_2_WEIGHT = 53,
  W_SAT_Q1_2_BIAS = 54,
  W_SAT_Q2_0_WEIGHT = 55,
  W_SAT_Q2_0_BIAS = 56,
  W_SAT_Q2_2_WEIGHT = 57,
  W_SAT_Q2_2_BIAS = 58,
  W_SAT_INPUT_NORM_WEIGHT = 59,
  W_SAT_INPUT_NORM_BIAS = 60,
  W_SAT_ENC0_WEIGHT = 61,
  W_SAT_ENC0_BIAS = 62,
  W_SAT_ENC2_WEIGHT = 63,
  W_SAT_ENC2_BIAS = 64,
  W_SAT_REF0_WEIGHT = 65,
  W_SAT_REF0_BIAS = 66,
  W_SAT_REF2_WEIGHT = 67,
  W_SAT_REF2_BIAS = 68,
  W_SAT_EGO_FUSION0_WEIGHT = 69,
  W_SAT_EGO_FUSION0_BIAS = 70,
  W_SAT_EGO_FUSION2_WEIGHT = 71,
  W_SAT_EGO_FUSION2_BIAS = 72,
  W_SAT_SUBSET_ENC0_WEIGHT = 73,
  W_SAT_SUBSET_ENC0_BIAS = 74,
  W_SAT_SUBSET_ENC2_WEIGHT = 75,
  W_SAT_SUBSET_ENC2_BIAS = 76,
  W_SAT_PROJECT0_WEIGHT = 77,
  W_SAT_PROJECT0_BIAS = 78,
  W_SAT_PROJECT2_WEIGHT = 79,
  W_SAT_PROJECT2_BIAS = 80,
  W_SAT_SCORER0_WEIGHT = 81,
  W_SAT_SCORER0_BIAS = 82,
  W_SAT_SCORER2_WEIGHT = 83,
  W_SAT_SCORER2_BIAS = 84,
  W_BW_EGO_NORM_WEIGHT = 85,
  W_BW_EGO_NORM_BIAS = 86,
  W_BW_SAT_NORM_WEIGHT = 87,
  W_BW_SAT_NORM_BIAS = 88,
  W_BW_USER_NORM_WEIGHT = 89,
  W_BW_USER_NORM_BIAS = 90,
  W_BW_EGO_ENC0_WEIGHT = 91,
  W_BW_EGO_ENC0_BIAS = 92,
  W_BW_EGO_ENC2_WEIGHT = 93,
  W_BW_EGO_ENC2_BIAS = 94,
  W_BW_SAT_ENC0_WEIGHT = 95,
  W_BW_SAT_ENC0_BIAS = 96,
  W_BW_SAT_ENC2_WEIGHT = 97,
  W_BW_SAT_ENC2_BIAS = 98,
  W_BW_USER_ENC0_WEIGHT = 99,
  W_BW_USER_ENC0_BIAS = 100,
  W_BW_USER_ENC2_WEIGHT = 101,
  W_BW_USER_ENC2_BIAS = 102,
  W_BW_SAT_REF0_WEIGHT = 103,
  W_BW_SAT_REF0_BIAS = 104,
  W_BW_Q1_0_WEIGHT = 105,
  W_BW_Q1_0_BIAS = 106,
  W_BW_SAT_CTX0_WEIGHT = 107,
  W_BW_SAT_CTX0_BIAS = 108,
  W_BW_SAT_CTX2_WEIGHT = 109,
  W_BW_SAT_CTX2_BIAS = 110,
  W_BW_GLOBAL0_WEIGHT = 111,
  W_BW_GLOBAL0_BIAS = 112,
  W_BW_GLOBAL2_WEIGHT = 113,
  W_BW_GLOBAL2_BIAS = 114,
  W_BW_USER_CTX_FUSION0_WEIGHT = 115,
  W_BW_USER_CTX_FUSION0_BIAS = 116,
  W_BW_USER_CTX_FUSION2_WEIGHT = 117,
  W_BW_USER_CTX_FUSION2_BIAS = 118,
  W_BW_SCORE0_WEIGHT = 119,
  W_BW_SCORE0_BIAS = 120,
  W_BW_SCORE2_WEIGHT = 121,
  W_BW_SCORE2_BIAS = 122,
  W_BW_TAU0_WEIGHT = 123,
  W_BW_TAU0_BIAS = 124,
  W_BW_TAU2_WEIGHT = 125,
  W_BW_TAU2_BIAS = 126,
  W_BW_KAPPA0_WEIGHT = 127,
  W_BW_KAPPA0_BIAS = 128,
  W_BW_KAPPA2_WEIGHT = 129,
  W_BW_KAPPA2_BIAS = 130,
};

constexpr int W_BW_COMP_BASE = 131;
constexpr int W_BW_COMP_STRIDE = 12;
constexpr int W_BW_COMP_ATTN_IN_PROJ_WEIGHT = 0;
constexpr int W_BW_COMP_ATTN_IN_PROJ_BIAS = 1;
constexpr int W_BW_COMP_ATTN_OUT_PROJ_WEIGHT = 2;
constexpr int W_BW_COMP_ATTN_OUT_PROJ_BIAS = 3;
constexpr int W_BW_COMP_NORM_ATTN_WEIGHT = 4;
constexpr int W_BW_COMP_NORM_ATTN_BIAS = 5;
constexpr int W_BW_COMP_FFN0_WEIGHT = 6;
constexpr int W_BW_COMP_FFN0_BIAS = 7;
constexpr int W_BW_COMP_FFN2_WEIGHT = 8;
constexpr int W_BW_COMP_FFN2_BIAS = 9;
constexpr int W_BW_COMP_NORM_FFN_WEIGHT = 10;
constexpr int W_BW_COMP_NORM_FFN_BIAS = 11;

enum ActorExtraMlpSlot : int {
  X_ACCEL_EGO_ENCODER = 0,
  X_ACCEL_CELL_ENCODER = 1,
  X_ACCEL_GU_ENCODER = 2,
  X_ACCEL_PEER_ENCODER = 3,
  X_ACCEL_SAT_ENCODER = 4,
  X_ACCEL_FUSION = 5,
  X_SAT_EGO_ENCODER = 6,
  X_SAT_DEMAND_ENCODER = 7,
  X_SAT_ROLE_ENCODER = 8,
  X_SAT_SAT_ENCODER = 9,
  X_SAT_CTX_ENCODER = 10,
  X_SAT_CONTEXT_FUSION = 11,
  X_SAT_LOGIT_HEAD = 12,
  X_SAT_COUNT_HEAD = 13,
  X_BW_EGO_ENCODER = 14,
  X_BW_SAT_ENCODER = 15,
  X_BW_USER_ENCODER = 16,
  X_BW_DOWN_CONTEXT = 17,
  X_BW_CTX0 = 18,
  X_BW_USER_CONTEXT_FUSION = 19,
  X_BW_SCORE_HEAD = 20,
  X_BW_TAU_HEAD = 21,
  X_BW_KAPPA_HEAD = 22,
};

enum RuntimeIntParamIndex : int {
  kParamNumEnvs = 0,
  kParamNumUav = 1,
  kParamNumGu = 2,
  kParamNumSat = 3,
  kParamUsersObsMax = 4,
  kParamSatsObsMax = 5,
  kParamVisibleSatsMax = 6,
  kParamSatNumSelect = 7,
  kParamHistoryCapacity = 8,
  kParamSatVisibleWidth = 43,
  kParamSubsetCount = 44,
  kParamUavNodeDim = 45,
  kParamUserNodeDim = 46,
  kParamSatNodeDim = 47,
  kParamUavGuEdgeDim = 84,
  kParamUavSatEdgeDim = 85,
  kParamUavUavEdgeDim = 86,
  kParamAccelSatWidth = 87,
  kParamAccessBwDecisionInterval = 91,
  kParamSatDecisionInterval = 92,
};

enum RuntimeFloatTensorIndex : int {
  kFLiveAccelObs0 = 228,
  kFLiveAccelObs1 = 233,
  kFLiveSatObs = 238,
  kFLiveBwObs = 242,
  kFLiveAccelAction = 245,
  kFLiveBwAction = 246,
  kFLiveBwRefAction = 247,
  kFLiveBwFlowProxyOverrideAction = 248,
  kFLiveAccelOldLogprob = 249,
  kFLiveSatOldLogprobPerAgent = 250,
  kFLiveBwOldLogprob = 251,
  kFLiveBwOldLogprobPerAgent = 252,
  kFHistBwActions = 305,
  kFHistBwOldLogprobs = 306,
  kFHistBwRefActions = 318,
  kFHistBwOldLogprobsPerAgent = 319,
  kFActorScratch = 387,
  kFLiveBwEntropyPerAgent = 401,
  kFLiveBwLogprobRawPerAgent = 402,
  kFLiveBwEntropyRawPerAgent = 403,
  kFLiveBwTau = 404,
  kFLiveBwKappa = 405,
  kFHistBwEntropyPerAgent = 406,
  kFHistBwLogprobRawPerAgent = 407,
  kFHistBwEntropyRawPerAgent = 408,
  kFHistBwTau = 409,
  kFHistBwKappa = 410,
  kFLiveSatEntropyPerAgent = 411,
  kFLiveAccelLatentAction = 412,
};

enum RuntimeLongTensorIndex : int {
  kLLiveSatSubsetMembers = 29,
  kLLiveSatSubsetIndex = 30,
  kLMainSatSubsetMembersBase = 36,
  kLLiveBwValidCount = 51,
  kLLiveBwLatentCount = 52,
  kLHistBwValidCount = 53,
  kLHistBwLatentCount = 54,
  kLLiveSatCandidateIds = 55,
  kLLiveSatActionIndices = 57,
};

enum RuntimeBoolTensorIndex : int {
  kBLiveAccelObs0 = 11,
  kBLiveAccelObs1 = 14,
  kBLiveSatObs = 17,
  kBLiveBwObs = 19,
  kBHistTerminated = 53,
  kBHistTruncated = 54,
};

constexpr int kBwEgoDim = 11;
constexpr int kBwSatTokenDim = 9;
constexpr int kBwGuTokenDim = 13;

struct RuntimePackedAbi {
  float* f[kMaxRuntimeFloatTensors];
  int64_t* l[kMaxRuntimeLongTensors];
  bool* b[kMaxRuntimeBoolTensors];
  int* i[kMaxRuntimeIntTensors];
  int64_t f_numel[kMaxRuntimeFloatTensors];
  int64_t l_numel[kMaxRuntimeLongTensors];
  int64_t b_numel[kMaxRuntimeBoolTensors];
  int64_t i_numel[kMaxRuntimeIntTensors];
  int64_t ip[kMaxRuntimeIntParams];
  double fp[kMaxRuntimeFloatParams];
  int nf;
  int nl;
  int nb;
  int ni;
  int nip;
  int nfp;
};

struct ActorPackedAbi {
  float* w[kMaxActorWeights];
  int* i[kMaxActorIntTensors];
  int64_t w_numel[kMaxActorWeights];
  int64_t i_numel[kMaxActorIntTensors];
  int64_t ip[kMaxActorIntParams];
  double fp[kMaxActorFloatParams];
  int nw;
  int ni;
  int nip;
  int nfp;
};

__constant__ RuntimePackedAbi cActorRuntimeAbi;
__constant__ ActorPackedAbi cActorAbi;

RuntimePackedAbi pack_runtime_abi(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params) {
  if (float_tensors.size() > kMaxRuntimeFloatTensors || long_tensors.size() > kMaxRuntimeLongTensors ||
      bool_tensors.size() > kMaxRuntimeBoolTensors || int_tensors.size() > kMaxRuntimeIntTensors ||
      int_params.size() > kMaxRuntimeIntParams || float_params.size() > kMaxRuntimeFloatParams) {
    throw std::runtime_error("native CUDA actor runtime ABI exceeds compiled slot capacity.");
  }
  RuntimePackedAbi out{};
  out.nf = static_cast<int>(float_tensors.size());
  out.nl = static_cast<int>(long_tensors.size());
  out.nb = static_cast<int>(bool_tensors.size());
  out.ni = static_cast<int>(int_tensors.size());
  out.nip = static_cast<int>(int_params.size());
  out.nfp = static_cast<int>(float_params.size());
  for (int idx = 0; idx < out.nf; ++idx) {
    out.f[idx] = float_tensors[static_cast<size_t>(idx)].data_ptr<float>();
    out.f_numel[idx] = float_tensors[static_cast<size_t>(idx)].numel();
  }
  for (int idx = 0; idx < out.nl; ++idx) {
    out.l[idx] = long_tensors[static_cast<size_t>(idx)].data_ptr<int64_t>();
    out.l_numel[idx] = long_tensors[static_cast<size_t>(idx)].numel();
  }
  for (int idx = 0; idx < out.nb; ++idx) {
    out.b[idx] = bool_tensors[static_cast<size_t>(idx)].data_ptr<bool>();
    out.b_numel[idx] = bool_tensors[static_cast<size_t>(idx)].numel();
  }
  for (int idx = 0; idx < out.ni; ++idx) {
    out.i[idx] = int_tensors[static_cast<size_t>(idx)].data_ptr<int>();
    out.i_numel[idx] = int_tensors[static_cast<size_t>(idx)].numel();
  }
  for (int idx = 0; idx < out.nip; ++idx) {
    out.ip[idx] = int_params[static_cast<size_t>(idx)];
  }
  for (int idx = 0; idx < out.nfp; ++idx) {
    out.fp[idx] = float_params[static_cast<size_t>(idx)];
  }
  return out;
}

ActorPackedAbi pack_actor_abi(
    const TensorVec& actor_weights,
    const TensorVec& actor_int_tensors,
    const std::vector<int64_t>& actor_int_params,
    const std::vector<double>& actor_float_params) {
  if (actor_weights.size() > kMaxActorWeights || actor_int_tensors.size() > kMaxActorIntTensors ||
      actor_int_params.size() > kMaxActorIntParams || actor_float_params.size() > kMaxActorFloatParams) {
    throw std::runtime_error("native CUDA actor ABI exceeds compiled slot capacity.");
  }
  ActorPackedAbi out{};
  out.nw = static_cast<int>(actor_weights.size());
  out.ni = static_cast<int>(actor_int_tensors.size());
  out.nip = static_cast<int>(actor_int_params.size());
  out.nfp = static_cast<int>(actor_float_params.size());
  for (int idx = 0; idx < out.nw; ++idx) {
    out.w[idx] = actor_weights[static_cast<size_t>(idx)].data_ptr<float>();
    out.w_numel[idx] = actor_weights[static_cast<size_t>(idx)].numel();
  }
  for (int idx = 0; idx < out.ni; ++idx) {
    out.i[idx] = actor_int_tensors[static_cast<size_t>(idx)].data_ptr<int>();
    out.i_numel[idx] = actor_int_tensors[static_cast<size_t>(idx)].numel();
  }
  for (int idx = 0; idx < out.nip; ++idx) {
    out.ip[idx] = actor_int_params[static_cast<size_t>(idx)];
  }
  for (int idx = 0; idx < out.nfp; ++idx) {
    out.fp[idx] = actor_float_params[static_cast<size_t>(idx)];
  }
  return out;
}

void copy_actor_abi_to_symbols(const RuntimePackedAbi& runtime, const ActorPackedAbi& actor, cudaStream_t stream) {
  C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(cActorRuntimeAbi, &runtime, sizeof(RuntimePackedAbi), 0, cudaMemcpyHostToDevice, stream));
  C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(cActorAbi, &actor, sizeof(ActorPackedAbi), 0, cudaMemcpyHostToDevice, stream));
}

__device__ __forceinline__ int64_t ip(const RuntimePackedAbi& a, int idx, int64_t default_value = 0) {
  return (idx >= 0 && idx < a.nip) ? a.ip[idx] : default_value;
}

__device__ __forceinline__ int64_t aip(const ActorPackedAbi& a, int idx, int64_t default_value = 0) {
  return (idx >= 0 && idx < a.nip) ? a.ip[idx] : default_value;
}

__device__ __forceinline__ float afp(const ActorPackedAbi& a, int idx, float default_value = 0.0f) {
  return (idx >= 0 && idx < a.nfp) ? static_cast<float>(a.fp[idx]) : default_value;
}

__device__ __forceinline__ bool has_w(const ActorPackedAbi& a, int idx) {
  return idx >= 0 && idx < a.nw && a.w[idx] != nullptr && a.w_numel[idx] > 0;
}

__device__ __forceinline__ bool has_f(const RuntimePackedAbi& a, int idx) {
  return idx >= 0 && idx < a.nf && a.f[idx] != nullptr && a.f_numel[idx] > 0;
}

__device__ __forceinline__ bool has_l(const RuntimePackedAbi& a, int idx) {
  return idx >= 0 && idx < a.nl && a.l[idx] != nullptr && a.l_numel[idx] > 0;
}

__device__ __forceinline__ bool has_b(const RuntimePackedAbi& a, int idx) {
  return idx >= 0 && idx < a.nb && a.b[idx] != nullptr && a.b_numel[idx] > 0;
}

__device__ __forceinline__ float relu_device(float x) {
  return x > 0.0f ? x : 0.0f;
}

__device__ __forceinline__ float sigmoid_device(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_device(float x) {
  return x * sigmoid_device(x);
}

struct Philox4x32 {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t w;
};

__device__ __forceinline__ uint32_t mulhilo_u32(uint32_t a, uint32_t b, uint32_t* hi) {
  const unsigned long long product =
      static_cast<unsigned long long>(a) * static_cast<unsigned long long>(b);
  *hi = static_cast<uint32_t>(product >> 32);
  return static_cast<uint32_t>(product);
}

__device__ __forceinline__ Philox4x32 philox4x32_round(Philox4x32 c, uint32_t key0, uint32_t key1) {
  uint32_t hi0 = 0;
  uint32_t hi1 = 0;
  const uint32_t lo0 = mulhilo_u32(0xD2511F53u, c.x, &hi0);
  const uint32_t lo1 = mulhilo_u32(0xCD9E8D57u, c.z, &hi1);
  Philox4x32 out{};
  out.x = hi1 ^ c.y ^ key0;
  out.y = lo1;
  out.z = hi0 ^ c.w ^ key1;
  out.w = lo0;
  return out;
}

__device__ __forceinline__ Philox4x32 philox4x32_10(Philox4x32 c, uint32_t key0, uint32_t key1) {
  #pragma unroll
  for (int round = 0; round < 10; ++round) {
    c = philox4x32_round(c, key0, key1);
    key0 += 0x9E3779B9u;
    key1 += 0xBB67AE85u;
  }
  return c;
}

__device__ __forceinline__ uint32_t philox_word(
    const ActorPackedAbi& actor,
    int64_t rng_step,
    int row,
    int stream,
    int draw) {
  const uint64_t step = static_cast<uint64_t>(rng_step);
  Philox4x32 counter{};
  counter.x = static_cast<uint32_t>(step);
  counter.y = static_cast<uint32_t>(step >> 32) ^ static_cast<uint32_t>(row);
  counter.z = static_cast<uint32_t>(stream);
  counter.w = static_cast<uint32_t>(draw >> 2);
  const uint32_t key0 = static_cast<uint32_t>(aip(actor, kActorRngSeedLo, 0xA123B456));
  const uint32_t key1 = static_cast<uint32_t>(aip(actor, kActorRngSeedHi, 0xC789D012));
  const Philox4x32 out = philox4x32_10(counter, key0, key1);
  switch (draw & 3) {
    case 0: return out.x;
    case 1: return out.y;
    case 2: return out.z;
    default: return out.w;
  }
}

__device__ __forceinline__ float uniform01(
    const ActorPackedAbi& actor,
    int64_t rng_step,
    int row,
    int stream,
    int draw = 0) {
  const uint32_t x = philox_word(actor, rng_step, row, stream, draw);
  return (static_cast<float>(x & 0x00ffffffu) + 1.0f) / 16777217.0f;
}

__device__ __forceinline__ float normal01(
    const ActorPackedAbi& actor,
    int64_t rng_step,
    int row,
    int stream) {
  const float u1 = fmaxf(uniform01(actor, rng_step, row, stream, 0), 1.0e-7f);
  const float u2 = uniform01(actor, rng_step, row, stream, 1);
  return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

__device__ void layer_norm_or_copy(
    const ActorPackedAbi& actor,
    const float* in,
    float* out,
    int dim,
    int weight_idx,
    int bias_idx) {
  if (!has_w(actor, weight_idx) || !has_w(actor, bias_idx)) {
    for (int i = 0; i < dim; ++i) {
      out[i] = in[i];
    }
    return;
  }
  float mean = 0.0f;
  for (int i = 0; i < dim; ++i) {
    mean += in[i];
  }
  mean /= fmaxf(static_cast<float>(dim), 1.0f);
  float var = 0.0f;
  for (int i = 0; i < dim; ++i) {
    const float d = in[i] - mean;
    var += d * d;
  }
  var /= fmaxf(static_cast<float>(dim), 1.0f);
  const float inv_std = rsqrtf(var + 1.0e-5f);
  const float* weight = actor.w[weight_idx];
  const float* bias = actor.w[bias_idx];
  for (int i = 0; i < dim; ++i) {
    out[i] = (in[i] - mean) * inv_std * weight[i] + bias[i];
  }
}

__device__ void linear_layer(
    const ActorPackedAbi& actor,
    const float* in,
    float* out,
    int in_dim,
    int out_dim,
    int weight_idx,
    int bias_idx,
    bool relu) {
  const float* weight = actor.w[weight_idx];
  const float* bias = actor.w[bias_idx];
  for (int o = 0; o < out_dim; ++o) {
    float acc = bias[o];
    const float* row = weight + o * in_dim;
    for (int i = 0; i < in_dim; ++i) {
      acc += row[i] * in[i];
    }
    out[o] = relu ? relu_device(acc) : acc;
  }
}

__device__ void mlp2(
    const ActorPackedAbi& actor,
    const float* in,
    float* out,
    float* scratch,
    int in_dim,
    int hidden_dim,
    int out_dim,
    int w0,
    int b0,
    int w1,
    int b1,
    bool final_relu) {
  linear_layer(actor, in, scratch, in_dim, hidden_dim, w0, b0, true);
  linear_layer(actor, scratch, out, hidden_dim, out_dim, w1, b1, final_relu);
}

__device__ void attend_tokens(
    const float* query,
    const float* tokens,
    const bool* mask,
    int count,
    int embed_dim,
    float* out) {
  for (int d = 0; d < embed_dim; ++d) {
    out[d] = 0.0f;
  }
  if (count <= 0) {
    return;
  }
  float max_score = kNegInf;
  bool any = false;
  const float scale = rsqrtf(fmaxf(static_cast<float>(embed_dim), 1.0f));
  for (int item = 0; item < count; ++item) {
    if (mask != nullptr && !mask[item]) {
      continue;
    }
    any = true;
    float score = 0.0f;
    const float* tok = tokens + item * embed_dim;
    for (int d = 0; d < embed_dim; ++d) {
      score += query[d] * tok[d];
    }
    score *= scale;
    max_score = fmaxf(max_score, score);
  }
  if (!any) {
    return;
  }
  float denom = 0.0f;
  for (int item = 0; item < count; ++item) {
    if (mask != nullptr && !mask[item]) {
      continue;
    }
    float score = 0.0f;
    const float* tok = tokens + item * embed_dim;
    for (int d = 0; d < embed_dim; ++d) {
      score += query[d] * tok[d];
    }
    const float w = expf(score * scale - max_score);
    denom += w;
    for (int d = 0; d < embed_dim; ++d) {
      out[d] += w * tok[d];
    }
  }
  const float inv = 1.0f / fmaxf(denom, 1.0e-8f);
  for (int d = 0; d < embed_dim; ++d) {
    out[d] *= inv;
  }
}

__device__ void concat2(const float* a, const float* b, float* out, int dim_a, int dim_b) {
  for (int i = 0; i < dim_a; ++i) {
    out[i] = a[i];
  }
  for (int i = 0; i < dim_b; ++i) {
    out[dim_a + i] = b[i];
  }
}

__device__ void concat4(const float* a, const float* b, const float* c, const float* d, float* out, int dim) {
  for (int i = 0; i < dim; ++i) {
    out[i] = a[i];
    out[dim + i] = b[i];
    out[2 * dim + i] = c[i];
    out[3 * dim + i] = d[i];
  }
}

__device__ float normal_logprob_device(float mean, float log_std, float z) {
  const float inv_std = expf(-log_std);
  const float n = (z - mean) * inv_std;
  return -0.5f * (n * n + kLog2Pi) - log_std;
}

__device__ __forceinline__ float softplus_device(float x) {
  return x > 20.0f ? x : log1pf(expf(x));
}

__device__ void block_zero(float* out, int dim) {
  for (int i = threadIdx.x; i < dim; i += blockDim.x) out[i] = 0.0f;
}

__device__ void block_copy(const float* in, float* out, int dim) {
  for (int i = threadIdx.x; i < dim; i += blockDim.x) out[i] = in[i];
}

__device__ void block_layer_norm_or_copy(
    const ActorPackedAbi& actor,
    const float* in,
    float* out,
    int dim,
    int weight_idx,
    int bias_idx,
    float* reduce_scratch) {
  if (!has_w(actor, weight_idx) || !has_w(actor, bias_idx)) {
    block_copy(in, out, dim);
    __syncthreads();
    return;
  }
  float partial = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) partial += in[i];
  reduce_scratch[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
    __syncthreads();
  }
  const float mean = reduce_scratch[0] / fmaxf(static_cast<float>(dim), 1.0f);
  partial = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float d = in[i] - mean;
    partial += d * d;
  }
  reduce_scratch[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv_std = rsqrtf(reduce_scratch[0] / fmaxf(static_cast<float>(dim), 1.0f) + 1.0e-5f);
  const float* weight = actor.w[weight_idx];
  const float* bias = actor.w[bias_idx];
  for (int i = threadIdx.x; i < dim; i += blockDim.x) out[i] = (in[i] - mean) * inv_std * weight[i] + bias[i];
  __syncthreads();
}

__device__ void block_linear(
    const ActorPackedAbi& actor,
    const float* in,
    float* out,
    int in_dim,
    int out_dim,
    int weight_idx,
    int bias_idx,
    bool relu) {
  const float* weight = actor.w[weight_idx];
  const float* bias = actor.w[bias_idx];
  for (int o = threadIdx.x; o < out_dim; o += blockDim.x) {
    float acc = bias[o];
    const float* row = weight + o * in_dim;
    for (int i = 0; i < in_dim; ++i) acc += row[i] * in[i];
    out[o] = relu ? relu_device(acc) : acc;
  }
  __syncthreads();
}

__device__ void block_mlp2(
    const ActorPackedAbi& actor,
    const float* in,
    float* out,
    float* hidden,
    int in_dim,
    int hidden_dim,
    int out_dim,
    int w0,
    int b0,
    int w1,
    int b1,
    bool final_relu) {
  block_linear(actor, in, hidden, in_dim, hidden_dim, w0, b0, true);
  block_linear(actor, hidden, out, hidden_dim, out_dim, w1, b1, final_relu);
}

__device__ void block_mlp2_silu(
    const ActorPackedAbi& actor,
    const float* in,
    float* out,
    float* hidden,
    int in_dim,
    int hidden_dim,
    int out_dim,
    int w0,
    int b0,
    int w1,
    int b1) {
  const float* weight0 = actor.w[w0];
  const float* bias0 = actor.w[b0];
  for (int h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
    float acc = bias0[h];
    const float* row = weight0 + h * in_dim;
    for (int i = 0; i < in_dim; ++i) acc += row[i] * in[i];
    hidden[h] = silu_device(acc);
  }
  __syncthreads();
  const float* weight1 = actor.w[w1];
  const float* bias1 = actor.w[b1];
  for (int o = threadIdx.x; o < out_dim; o += blockDim.x) {
    float acc = bias1[o];
    const float* row = weight1 + o * hidden_dim;
    for (int h = 0; h < hidden_dim; ++h) acc += row[h] * hidden[h];
    out[o] = acc;
  }
  __syncthreads();
}

__device__ __forceinline__ float actor_activate(float x, int activation_kind) {
  return activation_kind == 1 ? silu_device(x) : relu_device(x);
}

__device__ __forceinline__ int extra_mlp_weight_index(
    const ActorPackedAbi& actor,
    int slot,
    int extra_pair,
    bool bias) {
  const int base = static_cast<int>(aip(actor, kActorExtraMlpWeightBase, 0));
  const int stride = static_cast<int>(aip(actor, kActorExtraMlpWeightStride, 4));
  return base + slot * stride + extra_pair * 2 + (bias ? 1 : 0);
}

__device__ void block_linear_activation(
    const ActorPackedAbi& actor,
    const float* in,
    float* out,
    int in_dim,
    int out_dim,
    int weight_idx,
    int bias_idx,
    bool activate,
    int activation_kind) {
  const float* weight = actor.w[weight_idx];
  const float* bias = actor.w[bias_idx];
  for (int o = threadIdx.x; o < out_dim; o += blockDim.x) {
    float acc = bias[o];
    const float* row = weight + o * in_dim;
    for (int i = 0; i < in_dim; ++i) acc += row[i] * in[i];
    out[o] = activate ? actor_activate(acc, activation_kind) : acc;
  }
  __syncthreads();
}

__device__ void block_mlp_flex(
    const ActorPackedAbi& actor,
    const float* in,
    float* out,
    float* hidden0,
    float* hidden1,
    int in_dim,
    int hidden_dim,
    int out_dim,
    int w0,
    int b0,
    int w1,
    int b1,
    int extra_slot,
    int layers,
    bool final_activate,
    int activation_kind) {
  layers = min(max(layers, 1), 4);
  if (layers <= 1) {
    block_linear_activation(actor, in, out, in_dim, out_dim, w0, b0, final_activate, activation_kind);
    return;
  }
  block_linear_activation(actor, in, hidden0, in_dim, hidden_dim, w0, b0, true, activation_kind);
  if (layers == 2) {
    block_linear_activation(actor, hidden0, out, hidden_dim, out_dim, w1, b1, final_activate, activation_kind);
    return;
  }
  block_linear_activation(actor, hidden0, hidden1, hidden_dim, hidden_dim, w1, b1, true, activation_kind);
  if (layers == 3) {
    block_linear_activation(
        actor,
        hidden1,
        out,
        hidden_dim,
        out_dim,
        extra_mlp_weight_index(actor, extra_slot, 0, false),
        extra_mlp_weight_index(actor, extra_slot, 0, true),
        final_activate,
        activation_kind);
    return;
  }
  block_linear_activation(
      actor,
      hidden1,
      hidden0,
      hidden_dim,
      hidden_dim,
      extra_mlp_weight_index(actor, extra_slot, 0, false),
      extra_mlp_weight_index(actor, extra_slot, 0, true),
      true,
      activation_kind);
  block_linear_activation(
      actor,
      hidden0,
      out,
      hidden_dim,
      out_dim,
      extra_mlp_weight_index(actor, extra_slot, 1, false),
      extra_mlp_weight_index(actor, extra_slot, 1, true),
      final_activate,
      activation_kind);
}

__device__ void block_mlp_flex_contiguous(
    const ActorPackedAbi& actor,
    const float* in,
    float* out,
    float* hidden0,
    float* hidden1,
    int in_dim,
    int hidden_dim,
    int out_dim,
    int weight_base,
    int layers,
    bool final_activate,
    int activation_kind) {
  layers = min(max(layers, 1), 4);
  if (layers <= 1) {
    block_linear_activation(actor, in, out, in_dim, out_dim, weight_base + 0, weight_base + 1, final_activate, activation_kind);
    return;
  }
  block_linear_activation(actor, in, hidden0, in_dim, hidden_dim, weight_base + 0, weight_base + 1, true, activation_kind);
  if (layers == 2) {
    block_linear_activation(actor, hidden0, out, hidden_dim, out_dim, weight_base + 2, weight_base + 3, final_activate, activation_kind);
    return;
  }
  block_linear_activation(actor, hidden0, hidden1, hidden_dim, hidden_dim, weight_base + 2, weight_base + 3, true, activation_kind);
  if (layers == 3) {
    block_linear_activation(actor, hidden1, out, hidden_dim, out_dim, weight_base + 4, weight_base + 5, final_activate, activation_kind);
    return;
  }
  block_linear_activation(actor, hidden1, hidden0, hidden_dim, hidden_dim, weight_base + 4, weight_base + 5, true, activation_kind);
  block_linear_activation(actor, hidden0, out, hidden_dim, out_dim, weight_base + 6, weight_base + 7, final_activate, activation_kind);
}

__device__ void block_attention(
    const float* query,
    const float* tokens,
    const bool* mask,
    int count,
    int embed_dim,
    float* out,
    float* reduce_scratch) {
  for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) out[d] = 0.0f;
  __syncthreads();
  if (count <= 0) return;
  float* score_scratch = reduce_scratch + blockDim.x;
  const bool cache_scores = count <= blockDim.x;
  const float scale = rsqrtf(fmaxf(static_cast<float>(embed_dim), 1.0f));
  float local_max = kNegInf;
  bool local_any = false;
  for (int item = threadIdx.x; item < count; item += blockDim.x) {
    if (mask != nullptr && !mask[item]) continue;
    local_any = true;
    float score = 0.0f;
    const float* tok = tokens + item * embed_dim;
    for (int d = 0; d < embed_dim; ++d) score += query[d] * tok[d];
    score *= scale;
    if (cache_scores) score_scratch[item] = score;
    local_max = fmaxf(local_max, score);
  }
  reduce_scratch[threadIdx.x] = local_any ? local_max : kNegInf;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce_scratch[threadIdx.x] = fmaxf(reduce_scratch[threadIdx.x], reduce_scratch[threadIdx.x + stride]);
    __syncthreads();
  }
  const float max_score = reduce_scratch[0];
  if (max_score <= kNegInf * 0.5f) return;
  float local_denom = 0.0f;
  for (int item = threadIdx.x; item < count; item += blockDim.x) {
    if (mask != nullptr && !mask[item]) continue;
    float score = cache_scores ? score_scratch[item] : 0.0f;
    if (!cache_scores) {
      const float* tok = tokens + item * embed_dim;
      for (int d = 0; d < embed_dim; ++d) score += query[d] * tok[d];
      score *= scale;
    }
    local_denom += expf(score - max_score);
  }
  reduce_scratch[threadIdx.x] = local_denom;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
    __syncthreads();
  }
  const float denom = fmaxf(reduce_scratch[0], 1.0e-8f);
  for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int item = 0; item < count; ++item) {
      if (mask != nullptr && !mask[item]) continue;
      const float* tok = tokens + item * embed_dim;
      float score = cache_scores ? score_scratch[item] : 0.0f;
      if (!cache_scores) {
        for (int j = 0; j < embed_dim; ++j) score += query[j] * tok[j];
        score *= scale;
      }
      acc += expf(score - max_score) * tok[d];
    }
    out[d] = acc / denom;
  }
  __syncthreads();
}

__device__ void block_multi_query_attention(
    const float* queries,
    int query_count,
    const float* tokens,
    const bool* mask,
    int count,
    int embed_dim,
    float* out,
    float* reduce_scratch) {
  for (int idx = threadIdx.x; idx < query_count * embed_dim; idx += blockDim.x) out[idx] = 0.0f;
  __syncthreads();
  for (int q = 0; q < query_count; ++q) {
    block_attention(queries + q * embed_dim, tokens, mask, count, embed_dim, out + q * embed_dim, reduce_scratch);
  }
}

__device__ float block_logsumexp(const float* logits, const bool* mask, int count, float* reduce_scratch) {
  float local_max = kNegInf;
  bool any = false;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    if (mask != nullptr && !mask[i]) continue;
    any = true;
    local_max = fmaxf(local_max, logits[i]);
  }
  reduce_scratch[threadIdx.x] = any ? local_max : kNegInf;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce_scratch[threadIdx.x] = fmaxf(reduce_scratch[threadIdx.x], reduce_scratch[threadIdx.x + stride]);
    __syncthreads();
  }
  const float max_v = reduce_scratch[0];
  if (max_v <= kNegInf * 0.5f) return 0.0f;
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    if (mask != nullptr && !mask[i]) continue;
    local_sum += expf(logits[i] - max_v);
  }
  reduce_scratch[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
    __syncthreads();
  }
  return max_v + logf(fmaxf(reduce_scratch[0], 1.0e-8f));
}

__device__ int block_argmax_masked(const float* logits, const bool* mask, int count) {
  int best = -1;
  float best_v = kNegInf;
  for (int i = 0; i < count; ++i) {
    if (mask != nullptr && !mask[i]) continue;
    const float v = logits[i];
    if (best < 0 || v > best_v) {
      best = i;
      best_v = v;
    }
  }
  return best;
}

__device__ int sample_categorical_masked(
    const float* logits,
    const bool* mask,
    int count,
    float logsum,
    const ActorPackedAbi& actor,
    int64_t rng_step,
    int row,
    int stream) {
  const float u = uniform01(actor, rng_step, row, stream);
  float acc = 0.0f;
  int last = -1;
  for (int i = 0; i < count; ++i) {
    if (mask != nullptr && !mask[i]) continue;
    last = i;
    acc += expf(logits[i] - logsum);
    if (u <= acc) return i;
  }
  return last >= 0 ? last : -1;
}

__device__ __forceinline__ float* actor_row_scratch(const RuntimePackedAbi& runtime, int row, int* stride_out) {
  if (kFActorScratch >= runtime.nf || runtime.f[kFActorScratch] == nullptr) {
    *stride_out = 0;
    return nullptr;
  }
  const int rows = static_cast<int>(ip(runtime, kParamNumEnvs)) * static_cast<int>(ip(runtime, kParamNumUav));
  const int stride = rows > 0 ? static_cast<int>(runtime.f_numel[kFActorScratch] / rows) : 0;
  *stride_out = stride;
  return runtime.f[kFActorScratch] + static_cast<int64_t>(row) * stride;
}

__device__ __forceinline__ float* scratch_alloc(float*& cursor, int& remaining, int n) {
  if (n <= 0 || remaining < n) return nullptr;
  float* out = cursor;
  cursor += n;
  remaining -= n;
  return out;
}

__device__ void block_mlp2_items(
    const ActorPackedAbi& actor,
    const float* in,
    float* out,
    float* hidden,
    int count,
    int in_dim,
    int hidden_dim,
    int out_dim,
    int w0,
    int b0,
    int w1,
    int b1,
    bool final_relu) {
  const float* weight0 = actor.w[w0];
  const float* bias0 = actor.w[b0];
  for (int idx = threadIdx.x; idx < count * hidden_dim; idx += blockDim.x) {
    const int item = idx / hidden_dim;
    const int h = idx - item * hidden_dim;
    float acc = bias0[h];
    const float* row_w = weight0 + h * in_dim;
    const float* row_in = in + item * in_dim;
    for (int i = 0; i < in_dim; ++i) acc += row_w[i] * row_in[i];
    hidden[idx] = relu_device(acc);
  }
  __syncthreads();
  const float* weight1 = actor.w[w1];
  const float* bias1 = actor.w[b1];
  for (int idx = threadIdx.x; idx < count * out_dim; idx += blockDim.x) {
    const int item = idx / out_dim;
    const int o = idx - item * out_dim;
    float acc = bias1[o];
    const float* row_w = weight1 + o * hidden_dim;
    const float* row_h = hidden + item * hidden_dim;
    for (int h = 0; h < hidden_dim; ++h) acc += row_w[h] * row_h[h];
    out[idx] = final_relu ? relu_device(acc) : acc;
  }
  __syncthreads();
}

__device__ void block_mlp2_items_silu(
    const ActorPackedAbi& actor,
    const float* in,
    float* out,
    float* hidden,
    int count,
    int in_dim,
    int hidden_dim,
    int out_dim,
    int w0,
    int b0,
    int w1,
    int b1) {
  const float* weight0 = actor.w[w0];
  const float* bias0 = actor.w[b0];
  for (int idx = threadIdx.x; idx < count * hidden_dim; idx += blockDim.x) {
    const int item = idx / hidden_dim;
    const int h = idx - item * hidden_dim;
    float acc = bias0[h];
    const float* row_w = weight0 + h * in_dim;
    const float* row_in = in + item * in_dim;
    for (int i = 0; i < in_dim; ++i) acc += row_w[i] * row_in[i];
    hidden[idx] = silu_device(acc);
  }
  __syncthreads();
  const float* weight1 = actor.w[w1];
  const float* bias1 = actor.w[b1];
  for (int idx = threadIdx.x; idx < count * out_dim; idx += blockDim.x) {
    const int item = idx / out_dim;
    const int o = idx - item * out_dim;
    float acc = bias1[o];
    const float* row_w = weight1 + o * hidden_dim;
    const float* row_h = hidden + item * hidden_dim;
    for (int h = 0; h < hidden_dim; ++h) acc += row_w[h] * row_h[h];
    out[idx] = acc;
  }
  __syncthreads();
}

__device__ void block_linear_items_activation(
    const ActorPackedAbi& actor,
    const float* in,
    float* out,
    int count,
    int in_dim,
    int out_dim,
    int weight_idx,
    int bias_idx,
    bool activate,
    int activation_kind) {
  const float* weight = actor.w[weight_idx];
  const float* bias = actor.w[bias_idx];
  for (int idx = threadIdx.x; idx < count * out_dim; idx += blockDim.x) {
    const int item = idx / out_dim;
    const int o = idx - item * out_dim;
    float acc = bias[o];
    const float* row_w = weight + o * in_dim;
    const float* row_in = in + item * in_dim;
    for (int i = 0; i < in_dim; ++i) acc += row_w[i] * row_in[i];
    out[idx] = activate ? actor_activate(acc, activation_kind) : acc;
  }
  __syncthreads();
}

__device__ void block_mlp_items_flex(
    const ActorPackedAbi& actor,
    const float* in,
    float* out,
    float* hidden0,
    float* hidden1,
    int count,
    int in_dim,
    int hidden_dim,
    int out_dim,
    int w0,
    int b0,
    int w1,
    int b1,
    int extra_slot,
    int layers,
    bool final_activate,
    int activation_kind) {
  layers = min(max(layers, 1), 4);
  if (count <= 0) return;
  if (layers <= 1) {
    block_linear_items_activation(actor, in, out, count, in_dim, out_dim, w0, b0, final_activate, activation_kind);
    return;
  }
  block_linear_items_activation(actor, in, hidden0, count, in_dim, hidden_dim, w0, b0, true, activation_kind);
  if (layers == 2) {
    block_linear_items_activation(actor, hidden0, out, count, hidden_dim, out_dim, w1, b1, final_activate, activation_kind);
    return;
  }
  block_linear_items_activation(actor, hidden0, hidden1, count, hidden_dim, hidden_dim, w1, b1, true, activation_kind);
  if (layers == 3) {
    block_linear_items_activation(
        actor,
        hidden1,
        out,
        count,
        hidden_dim,
        out_dim,
        extra_mlp_weight_index(actor, extra_slot, 0, false),
        extra_mlp_weight_index(actor, extra_slot, 0, true),
        final_activate,
        activation_kind);
    return;
  }
  block_linear_items_activation(
      actor,
      hidden1,
      hidden0,
      count,
      hidden_dim,
      hidden_dim,
      extra_mlp_weight_index(actor, extra_slot, 0, false),
      extra_mlp_weight_index(actor, extra_slot, 0, true),
      true,
      activation_kind);
  block_linear_items_activation(
      actor,
      hidden0,
      out,
      count,
      hidden_dim,
      out_dim,
      extra_mlp_weight_index(actor, extra_slot, 1, false),
      extra_mlp_weight_index(actor, extra_slot, 1, true),
      final_activate,
      activation_kind);
}

__device__ void block_linear_items(
    const ActorPackedAbi& actor,
    const float* in,
    float* out,
    int count,
    int in_dim,
    int out_dim,
    int weight_idx,
    int bias_idx,
    bool relu) {
  const float* weight = actor.w[weight_idx];
  const float* bias = actor.w[bias_idx];
  for (int idx = threadIdx.x; idx < count * out_dim; idx += blockDim.x) {
    const int item = idx / out_dim;
    const int o = idx - item * out_dim;
    float acc = bias[o];
    const float* row_w = weight + o * in_dim;
    const float* row_in = in + item * in_dim;
    for (int i = 0; i < in_dim; ++i) acc += row_w[i] * row_in[i];
    out[idx] = relu ? relu_device(acc) : acc;
  }
  __syncthreads();
}

__device__ void block_layer_norm_items_or_copy(
    const ActorPackedAbi& actor,
    const float* in,
    float* out,
    int count,
    int dim,
    int weight_idx,
    int bias_idx,
    float* reduce_scratch) {
  if (!has_w(actor, weight_idx) || !has_w(actor, bias_idx)) {
    for (int idx = threadIdx.x; idx < count * dim; idx += blockDim.x) out[idx] = in[idx];
    __syncthreads();
    return;
  }
  const float* weight = actor.w[weight_idx];
  const float* bias = actor.w[bias_idx];
  for (int item = 0; item < count; ++item) {
    float partial = 0.0f;
    const float* row_in = in + item * dim;
    float* row_out = out + item * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) partial += row_in[d];
    reduce_scratch[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
      __syncthreads();
    }
    const float mean = reduce_scratch[0] / fmaxf(static_cast<float>(dim), 1.0f);
    partial = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      const float delta = row_in[d] - mean;
      partial += delta * delta;
    }
    reduce_scratch[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
      __syncthreads();
    }
    const float inv_std = rsqrtf(reduce_scratch[0] / fmaxf(static_cast<float>(dim), 1.0f) + 1.0e-5f);
    for (int d = threadIdx.x; d < dim; d += blockDim.x) row_out[d] = (row_in[d] - mean) * inv_std * weight[d] + bias[d];
    __syncthreads();
  }
}

__device__ void block_masked_simplex_from_logits(
    const float* logits,
    const bool* mask,
    int count,
    float* probs,
    float* reduce_scratch) {
  const float logsum = block_logsumexp(logits, mask, count, reduce_scratch);
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    probs[i] = (mask == nullptr || mask[i]) ? expf(logits[i] - logsum) : 0.0f;
  }
  __syncthreads();
}

__device__ void block_masked_simplex_from_alr_loc(
    const float* loc,
    const bool* mask,
    int count,
    float* logits_tmp,
    float* probs,
    float* reduce_scratch) {
  int ref = -1;
  for (int i = 0; i < count; ++i) {
    if (mask == nullptr || mask[i]) ref = i;
  }
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    logits_tmp[i] = (mask == nullptr || mask[i]) ? (i == ref ? 0.0f : loc[i]) : 0.0f;
  }
  __syncthreads();
  block_masked_simplex_from_logits(logits_tmp, mask, count, probs, reduce_scratch);
}

__device__ float masked_simplex_log_prob(
    const float* loc,
    const float* log_scale,
    const bool* mask,
    const float* action,
    int count) {
  int ref = -1;
  int valid_count = 0;
  float sum = 0.0f;
  for (int i = 0; i < count; ++i) {
    if (mask != nullptr && !mask[i]) continue;
    ref = i;
    ++valid_count;
    sum += fmaxf(action[i], 0.0f);
  }
  if (valid_count <= 1 || ref < 0) return 0.0f;
  sum = fmaxf(sum, 1.0e-8f);
  const float ref_prob = fmaxf(action[ref] / sum, 1.0e-8f);
  float normal_lp = 0.0f;
  float log_det = 0.0f;
  for (int i = 0; i < count; ++i) {
    if (mask != nullptr && !mask[i]) continue;
    const float p = fmaxf(action[i] / sum, 1.0e-8f);
    log_det += logf(p);
    if (i == ref) continue;
    const float latent = logf(p) - logf(ref_prob);
    const float inv_scale = expf(-log_scale[i]);
    const float z = (latent - loc[i]) * inv_scale;
    normal_lp += -0.5f * (z * z + kLog2Pi) - log_scale[i];
  }
  return normal_lp - log_det;
}

__device__ float block_masked_simplex_log_prob(
    const float* loc,
    const float* log_scale,
    const bool* mask,
    const float* action,
    int count,
    float* reduce_scratch) {
  float local_ref = -1.0f;
  float local_valid_count = 0.0f;
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    if (mask != nullptr && !mask[i]) continue;
    local_ref = fmaxf(local_ref, static_cast<float>(i));
    local_valid_count += 1.0f;
    local_sum += fmaxf(action[i], 0.0f);
  }
  reduce_scratch[threadIdx.x] = local_ref;
  reduce_scratch[blockDim.x + threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_scratch[threadIdx.x] = fmaxf(reduce_scratch[threadIdx.x], reduce_scratch[threadIdx.x + stride]);
      reduce_scratch[blockDim.x + threadIdx.x] += reduce_scratch[blockDim.x + threadIdx.x + stride];
    }
    __syncthreads();
  }
  const int ref = static_cast<int>(reduce_scratch[0]);
  const float sum = fmaxf(reduce_scratch[blockDim.x], 1.0e-8f);

  reduce_scratch[threadIdx.x] = local_valid_count;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
    __syncthreads();
  }
  const float valid_count = reduce_scratch[0];
  if (valid_count <= 1.0f || ref < 0) return 0.0f;

  const float ref_prob = fmaxf(action[ref] / sum, 1.0e-8f);
  float local_normal_lp = 0.0f;
  float local_log_det = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    if (mask != nullptr && !mask[i]) continue;
    const float p = fmaxf(action[i] / sum, 1.0e-8f);
    local_log_det += logf(p);
    if (i == ref) continue;
    const float latent = logf(p) - logf(ref_prob);
    const float inv_scale = expf(-log_scale[i]);
    const float z = (latent - loc[i]) * inv_scale;
    local_normal_lp += -0.5f * (z * z + kLog2Pi) - log_scale[i];
  }
  reduce_scratch[threadIdx.x] = local_normal_lp;
  reduce_scratch[blockDim.x + threadIdx.x] = local_log_det;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
      reduce_scratch[blockDim.x + threadIdx.x] += reduce_scratch[blockDim.x + threadIdx.x + stride];
    }
    __syncthreads();
  }
  return reduce_scratch[0] - reduce_scratch[blockDim.x];
}

__device__ float beta_log_prob_device(float x, float alpha, float beta);

__device__ __noinline__ float stickbreaking_log_prob(
    const float* mean,
    float kappa,
    const bool* mask,
    const float* action,
    int count) {
  int last = -1;
  int valid_count = 0;
  for (int i = 0; i < count; ++i) {
    if (mask != nullptr && !mask[i]) continue;
    last = i;
    ++valid_count;
  }
  if (valid_count <= 1 || last < 0) return 0.0f;
  float remaining_mean = 1.0f;
  float remaining_action = 1.0f;
  float lp = 0.0f;
  for (int i = 0; i < count; ++i) {
    if (mask != nullptr && !mask[i]) continue;
    const float mean_i = fmaxf(mean[i], 0.0f);
    const float action_i = fmaxf(action[i], 0.0f);
    if (i == last) break;
    const float factor_mean = fminf(fmaxf(mean_i / fmaxf(remaining_mean, 1.0e-6f), 1.0e-6f), 1.0f - 1.0e-6f);
    const float factor_action = fminf(fmaxf(action_i / fmaxf(remaining_action, 1.0e-6f), 1.0e-6f), 1.0f - 1.0e-6f);
    const float alpha = fmaxf(factor_mean * kappa, 1.0e-6f);
    const float beta = fmaxf((1.0f - factor_mean) * kappa, 1.0e-6f);
    lp += beta_log_prob_device(factor_action, alpha, beta);
    remaining_mean = fmaxf(remaining_mean - mean_i, 1.0e-6f);
    remaining_action = fmaxf(remaining_action - action_i, 1.0e-6f);
  }
  return lp;
}

__device__ int block_count_mask(const bool* mask, int count, float* reduce_scratch) {
  float local = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    if (mask == nullptr || mask[i]) local += 1.0f;
  }
  reduce_scratch[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
    __syncthreads();
  }
  return static_cast<int>(reduce_scratch[0] + 0.5f);
}

__device__ int block_last_mask_index(const bool* mask, int count, float* reduce_scratch) {
  float local = -1.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    if (mask == nullptr || mask[i]) local = fmaxf(local, static_cast<float>(i));
  }
  reduce_scratch[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_scratch[threadIdx.x] = fmaxf(reduce_scratch[threadIdx.x], reduce_scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  return static_cast<int>(reduce_scratch[0]);
}

__device__ __forceinline__ float dirichlet_mean_valid_device(float mean) {
  return fmaxf(mean, 1.0e-8f);
}

__device__ __forceinline__ float dirichlet_alpha_from_mean_device(float mean, float kappa, float mean_sum) {
  const float normalized_mean = dirichlet_mean_valid_device(mean) / fmaxf(mean_sum, 1.0e-8f);
  return fmaxf(normalized_mean * fmaxf(kappa, 1.0e-8f), 1.0e-8f);
}

__device__ __forceinline__ float dirichlet_alpha_legacy_device(float mean, float kappa) {
  return fmaxf(mean * kappa, 1.0e-8f);
}

__device__ __noinline__ float block_dirichlet_log_prob_fast(
    const float* mean,
    float kappa,
    const bool* mask,
    const float* action,
    int count,
    float* reduce_scratch,
    bool legacy_alpha) {
  float local_mean_sum0 = 0.0f;
  float local_count0 = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    if (mask != nullptr && !mask[i]) continue;
    local_count0 += 1.0f;
    local_mean_sum0 += dirichlet_mean_valid_device(mean[i]);
  }
  reduce_scratch[threadIdx.x] = local_mean_sum0;
  reduce_scratch[blockDim.x + threadIdx.x] = local_count0;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
      reduce_scratch[blockDim.x + threadIdx.x] += reduce_scratch[blockDim.x + threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean_sum = fmaxf(reduce_scratch[0], 1.0e-8f);
  const int valid_count0 = static_cast<int>(reduce_scratch[blockDim.x] + 0.5f);
  if (valid_count0 < 2) return 0.0f;

  float local_alpha0 = 0.0f;
  float local_log_gamma_sum = 0.0f;
  float local_action_sum = 0.0f;
  float local_count = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    if (mask != nullptr && !mask[i]) continue;
    local_count += 1.0f;
    const float alpha = legacy_alpha
        ? dirichlet_alpha_legacy_device(mean[i], kappa)
        : dirichlet_alpha_from_mean_device(mean[i], kappa, mean_sum);
    local_alpha0 += alpha;
    local_log_gamma_sum += lgammaf(alpha);
    local_action_sum += fmaxf(action[i], 0.0f);
  }
  reduce_scratch[threadIdx.x] = local_alpha0;
  reduce_scratch[blockDim.x + threadIdx.x] = local_log_gamma_sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
      reduce_scratch[blockDim.x + threadIdx.x] += reduce_scratch[blockDim.x + threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float alpha0 = reduce_scratch[0];
  const float log_gamma_sum = reduce_scratch[blockDim.x];
  reduce_scratch[threadIdx.x] = local_action_sum;
  reduce_scratch[blockDim.x + threadIdx.x] = local_count;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
      reduce_scratch[blockDim.x + threadIdx.x] += reduce_scratch[blockDim.x + threadIdx.x + stride];
    }
    __syncthreads();
  }
  const int valid_count = static_cast<int>(reduce_scratch[blockDim.x] + 0.5f);
  if (valid_count < 2) return 0.0f;
  const float action_sum = fmaxf(reduce_scratch[0], 1.0e-8f);
  float local_lp = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    if (mask != nullptr && !mask[i]) continue;
    const float alpha = legacy_alpha
        ? dirichlet_alpha_legacy_device(mean[i], kappa)
        : dirichlet_alpha_from_mean_device(mean[i], kappa, mean_sum);
    const float p = fmaxf(action[i] / action_sum, 1.0e-8f);
    local_lp += (alpha - 1.0f) * logf(p);
  }
  reduce_scratch[threadIdx.x] = local_lp;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
    __syncthreads();
  }
  return lgammaf(fmaxf(alpha0, 1.0e-8f)) - log_gamma_sum + reduce_scratch[0];
}

__device__ __noinline__ float block_dirichlet_log_prob_precise(
    const float* mean,
    float kappa,
    const bool* mask,
    const float* action,
    int count,
    float* reduce_scratch) {
  if (threadIdx.x == 0) {
    const double eps = 1.0e-8;
    int valid_count = 0;
    double mean_sum = 0.0;
    double action_sum = 0.0;
    for (int i = 0; i < count; ++i) {
      if (mask != nullptr && !mask[i]) continue;
      ++valid_count;
      mean_sum += fmax(static_cast<double>(mean[i]), eps);
      action_sum += fmax(static_cast<double>(action[i]), eps);
    }
    if (valid_count < 2) {
      reduce_scratch[0] = 0.0f;
    } else {
      mean_sum = fmax(mean_sum, eps);
      action_sum = fmax(action_sum, eps);
      const double kappa_s = fmax(static_cast<double>(kappa), eps);
      double alpha0 = 0.0;
      double log_gamma_sum = 0.0;
      double log_action_term = 0.0;
      for (int i = 0; i < count; ++i) {
        if (mask != nullptr && !mask[i]) continue;
        const double mean_valid = fmax(static_cast<double>(mean[i]), eps);
        const double alpha = fmax((mean_valid / mean_sum) * kappa_s, eps);
        const double action_valid = fmax(static_cast<double>(action[i]), eps);
        const double prob = fmax(action_valid / action_sum, eps);
        alpha0 += alpha;
        log_gamma_sum += lgamma(alpha);
        log_action_term += (alpha - 1.0) * log(prob);
      }
      const double raw = lgamma(fmax(alpha0, eps)) - log_gamma_sum + log_action_term;
      reduce_scratch[0] = static_cast<float>(raw);
    }
  }
  __syncthreads();
  return reduce_scratch[0];
}

__device__ __forceinline__ float digamma_approx(float x) {
  x = fmaxf(x, 1.0e-8f);
  float result = 0.0f;
  while (x < 8.0f) {
    result -= 1.0f / x;
    x += 1.0f;
  }
  const float inv = 1.0f / x;
  const float inv2 = inv * inv;
  result += logf(x) - 0.5f * inv - inv2 * (1.0f / 12.0f - inv2 * (1.0f / 120.0f - inv2 * (1.0f / 252.0f)));
  return result;
}

__device__ __noinline__ float block_dirichlet_entropy_fast(
    const float* mean,
    float kappa,
    const bool* mask,
    int count,
    float* reduce_scratch,
    bool legacy_alpha) {
  float local_mean_sum0 = 0.0f;
  float local_count0 = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    if (mask != nullptr && !mask[i]) continue;
    local_count0 += 1.0f;
    local_mean_sum0 += dirichlet_mean_valid_device(mean[i]);
  }
  reduce_scratch[threadIdx.x] = local_mean_sum0;
  reduce_scratch[blockDim.x + threadIdx.x] = local_count0;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
      reduce_scratch[blockDim.x + threadIdx.x] += reduce_scratch[blockDim.x + threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean_sum = fmaxf(reduce_scratch[0], 1.0e-8f);
  const int valid_count0 = static_cast<int>(reduce_scratch[blockDim.x] + 0.5f);
  if (valid_count0 < 2) return 0.0f;

  float local_alpha0 = 0.0f;
  float local_log_gamma_sum = 0.0f;
  float local_count = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    if (mask != nullptr && !mask[i]) continue;
    local_count += 1.0f;
    const float alpha = legacy_alpha
        ? dirichlet_alpha_legacy_device(mean[i], kappa)
        : dirichlet_alpha_from_mean_device(mean[i], kappa, mean_sum);
    local_alpha0 += alpha;
    local_log_gamma_sum += lgammaf(alpha);
  }
  reduce_scratch[threadIdx.x] = local_alpha0;
  reduce_scratch[blockDim.x + threadIdx.x] = local_log_gamma_sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
      reduce_scratch[blockDim.x + threadIdx.x] += reduce_scratch[blockDim.x + threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float alpha0 = reduce_scratch[0];
  const float log_gamma_sum = reduce_scratch[blockDim.x];
  reduce_scratch[threadIdx.x] = local_count;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
    __syncthreads();
  }
  const int valid_count = static_cast<int>(reduce_scratch[0] + 0.5f);
  if (valid_count < 2) return 0.0f;
  const float psi0 = digamma_approx(alpha0);
  float local_term = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    if (mask != nullptr && !mask[i]) continue;
    const float alpha = legacy_alpha
        ? dirichlet_alpha_legacy_device(mean[i], kappa)
        : dirichlet_alpha_from_mean_device(mean[i], kappa, mean_sum);
    local_term += (alpha - 1.0f) * digamma_approx(alpha);
  }
  reduce_scratch[threadIdx.x] = local_term;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
    __syncthreads();
  }
  return log_gamma_sum - lgammaf(fmaxf(alpha0, 1.0e-8f)) + (alpha0 - static_cast<float>(valid_count)) * psi0 - reduce_scratch[0];
}

__device__ float gamma_sample_mt_ge1(
    float alpha,
    const ActorPackedAbi& actor,
    int64_t rng_step,
    int row,
    int stream) {
  alpha = fmaxf(alpha, 1.0f);
  const float d = alpha - 1.0f / 3.0f;
  const float c = 1.0f / sqrtf(9.0f * d);
  for (int attempt = 0; attempt < 16; ++attempt) {
    const float x = normal01(actor, rng_step, row, stream + attempt * 7);
    const float v0 = 1.0f + c * x;
    if (v0 <= 0.0f) continue;
    const float v = v0 * v0 * v0;
    const float u = uniform01(actor, rng_step, row, stream + attempt * 7 + 3);
    if (u < 1.0f - 0.0331f * x * x * x * x) return d * v;
    if (logf(fmaxf(u, 1.0e-8f)) < 0.5f * x * x + d * (1.0f - v + logf(v))) return d * v;
  }
  return fmaxf(alpha, 1.0e-8f);
}

__device__ float gamma_sample_mt(
    float alpha,
    const ActorPackedAbi& actor,
    int64_t rng_step,
    int row,
    int stream) {
  alpha = fmaxf(alpha, 1.0e-8f);
  if (alpha < 1.0f) {
    const float g = gamma_sample_mt_ge1(alpha + 1.0f, actor, rng_step, row, stream + 17);
    const float u = fmaxf(uniform01(actor, rng_step, row, stream + 31), 1.0e-8f);
    return g * powf(u, 1.0f / alpha);
  }
  return gamma_sample_mt_ge1(alpha, actor, rng_step, row, stream);
}

__device__ void block_masked_mean(
    const float* values,
    const bool* mask,
    int count,
    int dim,
    float* out,
    float* reduce_scratch) {
  for (int d = 0; d < dim; ++d) {
    float partial_sum = 0.0f;
    float partial_count = 0.0f;
    for (int item = threadIdx.x; item < count; item += blockDim.x) {
      if (mask != nullptr && !mask[item]) continue;
      partial_sum += values[item * dim + d];
      partial_count += 1.0f;
    }
    reduce_scratch[threadIdx.x] = partial_sum;
    reduce_scratch[blockDim.x + threadIdx.x] = partial_count;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
        reduce_scratch[blockDim.x + threadIdx.x] += reduce_scratch[blockDim.x + threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) out[d] = reduce_scratch[blockDim.x] > 0.0f ? reduce_scratch[0] / fmaxf(reduce_scratch[blockDim.x], 1.0f) : 0.0f;
    __syncthreads();
  }
}

__device__ void block_masked_max(
    const float* values,
    const bool* mask,
    int count,
    int dim,
    float* out,
    float* reduce_scratch) {
  for (int d = 0; d < dim; ++d) {
    float local = kNegInf;
    bool any = false;
    for (int item = threadIdx.x; item < count; item += blockDim.x) {
      if (mask != nullptr && !mask[item]) continue;
      any = true;
      local = fmaxf(local, values[item * dim + d]);
    }
    reduce_scratch[threadIdx.x] = any ? local : kNegInf;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) reduce_scratch[threadIdx.x] = fmaxf(reduce_scratch[threadIdx.x], reduce_scratch[threadIdx.x + stride]);
      __syncthreads();
    }
    if (threadIdx.x == 0) out[d] = reduce_scratch[0] <= kNegInf * 0.5f ? 0.0f : reduce_scratch[0];
    __syncthreads();
  }
}

__device__ void block_uniform_simplex(const bool* mask, int count, float* out, float* reduce_scratch) {
  float local = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) if (mask == nullptr || mask[i]) local += 1.0f;
  reduce_scratch[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
    __syncthreads();
  }
  const float denom = reduce_scratch[0];
  for (int i = threadIdx.x; i < count; i += blockDim.x) out[i] = (denom > 0.0f && (mask == nullptr || mask[i])) ? 1.0f / denom : 0.0f;
  __syncthreads();
}

__device__ void block_normalize_simplex(const float* in, const bool* mask, int count, float* out, float* reduce_scratch) {
  float local = 0.0f;
  float local_count = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    if (mask != nullptr && !mask[i]) continue;
    local += fmaxf(in[i], 0.0f);
    local_count += 1.0f;
  }
  reduce_scratch[threadIdx.x] = local;
  reduce_scratch[blockDim.x + threadIdx.x] = local_count;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_scratch[threadIdx.x] += reduce_scratch[threadIdx.x + stride];
      reduce_scratch[blockDim.x + threadIdx.x] += reduce_scratch[blockDim.x + threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float denom = reduce_scratch[0];
  const float valid_count = reduce_scratch[blockDim.x];
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const bool valid = mask == nullptr || mask[i];
    if (!valid || valid_count <= 0.0f) out[i] = 0.0f;
    else if (valid_count == 1.0f) out[i] = 1.0f;
    else out[i] = denom > 1.0e-8f ? fmaxf(in[i], 0.0f) / denom : 1.0f / valid_count;
  }
  __syncthreads();
}

__device__ void block_loc_from_probs(const float* probs, const bool* mask, int count, float* loc) {
  int ref = -1;
  int valid_count = 0;
  for (int i = 0; i < count; ++i) {
    if (mask != nullptr && !mask[i]) continue;
    ref = i;
    ++valid_count;
  }
  const float ref_prob = ref >= 0 ? fmaxf(probs[ref], 1.0e-8f) : 1.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const bool latent = (mask == nullptr || mask[i]) && valid_count > 1 && i != ref;
    loc[i] = latent ? logf(fmaxf(probs[i], 1.0e-8f)) - logf(ref_prob) : 0.0f;
  }
  __syncthreads();
}

__device__ void block_topk_mask_by_rank(const float* score, const bool* valid_mask, int count, int k, bool* out) {
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    bool keep = false;
    if ((valid_mask == nullptr || valid_mask[i]) && k > 0) {
      int rank = 0;
      for (int j = 0; j < count; ++j) {
        if (valid_mask != nullptr && !valid_mask[j]) continue;
        if (score[j] > score[i] || (score[j] == score[i] && j < i)) ++rank;
      }
      keep = rank < k;
    }
    out[i] = keep;
  }
  __syncthreads();
}

__device__ float beta_log_prob_device(float x, float alpha, float beta) {
  x = fminf(fmaxf(x, 1.0e-6f), 1.0f - 1.0e-6f);
  alpha = fmaxf(alpha, 1.0e-6f);
  beta = fmaxf(beta, 1.0e-6f);
  return (alpha - 1.0f) * logf(x) + (beta - 1.0f) * logf(1.0f - x)
      + lgammaf(alpha + beta) - lgammaf(alpha) - lgammaf(beta);
}

__device__ void block_competition_layer(
    const ActorPackedAbi& actor,
    float* h,
    float* qkv,
    float* attn_ctx,
    float* projected,
    float* ffn_hidden,
    const bool* valid_mask,
    int count,
    int embed_dim,
    int hidden_dim,
    int heads,
    int weight_base,
    float* score_scratch,
    float* reduce_scratch) {
  if (count <= 0 || embed_dim <= 0 || hidden_dim <= 0) return;
  if (!has_w(actor, weight_base + W_BW_COMP_ATTN_IN_PROJ_WEIGHT) ||
      !has_w(actor, weight_base + W_BW_COMP_ATTN_OUT_PROJ_WEIGHT) ||
      !has_w(actor, weight_base + W_BW_COMP_FFN0_WEIGHT) ||
      !has_w(actor, weight_base + W_BW_COMP_FFN2_WEIGHT)) {
    return;
  }
  heads = max(heads, 1);
  if (embed_dim % heads != 0) heads = 1;
  const int head_dim = max(embed_dim / heads, 1);
  const float inv_sqrt_head = rsqrtf(fmaxf(static_cast<float>(head_dim), 1.0f));
  const float* in_w = actor.w[weight_base + W_BW_COMP_ATTN_IN_PROJ_WEIGHT];
  const float* in_b = actor.w[weight_base + W_BW_COMP_ATTN_IN_PROJ_BIAS];
  for (int idx = threadIdx.x; idx < count * 3 * embed_dim; idx += blockDim.x) {
    const int item = idx / (3 * embed_dim);
    const int qkv_d = idx - item * 3 * embed_dim;
    float acc = in_b[qkv_d];
    const float* row_w = in_w + qkv_d * embed_dim;
    const float* row_h = h + item * embed_dim;
    for (int d = 0; d < embed_dim; ++d) acc += row_w[d] * row_h[d];
    qkv[idx] = (valid_mask == nullptr || valid_mask[item]) ? acc : 0.0f;
  }
  __syncthreads();

  const bool cache_scores = score_scratch != nullptr;
  if (cache_scores) {
    for (int idx = threadIdx.x; idx < count * heads * count; idx += blockDim.x) {
      const int item = idx / (heads * count);
      const int rem = idx - item * heads * count;
      const int head = rem / count;
      const int key_item = rem - head * count;
      float score = kNegInf;
      if ((valid_mask == nullptr || valid_mask[item]) && (valid_mask == nullptr || valid_mask[key_item])) {
        const int head_off = head * head_dim;
        const float* q_ptr = qkv + item * 3 * embed_dim + head_off;
        const float* k_ptr = qkv + key_item * 3 * embed_dim + embed_dim + head_off;
        float raw = 0.0f;
        for (int hd = 0; hd < head_dim; ++hd) raw += q_ptr[hd] * k_ptr[hd];
        score = raw * inv_sqrt_head;
      }
      score_scratch[idx] = score;
    }
    __syncthreads();
  }

  for (int idx = threadIdx.x; idx < count * embed_dim; idx += blockDim.x) {
    const int item = idx / embed_dim;
    const int d = idx - item * embed_dim;
    if (valid_mask != nullptr && !valid_mask[item]) {
      attn_ctx[idx] = 0.0f;
      continue;
    }
    const int head = d / head_dim;
    const int head_off = head * head_dim;
    float max_score = kNegInf;
    for (int key_item = 0; key_item < count; ++key_item) {
      if (valid_mask != nullptr && !valid_mask[key_item]) continue;
      float score = cache_scores ? score_scratch[(item * heads + head) * count + key_item] : 0.0f;
      if (!cache_scores) {
        const float* q_ptr = qkv + item * 3 * embed_dim + head_off;
        const float* k_ptr = qkv + key_item * 3 * embed_dim + embed_dim + head_off;
        for (int hd = 0; hd < head_dim; ++hd) score += q_ptr[hd] * k_ptr[hd];
        score *= inv_sqrt_head;
      }
      max_score = fmaxf(max_score, score);
    }
    if (max_score <= kNegInf * 0.5f) {
      attn_ctx[idx] = 0.0f;
      continue;
    }
    float denom = 0.0f;
    float acc = 0.0f;
    for (int key_item = 0; key_item < count; ++key_item) {
      if (valid_mask != nullptr && !valid_mask[key_item]) continue;
      float score = cache_scores ? score_scratch[(item * heads + head) * count + key_item] : 0.0f;
      if (!cache_scores) {
        const float* q_ptr = qkv + item * 3 * embed_dim + head_off;
        const float* k_ptr = qkv + key_item * 3 * embed_dim + embed_dim + head_off;
        for (int hd = 0; hd < head_dim; ++hd) score += q_ptr[hd] * k_ptr[hd];
        score *= inv_sqrt_head;
      }
      const float w = expf(score - max_score);
      denom += w;
      acc += w * qkv[key_item * 3 * embed_dim + 2 * embed_dim + d];
    }
    attn_ctx[idx] = acc / fmaxf(denom, 1.0e-8f);
  }
  __syncthreads();

  const float* out_w = actor.w[weight_base + W_BW_COMP_ATTN_OUT_PROJ_WEIGHT];
  const float* out_b = actor.w[weight_base + W_BW_COMP_ATTN_OUT_PROJ_BIAS];
  for (int idx = threadIdx.x; idx < count * embed_dim; idx += blockDim.x) {
    const int item = idx / embed_dim;
    const int d = idx - item * embed_dim;
    float acc = out_b[d];
    const float* row_w = out_w + d * embed_dim;
    const float* row_ctx = attn_ctx + item * embed_dim;
    for (int j = 0; j < embed_dim; ++j) acc += row_w[j] * row_ctx[j];
    projected[idx] = (valid_mask == nullptr || valid_mask[item]) ? h[idx] + acc : 0.0f;
  }
  __syncthreads();
  block_layer_norm_items_or_copy(
      actor,
      projected,
      h,
      count,
      embed_dim,
      weight_base + W_BW_COMP_NORM_ATTN_WEIGHT,
      weight_base + W_BW_COMP_NORM_ATTN_BIAS,
      reduce_scratch);
  for (int idx = threadIdx.x; idx < count * embed_dim; idx += blockDim.x) {
    const int item = idx / embed_dim;
    if (valid_mask != nullptr && !valid_mask[item]) h[idx] = 0.0f;
  }
  __syncthreads();

  block_mlp2_items(
      actor,
      h,
      projected,
      ffn_hidden,
      count,
      embed_dim,
      hidden_dim,
      embed_dim,
      weight_base + W_BW_COMP_FFN0_WEIGHT,
      weight_base + W_BW_COMP_FFN0_BIAS,
      weight_base + W_BW_COMP_FFN2_WEIGHT,
      weight_base + W_BW_COMP_FFN2_BIAS,
      false);
  for (int idx = threadIdx.x; idx < count * embed_dim; idx += blockDim.x) {
    const int item = idx / embed_dim;
    projected[idx] = (valid_mask == nullptr || valid_mask[item]) ? h[idx] + projected[idx] : 0.0f;
  }
  __syncthreads();
  block_layer_norm_items_or_copy(
      actor,
      projected,
      h,
      count,
      embed_dim,
      weight_base + W_BW_COMP_NORM_FFN_WEIGHT,
      weight_base + W_BW_COMP_NORM_FFN_BIAS,
      reduce_scratch);
  for (int idx = threadIdx.x; idx < count * embed_dim; idx += blockDim.x) {
    const int item = idx / embed_dim;
    if (valid_mask != nullptr && !valid_mask[item]) h[idx] = 0.0f;
  }
  __syncthreads();
}

__global__ void actor_accel_live_kernel(int64_t active_idx, bool deterministic, int64_t rng_step) {
  const RuntimePackedAbi& runtime = cActorRuntimeAbi;
  const ActorPackedAbi& actor = cActorAbi;
  const int num_envs = static_cast<int>(ip(runtime, kParamNumEnvs));
  const int num_uav = static_cast<int>(ip(runtime, kParamNumUav));
  const int row = blockIdx.x;
  const int row_count = num_envs * num_uav;
  if (row >= row_count || num_uav <= 0) {
    return;
  }
  const int e = row / num_uav;
  const int u = row - e * num_uav;
  const int hidden_dim = static_cast<int>(aip(actor, kActorAccelHidden, aip(actor, kActorHidden)));
  const int embed_dim = static_cast<int>(aip(actor, kActorAccelEmbed, aip(actor, kActorEmbed)));
  const int ego_dim = static_cast<int>(aip(actor, kActorAccelEgoDim, 27));
  const int cell_dim = static_cast<int>(aip(actor, kActorAccelCellDim, 18));
  const int gu_token_dim = static_cast<int>(aip(actor, kActorAccelGuTokenDim, 27));
  const int peer_token_dim = static_cast<int>(aip(actor, kActorAccelPeerTokenDim, 28));
  const int sat_token_dim = static_cast<int>(aip(actor, kActorAccelSatTokenDim, 32));
  const int gu_query_count = static_cast<int>(aip(actor, kActorAccelGuQueryCount, 4));
  const int peer_query_count = static_cast<int>(aip(actor, kActorAccelPeerQueryCount, 2));
  const int sat_query_count = static_cast<int>(aip(actor, kActorAccelSatQueryCount, 2));
  const int encoder_layers = static_cast<int>(aip(actor, kActorAccelEncoderMlpLayers, 2));
  const int context_layers = static_cast<int>(aip(actor, kActorAccelContextMlpLayers, 2));
  const int head_layers = static_cast<int>(aip(actor, kActorAccelHeadMlpLayers, 1));
  const int interaction_layers = static_cast<int>(aip(actor, kActorAccelInteractionLayers, 0));
  const int interaction_heads = static_cast<int>(aip(actor, kActorAccelAttentionHeads, 1));
  const int interaction_base = static_cast<int>(aip(actor, kActorAccelBlockWeightBase, 0));
  const int interaction_stride = static_cast<int>(aip(actor, kActorAccelBlockWeightStride, 12));
  const int gu_count = static_cast<int>(ip(runtime, kParamNumGu));
  const int visible = static_cast<int>(ip(runtime, kParamAccelSatWidth));
  const int peer_count = max(num_uav - 1, 0);
  if (hidden_dim <= 0 || embed_dim <= 0 || kFLiveAccelAction >= runtime.nf || runtime.f[kFLiveAccelAction] == nullptr) {
    return;
  }
  const int live_f = active_idx == 0 ? kFLiveAccelObs0 : kFLiveAccelObs1;
  const int live_b = active_idx == 0 ? kBLiveAccelObs0 : kBLiveAccelObs1;
  int scratch_stride = 0;
  float* scratch_base = actor_row_scratch(runtime, row, &scratch_stride);
  if (scratch_base == nullptr || scratch_stride <= 0) return;
  float* cursor = scratch_base;
  int remaining = scratch_stride;
  float* reduce = scratch_alloc(cursor, remaining, 2 * blockDim.x);
  const int max_count = max(max(gu_count, peer_count), visible);
  const int max_in = max(max(max(ego_dim, cell_dim), max(gu_token_dim, peer_token_dim)), max(sat_token_dim, 2 * embed_dim));
  const int fusion_dim = (
      2 * embed_dim
      + (gu_query_count + 2) * embed_dim
      + (peer_query_count + 2) * embed_dim
      + (sat_query_count + 2) * embed_dim);
  float* ego_norm = scratch_alloc(cursor, remaining, ego_dim);
  float* cell_norm = scratch_alloc(cursor, remaining, cell_dim);
  float* ego_emb = scratch_alloc(cursor, remaining, embed_dim);
  float* cell_emb = scratch_alloc(cursor, remaining, embed_dim);
  float* query_src = scratch_alloc(cursor, remaining, 2 * embed_dim);
  float* gu_queries = scratch_alloc(cursor, remaining, gu_query_count * embed_dim);
  float* peer_queries = scratch_alloc(cursor, remaining, peer_query_count * embed_dim);
  float* sat_queries = scratch_alloc(cursor, remaining, sat_query_count * embed_dim);
  float* gu_attn = scratch_alloc(cursor, remaining, gu_query_count * embed_dim);
  float* peer_attn = scratch_alloc(cursor, remaining, peer_query_count * embed_dim);
  float* sat_attn = scratch_alloc(cursor, remaining, sat_query_count * embed_dim);
  float* gu_mean = scratch_alloc(cursor, remaining, embed_dim);
  float* gu_max = scratch_alloc(cursor, remaining, embed_dim);
  float* peer_mean = scratch_alloc(cursor, remaining, embed_dim);
  float* peer_max = scratch_alloc(cursor, remaining, embed_dim);
  float* sat_mean = scratch_alloc(cursor, remaining, embed_dim);
  float* sat_max = scratch_alloc(cursor, remaining, embed_dim);
  float* fusion_in = scratch_alloc(cursor, remaining, fusion_dim);
  float* fusion_out = scratch_alloc(cursor, remaining, hidden_dim);
  float* mean_out = scratch_alloc(cursor, remaining, 2);
  float* hidden = scratch_alloc(cursor, remaining, max(max_count, 1) * hidden_dim);
  float* hidden2 = scratch_alloc(cursor, remaining, max(max_count, 1) * hidden_dim);
  float* qkv = scratch_alloc(cursor, remaining, max(max_count, 1) * 3 * embed_dim);
  float* attn_ctx = scratch_alloc(cursor, remaining, max(max_count, 1) * embed_dim);
  float* projected = scratch_alloc(cursor, remaining, max(max_count, 1) * embed_dim);
  float* attn_scores = scratch_alloc(
      cursor,
      remaining,
      max(max_count, 1) * max(interaction_heads, 1) * max(max_count, 1));
  float* item_in = scratch_alloc(cursor, remaining, max(max_count, 1) * max_in);
  float* item_norm = scratch_alloc(cursor, remaining, max(max_count, 1) * max_in);
  float* peer_tok = scratch_alloc(cursor, remaining, max(peer_count, 1) * embed_dim);
  float* gu_tok = scratch_alloc(cursor, remaining, max(gu_count, 1) * embed_dim);
  float* sat_tok = scratch_alloc(cursor, remaining, max(visible, 1) * embed_dim);
  if (reduce == nullptr || sat_tok == nullptr) return;

  const float* ego_src = runtime.f[live_f + 0] + static_cast<int64_t>(row) * ego_dim;
  const float* cell_src = runtime.f[live_f + 1] + static_cast<int64_t>(row) * cell_dim;
  block_layer_norm_or_copy(actor, ego_src, ego_norm, ego_dim, W_ACCEL_EGO_NORM_WEIGHT, W_ACCEL_EGO_NORM_BIAS, reduce);
  block_mlp_flex(actor, ego_norm, ego_emb, hidden, hidden2, ego_dim, hidden_dim, embed_dim, W_ACCEL_EGO_ENC0_WEIGHT, W_ACCEL_EGO_ENC0_BIAS, W_ACCEL_EGO_ENC2_WEIGHT, W_ACCEL_EGO_ENC2_BIAS, X_ACCEL_EGO_ENCODER, encoder_layers, false, 0);
  block_layer_norm_or_copy(actor, cell_src, cell_norm, cell_dim, W_ACCEL_CELL_NORM_WEIGHT, W_ACCEL_CELL_NORM_BIAS, reduce);
  block_mlp_flex(actor, cell_norm, cell_emb, hidden, hidden2, cell_dim, hidden_dim, embed_dim, W_ACCEL_CELL_ENC0_WEIGHT, W_ACCEL_CELL_ENC0_BIAS, W_ACCEL_CELL_ENC2_WEIGHT, W_ACCEL_CELL_ENC2_BIAS, X_ACCEL_CELL_ENCODER, encoder_layers, false, 0);

  const float* gu_tokens = runtime.f[live_f + 2];
  const bool* gu_mask = runtime.b[live_b + 0] + static_cast<int64_t>(row) * gu_count;
  const float* peer_tokens = runtime.f[live_f + 3];
  const bool* peer_mask = runtime.b[live_b + 1] + static_cast<int64_t>(row) * peer_count;
  const float* sat_tokens = runtime.f[live_f + 4];
  const bool* sat_mask = runtime.b[live_b + 2] + static_cast<int64_t>(row) * visible;

  const float* gu_src = gu_tokens + static_cast<int64_t>(row) * gu_count * gu_token_dim;
  block_layer_norm_items_or_copy(actor, gu_src, item_norm, gu_count, gu_token_dim, W_ACCEL_GU_NORM_WEIGHT, W_ACCEL_GU_NORM_BIAS, reduce);
  block_mlp_items_flex(actor, item_norm, gu_tok, hidden, hidden2, gu_count, gu_token_dim, hidden_dim, embed_dim, W_ACCEL_GU_ENC0_WEIGHT, W_ACCEL_GU_ENC0_BIAS, W_ACCEL_GU_ENC2_WEIGHT, W_ACCEL_GU_ENC2_BIAS, X_ACCEL_GU_ENCODER, encoder_layers, false, 0);

  const float* peer_src = peer_tokens + static_cast<int64_t>(row) * peer_count * peer_token_dim;
  block_layer_norm_items_or_copy(actor, peer_src, item_norm, peer_count, peer_token_dim, W_ACCEL_PEER_NORM_WEIGHT, W_ACCEL_PEER_NORM_BIAS, reduce);
  block_mlp_items_flex(actor, item_norm, peer_tok, hidden, hidden2, peer_count, peer_token_dim, hidden_dim, embed_dim, W_ACCEL_PEER_ENC0_WEIGHT, W_ACCEL_PEER_ENC0_BIAS, W_ACCEL_PEER_ENC2_WEIGHT, W_ACCEL_PEER_ENC2_BIAS, X_ACCEL_PEER_ENCODER, encoder_layers, false, 0);

  const float* sat_src = sat_tokens + static_cast<int64_t>(row) * visible * sat_token_dim;
  block_layer_norm_items_or_copy(actor, sat_src, item_norm, visible, sat_token_dim, W_ACCEL_SAT_NORM_WEIGHT, W_ACCEL_SAT_NORM_BIAS, reduce);
  block_mlp_items_flex(actor, item_norm, sat_tok, hidden, hidden2, visible, sat_token_dim, hidden_dim, embed_dim, W_ACCEL_SAT_ENC0_WEIGHT, W_ACCEL_SAT_ENC0_BIAS, W_ACCEL_SAT_ENC2_WEIGHT, W_ACCEL_SAT_ENC2_BIAS, X_ACCEL_SAT_ENCODER, encoder_layers, false, 0);

  for (int layer = 0; layer < interaction_layers; ++layer) {
    const int base = interaction_base + layer * interaction_stride;
    block_competition_layer(actor, gu_tok, qkv, attn_ctx, projected, hidden, gu_mask, gu_count, embed_dim, hidden_dim, interaction_heads, base, attn_scores, reduce);
    block_competition_layer(actor, peer_tok, qkv, attn_ctx, projected, hidden, peer_mask, peer_count, embed_dim, hidden_dim, interaction_heads, base, attn_scores, reduce);
    block_competition_layer(actor, sat_tok, qkv, attn_ctx, projected, hidden, sat_mask, visible, embed_dim, hidden_dim, interaction_heads, base, attn_scores, reduce);
  }

  for (int d = threadIdx.x; d < 2 * embed_dim; d += blockDim.x) {
    query_src[d] = d < embed_dim ? ego_emb[d] : cell_emb[d - embed_dim];
  }
  __syncthreads();
  block_linear(actor, query_src, gu_queries, 2 * embed_dim, gu_query_count * embed_dim, W_ACCEL_GU_QUERY_WEIGHT, W_ACCEL_GU_QUERY_BIAS, false);
  block_linear(actor, query_src, peer_queries, 2 * embed_dim, peer_query_count * embed_dim, W_ACCEL_PEER_QUERY_WEIGHT, W_ACCEL_PEER_QUERY_BIAS, false);
  block_linear(actor, query_src, sat_queries, 2 * embed_dim, sat_query_count * embed_dim, W_ACCEL_SAT_QUERY_WEIGHT, W_ACCEL_SAT_QUERY_BIAS, false);

  block_multi_query_attention(gu_queries, gu_query_count, gu_tok, gu_mask, gu_count, embed_dim, gu_attn, reduce);
  block_masked_mean(gu_tok, gu_mask, gu_count, embed_dim, gu_mean, reduce);
  block_masked_max(gu_tok, gu_mask, gu_count, embed_dim, gu_max, reduce);
  block_multi_query_attention(peer_queries, peer_query_count, peer_tok, peer_mask, peer_count, embed_dim, peer_attn, reduce);
  block_masked_mean(peer_tok, peer_mask, peer_count, embed_dim, peer_mean, reduce);
  block_masked_max(peer_tok, peer_mask, peer_count, embed_dim, peer_max, reduce);
  block_multi_query_attention(sat_queries, sat_query_count, sat_tok, sat_mask, visible, embed_dim, sat_attn, reduce);
  block_masked_mean(sat_tok, sat_mask, visible, embed_dim, sat_mean, reduce);
  block_masked_max(sat_tok, sat_mask, visible, embed_dim, sat_max, reduce);

  for (int idx = threadIdx.x; idx < fusion_dim; idx += blockDim.x) fusion_in[idx] = 0.0f;
  __syncthreads();
  int off = 0;
  for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) fusion_in[off + d] = ego_emb[d];
  off += embed_dim;
  for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) fusion_in[off + d] = cell_emb[d];
  off += embed_dim;
  for (int d = threadIdx.x; d < gu_query_count * embed_dim; d += blockDim.x) fusion_in[off + d] = gu_attn[d];
  off += gu_query_count * embed_dim;
  for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) fusion_in[off + d] = gu_mean[d];
  off += embed_dim;
  for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) fusion_in[off + d] = gu_max[d];
  off += embed_dim;
  for (int d = threadIdx.x; d < peer_query_count * embed_dim; d += blockDim.x) fusion_in[off + d] = peer_attn[d];
  off += peer_query_count * embed_dim;
  for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) fusion_in[off + d] = peer_mean[d];
  off += embed_dim;
  for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) fusion_in[off + d] = peer_max[d];
  off += embed_dim;
  for (int d = threadIdx.x; d < sat_query_count * embed_dim; d += blockDim.x) fusion_in[off + d] = sat_attn[d];
  off += sat_query_count * embed_dim;
  for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) fusion_in[off + d] = sat_mean[d];
  off += embed_dim;
  for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) fusion_in[off + d] = sat_max[d];
  __syncthreads();

  block_mlp_flex(actor, fusion_in, fusion_out, hidden, hidden2, fusion_dim, hidden_dim, hidden_dim, W_ACCEL_FUSION0_WEIGHT, W_ACCEL_FUSION0_BIAS, W_ACCEL_FUSION2_WEIGHT, W_ACCEL_FUSION2_BIAS, X_ACCEL_FUSION, context_layers, true, 0);
  if (head_layers <= 1) {
    block_linear(actor, fusion_out, mean_out, hidden_dim, 2, W_ACCEL_MU_WEIGHT, W_ACCEL_MU_BIAS, false);
  } else {
    const int mu_head_base = static_cast<int>(aip(actor, kActorAccelMuHeadMlpWeightBase, 0));
    block_mlp_flex_contiguous(actor, fusion_out, mean_out, hidden, hidden2, hidden_dim, hidden_dim, 2, mu_head_base, head_layers, false, 0);
  }
  if (threadIdx.x == 0) {
    const float mean0 = mean_out[0];
    const float mean1 = mean_out[1];
    const float raw_log_std0 = has_w(actor, W_ACCEL_LOG_STD) ? actor.w[W_ACCEL_LOG_STD][0] : 0.0f;
    const float raw_log_std1 = has_w(actor, W_ACCEL_LOG_STD) ? actor.w[W_ACCEL_LOG_STD][1] : 0.0f;
    const float log_std0 = fminf(fmaxf(raw_log_std0, -5.0f), 2.0f);
    const float log_std1 = fminf(fmaxf(raw_log_std1, -5.0f), 2.0f);
    const float z0 = deterministic ? mean0 : mean0 + expf(log_std0) * normal01(actor, rng_step, row, 1);
    const float z1 = deterministic ? mean1 : mean1 + expf(log_std1) * normal01(actor, rng_step, row, 2);
    const float radius = sqrtf(z0 * z0 + z1 * z1);
    const float squashed_radius = tanhf(radius);
    const float scale = fmaxf(afp(actor, kActorAccelActionScale, 1.0f), 1.0e-8f);
    const float coeff = radius > 1.0e-8f ? scale * squashed_radius / radius : scale;
    runtime.f[kFLiveAccelAction][row * 2 + 0] = z0 * coeff;
    runtime.f[kFLiveAccelAction][row * 2 + 1] = z1 * coeff;
    if (kFLiveAccelLatentAction < runtime.nf && runtime.f[kFLiveAccelLatentAction] != nullptr) {
      runtime.f[kFLiveAccelLatentAction][row * 2 + 0] = z0;
      runtime.f[kFLiveAccelLatentAction][row * 2 + 1] = z1;
    }
    const float lp_z = normal_logprob_device(mean0, log_std0, z0) + normal_logprob_device(mean1, log_std1, z1);
    reduce[0] = lp_z;
    reduce[1] = 0.0f;
  }
  __syncthreads();
  if (threadIdx.x == 0 && kFLiveAccelOldLogprob < runtime.nf && runtime.f[kFLiveAccelOldLogprob] != nullptr) {
    runtime.f[kFLiveAccelOldLogprob][row] = reduce[0] + reduce[1];
  }
}

__global__ void actor_sat_live_kernel(bool deterministic, int64_t rng_step) {
  const RuntimePackedAbi& runtime = cActorRuntimeAbi;
  const ActorPackedAbi& actor = cActorAbi;
  const int num_envs = static_cast<int>(ip(runtime, kParamNumEnvs));
  const int num_uav = static_cast<int>(ip(runtime, kParamNumUav));
  const int row = blockIdx.x;
  const int row_count = num_envs * num_uav;
  if (row >= row_count || num_uav <= 0) {
    return;
  }
  const int e = row / num_uav;
  const int u = row - e * num_uav;
  (void)u;
  if (kLLiveSatSubsetIndex >= runtime.nl || runtime.l[kLLiveSatSubsetIndex] == nullptr) {
    return;
  }
  const int hidden_dim = static_cast<int>(aip(actor, kActorSatHidden, aip(actor, kActorHidden)));
  const int embed_dim = static_cast<int>(aip(actor, kActorSatEmbed, aip(actor, kActorEmbed)));
  const int visible = static_cast<int>(ip(runtime, kParamSatVisibleWidth));
  const int subset_count = static_cast<int>(ip(runtime, kParamSubsetCount));
  const int select_k = static_cast<int>(ip(runtime, kParamSatNumSelect));
  const int encoder_layers = static_cast<int>(aip(actor, kActorSatEncoderMlpLayers, 2));
  const int context_layers = static_cast<int>(aip(actor, kActorSatContextMlpLayers, 2));
  const int head_layers = static_cast<int>(aip(actor, kActorSatHeadMlpLayers, 2));
  const int heads = max(static_cast<int>(aip(actor, kActorSatAttentionHeads, 1)), 1);
  const int usable_heads = (embed_dim % heads == 0) ? heads : 1;
  const int head_dim = embed_dim / usable_heads;
  const int layer_count = max(static_cast<int>(aip(actor, kActorSatCompetitionLayers, 1)), 1);
  const int block_base = static_cast<int>(aip(actor, kActorSatBlockWeightBase, 0));
  const int block_stride = max(static_cast<int>(aip(actor, kActorSatBlockWeightStride, 12)), 12);
  constexpr int kSatEgoDim = 13;
  constexpr int kSatDemandDim = 8;
  constexpr int kSatRoleDim = 1;
  constexpr int kSatTokenDim = 26;
  int scratch_stride = 0;
  float* scratch_base = actor_row_scratch(runtime, row, &scratch_stride);
  if (scratch_base == nullptr || scratch_stride <= 0 || hidden_dim <= 0 || embed_dim <= 0 ||
      visible < 0 || visible > kMaxItems || subset_count <= 0 || subset_count > kMaxSubset ||
      select_k <= 0 || select_k > kMaxSelect) return;
  float* cursor = scratch_base;
  int remaining = scratch_stride;
  float* reduce = scratch_alloc(cursor, remaining, blockDim.x);
  const int max_count = max(max(visible, subset_count), 1);
  const int max_in = max(max(kSatTokenDim, 3 * embed_dim), 2 * embed_dim);
  float* ego_norm = scratch_alloc(cursor, remaining, kSatEgoDim);
  float* demand_norm = scratch_alloc(cursor, remaining, kSatDemandDim);
  float* ego_emb = scratch_alloc(cursor, remaining, embed_dim);
  float* demand_emb = scratch_alloc(cursor, remaining, embed_dim);
  float* role_emb = scratch_alloc(cursor, remaining, embed_dim);
  float* ctx_in = scratch_alloc(cursor, remaining, 3 * embed_dim);
  float* ctx0 = scratch_alloc(cursor, remaining, embed_dim);
  float* hidden = scratch_alloc(cursor, remaining, max_count * hidden_dim);
  float* hidden2 = scratch_alloc(cursor, remaining, max_count * hidden_dim);
  float* item_in = scratch_alloc(cursor, remaining, max_count * max_in);
  float* item_norm = scratch_alloc(cursor, remaining, max_count * max_in);
  float* sat_h = scratch_alloc(cursor, remaining, max(visible, 1) * embed_dim);
  float* sat_aux = scratch_alloc(cursor, remaining, max(visible, 1) * embed_dim);
  float* qkv = scratch_alloc(cursor, remaining, max(visible, 1) * 3 * embed_dim);
  float* attn_scores = scratch_alloc(cursor, remaining, max(visible, 1) * usable_heads * max(visible, 1));
  float* item_logits = scratch_alloc(cursor, remaining, max(visible, 1));
  float* count_logits = scratch_alloc(cursor, remaining, max(select_k + 1, 1));
  float* logits = scratch_alloc(cursor, remaining, max(subset_count, 1));
  if (logits == nullptr || count_logits == nullptr) return;

  __shared__ bool valid_s[kMaxItems];
  __shared__ bool subset_mask_s[kMaxSubset];
  __shared__ int valid_count_s;
  __shared__ int legal_count_s;

  const float* ego_src = runtime.f[kFLiveSatObs + 0] + static_cast<int64_t>(row) * kSatEgoDim;
  const float* demand_src = runtime.f[kFLiveSatObs + 1] + static_cast<int64_t>(row) * kSatDemandDim;
  const float* role_src = runtime.f[kFLiveSatObs + 2] + static_cast<int64_t>(row) * kSatRoleDim;
  const float* sat_src = runtime.f[kFLiveSatObs + 3] + static_cast<int64_t>(row) * visible * kSatTokenDim;
  const bool* sat_mask = runtime.b[kBLiveSatObs + 0] + static_cast<int64_t>(row) * visible;
  const bool* sat_valid = runtime.b[kBLiveSatObs + 1] + static_cast<int64_t>(row) * visible;
  if (kLMainSatSubsetMembersBase >= runtime.nl || runtime.l[kLMainSatSubsetMembersBase] == nullptr) {
    return;
  }
  const int64_t* members = runtime.l[kLMainSatSubsetMembersBase];

  int local_valid_count = 0;
  for (int s = threadIdx.x; s < visible; s += blockDim.x) {
    const bool valid = sat_mask[s] && sat_valid[s];
    valid_s[s] = valid;
    if (valid) ++local_valid_count;
  }
  reduce[threadIdx.x] = static_cast<float>(local_valid_count);
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) valid_count_s = static_cast<int>(reduce[0] + 0.5f);
  __syncthreads();

  block_layer_norm_or_copy(actor, ego_src, ego_norm, kSatEgoDim, W_SAT_EGO_NORM_WEIGHT, W_SAT_EGO_NORM_BIAS, reduce);
  block_mlp_flex(actor, ego_norm, ego_emb, hidden, hidden2, kSatEgoDim, hidden_dim, embed_dim, W_SAT_EGO_ENC0_WEIGHT, W_SAT_EGO_ENC0_BIAS, W_SAT_EGO_ENC2_WEIGHT, W_SAT_EGO_ENC2_BIAS, X_SAT_EGO_ENCODER, encoder_layers, false, 0);
  block_layer_norm_or_copy(actor, demand_src, demand_norm, kSatDemandDim, W_SAT_SUBSET_NORM_WEIGHT, W_SAT_SUBSET_NORM_BIAS, reduce);
  block_mlp_flex(actor, demand_norm, demand_emb, hidden, hidden2, kSatDemandDim, hidden_dim, embed_dim, W_SAT_Q1_0_WEIGHT, W_SAT_Q1_0_BIAS, W_SAT_Q1_2_WEIGHT, W_SAT_Q1_2_BIAS, X_SAT_DEMAND_ENCODER, encoder_layers, false, 0);
  block_mlp_flex(actor, role_src, role_emb, hidden, hidden2, kSatRoleDim, hidden_dim, embed_dim, W_SAT_Q2_0_WEIGHT, W_SAT_Q2_0_BIAS, W_SAT_Q2_2_WEIGHT, W_SAT_Q2_2_BIAS, X_SAT_ROLE_ENCODER, encoder_layers, false, 0);
  for (int d = threadIdx.x; d < 3 * embed_dim; d += blockDim.x) {
    ctx_in[d] = d < embed_dim ? ego_emb[d] : (d < 2 * embed_dim ? demand_emb[d - embed_dim] : role_emb[d - 2 * embed_dim]);
  }
  __syncthreads();
  block_mlp_flex(actor, ctx_in, ctx0, hidden, hidden2, 3 * embed_dim, hidden_dim, embed_dim, W_SAT_REF0_WEIGHT, W_SAT_REF0_BIAS, W_SAT_REF2_WEIGHT, W_SAT_REF2_BIAS, X_SAT_CTX_ENCODER, context_layers, false, 0);

  block_layer_norm_items_or_copy(actor, sat_src, item_norm, visible, kSatTokenDim, W_SAT_INPUT_NORM_WEIGHT, W_SAT_INPUT_NORM_BIAS, reduce);
  block_mlp_items_flex(actor, item_norm, sat_aux, hidden, hidden2, visible, kSatTokenDim, hidden_dim, embed_dim, W_SAT_ENC0_WEIGHT, W_SAT_ENC0_BIAS, W_SAT_ENC2_WEIGHT, W_SAT_ENC2_BIAS, X_SAT_SAT_ENCODER, encoder_layers, false, 0);
  for (int idx = threadIdx.x; idx < visible * (2 * embed_dim); idx += blockDim.x) {
    const int s = idx / (2 * embed_dim);
    const int d = idx - s * (2 * embed_dim);
    item_in[idx] = d < embed_dim ? sat_aux[s * embed_dim + d] : ctx0[d - embed_dim];
  }
  __syncthreads();
  block_mlp_items_flex(actor, item_in, sat_h, hidden, hidden2, visible, 2 * embed_dim, hidden_dim, embed_dim, W_SAT_EGO_FUSION0_WEIGHT, W_SAT_EGO_FUSION0_BIAS, W_SAT_EGO_FUSION2_WEIGHT, W_SAT_EGO_FUSION2_BIAS, X_SAT_CONTEXT_FUSION, context_layers, false, 0);
  for (int idx = threadIdx.x; idx < visible * embed_dim; idx += blockDim.x) {
    const int s = idx / embed_dim;
    if (!valid_s[s]) sat_h[idx] = 0.0f;
  }
  __syncthreads();

  for (int layer = 0; layer < layer_count; ++layer) {
    const int base = block_base + layer * block_stride;
    block_layer_norm_items_or_copy(actor, sat_h, item_norm, visible, embed_dim, base + 0, base + 1, reduce);
    block_linear_items(actor, item_norm, qkv, visible, embed_dim, 3 * embed_dim, base + 2, base + 3, false);
    if (attn_scores != nullptr) {
      const float inv_sqrt_head = rsqrtf(fmaxf(static_cast<float>(head_dim), 1.0f));
      for (int idx = threadIdx.x; idx < visible * usable_heads * visible; idx += blockDim.x) {
        const int s = idx / (usable_heads * visible);
        const int rem = idx - s * usable_heads * visible;
        const int h = rem / visible;
        const int j = rem - h * visible;
        float score = kNegInf;
        if (valid_s[s] && valid_s[j]) {
          float raw = 0.0f;
          for (int t = 0; t < head_dim; ++t) {
            const float q = qkv[s * 3 * embed_dim + h * head_dim + t];
            const float k = qkv[j * 3 * embed_dim + embed_dim + h * head_dim + t];
            raw += q * k;
          }
          score = raw * inv_sqrt_head;
        }
        attn_scores[idx] = score;
      }
      __syncthreads();
    }
    for (int idx = threadIdx.x; idx < visible * embed_dim; idx += blockDim.x) {
      const int s = idx / embed_dim;
      const int d = idx - s * embed_dim;
      if (!valid_s[s]) {
        sat_aux[idx] = 0.0f;
        continue;
      }
      const int h = d / head_dim;
      const int hd = d - h * head_dim;
      float max_score = kNegInf;
      for (int j = 0; j < visible; ++j) {
        if (!valid_s[j]) continue;
        float score = attn_scores != nullptr ? attn_scores[(s * usable_heads + h) * visible + j] : 0.0f;
        if (attn_scores == nullptr) {
          for (int t = 0; t < head_dim; ++t) {
            const float q = qkv[s * 3 * embed_dim + h * head_dim + t];
            const float k = qkv[j * 3 * embed_dim + embed_dim + h * head_dim + t];
            score += q * k;
          }
          score *= rsqrtf(fmaxf(static_cast<float>(head_dim), 1.0f));
        }
        max_score = fmaxf(max_score, score);
      }
      float denom = 0.0f;
      float value = 0.0f;
      for (int j = 0; j < visible; ++j) {
        if (!valid_s[j]) continue;
        float score = attn_scores != nullptr ? attn_scores[(s * usable_heads + h) * visible + j] : 0.0f;
        if (attn_scores == nullptr) {
          for (int t = 0; t < head_dim; ++t) {
            const float q = qkv[s * 3 * embed_dim + h * head_dim + t];
            const float k = qkv[j * 3 * embed_dim + embed_dim + h * head_dim + t];
            score += q * k;
          }
          score *= rsqrtf(fmaxf(static_cast<float>(head_dim), 1.0f));
        }
        const float w = expf(score - max_score);
        denom += w;
        value += w * qkv[j * 3 * embed_dim + 2 * embed_dim + h * head_dim + hd];
      }
      sat_aux[idx] = value / fmaxf(denom, 1.0e-8f);
    }
    __syncthreads();
    block_linear_items(actor, sat_aux, item_norm, visible, embed_dim, embed_dim, base + 4, base + 5, false);
    for (int idx = threadIdx.x; idx < visible * embed_dim; idx += blockDim.x) {
      const int s = idx / embed_dim;
      sat_h[idx] = valid_s[s] ? sat_h[idx] + item_norm[idx] : 0.0f;
    }
    __syncthreads();
    block_layer_norm_items_or_copy(actor, sat_h, item_norm, visible, embed_dim, base + 6, base + 7, reduce);
    block_mlp2_items(actor, item_norm, sat_aux, hidden, visible, embed_dim, hidden_dim, embed_dim, base + 8, base + 9, base + 10, base + 11, false);
    for (int idx = threadIdx.x; idx < visible * embed_dim; idx += blockDim.x) {
      const int s = idx / embed_dim;
      sat_h[idx] = valid_s[s] ? sat_h[idx] + sat_aux[idx] : 0.0f;
    }
    __syncthreads();
  }

  block_mlp_items_flex(actor, sat_h, item_logits, hidden, hidden2, visible, embed_dim, hidden_dim, 1, W_SAT_SUBSET_ENC0_WEIGHT, W_SAT_SUBSET_ENC0_BIAS, W_SAT_SUBSET_ENC2_WEIGHT, W_SAT_SUBSET_ENC2_BIAS, X_SAT_LOGIT_HEAD, head_layers, false, 0);
  block_mlp_flex(actor, ctx0, count_logits, hidden, hidden2, embed_dim, hidden_dim, select_k + 1, W_SAT_PROJECT0_WEIGHT, W_SAT_PROJECT0_BIAS, W_SAT_PROJECT2_WEIGHT, W_SAT_PROJECT2_BIAS, X_SAT_COUNT_HEAD, head_layers, false, 0);

  int local_legal = 0;
  for (int subset = threadIdx.x; subset < subset_count; subset += blockDim.x) {
    int size = 0;
    bool legal = true;
    float score = 0.0f;
    for (int k = 0; k < select_k; ++k) {
      const int slot = static_cast<int>(members[subset * select_k + k]);
      if (slot < 0) continue;
      ++size;
      if (slot >= visible || !valid_s[slot]) {
        legal = false;
      } else {
        score += item_logits[slot];
      }
    }
    if (valid_count_s <= 0) {
      legal = legal && size == 0 && subset == 0;
    } else {
      legal = legal && size > 0 && size <= select_k && size <= valid_count_s;
    }
    subset_mask_s[subset] = legal;
    logits[subset] = legal ? score + count_logits[min(size, select_k)] : kNegInf;
    if (legal) ++local_legal;
  }
  reduce[threadIdx.x] = static_cast<float>(local_legal);
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) legal_count_s = static_cast<int>(reduce[0] + 0.5f);
  __syncthreads();

  const bool any_active = legal_count_s > 0;
  float logsum = 0.0f;
  float entropy = 0.0f;
  int chosen = -1;
  if (any_active) {
    logsum = block_logsumexp(logits, subset_mask_s, subset_count, reduce);
    float local_entropy = 0.0f;
    if (legal_count_s > 1) {
      for (int subset = threadIdx.x; subset < subset_count; subset += blockDim.x) {
        if (!subset_mask_s[subset]) continue;
        const float p = expf(logits[subset] - logsum);
        local_entropy += p * (logsum - logits[subset]);
      }
    }
    reduce[threadIdx.x] = local_entropy;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
      __syncthreads();
    }
    entropy = reduce[0];
    if (threadIdx.x == 0) {
      chosen = deterministic
          ? block_argmax_masked(logits, subset_mask_s, subset_count)
          : sample_categorical_masked(logits, subset_mask_s, subset_count, logsum, actor, rng_step, row, 19);
      runtime.l[kLLiveSatSubsetIndex][row] = static_cast<int64_t>(chosen);
      if (kLLiveSatActionIndices < runtime.nl && runtime.l[kLLiveSatActionIndices] != nullptr) {
        for (int k = 0; k < select_k; ++k) {
          int64_t sid = -1;
          const int slot = (chosen >= 0 && chosen < subset_count) ? static_cast<int>(members[chosen * select_k + k]) : -1;
          if (slot >= 0 && slot < visible &&
              kLLiveSatCandidateIds < runtime.nl && runtime.l[kLLiveSatCandidateIds] != nullptr) {
            sid = runtime.l[kLLiveSatCandidateIds][static_cast<int64_t>(row) * visible + slot];
          }
          runtime.l[kLLiveSatActionIndices][static_cast<int64_t>(row) * select_k + k] = sid;
        }
      }
      if (kFLiveSatOldLogprobPerAgent < runtime.nf && runtime.f[kFLiveSatOldLogprobPerAgent] != nullptr) {
        runtime.f[kFLiveSatOldLogprobPerAgent][row] = (chosen >= 0 && legal_count_s > 1) ? logits[chosen] - logsum : 0.0f;
      }
      if (kFLiveSatEntropyPerAgent < runtime.nf && runtime.f[kFLiveSatEntropyPerAgent] != nullptr) {
        runtime.f[kFLiveSatEntropyPerAgent][row] = legal_count_s > 1 ? entropy : 0.0f;
      }
    }
  } else if (threadIdx.x == 0) {
    runtime.l[kLLiveSatSubsetIndex][row] = -1;
    if (kLLiveSatActionIndices < runtime.nl && runtime.l[kLLiveSatActionIndices] != nullptr) {
      for (int k = 0; k < select_k; ++k) {
        runtime.l[kLLiveSatActionIndices][static_cast<int64_t>(row) * select_k + k] = -1;
      }
    }
    if (kFLiveSatOldLogprobPerAgent < runtime.nf && runtime.f[kFLiveSatOldLogprobPerAgent] != nullptr) {
      runtime.f[kFLiveSatOldLogprobPerAgent][row] = 0.0f;
    }
    if (kFLiveSatEntropyPerAgent < runtime.nf && runtime.f[kFLiveSatEntropyPerAgent] != nullptr) {
      runtime.f[kFLiveSatEntropyPerAgent][row] = 0.0f;
    }
  }
}

__global__ void actor_sat_select_from_logits_kernel(
    const float* item_logits,
    const float* count_logits,
    bool deterministic,
    int64_t rng_step) {
  const RuntimePackedAbi& runtime = cActorRuntimeAbi;
  const ActorPackedAbi& actor = cActorAbi;
  const int num_envs = static_cast<int>(ip(runtime, kParamNumEnvs));
  const int num_uav = static_cast<int>(ip(runtime, kParamNumUav));
  const int row = blockIdx.x;
  const int row_count = num_envs * num_uav;
  if (row >= row_count || num_uav <= 0 || item_logits == nullptr || count_logits == nullptr) return;
  const int visible = static_cast<int>(ip(runtime, kParamSatVisibleWidth));
  const int subset_count = static_cast<int>(ip(runtime, kParamSubsetCount));
  const int select_k = static_cast<int>(ip(runtime, kParamSatNumSelect));
  if (visible < 0 || visible > kMaxItems || subset_count <= 0 || subset_count > kMaxSubset ||
      select_k <= 0 || select_k > kMaxSelect ||
      kLLiveSatSubsetIndex >= runtime.nl || runtime.l[kLLiveSatSubsetIndex] == nullptr ||
      kLMainSatSubsetMembersBase >= runtime.nl || runtime.l[kLMainSatSubsetMembersBase] == nullptr ||
      kBLiveSatObs + 1 >= runtime.nb || runtime.b[kBLiveSatObs] == nullptr || runtime.b[kBLiveSatObs + 1] == nullptr) {
    return;
  }

  extern __shared__ unsigned char shared_raw[];
  float* reduce = reinterpret_cast<float*>(shared_raw);
  bool* valid_s = reinterpret_cast<bool*>(reduce + blockDim.x);
  bool* subset_mask_s = valid_s + kMaxItems;
  float* logits_s = reinterpret_cast<float*>(subset_mask_s + kMaxSubset);

  const bool* sat_mask = runtime.b[kBLiveSatObs + 0] + static_cast<int64_t>(row) * visible;
  const bool* sat_valid = runtime.b[kBLiveSatObs + 1] + static_cast<int64_t>(row) * visible;
  const int64_t* members = runtime.l[kLMainSatSubsetMembersBase];
  const float* item_row = item_logits + static_cast<int64_t>(row) * visible;
  const float* count_row = count_logits + static_cast<int64_t>(row) * (select_k + 1);

  int local_valid_count = 0;
  for (int s = threadIdx.x; s < visible; s += blockDim.x) {
    const bool valid = sat_mask[s] && sat_valid[s];
    valid_s[s] = valid;
    if (valid) ++local_valid_count;
  }
  reduce[threadIdx.x] = static_cast<float>(local_valid_count);
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    __syncthreads();
  }
  const int valid_count = static_cast<int>(reduce[0] + 0.5f);

  int local_legal = 0;
  for (int subset = threadIdx.x; subset < subset_count; subset += blockDim.x) {
    int size = 0;
    bool legal = true;
    float score = 0.0f;
    for (int k = 0; k < select_k; ++k) {
      const int slot = static_cast<int>(members[subset * select_k + k]);
      if (slot < 0) continue;
      ++size;
      if (slot >= visible || !valid_s[slot]) {
        legal = false;
      } else {
        score += item_row[slot];
      }
    }
    if (valid_count <= 0) {
      legal = legal && size == 0 && subset == 0;
    } else {
      legal = legal && size > 0 && size <= select_k && size <= valid_count;
    }
    subset_mask_s[subset] = legal;
    logits_s[subset] = legal ? score + count_row[min(size, select_k)] : kNegInf;
    if (legal) ++local_legal;
  }
  reduce[threadIdx.x] = static_cast<float>(local_legal);
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    __syncthreads();
  }
  const int legal_count = static_cast<int>(reduce[0] + 0.5f);

  if (legal_count <= 0) {
    if (threadIdx.x == 0) {
      runtime.l[kLLiveSatSubsetIndex][row] = -1;
      if (kLLiveSatActionIndices < runtime.nl && runtime.l[kLLiveSatActionIndices] != nullptr) {
        for (int k = 0; k < select_k; ++k) {
          runtime.l[kLLiveSatActionIndices][static_cast<int64_t>(row) * select_k + k] = -1;
        }
      }
      if (kFLiveSatOldLogprobPerAgent < runtime.nf && runtime.f[kFLiveSatOldLogprobPerAgent] != nullptr) {
        runtime.f[kFLiveSatOldLogprobPerAgent][row] = 0.0f;
      }
      if (kFLiveSatEntropyPerAgent < runtime.nf && runtime.f[kFLiveSatEntropyPerAgent] != nullptr) {
        runtime.f[kFLiveSatEntropyPerAgent][row] = 0.0f;
      }
    }
    return;
  }

  const float logsum = block_logsumexp(logits_s, subset_mask_s, subset_count, reduce);
  float local_entropy = 0.0f;
  if (legal_count > 1) {
    for (int subset = threadIdx.x; subset < subset_count; subset += blockDim.x) {
      if (!subset_mask_s[subset]) continue;
      const float p = expf(logits_s[subset] - logsum);
      local_entropy += p * (logsum - logits_s[subset]);
    }
  }
  reduce[threadIdx.x] = local_entropy;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    __syncthreads();
  }
  const float entropy = reduce[0];

  if (threadIdx.x == 0) {
    const int chosen = deterministic
        ? block_argmax_masked(logits_s, subset_mask_s, subset_count)
        : sample_categorical_masked(logits_s, subset_mask_s, subset_count, logsum, actor, rng_step, row, 19);
    runtime.l[kLLiveSatSubsetIndex][row] = static_cast<int64_t>(chosen);
    if (kLLiveSatActionIndices < runtime.nl && runtime.l[kLLiveSatActionIndices] != nullptr) {
      for (int k = 0; k < select_k; ++k) {
        int64_t sid = -1;
        const int slot = (chosen >= 0 && chosen < subset_count) ? static_cast<int>(members[chosen * select_k + k]) : -1;
        if (slot >= 0 && slot < visible &&
            kLLiveSatCandidateIds < runtime.nl && runtime.l[kLLiveSatCandidateIds] != nullptr) {
          sid = runtime.l[kLLiveSatCandidateIds][static_cast<int64_t>(row) * visible + slot];
        }
        runtime.l[kLLiveSatActionIndices][static_cast<int64_t>(row) * select_k + k] = sid;
      }
    }
    if (kFLiveSatOldLogprobPerAgent < runtime.nf && runtime.f[kFLiveSatOldLogprobPerAgent] != nullptr) {
      runtime.f[kFLiveSatOldLogprobPerAgent][row] = (chosen >= 0 && legal_count > 1) ? logits_s[chosen] - logsum : 0.0f;
    }
    if (kFLiveSatEntropyPerAgent < runtime.nf && runtime.f[kFLiveSatEntropyPerAgent] != nullptr) {
      runtime.f[kFLiveSatEntropyPerAgent][row] = legal_count > 1 ? entropy : 0.0f;
    }
  }
}

__global__ void actor_accel_write_from_mean_kernel(
    const float* mean,
    bool deterministic,
    int64_t rng_step) {
  const RuntimePackedAbi& runtime = cActorRuntimeAbi;
  const ActorPackedAbi& actor = cActorAbi;
  const int num_envs = static_cast<int>(ip(runtime, kParamNumEnvs));
  const int num_uav = static_cast<int>(ip(runtime, kParamNumUav));
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_count = num_envs * num_uav;
  if (row >= row_count || mean == nullptr || kFLiveAccelAction >= runtime.nf || runtime.f[kFLiveAccelAction] == nullptr) return;
  const float mean0 = mean[row * 2 + 0];
  const float mean1 = mean[row * 2 + 1];
  const float raw_log_std0 = has_w(actor, W_ACCEL_LOG_STD) ? actor.w[W_ACCEL_LOG_STD][0] : 0.0f;
  const float raw_log_std1 = has_w(actor, W_ACCEL_LOG_STD) ? actor.w[W_ACCEL_LOG_STD][1] : 0.0f;
  const float log_std0 = fminf(fmaxf(raw_log_std0, -5.0f), 2.0f);
  const float log_std1 = fminf(fmaxf(raw_log_std1, -5.0f), 2.0f);
  const float z0 = deterministic ? mean0 : mean0 + expf(log_std0) * normal01(actor, rng_step, row, 1);
  const float z1 = deterministic ? mean1 : mean1 + expf(log_std1) * normal01(actor, rng_step, row, 2);
  const float radius = sqrtf(z0 * z0 + z1 * z1);
  const float squashed_radius = tanhf(radius);
  const float scale = fmaxf(afp(actor, kActorAccelActionScale, 1.0f), 1.0e-8f);
  const float coeff = radius > 1.0e-8f ? scale * squashed_radius / radius : scale;
  runtime.f[kFLiveAccelAction][row * 2 + 0] = z0 * coeff;
  runtime.f[kFLiveAccelAction][row * 2 + 1] = z1 * coeff;
  if (kFLiveAccelLatentAction < runtime.nf && runtime.f[kFLiveAccelLatentAction] != nullptr) {
    runtime.f[kFLiveAccelLatentAction][row * 2 + 0] = z0;
    runtime.f[kFLiveAccelLatentAction][row * 2 + 1] = z1;
  }
  if (kFLiveAccelOldLogprob < runtime.nf && runtime.f[kFLiveAccelOldLogprob] != nullptr) {
    const float lp_z = normal_logprob_device(mean0, log_std0, z0) + normal_logprob_device(mean1, log_std1, z1);
    runtime.f[kFLiveAccelOldLogprob][row] = lp_z;
  }
}

__device__ bool actor_restore_bw_macro_row_if_needed(
    const RuntimePackedAbi& runtime,
    int row,
    int history_slot,
    int num_uav,
    int num_gu);

__global__ void actor_bw_write_from_params_kernel(
    const float* det_mean_in,
    const float* kappa_in,
    const float* tau_in,
    const int64_t* row_indices,
    int64_t write_count64,
    bool deterministic,
    int64_t rng_step,
    int64_t history_slot64) {
  const RuntimePackedAbi& runtime = cActorRuntimeAbi;
  const ActorPackedAbi& actor = cActorAbi;
  const int num_envs = static_cast<int>(ip(runtime, kParamNumEnvs));
  const int num_uav = static_cast<int>(ip(runtime, kParamNumUav));
  const int num_gu = static_cast<int>(ip(runtime, kParamNumGu));
  const int local_row = blockIdx.x;
  const int write_count = static_cast<int>(write_count64);
  if (local_row >= write_count) return;
  const int row = row_indices != nullptr ? static_cast<int>(row_indices[local_row]) : local_row;
  const int row_count = num_envs * num_uav;
  if (row >= row_count || num_gu <= 0 || num_gu > kMaxItems || det_mean_in == nullptr || kappa_in == nullptr || tau_in == nullptr ||
      kFLiveBwAction >= runtime.nf || runtime.f[kFLiveBwAction] == nullptr ||
      kBLiveBwObs + 2 >= runtime.nb || runtime.b[kBLiveBwObs + 1] == nullptr || runtime.b[kBLiveBwObs + 2] == nullptr) {
    return;
  }
  if (actor_restore_bw_macro_row_if_needed(runtime, row, static_cast<int>(history_slot64), num_uav, num_gu)) {
    return;
  }
  extern __shared__ unsigned char shared_raw[];
  float* reduce = reinterpret_cast<float*>(shared_raw);
  bool* valid_s = reinterpret_cast<bool*>(reduce + 2 * blockDim.x);
  float* det_mean = reinterpret_cast<float*>(valid_s + kMaxItems);
  float* action = det_mean + kMaxItems;

  const int e = row / num_uav;
  const bool* gu_mask = runtime.b[kBLiveBwObs + 1] + static_cast<int64_t>(row) * num_gu;
  const bool* bw_valid_mask = runtime.b[kBLiveBwObs + 2] + static_cast<int64_t>(row) * num_gu;
  int local_count = 0;
  for (int g = threadIdx.x; g < num_gu; g += blockDim.x) {
    const bool valid = gu_mask[g] && bw_valid_mask[g];
    valid_s[g] = valid;
    det_mean[g] = valid ? det_mean_in[static_cast<int64_t>(local_row) * num_gu + g] : 0.0f;
    action[g] = 0.0f;
    if (valid) ++local_count;
  }
  reduce[threadIdx.x] = static_cast<float>(local_count);
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    __syncthreads();
  }
  const int valid_count = static_cast<int>(reduce[0] + 0.5f);
  if (valid_count <= 0) {
    for (int g = threadIdx.x; g < num_gu; g += blockDim.x) {
      const int idx = row * num_gu + g;
      runtime.f[kFLiveBwAction][idx] = 0.0f;
      if (kFLiveBwRefAction < runtime.nf && runtime.f[kFLiveBwRefAction] != nullptr) runtime.f[kFLiveBwRefAction][idx] = 0.0f;
      if (kFLiveBwFlowProxyOverrideAction < runtime.nf && runtime.f[kFLiveBwFlowProxyOverrideAction] != nullptr) runtime.f[kFLiveBwFlowProxyOverrideAction][idx] = 0.0f;
    }
    if (threadIdx.x == 0) {
      if (has_f(runtime, kFLiveBwOldLogprobPerAgent)) runtime.f[kFLiveBwOldLogprobPerAgent][row] = 0.0f;
      if (has_f(runtime, kFLiveBwEntropyPerAgent)) runtime.f[kFLiveBwEntropyPerAgent][row] = 0.0f;
      if (has_f(runtime, kFLiveBwLogprobRawPerAgent)) runtime.f[kFLiveBwLogprobRawPerAgent][row] = 0.0f;
      if (has_f(runtime, kFLiveBwEntropyRawPerAgent)) runtime.f[kFLiveBwEntropyRawPerAgent][row] = 0.0f;
      if (has_f(runtime, kFLiveBwTau)) runtime.f[kFLiveBwTau][row] = 0.0f;
      if (has_f(runtime, kFLiveBwKappa)) runtime.f[kFLiveBwKappa][row] = 0.0f;
      if (has_l(runtime, kLLiveBwValidCount)) runtime.l[kLLiveBwValidCount][row] = 0;
      if (has_l(runtime, kLLiveBwLatentCount)) runtime.l[kLLiveBwLatentCount][row] = 0;
    }
    return;
  }

  const float kappa = kappa_in[local_row];
  const float tau = tau_in[local_row];
  const int dirichlet_mode = static_cast<int>(aip(actor, kActorBwDirichletDiagnosticMode, 0));
  const bool legacy_alpha = dirichlet_mode == 2;
  const bool precise_logprob = dirichlet_mode == 0;
  if (deterministic || valid_count <= 1) {
    for (int g = threadIdx.x; g < num_gu; g += blockDim.x) action[g] = det_mean[g];
    __syncthreads();
  } else {
    float local_mean_sum = 0.0f;
    for (int g = threadIdx.x; g < num_gu; g += blockDim.x) {
      if (valid_s[g]) local_mean_sum += dirichlet_mean_valid_device(det_mean[g]);
    }
    reduce[threadIdx.x] = local_mean_sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
      __syncthreads();
    }
    const float mean_sum = fmaxf(reduce[0], 1.0e-8f);

    float local_sum = 0.0f;
    for (int g = threadIdx.x; g < num_gu; g += blockDim.x) {
      const float alpha = valid_s[g]
          ? (legacy_alpha ? dirichlet_alpha_legacy_device(det_mean[g], kappa) : dirichlet_alpha_from_mean_device(det_mean[g], kappa, mean_sum))
          : 0.0f;
      const float sample = valid_s[g] ? gamma_sample_mt(alpha, actor, rng_step, row, g + 17) : 0.0f;
      action[g] = sample;
      local_sum += sample;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
      __syncthreads();
    }
    const float denom = reduce[0];
    for (int g = threadIdx.x; g < num_gu; g += blockDim.x) action[g] = denom > 1.0e-8f && valid_s[g] ? action[g] / denom : 0.0f;
    __syncthreads();
  }

  const float raw_logprob = precise_logprob
      ? block_dirichlet_log_prob_precise(det_mean, kappa, valid_s, action, num_gu, reduce)
      : block_dirichlet_log_prob_fast(det_mean, kappa, valid_s, action, num_gu, reduce, legacy_alpha);
  const float entropy_raw = block_dirichlet_entropy_fast(det_mean, kappa, valid_s, num_gu, reduce, legacy_alpha);
  const float latent_denom = fmaxf(static_cast<float>(valid_count - 1), 1.0f);
  const float logprob = raw_logprob / latent_denom;
  const float entropy = entropy_raw / latent_denom;
  for (int g = threadIdx.x; g < num_gu; g += blockDim.x) {
    const int idx = row * num_gu + g;
    const float value = valid_s[g] ? action[g] : 0.0f;
    const float ref_value = valid_s[g] ? det_mean[g] : 0.0f;
    runtime.f[kFLiveBwAction][idx] = value;
    if (kFLiveBwRefAction < runtime.nf && runtime.f[kFLiveBwRefAction] != nullptr) runtime.f[kFLiveBwRefAction][idx] = ref_value;
    if (kFLiveBwFlowProxyOverrideAction < runtime.nf && runtime.f[kFLiveBwFlowProxyOverrideAction] != nullptr) runtime.f[kFLiveBwFlowProxyOverrideAction][idx] = ref_value;
  }
  if (threadIdx.x == 0) {
    if (has_f(runtime, kFLiveBwOldLogprob)) atomicAdd(runtime.f[kFLiveBwOldLogprob] + e, logprob);
    if (has_f(runtime, kFLiveBwOldLogprobPerAgent)) runtime.f[kFLiveBwOldLogprobPerAgent][row] = logprob;
    if (has_f(runtime, kFLiveBwEntropyPerAgent)) runtime.f[kFLiveBwEntropyPerAgent][row] = entropy;
    if (has_f(runtime, kFLiveBwLogprobRawPerAgent)) runtime.f[kFLiveBwLogprobRawPerAgent][row] = raw_logprob;
    if (has_f(runtime, kFLiveBwEntropyRawPerAgent)) runtime.f[kFLiveBwEntropyRawPerAgent][row] = entropy_raw;
    if (has_f(runtime, kFLiveBwTau)) runtime.f[kFLiveBwTau][row] = tau;
    if (has_f(runtime, kFLiveBwKappa)) runtime.f[kFLiveBwKappa][row] = kappa;
    if (has_l(runtime, kLLiveBwValidCount)) runtime.l[kLLiveBwValidCount][row] = static_cast<int64_t>(valid_count);
    if (has_l(runtime, kLLiveBwLatentCount)) runtime.l[kLLiveBwLatentCount][row] = static_cast<int64_t>(max(valid_count - 1, 0));
  }
}

__host__ bool host_has_tensor(const TensorVec& tensors, int idx) {
  return idx >= 0 && idx < static_cast<int>(tensors.size()) && tensors[static_cast<size_t>(idx)].defined() &&
      tensors[static_cast<size_t>(idx)].numel() > 0;
}

__host__ at::Tensor host_weight(const TensorVec& weights, int idx) {
  if (!host_has_tensor(weights, idx)) {
    throw std::runtime_error("native fused SAT actor missing required weight tensor.");
  }
  return weights[static_cast<size_t>(idx)];
}

__host__ int host_aip(const std::vector<int64_t>& params, int idx, int fallback = 0) {
  return idx >= 0 && idx < static_cast<int>(params.size()) ? static_cast<int>(params[static_cast<size_t>(idx)]) : fallback;
}

__host__ int host_extra_mlp_weight_index(const std::vector<int64_t>& actor_int_params, int slot, int extra_pair, bool bias) {
  const int base = host_aip(actor_int_params, kActorExtraMlpWeightBase, 0);
  const int stride = host_aip(actor_int_params, kActorExtraMlpWeightStride, 4);
  return base + slot * stride + extra_pair * 2 + (bias ? 1 : 0);
}

__host__ at::Tensor host_linear(const TensorVec& weights, const at::Tensor& input, int w_idx, int b_idx) {
  return at::linear(input, host_weight(weights, w_idx), host_weight(weights, b_idx));
}

__host__ at::Tensor host_apply_activation_if(const at::Tensor& value, bool enabled, int activation_kind) {
  if (!enabled) return value;
  return activation_kind == 1 ? at::silu(value) : at::relu(value);
}

__host__ at::Tensor host_layer_norm_or_copy(const TensorVec& weights, const at::Tensor& input, int w_idx, int b_idx) {
  if (!host_has_tensor(weights, w_idx) || !host_has_tensor(weights, b_idx)) {
    return input;
  }
  return at::layer_norm(
      input,
      {input.size(-1)},
      host_weight(weights, w_idx),
      host_weight(weights, b_idx),
      1.0e-5,
      false);
}

__host__ at::Tensor host_mlp_flex(
    const TensorVec& weights,
    const std::vector<int64_t>& actor_int_params,
    const at::Tensor& input,
    int w0,
    int b0,
    int w1,
    int b1,
    int extra_slot,
    int layers,
    bool final_activate,
    int activation_kind = 0) {
  layers = std::min(std::max(layers, 1), 4);
  if (layers <= 1) {
    return host_apply_activation_if(host_linear(weights, input, w0, b0), final_activate, activation_kind);
  }
  at::Tensor h = host_apply_activation_if(host_linear(weights, input, w0, b0), true, activation_kind);
  if (layers == 2) {
    return host_apply_activation_if(host_linear(weights, h, w1, b1), final_activate, activation_kind);
  }
  h = host_apply_activation_if(host_linear(weights, h, w1, b1), true, activation_kind);
  if (layers == 3) {
    return host_apply_activation_if(
        host_linear(
            weights,
            h,
            host_extra_mlp_weight_index(actor_int_params, extra_slot, 0, false),
            host_extra_mlp_weight_index(actor_int_params, extra_slot, 0, true)),
        final_activate,
        activation_kind);
  }
  h = host_apply_activation_if(host_linear(
      weights,
      h,
      host_extra_mlp_weight_index(actor_int_params, extra_slot, 0, false),
      host_extra_mlp_weight_index(actor_int_params, extra_slot, 0, true)),
      true,
      activation_kind);
  return host_apply_activation_if(
      host_linear(
          weights,
          h,
          host_extra_mlp_weight_index(actor_int_params, extra_slot, 1, false),
          host_extra_mlp_weight_index(actor_int_params, extra_slot, 1, true)),
      final_activate,
      activation_kind);
}

__host__ at::Tensor host_mlp_flex_contiguous(
    const TensorVec& weights,
    const at::Tensor& input,
    int weight_base,
    int layers,
    bool final_activate,
    int activation_kind = 0) {
  layers = std::min(std::max(layers, 1), 4);
  if (layers <= 1) {
    return host_apply_activation_if(host_linear(weights, input, weight_base + 0, weight_base + 1), final_activate, activation_kind);
  }
  at::Tensor h = host_apply_activation_if(host_linear(weights, input, weight_base + 0, weight_base + 1), true, activation_kind);
  if (layers == 2) {
    return host_apply_activation_if(host_linear(weights, h, weight_base + 2, weight_base + 3), final_activate, activation_kind);
  }
  h = host_apply_activation_if(host_linear(weights, h, weight_base + 2, weight_base + 3), true, activation_kind);
  if (layers == 3) {
    return host_apply_activation_if(host_linear(weights, h, weight_base + 4, weight_base + 5), final_activate, activation_kind);
  }
  h = host_apply_activation_if(host_linear(weights, h, weight_base + 4, weight_base + 5), true, activation_kind);
  return host_apply_activation_if(host_linear(weights, h, weight_base + 6, weight_base + 7), final_activate, activation_kind);
}

__host__ at::Tensor host_bw_tau_kappa_head_packed(
    const TensorVec& weights,
    const std::vector<int64_t>& actor_int_params,
    const at::Tensor& input,
    int layers,
    int activation_kind = 1) {
  if (layers != 2) return at::Tensor();
  const int base = host_aip(actor_int_params, kActorBwTauKappaPackedWeightBase, -1);
  if (base < 0 || !host_has_tensor(weights, base + 0) || !host_has_tensor(weights, base + 1) ||
      !host_has_tensor(weights, base + 2) || !host_has_tensor(weights, base + 3)) {
    return at::Tensor();
  }
  at::Tensor h = host_apply_activation_if(host_linear(weights, input, base + 0, base + 1), true, activation_kind);
  return host_linear(weights, h, base + 2, base + 3);
}

__host__ at::Tensor host_masked_softmax(
    const at::Tensor& scores,
    const at::Tensor& mask,
    int64_t dim,
    const at::Tensor& mask_f_in = at::Tensor()) {
  at::Tensor safe = scores.masked_fill(mask.logical_not(), kNegInf);
  at::Tensor max_scores = std::get<0>(safe.max(dim, true));
  at::Tensor has_valid = mask.any(dim, true);
  max_scores = at::where(has_valid, max_scores, at::zeros_like(max_scores));
  const at::Tensor mask_f = mask_f_in.defined() ? mask_f_in : mask.to(scores.scalar_type());
  at::Tensor exp_scores = at::exp(safe - max_scores) * mask_f;
  at::Tensor denom = exp_scores.sum(dim, true).clamp_min(1.0e-8);
  at::Tensor probs = exp_scores / denom;
  return at::where(has_valid, probs, at::zeros_like(probs));
}

__host__ at::Tensor host_masked_softmax_presafe(
    const at::Tensor& safe_scores,
    const at::Tensor& mask,
    int64_t dim,
    const at::Tensor& mask_f_in = at::Tensor()) {
  at::Tensor max_scores = std::get<0>(safe_scores.max(dim, true));
  at::Tensor has_valid = mask.any(dim, true);
  max_scores = at::where(has_valid, max_scores, at::zeros_like(max_scores));
  const at::Tensor mask_f = mask_f_in.defined() ? mask_f_in : mask.to(safe_scores.scalar_type());
  at::Tensor exp_scores = at::exp(safe_scores - max_scores) * mask_f;
  at::Tensor denom = exp_scores.sum(dim, true).clamp_min(1.0e-8);
  at::Tensor probs = exp_scores / denom;
  return at::where(has_valid, probs, at::zeros_like(probs));
}

__host__ at::Tensor host_multi_query_attention(
    const at::Tensor& queries,
    const at::Tensor& tokens,
    const at::Tensor& mask,
    const at::Tensor& mask_f_in = at::Tensor()) {
  if (tokens.size(-2) == 0) return at::zeros_like(queries);
  const double scale = 1.0 / std::max(std::sqrt(static_cast<double>(tokens.size(-1))), 1.0);
  at::Tensor scores = (queries.unsqueeze(-2) * tokens.unsqueeze(-3)).sum(-1) * scale;
  at::Tensor mask_q = mask.unsqueeze(-2).expand(scores.sizes());
  at::Tensor mask_q_f = mask_f_in.defined() ? mask_f_in.unsqueeze(-2).expand(scores.sizes()) : at::Tensor();
  at::Tensor weights = host_masked_softmax(scores, mask_q, -1, mask_q_f);
  return (weights.unsqueeze(-1) * tokens.unsqueeze(-3)).sum(-2);
}

__host__ at::Tensor host_masked_mean(
    const at::Tensor& tokens,
    const at::Tensor& mask,
    const at::Tensor& mask_f_in = at::Tensor()) {
  if (tokens.size(-2) == 0) return at::zeros({tokens.size(0), tokens.size(-1)}, tokens.options());
  at::Tensor mask_f = mask_f_in.defined() ? mask_f_in.unsqueeze(-1) : mask.to(tokens.scalar_type()).unsqueeze(-1);
  at::Tensor denom = mask_f.sum(-2).clamp_min(1.0);
  return (tokens * mask_f).sum(-2) / denom;
}

__host__ at::Tensor host_masked_sum(
    const at::Tensor& tokens,
    const at::Tensor& mask,
    const at::Tensor& mask_f_in = at::Tensor()) {
  if (tokens.size(-2) == 0) return at::zeros({tokens.size(0), tokens.size(-1)}, tokens.options());
  at::Tensor mask_f = mask_f_in.defined() ? mask_f_in.unsqueeze(-1) : mask.to(tokens.scalar_type()).unsqueeze(-1);
  return (tokens * mask_f).sum(-2);
}

__host__ at::Tensor host_masked_max(const at::Tensor& tokens, const at::Tensor& mask) {
  if (tokens.size(-2) == 0) return at::zeros({tokens.size(0), tokens.size(-1)}, tokens.options());
  at::Tensor safe = tokens.masked_fill(mask.logical_not().unsqueeze(-1), kNegInf);
  at::Tensor out = std::get<0>(safe.max(-2, false));
  return at::where(mask.any(-1, true), out, at::zeros_like(out));
}

__host__ at::Tensor host_mha_residual_block(
    const TensorVec& weights,
    const at::Tensor& h_in,
    const at::Tensor& valid,
    int embed_dim,
    int hidden_dim,
    int heads,
    int weight_base,
    const at::Tensor& valid_f_in = at::Tensor()) {
  if (h_in.size(1) == 0) return h_in;
  heads = std::max(heads, 1);
  if (embed_dim % heads != 0) heads = 1;
  const int head_dim = std::max(embed_dim / heads, 1);
  const int64_t rows = h_in.size(0);
  const int64_t count = h_in.size(1);
  at::Tensor valid_f = valid_f_in.defined() ? valid_f_in : valid.to(h_in.scalar_type()).unsqueeze(-1);
  at::Tensor h = h_in * valid_f;
  at::Tensor qkv = host_linear(weights, h, weight_base + W_BW_COMP_ATTN_IN_PROJ_WEIGHT, weight_base + W_BW_COMP_ATTN_IN_PROJ_BIAS);
  std::vector<at::Tensor> parts = qkv.chunk(3, -1);
  at::Tensor q = parts[0].view({rows, count, heads, head_dim}).transpose(1, 2);
  at::Tensor k = parts[1].view({rows, count, heads, head_dim}).transpose(1, 2);
  at::Tensor v = parts[2].view({rows, count, heads, head_dim}).transpose(1, 2);
  at::Tensor scores = at::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(head_dim));
  at::Tensor has_valid = valid.any(1, true);
  at::Tensor key_valid = at::where(has_valid, valid, at::ones_like(valid));
  scores = scores.masked_fill(key_valid.logical_not().unsqueeze(1).unsqueeze(1), kNegInf);
  at::Tensor weights_attn = at::softmax(scores, -1);
  at::Tensor attn = at::matmul(weights_attn, v).transpose(1, 2).contiguous().view({rows, count, embed_dim});
  at::Tensor projected = host_linear(weights, attn, weight_base + W_BW_COMP_ATTN_OUT_PROJ_WEIGHT, weight_base + W_BW_COMP_ATTN_OUT_PROJ_BIAS);
  h = host_layer_norm_or_copy(weights, h + projected, weight_base + W_BW_COMP_NORM_ATTN_WEIGHT, weight_base + W_BW_COMP_NORM_ATTN_BIAS) * valid_f;
  at::Tensor ffn = host_mlp_flex(
      weights,
      std::vector<int64_t>(),
      h,
      weight_base + W_BW_COMP_FFN0_WEIGHT,
      weight_base + W_BW_COMP_FFN0_BIAS,
      weight_base + W_BW_COMP_FFN2_WEIGHT,
      weight_base + W_BW_COMP_FFN2_BIAS,
      -1,
      2,
      false,
      0);
  return host_layer_norm_or_copy(weights, h + ffn, weight_base + W_BW_COMP_NORM_FFN_WEIGHT, weight_base + W_BW_COMP_NORM_FFN_BIAS) * valid_f;
}

__host__ at::Tensor host_sat_attention_block(
    const TensorVec& weights,
    const std::vector<int64_t>& actor_int_params,
    const at::Tensor& h_in,
    const at::Tensor& valid,
    int layer,
    int embed_dim,
    int hidden_dim,
    int heads,
    int block_base,
    int block_stride) {
  const int usable_heads = (embed_dim % std::max(heads, 1) == 0) ? std::max(heads, 1) : 1;
  const int head_dim = embed_dim / usable_heads;
  const int base = block_base + layer * block_stride;
  at::Tensor valid_f = valid.to(h_in.scalar_type()).unsqueeze(-1);
  at::Tensor h = h_in * valid_f;
  at::Tensor x = host_layer_norm_or_copy(weights, h, base + 0, base + 1);
  at::Tensor qkv = host_linear(weights, x, base + 2, base + 3);
  std::vector<at::Tensor> parts = qkv.chunk(3, -1);
  const int64_t rows = h.size(0);
  const int64_t visible = h.size(1);
  at::Tensor q = parts[0].view({rows, visible, usable_heads, head_dim}).transpose(1, 2);
  at::Tensor k = parts[1].view({rows, visible, usable_heads, head_dim}).transpose(1, 2);
  at::Tensor v = parts[2].view({rows, visible, usable_heads, head_dim}).transpose(1, 2);
  at::Tensor scores = at::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(std::max(head_dim, 1)));
  at::Tensor key_mask = valid.unsqueeze(1).unsqueeze(1);
  scores = scores.masked_fill(key_mask.logical_not(), kNegInf);
  at::Tensor row_has_valid = valid.any(1).view({rows, 1, 1, 1});
  scores = at::where(row_has_valid, scores, at::zeros_like(scores));
  at::Tensor attn_weights = at::softmax(scores, -1);
  attn_weights = attn_weights * key_mask.to(attn_weights.scalar_type());
  at::Tensor denom = attn_weights.sum(-1, true).clamp_min(1.0e-8);
  attn_weights = attn_weights / denom;
  at::Tensor attn = at::matmul(attn_weights, v).transpose(1, 2).contiguous().view({rows, visible, embed_dim});
  at::Tensor projected = host_linear(weights, attn, base + 4, base + 5) * valid_f;
  h = (h + projected) * valid_f;
  at::Tensor ffn_in = host_layer_norm_or_copy(weights, h, base + 6, base + 7);
  at::Tensor ffn = host_mlp_flex(
      weights,
      actor_int_params,
      ffn_in,
      base + 8,
      base + 9,
      base + 10,
      base + 11,
      -1,
      2,
      false);
  return (h + ffn) * valid_f;
}

__device__ int actor_hist_env_row(const RuntimePackedAbi& runtime, int slot, int e) {
  const int num_envs = static_cast<int>(ip(runtime, kParamNumEnvs));
  return slot * max(num_envs, 0) + e;
}

__device__ int actor_bw_macro_start_slot_for_env(const RuntimePackedAbi& runtime, int slot, int e) {
  const int interval = max(static_cast<int>(ip(runtime, kParamAccessBwDecisionInterval, 1)), 1);
  if (interval <= 1 || slot <= 0) return slot;
  int episode_start = 0;
  for (int back = 1; slot - back >= 0; ++back) {
    const int prev_row = actor_hist_env_row(runtime, slot - back, e);
    const bool prev_done =
        (has_b(runtime, kBHistTerminated) && runtime.b[kBHistTerminated][prev_row]) ||
        (has_b(runtime, kBHistTruncated) && runtime.b[kBHistTruncated][prev_row]);
    if (prev_done) {
      episode_start = slot - back + 1;
      break;
    }
  }
  const int age = max(slot - episode_start, 0);
  return slot - (age % interval);
}

__device__ bool actor_restore_bw_macro_row_if_needed(
    const RuntimePackedAbi& runtime,
    int row,
    int history_slot,
    int num_uav,
    int num_gu) {
  const int interval = max(static_cast<int>(ip(runtime, kParamAccessBwDecisionInterval, 1)), 1);
  if (interval <= 1 || history_slot <= 0 || num_uav <= 0 || num_gu <= 0) return false;
  if (!has_f(runtime, kFLiveBwAction)) return false;
  const int e = row / num_uav;
  const int u = row - e * num_uav;
  const int hist_cap = max(static_cast<int>(ip(runtime, kParamHistoryCapacity, 0)), 0);
  const int num_envs = static_cast<int>(ip(runtime, kParamNumEnvs));
  const int hist_rows = hist_cap * max(num_envs, 0);
  const int start_slot = actor_bw_macro_start_slot_for_env(runtime, history_slot, e);
  if (start_slot == history_slot) return false;
  const int src_row = actor_hist_env_row(runtime, start_slot, e);
  if (src_row < 0 || src_row >= hist_rows) return false;

  for (int g = threadIdx.x; g < num_gu; g += blockDim.x) {
    const int live_idx = row * num_gu + g;
    const int hist_idx = (src_row * num_uav + u) * num_gu + g;
    const float value = has_f(runtime, kFHistBwActions) ? runtime.f[kFHistBwActions][hist_idx] : 0.0f;
    runtime.f[kFLiveBwAction][live_idx] = value;
    if (has_f(runtime, kFLiveBwFlowProxyOverrideAction)) runtime.f[kFLiveBwFlowProxyOverrideAction][live_idx] = value;
    if (has_f(runtime, kFLiveBwRefAction)) {
      runtime.f[kFLiveBwRefAction][live_idx] =
          has_f(runtime, kFHistBwRefActions) ? runtime.f[kFHistBwRefActions][hist_idx] : value;
    }
  }
  if (threadIdx.x == 0) {
    const int hist_u = src_row * num_uav + u;
    if (has_f(runtime, kFLiveBwOldLogprobPerAgent)) {
      runtime.f[kFLiveBwOldLogprobPerAgent][row] =
          has_f(runtime, kFHistBwOldLogprobsPerAgent) ? runtime.f[kFHistBwOldLogprobsPerAgent][hist_u] : 0.0f;
    }
    if (has_f(runtime, kFLiveBwEntropyPerAgent)) {
      runtime.f[kFLiveBwEntropyPerAgent][row] =
          has_f(runtime, kFHistBwEntropyPerAgent) ? runtime.f[kFHistBwEntropyPerAgent][hist_u] : 0.0f;
    }
    if (has_f(runtime, kFLiveBwLogprobRawPerAgent)) {
      runtime.f[kFLiveBwLogprobRawPerAgent][row] =
          has_f(runtime, kFHistBwLogprobRawPerAgent) ? runtime.f[kFHistBwLogprobRawPerAgent][hist_u] : 0.0f;
    }
    if (has_f(runtime, kFLiveBwEntropyRawPerAgent)) {
      runtime.f[kFLiveBwEntropyRawPerAgent][row] =
          has_f(runtime, kFHistBwEntropyRawPerAgent) ? runtime.f[kFHistBwEntropyRawPerAgent][hist_u] : 0.0f;
    }
    if (has_f(runtime, kFLiveBwTau)) {
      runtime.f[kFLiveBwTau][row] = has_f(runtime, kFHistBwTau) ? runtime.f[kFHistBwTau][hist_u] : 0.0f;
    }
    if (has_f(runtime, kFLiveBwKappa)) {
      runtime.f[kFLiveBwKappa][row] = has_f(runtime, kFHistBwKappa) ? runtime.f[kFHistBwKappa][hist_u] : 0.0f;
    }
    if (has_l(runtime, kLLiveBwValidCount)) {
      runtime.l[kLLiveBwValidCount][row] =
          has_l(runtime, kLHistBwValidCount) ? runtime.l[kLHistBwValidCount][hist_u] : 0;
    }
    if (has_l(runtime, kLLiveBwLatentCount)) {
      runtime.l[kLLiveBwLatentCount][row] =
          has_l(runtime, kLHistBwLatentCount) ? runtime.l[kLHistBwLatentCount][hist_u] : 0;
    }
    if (u == 0 && has_f(runtime, kFLiveBwOldLogprob)) {
      runtime.f[kFLiveBwOldLogprob][e] =
          has_f(runtime, kFHistBwOldLogprobs) ? runtime.f[kFHistBwOldLogprobs][src_row] : 0.0f;
    }
  }
  return true;
}

__global__ void actor_bw_prepare_macro_start_mask_kernel(
    bool* start_mask,
    int64_t history_slot64,
    int num_uav,
    int num_gu,
    int row_count) {
  const RuntimePackedAbi& runtime = cActorRuntimeAbi;
  const int row = blockIdx.x;
  if (row >= row_count || start_mask == nullptr) return;
  const int interval = max(static_cast<int>(ip(runtime, kParamAccessBwDecisionInterval, 1)), 1);
  if (interval <= 1 || history_slot64 <= 0) {
    if (threadIdx.x == 0) start_mask[row] = true;
    return;
  }
  const int e = row / max(num_uav, 1);
  const int start_slot = actor_bw_macro_start_slot_for_env(runtime, static_cast<int>(history_slot64), e);
  const bool is_start = start_slot == static_cast<int>(history_slot64);
  if (threadIdx.x == 0) start_mask[row] = is_start;
  if (!is_start) {
    // Restore continuation rows before the actor forward so the later fused
    // writer only needs to process true macro-start rows.
    actor_restore_bw_macro_row_if_needed(runtime, row, static_cast<int>(history_slot64), num_uav, num_gu);
  }
}

__global__ void actor_bw_live_kernel(bool deterministic, int64_t rng_step, int64_t history_slot64) {
  const RuntimePackedAbi& runtime = cActorRuntimeAbi;
  const ActorPackedAbi& actor = cActorAbi;
  const int num_envs = static_cast<int>(ip(runtime, kParamNumEnvs));
  const int num_uav = static_cast<int>(ip(runtime, kParamNumUav));
  const int num_gu = static_cast<int>(ip(runtime, kParamNumGu));
  const int select_k = static_cast<int>(ip(runtime, kParamSatNumSelect));
  const int row = blockIdx.x;
  const int row_count = num_envs * num_uav;
  if (row >= row_count || num_uav <= 0 || num_gu <= 0 || num_gu > kMaxItems || select_k <= 0 || select_k > kMaxItems) {
    return;
  }
  const int e = row / num_uav;
  if (kFLiveBwAction >= runtime.nf || runtime.f[kFLiveBwAction] == nullptr) return;
  if (actor_restore_bw_macro_row_if_needed(runtime, row, static_cast<int>(history_slot64), num_uav, num_gu)) {
    return;
  }
  const int hidden_dim = static_cast<int>(aip(actor, kActorBwHidden, aip(actor, kActorHidden)));
  const int embed_dim = static_cast<int>(aip(actor, kActorBwEmbed, aip(actor, kActorEmbed)));
  const int down_queries = max(static_cast<int>(aip(actor, kActorBwDownQueryCount, 1)), 1);
  const int encoder_layers = static_cast<int>(aip(actor, kActorBwEncoderMlpLayers, 2));
  const int context_layers = static_cast<int>(aip(actor, kActorBwContextMlpLayers, 2));
  const int head_layers = static_cast<int>(aip(actor, kActorBwHeadMlpLayers, 2));
  const int competition_layers = static_cast<int>(aip(actor, kActorCompetitionLayers, 0));
  const int competition_heads = static_cast<int>(aip(actor, kActorCompetitionHeads, 1));
  if (hidden_dim <= 0 || embed_dim <= 0 || embed_dim > kMaxEmbed || hidden_dim > kMaxHidden) return;

  int scratch_stride = 0;
  float* scratch_base = actor_row_scratch(runtime, row, &scratch_stride);
  if (scratch_base == nullptr || scratch_stride <= 0) return;
  float* cursor = scratch_base;
  int remaining = scratch_stride;
  const int max_count = max(max(num_gu, select_k), 1);
  const int max_item_dim = max(max(kBwGuTokenDim, kBwSatTokenDim), 2 * embed_dim);
  float* reduce = scratch_alloc(cursor, remaining, 2 * blockDim.x);
  float* ego_norm = scratch_alloc(cursor, remaining, kBwEgoDim);
  float* ego_emb = scratch_alloc(cursor, remaining, embed_dim);
  float* queries = scratch_alloc(cursor, remaining, down_queries * embed_dim);
  float* down_attn = scratch_alloc(cursor, remaining, down_queries * embed_dim);
  float* sat_add_pool = scratch_alloc(cursor, remaining, embed_dim);
  float* down_in = scratch_alloc(cursor, remaining, (down_queries + 1) * embed_dim);
  float* down_ctx = scratch_alloc(cursor, remaining, embed_dim);
  float* ctx_in = scratch_alloc(cursor, remaining, 2 * embed_dim);
  float* ctx0 = scratch_alloc(cursor, remaining, embed_dim);
  float* scalar = scratch_alloc(cursor, remaining, 1);
  float* hidden = scratch_alloc(cursor, remaining, max(max_count, 1) * hidden_dim);
  float* hidden2 = scratch_alloc(cursor, remaining, max(max_count, 1) * hidden_dim);
  float* item_in = scratch_alloc(cursor, remaining, max(max_count, 1) * max_item_dim);
  float* item_norm = scratch_alloc(cursor, remaining, max(max_count, 1) * max_item_dim);
  float* sat_emb = scratch_alloc(cursor, remaining, max(select_k, 1) * embed_dim);
  float* gu_emb = scratch_alloc(cursor, remaining, max(num_gu, 1) * embed_dim);
  float* gu_h = scratch_alloc(cursor, remaining, max(num_gu, 1) * embed_dim);
  float* gu_tmp = scratch_alloc(cursor, remaining, max(num_gu, 1) * embed_dim);
  float* attn_scores = scratch_alloc(cursor, remaining, max_count * max(competition_heads, 1) * max_count);
  float* score = scratch_alloc(cursor, remaining, num_gu);
  float* scaled_score = scratch_alloc(cursor, remaining, num_gu);
  float* det_mean = scratch_alloc(cursor, remaining, num_gu);
  float* action = scratch_alloc(cursor, remaining, num_gu);
  if (
      reduce == nullptr || ego_norm == nullptr || ego_emb == nullptr || queries == nullptr || down_attn == nullptr ||
      sat_add_pool == nullptr || down_in == nullptr || down_ctx == nullptr || ctx_in == nullptr || ctx0 == nullptr ||
      scalar == nullptr || hidden == nullptr || hidden2 == nullptr || item_in == nullptr || item_norm == nullptr || sat_emb == nullptr ||
      gu_emb == nullptr || gu_h == nullptr || gu_tmp == nullptr || score == nullptr || scaled_score == nullptr ||
      det_mean == nullptr || action == nullptr) {
    return;
  }

  __shared__ bool valid_mask_s[kMaxItems];
  const bool* sat_mask = runtime.b[kBLiveBwObs + 0] + row * select_k;
  const bool* gu_mask = runtime.b[kBLiveBwObs + 1] + row * num_gu;
  const bool* bw_valid_mask = runtime.b[kBLiveBwObs + 2] + row * num_gu;
  for (int g = threadIdx.x; g < num_gu; g += blockDim.x) {
    valid_mask_s[g] = gu_mask[g] && bw_valid_mask[g];
    score[g] = 0.0f;
    scaled_score[g] = 0.0f;
    det_mean[g] = 0.0f;
    action[g] = 0.0f;
  }
  __syncthreads();

  const int valid_count = block_count_mask(valid_mask_s, num_gu, reduce);
  if (valid_count <= 0) {
    for (int g = threadIdx.x; g < num_gu; g += blockDim.x) {
      const int idx = row * num_gu + g;
      runtime.f[kFLiveBwAction][idx] = 0.0f;
      if (kFLiveBwRefAction < runtime.nf && runtime.f[kFLiveBwRefAction] != nullptr) runtime.f[kFLiveBwRefAction][idx] = 0.0f;
      if (kFLiveBwFlowProxyOverrideAction < runtime.nf && runtime.f[kFLiveBwFlowProxyOverrideAction] != nullptr) runtime.f[kFLiveBwFlowProxyOverrideAction][idx] = 0.0f;
    }
    if (threadIdx.x == 0) {
      if (has_f(runtime, kFLiveBwOldLogprobPerAgent)) runtime.f[kFLiveBwOldLogprobPerAgent][row] = 0.0f;
      if (has_f(runtime, kFLiveBwEntropyPerAgent)) runtime.f[kFLiveBwEntropyPerAgent][row] = 0.0f;
      if (has_f(runtime, kFLiveBwLogprobRawPerAgent)) runtime.f[kFLiveBwLogprobRawPerAgent][row] = 0.0f;
      if (has_f(runtime, kFLiveBwEntropyRawPerAgent)) runtime.f[kFLiveBwEntropyRawPerAgent][row] = 0.0f;
      if (has_f(runtime, kFLiveBwTau)) runtime.f[kFLiveBwTau][row] = 0.0f;
      if (has_f(runtime, kFLiveBwKappa)) runtime.f[kFLiveBwKappa][row] = 0.0f;
      if (has_l(runtime, kLLiveBwValidCount)) runtime.l[kLLiveBwValidCount][row] = 0;
      if (has_l(runtime, kLLiveBwLatentCount)) runtime.l[kLLiveBwLatentCount][row] = 0;
    }
    return;
  }

  const float* ego_src = runtime.f[kFLiveBwObs + 0] + row * kBwEgoDim;
  block_layer_norm_or_copy(actor, ego_src, ego_norm, kBwEgoDim, W_BW_EGO_NORM_WEIGHT, W_BW_EGO_NORM_BIAS, reduce);
  block_mlp_flex(actor, ego_norm, ego_emb, hidden, hidden2, kBwEgoDim, hidden_dim, embed_dim, W_BW_EGO_ENC0_WEIGHT, W_BW_EGO_ENC0_BIAS, W_BW_EGO_ENC2_WEIGHT, W_BW_EGO_ENC2_BIAS, X_BW_EGO_ENCODER, encoder_layers, false, 1);

  const float* sat_tokens = runtime.f[kFLiveBwObs + 1] + static_cast<int64_t>(row) * select_k * kBwSatTokenDim;
  block_layer_norm_items_or_copy(actor, sat_tokens, item_norm, select_k, kBwSatTokenDim, W_BW_SAT_NORM_WEIGHT, W_BW_SAT_NORM_BIAS, reduce);
  block_mlp_items_flex(actor, item_norm, sat_emb, hidden, hidden2, select_k, kBwSatTokenDim, hidden_dim, embed_dim, W_BW_SAT_ENC0_WEIGHT, W_BW_SAT_ENC0_BIAS, W_BW_SAT_ENC2_WEIGHT, W_BW_SAT_ENC2_BIAS, X_BW_SAT_ENCODER, encoder_layers, false, 1);

  const float* gu_tokens = runtime.f[kFLiveBwObs + 2] + static_cast<int64_t>(row) * num_gu * kBwGuTokenDim;
  block_layer_norm_items_or_copy(actor, gu_tokens, item_norm, num_gu, kBwGuTokenDim, W_BW_USER_NORM_WEIGHT, W_BW_USER_NORM_BIAS, reduce);
  block_mlp_items_flex(actor, item_norm, gu_emb, hidden, hidden2, num_gu, kBwGuTokenDim, hidden_dim, embed_dim, W_BW_USER_ENC0_WEIGHT, W_BW_USER_ENC0_BIAS, W_BW_USER_ENC2_WEIGHT, W_BW_USER_ENC2_BIAS, X_BW_USER_ENCODER, encoder_layers, false, 1);

  if (has_w(actor, W_BW_Q1_0_WEIGHT) && has_w(actor, W_BW_Q1_0_BIAS)) {
    block_linear(actor, ego_emb, queries, embed_dim, down_queries * embed_dim, W_BW_Q1_0_WEIGHT, W_BW_Q1_0_BIAS, false);
  } else {
    for (int idx = threadIdx.x; idx < down_queries * embed_dim; idx += blockDim.x) {
      queries[idx] = ego_emb[idx % embed_dim];
    }
    __syncthreads();
  }
  for (int q = 0; q < down_queries; ++q) {
    block_attention(queries + q * embed_dim, sat_emb, sat_mask, select_k, embed_dim, down_attn + q * embed_dim, reduce);
  }

  for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) {
    float sum = 0.0f;
    for (int k = 0; k < select_k; ++k) {
      if (!sat_mask[k]) continue;
      float acc = has_w(actor, W_BW_SAT_REF0_BIAS) ? actor.w[W_BW_SAT_REF0_BIAS][d] : 0.0f;
      if (has_w(actor, W_BW_SAT_REF0_WEIGHT)) {
        const float* row_w = actor.w[W_BW_SAT_REF0_WEIGHT] + d * embed_dim;
        const float* tok = sat_emb + k * embed_dim;
        for (int i = 0; i < embed_dim; ++i) acc += row_w[i] * tok[i];
      } else {
        acc += sat_emb[k * embed_dim + d];
      }
      sum += acc;
    }
    sat_add_pool[d] = sum;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < down_queries * embed_dim; idx += blockDim.x) down_in[idx] = down_attn[idx];
  for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) down_in[down_queries * embed_dim + d] = sat_add_pool[d];
  __syncthreads();
  block_mlp_flex(actor, down_in, down_ctx, hidden, hidden2, (down_queries + 1) * embed_dim, hidden_dim, embed_dim, W_BW_SAT_CTX0_WEIGHT, W_BW_SAT_CTX0_BIAS, W_BW_SAT_CTX2_WEIGHT, W_BW_SAT_CTX2_BIAS, X_BW_DOWN_CONTEXT, context_layers, false, 1);

  for (int d = threadIdx.x; d < 2 * embed_dim; d += blockDim.x) ctx_in[d] = d < embed_dim ? ego_emb[d] : down_ctx[d - embed_dim];
  __syncthreads();
  block_mlp_flex(actor, ctx_in, ctx0, hidden, hidden2, 2 * embed_dim, hidden_dim, embed_dim, W_BW_GLOBAL0_WEIGHT, W_BW_GLOBAL0_BIAS, W_BW_GLOBAL2_WEIGHT, W_BW_GLOBAL2_BIAS, X_BW_CTX0, context_layers, false, 1);

  for (int idx = threadIdx.x; idx < num_gu * (2 * embed_dim); idx += blockDim.x) {
    const int g = idx / (2 * embed_dim);
    const int d = idx - g * (2 * embed_dim);
    item_in[idx] = d < embed_dim ? gu_emb[g * embed_dim + d] : ctx0[d - embed_dim];
  }
  __syncthreads();
  block_mlp_items_flex(actor, item_in, gu_h, hidden, hidden2, num_gu, 2 * embed_dim, hidden_dim, embed_dim, W_BW_USER_CTX_FUSION0_WEIGHT, W_BW_USER_CTX_FUSION0_BIAS, W_BW_USER_CTX_FUSION2_WEIGHT, W_BW_USER_CTX_FUSION2_BIAS, X_BW_USER_CONTEXT_FUSION, context_layers, false, 1);

  for (int layer = 0; layer < competition_layers; ++layer) {
    block_competition_layer(
        actor,
        gu_h,
        item_in,
        item_norm,
        gu_tmp,
        hidden,
        valid_mask_s,
        num_gu,
        embed_dim,
        hidden_dim,
        competition_heads,
        W_BW_COMP_BASE + layer * W_BW_COMP_STRIDE,
        attn_scores,
        reduce);
  }

  block_mlp_items_flex(actor, gu_h, score, hidden, hidden2, num_gu, embed_dim, hidden_dim, 1, W_BW_SCORE0_WEIGHT, W_BW_SCORE0_BIAS, W_BW_SCORE2_WEIGHT, W_BW_SCORE2_BIAS, X_BW_SCORE_HEAD, head_layers, false, 1);
  for (int g = threadIdx.x; g < num_gu; g += blockDim.x) {
    if (!valid_mask_s[g]) score[g] = 0.0f;
  }
  __syncthreads();

  const float fixed_tau = afp(actor, kActorBwFixedTau, -1.0f);
  float tau = fixed_tau > 0.0f ? fixed_tau : 0.0f;
  if (!(fixed_tau > 0.0f)) {
    block_mlp_flex(actor, ctx0, scalar, hidden, hidden2, embed_dim, hidden_dim, 1, W_BW_TAU0_WEIGHT, W_BW_TAU0_BIAS, W_BW_TAU2_WEIGHT, W_BW_TAU2_BIAS, X_BW_TAU_HEAD, head_layers, false, 1);
    tau = afp(actor, kActorBwTauMin, 0.7f) + (afp(actor, kActorBwTauMax, 1.3f) - afp(actor, kActorBwTauMin, 0.7f)) * sigmoid_device(scalar[0]);
  }
  const float fixed_kappa = afp(actor, kActorBwFixedKappa, -1.0f);
  float kappa = fixed_kappa > 0.0f ? fixed_kappa : 0.0f;
  if (!(fixed_kappa > 0.0f)) {
    block_mlp_flex(actor, ctx0, scalar, hidden, hidden2, embed_dim, hidden_dim, 1, W_BW_KAPPA0_WEIGHT, W_BW_KAPPA0_BIAS, W_BW_KAPPA2_WEIGHT, W_BW_KAPPA2_BIAS, X_BW_KAPPA_HEAD, head_layers, false, 1);
    kappa = afp(actor, kActorBwKappaMin, 0.5f) + (afp(actor, kActorBwKappaMax, 32.0f) - afp(actor, kActorBwKappaMin, 0.5f)) * sigmoid_device(scalar[0]);
  }

  for (int g = threadIdx.x; g < num_gu; g += blockDim.x) scaled_score[g] = valid_mask_s[g] ? score[g] / fmaxf(tau, 1.0e-6f) : 0.0f;
  __syncthreads();
  block_masked_simplex_from_logits(scaled_score, valid_mask_s, num_gu, det_mean, reduce);
  if (valid_count == 1) {
    for (int g = threadIdx.x; g < num_gu; g += blockDim.x) det_mean[g] = valid_mask_s[g] ? 1.0f : 0.0f;
    __syncthreads();
  }

  const int dirichlet_mode = static_cast<int>(aip(actor, kActorBwDirichletDiagnosticMode, 0));
  const bool legacy_alpha = dirichlet_mode == 2;
  const bool precise_logprob = dirichlet_mode == 0;
  if (deterministic || valid_count <= 1) {
    for (int g = threadIdx.x; g < num_gu; g += blockDim.x) action[g] = det_mean[g];
    __syncthreads();
  } else {
    float local_mean_sum = 0.0f;
    for (int g = threadIdx.x; g < num_gu; g += blockDim.x) {
      if (valid_mask_s[g]) local_mean_sum += dirichlet_mean_valid_device(det_mean[g]);
    }
    reduce[threadIdx.x] = local_mean_sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
      __syncthreads();
    }
    const float mean_sum = fmaxf(reduce[0], 1.0e-8f);

    float local_sum = 0.0f;
    for (int g = threadIdx.x; g < num_gu; g += blockDim.x) {
      const float alpha = valid_mask_s[g]
          ? (legacy_alpha ? dirichlet_alpha_legacy_device(det_mean[g], kappa) : dirichlet_alpha_from_mean_device(det_mean[g], kappa, mean_sum))
          : 0.0f;
      const float sample = valid_mask_s[g] ? gamma_sample_mt(alpha, actor, rng_step, row, g + 17) : 0.0f;
      action[g] = sample;
      local_sum += sample;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
      __syncthreads();
    }
    const float denom = reduce[0];
    for (int g = threadIdx.x; g < num_gu; g += blockDim.x) action[g] = denom > 1.0e-8f && valid_mask_s[g] ? action[g] / denom : 0.0f;
    __syncthreads();
  }

  const float raw_logprob = precise_logprob
      ? block_dirichlet_log_prob_precise(det_mean, kappa, valid_mask_s, action, num_gu, reduce)
      : block_dirichlet_log_prob_fast(det_mean, kappa, valid_mask_s, action, num_gu, reduce, legacy_alpha);
  const float entropy_raw = block_dirichlet_entropy_fast(det_mean, kappa, valid_mask_s, num_gu, reduce, legacy_alpha);
  const float latent_denom = fmaxf(static_cast<float>(valid_count - 1), 1.0f);
  const float logprob = raw_logprob / latent_denom;
  const float entropy = entropy_raw / latent_denom;
  for (int g = threadIdx.x; g < num_gu; g += blockDim.x) {
    const int idx = row * num_gu + g;
    const float value = valid_mask_s[g] ? action[g] : 0.0f;
    const float ref_value = valid_mask_s[g] ? det_mean[g] : 0.0f;
    runtime.f[kFLiveBwAction][idx] = value;
    if (kFLiveBwRefAction < runtime.nf && runtime.f[kFLiveBwRefAction] != nullptr) runtime.f[kFLiveBwRefAction][idx] = ref_value;
    if (kFLiveBwFlowProxyOverrideAction < runtime.nf && runtime.f[kFLiveBwFlowProxyOverrideAction] != nullptr) runtime.f[kFLiveBwFlowProxyOverrideAction][idx] = ref_value;
  }
  if (threadIdx.x == 0) {
    if (has_f(runtime, kFLiveBwOldLogprob)) atomicAdd(runtime.f[kFLiveBwOldLogprob] + e, logprob);
    if (has_f(runtime, kFLiveBwOldLogprobPerAgent)) runtime.f[kFLiveBwOldLogprobPerAgent][row] = logprob;
    if (has_f(runtime, kFLiveBwEntropyPerAgent)) runtime.f[kFLiveBwEntropyPerAgent][row] = entropy;
    if (has_f(runtime, kFLiveBwLogprobRawPerAgent)) runtime.f[kFLiveBwLogprobRawPerAgent][row] = raw_logprob;
    if (has_f(runtime, kFLiveBwEntropyRawPerAgent)) runtime.f[kFLiveBwEntropyRawPerAgent][row] = entropy_raw;
    if (has_f(runtime, kFLiveBwTau)) runtime.f[kFLiveBwTau][row] = tau;
    if (has_f(runtime, kFLiveBwKappa)) runtime.f[kFLiveBwKappa][row] = kappa;
    if (has_l(runtime, kLLiveBwValidCount)) runtime.l[kLLiveBwValidCount][row] = static_cast<int64_t>(valid_count);
    if (has_l(runtime, kLLiveBwLatentCount)) runtime.l[kLLiveBwLatentCount][row] = static_cast<int64_t>(max(valid_count - 1, 0));
  }
}

void launch_actor_kernel(
    const RuntimePackedAbi& runtime,
    const ActorPackedAbi& actor,
    const at::Tensor& device_anchor,
    int which,
    int64_t active_idx,
    bool deterministic,
    int64_t rng_step,
    int64_t history_slot = -1) {
  const c10::cuda::CUDAGuard guard(device_anchor.device());
  const int num_envs = static_cast<int>(runtime.ip[kParamNumEnvs]);
  if (num_envs <= 0) {
    return;
  }
  const int num_uav = static_cast<int>(runtime.ip[kParamNumUav]);
  const int row_count = num_envs * max(num_uav, 1);
  const dim3 grid(which == 0 || which == 1 || which == 2 ? row_count : num_envs);
  const dim3 block(which == 2 ? 256 : 128);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_anchor.device().index());
  copy_actor_abi_to_symbols(runtime, actor, stream);
  if (which == 0) {
    actor_accel_live_kernel<<<grid, block, 0, stream>>>(active_idx, deterministic, rng_step);
  } else if (which == 1) {
    actor_sat_live_kernel<<<grid, block, 0, stream>>>(deterministic, rng_step);
  } else {
    const int num_gu = static_cast<int>(runtime.ip[kParamNumGu]);
    const int select_k = static_cast<int>(runtime.ip[kParamSatNumSelect]);
    const int hidden_dim = static_cast<int>(actor.nip > kActorBwHidden ? actor.ip[kActorBwHidden] : actor.ip[kActorHidden]);
    const int embed_dim = static_cast<int>(actor.nip > kActorBwEmbed ? actor.ip[kActorBwEmbed] : actor.ip[kActorEmbed]);
    if (num_gu <= 0 || num_gu > kMaxItems) {
      throw std::runtime_error("native BW actor CUDA requires 0 < num_gu <= compiled kMaxItems.");
    }
    if (select_k <= 0 || select_k > kMaxItems) {
      throw std::runtime_error("native BW actor CUDA requires 0 < sat_num_select <= compiled kMaxItems.");
    }
    if (hidden_dim <= 0 || hidden_dim > kMaxHidden) {
      throw std::runtime_error("native BW actor CUDA hidden_dim exceeds compiled boundary.");
    }
    if (embed_dim <= 0 || embed_dim > kMaxEmbed) {
      throw std::runtime_error("native BW actor CUDA embed_dim exceeds compiled boundary.");
    }
    if (kFLiveBwOldLogprob < runtime.nf && runtime.f[kFLiveBwOldLogprob] != nullptr && runtime.f_numel[kFLiveBwOldLogprob] >= num_envs) {
      C10_CUDA_CHECK(cudaMemsetAsync(runtime.f[kFLiveBwOldLogprob], 0, sizeof(float) * static_cast<size_t>(num_envs), stream));
    }
    actor_bw_live_kernel<<<grid, block, 0, stream>>>(deterministic, rng_step, history_slot);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

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
    int64_t rng_step) {
  RuntimePackedAbi runtime = pack_runtime_abi(float_tensors, long_tensors, bool_tensors, int_tensors, int_params, float_params);
  ActorPackedAbi actor = pack_actor_abi(actor_weights, actor_int_tensors, actor_int_params, actor_float_params);
  launch_actor_kernel(runtime, actor, float_tensors.at(0), 0, active_idx, deterministic, rng_step);
}

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
    int64_t rng_step) {
  RuntimePackedAbi runtime = pack_runtime_abi(float_tensors, long_tensors, bool_tensors, int_tensors, int_params, float_params);
  ActorPackedAbi actor = pack_actor_abi(actor_weights, actor_int_tensors, actor_int_params, actor_float_params);
  launch_actor_kernel(runtime, actor, float_tensors.at(0), 1, 0, deterministic, rng_step);
}

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
    int64_t rng_step) {
  RuntimePackedAbi runtime = pack_runtime_abi(float_tensors, long_tensors, bool_tensors, int_tensors, int_params, float_params);
  ActorPackedAbi actor = pack_actor_abi(actor_weights, actor_int_tensors, actor_int_params, actor_float_params);
  const int num_envs = host_aip(int_params, kParamNumEnvs, 0);
  const int num_uav = host_aip(int_params, kParamNumUav, 0);
  const int row_count = num_envs * num_uav;
  const int visible = host_aip(int_params, kParamSatVisibleWidth, 0);
  const int subset_count = host_aip(int_params, kParamSubsetCount, 0);
  const int select_k = host_aip(int_params, kParamSatNumSelect, 0);
  const int hidden_dim = host_aip(actor_int_params, kActorSatHidden, host_aip(actor_int_params, kActorHidden, 0));
  const int embed_dim = host_aip(actor_int_params, kActorSatEmbed, host_aip(actor_int_params, kActorEmbed, 0));
  if (row_count <= 0) return;
  if (visible < 0 || visible > kMaxItems || subset_count <= 0 || subset_count > kMaxSubset ||
      select_k <= 0 || select_k > kMaxSelect) {
    throw std::runtime_error("native fused SAT actor received unsupported visible/subset/select dimensions.");
  }
  if (hidden_dim <= 0 || hidden_dim > kMaxHidden || embed_dim <= 0 || embed_dim > kMaxEmbed) {
    throw std::runtime_error("native fused SAT actor hidden/embed dimension exceeds compiled boundary.");
  }
  if (!host_has_tensor(float_tensors, kFLiveSatObs + 0) || !host_has_tensor(float_tensors, kFLiveSatObs + 1) ||
      !host_has_tensor(float_tensors, kFLiveSatObs + 2) || !host_has_tensor(float_tensors, kFLiveSatObs + 3) ||
      !host_has_tensor(bool_tensors, kBLiveSatObs + 0) || !host_has_tensor(bool_tensors, kBLiveSatObs + 1)) {
    throw std::runtime_error("native fused SAT actor requires live SAT obs tensors.");
  }

  const c10::cuda::CUDAGuard guard(float_tensors.at(0).device());
  const c10::InferenceMode inference_guard(true);
  at::Tensor ego = float_tensors[static_cast<size_t>(kFLiveSatObs + 0)].view({row_count, 13});
  at::Tensor demand = float_tensors[static_cast<size_t>(kFLiveSatObs + 1)].view({row_count, 8});
  at::Tensor role = float_tensors[static_cast<size_t>(kFLiveSatObs + 2)].view({row_count, 1});
  at::Tensor sat_tokens = float_tensors[static_cast<size_t>(kFLiveSatObs + 3)].view({row_count, visible, 26});
  at::Tensor sat_mask = bool_tensors[static_cast<size_t>(kBLiveSatObs + 0)].view({row_count, visible});
  at::Tensor sat_valid = bool_tensors[static_cast<size_t>(kBLiveSatObs + 1)].view({row_count, visible});
  at::Tensor valid = sat_mask.logical_and(sat_valid);

  const int encoder_layers = host_aip(actor_int_params, kActorSatEncoderMlpLayers, 2);
  const int context_layers = host_aip(actor_int_params, kActorSatContextMlpLayers, 2);
  const int head_layers = host_aip(actor_int_params, kActorSatHeadMlpLayers, 2);
  const int heads = std::max(host_aip(actor_int_params, kActorSatAttentionHeads, 1), 1);
  const int layer_count = std::max(host_aip(actor_int_params, kActorSatCompetitionLayers, 1), 1);
  const int block_base = host_aip(actor_int_params, kActorSatBlockWeightBase, 0);
  const int block_stride = std::max(host_aip(actor_int_params, kActorSatBlockWeightStride, 12), 12);

  at::Tensor ego_emb = host_mlp_flex(
      actor_weights,
      actor_int_params,
      host_layer_norm_or_copy(actor_weights, ego, W_SAT_EGO_NORM_WEIGHT, W_SAT_EGO_NORM_BIAS),
      W_SAT_EGO_ENC0_WEIGHT,
      W_SAT_EGO_ENC0_BIAS,
      W_SAT_EGO_ENC2_WEIGHT,
      W_SAT_EGO_ENC2_BIAS,
      X_SAT_EGO_ENCODER,
      encoder_layers,
      false);
  at::Tensor demand_emb = host_mlp_flex(
      actor_weights,
      actor_int_params,
      host_layer_norm_or_copy(actor_weights, demand, W_SAT_SUBSET_NORM_WEIGHT, W_SAT_SUBSET_NORM_BIAS),
      W_SAT_Q1_0_WEIGHT,
      W_SAT_Q1_0_BIAS,
      W_SAT_Q1_2_WEIGHT,
      W_SAT_Q1_2_BIAS,
      X_SAT_DEMAND_ENCODER,
      encoder_layers,
      false);
  at::Tensor role_emb = host_mlp_flex(
      actor_weights,
      actor_int_params,
      role,
      W_SAT_Q2_0_WEIGHT,
      W_SAT_Q2_0_BIAS,
      W_SAT_Q2_2_WEIGHT,
      W_SAT_Q2_2_BIAS,
      X_SAT_ROLE_ENCODER,
      encoder_layers,
      false);
  at::Tensor ctx0 = host_mlp_flex(
      actor_weights,
      actor_int_params,
      at::cat({ego_emb, demand_emb, role_emb}, -1),
      W_SAT_REF0_WEIGHT,
      W_SAT_REF0_BIAS,
      W_SAT_REF2_WEIGHT,
      W_SAT_REF2_BIAS,
      X_SAT_CTX_ENCODER,
      context_layers,
      false);
  at::Tensor sat_emb = host_mlp_flex(
      actor_weights,
      actor_int_params,
      host_layer_norm_or_copy(actor_weights, sat_tokens, W_SAT_INPUT_NORM_WEIGHT, W_SAT_INPUT_NORM_BIAS),
      W_SAT_ENC0_WEIGHT,
      W_SAT_ENC0_BIAS,
      W_SAT_ENC2_WEIGHT,
      W_SAT_ENC2_BIAS,
      X_SAT_SAT_ENCODER,
      encoder_layers,
      false);
  at::Tensor ctx_expanded = ctx0.unsqueeze(1).expand({row_count, visible, embed_dim});
  at::Tensor sat_h = host_mlp_flex(
      actor_weights,
      actor_int_params,
      at::cat({sat_emb, ctx_expanded}, -1),
      W_SAT_EGO_FUSION0_WEIGHT,
      W_SAT_EGO_FUSION0_BIAS,
      W_SAT_EGO_FUSION2_WEIGHT,
      W_SAT_EGO_FUSION2_BIAS,
      X_SAT_CONTEXT_FUSION,
      context_layers,
      false);
  sat_h = sat_h * valid.to(sat_h.scalar_type()).unsqueeze(-1);
  for (int layer = 0; layer < layer_count; ++layer) {
    sat_h = host_sat_attention_block(actor_weights, actor_int_params, sat_h, valid, layer, embed_dim, hidden_dim, heads, block_base, block_stride);
  }
  at::Tensor item_logits = host_mlp_flex(
      actor_weights,
      actor_int_params,
      sat_h,
      W_SAT_SUBSET_ENC0_WEIGHT,
      W_SAT_SUBSET_ENC0_BIAS,
      W_SAT_SUBSET_ENC2_WEIGHT,
      W_SAT_SUBSET_ENC2_BIAS,
      X_SAT_LOGIT_HEAD,
      head_layers,
      false)
                             .squeeze(-1);
  item_logits = at::where(valid, item_logits, at::zeros_like(item_logits)).contiguous();
  at::Tensor count_logits = host_mlp_flex(
      actor_weights,
      actor_int_params,
      ctx0,
      W_SAT_PROJECT0_WEIGHT,
      W_SAT_PROJECT0_BIAS,
      W_SAT_PROJECT2_WEIGHT,
      W_SAT_PROJECT2_BIAS,
      X_SAT_COUNT_HEAD,
      head_layers,
      false)
                                .contiguous();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(float_tensors.at(0).device().index());
  copy_actor_abi_to_symbols(runtime, actor, stream);
  const dim3 grid(row_count);
  const dim3 block(128);
  const size_t shared_bytes =
      sizeof(float) * static_cast<size_t>(block.x) +
      sizeof(bool) * static_cast<size_t>(kMaxItems + kMaxSubset) +
      sizeof(float) * static_cast<size_t>(kMaxSubset);
  actor_sat_select_from_logits_kernel<<<grid, block, shared_bytes, stream>>>(
      item_logits.data_ptr<float>(),
      count_logits.data_ptr<float>(),
      deterministic,
      rng_step);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    int64_t rng_step) {
  RuntimePackedAbi runtime = pack_runtime_abi(float_tensors, long_tensors, bool_tensors, int_tensors, int_params, float_params);
  ActorPackedAbi actor = pack_actor_abi(actor_weights, actor_int_tensors, actor_int_params, actor_float_params);
  const int num_envs = host_aip(int_params, kParamNumEnvs, 0);
  const int num_uav = host_aip(int_params, kParamNumUav, 0);
  const int row_count = num_envs * num_uav;
  if (row_count <= 0) return;
  const int active = static_cast<int>(active_idx) == 0 ? 0 : 1;
  const int live_f = active == 0 ? kFLiveAccelObs0 : kFLiveAccelObs1;
  const int live_b = active == 0 ? kBLiveAccelObs0 : kBLiveAccelObs1;
  const int ego_dim = host_aip(actor_int_params, kActorAccelEgoDim, 28);
  const int cell_dim = host_aip(actor_int_params, kActorAccelCellDim, 18);
  const int gu_dim = host_aip(actor_int_params, kActorAccelGuTokenDim, 27);
  const int peer_dim = host_aip(actor_int_params, kActorAccelPeerTokenDim, 28);
  const int sat_dim = host_aip(actor_int_params, kActorAccelSatTokenDim, 32);
  const int num_gu = host_aip(int_params, kParamNumGu, 0);
  const int visible = host_aip(int_params, kParamAccelSatWidth, 0);
  const int peer_count = std::max(num_uav - 1, 0);
  const int hidden_dim = host_aip(actor_int_params, kActorAccelHidden, host_aip(actor_int_params, kActorHidden, 0));
  const int embed_dim = host_aip(actor_int_params, kActorAccelEmbed, host_aip(actor_int_params, kActorEmbed, 0));
  const int gu_query_count = std::max(host_aip(actor_int_params, kActorAccelGuQueryCount, 4), 1);
  const int peer_query_count = std::max(host_aip(actor_int_params, kActorAccelPeerQueryCount, 2), 1);
  const int sat_query_count = std::max(host_aip(actor_int_params, kActorAccelSatQueryCount, 2), 1);
  if (hidden_dim <= 0 || hidden_dim > kMaxHidden || embed_dim <= 0 || embed_dim > kMaxEmbed) {
    throw std::runtime_error("native fused accel actor hidden/embed dimension exceeds compiled boundary.");
  }
  if (!host_has_tensor(float_tensors, live_f + 0) || !host_has_tensor(float_tensors, live_f + 1) ||
      !host_has_tensor(float_tensors, live_f + 2) || !host_has_tensor(float_tensors, live_f + 3) ||
      !host_has_tensor(float_tensors, live_f + 4) || !host_has_tensor(bool_tensors, live_b + 0) ||
      !host_has_tensor(bool_tensors, live_b + 1) || !host_has_tensor(bool_tensors, live_b + 2)) {
    throw std::runtime_error("native fused accel actor requires live accel obs tensors.");
  }

  const c10::cuda::CUDAGuard guard(float_tensors.at(0).device());
  const c10::InferenceMode inference_guard(true);
  at::Tensor ego = float_tensors[static_cast<size_t>(live_f + 0)].view({row_count, ego_dim});
  at::Tensor cell = float_tensors[static_cast<size_t>(live_f + 1)].view({row_count, cell_dim});
  at::Tensor gu_tokens = float_tensors[static_cast<size_t>(live_f + 2)].view({row_count, num_gu, gu_dim});
  at::Tensor peer_tokens = float_tensors[static_cast<size_t>(live_f + 3)].view({row_count, peer_count, peer_dim});
  at::Tensor sat_tokens = float_tensors[static_cast<size_t>(live_f + 4)].view({row_count, visible, sat_dim});
  at::Tensor gu_mask = bool_tensors[static_cast<size_t>(live_b + 0)].view({row_count, num_gu});
  at::Tensor peer_mask = bool_tensors[static_cast<size_t>(live_b + 1)].view({row_count, peer_count});
  at::Tensor sat_mask = bool_tensors[static_cast<size_t>(live_b + 2)].view({row_count, visible});

  const int encoder_layers = host_aip(actor_int_params, kActorAccelEncoderMlpLayers, 2);
  const int context_layers = host_aip(actor_int_params, kActorAccelContextMlpLayers, 2);
  const int head_layers = host_aip(actor_int_params, kActorAccelHeadMlpLayers, 1);
  const int interaction_layers = std::max(host_aip(actor_int_params, kActorAccelInteractionLayers, 0), 0);
  const int interaction_heads = std::max(host_aip(actor_int_params, kActorAccelAttentionHeads, 1), 1);
  const int interaction_base = host_aip(actor_int_params, kActorAccelBlockWeightBase, 0);
  const int interaction_stride = std::max(host_aip(actor_int_params, kActorAccelBlockWeightStride, 12), 12);

  at::Tensor ego_emb = host_mlp_flex(
      actor_weights,
      actor_int_params,
      host_layer_norm_or_copy(actor_weights, ego, W_ACCEL_EGO_NORM_WEIGHT, W_ACCEL_EGO_NORM_BIAS),
      W_ACCEL_EGO_ENC0_WEIGHT,
      W_ACCEL_EGO_ENC0_BIAS,
      W_ACCEL_EGO_ENC2_WEIGHT,
      W_ACCEL_EGO_ENC2_BIAS,
      X_ACCEL_EGO_ENCODER,
      encoder_layers,
      false);
  at::Tensor cell_emb = host_mlp_flex(
      actor_weights,
      actor_int_params,
      host_layer_norm_or_copy(actor_weights, cell, W_ACCEL_CELL_NORM_WEIGHT, W_ACCEL_CELL_NORM_BIAS),
      W_ACCEL_CELL_ENC0_WEIGHT,
      W_ACCEL_CELL_ENC0_BIAS,
      W_ACCEL_CELL_ENC2_WEIGHT,
      W_ACCEL_CELL_ENC2_BIAS,
      X_ACCEL_CELL_ENCODER,
      encoder_layers,
      false);
  at::Tensor gu_h = host_mlp_flex(
      actor_weights,
      actor_int_params,
      host_layer_norm_or_copy(actor_weights, gu_tokens, W_ACCEL_GU_NORM_WEIGHT, W_ACCEL_GU_NORM_BIAS),
      W_ACCEL_GU_ENC0_WEIGHT,
      W_ACCEL_GU_ENC0_BIAS,
      W_ACCEL_GU_ENC2_WEIGHT,
      W_ACCEL_GU_ENC2_BIAS,
      X_ACCEL_GU_ENCODER,
      encoder_layers,
      false);
  at::Tensor peer_h = host_mlp_flex(
      actor_weights,
      actor_int_params,
      host_layer_norm_or_copy(actor_weights, peer_tokens, W_ACCEL_PEER_NORM_WEIGHT, W_ACCEL_PEER_NORM_BIAS),
      W_ACCEL_PEER_ENC0_WEIGHT,
      W_ACCEL_PEER_ENC0_BIAS,
      W_ACCEL_PEER_ENC2_WEIGHT,
      W_ACCEL_PEER_ENC2_BIAS,
      X_ACCEL_PEER_ENCODER,
      encoder_layers,
      false);
  at::Tensor sat_h = host_mlp_flex(
      actor_weights,
      actor_int_params,
      host_layer_norm_or_copy(actor_weights, sat_tokens, W_ACCEL_SAT_NORM_WEIGHT, W_ACCEL_SAT_NORM_BIAS),
      W_ACCEL_SAT_ENC0_WEIGHT,
      W_ACCEL_SAT_ENC0_BIAS,
      W_ACCEL_SAT_ENC2_WEIGHT,
      W_ACCEL_SAT_ENC2_BIAS,
      X_ACCEL_SAT_ENCODER,
      encoder_layers,
      false);
  gu_h = gu_h * gu_mask.to(gu_h.scalar_type()).unsqueeze(-1);
  peer_h = peer_h * peer_mask.to(peer_h.scalar_type()).unsqueeze(-1);
  sat_h = sat_h * sat_mask.to(sat_h.scalar_type()).unsqueeze(-1);
  for (int layer = 0; layer < interaction_layers; ++layer) {
    const int base = interaction_base + layer * interaction_stride;
    gu_h = host_mha_residual_block(actor_weights, gu_h, gu_mask, embed_dim, hidden_dim, interaction_heads, base);
    peer_h = host_mha_residual_block(actor_weights, peer_h, peer_mask, embed_dim, hidden_dim, interaction_heads, base);
    sat_h = host_mha_residual_block(actor_weights, sat_h, sat_mask, embed_dim, hidden_dim, interaction_heads, base);
  }
  at::Tensor query_src = at::cat({ego_emb, cell_emb}, -1);
  at::Tensor gu_queries = host_linear(actor_weights, query_src, W_ACCEL_GU_QUERY_WEIGHT, W_ACCEL_GU_QUERY_BIAS)
                              .view({row_count, gu_query_count, embed_dim});
  at::Tensor peer_queries = host_linear(actor_weights, query_src, W_ACCEL_PEER_QUERY_WEIGHT, W_ACCEL_PEER_QUERY_BIAS)
                                .view({row_count, peer_query_count, embed_dim});
  at::Tensor sat_queries = host_linear(actor_weights, query_src, W_ACCEL_SAT_QUERY_WEIGHT, W_ACCEL_SAT_QUERY_BIAS)
                               .view({row_count, sat_query_count, embed_dim});
  at::Tensor gu_attn = host_multi_query_attention(gu_queries, gu_h, gu_mask).flatten(1);
  at::Tensor peer_attn = host_multi_query_attention(peer_queries, peer_h, peer_mask).flatten(1);
  at::Tensor sat_attn = host_multi_query_attention(sat_queries, sat_h, sat_mask).flatten(1);
  at::Tensor fusion_in = at::cat(
      {
          ego_emb,
          cell_emb,
          gu_attn,
          host_masked_mean(gu_h, gu_mask),
          host_masked_max(gu_h, gu_mask),
          peer_attn,
          host_masked_mean(peer_h, peer_mask),
          host_masked_max(peer_h, peer_mask),
          sat_attn,
          host_masked_mean(sat_h, sat_mask),
          host_masked_max(sat_h, sat_mask),
      },
      -1);
  at::Tensor fusion = host_mlp_flex(
      actor_weights,
      actor_int_params,
      fusion_in,
      W_ACCEL_FUSION0_WEIGHT,
      W_ACCEL_FUSION0_BIAS,
      W_ACCEL_FUSION2_WEIGHT,
      W_ACCEL_FUSION2_BIAS,
      X_ACCEL_FUSION,
      context_layers,
      true);
  at::Tensor mean = head_layers <= 1
      ? host_linear(actor_weights, fusion, W_ACCEL_MU_WEIGHT, W_ACCEL_MU_BIAS)
      : host_mlp_flex_contiguous(
            actor_weights,
            fusion,
            host_aip(actor_int_params, kActorAccelMuHeadMlpWeightBase, 0),
            head_layers,
            false);
  mean = mean.contiguous();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(float_tensors.at(0).device().index());
  copy_actor_abi_to_symbols(runtime, actor, stream);
  const dim3 block(128);
  const dim3 grid((row_count + static_cast<int>(block.x) - 1) / static_cast<int>(block.x));
  actor_accel_write_from_mean_kernel<<<grid, block, 0, stream>>>(mean.data_ptr<float>(), deterministic, rng_step);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    int64_t history_slot) {
  RuntimePackedAbi runtime = pack_runtime_abi(float_tensors, long_tensors, bool_tensors, int_tensors, int_params, float_params);
  ActorPackedAbi actor = pack_actor_abi(actor_weights, actor_int_tensors, actor_int_params, actor_float_params);
  const int num_envs = host_aip(int_params, kParamNumEnvs, 0);
  const int num_uav = host_aip(int_params, kParamNumUav, 0);
  const int num_gu = host_aip(int_params, kParamNumGu, 0);
  const int select_k = host_aip(int_params, kParamSatNumSelect, 0);
  const int row_count = num_envs * num_uav;
  const int hidden_dim = host_aip(actor_int_params, kActorBwHidden, host_aip(actor_int_params, kActorHidden, 0));
  const int embed_dim = host_aip(actor_int_params, kActorBwEmbed, host_aip(actor_int_params, kActorEmbed, 0));
  if (row_count <= 0) return;
  if (num_gu <= 0 || num_gu > kMaxItems || select_k <= 0 || select_k > kMaxItems) {
    throw std::runtime_error("native fused BW actor received unsupported GU/select dimensions.");
  }
  if (hidden_dim <= 0 || hidden_dim > kMaxHidden || embed_dim <= 0 || embed_dim > kMaxEmbed) {
    throw std::runtime_error("native fused BW actor hidden/embed dimension exceeds compiled boundary.");
  }
  if (!host_has_tensor(float_tensors, kFLiveBwObs + 0) || !host_has_tensor(float_tensors, kFLiveBwObs + 1) ||
      !host_has_tensor(float_tensors, kFLiveBwObs + 2) || !host_has_tensor(bool_tensors, kBLiveBwObs + 0) ||
      !host_has_tensor(bool_tensors, kBLiveBwObs + 1) || !host_has_tensor(bool_tensors, kBLiveBwObs + 2)) {
    throw std::runtime_error("native fused BW actor requires live BW obs tensors.");
  }

  const c10::cuda::CUDAGuard guard(float_tensors.at(0).device());
  const c10::InferenceMode inference_guard(true);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(float_tensors.at(0).device().index());
  copy_actor_abi_to_symbols(runtime, actor, stream);
  if (kFLiveBwOldLogprob < runtime.nf && runtime.f[kFLiveBwOldLogprob] != nullptr && runtime.f_numel[kFLiveBwOldLogprob] >= num_envs) {
    C10_CUDA_CHECK(cudaMemsetAsync(runtime.f[kFLiveBwOldLogprob], 0, sizeof(float) * static_cast<size_t>(num_envs), stream));
  }

  at::Tensor row_indices;
  int eval_row_count = row_count;
  const int macro_interval = host_aip(int_params, kParamAccessBwDecisionInterval, 1);
  if (macro_interval > 1 && history_slot >= 0) {
    at::Tensor start_mask = at::empty({row_count}, float_tensors.at(0).options().dtype(at::kBool));
    const dim3 prep_grid(row_count);
    const dim3 prep_block(256);
    actor_bw_prepare_macro_start_mask_kernel<<<prep_grid, prep_block, 0, stream>>>(
        start_mask.data_ptr<bool>(),
        history_slot,
        num_uav,
        num_gu,
        row_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    row_indices = at::nonzero(start_mask).flatten().contiguous();
    eval_row_count = static_cast<int>(row_indices.numel());
    if (eval_row_count <= 0) {
      return;
    }
  }

  at::Tensor ego_full = float_tensors[static_cast<size_t>(kFLiveBwObs + 0)].view({row_count, 11});
  at::Tensor sat_tokens_full = float_tensors[static_cast<size_t>(kFLiveBwObs + 1)].view({row_count, select_k, 9});
  at::Tensor gu_tokens_full = float_tensors[static_cast<size_t>(kFLiveBwObs + 2)].view({row_count, num_gu, 13});
  at::Tensor sat_mask_full = bool_tensors[static_cast<size_t>(kBLiveBwObs + 0)].view({row_count, select_k});
  at::Tensor gu_mask_full = bool_tensors[static_cast<size_t>(kBLiveBwObs + 1)].view({row_count, num_gu});
  at::Tensor bw_valid_full = bool_tensors[static_cast<size_t>(kBLiveBwObs + 2)].view({row_count, num_gu});

  at::Tensor ego = row_indices.defined() ? ego_full.index_select(0, row_indices) : ego_full;
  at::Tensor sat_tokens = row_indices.defined() ? sat_tokens_full.index_select(0, row_indices) : sat_tokens_full;
  at::Tensor gu_tokens = row_indices.defined() ? gu_tokens_full.index_select(0, row_indices) : gu_tokens_full;
  at::Tensor sat_mask = row_indices.defined() ? sat_mask_full.index_select(0, row_indices) : sat_mask_full;
  at::Tensor gu_mask = row_indices.defined() ? gu_mask_full.index_select(0, row_indices) : gu_mask_full;
  at::Tensor bw_valid = row_indices.defined() ? bw_valid_full.index_select(0, row_indices) : bw_valid_full;
  at::Tensor valid = gu_mask.logical_and(bw_valid);
  at::Tensor sat_mask_f = sat_mask.to(sat_tokens.scalar_type());
  at::Tensor valid_f = valid.to(gu_tokens.scalar_type());
  at::Tensor valid_f_unsqueezed = valid_f.unsqueeze(-1);

  const int encoder_layers = host_aip(actor_int_params, kActorBwEncoderMlpLayers, 2);
  const int context_layers = host_aip(actor_int_params, kActorBwContextMlpLayers, 2);
  const int head_layers = host_aip(actor_int_params, kActorBwHeadMlpLayers, 2);
  const int down_queries = std::max(host_aip(actor_int_params, kActorBwDownQueryCount, 1), 1);
  const int competition_layers = std::max(host_aip(actor_int_params, kActorCompetitionLayers, 1), 1);
  const int competition_heads = std::max(host_aip(actor_int_params, kActorCompetitionHeads, 1), 1);

  at::Tensor ego_emb = host_mlp_flex(
      actor_weights,
      actor_int_params,
      host_layer_norm_or_copy(actor_weights, ego, W_BW_EGO_NORM_WEIGHT, W_BW_EGO_NORM_BIAS),
      W_BW_EGO_ENC0_WEIGHT,
      W_BW_EGO_ENC0_BIAS,
      W_BW_EGO_ENC2_WEIGHT,
      W_BW_EGO_ENC2_BIAS,
      X_BW_EGO_ENCODER,
      encoder_layers,
      false,
      1);
  at::Tensor sat_emb = host_mlp_flex(
      actor_weights,
      actor_int_params,
      host_layer_norm_or_copy(actor_weights, sat_tokens, W_BW_SAT_NORM_WEIGHT, W_BW_SAT_NORM_BIAS),
      W_BW_SAT_ENC0_WEIGHT,
      W_BW_SAT_ENC0_BIAS,
      W_BW_SAT_ENC2_WEIGHT,
      W_BW_SAT_ENC2_BIAS,
      X_BW_SAT_ENCODER,
      encoder_layers,
      false,
      1);
  at::Tensor gu_emb = host_mlp_flex(
      actor_weights,
      actor_int_params,
      host_layer_norm_or_copy(actor_weights, gu_tokens, W_BW_USER_NORM_WEIGHT, W_BW_USER_NORM_BIAS),
      W_BW_USER_ENC0_WEIGHT,
      W_BW_USER_ENC0_BIAS,
      W_BW_USER_ENC2_WEIGHT,
      W_BW_USER_ENC2_BIAS,
      X_BW_USER_ENCODER,
      encoder_layers,
      false,
      1);
  at::Tensor queries = host_has_tensor(actor_weights, W_BW_Q1_0_WEIGHT)
      ? host_linear(actor_weights, ego_emb, W_BW_Q1_0_WEIGHT, W_BW_Q1_0_BIAS).view({eval_row_count, down_queries, embed_dim})
      : ego_emb.unsqueeze(1).expand({eval_row_count, down_queries, embed_dim});
  at::Tensor down_attn = host_multi_query_attention(queries, sat_emb, sat_mask, sat_mask_f);
  at::Tensor sat_add = host_masked_sum(host_linear(actor_weights, sat_emb, W_BW_SAT_REF0_WEIGHT, W_BW_SAT_REF0_BIAS), sat_mask, sat_mask_f);
  at::Tensor down_ctx = host_mlp_flex(
      actor_weights,
      actor_int_params,
      at::cat({down_attn.flatten(1), sat_add}, -1),
      W_BW_SAT_CTX0_WEIGHT,
      W_BW_SAT_CTX0_BIAS,
      W_BW_SAT_CTX2_WEIGHT,
      W_BW_SAT_CTX2_BIAS,
      X_BW_DOWN_CONTEXT,
      context_layers,
      false,
      1);
  at::Tensor ctx0 = host_mlp_flex(
      actor_weights,
      actor_int_params,
      at::cat({ego_emb, down_ctx}, -1),
      W_BW_GLOBAL0_WEIGHT,
      W_BW_GLOBAL0_BIAS,
      W_BW_GLOBAL2_WEIGHT,
      W_BW_GLOBAL2_BIAS,
      X_BW_CTX0,
      context_layers,
      false,
      1);
  at::Tensor ctx_expand = ctx0.unsqueeze(1).expand({eval_row_count, num_gu, embed_dim});
  at::Tensor gu_h = host_mlp_flex(
      actor_weights,
      actor_int_params,
      at::cat({gu_emb, ctx_expand}, -1),
      W_BW_USER_CTX_FUSION0_WEIGHT,
      W_BW_USER_CTX_FUSION0_BIAS,
      W_BW_USER_CTX_FUSION2_WEIGHT,
      W_BW_USER_CTX_FUSION2_BIAS,
      X_BW_USER_CONTEXT_FUSION,
      context_layers,
      false,
      1);
  gu_h = gu_h * valid_f_unsqueezed;
  for (int layer = 0; layer < competition_layers; ++layer) {
    gu_h = host_mha_residual_block(
        actor_weights,
        gu_h,
        valid,
        embed_dim,
        hidden_dim,
        competition_heads,
        W_BW_COMP_BASE + layer * W_BW_COMP_STRIDE,
        valid_f_unsqueezed);
  }
  at::Tensor raw_score = host_mlp_flex(
      actor_weights,
      actor_int_params,
      gu_h,
      W_BW_SCORE0_WEIGHT,
      W_BW_SCORE0_BIAS,
      W_BW_SCORE2_WEIGHT,
      W_BW_SCORE2_BIAS,
      X_BW_SCORE_HEAD,
      head_layers,
      false,
      1)
                            .squeeze(-1);
  at::Tensor tau_raw;
  at::Tensor kappa_raw;
  at::Tensor tau_kappa_raw = host_bw_tau_kappa_head_packed(actor_weights, actor_int_params, ctx0, head_layers, 1);
  if (tau_kappa_raw.defined()) {
    tau_raw = tau_kappa_raw.select(-1, 0);
    kappa_raw = tau_kappa_raw.select(-1, 1);
  } else {
    tau_raw = host_mlp_flex(
        actor_weights,
        actor_int_params,
        ctx0,
        W_BW_TAU0_WEIGHT,
        W_BW_TAU0_BIAS,
        W_BW_TAU2_WEIGHT,
        W_BW_TAU2_BIAS,
        X_BW_TAU_HEAD,
        head_layers,
        false,
        1)
                              .squeeze(-1);
    kappa_raw = host_mlp_flex(
        actor_weights,
        actor_int_params,
        ctx0,
        W_BW_KAPPA0_WEIGHT,
        W_BW_KAPPA0_BIAS,
        W_BW_KAPPA2_WEIGHT,
        W_BW_KAPPA2_BIAS,
        X_BW_KAPPA_HEAD,
        head_layers,
        false,
        1)
                                 .squeeze(-1);
  }
  at::Tensor valid_count = valid.sum(-1);
  const double fixed_tau = actor_float_params.size() > kActorBwFixedTau ? actor_float_params[static_cast<size_t>(kActorBwFixedTau)] : -1.0;
  at::Tensor tau;
  if (fixed_tau > 0.0) {
    tau = at::full_like(tau_raw, fixed_tau);
  } else {
    const double tau_min = actor_float_params.size() > kActorBwTauMin ? actor_float_params[static_cast<size_t>(kActorBwTauMin)] : 0.7;
    const double tau_max = actor_float_params.size() > kActorBwTauMax ? actor_float_params[static_cast<size_t>(kActorBwTauMax)] : 1.3;
    tau = tau_min + (tau_max - tau_min) * at::sigmoid(tau_raw);
  }
  at::Tensor score = raw_score.masked_fill(valid.logical_not(), kNegInf);
  at::Tensor det_mean = host_masked_softmax_presafe(score / tau.unsqueeze(1), valid, -1, valid_f);
  det_mean = at::where((valid_count.unsqueeze(-1) == 1), valid_f, det_mean);
  det_mean = at::where((valid_count.unsqueeze(-1) <= 0), at::zeros_like(det_mean), det_mean);
  const double fixed_kappa = actor_float_params.size() > kActorBwFixedKappa ? actor_float_params[static_cast<size_t>(kActorBwFixedKappa)] : -1.0;
  at::Tensor kappa;
  if (fixed_kappa > 0.0) {
    kappa = at::full_like(kappa_raw, fixed_kappa);
  } else {
    const double kappa_min = actor_float_params.size() > kActorBwKappaMin ? actor_float_params[static_cast<size_t>(kActorBwKappaMin)] : 0.5;
    const double kappa_max = actor_float_params.size() > kActorBwKappaMax ? actor_float_params[static_cast<size_t>(kActorBwKappaMax)] : 32.0;
    kappa = kappa_min + (kappa_max - kappa_min) * at::sigmoid(kappa_raw);
  }
  det_mean = det_mean.contiguous();
  kappa = kappa.contiguous();
  tau = tau.contiguous();

  const dim3 grid(eval_row_count);
  const dim3 block(256);
  const size_t shared_bytes =
      sizeof(float) * static_cast<size_t>(2 * block.x) +
      sizeof(bool) * static_cast<size_t>(kMaxItems) +
      sizeof(float) * static_cast<size_t>(2 * kMaxItems);
  const int64_t* row_index_ptr = row_indices.defined() ? row_indices.data_ptr<int64_t>() : nullptr;
  actor_bw_write_from_params_kernel<<<grid, block, shared_bytes, stream>>>(
      det_mean.data_ptr<float>(),
      kappa.data_ptr<float>(),
      tau.data_ptr<float>(),
      row_index_ptr,
      static_cast<int64_t>(eval_row_count),
      deterministic,
      rng_step,
      history_slot);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    int64_t history_slot) {
  RuntimePackedAbi runtime = pack_runtime_abi(float_tensors, long_tensors, bool_tensors, int_tensors, int_params, float_params);
  ActorPackedAbi actor = pack_actor_abi(actor_weights, actor_int_tensors, actor_int_params, actor_float_params);
  launch_actor_kernel(runtime, actor, float_tensors.at(0), 2, 0, deterministic, rng_step, history_slot);
}
