#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using TensorVec = std::vector<at::Tensor>;

namespace {

constexpr int kMaxFloatTensors = 768;
constexpr int kMaxLongTensors = 128;
constexpr int kMaxBoolTensors = 128;
constexpr int kMaxIntTensors = 64;
constexpr int kMaxIntParams = 96;
constexpr int kMaxFloatParams = 224;
constexpr int kMaxVisibleTopK = 64;
constexpr int kAccelCellSharedMaxUav = 64;
constexpr int kFinishProfileSegments = 34;
constexpr int kRefreshCostScalarCount = 9;

constexpr int kSourcePolicy = 0;
constexpr int kSourceZero = 1;
constexpr int kSourceUniform = 2;
constexpr int kSourceRandom = 3;
constexpr int kSourceLinkPriority = 4;
constexpr int kSourceDemandPriority = 5;
constexpr int kSourceQueueAware = 6;
constexpr int kSourceClusterCenterQueueAware = 7;
constexpr int kSourceLyapunov = 8;
constexpr int kSourceTopologyDppBw = 9;
constexpr int kSourceDppResourceBw = kSourceTopologyDppBw;  // Legacy alias.
constexpr int kSourceTopologyDppSat = 10;
constexpr int kSourceTopologyDppAccel = 11;
constexpr int kSourceObservableClusterQueueAware = 12;

enum FinishProfileSegment : int {
  kFinishProfArrivalAndFlowProxy = 0,
  kFinishProfAccess = 1,
  kFinishProfBackhaulSetup = 2,
  kFinishProfGuQueue = 3,
  kFinishProfUavQueue = 4,
  kFinishProfSatQueue = 5,
  kFinishProfRefreshQueueDerived = 6,
  kFinishProfRewardStats = 7,
  kFinishProfHistoryAndDanger = 8,
  kFinishProfTerminalWorld = 9,
  kFinishProfCommitOrReset = 10,
  kFinishProfNextRandomTape = 11,
  kFinishProfNextPrepareStage = 12,
  kFinishProfNextWorld = 13,
  kFinishProfNextSnapshots = 14,
  kFinishProfNextAccelObs = 15,
  kFinishProfAccelObsEgoCell = 16,
  kFinishProfAccelObsGuTokens = 17,
  kFinishProfAccelObsPeerTokens = 18,
  kFinishProfAccelObsSatTokens = 19,
  kFinishProfRefreshMeanCost = 20,
  kFinishProfRefreshAssoc = 21,
  kFinishProfRefreshCandidateAccess = 22,
  kFinishProfRefreshFullAccessGain = 23,
  kFinishProfRefreshProxyAndCost = 24,
  kFinishProfRefreshFullUsGeometry = 25,
  kFinishProfRefreshVisibleTopk = 26,
  kFinishProfRefreshActiveSat = 27,
  kFinishProfRefreshActiveUs = 28,
  kFinishProfRefreshLastSelection = 29,
  kFinishProfQueueRefreshMeanCost = 30,
  kFinishProfQueueRefreshProxyAndCost = 31,
  kFinishProfQueueRefreshActiveSat = 32,
  kFinishProfQueueRefreshFullUsSatQueue = 33,
};

enum RefreshCostScalarIndex : int {
  kRefreshCostScalarMeanSat = 0,
  kRefreshCostScalarMeanUav = 1,
  kRefreshCostScalarMeanLocalGu = 2,
  kRefreshCostScalarMeanAssocUav = 3,
  kRefreshCostScalarMeanAssocSat = 4,
  kRefreshCostScalarMeanWeightedQueue = 5,
  kRefreshCostScalarWeightedQueueRef = 6,
  kRefreshCostScalarBaseArrival = 7,
  kRefreshCostScalarReserved = 8,
};

__host__ __device__ inline int refresh_cost_cache_size_for_dims(int sat, int ucount, int gu) {
  return sat + 2 * ucount + 5 * gu + kRefreshCostScalarCount;
}

enum IntParamIndex : int {
  kParamNumEnvs = 0,
  kParamNumUav = 1,
  kParamNumGu = 2,
  kParamNumSat = 3,
  kParamUsersObsMax = 4,
  kParamSatsObsMax = 5,
  kParamVisibleSatsMax = 6,
  kParamSatNumSelect = 7,
  kParamHistoryCapacity = 8,
  kParamAccelSourceMode = 9,
  kParamSatSourceMode = 10,
  kParamBwSourceMode = 11,
  kParamFlowBaseActionMode = 12,
  kParamCandidateMode = 13,
  kParamSatCandidateMode = 14,
  kParamBoundaryMode = 15,
  kParamRewardMode = 16,
  kParamTrafficModel = 17,
  kParamPathlossMode = 18,
  kParamEnergyModel = 19,
  kParamCandidateK = 20,
  kParamCandidateRadiusPresent = 21,
  kParamEnableBwAction = 22,
  kParamInterferenceEnabled = 23,
  kParamDopplerEnabled = 24,
  kParamDopplerAttenEnabled = 25,
  kParamDopplerObserved = 26,
  kParamDopplerPrecompEnabled = 27,
  kParamUseAvoidance = 28,
  kParamUseEnergySafety = 29,
  kParamBoundaryHardFilter = 30,
  kParamPairwiseHardFilter = 31,
  kParamDeadlineEnabled = 32,
  kParamEnergyEnabled = 33,
  kParamFlowProxyEnabled = 34,
  kParamFlowProxyRewardMode = 35,
  kParamObsOwnAssocCost = 36,
  kParamObsOwnUavId = 37,
  kParamObsSatCost = 38,
  kParamUserProxyDim = 39,
  kParamLocalNumUav = 40,
  kParamLocalNumGu = 41,
  kParamLocalNumSat = 42,
  kParamSatVisibleWidth = 43,
  kParamSubsetCount = 44,
  kParamUavNodeDim = 45,
  kParamUserNodeDim = 46,
  kParamSatNodeDim = 47,
  kParamQueuePenaltyMode = 48,
  kParamQueueDeltaMode = 49,
  kParamQueueRewardUseArrivalNorm = 50,
  kParamUseQueueLogSmoothing = 51,
  kParamUseActiveQueueDelta = 52,
  kParamUseEnergyReward = 53,
  kParamUseRewardTanh = 54,
  kParamCentroidCrossAnnealEnabled = 55,
  kParamEtaCentroidFinalPresent = 56,
  kParamCloseRiskEnabled = 57,
  kParamDangerImitationEnabled = 58,
  kParamDangerTriggerMode = 59,
  kParamAvoidancePrealertFactorPresent = 60,
  kParamAvoidancePrealertModeTtc = 61,
  kParamAvoidancePrealertDistCapPresent = 62,
  kParamAtmLossEnabled = 63,
  kParamAvoidanceRepulseMode = 64,
  kParamAvoidanceRepulseClip = 65,
  kParamAvoidanceClosingGainEnabled = 66,
  kParamAvoidanceClosingGainTop1Only = 67,
  kParamRainLossEnabled = 68,
  kParamObsUserIncludeArrivalRate = 69,
  kParamObsUserIncludeRecentArrival = 70,
  kParamObsUserIncludeRecentService = 71,
  kParamObsUserIncludeQueueHeadroom = 72,
  kParamObsUserIncludeLocalGuServiceCost = 73,
  kParamObsUserIncludeAssocUavCost = 74,
  kParamObsUserIncludeAssocSatCostMean = 75,
  kParamObsUserIncludeWeightedQueueCost = 76,
  kParamObsUserIncludeWeightedQueueCostRelative = 77,
  kParamObsUserIncludeUrgencyRisk = 78,
  kParamObsUserIncludeDownstreamPressure = 79,
  kParamObsUserIncludeServiceGap = 80,
  kParamObsUserIncludeServiceGapRisk = 81,
  kParamObsUserIncludeDeadlineSlack = 82,
  kParamObsUserIncludeDeadlineRisk = 83,
  kParamUavGuEdgeDim = 84,
  kParamUavSatEdgeDim = 85,
  kParamUavUavEdgeDim = 86,
  kParamAccelSatWidth = 87,
  kParamSafetyShieldNative = 88,
  kParamSafetyShieldIters = 89,
  kParamHistorySnapshotsEnabled = 90,
  kParamAccessBwDecisionInterval = 91,
  kParamSatDecisionInterval = 92,
  kParamTopologyDppGuMaxSelect = 93,
  kParamTopologyDppAccelNumCandidates = 94,
};

enum AccelEgoField : int {
  kAccelEgoX = 0,
  kAccelEgoY = 1,
  kAccelEgoVx = 2,
  kAccelEgoVy = 3,
  kAccelEgoSpeed = 4,
  kAccelEgoEnergy = 5,
  kAccelEgoBoundaryLeft = 6,
  kAccelEgoBoundaryRight = 7,
  kAccelEgoBoundaryBottom = 8,
  kAccelEgoBoundaryTop = 9,
  kAccelEgoUavQueueSteps = 10,
  kAccelEgoUavQueueFill = 11,
  kAccelEgoUavLastInflowSteps = 12,
  kAccelEgoUavLastOutflowSteps = 13,
  kAccelEgoUavLastDropSteps = 14,
  kAccelEgoUavServiceEmaSteps = 15,
  kAccelEgoUavLocalCostLogRatio = 16,
  kAccelEgoUavLastTotalCostLogRatio = 17,
  kAccelEgoUavLastWorkloadLog1p = 18,
  kAccelEgoUavLastAccessInterferenceLog1p = 19,
  kAccelEgoLastPolicyAccelX = 20,
  kAccelEgoLastPolicyAccelY = 21,
  kAccelEgoLastExecAccelX = 22,
  kAccelEgoLastExecAccelY = 23,
  kAccelEgoLastInterventionDx = 24,
  kAccelEgoLastInterventionDy = 25,
  kAccelEgoLastInterventionL2 = 26,
  kAccelEgoRemainingHorizonFrac = 27,
  kAccelEgoDim = 28,
};

enum AccelCellField : int {
  kAccelCellGuCountFrac = 0,
  kAccelCellQueueStepsSum = 1,
  kAccelCellExpectedArrivalStepsSum = 2,
  kAccelCellLastArrivalStepsSum = 3,
  kAccelCellLastOutflowStepsSum = 4,
  kAccelCellLastDropStepsSum = 5,
  kAccelCellLastWorkloadLog1pSum = 6,
  kAccelCellWorkloadShareGap = 7,
  kAccelCellBoundaryWorkloadSum = 8,
  kAccelCellWeakLinkWorkloadSum = 9,
  kAccelCellAccessPressure = 10,
  kAccelCellInterferenceExposure = 11,
  kAccelCellDemandMomentX = 12,
  kAccelCellDemandMomentY = 13,
  kAccelCellBoundaryMomentX = 14,
  kAccelCellBoundaryMomentY = 15,
  kAccelCellWeakLinkMomentX = 16,
  kAccelCellWeakLinkMomentY = 17,
  kAccelCellDim = 18,
};

enum AccelGuField : int {
  kAccelGuX = 0,
  kAccelGuY = 1,
  kAccelGuQueueSteps = 2,
  kAccelGuQueueFill = 3,
  kAccelGuExpectedArrivalSteps = 4,
  kAccelGuLastArrivalSteps = 5,
  kAccelGuLastOutflowSteps = 6,
  kAccelGuLastDropSteps = 7,
  kAccelGuServiceEmaSteps = 8,
  kAccelGuLocalCostLogRatio = 9,
  kAccelGuLastTotalCostLogRatio = 10,
  kAccelGuLastWorkloadLog1p = 11,
  kAccelGuRelX = 12,
  kAccelGuRelY = 13,
  kAccelGuDist = 14,
  kAccelGuAccessSeRef = 15,
  kAccelGuLastAssocToEgo = 16,
  kAccelGuLastBwFractionEgo = 17,
  kAccelGuLastServedByEgo = 18,
  kAccelGuPreOwnerIsEgo = 19,
  kAccelGuHandoffMarginEgo = 20,
  kAccelGuOwnerStabilityMargin = 21,
  kAccelGuEgoTakeoverGap = 22,
  kAccelGuPartitionBoundaryWeight = 23,
  kAccelGuEgoLinkWeakness = 24,
  kAccelGuLastBwSum = 25,
  kAccelGuLastAccessActive = 25,
  kAccelGuLastNonselfInterferenceLog1p = 26,
  kAccelGuTokenDim = 27,
};

enum AccelPeerField : int {
  kAccelPeerRelX = 0,
  kAccelPeerRelY = 1,
  kAccelPeerRelVx = 2,
  kAccelPeerRelVy = 3,
  kAccelPeerDist = 4,
  kAccelPeerClosingSpeed = 5,
  kAccelPeerSafeDistanceMargin = 6,
  kAccelPeerUnsafeFlag = 7,
  kAccelPeerAlertFlag = 8,
  kAccelPeerLastSharedSatFrac = 9,
  kAccelPeerCellOffset = 10,
  kAccelPeerTokenDim = 28,
};

enum AccelSatField : int {
  kAccelSatX = 0,
  kAccelSatY = 1,
  kAccelSatZ = 2,
  kAccelSatVx = 3,
  kAccelSatVy = 4,
  kAccelSatVz = 5,
  kAccelSatQueueSteps = 6,
  kAccelSatQueueFill = 7,
  kAccelSatLastIncomingSteps = 8,
  kAccelSatLastProcessedSteps = 9,
  kAccelSatLastDropSteps = 10,
  kAccelSatServiceEmaSteps = 11,
  kAccelSatCostLogRatio = 12,
  kAccelSatLastWorkloadLog1p = 13,
  kAccelSatLastSelectedLoadFrac = 14,
  kAccelSatProcCapacitySteps = 15,
  kAccelSatRelX = 16,
  kAccelSatRelY = 17,
  kAccelSatRelZ = 18,
  kAccelSatRelVx = 19,
  kAccelSatRelVy = 20,
  kAccelSatRelVz = 21,
  kAccelSatRange = 22,
  kAccelSatRadialVelocity = 23,
  kAccelSatElevation = 24,
  kAccelSatDopplerRatio = 25,
  kAccelSatDopplerAbsRatio = 26,
  kAccelSatBackhaulSeRef = 27,
  kAccelSatVisibleFlag = 28,
  kAccelSatValidFlag = 29,
  kAccelSatLastSelectedFlag = 30,
  kAccelSatLastOutflowSteps = 31,
  kAccelSatTokenDim = 32,
};

enum BwEgoField : int {
  kBwEgoUavQueueSteps = 0,
  kBwEgoUavQueueFill = 1,
  kBwEgoUavLastInflowSteps = 2,
  kBwEgoUavLastOutflowSteps = 3,
  kBwEgoUavLastDropSteps = 4,
  kBwEgoUavServiceEmaSteps = 5,
  kBwEgoUavLocalCostLogRatio = 6,
  kBwEgoUavLastTotalCostLogRatio = 7,
  kBwEgoUavLastWorkloadLog1p = 8,
  kBwEgoUavLastAccessInterferenceLog1p = 9,
  kBwEgoRemainingHorizonFrac = 10,
  kBwEgoDim = 11,
};

enum BwSatTokenField : int {
  kBwSatPrefixBackhaulCapacitySteps = 0,
  kBwSatQueueSteps = 1,
  kBwSatQueueFill = 2,
  kBwSatLastIncomingSteps = 3,
  kBwSatLastProcessedSteps = 4,
  kBwSatLastDropSteps = 5,
  kBwSatServiceEmaSteps = 6,
  kBwSatCostLogRatio = 7,
  kBwSatLastWorkloadLog1p = 8,
  kBwSatTokenDim = 9,
};

enum BwGuTokenField : int {
  kBwGuQueueSteps = 0,
  kBwGuQueueFill = 1,
  kBwGuExpectedArrivalSteps = 2,
  kBwGuLastArrivalSteps = 3,
  kBwGuLastOutflowSteps = 4,
  kBwGuLastDropSteps = 5,
  kBwGuServiceEmaSteps = 6,
  kBwGuLocalCostLogRatio = 7,
  kBwGuLastTotalCostLogRatio = 8,
  kBwGuLastWorkloadLog1p = 9,
  kBwGuAccessRateFullBwRefSteps = 10,
  kBwGuCrossInterferenceMeanLog1p = 11,
  kBwGuCrossInterferenceMaxLog1p = 12,
  kBwGuTokenDim = 13,
};

enum FloatParamIndex : int {
  kFpAccessEtaQuantum = 0,
  kFpAccessRateQuantum = 1,
  kFpFlowBitsQuantum = 2,
  kFpQueueStateQuantum = 3,
  kFpSummaryMetricQuantum = 4,
  kFpAccessGainQuantum = 5,
  kFpAccessPathlossDbQuantum = 6,
  kFpPathlossConstDb = 7,
  kFpCarrierFreq = 8,
  kFpXiLos = 9,
  kFpXiNlos = 10,
  kFpLosA = 11,
  kFpLosB = 12,
  kFpCandidateUavHeight = 13,
  kFpPlThresholdDb = 14,
  kFpCandidateRadius = 15,
  kFpRefLatDeg = 16,
  kFpRefLonDeg = 17,
  kFpEarthRadius = 18,
  kFpSatGeomUavHeight = 19,
  kFpThetaMinRad = 20,
  kFpSatCarrierFreq = 21,
  kFpSpeedOfLight = 22,
  kFpNuMax = 23,
  kFpSatQueueMax = 24,
  kFpSatUavTxPower = 25,
  kFpSatNoiseDensity = 26,
  kFpSubcarrierSpacing = 27,
  kFpSatCandidateElevationWeight = 28,
  kFpSatCandidateQueueWeight = 29,
  kFpSatCandidateSeWeight = 30,
  kFpTau0 = 31,
  kFpMapSize = 32,
  kFpVMax = 33,
  kFpUavEnergyInit = 34,
  kFpQueueMaxGu = 35,
  kFpQueueMaxUav = 36,
  kFpQueueMaxSat = 37,
  kFpLocalEarthRadius = 38,
  kFpSatHeight = 39,
  kFpLocalUavTxPower = 40,
  kFpLocalNoiseDensity = 41,
  kFpLocalNuMax = 42,
  kFpLocalSubcarrierSpacing = 43,
  kFpServiceGapCapSteps = 44,
  kFpAvoidanceAlertFactor = 45,
  kFpDSafe = 46,
  kFpAccelAMax = 47,
  kFpAccelVMax = 48,
  kFpAccelTau0 = 49,
  kFpAccelMapSize = 50,
  kFpAccelDSafe = 51,
  kFpAccelEnergyInit = 52,
  kFpUavOptSpeed = 53,
  kFpEnergySafeThreshold = 54,
  kFpBoundaryMargin = 55,
  kFpBwTau0 = 56,
  kFpBwQueueMaxGu = 57,
  kFpBwQueueMaxUav = 58,
  kFpBwQueueMaxSat = 59,
  kFpServiceGapIncrement = 60,
  kFpServiceGapReliefCoef = 61,
  kFpBwServiceGapCap = 62,
  kFpDeadlineAgeIncrement = 63,
  kFpDeadlineServiceReliefCoef = 64,
  kFpDeadlineAgeCap = 65,
  kFpDeadlineExpireRate = 66,
  kFpBwLinkTau0 = 67,
  kFpBackhaulRateQuantum = 68,
  kFpBSatTotal = 69,
  kFpBSatTotalScale = 70,
  kFpBwUavTxPower = 71,
  kFpBwNoiseDensity = 72,
  kFpPCommLink = 73,
  kFpBwLinkQueueMaxSat = 74,
  kFpRewardTSteps = 75,
  kFpOmegaQ = 76,
  kFpOmegaE = 77,
  kFpEtaService = 78,
  kFpEtaQDelta = 79,
  kFpEtaBatt = 80,
  kFpEtaCrash = 81,
  kFpEtaAccel = 82,
  kFpEtaDrop = 83,
  kFpEtaDropStep = 84,
  kFpEtaDropGu = 85,
  kFpEtaDropUav = 86,
  kFpEtaDropSat = 87,
  kFpRewardWAccess = 88,
  kFpRewardWRelay = 89,
  kFpRewardWPreBacklog = 90,
  kFpRewardWPreDrop = 91,
  kFpRewardWPreServiceGap = 92,
  kFpRewardWPreOverflowRisk = 93,
  kFpBwUavOrbitRadius = 94,
  kFpBwUavOrbitRadiusSq = 95,
  kFpBwSatOrbitRadiusSq = 96,
  kFpBwBackhaulGainConst = 97,
  kFpBwEffectiveBSatTotal = 98,
  kFpBwInvAMax = 99,
  kFpBwDopplerCap = 100,
  kFpBwDopplerRho = 101,
  kFpBwDopplerSigma = 102,
  kFpAccessGuTxPower = 103,
  kFpAccessNoiseDensity = 104,
  kFpAccessBAcc = 105,
  kFpAccessInterferenceQuantum = 106,
  kFpWorkloadEps = 107,
  kFpBwWorkloadEmaDecay = 108,
  kFpEtaThroughputAccess = 109,
  kFpEtaThroughputBackhaul = 110,
  kFpEtaCloseRisk = 111,
  kFpThroughputOnlyAccessCoef = 112,
  kFpThroughputOnlyBackhaulCoef = 113,
  kFpThroughputOnlyGuQueueCoef = 114,
  kFpQueueNormK = 115,
  kFpQueueNormArrivalFloor = 116,
  kFpQueueLogK = 117,
  kFpOmegaQGu = 118,
  kFpOmegaQUav = 119,
  kFpOmegaQSat = 120,
  kFpOmegaQTail = 121,
  kFpQNormTailQ0 = 122,
  kFpTailQSmall = 123,
  kFpTailEtaAccelGain = 124,
  kFpEtaCentroid = 125,
  kFpEtaCentroidFinal = 126,
  kFpEtaCentroidDecaySteps = 127,
  kFpCentroidDistScale = 128,
  kFpCentroidCrossQueueGain = 129,
  kFpCentroidCrossQDeltaGain = 130,
  kFpCentroidCrossCrashGain = 131,
  kFpPFlyBase = 132,
  kFpPFlyCoeff = 133,
  kFpRotorP0 = 134,
  kFpRotorPi = 135,
  kFpRotorUTip = 136,
  kFpRotorV0 = 137,
  kFpRotorD0 = 138,
  kFpRotorRho = 139,
  kFpRotorS = 140,
  kFpRotorArea = 141,
  kFpNrf = 142,
  kFpFlowProxyAuxDelta = 143,
  kFpFlowProxyEps = 144,
  kFpAvoidancePrealertFactor = 145,
  kFpAvoidancePrealertClosingSpeed = 146,
  kFpAvoidancePrealertTtc = 147,
  kFpAvoidancePrealertDistCap = 148,
  kFpCloseRiskCap = 149,
  kFpDangerCloseRiskThresh = 150,
  kFpDangerInterventionThresh = 151,
  kFpAtmLossDb = 152,
  kFpBaselineAccelGain = 153,
  kFpBaselineAssocBonus = 154,
  kFpBaselineRepulseGain = 155,
  kFpBaselineRepulseRadiusFactor = 156,
  kFpBaselineEnergyWeight = 157,
  kFpBaselineEnergyLow = 158,
  kFpBaselineSatSeWeight = 159,
  kFpBaselineSatQueuePenalty = 160,
  kFpBaselineSatLoadPenalty = 161,
  kFpBaselineSatBwReward = 162,
  kFpBaselineSatStayBonus = 163,
  kFpBaselineSatSwitchMargin = 164,
  kFpBaselineClusterStopRadius = 165,
  kFpBaselineClusterSpeedTol = 166,
  kFpBaselineClusterSlowRadius = 167,
  kFpBaselineClusterCruiseSpeed = 168,
  kFpBaselineClusterVelGain = 169,
  kFpSatLogitScale = 170,
  kFpAvoidanceClosingGainCap = 171,
  kFpAvoidanceEta = 172,
  kFpBwWorkloadSatActiveRef = 173,
  kFpAccessNoiseFigureLinear = 174,
  kFpSatNoiseFigureLinear = 175,
  kFpBwNoiseFigureLinear = 176,
  kFpAccessFadingModeCode = 177,
  kFpAccessRicianK = 178,
  kFpRainRate001 = 179,
  kFpRainHeightKm = 180,
  kFpRainStationHeightKm = 181,
  kFpRainLatitudeDeg = 182,
  kFpRainExceedancePct = 183,
  kFpRainPolarizationTiltDeg = 184,
  kFpSafetyShieldBuffer = 185,
  kFpSafetyShieldASafe = 186,
  kFpSafetyShieldStepGain = 187,
  kFpSafetyShieldTol = 188,
  kFpSafetyShieldBrakeRho = 189,
  kFpBaselineLyapunovV = 190,
  kFpBaselineLyapunovUrgencyAlpha = 191,
  kFpBaselineLyapunovDriftWeight = 192,
  kFpBaselineLyapunovActionCost = 193,
  kFpBaselineLyapunovEmaBeta = 194,
  kFpBaselineLyapunovBwTemp = 195,
  kFpBaselineLyapunovBwFloor = 196,
  kFpBaselineLyapunovBwServiceScale = 197,
  kFpBaselineLyapunovSatDriftWeight = 198,
  kFpBaselineLyapunovSatSwitchBias = 199,
  kFpBaselineLyapunovSatAbsSeWeight = 200,
  kFpBaselineLyapunovSatDopplerPenalty = 201,
  kFpTopologyDppBwTemp = 202,
  kFpTopologyDppBwFloor = 203,
  kFpTopologyDppDistPenalty = 204,
  kFpTopologyDppSatQueueGapWeight = 205,
  kFpTopologyDppSatSubsetPenalty = 206,
  kFpTopologyDppSatContentionWeight = 207,
  kFpTopologyDppAccelStepScale = 208,
  kFpTopologyDppAccessWeight = 209,
  kFpTopologyDppBackhaulWeight = 210,
  kFpTopologyDppMobilityWeight = 211,
  kFpTopologyDppAccelCost = 212,
  kFpTopologyDppSmoothness = 213,
  kFpTopologyDppAccelSafetyWeight = 214,
  kFpTopologyDppAccelRoleWeight = 215,
};

enum FloatTensorIndex : int {
  kFStateUavPos = 0,
  kFStateUavVel = 1,
  kFStateUavEnergy = 2,
  kFStateUavQueue = 3,
  kFStateGuPos = 4,
  kFStateGuQueue = 5,
  kFStateSatQueue = 6,
  kFStateSatPos = 7,
  kFStateSatVel = 8,
  kFStateArrivalRef = 9,
  kFStateEffectiveArrivalRate = 10,
  kFStateArrivalBaseScale = 11,
  kFStateGuEma = 12,
  kFStateUavEma = 13,
  kFStateSatEma = 14,
  kFStateLastArrivalRateVec = 15,
  kFStateGuDeadlineSteps = 16,
  kFStateLastGuArrival = 17,
  kFStateLastGuOutflow = 18,
  kFStateUrgencyRisk = 19,
  kFStateDownstreamPressure = 20,
  kFStateServiceGapRisk = 21,
  kFStateDeadlineSlack = 22,
  kFStateDeadlineRisk = 23,
  kFStateServiceGap = 24,
  kFStateDeadlineAge = 25,
  kFStateLastExecAccel = 26,
  kFStateLastPolicyAccel = 27,
  kFStateAvoidanceEtaEff = 28,
  kFStateLastAvoidanceEtaExec = 29,
  kFStateDopplerResidual = 30,
  kFStatePrevQueueSumGu = 31,
  kFStatePrevQueueSumUav = 32,
  kFStatePrevQueueSumSat = 33,
  kFStatePrevQNormActive = 34,
  kFStatePrevGuQueueVec = 35,
  kFStatePrevUavQueueVec = 36,
  kFStatePrevSatQueueVec = 37,
  kFRandomArrivals = 38,
  kFRandomArrivalRates = 39,
  kFRandomFadingGain = 40,
  kFRandomDopplerNoise = 41,
  kFRandomArrivalTape = 42,
  kFRandomArrivalRateTape = 43,
  kFRandomFadingGainTape = 44,
  kFRandomDopplerNoiseTape = 45,
  kFRandomResetGuPosTape = 46,
  kFRandomResetUavPosTape = 47,
  kFRandomResetUavVelTape = 48,
  kFRandomResetGuQueueTape = 49,
  kFRandomResetUavQueueTape = 50,
  kFRandomResetSatQueueTape = 51,
  kFRandomResetArrivalBaseScaleTape = 52,
  kFRandomResetDeadlineStepsTape = 53,
  kFRandomResetDopplerResidualTape = 54,
  kFRandomResetEffectiveArrivalRateTape = 55,
  kFRandomResetArrivalRateVecTape = 56,
  kFRandomResetArrivalRefTape = 57,
  kFRandomResetFollowupArrivalTape = 58,
  kFRandomResetFollowupArrivalRateTape = 59,
  kFStageBase = 60,
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
  kFBwInputBase = 253,
  kFMainSatSubsetSizes = 263,
  kFHistAccelWorld = 264,
  kFHistSatWorld = 270,
  kFHistBwWorld = 276,
  kFHistTerminalWorld = 282,
  kFHistAccelLocal = 288,
  kFHistSatLocal = 293,
  kFHistBwLocal = 297,
  kFHistAccelActions = 300,
  kFHistAccelOldLogprobs = 301,
  kFHistAccelValues = 302,
  kFHistSatOldLogprobs = 303,
  kFHistSatValues = 304,
  kFHistBwActions = 305,
  kFHistBwOldLogprobs = 306,
  kFHistBwValues = 307,
  kFHistBwRewards = 308,
  kFHistBwAccessRewards = 309,
  kFHistBwWeightedWorkloadDeltaRewards = 310,
  kFHistBwWeightedWorkloadLevelRewards = 311,
  kFHistBwGuQueueLevelRewards = 312,
  kFHistBwSystemQueueLevelRewards = 313,
  kFHistBwGuServiceQueueRewards = 314,
  kFHistBwFlowProxyScores = 315,
  kFHistBwFlowProxyMasks = 316,
  kFHistBwFlowProxyDeltas = 317,
  kFHistBwRefActions = 318,
  kFHistBwOldLogprobsPerAgent = 319,
  kFHistDangerTargets = 320,
  kFHistDangerMasks = 321,
  kFHistRewardParts = 322,
  kFMainBwLinkUavEnergy = 366,
  kFMainBwLinkLastEnergyCost = 367,
  kFMainBwLinkRateMatrix = 368,
  kFMainBwLinkSatLoads = 369,
  kFMainBwLinkLastSatScore = 370,
  kFMainBwLinkOverrideUavEnergy = 371,
  kFMainBwLinkOverrideLastEnergyCost = 372,
  kFMainBwLinkOverrideRateMatrix = 373,
  kFMainBwLinkOverrideSatLoads = 374,
  kFMainBwLinkOverrideLastSatScore = 375,
  kFMainBwLinkOverrideActive = 376,
  kFMainBwSatComputeRates = 377,
  kFMainFadingGainUnity = 378,
  kFMainDopplerNoiseZero = 379,
  kFStateLastSatConnectionCounts = 380,
  kFStateGuClusterCenters = 381,
  kFStateGuClusterCounts = 382,
  kFRandomResetGuClusterCentersTape = 383,
  kFRandomResetGuClusterCountsTape = 384,
  kFOrbitPosTable = 385,
  kFOrbitVelTable = 386,
  kFMainNativeActorScratch = 387,
  kFStateGuDrop = 388,
  kFStateUavDrop = 389,
  kFStateSatDrop = 390,
  kFStateLastAccessInterferenceByUav = 391,
  kFStateLastBwFractionByUavGu = 392,
  kFStateLastGuToUavInflowByUav = 393,
  kFStateLastUavToSatOutflowMatrix = 394,
  kFStateLastSelectedMaskByUavSat = 395,
  kFStateLastSatProcessed = 396,
  kFHistAccelWorldGlobalScalars = 397,
  kFHistSatWorldGlobalScalars = 398,
  kFHistBwWorldGlobalScalars = 399,
  kFHistTerminalWorldGlobalScalars = 400,
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
  kFHistAccelLatentActions = 413,
  kFHistSatOldLogprobsPerAgent = 414,
  kFStateHotspotMemberMask = 415,
  kFRandomResetHotspotMemberMaskTape = 416,
  kFHistBwRuntimeStateBase = 417,
  kFHistBwRuntimeCacheBase = 468,
  kFHistBwRuntimeStageBase = 478,
  kFHistAccelRuntimeStateBase = 520,
  kFHistAccelRuntimeStageBase = 571,
  kFHistSatRuntimeStateBase = 613,
  kFHistSatRuntimeStageBase = 664,
  kFLyapunovPressureEma = 706,
  kFLyapunovVirtualQueue = 707,
  kFLyapunovServiceEst = 708,
  kFLyapunovInstantPressure = 709,
};

enum StageFloatField : int {
  kSfEffectiveBSatTotal = 0,
  kSfUavPos = 1,
  kSfUavVel = 2,
  kSfUavEnergy = 3,
  kSfUavQueue = 4,
  kSfGuPos = 5,
  kSfGuQueue = 6,
  kSfSatQueue = 7,
  kSfSatLoads = 8,
  kSfSatPos = 9,
  kSfSatVel = 10,
  kSfBwValidMask = 11,
  kSfCandidateFlag = 12,
  kSfBwValidFlag = 13,
  kSfPrevAssocFlag = 14,
  kSfEtaRefFeature = 15,
  kSfEtaSlots = 16,
  kSfGuProxyFeatures = 17,
  kSfUavAssocUavCost = 18,
  kSfSatCostNorm = 19,
  kSfAccessGainMatrix = 20,
  kSfVisibleFlagAll = 21,
  kSfElevationMatrix = 22,
  kSfUavEcefAll = 23,
  kSfUavVelEcefAll = 24,
  kSfSatPosActive = 25,
  kSfSatVelActive = 26,
  kSfSatQueueActive = 27,
  kSfSatLoadActive = 28,
  kSfSatCostNormActive = 29,
  kSfUsRelPosActive = 30,
  kSfUsRelVelActive = 31,
  kSfUsGainActive = 32,
  kSfUsNuEffActive = 33,
  kSfVisibleFlagActive = 34,
  kSfUsValidFlagActive = 35,
  kSfUsRelPosAll = 36,
  kSfUsRelVelAll = 37,
  kSfUsGainAll = 38,
  kSfUsNuEffAll = 39,
  kSfUsValidFlagAll = 40,
  kSfUsSatQueueAll = 41,
};

enum StageLongField : int {
  kSlStageId = 0,
  kSlAssoc = 1,
  kSlPrevAssociation = 2,
  kSlCandidateIndices = 3,
  kSlSatSelectionMatrix = 4,
  kSlVisibleIds = 5,
  kSlActiveSatIds = 6,
};

enum LongTensorIndex : int {
  kLStateLastSatSelectionMatrix = 0,
  kLStageBase = 1,
  kLLiveSatSubsetMembers = 29,
  kLLiveSatSubsetIndex = 30,
  kLBwInputBase = 31,
  kLMainSatSubsetMembersBase = 36,
  kLMainCandidateSlotIds = 37,
  kLMainCandidateEnvIds = 38,
  kLMainCandidateGuIds = 39,
  kLMainCandidateUavIds = 40,
  kLMainSatAllIds = 41,
  kLMainAccelUavIndexOrder = 42,
  kLMainAccelNeighborIndices = 43,
  kLHistSatSubsetMembers = 44,
  kLHistSatActions = 45,
  kLMainSelectedEnvMapping = 46,
  kLHistAccelWorldSatIds = 47,
  kLHistSatWorldSatIds = 48,
  kLHistBwWorldSatIds = 49,
  kLHistTerminalWorldSatIds = 50,
  kLLiveBwValidCount = 51,
  kLLiveBwLatentCount = 52,
  kLHistBwValidCount = 53,
  kLHistBwLatentCount = 54,
  kLLiveSatCandidateIds = 55,
  kLHistSatCandidateIds = 56,
  kLLiveSatActionIndices = 57,
  kLHistSatActionIndices = 58,
  kLHistBwRuntimeStateBase = 59,
  kLHistBwRuntimeCacheBase = 60,
  kLHistBwRuntimeStageBase = 65,
  kLHistAccelRuntimeStateBase = 72,
  kLHistAccelRuntimeStageBase = 73,
  kLHistSatRuntimeStateBase = 80,
  kLHistSatRuntimeStageBase = 81,
};

enum BoolTensorIndex : int {
  kBStateHotspotMemberMask = 0,
  kBRandomResetHotspotMemberMaskTape = 1,
  kBRandomHotspotMaskTape = 2,
  kBStageBase = 3,
  kBLiveAccelObs0 = 11,
  kBLiveAccelObs1 = 14,
  kBLiveSatObs = 17,
  kBLiveBwObs = 19,
  kBBwInputCandidateMask = 22,
  kBMainAccelOffdiagMask = 23,
  kBMainUavPairUpperMask = 24,
  kBHistAccelWorld = 25,
  kBHistSatWorld = 30,
  kBHistBwWorld = 35,
  kBHistTerminalWorld = 40,
  kBHistAccelLocal = 45,
  kBHistSatLocal = 48,
  kBHistBwLocal = 50,
  kBHistTerminated = 53,
  kBHistTruncated = 54,
  kBHistTerminalNextWorldMask = 55,
  kBLiveSatSubsetMask = 56,
  kBHistSatSubsetMask = 57,
  kBHistBwRuntimeStateBase = 58,
  kBHistBwRuntimeCacheBase = 59,
  kBHistBwRuntimeStageBase = 60,
  kBHistAccelRuntimeStateBase = 62,
  kBHistAccelRuntimeStageBase = 63,
  kBHistSatRuntimeStateBase = 65,
  kBHistSatRuntimeStageBase = 66,
};

enum IntTensorIndex : int {
  kIMarker = 0,
  kIStatePrevAssociation = 1,
  kIStateLastAssociation = 2,
  kIStateLastSatConnectionCounts = 3,
  kIStateHotspotActiveIdx = 4,
  kIStateHotspotSubsetCount = 5,
  kIStateTrafficResetStep = 6,
  kIStateTrafficResetOrdinal = 7,
  kIStateEpisodeIdx = 8,
  kIStateT = 9,
  kIStateGlobalStep = 10,
  kIRandomHotspotActiveAfterTape = 11,
  kIRandomResetFollowupHotspotActiveAfterTape = 12,
  kIRandomResetEpisodeIdxTape = 13,
  kIRandomResetHotspotActiveIdxTape = 14,
  kIRandomResetHotspotSubsetCountTape = 15,
  kIRandomStepTensor = 16,
  kIRandomResetCount = 17,
  kIHistBwRuntimeStateBase = 18,
  kIHistAccelRuntimeStateBase = 27,
  kIHistSatRuntimeStateBase = 36,
};

struct PackedAbi {
  float* f[kMaxFloatTensors];
  int64_t* l[kMaxLongTensors];
  bool* b[kMaxBoolTensors];
  int* i[kMaxIntTensors];
  int64_t f_numel[kMaxFloatTensors];
  int64_t l_numel[kMaxLongTensors];
  int64_t b_numel[kMaxBoolTensors];
  int64_t i_numel[kMaxIntTensors];
  int64_t ip[kMaxIntParams];
  double fp[kMaxFloatParams];
  int nf;
  int nl;
  int nb;
  int ni;
  int nip;
  int nfp;
};

__constant__ PackedAbi cBranchSourceAbi;
__constant__ PackedAbi cBranchTargetAbi;
__constant__ PackedAbi cLiveAbi;

__host__ PackedAbi pack_abi(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params) {
  if (float_tensors.size() > kMaxFloatTensors || long_tensors.size() > kMaxLongTensors ||
      bool_tensors.size() > kMaxBoolTensors || int_tensors.size() > kMaxIntTensors ||
      int_params.size() > kMaxIntParams || float_params.size() > kMaxFloatParams) {
    throw std::runtime_error("native CUDA ABI exceeds compiled tensor/parameter slot capacity.");
  }
  PackedAbi out{};
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

void copy_live_abi_to_symbol(const PackedAbi& abi, cudaStream_t stream) {
  C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(cLiveAbi, &abi, sizeof(PackedAbi), 0, cudaMemcpyHostToDevice, stream));
}

void check_cuda_group(const TensorVec& tensors, at::ScalarType dtype, const char* group_name) {
  for (size_t idx = 0; idx < tensors.size(); ++idx) {
    const at::Tensor& tensor = tensors[idx];
    if (!tensor.defined() || !tensor.is_cuda() || !tensor.is_contiguous() || tensor.scalar_type() != dtype) {
      throw std::runtime_error(std::string("native CUDA ABI tensor group ") + group_name + " contains an invalid tensor.");
    }
  }
}

void check_launch_contract(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    int64_t active_idx,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode) {
  check_cuda_group(float_tensors, at::ScalarType::Float, "float_tensors");
  check_cuda_group(long_tensors, at::ScalarType::Long, "long_tensors");
  check_cuda_group(bool_tensors, at::ScalarType::Bool, "bool_tensors");
  check_cuda_group(int_tensors, at::ScalarType::Int, "int_tensors");
  if (active_idx != 0 && active_idx != 1) {
    throw std::runtime_error("native CUDA ABI active_idx must be 0 or 1.");
  }
  if (int_tensors.empty()) {
    throw std::runtime_error("native CUDA ABI requires at least one int32 marker/state tensor.");
  }
  if (static_cast<int>(int_params.size()) <= kParamBwSourceMode) {
    throw std::runtime_error("native CUDA ABI is missing actor source mode parameters.");
  }
  if (int_params[kParamAccelSourceMode] != accel_source_mode ||
      int_params[kParamSatSourceMode] != sat_source_mode ||
      int_params[kParamBwSourceMode] != bw_source_mode) {
    throw std::runtime_error("native CUDA actor source mode scalars do not match the frozen ABI.");
  }
  for (int64_t mode : {accel_source_mode, sat_source_mode, bw_source_mode}) {
    if (mode < kSourcePolicy || mode > kSourceObservableClusterQueueAware) {
      throw std::runtime_error("native CUDA actor source mode is not supported.");
    }
  }
}

__device__ __forceinline__ int64_t ip(const PackedAbi& a, int idx, int64_t default_value = 0) {
  return (idx >= 0 && idx < a.nip) ? a.ip[idx] : default_value;
}

__device__ __forceinline__ float fp(const PackedAbi& a, int idx, float default_value = 0.0f) {
  return (idx >= 0 && idx < a.nfp) ? static_cast<float>(a.fp[idx]) : default_value;
}

__device__ __forceinline__ bool has_f(const PackedAbi& a, int idx) {
  return idx >= 0 && idx < a.nf && a.f[idx] != nullptr && a.f_numel[idx] > 0;
}

__device__ __forceinline__ bool has_l(const PackedAbi& a, int idx) {
  return idx >= 0 && idx < a.nl && a.l[idx] != nullptr && a.l_numel[idx] > 0;
}

__device__ __forceinline__ bool has_b(const PackedAbi& a, int idx) {
  return idx >= 0 && idx < a.nb && a.b[idx] != nullptr && a.b_numel[idx] > 0;
}

__device__ __forceinline__ bool has_i(const PackedAbi& a, int idx) {
  return idx >= 0 && idx < a.ni && a.i[idx] != nullptr && a.i_numel[idx] > 0;
}

__device__ __forceinline__ void finish_profile_mark(
    float* profile_out,
    int profile_stride,
    int e,
    int segment,
    unsigned long long* last_clock) {
  if (profile_out == nullptr || profile_stride <= segment || last_clock == nullptr) return;
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned long long now = clock64();
    profile_out[e * profile_stride + segment] += static_cast<float>(now - *last_clock);
    *last_clock = now;
  }
}

__device__ __forceinline__ int stage_f(int stage_slot, int field) {
  return kFStageBase + stage_slot * 42 + field;
}

__device__ __forceinline__ int stage_l(int stage_slot, int field) {
  return kLStageBase + stage_slot * 7 + field;
}

__device__ __forceinline__ int stage_b(int stage_slot, int field) {
  return kBStageBase + stage_slot * 2 + field;
}

__device__ __forceinline__ float clampf_device(float value, float lo, float hi) {
  return fminf(fmaxf(value, lo), hi);
}

__device__ __forceinline__ float remaining_horizon_frac_device(const PackedAbi& a, int e) {
  const float t_steps = fmaxf(fp(a, kFpRewardTSteps, 1.0f), 1.0f);
  const float denom = fmaxf(t_steps - 1.0f, 1.0f);
  const float t_now = static_cast<float>(has_i(a, kIStateT) ? a.i[kIStateT][e] : 0);
  return clampf_device((t_steps - 1.0f - t_now) / denom, 0.0f, 1.0f);
}

constexpr float kNormDenomEps = 1.0e-9f;
constexpr float kRuntimeRatioZeroTol = 1.0e-9f;
constexpr float kGeometryDenomEps = 1.0e-9f;
constexpr float kLogRatioEps = 1.0e-12f;
constexpr float kRelativeLogEps = 1.0e-6f;
constexpr float kAngleRadEps = 1.0e-6f;
constexpr float kDynamicsDenomEps = 1.0e-6f;
constexpr float kTrigDenomEps = 1.0e-6f;
constexpr float kFrequencyGhzEps = 1.0e-6f;
constexpr float kMinRainReductionFactor = 1.0e-3f;
constexpr float kProbabilityEqualityTol = 1.0e-12f;
constexpr float kDefaultNoiseDensity = 1.0e-9f;
constexpr float kDefaultRateQuantum = 1.0e-6f;

__device__ __forceinline__ float positive_config_scale(float value) {
  return fmaxf(value, kNormDenomEps);
}

__device__ __forceinline__ float require_positive_reward_ref(float value) {
  return (isfinite(value) && value > 0.0f) ? value : nanf("");
}

__device__ __forceinline__ float geometry_denominator(float value) {
  return fmaxf(value, kGeometryDenomEps);
}

__device__ __forceinline__ float dynamics_denominator(float value) {
  return fmaxf(value, kDynamicsDenomEps);
}

__device__ __forceinline__ void project_l2_device(float* x, float* y, float max_norm) {
  const float limit = fmaxf(max_norm, 0.0f);
  if (limit <= 0.0f) {
    *x = 0.0f;
    *y = 0.0f;
    return;
  }
  const float norm = sqrtf((*x) * (*x) + (*y) * (*y));
  if (norm > limit) {
    const float scale = limit / dynamics_denominator(norm);
    *x *= scale;
    *y *= scale;
  }
}

__device__ __forceinline__ float log_argument(float value) {
  return fmaxf(value, kRelativeLogEps);
}

__device__ __forceinline__ float log1p_nonnegative(float value) {
  return log1pf(fmaxf(value, 0.0f));
}

__device__ __forceinline__ float positive_coeff(float value) {
  return fmaxf(value, kLogRatioEps);
}

__device__ __forceinline__ float ratio_or_zero(float num, float den) {
  return fabsf(den) > kRuntimeRatioZeroTol ? num / den : 0.0f;
}

__device__ __forceinline__ float divide_or_zero(float num, float den) {
  return fabsf(den) > kNormDenomEps ? num / den : 0.0f;
}

__device__ __forceinline__ float safe_div(float num, float den) {
  return divide_or_zero(num, den);
}

__device__ __forceinline__ int active_width(const PackedAbi& a) {
  const int u = static_cast<int>(ip(a, kParamNumUav));
  const int s = static_cast<int>(ip(a, kParamNumSat));
  const int visible = static_cast<int>(ip(a, kParamVisibleSatsMax));
  const int width = visible * max(u, 0);
  return min(max(width, 0), max(s, 0));
}

constexpr int kCriticGuDim = 15;
constexpr int kCriticUavDim = 19;
constexpr int kCriticSatDim = 18;
constexpr int kCriticUgDim = 9;
constexpr int kCriticUsDim = 18;
constexpr int kCriticUuDim = 11;
constexpr int kCriticGlobalDim = 27;

enum CriticGuField : int {
  kGuX = 0,
  kGuY = 1,
  kGuQueueSteps = 2,
  kGuQueueFill = 3,
  kGuExpectedArrivalSteps = 4,
  kGuLastArrivalSteps = 5,
  kGuLastOutflowSteps = 6,
  kGuLastDropSteps = 7,
  kGuServiceEmaSteps = 8,
  kGuLocalCostLogRatio = 9,
  kGuLastTotalCostLogRatio = 10,
  kGuPrefixTotalCostLogRatio = 11,
  kGuPrefixCostKnown = 12,
  kGuLastWorkloadLog1p = 13,
  kGuPrefixWorkloadLog1p = 14,
};

enum CriticUavField : int {
  kUavX = 0,
  kUavY = 1,
  kUavVx = 2,
  kUavVy = 3,
  kUavEnergy = 4,
  kUavQueueSteps = 5,
  kUavQueueFill = 6,
  kUavLastInflowSteps = 7,
  kUavLastOutflowSteps = 8,
  kUavLastDropSteps = 9,
  kUavServiceEmaSteps = 10,
  kUavLocalCostLogRatio = 11,
  kUavLastTotalCostLogRatio = 12,
  kUavPrefixTotalCostLogRatio = 13,
  kUavPrefixCostKnown = 14,
  kUavLastWorkloadLog1p = 15,
  kUavPrefixWorkloadLog1p = 16,
  kUavPrefixBwValidCountFrac = 17,
  kUavLastAccessInterferenceLog1p = 18,
};

enum CriticSatField : int {
  kSatX = 0,
  kSatY = 1,
  kSatZ = 2,
  kSatVx = 3,
  kSatVy = 4,
  kSatVz = 5,
  kSatQueueSteps = 6,
  kSatQueueFill = 7,
  kSatLastIncomingSteps = 8,
  kSatLastProcessedSteps = 9,
  kSatLastDropSteps = 10,
  kSatServiceEmaSteps = 11,
  kSatCostLogRatio = 12,
  kSatLastWorkloadLog1p = 13,
  kSatPrefixSelectedLoadFrac = 14,
  kSatPrefixLoadKnown = 15,
  kSatLastSelectedLoadFrac = 16,
  kSatProcCapacitySteps = 17,
};

enum CriticUgField : int {
  kUgRelX = 0,
  kUgRelY = 1,
  kUgHorizontalDist = 2,
  kUgElevationNorm = 3,
  kUgAccessSeRef = 4,
  kUgLastBwFraction = 5,
  kUgLastServedFlag = 6,
  kUgPrefixBwValidFlag = 7,
  kUgPrefixBwValidKnown = 8,
};

enum CriticUsField : int {
  kUsRelX = 0,
  kUsRelY = 1,
  kUsRelZ = 2,
  kUsRelVx = 3,
  kUsRelVy = 4,
  kUsRelVz = 5,
  kUsRadialVelocityNorm = 6,
  kUsRangeNorm = 7,
  kUsElevationNorm = 8,
  kUsDopplerNorm = 9,
  kUsDopplerMargin = 10,
  kUsBackhaulSeRef = 11,
  kUsVisibleFlag = 12,
  kUsValidFlag = 13,
  kUsLastSelectedFlag = 14,
  kUsPrefixSelectedFlag = 15,
  kUsPrefixSelectedKnown = 16,
  kUsPrefixBackhaulCapacitySteps = 17,
};

enum CriticUuField : int {
  kUuRelX = 0,
  kUuRelY = 1,
  kUuRelVx = 2,
  kUuRelVy = 3,
  kUuDistNorm = 4,
  kUuClosingSpeedNorm = 5,
  kUuAlertFlag = 6,
  kUuUnsafeFlag = 7,
  kUuLastSharedSatFrac = 8,
  kUuPrefixSharedSatFrac = 9,
  kUuPrefixSharedSatKnown = 10,
};

enum CriticGlobalField : int {
  kGlobalTotalGuQueueSteps = 0,
  kGlobalTotalUavQueueSteps = 1,
  kGlobalTotalSatQueueSteps = 2,
  kGlobalTotalGuDropSteps = 3,
  kGlobalTotalUavDropSteps = 4,
  kGlobalTotalSatDropSteps = 5,
  kGlobalTotalExpectedArrivalSteps = 6,
  kGlobalTotalLastGuOutflowSteps = 7,
  kGlobalTotalLastUavOutflowSteps = 8,
  kGlobalTotalLastSatProcessedSteps = 9,
  kGlobalTotalLastWeightedWorkloadSteps = 10,
  kGlobalTotalPrefixWeightedWorkloadSteps = 11,
  kGlobalPrefixWorkloadKnown = 12,
  kGlobalLastInterferenceMean = 13,
  kGlobalLastInterferenceMax = 14,
  kGlobalSelectedSatLoadMean = 15,
  kGlobalSelectedSatLoadMax = 16,
  kGlobalSelectedSatLoadKnown = 17,
  kGlobalLastSelectedSatLoadMean = 18,
  kGlobalLastSelectedSatLoadMax = 19,
  kGlobalNonTokenSatCountFrac = 20,
  kGlobalNonTokenSatQueueSteps = 21,
  kGlobalNonTokenSatDropSteps = 22,
  kGlobalNonTokenSatLastProcessedSteps = 23,
  kGlobalNonTokenSatWorkloadSteps = 24,
  kGlobalNonTokenSatDropWorkloadSteps = 25,
  kGlobalRemainingHorizonFrac = 26,
};

__device__ __forceinline__ int world_global_f_index(int world_f) {
  if (world_f == kFHistAccelWorld) return kFHistAccelWorldGlobalScalars;
  if (world_f == kFHistSatWorld) return kFHistSatWorldGlobalScalars;
  if (world_f == kFHistBwWorld) return kFHistBwWorldGlobalScalars;
  if (world_f == kFHistTerminalWorld) return kFHistTerminalWorldGlobalScalars;
  return -1;
}

__device__ __forceinline__ int world_sat_ids_l_index(int world_f) {
  if (world_f == kFHistAccelWorld) return kLHistAccelWorldSatIds;
  if (world_f == kFHistSatWorld) return kLHistSatWorldSatIds;
  if (world_f == kFHistBwWorld) return kLHistBwWorldSatIds;
  if (world_f == kFHistTerminalWorld) return kLHistTerminalWorldSatIds;
  return -1;
}

__device__ __forceinline__ int sat_visible_width(const PackedAbi& a) {
  return static_cast<int>(ip(a, kParamSatVisibleWidth));
}

__device__ __forceinline__ int accel_sat_width(const PackedAbi& a) {
  return static_cast<int>(ip(a, kParamAccelSatWidth));
}

__device__ __forceinline__ int row_local(const PackedAbi& a, int e, int u) {
  return e * static_cast<int>(ip(a, kParamNumUav)) + u;
}

__device__ __forceinline__ int hist_env_row(const PackedAbi& a, int slot, int e) {
  return slot * static_cast<int>(ip(a, kParamNumEnvs)) + e;
}

__device__ __forceinline__ int hist_local_row(const PackedAbi& a, int slot, int e, int u) {
  const int ecount = static_cast<int>(ip(a, kParamNumEnvs));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  return slot * ecount * ucount + e * ucount + u;
}

__device__ __forceinline__ bool history_snapshots_enabled(const PackedAbi& a) {
  return ip(a, kParamHistorySnapshotsEnabled, 1) != 0;
}

__device__ __forceinline__ int row_width_or_zero(int64_t numel, int rows) {
  return rows > 0 && numel > 0 ? static_cast<int>(numel / rows) : 0;
}

__device__ void copy_float_row_parallel(
    const PackedAbi& a,
    int dst_tensor,
    int src_tensor,
    int dst_row,
    int src_row,
    int dst_rows,
    int src_rows) {
  if (!has_f(a, dst_tensor) || !has_f(a, src_tensor)) return;
  const int dst_width = row_width_or_zero(a.f_numel[dst_tensor], dst_rows);
  const int src_width = row_width_or_zero(a.f_numel[src_tensor], src_rows);
  if (dst_width != src_width) return;
  const int width = dst_width;
  if (width <= 0) return;
  for (int idx = threadIdx.x; idx < width; idx += blockDim.x) {
    a.f[dst_tensor][static_cast<int64_t>(dst_row) * dst_width + idx] =
        a.f[src_tensor][static_cast<int64_t>(src_row) * src_width + idx];
  }
}

__device__ void copy_long_row_parallel(
    const PackedAbi& a,
    int dst_tensor,
    int src_tensor,
    int dst_row,
    int src_row,
    int dst_rows,
    int src_rows) {
  if (!has_l(a, dst_tensor) || !has_l(a, src_tensor)) return;
  const int dst_width = row_width_or_zero(a.l_numel[dst_tensor], dst_rows);
  const int src_width = row_width_or_zero(a.l_numel[src_tensor], src_rows);
  if (dst_width != src_width) return;
  const int width = dst_width;
  if (width <= 0) return;
  for (int idx = threadIdx.x; idx < width; idx += blockDim.x) {
    a.l[dst_tensor][static_cast<int64_t>(dst_row) * dst_width + idx] =
        a.l[src_tensor][static_cast<int64_t>(src_row) * src_width + idx];
  }
}

__device__ void copy_bool_row_parallel(
    const PackedAbi& a,
    int dst_tensor,
    int src_tensor,
    int dst_row,
    int src_row,
    int dst_rows,
    int src_rows) {
  if (!has_b(a, dst_tensor) || !has_b(a, src_tensor)) return;
  const int dst_width = row_width_or_zero(a.b_numel[dst_tensor], dst_rows);
  const int src_width = row_width_or_zero(a.b_numel[src_tensor], src_rows);
  if (dst_width != src_width) return;
  const int width = dst_width;
  if (width <= 0) return;
  for (int idx = threadIdx.x; idx < width; idx += blockDim.x) {
    a.b[dst_tensor][static_cast<int64_t>(dst_row) * dst_width + idx] =
        a.b[src_tensor][static_cast<int64_t>(src_row) * src_width + idx];
  }
}

__device__ void copy_int_row_parallel(
    const PackedAbi& a,
    int dst_tensor,
    int src_tensor,
    int dst_row,
    int src_row,
    int dst_rows,
    int src_rows) {
  if (!has_i(a, dst_tensor) || !has_i(a, src_tensor)) return;
  const int dst_width = row_width_or_zero(a.i_numel[dst_tensor], dst_rows);
  const int src_width = row_width_or_zero(a.i_numel[src_tensor], src_rows);
  if (dst_width != src_width) return;
  const int width = dst_width;
  if (width <= 0) return;
  for (int idx = threadIdx.x; idx < width; idx += blockDim.x) {
    a.i[dst_tensor][static_cast<int64_t>(dst_row) * dst_width + idx] =
        a.i[src_tensor][static_cast<int64_t>(src_row) * src_width + idx];
  }
}

__device__ void copy_float_row_between_parallel(
    const PackedAbi& dst,
    int dst_tensor,
    const PackedAbi& src,
    int src_tensor,
    int dst_row,
    int src_row,
    int dst_rows,
    int src_rows) {
  if (!has_f(dst, dst_tensor) || !has_f(src, src_tensor)) return;
  const int dst_width = row_width_or_zero(dst.f_numel[dst_tensor], dst_rows);
  const int src_width = row_width_or_zero(src.f_numel[src_tensor], src_rows);
  if (dst_width != src_width) return;
  const int width = dst_width;
  if (width <= 0) return;
  for (int idx = threadIdx.x; idx < width; idx += blockDim.x) {
    dst.f[dst_tensor][static_cast<int64_t>(dst_row) * dst_width + idx] =
        src.f[src_tensor][static_cast<int64_t>(src_row) * src_width + idx];
  }
}

__device__ void copy_long_row_between_parallel(
    const PackedAbi& dst,
    int dst_tensor,
    const PackedAbi& src,
    int src_tensor,
    int dst_row,
    int src_row,
    int dst_rows,
    int src_rows) {
  if (!has_l(dst, dst_tensor) || !has_l(src, src_tensor)) return;
  const int dst_width = row_width_or_zero(dst.l_numel[dst_tensor], dst_rows);
  const int src_width = row_width_or_zero(src.l_numel[src_tensor], src_rows);
  if (dst_width != src_width) return;
  const int width = dst_width;
  if (width <= 0) return;
  for (int idx = threadIdx.x; idx < width; idx += blockDim.x) {
    dst.l[dst_tensor][static_cast<int64_t>(dst_row) * dst_width + idx] =
        src.l[src_tensor][static_cast<int64_t>(src_row) * src_width + idx];
  }
}

__device__ void copy_bool_row_between_parallel(
    const PackedAbi& dst,
    int dst_tensor,
    const PackedAbi& src,
    int src_tensor,
    int dst_row,
    int src_row,
    int dst_rows,
    int src_rows) {
  if (!has_b(dst, dst_tensor) || !has_b(src, src_tensor)) return;
  const int dst_width = row_width_or_zero(dst.b_numel[dst_tensor], dst_rows);
  const int src_width = row_width_or_zero(src.b_numel[src_tensor], src_rows);
  if (dst_width != src_width) return;
  const int width = dst_width;
  if (width <= 0) return;
  for (int idx = threadIdx.x; idx < width; idx += blockDim.x) {
    dst.b[dst_tensor][static_cast<int64_t>(dst_row) * dst_width + idx] =
        src.b[src_tensor][static_cast<int64_t>(src_row) * src_width + idx];
  }
}

__device__ void copy_int_row_between_parallel(
    const PackedAbi& dst,
    int dst_tensor,
    const PackedAbi& src,
    int src_tensor,
    int dst_row,
    int src_row,
    int dst_rows,
    int src_rows) {
  if (!has_i(dst, dst_tensor) || !has_i(src, src_tensor)) return;
  const int dst_width = row_width_or_zero(dst.i_numel[dst_tensor], dst_rows);
  const int src_width = row_width_or_zero(src.i_numel[src_tensor], src_rows);
  if (dst_width != src_width) return;
  const int width = dst_width;
  if (width <= 0) return;
  for (int idx = threadIdx.x; idx < width; idx += blockDim.x) {
    dst.i[dst_tensor][static_cast<int64_t>(dst_row) * dst_width + idx] =
        src.i[src_tensor][static_cast<int64_t>(src_row) * src_width + idx];
  }
}

__device__ float state_arrival_rate(const PackedAbi& a, int e, int g, int slot) {
  const int ecount = static_cast<int>(ip(a, kParamNumEnvs));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  if (has_f(a, kFRandomResetFollowupArrivalRateTape) &&
      has_i(a, kIStateTrafficResetStep) &&
      has_i(a, kIStateTrafficResetOrdinal)) {
    const int reset_step = a.i[kIStateTrafficResetStep][e];
    const int reset_ordinal = a.i[kIStateTrafficResetOrdinal][e];
    if (reset_step >= 0 && reset_ordinal >= 0 && reset_step < slot) {
      const int cap = max(static_cast<int>(ip(a, kParamHistoryCapacity, 1)), 1);
      const int64_t denom = max(static_cast<int64_t>(cap) * ecount * max(gu, 1), static_cast<int64_t>(1));
      const int reset_rows = max(static_cast<int>(a.f_numel[kFRandomResetFollowupArrivalRateTape] / denom), 1);
      const int reset_idx = min(max(reset_ordinal, 0), reset_rows - 1);
      const int step_idx = min(max(slot - reset_step - 1, 0), cap - 1);
      const int64_t idx = (((static_cast<int64_t>(reset_idx) * cap + step_idx) * ecount + e) * gu) + g;
      if (idx >= 0 && idx < a.f_numel[kFRandomResetFollowupArrivalRateTape]) {
        return fmaxf(a.f[kFRandomResetFollowupArrivalRateTape][idx], 0.0f);
      }
    }
  }
  if (has_f(a, kFRandomArrivalRateTape)) {
    const int64_t idx = (static_cast<int64_t>(slot) * ecount + e) * gu + g;
    if (idx >= 0 && idx < a.f_numel[kFRandomArrivalRateTape]) {
      return fmaxf(a.f[kFRandomArrivalRateTape][idx], 0.0f);
    }
  }
  if (has_f(a, kFStateLastArrivalRateVec)) {
    return fmaxf(a.f[kFStateLastArrivalRateVec][e * gu + g], 0.0f);
  }
  if (has_f(a, kFStateEffectiveArrivalRate)) {
    return fmaxf(a.f[kFStateEffectiveArrivalRate][e], 0.0f);
  }
  return 0.0f;
}

__device__ float state_arrival_bits(const PackedAbi& a, int e, int g, int slot) {
  const int ecount = static_cast<int>(ip(a, kParamNumEnvs));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  if (has_f(a, kFRandomResetFollowupArrivalTape) &&
      has_i(a, kIStateTrafficResetStep) &&
      has_i(a, kIStateTrafficResetOrdinal)) {
    const int reset_step = a.i[kIStateTrafficResetStep][e];
    const int reset_ordinal = a.i[kIStateTrafficResetOrdinal][e];
    if (reset_step >= 0 && reset_ordinal >= 0 && reset_step < slot) {
      const int cap = max(static_cast<int>(ip(a, kParamHistoryCapacity, 1)), 1);
      const int64_t denom = max(static_cast<int64_t>(cap) * ecount * max(gu, 1), static_cast<int64_t>(1));
      const int reset_rows = max(static_cast<int>(a.f_numel[kFRandomResetFollowupArrivalTape] / denom), 1);
      const int reset_idx = min(max(reset_ordinal, 0), reset_rows - 1);
      const int step_idx = min(max(slot - reset_step - 1, 0), cap - 1);
      const int64_t idx = (((static_cast<int64_t>(reset_idx) * cap + step_idx) * ecount + e) * gu) + g;
      if (idx >= 0 && idx < a.f_numel[kFRandomResetFollowupArrivalTape]) {
        return fmaxf(a.f[kFRandomResetFollowupArrivalTape][idx], 0.0f);
      }
    }
  }
  if (has_f(a, kFRandomArrivalTape)) {
    const int64_t idx = (static_cast<int64_t>(slot) * ecount + e) * gu + g;
    if (idx >= 0 && idx < a.f_numel[kFRandomArrivalTape]) {
      return fmaxf(a.f[kFRandomArrivalTape][idx], 0.0f);
    }
  }
  return state_arrival_rate(a, e, g, slot);
}

__device__ bool hotspot_active_after_for_step(const PackedAbi& a, int e, int slot, int* out_value) {
  const int ecount = static_cast<int>(ip(a, kParamNumEnvs));
  if (has_i(a, kIRandomResetFollowupHotspotActiveAfterTape) &&
      has_i(a, kIStateTrafficResetStep) &&
      has_i(a, kIStateTrafficResetOrdinal)) {
    const int reset_step = a.i[kIStateTrafficResetStep][e];
    const int reset_ordinal = a.i[kIStateTrafficResetOrdinal][e];
    if (reset_step >= 0 && reset_ordinal >= 0 && reset_step < slot) {
      const int cap = max(static_cast<int>(ip(a, kParamHistoryCapacity, 1)), 1);
      const int64_t denom = max(static_cast<int64_t>(cap) * ecount, static_cast<int64_t>(1));
      const int reset_rows = max(static_cast<int>(a.i_numel[kIRandomResetFollowupHotspotActiveAfterTape] / denom), 1);
      const int reset_idx = min(max(reset_ordinal, 0), reset_rows - 1);
      const int step_idx = min(max(slot - reset_step - 1, 0), cap - 1);
      const int64_t idx = (static_cast<int64_t>(reset_idx) * cap + step_idx) * ecount + e;
      if (idx >= 0 && idx < a.i_numel[kIRandomResetFollowupHotspotActiveAfterTape]) {
        *out_value = a.i[kIRandomResetFollowupHotspotActiveAfterTape][idx];
        return true;
      }
    }
  }
  if (has_i(a, kIRandomHotspotActiveAfterTape)) {
    const int64_t idx = static_cast<int64_t>(slot) * ecount + e;
    if (idx >= 0 && idx < a.i_numel[kIRandomHotspotActiveAfterTape]) {
      *out_value = a.i[kIRandomHotspotActiveAfterTape][idx];
      return true;
    }
  }
  return false;
}

__device__ void copy_random_step_tapes_parallel(const PackedAbi& a, int step, int e) {
  const int ecount = static_cast<int>(ip(a, kParamNumEnvs));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  if (step < 0 || e < 0 || e >= ecount) return;
  if (has_f(a, kFRandomFadingGain)) {
    const int64_t per_step = static_cast<int64_t>(ecount) * gu * ucount;
    const int64_t base = static_cast<int64_t>(step) * per_step + static_cast<int64_t>(e) * gu * ucount;
    for (int idx = threadIdx.x; idx < gu * ucount; idx += blockDim.x) {
      const int64_t src = base + idx;
      const float value = (has_f(a, kFRandomFadingGainTape) && src >= 0 && src < a.f_numel[kFRandomFadingGainTape])
          ? a.f[kFRandomFadingGainTape][src]
          : 1.0f;
      a.f[kFRandomFadingGain][static_cast<int64_t>(e) * gu * ucount + idx] = value;
    }
  }
  if (has_f(a, kFRandomDopplerNoise)) {
    const int64_t per_step = static_cast<int64_t>(ecount) * ucount * sat;
    const int64_t base = static_cast<int64_t>(step) * per_step + static_cast<int64_t>(e) * ucount * sat;
    for (int idx = threadIdx.x; idx < ucount * sat; idx += blockDim.x) {
      const int64_t src = base + idx;
      const float value = (has_f(a, kFRandomDopplerNoiseTape) && src >= 0 && src < a.f_numel[kFRandomDopplerNoiseTape])
          ? a.f[kFRandomDopplerNoiseTape][src]
          : 0.0f;
      a.f[kFRandomDopplerNoise][static_cast<int64_t>(e) * ucount * sat + idx] = value;
    }
  }
}

__device__ int nearest_uav_for_gu(const PackedAbi& a, int stage_slot, int e, int g) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  float best_dist2 = 3.402823466e38f;
  int best_u = 0;
  const float gx = a.f[stage_f(stage_slot, kSfGuPos)][(e * static_cast<int>(ip(a, kParamNumGu)) + g) * 2 + 0];
  const float gy = a.f[stage_f(stage_slot, kSfGuPos)][(e * static_cast<int>(ip(a, kParamNumGu)) + g) * 2 + 1];
  for (int u = 0; u < ucount; ++u) {
    const float ux = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 0];
    const float uy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 1];
    const float dx = gx - ux;
    const float dy = gy - uy;
    const float dist2 = dx * dx + dy * dy;
    if (dist2 < best_dist2) {
      best_dist2 = dist2;
      best_u = u;
    }
  }
  return best_u;
}

__device__ float simple_access_gain(const PackedAbi& a, float dx, float dy) {
  const float h = fmaxf(fp(a, kFpCandidateUavHeight, 1.0f), 1.0f);
  const float d = sqrtf(fmaxf(dx * dx + dy * dy + h * h, 1.0e-18f));
  const float phi_deg = asinf(clampf_device(h / geometry_denominator(d), -1.0f, 1.0f)) * 57.29577951308232f;
  const float los_a = fp(a, kFpLosA, 9.61f);
  const float los_b = fp(a, kFpLosB, 0.16f);
  const float p_los = 1.0f / (1.0f + los_a * expf(-los_b * (phi_deg - los_a)));
  const float pl_base = fp(a, kFpPathlossConstDb, 32.44f) +
      20.0f * log10f(fmaxf(fp(a, kFpCarrierFreq, 2.0e9f), 1.0f) / 1.0e9f);
  const float pl_los = pl_base + fp(a, kFpXiLos, 1.0f) + 20.0f * log10f(geometry_denominator(d));
  const float pl_nlos = pl_base + fp(a, kFpXiNlos, 20.0f) + 20.0f * log10f(geometry_denominator(d));
  const bool free_space = ip(a, kParamPathlossMode) == 1;
  const float pl_db = free_space ? pl_los : (p_los * pl_los + (1.0f - p_los) * pl_nlos);
  const float raw = powf(10.0f, -pl_db / 10.0f);
  const float quantum = fmaxf(fp(a, kFpAccessGainQuantum, 0.0f), 0.0f);
  if (quantum > 0.0f) {
    return rintf(raw / quantum) * quantum;
  }
  return raw;
}

__device__ __forceinline__ float access_spectral_efficiency_device(const PackedAbi& a, float snr);
__device__ __forceinline__ float quantize_device(float value, float quantum);

__device__ float simple_eta_from_gain(const PackedAbi& a, float gain) {
  const float bw = fmaxf(fp(a, kFpAccessBAcc, 0.0f), 0.0f);
  if (bw <= 0.0f) return 0.0f;
  const float denom = fp(a, kFpAccessNoiseDensity, fp(a, kFpLocalNoiseDensity, kDefaultNoiseDensity)) *
      fp(a, kFpAccessNoiseFigureLinear, 1.0f) * bw;
  if (denom <= 0.0f) return 0.0f;
  const float snr = fmaxf(fp(a, kFpAccessGuTxPower, fp(a, kFpLocalUavTxPower, 1.0f)), 0.0f) * gain / denom;
  return quantize_device(access_spectral_efficiency_device(a, snr), fp(a, kFpAccessEtaQuantum, kDefaultRateQuantum));
}

__device__ __forceinline__ float quantize_device(float value, float quantum) {
  return quantum > 0.0f ? rintf(value / quantum) * quantum : value;
}

__device__ __forceinline__ float spectral_efficiency_device(float snr) {
  return log2f(1.0f + fmaxf(snr, 0.0f));
}

__device__ float rician_ergodic_spectral_efficiency_device(const PackedAbi& a, float snr) {
  const float snr_safe = fmaxf(snr, 0.0f);
  if (snr_safe <= 0.0f) return 0.0f;
  const float k = fmaxf(fp(a, kFpAccessRicianK, 0.0f), 0.0f);
  const float sigma = sqrtf(1.0f / (2.0f * (k + 1.0f)));
  const float mean_real = sqrtf(k / (k + 1.0f));
  const float nodes[16] = {
      -4.688738939305819f, -3.869447904860123f, -3.176999161979957f, -2.546202157847481f,
      -1.951787990916254f, -1.380258539198881f, -0.822951449144656f, -0.273481046138152f,
      0.273481046138152f, 0.822951449144656f, 1.380258539198881f, 1.951787990916254f,
      2.546202157847481f, 3.176999161979957f, 3.869447904860123f, 4.688738939305819f};
  const float weights[16] = {
      0.000000000265481f, 0.000000232098084f, 0.000027118600925f, 0.000932284008624f,
      0.012880311535510f, 0.083810041398986f, 0.280647458528534f, 0.507929479016614f,
      0.507929479016614f, 0.280647458528534f, 0.083810041398986f, 0.012880311535510f,
      0.000932284008624f, 0.000027118600925f, 0.000000232098084f, 0.000000000265481f};
  float acc = 0.0f;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float real = mean_real + sigma * 1.4142135623730951f * nodes[i];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float imag = sigma * 1.4142135623730951f * nodes[j];
      const float gain = real * real + imag * imag;
      acc += weights[i] * weights[j] * log2f(1.0f + snr_safe * gain);
    }
  }
  return acc * 0.3183098861837907f;
}

__device__ __forceinline__ float access_spectral_efficiency_device(const PackedAbi& a, float snr) {
  const float mode = fp(a, kFpAccessFadingModeCode, 0.0f);
  return (mode > 0.5f && mode < 1.5f)
      ? rician_ergodic_spectral_efficiency_device(a, snr)
      : spectral_efficiency_device(snr);
}

__device__ float p838_curve_device(float log_f, const float* aa, const float* bb, const float* cc, int n) {
  float out = 0.0f;
  for (int i = 0; i < n; ++i) {
    const float ratio = (log_f - bb[i]) / fabsf(cc[i]);
    out += aa[i] * expf(-(ratio * ratio));
  }
  return out;
}

__device__ void rain_frequency_terms_device(float carrier_freq_hz, float* k_h, float* k_v, float* alpha_h, float* alpha_v) {
  const float freq_ghz = fmaxf(carrier_freq_hz / 1.0e9f, kFrequencyGhzEps);
  const float log_f = log10f(freq_ghz);
  const float a_kh[4] = {-5.33980f, -0.35351f, -0.23789f, -0.94158f};
  const float b_kh[4] = {-0.10008f, 1.26970f, 0.86036f, 0.64552f};
  const float c_kh[4] = {1.13098f, 0.45400f, 0.15354f, 0.16817f};
  const float a_kv[4] = {-3.80595f, -3.44965f, -0.39902f, 0.50167f};
  const float b_kv[4] = {0.56934f, -0.22911f, 0.73042f, 1.07319f};
  const float c_kv[4] = {0.81061f, 0.51059f, 0.11899f, 0.27195f};
  const float a_ah[5] = {-0.14318f, 0.29591f, 0.32177f, -5.37610f, 16.1721f};
  const float b_ah[5] = {1.82442f, 0.77564f, 0.63773f, -0.96230f, -3.29980f};
  const float c_ah[5] = {-0.55187f, 0.19822f, 0.13164f, 1.47828f, 3.43990f};
  const float a_av[5] = {-0.07771f, 0.56727f, -0.20238f, -48.2991f, 48.5833f};
  const float b_av[5] = {2.33840f, 0.95545f, 1.14520f, 0.791669f, 0.791459f};
  const float c_av[5] = {-0.76284f, 0.54039f, 0.26809f, 0.116226f, 0.116479f};
  *k_h = powf(10.0f, p838_curve_device(log_f, a_kh, b_kh, c_kh, 4) - 0.18961f * log_f + 0.71147f);
  *k_v = powf(10.0f, p838_curve_device(log_f, a_kv, b_kv, c_kv, 4) - 0.16398f * log_f + 0.63297f);
  *alpha_h = p838_curve_device(log_f, a_ah, b_ah, c_ah, 5) + 0.67849f * log_f - 1.95537f;
  *alpha_v = p838_curve_device(log_f, a_av, b_av, c_av, 5) - 0.053739f * log_f + 0.83433f;
}

__device__ float rain_attenuation_db_device(const PackedAbi& a, float elevation) {
  const float rain_rate = fmaxf(fp(a, kFpRainRate001, 0.0f), 0.0f);
  const float rain_height = fp(a, kFpRainHeightKm, 5.0f);
  const float station_height = fp(a, kFpRainStationHeightKm, 0.0f);
  if (rain_rate <= 0.0f || rain_height <= station_height) return 0.0f;
  const float theta = fmaxf(elevation, kAngleRadEps);
  const float theta_deg = theta * 57.29577951308232f;
  const float sin_el = fmaxf(sinf(theta), kTrigDenomEps);
  const float cos_el = cosf(theta);
  const float freq_hz = fmaxf(fp(a, kFpSatCarrierFreq, fp(a, kFpCarrierFreq, 2.0e9f)), 1.0f);
  const float freq_ghz = fmaxf(freq_hz / 1.0e9f, kFrequencyGhzEps);
  float k_h = 0.0f, k_v = 0.0f, alpha_h = 0.0f, alpha_v = 0.0f;
  rain_frequency_terms_device(freq_hz, &k_h, &k_v, &alpha_h, &alpha_v);
  const float tau = fp(a, kFpRainPolarizationTiltDeg, 45.0f) * 0.017453292519943295f;
  const float cos_term = cos_el * cos_el * cosf(2.0f * tau);
  const float k = positive_coeff(0.5f * (k_h + k_v + (k_h - k_v) * cos_term));
  const float alpha = 0.5f *
      (k_h * alpha_h + k_v * alpha_v + (k_h * alpha_h - k_v * alpha_v) * cos_term) / k;
  const float gamma_r = k * powf(rain_rate, alpha);
  const float delta_h = fmaxf(rain_height - station_height, 0.0f);
  const float l_s = delta_h / sin_el;
  const float l_g = l_s * cos_el;
  float r_001 = 1.0f / (1.0f + 0.78f * sqrtf(fmaxf(l_g * gamma_r / freq_ghz, 0.0f)) -
      0.38f * (1.0f - expf(-2.0f * fmaxf(l_g, 0.0f))));
  r_001 = fmaxf(r_001, kMinRainReductionFactor);
  const float zeta = atan2f(delta_h, geometry_denominator(l_g * r_001));
  const float l_r = (zeta > theta)
      ? fmaxf(l_g, 0.0f) * r_001 / fmaxf(cos_el, kTrigDenomEps)
      : l_s;
  const float chi = fmaxf(36.0f - fabsf(fp(a, kFpRainLatitudeDeg, 0.0f)), 0.0f);
  float v_001 = 1.0f / (1.0f + sqrtf(sin_el) *
      (31.0f * (1.0f - expf(-theta_deg / (1.0f + chi))) *
          sqrtf(fmaxf(l_r * gamma_r, 0.0f)) / (freq_ghz * freq_ghz) - 0.45f));
  v_001 = fmaxf(v_001, kMinRainReductionFactor);
  const float a_001 = gamma_r * l_r * v_001;
  const float p = fminf(fmaxf(fp(a, kFpRainExceedancePct, 0.1f), 0.001f), 5.0f);
  if (fabsf(p - 0.01f) < kProbabilityEqualityTol) return a_001;
  const float lat_abs = fabsf(fp(a, kFpRainLatitudeDeg, 0.0f));
  float beta = 0.0f;
  if (p < 1.0f && lat_abs < 36.0f) {
    beta = theta_deg >= 25.0f
        ? -0.005f * (lat_abs - 36.0f)
        : -0.005f * (lat_abs - 36.0f) + 1.8f - 4.25f * sin_el;
  }
  const float exponent = -(0.655f + 0.033f * logf(p) - 0.045f * logf(log_argument(a_001)) -
      beta * (1.0f - p) * sin_el);
  return a_001 * powf(p / 0.01f, exponent);
}

__device__ __forceinline__ float atmospheric_loss_factor_device(const PackedAbi& a, float elevation) {
  float loss_db = 0.0f;
  if (ip(a, kParamAtmLossEnabled)) {
    const float sin_el = fmaxf(sinf(elevation), 1.0e-3f);
    loss_db += fp(a, kFpAtmLossDb, 0.0f) / sin_el;
  }
  if (ip(a, kParamRainLossEnabled)) {
    loss_db += rain_attenuation_db_device(a, elevation);
  }
  return powf(10.0f, -loss_db / 10.0f);
}

__device__ __forceinline__ float gu_distance2_for_stage(const PackedAbi& a, int stage_slot, int e, int u, int g) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const float gx = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + 0];
  const float gy = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + 1];
  const float ux = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 0];
  const float uy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 1];
  const float dx = gx - ux;
  const float dy = gy - uy;
  return dx * dx + dy * dy;
}

__device__ bool candidate_gid_better(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    int lhs,
    int rhs,
    int mode,
    bool radius_filter) {
  if (rhs < 0) return true;
  if (lhs < 0) return false;
  if (mode == 0) return lhs < rhs;
  if (mode == 1) {
    const int gu = static_cast<int>(ip(a, kParamNumGu));
    const float lq = a.f[stage_f(stage_slot, kSfGuQueue)][e * gu + lhs];
    const float rq = a.f[stage_f(stage_slot, kSfGuQueue)][e * gu + rhs];
    return lq > rq || (lq == rq && lhs < rhs);
  }
  const float ld2 = gu_distance2_for_stage(a, stage_slot, e, u, lhs);
  const float rd2 = gu_distance2_for_stage(a, stage_slot, e, u, rhs);
  (void)radius_filter;
  return ld2 < rd2 || (ld2 == rd2 && lhs < rhs);
}

__device__ int candidate_rank_for_gid(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    int gid,
    int mode,
    bool radius_filter) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  int rank = 0;
  for (int other = 0; other < gu; ++other) {
    if (other == gid) continue;
    if (mode <= 1) {
      if (a.l[stage_l(stage_slot, kSlAssoc)][e * gu + other] != u) continue;
    } else if (radius_filter) {
      const float radius = fp(a, kFpCandidateRadius, 0.0f);
      if (sqrtf(gu_distance2_for_stage(a, stage_slot, e, u, other)) > radius) continue;
    }
    if (candidate_gid_better(a, stage_slot, e, u, other, gid, mode, radius_filter)) ++rank;
  }
  return rank;
}

__device__ int select_candidate_gid_for_slot(const PackedAbi& a, int stage_slot, int e, int u, int c, int* valid_count_out) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int users_obs = static_cast<int>(ip(a, kParamUsersObsMax));
  const int mode_code = static_cast<int>(ip(a, kParamCandidateMode));
  const int raw_keep = static_cast<int>(ip(a, kParamCandidateK, users_obs));
  const int keep = min(max(raw_keep, 0), min(gu, users_obs));
  if (valid_count_out != nullptr) *valid_count_out = 0;
  if (keep <= 0 || c >= keep) return -1;

  if (mode_code == 0 || mode_code == 1) {
    int assoc_count = 0;
    for (int g = 0; g < gu; ++g) {
      if (a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g] == u) ++assoc_count;
    }
    const int valid_count = min(assoc_count, keep);
    if (valid_count_out != nullptr) *valid_count_out = valid_count;
    if (c >= valid_count) return -1;
    const int order_mode = (mode_code == 1 || assoc_count > keep) ? 1 : 0;
    for (int g = 0; g < gu; ++g) {
      if (a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g] != u) continue;
      if (candidate_rank_for_gid(a, stage_slot, e, u, g, order_mode, false) == c) return g;
    }
    return -1;
  }

  bool radius_filter = false;
  int valid_count = keep;
  if (mode_code == 2 && ip(a, kParamCandidateRadiusPresent) && fp(a, kFpCandidateRadius, 0.0f) > 0.0f) {
    int within_count = 0;
    const float radius = fp(a, kFpCandidateRadius, 0.0f);
    for (int g = 0; g < gu; ++g) {
      if (sqrtf(gu_distance2_for_stage(a, stage_slot, e, u, g)) <= radius) ++within_count;
    }
    if (within_count > 0) {
      radius_filter = true;
      valid_count = min(within_count, keep);
    }
  }
  if (valid_count_out != nullptr) *valid_count_out = valid_count;
  if (c >= valid_count) return -1;
  for (int g = 0; g < gu; ++g) {
    if (radius_filter && sqrtf(gu_distance2_for_stage(a, stage_slot, e, u, g)) > fp(a, kFpCandidateRadius, 0.0f)) continue;
    if (candidate_rank_for_gid(a, stage_slot, e, u, g, 3, radius_filter) == c) return g;
  }
  return -1;
}

__device__ __forceinline__ float sinc_pi_device(float x) {
  const float pix = 3.14159265358979323846f * x;
  return fabsf(pix) < kAngleRadEps ? 1.0f : sinf(pix) / pix;
}

__device__ bool bw_candidate_assoc_match(const PackedAbi& a, int stage_slot, int e, int u, int c, int* gid_out = nullptr) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int users_obs = static_cast<int>(ip(a, kParamUsersObsMax));
  if (c < 0 || c >= users_obs) return false;
  const int64_t gid64 = a.l[stage_l(stage_slot, kSlCandidateIndices)][(e * static_cast<int>(ip(a, kParamNumUav)) + u) * users_obs + c];
  const bool valid = a.b[stage_b(stage_slot, 0)][(e * static_cast<int>(ip(a, kParamNumUav)) + u) * users_obs + c] &&
      gid64 >= 0 && gid64 < gu;
  if (gid_out != nullptr) *gid_out = valid ? static_cast<int>(gid64) : -1;
  if (!valid) return false;
  return a.l[stage_l(stage_slot, kSlAssoc)][e * gu + static_cast<int>(gid64)] == u;
}

__device__ float bw_action_slot_value(const PackedAbi& a, int e, int u, int c, int bw_mode) {
  if (!ip(a, kParamEnableBwAction)) return 1.0f;
  if (bw_mode == kSourceZero) return 0.0f;
  int gid = -1;
  if (!bw_candidate_assoc_match(a, 3, e, u, c, &gid)) return 0.0f;
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  return a.f[kFLiveBwAction][(e * static_cast<int>(ip(a, kParamNumUav)) + u) * gu + gid];
}

__device__ float bw_action_gu_value(const PackedAbi& a, int e, int u, int g, int bw_mode) {
  if (!ip(a, kParamEnableBwAction)) return 1.0f;
  if (bw_mode == kSourceZero) return 0.0f;
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  if (g < 0 || g >= gu) return 0.0f;
  return a.f[kFLiveBwAction][(e * static_cast<int>(ip(a, kParamNumUav)) + u) * gu + g];
}

__device__ bool bw_gu_assoc_match(const PackedAbi& a, int stage_slot, int e, int u, int g) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  if (g < 0 || g >= gu || u < 0 || u >= ucount) return false;
  return a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g] == u;
}

__device__ int bw_assoc_count_for_u_all_gu(const PackedAbi& a, int stage_slot, int e, int u) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  int count = 0;
  for (int g = 0; g < gu; ++g) {
    count += bw_gu_assoc_match(a, stage_slot, e, u, g) ? 1 : 0;
  }
  return count;
}

__device__ float bw_beta_for_gu_full(const PackedAbi& a, int stage_slot, int e, int u, int g, int bw_mode) {
  if (!bw_gu_assoc_match(a, stage_slot, e, u, g)) return 0.0f;
  if (ip(a, kParamEnableBwAction)) {
    return bw_action_gu_value(a, e, u, g, bw_mode);
  }
  const int assoc_count = bw_assoc_count_for_u_all_gu(a, stage_slot, e, u);
  if (assoc_count <= 0) return 0.0f;
  return 1.0f / static_cast<float>(assoc_count);
}

__device__ float bw_beta_for_slot(const PackedAbi& a, int stage_slot, int e, int u, int c, int bw_mode) {
  int gid = -1;
  if (!bw_candidate_assoc_match(a, stage_slot, e, u, c, &gid)) return 0.0f;
  return bw_beta_for_gu_full(a, stage_slot, e, u, gid, bw_mode);
}

__device__ float bw_band_fraction_for_gu(const PackedAbi& a, int stage_slot, int e, int g, int bw_mode) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  if (g < 0 || g >= gu) return 0.0f;
  const int64_t assoc64 = a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g];
  if (assoc64 < 0 || assoc64 >= ucount) return 0.0f;
  return bw_beta_for_gu_full(a, stage_slot, e, static_cast<int>(assoc64), g, bw_mode);
}

__device__ float access_interference_for_u(const PackedAbi& a, int stage_slot, int e, int u, int bw_mode) {
  if (!ip(a, kParamInterferenceEnabled)) return 0.0f;
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const float p = fp(a, kFpAccessGuTxPower, fp(a, kFpLocalUavTxPower, 1.0f));
  float total_received = 0.0f;
  float same_cell = 0.0f;
  for (int g = 0; g < gu; ++g) {
    const float band = bw_band_fraction_for_gu(a, stage_slot, e, g, bw_mode);
    const int64_t assoc64 = a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g];
    if (assoc64 < 0 || assoc64 >= ucount || band <= 0.0f) continue;
    const float gain = a.f[stage_f(stage_slot, kSfAccessGainMatrix)][(e * gu + g) * ucount + u];
    const float received = p * gain * band;
    total_received += received;
    if (assoc64 == u) same_cell += received;
  }
  return fmaxf(quantize_device(total_received - same_cell, fp(a, kFpAccessInterferenceQuantum, 0.0f)), 0.0f);
}

__device__ float access_rate_for_gu_with_interference(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    int gid,
    int bw_mode,
    float interference);

__device__ float access_rate_for_gu(const PackedAbi& a, int stage_slot, int e, int u, int gid, int bw_mode) {
  return access_rate_for_gu_with_interference(a, stage_slot, e, u, gid, bw_mode, access_interference_for_u(a, stage_slot, e, u, bw_mode));
}

__device__ float access_rate_for_gu_with_interference(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    int gid,
    int bw_mode,
    float interference) {
  if (!bw_gu_assoc_match(a, stage_slot, e, u, gid)) return 0.0f;
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const float beta = bw_beta_for_gu_full(a, stage_slot, e, u, gid, bw_mode);
  if (beta <= 0.0f) return 0.0f;
  const float gain = a.f[stage_f(stage_slot, kSfAccessGainMatrix)][(e * gu + gid) * ucount + u];
  const float eff_bw = beta * fp(a, kFpAccessBAcc, 1.0f);
  if (eff_bw <= 0.0f) return 0.0f;
  const float eff_interference = ip(a, kParamInterferenceEnabled) ? beta * interference : 0.0f;
  const float denom = fp(a, kFpAccessNoiseDensity, fp(a, kFpLocalNoiseDensity, kDefaultNoiseDensity)) *
      fp(a, kFpAccessNoiseFigureLinear, 1.0f) * eff_bw + eff_interference;
  if (denom <= 0.0f) return 0.0f;
  const float snr = fp(a, kFpAccessGuTxPower, fp(a, kFpLocalUavTxPower, 1.0f)) * gain / denom;
  const float se = quantize_device(access_spectral_efficiency_device(a, snr), fp(a, kFpAccessEtaQuantum, kDefaultRateQuantum));
  return quantize_device(eff_bw * se, fp(a, kFpAccessRateQuantum, 32.0f));
}

__device__ float access_rate_for_slot(const PackedAbi& a, int stage_slot, int e, int u, int c, int bw_mode) {
  int gid = -1;
  if (!bw_candidate_assoc_match(a, stage_slot, e, u, c, &gid)) return 0.0f;
  return access_rate_for_gu(a, stage_slot, e, u, gid, bw_mode);
}

__device__ int sat_selected_count(const PackedAbi& a, int e, int sid) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  int count = 0;
  for (int u = 0; u < ucount; ++u) {
    for (int k = 0; k < select_k; ++k) {
      const int64_t cur = a.l[stage_l(3, kSlSatSelectionMatrix)][(e * ucount + u) * select_k + k];
      if (cur == sid) ++count;
    }
  }
  return count;
}

__device__ int sat_selected_count_for_stage(const PackedAbi& a, int stage_slot, int e, int sid) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  int count = 0;
  for (int u = 0; u < ucount; ++u) {
    for (int k = 0; k < select_k; ++k) {
      const int64_t cur = a.l[stage_l(stage_slot, kSlSatSelectionMatrix)][(e * ucount + u) * select_k + k];
      if (cur == sid) ++count;
    }
  }
  return count;
}

__device__ __forceinline__ float sat_selected_load_frac_for_stage(const PackedAbi& a, int stage_slot, int e, int sid) {
  const float ucount = fmaxf(static_cast<float>(ip(a, kParamNumUav)), 1.0f);
  return safe_div(static_cast<float>(sat_selected_count_for_stage(a, stage_slot, e, sid)), ucount);
}

__device__ int active_index_for_sat(const PackedAbi& a, int stage_slot, int e, int sid) {
  const int active = active_width(a);
  for (int aidx = 0; aidx < active; ++aidx) {
    if (a.l[stage_l(stage_slot, kSlActiveSatIds)][e * active + aidx] == sid) return aidx;
  }
  return -1;
}

__device__ float backhaul_rate_for_selected_us(const PackedAbi& a, int stage_slot, int e, int u, int sid);

__device__ float backhaul_rate_for_us(const PackedAbi& a, int stage_slot, int e, int u, int sid) {
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  if (sid < 0 || sid >= sat) return 0.0f;
  bool selected = false;
  for (int k = 0; k < select_k; ++k) {
    selected = selected || (a.l[stage_l(stage_slot, kSlSatSelectionMatrix)][(e * ucount + u) * select_k + k] == sid);
  }
  if (!selected) return 0.0f;
  return backhaul_rate_for_selected_us(a, stage_slot, e, u, sid);
}

__device__ float sat_compute_rate_for(const PackedAbi& a, int e, int s) {
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  if (has_f(a, kFMainBwSatComputeRates)) {
    const int envs = static_cast<int>(ip(a, kParamNumEnvs));
    if (a.f_numel[kFMainBwSatComputeRates] >= static_cast<int64_t>(envs) * sat) {
      return a.f[kFMainBwSatComputeRates][e * sat + s];
    }
    if (a.f_numel[kFMainBwSatComputeRates] > e) return a.f[kFMainBwSatComputeRates][e];
    if (a.f_numel[kFMainBwSatComputeRates] > 0) return a.f[kFMainBwSatComputeRates][0];
  }
  return fp(a, kFpBackhaulRateQuantum, 0.0f);
}

__device__ float doppler_eff_hz(const PackedAbi& a, int e, int u, int sid, const float rel[3], const float relv[3]) {
  if (!ip(a, kParamDopplerEnabled) && !ip(a, kParamDopplerAttenEnabled) && !ip(a, kParamDopplerObserved)) {
    return 0.0f;
  }
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  if (ip(a, kParamDopplerPrecompEnabled) && has_f(a, kFStateDopplerResidual) && sid >= 0 && sid < sat) {
    return a.f[kFStateDopplerResidual][(e * ucount + u) * sat + sid];
  }
  const float dist = sqrtf(geometry_denominator(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]));
  const float dot = rel[0] * relv[0] + rel[1] * relv[1] + rel[2] * relv[2];
  return fp(a, kFpSatCarrierFreq, fp(a, kFpCarrierFreq, 0.0f)) / fmaxf(fp(a, kFpSpeedOfLight, 299792458.0f), 1.0f) * dot / dist;
}

__device__ void uav_local_to_ecef_device(const PackedAbi& a, float x, float y, float out[3]) {
  constexpr float kDegToRad = 0.017453292519943295769f;
  const float earth = fmaxf(fp(a, kFpLocalEarthRadius, fp(a, kFpEarthRadius, 6371000.0f)), 1.0f);
  const float lat0 = fp(a, kFpRefLatDeg, 0.0f) * kDegToRad;
  const float lon0 = fp(a, kFpRefLonDeg, 0.0f) * kDegToRad;
  const float lat = lat0 + y / earth;
  const float lon = lon0 + x / positive_config_scale(earth * cosf(lat0));
  const float r = earth + fp(a, kFpSatGeomUavHeight, 0.0f);
  const float cos_lat = cosf(lat);
  const float sin_lat = sinf(lat);
  const float cos_lon = cosf(lon);
  const float sin_lon = sinf(lon);
  out[0] = r * cos_lat * cos_lon;
  out[1] = r * cos_lat * sin_lon;
  out[2] = r * sin_lat;
}

__device__ void uav_vel_local_to_ecef_device(const PackedAbi& a, float x, float y, float vx, float vy, float out[3]) {
  constexpr float kDegToRad = 0.017453292519943295769f;
  const float earth = fmaxf(fp(a, kFpLocalEarthRadius, fp(a, kFpEarthRadius, 6371000.0f)), 1.0f);
  const float lat0 = fp(a, kFpRefLatDeg, 0.0f) * kDegToRad;
  const float lon0 = fp(a, kFpRefLonDeg, 0.0f) * kDegToRad;
  const float lat = lat0 + y / earth;
  const float lon = lon0 + x / positive_config_scale(earth * cosf(lat0));
  const float sin_lat = sinf(lat);
  const float cos_lat = cosf(lat);
  const float sin_lon = sinf(lon);
  const float cos_lon = cosf(lon);
  out[0] = -sin_lon * vx - sin_lat * cos_lon * vy;
  out[1] = cos_lon * vx - sin_lat * sin_lon * vy;
  out[2] = cos_lat * vy;
}

__device__ void sat_rel_for_stage_us(const PackedAbi& a, int stage_slot, int e, int u, int sid, float rel[3], float relv[3]) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const float ux = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 0];
  const float uy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 1];
  const float uvx = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 0];
  const float uvy = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 1];
  float uecef[3];
  float uvecef[3];
  uav_local_to_ecef_device(a, ux, uy, uecef);
  uav_vel_local_to_ecef_device(a, ux, uy, uvx, uvy, uvecef);
  for (int d = 0; d < 3; ++d) {
    const float spos = (sid >= 0 && sid < sat) ? a.f[stage_f(stage_slot, kSfSatPos)][(e * sat + sid) * 3 + d] : 0.0f;
    const float svel = (sid >= 0 && sid < sat) ? a.f[stage_f(stage_slot, kSfSatVel)][(e * sat + sid) * 3 + d] : 0.0f;
    rel[d] = spos - uecef[d];
    relv[d] = svel - uvecef[d];
  }
}

__device__ __forceinline__ float sat_elevation_from_rel_device(const PackedAbi& a, const float rel[3]) {
  const float dist2 = geometry_denominator(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]);
  const float dist = sqrtf(dist2);
  const float sat_r2 = fmaxf(fp(a, kFpBwSatOrbitRadiusSq, 0.0f), 1.0f);
  const float uav_r2 = fmaxf(fp(a, kFpBwUavOrbitRadiusSq, 0.0f), 1.0f);
  const float uav_r = sqrtf(uav_r2);
  const float arg = ratio_or_zero(sat_r2 - uav_r2 - dist2, 2.0f * uav_r * dist);
  return asinf(clampf_device(arg, -1.0f, 1.0f));
}

__device__ float sat_elevation_for_stage_us(const PackedAbi& a, int stage_slot, int e, int u, int sid) {
  float rel[3];
  float relv[3];
  sat_rel_for_stage_us(a, stage_slot, e, u, sid, rel, relv);
  return sat_elevation_from_rel_device(a, rel);
}

__device__ bool last_sat_selected_for_u(const PackedAbi& a, int e, int u, int sid) {
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  if (!has_l(a, kLStateLastSatSelectionMatrix)) return false;
  for (int k = 0; k < select_k; ++k) {
    if (a.l[kLStateLastSatSelectionMatrix][(e * ucount + u) * select_k + k] == sid) return true;
  }
  return false;
}

__device__ float sat_score_se_for_stage_us(const PackedAbi& a, int stage_slot, int e, int u, int sid, float elevation) {
  float rel[3];
  float relv[3];
  sat_rel_for_stage_us(a, stage_slot, e, u, sid, rel, relv);
  const float dist2 = geometry_denominator(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]);
  float gain = positive_coeff(fp(a, kFpBwBackhaulGainConst, 1.0f)) / dist2;
  gain *= atmospheric_loss_factor_device(a, elevation);
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const float sat_load = sid >= 0 && sid < sat ? a.f[stage_f(stage_slot, kSfSatLoads)][e * sat + sid] : 0.0f;
  const float projected_count = fmaxf(sat_load + (last_sat_selected_for_u(a, e, u, sid) ? 0.0f : 1.0f), 1.0f);
  const float bandwidth = fp(a, kFpBwEffectiveBSatTotal, fp(a, kFpBSatTotal, 0.0f)) / projected_count;
  if (bandwidth <= 0.0f) return 0.0f;
  const float denom = fp(a, kFpSatNoiseDensity, fp(a, kFpBwNoiseDensity, kDefaultNoiseDensity)) *
      fp(a, kFpSatNoiseFigureLinear, 1.0f) * bandwidth;
  if (denom <= 0.0f) return 0.0f;
  float snr = fp(a, kFpSatUavTxPower, fp(a, kFpBwUavTxPower, 1.0f)) * gain / denom;
  if (ip(a, kParamDopplerAttenEnabled)) {
    const float spacing = fp(a, kFpSubcarrierSpacing, 0.0f);
    if (spacing > 0.0f) {
      const float nu_eff = doppler_eff_hz(a, e, u, sid, rel, relv);
      const float s = sinc_pi_device(nu_eff / spacing);
      snr *= s * s;
    }
  }
  return spectral_efficiency_device(snr);
}

__device__ float sat_score_se_from_cached_stage_us(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    int sid,
    float gain,
    float nu_eff) {
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const float sat_load = sid >= 0 && sid < sat ? a.f[stage_f(stage_slot, kSfSatLoads)][e * sat + sid] : 0.0f;
  const float projected_count = fmaxf(sat_load + (last_sat_selected_for_u(a, e, u, sid) ? 0.0f : 1.0f), 1.0f);
  const float bandwidth = fp(a, kFpBwEffectiveBSatTotal, fp(a, kFpBSatTotal, 0.0f)) / projected_count;
  if (bandwidth <= 0.0f) return 0.0f;
  const float denom = fp(a, kFpSatNoiseDensity, fp(a, kFpBwNoiseDensity, kDefaultNoiseDensity)) *
      fp(a, kFpSatNoiseFigureLinear, 1.0f) * bandwidth;
  if (denom <= 0.0f) return 0.0f;
  float snr = fp(a, kFpSatUavTxPower, fp(a, kFpBwUavTxPower, 1.0f)) * gain / denom;
  if (ip(a, kParamDopplerAttenEnabled)) {
    const float spacing = fp(a, kFpSubcarrierSpacing, 0.0f);
    if (spacing > 0.0f) {
      const float s = sinc_pi_device(nu_eff / spacing);
      snr *= s * s;
    }
  }
  return spectral_efficiency_device(snr);
}

__device__ float sat_score_se_cached_or_stage_us(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    int sid,
    float elevation) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  if (sid >= 0 && sid < sat && ip(a, kParamSatCandidateMode) != 0 &&
      has_f(a, stage_f(stage_slot, kSfUsGainAll)) &&
      has_f(a, stage_f(stage_slot, kSfUsNuEffAll))) {
    const int idx_us = (e * ucount + u) * sat + sid;
    return sat_score_se_from_cached_stage_us(
        a,
        stage_slot,
        e,
        u,
        sid,
        a.f[stage_f(stage_slot, kSfUsGainAll)][idx_us],
        a.f[stage_f(stage_slot, kSfUsNuEffAll)][idx_us]);
  }
  return sat_score_se_for_stage_us(a, stage_slot, e, u, sid, elevation);
}

__device__ float close_risk_trigger_dist(const PackedAbi& a) {
  const float d_safe = fp(a, kFpDSafe, fp(a, kFpAccelDSafe, 0.0f));
  const float d_alert = fp(a, kFpAvoidanceAlertFactor, 1.0f) * d_safe;
  float trigger_dist = d_alert;
  if (ip(a, kParamAvoidancePrealertFactorPresent)) {
    trigger_dist = fmaxf(fp(a, kFpAvoidancePrealertFactor, 0.0f) * d_safe, d_alert);
  }
  if (ip(a, kParamAvoidancePrealertModeTtc) && ip(a, kParamAvoidancePrealertDistCapPresent)) {
    trigger_dist = fmaxf(fp(a, kFpAvoidancePrealertDistCap, 0.0f), d_alert);
  }
  return trigger_dist;
}

__device__ float close_risk_pair_value(const PackedAbi& a, int stage_slot, int e, int u, int v, bool* valid_pair, bool* collision) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const float ux = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 0];
  const float uy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 1];
  const float vx = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + v) * 2 + 0];
  const float vy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + v) * 2 + 1];
  const float dux = ux - vx;
  const float duy = uy - vy;
  const float dist = sqrtf(dux * dux + duy * duy);
  const bool valid = dist > kDynamicsDenomEps;
  if (valid_pair != nullptr) *valid_pair = valid;
  const float d_safe = fp(a, kFpDSafe, fp(a, kFpAccelDSafe, 0.0f));
  if (collision != nullptr) *collision = dist < d_safe;
  if (!valid) return 0.0f;

  const float uvelx = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 0];
  const float uvely = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 1];
  const float vvelx = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + v) * 2 + 0];
  const float vvely = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + v) * 2 + 1];
  const float relvx = uvelx - vvelx;
  const float relvy = uvely - vvely;
  const float closing_speed = fmaxf(-(dux * relvx + duy * relvy) / dynamics_denominator(dist), 0.0f);
  const float d_alert = fp(a, kFpAvoidanceAlertFactor, 1.0f) * d_safe;
  const float trigger_dist = close_risk_trigger_dist(a);
  const float closing_thresh = fmaxf(fp(a, kFpAvoidancePrealertClosingSpeed, 0.0f), 0.0f);
  if (closing_speed <= closing_thresh || dist >= trigger_dist) return 0.0f;
  if (ip(a, kParamAvoidancePrealertModeTtc) && dist >= d_alert) {
    const float ttc_limit = fmaxf(fp(a, kFpAvoidancePrealertTtc, 0.0f), 0.0f);
    if (ttc_limit <= 0.0f) return 0.0f;
    const float ttc_to_alert = (dist - d_alert) / dynamics_denominator(closing_speed);
    if (ttc_to_alert >= ttc_limit) return 0.0f;
  }
  const float dist_denom = dynamics_denominator(trigger_dist - d_alert);
  const float close_scale = dynamics_denominator(closing_thresh);
  const float dist_ratio = clampf_device((trigger_dist - dist) / dist_denom, 0.0f, 1.0f);
  const float close_ratio = clampf_device((closing_speed - closing_thresh) / close_scale, 0.0f, fmaxf(fp(a, kFpCloseRiskCap, 1.0f), 0.0f));
  return dist_ratio * close_ratio;
}

__device__ bool accel_avoidance_pair_term(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    int v,
    float* dirx,
    float* diry,
    float* strength,
    float* closing_gain,
    float* closing_bonus_score,
    float* ttc_urgency,
    bool* in_core_alert,
    float* dist_out) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  if (u == v) return false;
  const float ux = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 0];
  const float uy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 1];
  const float vx = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + v) * 2 + 0];
  const float vy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + v) * 2 + 1];
  const float dx = ux - vx;
  const float dy = uy - vy;
  const float dist = sqrtf(dx * dx + dy * dy);
  if (dist <= kDynamicsDenomEps) return false;

  const float uvelx = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 0];
  const float uvely = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 1];
  const float vvelx = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + v) * 2 + 0];
  const float vvely = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + v) * 2 + 1];
  const float relvx = uvelx - vvelx;
  const float relvy = uvely - vvely;
  const float closing_speed = -(dx * relvx + dy * relvy) / dynamics_denominator(dist);

  const float d_safe = fp(a, kFpAccelDSafe, fp(a, kFpDSafe, 0.0f));
  const float d_alert = fp(a, kFpAvoidanceAlertFactor, 1.0f) * d_safe;
  float d_prealert = 0.0f;
  if (ip(a, kParamAvoidancePrealertFactorPresent)) {
    d_prealert = fmaxf(fp(a, kFpAvoidancePrealertFactor, 0.0f) * d_safe, d_alert);
  }
  const float closing_thresh = fmaxf(fp(a, kFpAvoidancePrealertClosingSpeed, 0.0f), 0.0f);
  float prealert_trigger_dist = d_prealert;
  if (ip(a, kParamAvoidancePrealertModeTtc)) {
    if (ip(a, kParamAvoidancePrealertDistCapPresent)) {
      prealert_trigger_dist = fmaxf(fp(a, kFpAvoidancePrealertDistCap, 0.0f), d_alert);
    } else if (d_prealert > 0.0f) {
      prealert_trigger_dist = d_prealert;
    }
  }

  float trigger_dist = d_alert;
  const bool core = d_alert > 0.0f && dist < d_alert;
  bool prealert = false;
  float ttc_to_alert = 3.402823466e38f;
  if (ip(a, kParamAvoidancePrealertModeTtc)) {
    const float ttc_limit = fmaxf(fp(a, kFpAvoidancePrealertTtc, 0.0f), 0.0f);
    if (prealert_trigger_dist > d_alert && dist < prealert_trigger_dist &&
        closing_speed > closing_thresh && ttc_limit > 0.0f) {
      ttc_to_alert = (dist - d_alert) / dynamics_denominator(closing_speed);
      prealert = ttc_to_alert < ttc_limit;
    }
  } else {
    prealert = d_prealert > d_alert && dist < d_prealert && closing_speed > closing_thresh;
  }
  if (prealert && !core) {
    trigger_dist = ip(a, kParamAvoidancePrealertModeTtc) ? prealert_trigger_dist : d_prealert;
  }
  if (!core && !prealert) return false;

  const int repulse_mode = static_cast<int>(ip(a, kParamAvoidanceRepulseMode, 0));
  float term_strength = 0.0f;
  if (repulse_mode == 1) {
    const float denom = dynamics_denominator(trigger_dist - d_safe);
    term_strength = clampf_device((trigger_dist - dist) / denom, 0.0f, 1.0f);
  } else if (repulse_mode == 2) {
    const float denom = dynamics_denominator(trigger_dist - d_safe);
    const float base = clampf_device((trigger_dist - dist) / denom, 0.0f, 1.0f);
    term_strength = base * base;
  } else {
    term_strength = 1.0f / dynamics_denominator(dist) - 1.0f / dynamics_denominator(trigger_dist);
  }

  float gain = 1.0f;
  float ratio_raw = 1.0f;
  if (ip(a, kParamAvoidanceClosingGainEnabled) && closing_thresh > kDynamicsDenomEps && closing_speed > closing_thresh) {
    ratio_raw = closing_speed / closing_thresh;
    gain = clampf_device(ratio_raw, 1.0f, fmaxf(fp(a, kFpAvoidanceClosingGainCap, 2.0f), 1.0f));
  }

  if (dirx != nullptr) *dirx = dx / dynamics_denominator(dist);
  if (diry != nullptr) *diry = dy / dynamics_denominator(dist);
  if (strength != nullptr) *strength = term_strength;
  if (closing_gain != nullptr) *closing_gain = gain;
  if (closing_bonus_score != nullptr) *closing_bonus_score = term_strength * fmaxf(ratio_raw - 1.0f, 0.0f);
  if (ttc_urgency != nullptr) *ttc_urgency = isfinite(ttc_to_alert) ? 1.0f / dynamics_denominator(ttc_to_alert) : 0.0f;
  if (in_core_alert != nullptr) *in_core_alert = core;
  if (dist_out != nullptr) *dist_out = dist;
  return true;
}

__device__ void accel_apply_avoidance(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    float policy_ax,
    float policy_ay,
    float* exec_ax,
    float* exec_ay) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const float amax = positive_config_scale(fp(a, kFpAccelAMax, 1.0f));
  float repx = 0.0f;
  float repy = 0.0f;
  if (ip(a, kParamUseAvoidance) && ucount > 1) {
    const float eta = has_f(a, kFStateAvoidanceEtaEff)
        ? a.f[kFStateAvoidanceEtaEff][e]
        : fp(a, kFpAvoidanceEta, 0.0f);

    int best_v = -1;
    int best_core = -1;
    float best_bonus = -3.402823466e38f;
    float best_ttc = -3.402823466e38f;
    float best_strength = -3.402823466e38f;
    float best_neg_dist = -3.402823466e38f;
    if (ip(a, kParamAvoidanceClosingGainEnabled) && ip(a, kParamAvoidanceClosingGainTop1Only)) {
      for (int v = 0; v < ucount; ++v) {
        float dirx = 0.0f, diry = 0.0f, strength = 0.0f, gain = 1.0f;
        float bonus = 0.0f, ttc = 0.0f, dist = 0.0f;
        bool core = false;
        if (!accel_avoidance_pair_term(a, stage_slot, e, u, v, &dirx, &diry, &strength, &gain, &bonus, &ttc, &core, &dist)) {
          continue;
        }
        if (gain <= 1.0f) continue;
        const int core_key = core ? 1 : 0;
        const float neg_dist = -dist;
        const bool better =
            best_v < 0 ||
            core_key > best_core ||
            (core_key == best_core && bonus > best_bonus) ||
            (core_key == best_core && bonus == best_bonus && ttc > best_ttc) ||
            (core_key == best_core && bonus == best_bonus && ttc == best_ttc && strength > best_strength) ||
            (core_key == best_core && bonus == best_bonus && ttc == best_ttc && strength == best_strength && neg_dist > best_neg_dist);
        if (better) {
          best_v = v;
          best_core = core_key;
          best_bonus = bonus;
          best_ttc = ttc;
          best_strength = strength;
          best_neg_dist = neg_dist;
        }
      }
    }

    for (int v = 0; v < ucount; ++v) {
      float dirx = 0.0f, diry = 0.0f, strength = 0.0f, gain = 1.0f;
      float bonus = 0.0f, ttc = 0.0f, dist = 0.0f;
      bool core = false;
      if (!accel_avoidance_pair_term(a, stage_slot, e, u, v, &dirx, &diry, &strength, &gain, &bonus, &ttc, &core, &dist)) {
        continue;
      }
      if (ip(a, kParamAvoidanceClosingGainEnabled) && ip(a, kParamAvoidanceClosingGainTop1Only) && best_v >= 0 && v != best_v) {
        gain = 1.0f;
      }
      repx += eta * strength * gain * dirx;
      repy += eta * strength * gain * diry;
    }
    if (ip(a, kParamAvoidanceRepulseClip, 1)) {
      project_l2_device(&repx, &repy, amax);
    }
  }
  if (has_f(a, kFStateLastAvoidanceEtaExec)) {
    a.f[kFStateLastAvoidanceEtaExec][e] = has_f(a, kFStateAvoidanceEtaEff)
        ? a.f[kFStateAvoidanceEtaEff][e]
        : fp(a, kFpAvoidanceEta, 0.0f);
  }
  float ax = policy_ax + repx;
  float ay = policy_ay + repy;
  project_l2_device(&ax, &ay, amax);
  *exec_ax = ax;
  *exec_ay = ay;
}

__device__ __forceinline__ void project_accel_and_next_speed_device(
    float* ax,
    float* ay,
    float vx,
    float vy,
    float tau,
    float amax,
    float vmax) {
  project_l2_device(ax, ay, amax);
  float vx_next = vx + (*ax) * tau;
  float vy_next = vy + (*ay) * tau;
  project_l2_device(&vx_next, &vy_next, vmax);
  *ax = (vx_next - vx) / dynamics_denominator(tau);
  *ay = (vy_next - vy) / dynamics_denominator(tau);
  project_l2_device(ax, ay, amax);
}

__device__ __forceinline__ void safety_pair_direction_device(
    float rx,
    float ry,
    float wx,
    float wy,
    float* nx,
    float* ny) {
  const float r_norm = sqrtf(rx * rx + ry * ry);
  if (r_norm > kDynamicsDenomEps) {
    *nx = rx / r_norm;
    *ny = ry / r_norm;
    return;
  }
  const float w_norm = sqrtf(wx * wx + wy * wy);
  if (w_norm > kDynamicsDenomEps) {
    *nx = -wx / w_norm;
    *ny = -wy / w_norm;
    return;
  }
  *nx = 1.0f;
  *ny = 0.0f;
}

__device__ float native_safety_pair_margin_device(
    float rx,
    float ry,
    float wx,
    float wy,
    float ax_rel,
    float ay_rel,
    float tau,
    float d_required,
    float a_safe,
    float* nx_out,
    float* ny_out,
    float* c_next_out) {
  float nx = 1.0f;
  float ny = 0.0f;
  safety_pair_direction_device(rx, ry, wx, wy, &nx, &ny);
  const float w_next_x = wx + tau * ax_rel;
  const float w_next_y = wy + tau * ay_rel;
  const float r_next_proj = nx * (rx + tau * wx + tau * tau * ax_rel) +
      ny * (ry + tau * wy + tau * tau * ay_rel);
  const float c_next = fmaxf(0.0f, -(nx * w_next_x + ny * w_next_y));
  if (nx_out != nullptr) *nx_out = nx;
  if (ny_out != nullptr) *ny_out = ny;
  if (c_next_out != nullptr) *c_next_out = c_next;
  return r_next_proj - (d_required + c_next * c_next / (2.0f * dynamics_denominator(a_safe)));
}

__device__ void native_safety_project_env(const PackedAbi& a, int stage_slot, int e) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  if (ucount <= 1) return;
  const float tau = dynamics_denominator(fp(a, kFpAccelTau0, fp(a, kFpTau0, 1.0f)));
  const float amax = positive_config_scale(fp(a, kFpAccelAMax, 1.0f));
  const float vmax = positive_config_scale(fp(a, kFpAccelVMax, fp(a, kFpVMax, 1.0f)));
  const float d_safe = fmaxf(fp(a, kFpAccelDSafe, fp(a, kFpDSafe, 0.0f)), 0.0f);
  const float buffer = fmaxf(fp(a, kFpSafetyShieldBuffer, 0.0f), 0.0f);
  const float rho = fmaxf(fp(a, kFpSafetyShieldBrakeRho, 0.8f), 1.0e-6f);
  float a_safe = fmaxf(fp(a, kFpSafetyShieldASafe, 0.0f), 0.0f);
  if (a_safe <= 0.0f) a_safe = rho * 2.0f * amax;
  a_safe = dynamics_denominator(a_safe);
  const float gain = fmaxf(fp(a, kFpSafetyShieldStepGain, 1.0f), 0.0f);
  const float tol = fmaxf(fp(a, kFpSafetyShieldTol, 1.0e-5f), 0.0f);
  const int iters = max(static_cast<int>(ip(a, kParamSafetyShieldIters, 8)), 1);
  const float d_required = d_safe + buffer;

  for (int u = 0; u < ucount; ++u) {
    float ax = a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 0];
    float ay = a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 1];
    const float vx = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 0];
    const float vy = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 1];
    project_accel_and_next_speed_device(&ax, &ay, vx, vy, tau, amax, vmax);
    a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 0] = ax;
    a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 1] = ay;
  }

  for (int pass = 0; pass < iters; ++pass) {
    bool changed = false;
    for (int u = 0; u < ucount; ++u) {
      for (int v = u + 1; v < ucount; ++v) {
        const float ux = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 0];
        const float uy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 1];
        const float vx_pos = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + v) * 2 + 0];
        const float vy_pos = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + v) * 2 + 1];
        const float uvx = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 0];
        const float uvy = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 1];
        const float vvx = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + v) * 2 + 0];
        const float vvy = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + v) * 2 + 1];
        const float axu = a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 0];
        const float ayu = a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 1];
        const float axv = a.f[kFStateLastExecAccel][(e * ucount + v) * 2 + 0];
        const float ayv = a.f[kFStateLastExecAccel][(e * ucount + v) * 2 + 1];
        const float rx = ux - vx_pos;
        const float ry = uy - vy_pos;
        const float wx = uvx - vvx;
        const float wy = uvy - vvy;
        float nx = 1.0f;
        float ny = 0.0f;
        float c_next = 0.0f;
        const float margin = native_safety_pair_margin_device(
            rx,
            ry,
            wx,
            wy,
            axu - axv,
            ayu - ayv,
            tau,
            d_required,
            a_safe,
            &nx,
            &ny,
            &c_next);
        if (margin >= -tol || gain <= 0.0f) continue;

        const float denom = dynamics_denominator(2.0f * tau * tau + 2.0f * tau * c_next / a_safe);
        const float delta = fminf(gain * (-margin + tol) / denom, 2.0f * amax);
        float axu_new = axu + delta * nx;
        float ayu_new = ayu + delta * ny;
        float axv_new = axv - delta * nx;
        float ayv_new = ayv - delta * ny;
        project_accel_and_next_speed_device(&axu_new, &ayu_new, uvx, uvy, tau, amax, vmax);
        project_accel_and_next_speed_device(&axv_new, &ayv_new, vvx, vvy, tau, amax, vmax);
        a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 0] = axu_new;
        a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 1] = ayu_new;
        a.f[kFStateLastExecAccel][(e * ucount + v) * 2 + 0] = axv_new;
        a.f[kFStateLastExecAccel][(e * ucount + v) * 2 + 1] = ayv_new;
        changed = true;
      }
    }
    if (!changed) break;
  }
}

__device__ float backhaul_rate_for_selected_us(const PackedAbi& a, int stage_slot, int e, int u, int sid) {
  if (sid < 0 || sid >= static_cast<int>(ip(a, kParamNumSat))) return 0.0f;
  float rel[3];
  float relv[3];
  sat_rel_for_stage_us(a, stage_slot, e, u, sid, rel, relv);
  const float dist2 = fmaxf(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2], 1.0f);
  const float elevation = sat_elevation_from_rel_device(a, rel);
  const float gain = positive_coeff(fp(a, kFpBwBackhaulGainConst, 1.0f)) / dist2 *
      atmospheric_loss_factor_device(a, elevation);
  const float nu_eff = doppler_eff_hz(a, e, u, sid, rel, relv);
  const bool elevation_valid = elevation >= fp(a, kFpThetaMinRad, -3.1415926f);
  const bool selected_valid = elevation_valid && (!ip(a, kParamDopplerEnabled) || fabsf(nu_eff) <= fmaxf(fp(a, kFpNuMax, 1.0f), 1.0f));
  const float count = fmaxf(static_cast<float>(sat_selected_count_for_stage(a, stage_slot, e, sid)), 1.0f);
  float b_ul = fp(a, kFpBwEffectiveBSatTotal, fp(a, kFpBSatTotal, 0.0f)) / count;
  if (b_ul <= 0.0f) return 0.0f;
  const float denom = fp(a, kFpBwNoiseDensity, kDefaultNoiseDensity) * fp(a, kFpBwNoiseFigureLinear, 1.0f) * b_ul;
  if (denom <= 0.0f) return 0.0f;
  float snr = fp(a, kFpBwUavTxPower, 1.0f) * gain / denom;
  if (ip(a, kParamDopplerAttenEnabled)) {
    const float spacing = fp(a, kFpSubcarrierSpacing, 0.0f);
    if (spacing > 0.0f) {
      const float s = sinc_pi_device(nu_eff / spacing);
      snr *= s * s;
    }
  }
  const float rate = selected_valid ? b_ul * spectral_efficiency_device(snr) : 0.0f;
  return quantize_device(rate, fp(a, kFpBackhaulRateQuantum, 32.0f));
}

__device__ void copy_state_to_stage_base(const PackedAbi& a, int stage_slot, int e, int stage_id) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  a.l[stage_l(stage_slot, kSlStageId)][e] = static_cast<int64_t>(stage_id);
  a.f[stage_f(stage_slot, kSfEffectiveBSatTotal)][e] = fp(a, kFpBwEffectiveBSatTotal, fp(a, kFpBSatTotal, 0.0f));
  for (int u = 0; u < ucount; ++u) {
    for (int d = 0; d < 2; ++d) {
      a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + d] = a.f[kFStateUavPos][(e * ucount + u) * 2 + d];
      a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + d] = a.f[kFStateUavVel][(e * ucount + u) * 2 + d];
    }
    a.f[stage_f(stage_slot, kSfUavEnergy)][e * ucount + u] = a.f[kFStateUavEnergy][e * ucount + u];
    a.f[stage_f(stage_slot, kSfUavQueue)][e * ucount + u] = a.f[kFStateUavQueue][e * ucount + u];
  }
  for (int g = 0; g < gu; ++g) {
    for (int d = 0; d < 2; ++d) {
      a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + d] = a.f[kFStateGuPos][(e * gu + g) * 2 + d];
    }
    a.f[stage_f(stage_slot, kSfGuQueue)][e * gu + g] = a.f[kFStateGuQueue][e * gu + g];
    a.l[stage_l(stage_slot, kSlPrevAssociation)][e * gu + g] =
        has_i(a, kIStateLastAssociation) ? static_cast<int64_t>(a.i[kIStateLastAssociation][e * gu + g]) : -1;
  }
  for (int s = 0; s < sat; ++s) {
    a.f[stage_f(stage_slot, kSfSatQueue)][e * sat + s] = a.f[kFStateSatQueue][e * sat + s];
    a.f[stage_f(stage_slot, kSfSatLoads)][e * sat + s] =
        has_f(a, kFStateLastSatConnectionCounts) ? a.f[kFStateLastSatConnectionCounts][e * sat + s] : 0.0f;
    for (int d = 0; d < 3; ++d) {
      a.f[stage_f(stage_slot, kSfSatPos)][(e * sat + s) * 3 + d] = a.f[kFStateSatPos][(e * sat + s) * 3 + d];
      a.f[stage_f(stage_slot, kSfSatVel)][(e * sat + s) * 3 + d] = a.f[kFStateSatVel][(e * sat + s) * 3 + d];
    }
  }
}

__device__ void write_ego_uav(const PackedAbi& a, int stage_slot, int e, int u, float* out, int out_row, int node_dim) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const float map_size = positive_config_scale(fp(a, kFpMapSize, 1.0f));
  const float vmax = positive_config_scale(fp(a, kFpVMax, 1.0f));
  const float emax = positive_config_scale(fp(a, kFpUavEnergyInit, 1.0f));
  const float qmax_uav = positive_config_scale(fp(a, kFpQueueMaxUav, 1.0f));
  for (int d = 0; d < node_dim; ++d) out[out_row * node_dim + d] = 0.0f;
  if (node_dim > 0) out[out_row * node_dim + 0] = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 0] / map_size;
  if (node_dim > 1) out[out_row * node_dim + 1] = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 1] / map_size;
  if (node_dim > 2) out[out_row * node_dim + 2] = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 0] / vmax;
  if (node_dim > 3) out[out_row * node_dim + 3] = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 1] / vmax;
  if (node_dim > 4) out[out_row * node_dim + 4] = a.f[stage_f(stage_slot, kSfUavEnergy)][e * ucount + u] / emax;
  if (node_dim > 5) out[out_row * node_dim + 5] = a.f[stage_f(stage_slot, kSfUavQueue)][e * ucount + u] / qmax_uav;
  int feat_col = 7;
  if (ip(a, kParamObsOwnAssocCost)) {
    if (feat_col < node_dim) out[out_row * node_dim + feat_col] = a.f[stage_f(stage_slot, kSfUavAssocUavCost)][e * ucount + u];
    ++feat_col;
  }
  if (ip(a, kParamObsOwnUavId) && feat_col < node_dim) {
    out[out_row * node_dim + feat_col] = (static_cast<float>(u) + 0.5f) / fmaxf(static_cast<float>(ucount), 1.0f);
  }
}

__device__ void write_sat_obs_row(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    float* sat_nodes,
    float* sat_edges,
    bool* sat_mask,
    int out_row,
    int node_dim,
    int edge_dim,
    int width) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int visible = static_cast<int>(ip(a, kParamVisibleSatsMax));
  const float queue_max_sat = positive_config_scale(fp(a, kFpQueueMaxSat, 1.0f));
  const float orbit = fmaxf(fp(a, kFpLocalEarthRadius, 0.0f) + fp(a, kFpSatHeight, 0.0f), 1.0f);
  const float nu_max = fmaxf(fp(a, kFpLocalNuMax, 1.0f), 1.0f);
  for (int j = 0; j < width; ++j) {
    const int sid = (j < visible) ? static_cast<int>(a.l[stage_l(stage_slot, kSlVisibleIds)][(e * ucount + u) * visible + j]) : -1;
    float rel[3];
    float relv[3];
    if (sid >= 0 && sid < sat) sat_rel_for_stage_us(a, stage_slot, e, u, sid, rel, relv);
    const float nu_eff = (sid >= 0 && sid < sat) ? doppler_eff_hz(a, e, u, sid, rel, relv) : 0.0f;
    const float elevation = (sid >= 0 && sid < sat) ? sat_elevation_from_rel_device(a, rel) : -3.1415926f;
    const bool doppler_ok = !ip(a, kParamDopplerEnabled) || fabsf(nu_eff) <= fmaxf(fp(a, kFpNuMax, 1.0f), 1.0f);
    const bool valid = sid >= 0 && sid < sat && elevation >= fp(a, kFpThetaMinRad, -3.1415926f) && doppler_ok;
    sat_mask[out_row * width + j] = valid;
    for (int d = 0; d < node_dim; ++d) sat_nodes[(out_row * width + j) * node_dim + d] = 0.0f;
    for (int d = 0; d < edge_dim; ++d) sat_edges[(out_row * width + j) * edge_dim + d] = 0.0f;
    if (!valid) continue;
    for (int d = 0; d < 3 && d < node_dim; ++d) {
      sat_nodes[(out_row * width + j) * node_dim + d] = a.f[stage_f(stage_slot, kSfSatPos)][(e * sat + sid) * 3 + d] / orbit;
    }
    for (int d = 0; d < 3 && d + 3 < node_dim; ++d) {
      sat_nodes[(out_row * width + j) * node_dim + 3 + d] = a.f[stage_f(stage_slot, kSfSatVel)][(e * sat + sid) * 3 + d] / orbit;
    }
    if (node_dim > 6) sat_nodes[(out_row * width + j) * node_dim + 6] = a.f[stage_f(stage_slot, kSfSatQueue)][e * sat + sid] / queue_max_sat;
    if (node_dim > 7) sat_nodes[(out_row * width + j) * node_dim + 7] = a.f[stage_f(stage_slot, kSfSatLoads)][e * sat + sid] / fmaxf(static_cast<float>(ucount), 1.0f);
    if (node_dim > 8) sat_nodes[(out_row * width + j) * node_dim + 8] = a.f[stage_f(stage_slot, kSfSatCostNorm)][e * sat + sid];
    for (int d = 0; d < 3 && d < edge_dim; ++d) sat_edges[(out_row * width + j) * edge_dim + d] = rel[d] / orbit;
    for (int d = 0; d < 3 && d + 3 < edge_dim; ++d) sat_edges[(out_row * width + j) * edge_dim + 3 + d] = relv[d] / orbit;
    if (edge_dim > 6) sat_edges[(out_row * width + j) * edge_dim + 6] = nu_eff / nu_max;
    if (edge_dim > 7) {
      const float dist2 = fmaxf(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2], 1.0f);
      const float gain = positive_coeff(fp(a, kFpBwBackhaulGainConst, 1.0f)) / dist2 *
          atmospheric_loss_factor_device(a, elevation);
      const float bw = fp(a, kFpBwEffectiveBSatTotal, fp(a, kFpBSatTotal, 0.0f));
      const float denom = fp(a, kFpBwNoiseDensity, kDefaultNoiseDensity) * fp(a, kFpBwNoiseFigureLinear, fp(a, kFpSatNoiseFigureLinear, 1.0f)) * bw;
      const float snr = (bw > 0.0f && denom > 0.0f)
          ? fmaxf(fp(a, kFpSatUavTxPower, fp(a, kFpBwUavTxPower, fp(a, kFpLocalUavTxPower, 1.0f))), 0.0f) * gain / denom
          : 0.0f;
      sat_edges[(out_row * width + j) * edge_dim + 7] = spectral_efficiency_device(snr);
    }
    if (edge_dim > 8) sat_edges[(out_row * width + j) * edge_dim + 8] = a.f[stage_f(stage_slot, kSfSatQueue)][e * sat + sid] / queue_max_sat;
    if (edge_dim > 9) sat_edges[(out_row * width + j) * edge_dim + 9] = a.f[stage_f(stage_slot, kSfSatLoads)][e * sat + sid] / fmaxf(static_cast<float>(ucount), 1.0f);
    if (edge_dim > 10) sat_edges[(out_row * width + j) * edge_dim + 10] = 1.0f / fmaxf(a.f[stage_f(stage_slot, kSfSatLoads)][e * sat + sid], 1.0f);
    if (edge_dim > 11) sat_edges[(out_row * width + j) * edge_dim + 11] = 1.0f;
    if (edge_dim > 12) sat_edges[(out_row * width + j) * edge_dim + 12] = doppler_ok ? 1.0f : 0.0f;
    if (edge_dim > 13) {
      const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
      float selected = 0.0f;
      for (int k = 0; k < select_k; ++k) {
        selected = selected || (a.l[stage_l(stage_slot, kSlSatSelectionMatrix)][(e * ucount + u) * select_k + k] == sid);
      }
      sat_edges[(out_row * width + j) * edge_dim + 13] = selected;
    }
  }
}

__device__ float sat_cost_current(const PackedAbi& a, int e, int s);
__device__ float uav_cost_last_route(const PackedAbi& a, int e, int u);
__device__ float gu_cost_last_route(const PackedAbi& a, int e, int g);
__device__ void last_route_cost_cache_fill_parallel(const PackedAbi& a, int e, float* cache);
__device__ float uav_cost_last_route_cached(const PackedAbi& a, int e, int u, float* cache);
__device__ float gu_cost_last_route_cached(const PackedAbi& a, int e, int g, float* cache);
__device__ float sat_flow_ref_current(const PackedAbi& a, float arrival_ref);
__device__ float selected_shared_fraction_last(const PackedAbi& a, int e, int u, int v);

__device__ __forceinline__ float accel_arrival_ref(const PackedAbi& a, int e) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  return require_positive_reward_ref(has_f(a, kFStateArrivalRef) ? a.f[kFStateArrivalRef][e] : static_cast<float>(gu) * fp(a, kFpTau0, 1.0f));
}

__device__ __forceinline__ float accel_access_noise_ref(const PackedAbi& a) {
  return require_positive_reward_ref(
      fp(a, kFpAccessNoiseDensity, fp(a, kFpLocalNoiseDensity, kDefaultNoiseDensity)) *
      fp(a, kFpAccessNoiseFigureLinear, 1.0f) *
      positive_config_scale(fp(a, kFpAccessBAcc, 1.0f)));
}

__device__ __forceinline__ float accel_backhaul_noise_ref(const PackedAbi& a) {
  const float b_ref = positive_config_scale(fp(a, kFpBwEffectiveBSatTotal, fp(a, kFpBSatTotal, 1.0f)));
  return require_positive_reward_ref(fp(a, kFpBwNoiseDensity, kDefaultNoiseDensity) * fp(a, kFpBwNoiseFigureLinear, 1.0f) * b_ref);
}

__device__ __forceinline__ float accel_access_se_from_gain(const PackedAbi& a, float gain, float access_noise_ref) {
  const float snr = fp(a, kFpAccessGuTxPower, fp(a, kFpLocalUavTxPower, 1.0f)) * gain / access_noise_ref;
  return access_spectral_efficiency_device(a, snr);
}

__device__ float accel_uav_last_outflow(const PackedAbi& a, int e, int u) {
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  float outflow = 0.0f;
  if (has_f(a, kFStateLastUavToSatOutflowMatrix)) {
    for (int s = 0; s < sat; ++s) outflow += a.f[kFStateLastUavToSatOutflowMatrix][(e * ucount + u) * sat + s];
  }
  return outflow;
}

__device__ float accel_sat_last_incoming(const PackedAbi& a, int e, int s) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  float incoming = 0.0f;
  if (has_f(a, kFStateLastUavToSatOutflowMatrix)) {
    for (int u = 0; u < ucount; ++u) incoming += a.f[kFStateLastUavToSatOutflowMatrix][(e * ucount + u) * sat + s];
  }
  return incoming;
}

__device__ float accel_gu_last_bw_sum(const PackedAbi& a, int e, int g) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  float total = 0.0f;
  if (has_f(a, kFStateLastBwFractionByUavGu)) {
    for (int u = 0; u < ucount; ++u) total += a.f[kFStateLastBwFractionByUavGu][(e * ucount + u) * gu + g];
  }
  return total;
}

__device__ float accel_gu_last_bw_scale(const PackedAbi& a, int e, int g) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int last_assoc = has_i(a, kIStateLastAssociation) ? a.i[kIStateLastAssociation][e * gu + g] : -1;
  if (last_assoc < 0 || last_assoc >= ucount) return 0.0f;
  return fmaxf(accel_gu_last_bw_sum(a, e, g), 0.0f);
}

__device__ float accel_gu_dist(const PackedAbi& a, int stage_slot, int e, int u, int g) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const float gx = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + 0];
  const float gy = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + 1];
  const float ux = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 0];
  const float uy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 1];
  const float dx = gx - ux;
  const float dy = gy - uy;
  return sqrtf(dx * dx + dy * dy);
}

__device__ float accel_owner_stability_margin(const PackedAbi& a, int stage_slot, int e, int g, float map_size) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int owner = static_cast<int>(a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g]);
  if (ucount <= 1) return 1.0f;
  const float d_owner = accel_gu_dist(a, stage_slot, e, owner, g);
  float d_second = 3.402823466e38f;
  for (int v = 0; v < ucount; ++v) {
    if (v == owner) continue;
    d_second = fminf(d_second, accel_gu_dist(a, stage_slot, e, v, g));
  }
  return (d_second - d_owner) / map_size;
}

__device__ float accel_cell_summary_value(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    int field,
    float arrival_ref,
    float gu_flow_ref,
    float access_noise_ref,
    float* last_route_cost_cache = nullptr) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const float map_size = positive_config_scale(fp(a, kFpMapSize, 1.0f));
  const float tau = fp(a, kFpBwTau0, fp(a, kFpTau0, 1.0f));
  const float eps = positive_coeff(fp(a, kFpWorkloadEps, kRelativeLogEps));
  float sum = 0.0f;
  float denom = 0.0f;
  float mx = 0.0f;
  float my = 0.0f;

  if (field == kAccelCellWorkloadShareGap) {
    float own = 0.0f;
    float total = 0.0f;
    for (int v = 0; v < ucount; ++v) {
      float raw = 0.0f;
      for (int g = 0; g < gu; ++g) {
        const int owner = static_cast<int>(a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g]);
        if (owner != v) continue;
        const float q = a.f[stage_f(stage_slot, kSfGuQueue)][e * gu + g];
        raw += fmaxf(gu_cost_last_route_cached(a, e, g, last_route_cost_cache) * q, 0.0f);
      }
      const float value = fmaxf(raw / fmaxf(static_cast<float>(gu), 1.0f), 0.0f);
      total += value;
      if (v == u) own = value;
    }
    return total > 0.0f ? own / total - 1.0f / static_cast<float>(ucount) : 0.0f;
  }

  if (field == kAccelCellInterferenceExposure) {
    for (int g = 0; g < gu; ++g) {
      const int last_assoc = has_i(a, kIStateLastAssociation) ? a.i[kIStateLastAssociation][e * gu + g] : -1;
      if (last_assoc == u) continue;
      const float bw_scale = accel_gu_last_bw_scale(a, e, g);
      const float gain = a.f[stage_f(stage_slot, kSfAccessGainMatrix)][(e * gu + g) * ucount + u];
      sum += fp(a, kFpAccessGuTxPower, fp(a, kFpLocalUavTxPower, 1.0f)) * gain * bw_scale;
    }
    return log1pf(fmaxf(sum / access_noise_ref, 0.0f));
  }

  for (int g = 0; g < gu; ++g) {
    const int owner = static_cast<int>(a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g]);
    if (owner != u) continue;
    const float q = a.f[stage_f(stage_slot, kSfGuQueue)][e * gu + g];
    const float expected = has_f(a, kFStateLastArrivalRateVec) ? a.f[kFStateLastArrivalRateVec][e * gu + g] * tau : 0.0f;
    const float last_arrival = has_f(a, kFStateLastGuArrival) ? a.f[kFStateLastGuArrival][e * gu + g] : 0.0f;
    const float last_outflow = has_f(a, kFStateLastGuOutflow) ? a.f[kFStateLastGuOutflow][e * gu + g] : 0.0f;
    const float drop = has_f(a, kFStateGuDrop) ? a.f[kFStateGuDrop][e * gu + g] : 0.0f;
    const float queue_steps = q / gu_flow_ref;
    const float expected_steps = expected / gu_flow_ref;
    const float last_arrival_steps = last_arrival / gu_flow_ref;
    const float last_outflow_steps = last_outflow / gu_flow_ref;
    const float last_drop_steps = drop / gu_flow_ref;
    const float demand_steps = queue_steps + expected_steps;
    const float last_cost = gu_cost_last_route(a, e, g);
    const float workload = fmaxf(last_cost * q, 0.0f);
    const float stability = accel_owner_stability_margin(a, stage_slot, e, g, map_size);
    const float boundary_weight = ucount > 1 ? expf(-stability) : 0.0f;
    const float owner_gain = a.f[stage_f(stage_slot, kSfAccessGainMatrix)][(e * gu + g) * ucount + owner];
    const float owner_se = accel_access_se_from_gain(a, owner_gain, access_noise_ref);
    const float weak_weight = 1.0f / (1.0f + owner_se);
    const float ux = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + owner) * 2 + 0];
    const float uy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + owner) * 2 + 1];
    const float gx = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + 0];
    const float gy = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + 1];
    const float relx = (gx - ux) / map_size;
    const float rely = (gy - uy) / map_size;

    switch (field) {
      case kAccelCellGuCountFrac: sum += 1.0f; break;
      case kAccelCellQueueStepsSum: sum += queue_steps; break;
      case kAccelCellExpectedArrivalStepsSum: sum += expected_steps; break;
      case kAccelCellLastArrivalStepsSum: sum += last_arrival_steps; break;
      case kAccelCellLastOutflowStepsSum: sum += last_outflow_steps; break;
      case kAccelCellLastDropStepsSum: sum += last_drop_steps; break;
      case kAccelCellLastWorkloadLog1pSum: sum += workload; break;
      case kAccelCellBoundaryWorkloadSum: sum += workload * boundary_weight; break;
      case kAccelCellWeakLinkWorkloadSum: sum += workload * weak_weight; break;
      case kAccelCellAccessPressure: sum += fmaxf(demand_steps - last_outflow_steps, 0.0f) + last_drop_steps; break;
      case kAccelCellDemandMomentX:
      case kAccelCellDemandMomentY:
        denom += demand_steps;
        mx += demand_steps * relx;
        my += demand_steps * rely;
        break;
      case kAccelCellBoundaryMomentX:
      case kAccelCellBoundaryMomentY: {
        const float w = workload * boundary_weight;
        denom += w;
        mx += w * relx;
        my += w * rely;
        break;
      }
      case kAccelCellWeakLinkMomentX:
      case kAccelCellWeakLinkMomentY: {
        const float w = workload * weak_weight;
        denom += w;
        mx += w * relx;
        my += w * rely;
        break;
      }
      default:
        break;
    }
  }
  if (field >= kAccelCellDemandMomentX && field <= kAccelCellWeakLinkMomentY) {
    return denom > 0.0f ? ((field == kAccelCellDemandMomentX || field == kAccelCellBoundaryMomentX || field == kAccelCellWeakLinkMomentX) ? mx / denom : my / denom) : 0.0f;
  }
  const float normalized = sum / fmaxf(static_cast<float>(gu), 1.0f);
  switch (field) {
    case kAccelCellQueueStepsSum:
    case kAccelCellExpectedArrivalStepsSum:
    case kAccelCellLastArrivalStepsSum:
    case kAccelCellLastOutflowStepsSum:
    case kAccelCellLastDropStepsSum:
    case kAccelCellLastWorkloadLog1pSum:
    case kAccelCellBoundaryWorkloadSum:
    case kAccelCellWeakLinkWorkloadSum:
      return log1p_nonnegative(normalized);
    default:
      return normalized;
  }
}

__device__ float block_reduce_sum_128(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) shared[threadIdx.x] += shared[threadIdx.x + offset];
    __syncthreads();
  }
  return shared[0];
}

__device__ float block_reduce_max_128(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + offset]);
    __syncthreads();
  }
  return shared[0];
}

__device__ void copy_state_to_stage_base_parallel(const PackedAbi& a, int stage_slot, int e, int stage_id) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  if (threadIdx.x == 0) {
    a.l[stage_l(stage_slot, kSlStageId)][e] = static_cast<int64_t>(stage_id);
    a.f[stage_f(stage_slot, kSfEffectiveBSatTotal)][e] = fp(a, kFpBwEffectiveBSatTotal, fp(a, kFpBSatTotal, 0.0f));
  }
  for (int idx = threadIdx.x; idx < ucount * 2; idx += blockDim.x) {
    const int u = idx / 2;
    const int d = idx - u * 2;
    a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + d] = a.f[kFStateUavPos][(e * ucount + u) * 2 + d];
    a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + d] = a.f[kFStateUavVel][(e * ucount + u) * 2 + d];
  }
  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    a.f[stage_f(stage_slot, kSfUavEnergy)][e * ucount + u] = a.f[kFStateUavEnergy][e * ucount + u];
    a.f[stage_f(stage_slot, kSfUavQueue)][e * ucount + u] = a.f[kFStateUavQueue][e * ucount + u];
  }
  for (int idx = threadIdx.x; idx < gu * 2; idx += blockDim.x) {
    const int g = idx / 2;
    const int d = idx - g * 2;
    a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + d] = a.f[kFStateGuPos][(e * gu + g) * 2 + d];
  }
  for (int g = threadIdx.x; g < gu; g += blockDim.x) {
    a.f[stage_f(stage_slot, kSfGuQueue)][e * gu + g] = a.f[kFStateGuQueue][e * gu + g];
    a.l[stage_l(stage_slot, kSlPrevAssociation)][e * gu + g] =
        has_i(a, kIStateLastAssociation) ? static_cast<int64_t>(a.i[kIStateLastAssociation][e * gu + g]) : -1;
  }
  for (int s = threadIdx.x; s < sat; s += blockDim.x) {
    a.f[stage_f(stage_slot, kSfSatQueue)][e * sat + s] = a.f[kFStateSatQueue][e * sat + s];
    a.f[stage_f(stage_slot, kSfSatLoads)][e * sat + s] =
        has_f(a, kFStateLastSatConnectionCounts) ? a.f[kFStateLastSatConnectionCounts][e * sat + s] : 0.0f;
  }
  for (int idx = threadIdx.x; idx < sat * 3; idx += blockDim.x) {
    const int s = idx / 3;
    const int d = idx - s * 3;
    a.f[stage_f(stage_slot, kSfSatPos)][(e * sat + s) * 3 + d] = a.f[kFStateSatPos][(e * sat + s) * 3 + d];
    a.f[stage_f(stage_slot, kSfSatVel)][(e * sat + s) * 3 + d] = a.f[kFStateSatVel][(e * sat + s) * 3 + d];
  }
}

__device__ int orbit_table_steps(const PackedAbi& a) {
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int64_t row_width = static_cast<int64_t>(sat) * 3;
  if (sat <= 0 || row_width <= 0 || !has_f(a, kFOrbitPosTable) || !has_f(a, kFOrbitVelTable)) return 0;
  const int64_t pos_steps = a.f_numel[kFOrbitPosTable] / row_width;
  const int64_t vel_steps = a.f_numel[kFOrbitVelTable] / row_width;
  const int64_t steps = pos_steps < vel_steps ? pos_steps : vel_steps;
  if (steps <= 0) return 0;
  return steps > 2147483647LL ? 2147483647 : static_cast<int>(steps);
}

__device__ void sync_state_orbit_from_t_parallel(const PackedAbi& a, int e, int t) {
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int steps = orbit_table_steps(a);
  if (sat <= 0 || steps <= 0 || !has_f(a, kFStateSatPos) || !has_f(a, kFStateSatVel)) return;
  int table_t = t;
  if (table_t < 0) table_t = 0;
  if (table_t >= steps) table_t = steps - 1;
  const int64_t row_width = static_cast<int64_t>(sat) * 3;
  for (int idx = threadIdx.x; idx < sat * 3; idx += blockDim.x) {
    const int64_t src = static_cast<int64_t>(table_t) * row_width + idx;
    const int64_t dst = (static_cast<int64_t>(e) * sat * 3) + idx;
    if (src < a.f_numel[kFOrbitPosTable] && dst < a.f_numel[kFStateSatPos]) {
      a.f[kFStateSatPos][dst] = a.f[kFOrbitPosTable][src];
    }
    if (src < a.f_numel[kFOrbitVelTable] && dst < a.f_numel[kFStateSatVel]) {
      a.f[kFStateSatVel][dst] = a.f[kFOrbitVelTable][src];
    }
  }
}

enum GuProxyFeatureCode : int {
  kGuProxyArrivalRate = 0,
  kGuProxyRecentArrival = 1,
  kGuProxyRecentService = 2,
  kGuProxyQueueHeadroom = 3,
  kGuProxyLocalGuServiceCost = 4,
  kGuProxyAssocUavCost = 5,
  kGuProxyAssocSatCostMean = 6,
  kGuProxyWeightedQueueCost = 7,
  kGuProxyWeightedQueueCostRelative = 8,
  kGuProxyUrgencyRisk = 9,
  kGuProxyDownstreamPressure = 10,
  kGuProxyServiceGap = 11,
  kGuProxyServiceGapRisk = 12,
  kGuProxyDeadlineSlack = 13,
  kGuProxyDeadlineRisk = 14,
};

__device__ float gu_proxy_feature_value(const PackedAbi& a, int stage_slot, int e, int g, int p);
__device__ float sat_cost_current(const PackedAbi& a, int e, int s);
__device__ float mean_sat_cost_current(const PackedAbi& a, int e);
__device__ float uav_cost_current(const PackedAbi& a, int e, int u);
__device__ float mean_uav_cost_current(const PackedAbi& a, int e);
__device__ void refresh_cost_cache_fill_base_parallel(const PackedAbi& a, int e, float* cache);
__device__ void refresh_cost_cache_fill_gu_parallel(const PackedAbi& a, int stage_slot, int e, float* cache);
__device__ float gu_proxy_feature_value_cached(const PackedAbi& a, int stage_slot, int e, int g, int p, float* cache);
__device__ float refresh_cached_sat_cost(const PackedAbi& a, float* cache, int s);
__device__ float refresh_cached_uav_cost(const PackedAbi& a, float* cache, int u);
__device__ float refresh_cached_gu_cost(const PackedAbi& a, float* cache, int g);

__device__ void refresh_stage_derived_parallel(
    const PackedAbi& a,
    int stage_slot,
    int e,
    float* refresh_profile_out = nullptr,
    int refresh_profile_stride = 0,
    unsigned long long* refresh_profile_clock = nullptr,
    float* refresh_cost_cache = nullptr) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int users_obs = static_cast<int>(ip(a, kParamUsersObsMax));
  const int visible_max = static_cast<int>(ip(a, kParamVisibleSatsMax));
  const int active = active_width(a);
  const int proxy_dim = static_cast<int>(ip(a, kParamUserProxyDim));
  const float queue_max_gu = positive_config_scale(fp(a, kFpQueueMaxGu, 1.0f));
  const float queue_max_uav = positive_config_scale(fp(a, kFpQueueMaxUav, 1.0f));
  const float queue_max_sat = positive_config_scale(fp(a, kFpQueueMaxSat, 1.0f));
  const float backhaul_gain = positive_coeff(fp(a, kFpBwBackhaulGainConst, 1.0f));
  __shared__ float sh_uav_cost_mean_current;
  __shared__ float sh_sat_cost_mean_current;

  if (refresh_cost_cache != nullptr) {
    refresh_cost_cache_fill_base_parallel(a, e, refresh_cost_cache);
    if (threadIdx.x == 0) {
      float* scalars = refresh_cost_cache + sat + 2 * ucount + 5 * gu;
      sh_sat_cost_mean_current = scalars[kRefreshCostScalarMeanSat];
      sh_uav_cost_mean_current = scalars[kRefreshCostScalarMeanUav];
    }
  } else if (threadIdx.x == 0) {
      sh_sat_cost_mean_current = mean_sat_cost_current(a, e);
      sh_uav_cost_mean_current = mean_uav_cost_current(a, e);
  }
  __syncthreads();
  finish_profile_mark(
      refresh_profile_out,
      refresh_profile_stride,
      e,
      kFinishProfRefreshMeanCost,
      refresh_profile_clock);

  for (int g = threadIdx.x; g < gu; g += blockDim.x) {
    const int assoc = nearest_uav_for_gu(a, stage_slot, e, g);
    a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g] = assoc;
  }
  __syncthreads();
  if (refresh_cost_cache != nullptr) {
    refresh_cost_cache_fill_gu_parallel(a, stage_slot, e, refresh_cost_cache);
  }
  finish_profile_mark(
      refresh_profile_out,
      refresh_profile_stride,
      e,
      kFinishProfRefreshAssoc,
      refresh_profile_clock);

  for (int idx = threadIdx.x; idx < ucount * gu; idx += blockDim.x) {
    const int u = idx / gu;
    const int g = idx - u * gu;
    const int idx_ug = (e * ucount + u) * gu + g;
    const int64_t prev_assoc = a.l[stage_l(stage_slot, kSlPrevAssociation)][e * gu + g];
    a.f[stage_f(stage_slot, kSfCandidateFlag)][idx_ug] = 0.0f;
    a.f[stage_f(stage_slot, kSfBwValidFlag)][idx_ug] = 0.0f;
    a.f[stage_f(stage_slot, kSfPrevAssocFlag)][idx_ug] = (prev_assoc == u) ? 1.0f : 0.0f;
    a.f[stage_f(stage_slot, kSfEtaRefFeature)][idx_ug] = 0.0f;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < ucount * users_obs; idx += blockDim.x) {
    const int u = idx / users_obs;
    const int c = idx - u * users_obs;
    int valid_count = 0;
    int gid = select_candidate_gid_for_slot(a, stage_slot, e, u, c, &valid_count);
    bool valid = gid >= 0 && gid < gu && c < valid_count;
    const int idx_c = (e * ucount + u) * users_obs + c;
    a.l[stage_l(stage_slot, kSlCandidateIndices)][idx_c] = valid ? gid : -1;
    a.b[stage_b(stage_slot, 0)][idx_c] = valid;
    a.f[stage_f(stage_slot, kSfBwValidMask)][idx_c] = valid ? 1.0f : 0.0f;
    float eta = 0.0f;
    if (valid) {
      const float gx = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + gid) * 2 + 0];
      const float gy = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + gid) * 2 + 1];
      const float ux = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 0];
      const float uy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 1];
      float gain = simple_access_gain(a, gx - ux, gy - uy);
      if (fp(a, kFpAccessFadingModeCode, 0.0f) > 1.5f && has_f(a, kFRandomFadingGain)) {
        gain *= fmaxf(a.f[kFRandomFadingGain][(e * gu + gid) * ucount + u], 0.0f);
      }
      a.f[stage_f(stage_slot, kSfAccessGainMatrix)][(e * gu + gid) * ucount + u] = gain;
      eta = simple_eta_from_gain(a, gain);
      const int idx_ug = (e * ucount + u) * gu + gid;
      a.f[stage_f(stage_slot, kSfCandidateFlag)][idx_ug] = 1.0f;
      a.f[stage_f(stage_slot, kSfBwValidFlag)][idx_ug] = 1.0f;
      a.f[stage_f(stage_slot, kSfEtaRefFeature)][idx_ug] = eta;
    }
    a.f[stage_f(stage_slot, kSfEtaSlots)][idx_c] = eta;
  }
  finish_profile_mark(
      refresh_profile_out,
      refresh_profile_stride,
      e,
      kFinishProfRefreshCandidateAccess,
      refresh_profile_clock);

  for (int idx = threadIdx.x; idx < gu * ucount; idx += blockDim.x) {
    const int g = idx / ucount;
    const int u = idx - g * ucount;
    const float gx = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + 0];
    const float gy = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + 1];
    const float ux = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 0];
    const float uy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 1];
    float gain = simple_access_gain(a, gx - ux, gy - uy);
    if (fp(a, kFpAccessFadingModeCode, 0.0f) > 1.5f && has_f(a, kFRandomFadingGain)) {
      gain *= fmaxf(a.f[kFRandomFadingGain][(e * gu + g) * ucount + u], 0.0f);
    }
    a.f[stage_f(stage_slot, kSfAccessGainMatrix)][(e * gu + g) * ucount + u] = gain;
  }
  finish_profile_mark(
      refresh_profile_out,
      refresh_profile_stride,
      e,
      kFinishProfRefreshFullAccessGain,
      refresh_profile_clock);

  for (int idx = threadIdx.x; idx < gu * proxy_dim; idx += blockDim.x) {
    const int g = idx / max(proxy_dim, 1);
    const int p = idx - g * max(proxy_dim, 1);
    if (g < gu && p < proxy_dim) {
      const float value = gu_proxy_feature_value_cached(a, stage_slot, e, g, p, refresh_cost_cache);
      a.f[stage_f(stage_slot, kSfGuProxyFeatures)][(e * gu + g) * proxy_dim + p] = value;
    }
  }
  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    const float cost = refresh_cost_cache != nullptr ? refresh_cached_uav_cost(a, refresh_cost_cache, u) : uav_cost_current(a, e, u);
    a.f[stage_f(stage_slot, kSfUavAssocUavCost)][e * ucount + u] =
        logf(log_argument(positive_coeff(cost) / positive_coeff(sh_uav_cost_mean_current)));
  }
  for (int s = threadIdx.x; s < sat; s += blockDim.x) {
    const float cost = refresh_cost_cache != nullptr ? refresh_cached_sat_cost(a, refresh_cost_cache, s) : sat_cost_current(a, e, s);
    a.f[stage_f(stage_slot, kSfSatCostNorm)][e * sat + s] =
        logf(log_argument(positive_coeff(cost) / positive_coeff(sh_sat_cost_mean_current)));
  }
  finish_profile_mark(
      refresh_profile_out,
      refresh_profile_stride,
      e,
      kFinishProfRefreshProxyAndCost,
      refresh_profile_clock);
  for (int idx = threadIdx.x; idx < ucount * sat; idx += blockDim.x) {
    const int u = idx / sat;
    const int s = idx - u * sat;
    const int out = (e * ucount + u) * sat + s;
    float rel[3];
    float relv[3];
    sat_rel_for_stage_us(a, stage_slot, e, u, s, rel, relv);
    const float dist2 = geometry_denominator(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]);
    const float elevation = sat_elevation_from_rel_device(a, rel);
    const float gain = positive_coeff(fp(a, kFpBwBackhaulGainConst, 1.0f)) / fmaxf(dist2, 1.0f) *
        atmospheric_loss_factor_device(a, elevation);
    const float nu_eff = doppler_eff_hz(a, e, u, s, rel, relv);
    const bool above = elevation >= fp(a, kFpThetaMinRad, -3.1415926f);
    const bool doppler_ok = !ip(a, kParamDopplerEnabled) || fabsf(nu_eff) <= fmaxf(fp(a, kFpNuMax, 1.0f), 1.0f);
    a.f[stage_f(stage_slot, kSfElevationMatrix)][out] = elevation;
    a.f[stage_f(stage_slot, kSfVisibleFlagAll)][out] = 0.0f;
    if (ip(a, kParamSatCandidateMode) != 0) {
      if (has_f(a, stage_f(stage_slot, kSfUsRelPosAll))) {
        for (int d = 0; d < 3; ++d) {
          a.f[stage_f(stage_slot, kSfUsRelPosAll)][out * 3 + d] = rel[d];
          a.f[stage_f(stage_slot, kSfUsRelVelAll)][out * 3 + d] = relv[d];
        }
      }
      if (has_f(a, stage_f(stage_slot, kSfUsGainAll))) {
        a.f[stage_f(stage_slot, kSfUsGainAll)][out] = gain;
        a.f[stage_f(stage_slot, kSfUsNuEffAll)][out] = nu_eff;
        a.f[stage_f(stage_slot, kSfUsValidFlagAll)][out] = (above && doppler_ok) ? 1.0f : 0.0f;
        a.f[stage_f(stage_slot, kSfUsSatQueueAll)][out] = safe_div(a.f[stage_f(stage_slot, kSfSatQueue)][e * sat + s], queue_max_sat);
      }
    }
  }
  __syncthreads();
  finish_profile_mark(
      refresh_profile_out,
      refresh_profile_stride,
      e,
      kFinishProfRefreshFullUsGeometry,
      refresh_profile_clock);

  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    float elev_min = 3.402823466e38f;
    float elev_max = -3.402823466e38f;
    float se_min = 3.402823466e38f;
    float se_max = -3.402823466e38f;
    bool any_above = false;
    if (ip(a, kParamSatCandidateMode) != 0) {
      for (int s = 0; s < sat; ++s) {
        const int flag_idx = (e * ucount + u) * sat + s;
        const float elevation = a.f[stage_f(stage_slot, kSfElevationMatrix)][flag_idx];
        if (elevation < fp(a, kFpThetaMinRad, -3.1415926f)) continue;
        const float se = sat_score_se_cached_or_stage_us(a, stage_slot, e, u, s, elevation);
        elev_min = fminf(elev_min, elevation);
        elev_max = fmaxf(elev_max, elevation);
        se_min = fminf(se_min, se);
        se_max = fmaxf(se_max, se);
        any_above = true;
      }
    }
    if (visible_max <= kMaxVisibleTopK) {
      int top_sid[kMaxVisibleTopK];
      float top_score[kMaxVisibleTopK];
      for (int j = 0; j < visible_max; ++j) {
        top_sid[j] = -1;
        top_score[j] = -3.402823466e38f;
      }
      for (int s = 0; s < sat; ++s) {
        const int flag_idx = (e * ucount + u) * sat + s;
        const float elevation = a.f[stage_f(stage_slot, kSfElevationMatrix)][flag_idx];
        if (elevation < fp(a, kFpThetaMinRad, -3.1415926f)) continue;
        float score = elevation;
        if (ip(a, kParamSatCandidateMode) != 0 && any_above) {
          const float elev_span = elev_max - elev_min;
          const float se = sat_score_se_cached_or_stage_us(a, stage_slot, e, u, s, elevation);
          const float se_span = se_max - se_min;
          const float elev_norm = elev_span > kNormDenomEps ? (elevation - elev_min) / elev_span : 0.0f;
          const float se_norm = se_span > kNormDenomEps ? (se - se_min) / se_span : 0.0f;
          const float queue_norm = safe_div(a.f[stage_f(stage_slot, kSfSatQueue)][e * sat + s], queue_max_sat);
          score = fp(a, kFpSatCandidateElevationWeight, 1.0f) * elev_norm +
              fp(a, kFpSatCandidateSeWeight, 0.0f) * se_norm -
              fp(a, kFpSatCandidateQueueWeight, 0.0f) * queue_norm;
        }
        int insert_pos = -1;
        for (int j = 0; j < visible_max; ++j) {
          if (score > top_score[j] || (score == top_score[j] && (top_sid[j] < 0 || s < top_sid[j]))) {
            insert_pos = j;
            break;
          }
        }
        if (insert_pos >= 0) {
          for (int j = visible_max - 1; j > insert_pos; --j) {
            top_sid[j] = top_sid[j - 1];
            top_score[j] = top_score[j - 1];
          }
          top_sid[insert_pos] = s;
          top_score[insert_pos] = score;
        }
      }
      for (int j = 0; j < visible_max; ++j) {
        const int sid = top_sid[j];
        const int out = (e * ucount + u) * visible_max + j;
        a.l[stage_l(stage_slot, kSlVisibleIds)][out] = sid;
        a.b[stage_b(stage_slot, 1)][out] = sid >= 0;
        if (sid >= 0) {
          a.f[stage_f(stage_slot, kSfVisibleFlagAll)][(e * ucount + u) * sat + sid] = 1.0f;
        }
      }
    } else {
      for (int j = 0; j < visible_max; ++j) {
        int best_sid = -1;
        float best_score = -3.402823466e38f;
        for (int s = 0; s < sat; ++s) {
          const int flag_idx = (e * ucount + u) * sat + s;
          if (a.f[stage_f(stage_slot, kSfVisibleFlagAll)][flag_idx] > 0.5f) continue;
          const float elevation = a.f[stage_f(stage_slot, kSfElevationMatrix)][flag_idx];
          if (elevation < fp(a, kFpThetaMinRad, -3.1415926f)) continue;
          float score = elevation;
          if (ip(a, kParamSatCandidateMode) != 0 && any_above) {
            const float elev_span = elev_max - elev_min;
            const float se = sat_score_se_cached_or_stage_us(a, stage_slot, e, u, s, elevation);
            const float se_span = se_max - se_min;
            const float elev_norm = elev_span > kNormDenomEps ? (elevation - elev_min) / elev_span : 0.0f;
            const float se_norm = se_span > kNormDenomEps ? (se - se_min) / se_span : 0.0f;
            const float queue_norm = safe_div(a.f[stage_f(stage_slot, kSfSatQueue)][e * sat + s], queue_max_sat);
            score = fp(a, kFpSatCandidateElevationWeight, 1.0f) * elev_norm +
                fp(a, kFpSatCandidateSeWeight, 0.0f) * se_norm -
                fp(a, kFpSatCandidateQueueWeight, 0.0f) * queue_norm;
          }
          if (score > best_score || (score == best_score && (best_sid < 0 || s < best_sid))) {
            best_score = score;
            best_sid = s;
          }
        }
        const int out = (e * ucount + u) * visible_max + j;
        a.l[stage_l(stage_slot, kSlVisibleIds)][out] = best_sid;
        a.b[stage_b(stage_slot, 1)][out] = best_sid >= 0;
        if (best_sid >= 0) {
          a.f[stage_f(stage_slot, kSfVisibleFlagAll)][(e * ucount + u) * sat + best_sid] = 1.0f;
        }
      }
    }
  }
  __syncthreads();
  finish_profile_mark(
      refresh_profile_out,
      refresh_profile_stride,
      e,
      kFinishProfRefreshVisibleTopk,
      refresh_profile_clock);

  for (int aidx = threadIdx.x; aidx < active; aidx += blockDim.x) {
    int selected_sid = -1;
    int rank = 0;
    for (int u = 0; u < ucount && selected_sid < 0; ++u) {
      for (int j = 0; j < visible_max; ++j) {
        const int visible_offset = (e * ucount + u) * visible_max + j;
        if (!a.b[stage_b(stage_slot, 1)][visible_offset]) continue;
        const int sid = static_cast<int>(a.l[stage_l(stage_slot, kSlVisibleIds)][visible_offset]);
        if (sid < 0 || sid >= sat) continue;
        bool duplicate = false;
        for (int prev_u = 0; prev_u <= u && !duplicate; ++prev_u) {
          const int prev_limit = prev_u < u ? visible_max : j;
          for (int prev_j = 0; prev_j < prev_limit; ++prev_j) {
            const int prev_offset = (e * ucount + prev_u) * visible_max + prev_j;
            if (!a.b[stage_b(stage_slot, 1)][prev_offset]) continue;
            const int prev_sid = static_cast<int>(a.l[stage_l(stage_slot, kSlVisibleIds)][prev_offset]);
            if (prev_sid == sid) {
              duplicate = true;
              break;
            }
          }
        }
        if (duplicate) continue;
        if (rank == aidx) {
          selected_sid = sid;
          break;
        }
        ++rank;
      }
    }
    a.l[stage_l(stage_slot, kSlActiveSatIds)][e * active + aidx] = selected_sid;
  }
  finish_profile_mark(
      refresh_profile_out,
      refresh_profile_stride,
      e,
      kFinishProfRefreshActiveSat,
      refresh_profile_clock);
  for (int idx = threadIdx.x; idx < ucount; idx += blockDim.x) {
    const int u = idx;
    const float ux = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 0];
    const float uy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 1];
    const float vx = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 0];
    const float vy = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 1];
    float uecef[3];
    float uvecef[3];
    uav_local_to_ecef_device(a, ux, uy, uecef);
    uav_vel_local_to_ecef_device(a, ux, uy, vx, vy, uvecef);
    for (int d = 0; d < 3; ++d) {
      a.f[stage_f(stage_slot, kSfUavEcefAll)][(e * ucount + u) * 3 + d] = uecef[d];
      a.f[stage_f(stage_slot, kSfUavVelEcefAll)][(e * ucount + u) * 3 + d] = uvecef[d];
    }
  }
  __syncthreads();

  for (int aidx = threadIdx.x; aidx < active; aidx += blockDim.x) {
    const int sid = static_cast<int>(a.l[stage_l(stage_slot, kSlActiveSatIds)][e * active + aidx]);
    for (int d = 0; d < 3; ++d) {
      const float pos = sid >= 0 ? a.f[stage_f(stage_slot, kSfSatPos)][(e * sat + sid) * 3 + d] : 0.0f;
      const float vel = sid >= 0 ? a.f[stage_f(stage_slot, kSfSatVel)][(e * sat + sid) * 3 + d] : 0.0f;
      a.f[stage_f(stage_slot, kSfSatPosActive)][(e * active + aidx) * 3 + d] = pos;
      a.f[stage_f(stage_slot, kSfSatVelActive)][(e * active + aidx) * 3 + d] = vel;
    }
    a.f[stage_f(stage_slot, kSfSatQueueActive)][e * active + aidx] =
        sid >= 0 ? a.f[stage_f(stage_slot, kSfSatQueue)][e * sat + sid] : 0.0f;
    a.f[stage_f(stage_slot, kSfSatLoadActive)][e * active + aidx] =
        sid >= 0 ? a.f[stage_f(stage_slot, kSfSatLoads)][e * sat + sid] : 0.0f;
    a.f[stage_f(stage_slot, kSfSatCostNormActive)][e * active + aidx] =
        sid >= 0 ? a.f[stage_f(stage_slot, kSfSatCostNorm)][e * sat + sid] : 0.0f;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < ucount * active; idx += blockDim.x) {
    const int u = idx / active;
    const int aidx = idx - u * active;
    const int sid = a.l[stage_l(stage_slot, kSlActiveSatIds)][e * active + aidx];
    const int active_us_idx = e * ucount * active + idx;
    const float ux = a.f[stage_f(stage_slot, kSfUavEcefAll)][(e * ucount + u) * 3 + 0];
    const float uy = a.f[stage_f(stage_slot, kSfUavEcefAll)][(e * ucount + u) * 3 + 1];
    const float uz = a.f[stage_f(stage_slot, kSfUavEcefAll)][(e * ucount + u) * 3 + 2];
    const float uvx = a.f[stage_f(stage_slot, kSfUavVelEcefAll)][(e * ucount + u) * 3 + 0];
    const float uvy = a.f[stage_f(stage_slot, kSfUavVelEcefAll)][(e * ucount + u) * 3 + 1];
    const float uvz = a.f[stage_f(stage_slot, kSfUavVelEcefAll)][(e * ucount + u) * 3 + 2];
    float rel[3];
    float relv[3];
    for (int d = 0; d < 3; ++d) {
      const float spos = sid >= 0 ? a.f[stage_f(stage_slot, kSfSatPosActive)][(e * active + aidx) * 3 + d] : 0.0f;
      const float svel = sid >= 0 ? a.f[stage_f(stage_slot, kSfSatVelActive)][(e * active + aidx) * 3 + d] : 0.0f;
      const float upos = (d == 0 ? ux : (d == 1 ? uy : uz));
      const float uvel = (d == 0 ? uvx : (d == 1 ? uvy : uvz));
      rel[d] = spos - upos;
      relv[d] = svel - uvel;
      a.f[stage_f(stage_slot, kSfUsRelPosActive)][active_us_idx * 3 + d] = rel[d];
      a.f[stage_f(stage_slot, kSfUsRelVelActive)][active_us_idx * 3 + d] = relv[d];
      if (ip(a, kParamSatCandidateMode) != 0 && has_f(a, stage_f(stage_slot, kSfUsRelPosAll)) && sid >= 0) {
        a.f[stage_f(stage_slot, kSfUsRelPosAll)][((e * ucount + u) * sat + sid) * 3 + d] = rel[d];
        a.f[stage_f(stage_slot, kSfUsRelVelAll)][((e * ucount + u) * sat + sid) * 3 + d] = relv[d];
      }
    }
    const float dist2 = fmaxf(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2], 1.0f);
    const float elevation = sat_elevation_from_rel_device(a, rel);
    const float gain = (backhaul_gain / dist2) * atmospheric_loss_factor_device(a, elevation);
    const float nu_eff = sid >= 0 ? doppler_eff_hz(a, e, u, sid, rel, relv) : 0.0f;
    const bool visible =
        sid >= 0 && sid < sat && a.f[stage_f(stage_slot, kSfVisibleFlagAll)][(e * ucount + u) * sat + sid] > 0.5f;
    const bool doppler_ok = !ip(a, kParamDopplerEnabled) || fabsf(nu_eff) <= fmaxf(fp(a, kFpNuMax, 1.0f), 1.0f);
    a.f[stage_f(stage_slot, kSfUsGainActive)][active_us_idx] = sid >= 0 ? gain : 0.0f;
    a.f[stage_f(stage_slot, kSfUsNuEffActive)][active_us_idx] = nu_eff;
    a.f[stage_f(stage_slot, kSfVisibleFlagActive)][active_us_idx] = visible ? 1.0f : 0.0f;
    a.f[stage_f(stage_slot, kSfUsValidFlagActive)][active_us_idx] = (visible && doppler_ok) ? 1.0f : 0.0f;
    if (ip(a, kParamSatCandidateMode) != 0 && sid >= 0 && has_f(a, stage_f(stage_slot, kSfUsGainAll))) {
      const int idx_us = (e * ucount + u) * sat + sid;
      a.f[stage_f(stage_slot, kSfUsGainAll)][idx_us] = gain;
      a.f[stage_f(stage_slot, kSfUsNuEffAll)][idx_us] = nu_eff;
      a.f[stage_f(stage_slot, kSfUsValidFlagAll)][idx_us] = (visible && doppler_ok) ? 1.0f : 0.0f;
      a.f[stage_f(stage_slot, kSfUsSatQueueAll)][idx_us] = safe_div(a.f[stage_f(stage_slot, kSfSatQueue)][e * sat + sid], queue_max_sat);
    }
  }
  finish_profile_mark(
      refresh_profile_out,
      refresh_profile_stride,
      e,
      kFinishProfRefreshActiveUs,
      refresh_profile_clock);
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  if (stage_slot != 3) {
    for (int idx = threadIdx.x; idx < ucount * select_k; idx += blockDim.x) {
      const int u = idx / select_k;
      const int k = idx - u * select_k;
      const int64_t last = has_l(a, kLStateLastSatSelectionMatrix) ? a.l[kLStateLastSatSelectionMatrix][(e * ucount + u) * select_k + k] : -1;
      a.l[stage_l(stage_slot, kSlSatSelectionMatrix)][(e * ucount + u) * select_k + k] = (last >= 0 && last < sat) ? last : -1;
    }
  }
  finish_profile_mark(
      refresh_profile_out,
      refresh_profile_stride,
      e,
      kFinishProfRefreshLastSelection,
      refresh_profile_clock);
}

__device__ void refresh_stage_queue_derived_parallel(
    const PackedAbi& a,
    int stage_slot,
    int e,
    float* refresh_profile_out = nullptr,
    int refresh_profile_stride = 0,
    unsigned long long* refresh_profile_clock = nullptr,
    float* refresh_cost_cache = nullptr,
    bool write_stage_fields = true) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int active = active_width(a);
  const int proxy_dim = static_cast<int>(ip(a, kParamUserProxyDim));
  const float queue_max_uav = positive_config_scale(fp(a, kFpQueueMaxUav, 1.0f));
  const float queue_max_sat = positive_config_scale(fp(a, kFpQueueMaxSat, 1.0f));
  __shared__ float sh_uav_cost_mean_current;
  __shared__ float sh_sat_cost_mean_current;

  if (refresh_cost_cache != nullptr) {
    refresh_cost_cache_fill_base_parallel(a, e, refresh_cost_cache);
    refresh_cost_cache_fill_gu_parallel(a, stage_slot, e, refresh_cost_cache);
    if (threadIdx.x == 0) {
      float* scalars = refresh_cost_cache + sat + 2 * ucount + 5 * gu;
      sh_sat_cost_mean_current = scalars[kRefreshCostScalarMeanSat];
      sh_uav_cost_mean_current = scalars[kRefreshCostScalarMeanUav];
    }
  } else if (threadIdx.x == 0) {
    sh_sat_cost_mean_current = mean_sat_cost_current(a, e);
    sh_uav_cost_mean_current = mean_uav_cost_current(a, e);
  }
  __syncthreads();
  finish_profile_mark(
      refresh_profile_out,
      refresh_profile_stride,
      e,
      kFinishProfQueueRefreshMeanCost,
      refresh_profile_clock);

  if (write_stage_fields) {
    for (int idx = threadIdx.x; idx < gu * proxy_dim; idx += blockDim.x) {
      const int g = idx / max(proxy_dim, 1);
      const int p = idx - g * max(proxy_dim, 1);
      if (g < gu && p < proxy_dim) {
        const float value = gu_proxy_feature_value_cached(a, stage_slot, e, g, p, refresh_cost_cache);
        a.f[stage_f(stage_slot, kSfGuProxyFeatures)][(e * gu + g) * proxy_dim + p] = value;
      }
    }

    for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
      const float cost = refresh_cost_cache != nullptr ? refresh_cached_uav_cost(a, refresh_cost_cache, u) : uav_cost_current(a, e, u);
      a.f[stage_f(stage_slot, kSfUavAssocUavCost)][e * ucount + u] =
          logf(log_argument(positive_coeff(cost) / positive_coeff(sh_uav_cost_mean_current)));
    }

    for (int s = threadIdx.x; s < sat; s += blockDim.x) {
      const float cost = refresh_cost_cache != nullptr ? refresh_cached_sat_cost(a, refresh_cost_cache, s) : sat_cost_current(a, e, s);
      a.f[stage_f(stage_slot, kSfSatCostNorm)][e * sat + s] =
          logf(log_argument(positive_coeff(cost) / positive_coeff(sh_sat_cost_mean_current)));
    }
  }
  __syncthreads();
  finish_profile_mark(
      refresh_profile_out,
      refresh_profile_stride,
      e,
      kFinishProfQueueRefreshProxyAndCost,
      refresh_profile_clock);

  if (write_stage_fields) {
    for (int aidx = threadIdx.x; aidx < active; aidx += blockDim.x) {
      const int sid = static_cast<int>(a.l[stage_l(stage_slot, kSlActiveSatIds)][e * active + aidx]);
      a.f[stage_f(stage_slot, kSfSatQueueActive)][e * active + aidx] =
          sid >= 0 ? a.f[stage_f(stage_slot, kSfSatQueue)][e * sat + sid] : 0.0f;
      a.f[stage_f(stage_slot, kSfSatCostNormActive)][e * active + aidx] =
          sid >= 0 ? a.f[stage_f(stage_slot, kSfSatCostNorm)][e * sat + sid] : 0.0f;
    }
  }
  finish_profile_mark(
      refresh_profile_out,
      refresh_profile_stride,
      e,
      kFinishProfQueueRefreshActiveSat,
      refresh_profile_clock);

  if (write_stage_fields && ip(a, kParamSatCandidateMode) != 0 && has_f(a, stage_f(stage_slot, kSfUsSatQueueAll))) {
    for (int idx = threadIdx.x; idx < ucount * sat; idx += blockDim.x) {
      const int u = idx / sat;
      const int s = idx - u * sat;
      const int out = (e * ucount + u) * sat + s;
      a.f[stage_f(stage_slot, kSfUsSatQueueAll)][out] =
          safe_div(a.f[stage_f(stage_slot, kSfSatQueue)][e * sat + s], queue_max_sat);
    }
  }
  finish_profile_mark(
      refresh_profile_out,
      refresh_profile_stride,
      e,
      kFinishProfQueueRefreshFullUsSatQueue,
      refresh_profile_clock);
}

__device__ void prepare_stage_from_state_parallel(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int stage_id,
    float* refresh_profile_out = nullptr,
    int refresh_profile_stride = 0,
    unsigned long long* refresh_profile_clock = nullptr,
    float* refresh_cost_cache = nullptr) {
  copy_state_to_stage_base_parallel(a, stage_slot, e, stage_id);
  __syncthreads();
  refresh_stage_derived_parallel(
      a,
      stage_slot,
      e,
      refresh_profile_out,
      refresh_profile_stride,
      refresh_profile_clock,
      refresh_cost_cache);
}

__device__ void copy_stage_env_parallel(const PackedAbi& a, int src_slot, int dst_slot, int e, int dst_stage_id) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int users_obs = static_cast<int>(ip(a, kParamUsersObsMax));
  const int visible = static_cast<int>(ip(a, kParamVisibleSatsMax));
  const int active = active_width(a);
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  const int proxy_dim = static_cast<int>(ip(a, kParamUserProxyDim));
  if (threadIdx.x == 0) {
    a.l[stage_l(dst_slot, kSlStageId)][e] = dst_stage_id;
    a.f[stage_f(dst_slot, kSfEffectiveBSatTotal)][e] = a.f[stage_f(src_slot, kSfEffectiveBSatTotal)][e];
  }
  for (int idx = threadIdx.x; idx < ucount * 2; idx += blockDim.x) {
    const int u = idx / 2;
    const int d = idx - u * 2;
    a.f[stage_f(dst_slot, kSfUavPos)][(e * ucount + u) * 2 + d] = a.f[stage_f(src_slot, kSfUavPos)][(e * ucount + u) * 2 + d];
    a.f[stage_f(dst_slot, kSfUavVel)][(e * ucount + u) * 2 + d] = a.f[stage_f(src_slot, kSfUavVel)][(e * ucount + u) * 2 + d];
  }
  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    a.f[stage_f(dst_slot, kSfUavEnergy)][e * ucount + u] = a.f[stage_f(src_slot, kSfUavEnergy)][e * ucount + u];
    a.f[stage_f(dst_slot, kSfUavQueue)][e * ucount + u] = a.f[stage_f(src_slot, kSfUavQueue)][e * ucount + u];
    a.f[stage_f(dst_slot, kSfUavAssocUavCost)][e * ucount + u] = a.f[stage_f(src_slot, kSfUavAssocUavCost)][e * ucount + u];
  }
  for (int idx = threadIdx.x; idx < ucount * users_obs; idx += blockDim.x) {
    const int u = idx / users_obs;
    const int c = idx - u * users_obs;
    const int out = (e * ucount + u) * users_obs + c;
    a.l[stage_l(dst_slot, kSlCandidateIndices)][out] = a.l[stage_l(src_slot, kSlCandidateIndices)][out];
    a.b[stage_b(dst_slot, 0)][out] = a.b[stage_b(src_slot, 0)][out];
    a.f[stage_f(dst_slot, kSfBwValidMask)][out] = a.f[stage_f(src_slot, kSfBwValidMask)][out];
    a.f[stage_f(dst_slot, kSfEtaSlots)][out] = a.f[stage_f(src_slot, kSfEtaSlots)][out];
  }
  for (int idx = threadIdx.x; idx < ucount * visible; idx += blockDim.x) {
    const int u = idx / visible;
    const int j = idx - u * visible;
    const int out = (e * ucount + u) * visible + j;
    a.l[stage_l(dst_slot, kSlVisibleIds)][out] = a.l[stage_l(src_slot, kSlVisibleIds)][out];
    a.b[stage_b(dst_slot, 1)][out] = a.b[stage_b(src_slot, 1)][out];
  }
  for (int idx = threadIdx.x; idx < ucount * select_k; idx += blockDim.x) {
    const int out = (e * ucount) * select_k + idx;
    a.l[stage_l(dst_slot, kSlSatSelectionMatrix)][out] = a.l[stage_l(src_slot, kSlSatSelectionMatrix)][out];
  }
  for (int idx = threadIdx.x; idx < gu * 2; idx += blockDim.x) {
    const int g = idx / 2;
    const int d = idx - g * 2;
    a.f[stage_f(dst_slot, kSfGuPos)][(e * gu + g) * 2 + d] = a.f[stage_f(src_slot, kSfGuPos)][(e * gu + g) * 2 + d];
  }
  for (int g = threadIdx.x; g < gu; g += blockDim.x) {
    a.f[stage_f(dst_slot, kSfGuQueue)][e * gu + g] = a.f[stage_f(src_slot, kSfGuQueue)][e * gu + g];
    a.l[stage_l(dst_slot, kSlAssoc)][e * gu + g] = a.l[stage_l(src_slot, kSlAssoc)][e * gu + g];
    a.l[stage_l(dst_slot, kSlPrevAssociation)][e * gu + g] = a.l[stage_l(src_slot, kSlPrevAssociation)][e * gu + g];
  }
  for (int idx = threadIdx.x; idx < gu * proxy_dim; idx += blockDim.x) {
    const int g = idx / max(proxy_dim, 1);
    const int p = idx - g * max(proxy_dim, 1);
    if (g < gu && p < proxy_dim) {
      a.f[stage_f(dst_slot, kSfGuProxyFeatures)][(e * gu + g) * proxy_dim + p] =
          a.f[stage_f(src_slot, kSfGuProxyFeatures)][(e * gu + g) * proxy_dim + p];
    }
  }
  for (int idx = threadIdx.x; idx < ucount * gu; idx += blockDim.x) {
    const int u = idx / gu;
    const int g = idx - u * gu;
    const int out = (e * ucount + u) * gu + g;
    a.f[stage_f(dst_slot, kSfCandidateFlag)][out] = a.f[stage_f(src_slot, kSfCandidateFlag)][out];
    a.f[stage_f(dst_slot, kSfBwValidFlag)][out] = a.f[stage_f(src_slot, kSfBwValidFlag)][out];
    a.f[stage_f(dst_slot, kSfPrevAssocFlag)][out] = a.f[stage_f(src_slot, kSfPrevAssocFlag)][out];
    a.f[stage_f(dst_slot, kSfEtaRefFeature)][out] = a.f[stage_f(src_slot, kSfEtaRefFeature)][out];
    a.f[stage_f(dst_slot, kSfAccessGainMatrix)][(e * gu + g) * ucount + u] =
        a.f[stage_f(src_slot, kSfAccessGainMatrix)][(e * gu + g) * ucount + u];
  }
  for (int s = threadIdx.x; s < sat; s += blockDim.x) {
    a.f[stage_f(dst_slot, kSfSatQueue)][e * sat + s] = a.f[stage_f(src_slot, kSfSatQueue)][e * sat + s];
    a.f[stage_f(dst_slot, kSfSatLoads)][e * sat + s] = a.f[stage_f(src_slot, kSfSatLoads)][e * sat + s];
    a.f[stage_f(dst_slot, kSfSatCostNorm)][e * sat + s] = a.f[stage_f(src_slot, kSfSatCostNorm)][e * sat + s];
  }
  for (int idx = threadIdx.x; idx < sat * 3; idx += blockDim.x) {
    const int s = idx / 3;
    const int d = idx - s * 3;
    a.f[stage_f(dst_slot, kSfSatPos)][(e * sat + s) * 3 + d] = a.f[stage_f(src_slot, kSfSatPos)][(e * sat + s) * 3 + d];
    a.f[stage_f(dst_slot, kSfSatVel)][(e * sat + s) * 3 + d] = a.f[stage_f(src_slot, kSfSatVel)][(e * sat + s) * 3 + d];
  }
  for (int idx = threadIdx.x; idx < ucount * sat; idx += blockDim.x) {
    const int u = idx / sat;
    const int s = idx - u * sat;
    const int out = (e * ucount + u) * sat + s;
    a.f[stage_f(dst_slot, kSfVisibleFlagAll)][out] = a.f[stage_f(src_slot, kSfVisibleFlagAll)][out];
    a.f[stage_f(dst_slot, kSfElevationMatrix)][out] = a.f[stage_f(src_slot, kSfElevationMatrix)][out];
  }
  for (int idx = threadIdx.x; idx < ucount * 3; idx += blockDim.x) {
    const int u = idx / 3;
    const int d = idx - u * 3;
    a.f[stage_f(dst_slot, kSfUavEcefAll)][(e * ucount + u) * 3 + d] = a.f[stage_f(src_slot, kSfUavEcefAll)][(e * ucount + u) * 3 + d];
    a.f[stage_f(dst_slot, kSfUavVelEcefAll)][(e * ucount + u) * 3 + d] = a.f[stage_f(src_slot, kSfUavVelEcefAll)][(e * ucount + u) * 3 + d];
  }
  for (int aidx = threadIdx.x; aidx < active; aidx += blockDim.x) {
    a.l[stage_l(dst_slot, kSlActiveSatIds)][e * active + aidx] = a.l[stage_l(src_slot, kSlActiveSatIds)][e * active + aidx];
    a.f[stage_f(dst_slot, kSfSatQueueActive)][e * active + aidx] = a.f[stage_f(src_slot, kSfSatQueueActive)][e * active + aidx];
    a.f[stage_f(dst_slot, kSfSatLoadActive)][e * active + aidx] = a.f[stage_f(src_slot, kSfSatLoadActive)][e * active + aidx];
    a.f[stage_f(dst_slot, kSfSatCostNormActive)][e * active + aidx] = a.f[stage_f(src_slot, kSfSatCostNormActive)][e * active + aidx];
  }
  for (int idx = threadIdx.x; idx < active * 3; idx += blockDim.x) {
    const int aidx = idx / 3;
    const int d = idx - aidx * 3;
    a.f[stage_f(dst_slot, kSfSatPosActive)][(e * active + aidx) * 3 + d] = a.f[stage_f(src_slot, kSfSatPosActive)][(e * active + aidx) * 3 + d];
    a.f[stage_f(dst_slot, kSfSatVelActive)][(e * active + aidx) * 3 + d] = a.f[stage_f(src_slot, kSfSatVelActive)][(e * active + aidx) * 3 + d];
  }
  for (int idx = threadIdx.x; idx < ucount * active; idx += blockDim.x) {
    a.f[stage_f(dst_slot, kSfUsGainActive)][(e * ucount) * active + idx] = a.f[stage_f(src_slot, kSfUsGainActive)][(e * ucount) * active + idx];
    a.f[stage_f(dst_slot, kSfUsNuEffActive)][(e * ucount) * active + idx] = a.f[stage_f(src_slot, kSfUsNuEffActive)][(e * ucount) * active + idx];
    a.f[stage_f(dst_slot, kSfVisibleFlagActive)][(e * ucount) * active + idx] = a.f[stage_f(src_slot, kSfVisibleFlagActive)][(e * ucount) * active + idx];
    a.f[stage_f(dst_slot, kSfUsValidFlagActive)][(e * ucount) * active + idx] = a.f[stage_f(src_slot, kSfUsValidFlagActive)][(e * ucount) * active + idx];
  }
  for (int idx = threadIdx.x; idx < ucount * active * 3; idx += blockDim.x) {
    a.f[stage_f(dst_slot, kSfUsRelPosActive)][(e * ucount) * active * 3 + idx] = a.f[stage_f(src_slot, kSfUsRelPosActive)][(e * ucount) * active * 3 + idx];
    a.f[stage_f(dst_slot, kSfUsRelVelActive)][(e * ucount) * active * 3 + idx] = a.f[stage_f(src_slot, kSfUsRelVelActive)][(e * ucount) * active * 3 + idx];
  }
}

__device__ void write_accel_obs_parallel(
    const PackedAbi& a,
    int stage_slot,
    int live_index,
    int e,
    float* accel_cell_summary_cache = nullptr,
    float* last_route_cost_cache = nullptr,
    float* accel_obs_profile_out = nullptr,
    int accel_obs_profile_stride = 0,
    unsigned long long* accel_obs_profile_clock = nullptr) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int visible_all = sat_visible_width(a);
  const int sat_width = accel_sat_width(a);
  const int live_f = live_index == 0 ? kFLiveAccelObs0 : kFLiveAccelObs1;
  const int live_b = live_index == 0 ? kBLiveAccelObs0 : kBLiveAccelObs1;
  const float map_size = positive_config_scale(fp(a, kFpMapSize, 1.0f));
  const float vmax = positive_config_scale(fp(a, kFpVMax, 1.0f));
  const float accel_ref = positive_config_scale(fp(a, kFpAccelAMax, 1.0f));
  const float emax = positive_config_scale(fp(a, kFpUavEnergyInit, 1.0f));
  const float qmax_gu = positive_config_scale(fp(a, kFpQueueMaxGu, 1.0f));
  const float qmax_uav = positive_config_scale(fp(a, kFpQueueMaxUav, 1.0f));
  const float qmax_sat = positive_config_scale(fp(a, kFpQueueMaxSat, 1.0f));
  const float orbit = fmaxf(fp(a, kFpLocalEarthRadius, 0.0f) + fp(a, kFpSatHeight, 0.0f), 1.0f);
  const float sat_speed = sqrtf(3.986004418e14f / orbit);
  const float tau = fp(a, kFpBwTau0, fp(a, kFpTau0, 1.0f));
  const float arrival_ref = accel_arrival_ref(a, e);
  const float gu_flow_ref = positive_config_scale(arrival_ref / static_cast<float>(gu));
  const float uav_flow_ref = positive_config_scale(arrival_ref / static_cast<float>(ucount));
  const float sat_flow_ref = sat_flow_ref_current(a, arrival_ref);
  const float eps = positive_coeff(fp(a, kFpWorkloadEps, kRelativeLogEps));
  const float gu_local_cost_ref = safe_div(1.0f, fmaxf(gu_flow_ref, eps));
  const float uav_flow_cost_ref = safe_div(1.0f, fmaxf(uav_flow_ref, eps));
  const float sat_cost_ref = safe_div(1.0f, fmaxf(sat_flow_ref, eps));
  const float uav_total_cost_ref = uav_flow_cost_ref + sat_cost_ref;
  const float gu_total_cost_ref = gu_local_cost_ref + uav_total_cost_ref;
  const float access_noise_ref = accel_access_noise_ref(a);
  const float backhaul_noise_ref = accel_backhaul_noise_ref(a);
  const int peer_width = max(ucount - 1, 0);
  const int select_ref = min(min(sat, static_cast<int>(fp(a, kFpNrf, 1.0f))), static_cast<int>(ip(a, kParamSatNumSelect)));
  if (select_ref <= 0) {
    asm("trap;");
  }
  const float d_alert = fp(a, kFpAvoidanceAlertFactor, 1.5f) * fp(a, kFpDSafe, 0.0f);
  if (last_route_cost_cache != nullptr) {
    last_route_cost_cache_fill_parallel(a, e, last_route_cost_cache);
  }

  for (int idx = threadIdx.x; idx < ucount * kAccelEgoDim; idx += blockDim.x) {
    const int u = idx / kAccelEgoDim;
    const int d = idx - u * kAccelEgoDim;
    const int row = row_local(a, e, u);
    float* ego = a.f[live_f + 0] + static_cast<int64_t>(row) * kAccelEgoDim;
    ego[d] = 0.0f;
  }
  __shared__ float sh_accel_cell[kAccelCellSharedMaxUav * kAccelCellDim];
  __shared__ float sh_accel_cell_demand_denom[kAccelCellSharedMaxUav];
  __shared__ float sh_accel_cell_boundary_denom[kAccelCellSharedMaxUav];
  __shared__ float sh_accel_cell_weak_denom[kAccelCellSharedMaxUav];
  if (ucount <= kAccelCellSharedMaxUav) {
    for (int idx = threadIdx.x; idx < ucount * kAccelCellDim; idx += blockDim.x) {
      sh_accel_cell[idx] = 0.0f;
    }
    for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
      sh_accel_cell_demand_denom[u] = 0.0f;
      sh_accel_cell_boundary_denom[u] = 0.0f;
      sh_accel_cell_weak_denom[u] = 0.0f;
    }
    __syncthreads();

    for (int g = threadIdx.x; g < gu; g += blockDim.x) {
      const int owner = static_cast<int>(a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g]);
      if (owner >= 0 && owner < ucount) {
        const float q = a.f[stage_f(stage_slot, kSfGuQueue)][e * gu + g];
        atomicAdd(
            &sh_accel_cell[owner * kAccelCellDim + kAccelCellWorkloadShareGap],
            fmaxf(gu_cost_last_route_cached(a, e, g, last_route_cost_cache) * q, 0.0f));
      }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < ucount * gu; idx += blockDim.x) {
      const int u = idx / gu;
      const int g = idx - u * gu;
      const int last_assoc = has_i(a, kIStateLastAssociation) ? a.i[kIStateLastAssociation][e * gu + g] : -1;
      if (last_assoc != u) {
        const float bw_scale = accel_gu_last_bw_scale(a, e, g);
        const float gain = a.f[stage_f(stage_slot, kSfAccessGainMatrix)][(e * gu + g) * ucount + u];
        atomicAdd(
            &sh_accel_cell[u * kAccelCellDim + kAccelCellInterferenceExposure],
            fp(a, kFpAccessGuTxPower, fp(a, kFpLocalUavTxPower, 1.0f)) * gain * bw_scale);
      }

      const int owner = static_cast<int>(a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g]);
      if (owner != u) continue;
      const float q = a.f[stage_f(stage_slot, kSfGuQueue)][e * gu + g];
      const float expected = has_f(a, kFStateLastArrivalRateVec) ? a.f[kFStateLastArrivalRateVec][e * gu + g] * tau : 0.0f;
      const float last_arrival = has_f(a, kFStateLastGuArrival) ? a.f[kFStateLastGuArrival][e * gu + g] : 0.0f;
      const float last_outflow = has_f(a, kFStateLastGuOutflow) ? a.f[kFStateLastGuOutflow][e * gu + g] : 0.0f;
      const float drop = has_f(a, kFStateGuDrop) ? a.f[kFStateGuDrop][e * gu + g] : 0.0f;
      const float queue_steps = q / gu_flow_ref;
      const float expected_steps = expected / gu_flow_ref;
      const float last_arrival_steps = last_arrival / gu_flow_ref;
      const float last_outflow_steps = last_outflow / gu_flow_ref;
      const float last_drop_steps = drop / gu_flow_ref;
      const float demand_steps = queue_steps + expected_steps;
      const float last_cost = gu_cost_last_route(a, e, g);
      const float workload = fmaxf(last_cost * q, 0.0f);
      const float stability = accel_owner_stability_margin(a, stage_slot, e, g, map_size);
      const float boundary_weight = ucount > 1 ? expf(-stability) : 0.0f;
      const float owner_gain = a.f[stage_f(stage_slot, kSfAccessGainMatrix)][(e * gu + g) * ucount + owner];
      const float owner_se = accel_access_se_from_gain(a, owner_gain, access_noise_ref);
      const float weak_weight = 1.0f / (1.0f + owner_se);
      const float ux = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + owner) * 2 + 0];
      const float uy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + owner) * 2 + 1];
      const float gx = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + 0];
      const float gy = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + 1];
      const float relx = (gx - ux) / map_size;
      const float rely = (gy - uy) / map_size;

      float* accum = sh_accel_cell + u * kAccelCellDim;
      atomicAdd(&accum[kAccelCellGuCountFrac], 1.0f);
      atomicAdd(&accum[kAccelCellQueueStepsSum], queue_steps);
      atomicAdd(&accum[kAccelCellExpectedArrivalStepsSum], expected_steps);
      atomicAdd(&accum[kAccelCellLastArrivalStepsSum], last_arrival_steps);
      atomicAdd(&accum[kAccelCellLastOutflowStepsSum], last_outflow_steps);
      atomicAdd(&accum[kAccelCellLastDropStepsSum], last_drop_steps);
      atomicAdd(&accum[kAccelCellLastWorkloadLog1pSum], workload);
      const float boundary_w = workload * boundary_weight;
      const float weak_w = workload * weak_weight;
      atomicAdd(&accum[kAccelCellBoundaryWorkloadSum], boundary_w);
      atomicAdd(&accum[kAccelCellWeakLinkWorkloadSum], weak_w);
      atomicAdd(&accum[kAccelCellAccessPressure], fmaxf(demand_steps - last_outflow_steps, 0.0f) + last_drop_steps);
      atomicAdd(&accum[kAccelCellDemandMomentX], demand_steps * relx);
      atomicAdd(&accum[kAccelCellDemandMomentY], demand_steps * rely);
      atomicAdd(&sh_accel_cell_demand_denom[u], demand_steps);
      atomicAdd(&accum[kAccelCellBoundaryMomentX], boundary_w * relx);
      atomicAdd(&accum[kAccelCellBoundaryMomentY], boundary_w * rely);
      atomicAdd(&sh_accel_cell_boundary_denom[u], boundary_w);
      atomicAdd(&accum[kAccelCellWeakLinkMomentX], weak_w * relx);
      atomicAdd(&accum[kAccelCellWeakLinkMomentY], weak_w * rely);
      atomicAdd(&sh_accel_cell_weak_denom[u], weak_w);
    }
    __syncthreads();

    for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
      const int row = row_local(a, e, u);
      float* cell = a.f[live_f + 1] + static_cast<int64_t>(row) * kAccelCellDim;
      const float inv_gu = safe_div(1.0f, fmaxf(static_cast<float>(gu), 1.0f));
      float workload_total = 0.0f;
      for (int v = 0; v < ucount; ++v) {
        workload_total += fmaxf(sh_accel_cell[v * kAccelCellDim + kAccelCellWorkloadShareGap] * inv_gu, 0.0f);
      }
      const float workload_own = fmaxf(sh_accel_cell[u * kAccelCellDim + kAccelCellWorkloadShareGap] * inv_gu, 0.0f);
      cell[kAccelCellGuCountFrac] = sh_accel_cell[u * kAccelCellDim + kAccelCellGuCountFrac] * inv_gu;
      cell[kAccelCellQueueStepsSum] = log1p_nonnegative(sh_accel_cell[u * kAccelCellDim + kAccelCellQueueStepsSum] * inv_gu);
      cell[kAccelCellExpectedArrivalStepsSum] =
          log1p_nonnegative(sh_accel_cell[u * kAccelCellDim + kAccelCellExpectedArrivalStepsSum] * inv_gu);
      cell[kAccelCellLastArrivalStepsSum] =
          log1p_nonnegative(sh_accel_cell[u * kAccelCellDim + kAccelCellLastArrivalStepsSum] * inv_gu);
      cell[kAccelCellLastOutflowStepsSum] =
          log1p_nonnegative(sh_accel_cell[u * kAccelCellDim + kAccelCellLastOutflowStepsSum] * inv_gu);
      cell[kAccelCellLastDropStepsSum] =
          log1p_nonnegative(sh_accel_cell[u * kAccelCellDim + kAccelCellLastDropStepsSum] * inv_gu);
      cell[kAccelCellLastWorkloadLog1pSum] =
          log1p_nonnegative(sh_accel_cell[u * kAccelCellDim + kAccelCellLastWorkloadLog1pSum] * inv_gu);
      cell[kAccelCellWorkloadShareGap] =
          workload_total > 0.0f ? workload_own / workload_total - 1.0f / static_cast<float>(ucount) : 0.0f;
      cell[kAccelCellBoundaryWorkloadSum] =
          log1p_nonnegative(sh_accel_cell[u * kAccelCellDim + kAccelCellBoundaryWorkloadSum] * inv_gu);
      cell[kAccelCellWeakLinkWorkloadSum] =
          log1p_nonnegative(sh_accel_cell[u * kAccelCellDim + kAccelCellWeakLinkWorkloadSum] * inv_gu);
      cell[kAccelCellAccessPressure] = sh_accel_cell[u * kAccelCellDim + kAccelCellAccessPressure] * inv_gu;
      cell[kAccelCellInterferenceExposure] =
          log1pf(fmaxf(sh_accel_cell[u * kAccelCellDim + kAccelCellInterferenceExposure] / access_noise_ref, 0.0f));
      const float demand_denom = sh_accel_cell_demand_denom[u];
      const float boundary_denom = sh_accel_cell_boundary_denom[u];
      const float weak_denom = sh_accel_cell_weak_denom[u];
      cell[kAccelCellDemandMomentX] =
          demand_denom > 0.0f ? sh_accel_cell[u * kAccelCellDim + kAccelCellDemandMomentX] / demand_denom : 0.0f;
      cell[kAccelCellDemandMomentY] =
          demand_denom > 0.0f ? sh_accel_cell[u * kAccelCellDim + kAccelCellDemandMomentY] / demand_denom : 0.0f;
      cell[kAccelCellBoundaryMomentX] =
          boundary_denom > 0.0f ? sh_accel_cell[u * kAccelCellDim + kAccelCellBoundaryMomentX] / boundary_denom : 0.0f;
      cell[kAccelCellBoundaryMomentY] =
          boundary_denom > 0.0f ? sh_accel_cell[u * kAccelCellDim + kAccelCellBoundaryMomentY] / boundary_denom : 0.0f;
      cell[kAccelCellWeakLinkMomentX] =
          weak_denom > 0.0f ? sh_accel_cell[u * kAccelCellDim + kAccelCellWeakLinkMomentX] / weak_denom : 0.0f;
      cell[kAccelCellWeakLinkMomentY] =
          weak_denom > 0.0f ? sh_accel_cell[u * kAccelCellDim + kAccelCellWeakLinkMomentY] / weak_denom : 0.0f;
      if (accel_cell_summary_cache != nullptr) {
        for (int d = 0; d < kAccelCellDim; ++d) {
          accel_cell_summary_cache[u * kAccelCellDim + d] = cell[d];
        }
      }
    }
  } else {
    for (int idx = threadIdx.x; idx < ucount * kAccelCellDim; idx += blockDim.x) {
      const int u = idx / kAccelCellDim;
      const int d = idx - u * kAccelCellDim;
      const int row = row_local(a, e, u);
      float* cell = a.f[live_f + 1] + static_cast<int64_t>(row) * kAccelCellDim;
      const float value = accel_cell_summary_value(
          a, stage_slot, e, u, d, arrival_ref, gu_flow_ref, access_noise_ref, last_route_cost_cache);
      cell[d] = value;
      if (accel_cell_summary_cache != nullptr) {
        accel_cell_summary_cache[u * kAccelCellDim + d] = value;
      }
    }
  }
  __syncthreads();

  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    const int row = row_local(a, e, u);
    float* ego = a.f[live_f + 0] + static_cast<int64_t>(row) * kAccelEgoDim;
    const float ux = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 0];
    const float uy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 1];
    const float uvx = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 0];
    const float uvy = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 1];
    const float uq = a.f[stage_f(stage_slot, kSfUavQueue)][e * ucount + u];
    const float last_uav_cost = uav_cost_last_route_cached(a, e, u, last_route_cost_cache);
    const float uav_local_cost = safe_div(1.0f, fmaxf(a.f[kFStateUavEma][e * ucount + u], eps));
    const float last_policy_x = has_f(a, kFStateLastPolicyAccel) ? a.f[kFStateLastPolicyAccel][(e * ucount + u) * 2 + 0] : 0.0f;
    const float last_policy_y = has_f(a, kFStateLastPolicyAccel) ? a.f[kFStateLastPolicyAccel][(e * ucount + u) * 2 + 1] : 0.0f;
    const float last_exec_x = has_f(a, kFStateLastExecAccel) ? a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 0] : 0.0f;
    const float last_exec_y = has_f(a, kFStateLastExecAccel) ? a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 1] : 0.0f;
    const float int_dx = last_exec_x - last_policy_x;
    const float int_dy = last_exec_y - last_policy_y;
    ego[kAccelEgoX] = ux / map_size;
    ego[kAccelEgoY] = uy / map_size;
    ego[kAccelEgoVx] = uvx / vmax;
    ego[kAccelEgoVy] = uvy / vmax;
    ego[kAccelEgoSpeed] = sqrtf(uvx * uvx + uvy * uvy) / vmax;
    ego[kAccelEgoEnergy] = a.f[stage_f(stage_slot, kSfUavEnergy)][e * ucount + u] / emax;
    ego[kAccelEgoBoundaryLeft] = ux / map_size;
    ego[kAccelEgoBoundaryRight] = (map_size - ux) / map_size;
    ego[kAccelEgoBoundaryBottom] = uy / map_size;
    ego[kAccelEgoBoundaryTop] = (map_size - uy) / map_size;
    ego[kAccelEgoUavQueueSteps] = log1p_nonnegative(uq / uav_flow_ref);
    ego[kAccelEgoUavQueueFill] = uq / qmax_uav;
    ego[kAccelEgoUavLastInflowSteps] = has_f(a, kFStateLastGuToUavInflowByUav) ? log1p_nonnegative(a.f[kFStateLastGuToUavInflowByUav][e * ucount + u] / uav_flow_ref) : 0.0f;
    ego[kAccelEgoUavLastOutflowSteps] = log1p_nonnegative(accel_uav_last_outflow(a, e, u) / uav_flow_ref);
    ego[kAccelEgoUavLastDropSteps] = has_f(a, kFStateUavDrop) ? log1p_nonnegative(a.f[kFStateUavDrop][e * ucount + u] / uav_flow_ref) : 0.0f;
    ego[kAccelEgoUavServiceEmaSteps] = log1p_nonnegative(a.f[kFStateUavEma][e * ucount + u] / uav_flow_ref);
    ego[kAccelEgoUavLocalCostLogRatio] = logf(log_argument(uav_local_cost / uav_flow_cost_ref));
    ego[kAccelEgoUavLastTotalCostLogRatio] = logf(log_argument(last_uav_cost / uav_total_cost_ref));
    ego[kAccelEgoUavLastWorkloadLog1p] = log1pf(fmaxf(last_uav_cost * uq, 0.0f));
    ego[kAccelEgoUavLastAccessInterferenceLog1p] = log1pf(fmaxf((has_f(a, kFStateLastAccessInterferenceByUav) ? a.f[kFStateLastAccessInterferenceByUav][e * ucount + u] : 0.0f) / access_noise_ref, 0.0f));
    ego[kAccelEgoLastPolicyAccelX] = last_policy_x / accel_ref;
    ego[kAccelEgoLastPolicyAccelY] = last_policy_y / accel_ref;
    ego[kAccelEgoLastExecAccelX] = last_exec_x / accel_ref;
    ego[kAccelEgoLastExecAccelY] = last_exec_y / accel_ref;
    ego[kAccelEgoLastInterventionDx] = int_dx / accel_ref;
    ego[kAccelEgoLastInterventionDy] = int_dy / accel_ref;
    ego[kAccelEgoLastInterventionL2] = sqrtf(int_dx * int_dx + int_dy * int_dy) / accel_ref;
    ego[kAccelEgoRemainingHorizonFrac] = remaining_horizon_frac_device(a, e);
  }
  finish_profile_mark(
      accel_obs_profile_out,
      accel_obs_profile_stride,
      e,
      kFinishProfAccelObsEgoCell,
      accel_obs_profile_clock);

  for (int idx = threadIdx.x; idx < ucount * gu; idx += blockDim.x) {
    const int u = idx / gu;
    const int g = idx - u * gu;
    const int row = row_local(a, e, u);
    float* tok = a.f[live_f + 2] + (static_cast<int64_t>(row) * gu + g) * kAccelGuTokenDim;
    for (int d = 0; d < kAccelGuTokenDim; ++d) tok[d] = 0.0f;
    const float ux = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 0];
    const float uy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 1];
    const int owner = static_cast<int>(a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g]);
    const int last_assoc = has_i(a, kIStateLastAssociation) ? a.i[kIStateLastAssociation][e * gu + g] : -1;
    const float gx = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + 0];
    const float gy = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + 1];
    const float q = a.f[stage_f(stage_slot, kSfGuQueue)][e * gu + g];
    const float expected = has_f(a, kFStateLastArrivalRateVec) ? a.f[kFStateLastArrivalRateVec][e * gu + g] * tau : 0.0f;
    const float last_arrival = has_f(a, kFStateLastGuArrival) ? a.f[kFStateLastGuArrival][e * gu + g] : 0.0f;
    const float last_outflow = has_f(a, kFStateLastGuOutflow) ? a.f[kFStateLastGuOutflow][e * gu + g] : 0.0f;
    const float drop = has_f(a, kFStateGuDrop) ? a.f[kFStateGuDrop][e * gu + g] : 0.0f;
    const float dist_ego = accel_gu_dist(a, stage_slot, e, u, g);
    const float gain_ego = a.f[stage_f(stage_slot, kSfAccessGainMatrix)][(e * gu + g) * ucount + u];
    const float se_ego = accel_access_se_from_gain(a, gain_ego, access_noise_ref);
    const float last_bw_ego = has_f(a, kFStateLastBwFractionByUavGu) ? a.f[kFStateLastBwFractionByUavGu][(e * ucount + u) * gu + g] : 0.0f;
    const float stability = accel_owner_stability_margin(a, stage_slot, e, g, map_size);
    float best_other_ego = dist_ego + map_size;
    float d_owner = dist_ego;
    if (owner >= 0 && owner < ucount) d_owner = accel_gu_dist(a, stage_slot, e, owner, g);
    if (ucount > 1) {
      best_other_ego = 3.402823466e38f;
      for (int v = 0; v < ucount; ++v) {
        if (v == u) continue;
        best_other_ego = fminf(best_other_ego, accel_gu_dist(a, stage_slot, e, v, g));
      }
    }
    const float bw_scale = accel_gu_last_bw_scale(a, e, g);
    tok[kAccelGuX] = gx / map_size;
    tok[kAccelGuY] = gy / map_size;
    tok[kAccelGuQueueSteps] = log1p_nonnegative(q / gu_flow_ref);
    tok[kAccelGuQueueFill] = q / qmax_gu;
    tok[kAccelGuExpectedArrivalSteps] = log1p_nonnegative(expected / gu_flow_ref);
    tok[kAccelGuLastArrivalSteps] = log1p_nonnegative(last_arrival / gu_flow_ref);
    tok[kAccelGuLastOutflowSteps] = log1p_nonnegative(last_outflow / gu_flow_ref);
    tok[kAccelGuLastDropSteps] = log1p_nonnegative(drop / gu_flow_ref);
    tok[kAccelGuServiceEmaSteps] = log1p_nonnegative(a.f[kFStateGuEma][e * gu + g] / gu_flow_ref);
    tok[kAccelGuLocalCostLogRatio] = logf(log_argument(safe_div(1.0f, fmaxf(a.f[kFStateGuEma][e * gu + g], eps)) / gu_local_cost_ref));
    const float last_gu_cost = gu_cost_last_route_cached(a, e, g, last_route_cost_cache);
    tok[kAccelGuLastTotalCostLogRatio] = logf(log_argument(last_gu_cost / gu_total_cost_ref));
    tok[kAccelGuLastWorkloadLog1p] = log1pf(fmaxf(last_gu_cost * q, 0.0f));
    tok[kAccelGuRelX] = (gx - ux) / map_size;
    tok[kAccelGuRelY] = (gy - uy) / map_size;
    tok[kAccelGuDist] = dist_ego / map_size;
    tok[kAccelGuAccessSeRef] = se_ego;
    tok[kAccelGuLastAssocToEgo] = last_assoc == u ? 1.0f : 0.0f;
    tok[kAccelGuLastBwFractionEgo] = last_bw_ego;
    tok[kAccelGuLastServedByEgo] = last_bw_ego;
    tok[kAccelGuPreOwnerIsEgo] = owner == u ? 1.0f : 0.0f;
    tok[kAccelGuHandoffMarginEgo] = ucount > 1 ? (best_other_ego - dist_ego) / map_size : 1.0f;
    tok[kAccelGuOwnerStabilityMargin] = stability;
    tok[kAccelGuEgoTakeoverGap] = ucount > 1 ? (dist_ego - d_owner) / map_size : 0.0f;
    tok[kAccelGuPartitionBoundaryWeight] = ucount > 1 ? expf(-stability) : 0.0f;
    tok[kAccelGuEgoLinkWeakness] = 1.0f / (1.0f + se_ego);
    tok[kAccelGuLastBwSum] = bw_scale;
    const float nonself_power = fp(a, kFpAccessGuTxPower, fp(a, kFpLocalUavTxPower, 1.0f)) * gain_ego * bw_scale * (last_assoc != u ? 1.0f : 0.0f);
    tok[kAccelGuLastNonselfInterferenceLog1p] = log1pf(fmaxf(nonself_power / access_noise_ref, 0.0f));
    a.b[live_b + 0][row * gu + g] = true;
  }
  finish_profile_mark(
      accel_obs_profile_out,
      accel_obs_profile_stride,
      e,
      kFinishProfAccelObsGuTokens,
      accel_obs_profile_clock);

  for (int idx = threadIdx.x; idx < ucount * peer_width; idx += blockDim.x) {
    const int u = idx / max(peer_width, 1);
    const int peer_slot = idx - u * max(peer_width, 1);
    const int v = peer_slot < u ? peer_slot : peer_slot + 1;
    const int row = row_local(a, e, u);
    float* peer = a.f[live_f + 3] + (static_cast<int64_t>(row) * peer_width + peer_slot) * kAccelPeerTokenDim;
    for (int d = 0; d < kAccelPeerTokenDim; ++d) peer[d] = 0.0f;
    const float ux = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 0];
    const float uy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 1];
    const float uvx = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 0];
    const float uvy = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 1];
    const float vx = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + v) * 2 + 0];
    const float vy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + v) * 2 + 1];
    const float vvx = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + v) * 2 + 0];
    const float vvy = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + v) * 2 + 1];
    const float relx = ux - vx;
    const float rely = uy - vy;
    const float relvx = uvx - vvx;
    const float relvy = uvy - vvy;
    const float dist = sqrtf(relx * relx + rely * rely);
    peer[kAccelPeerRelX] = relx / map_size;
    peer[kAccelPeerRelY] = rely / map_size;
    peer[kAccelPeerRelVx] = relvx / vmax;
    peer[kAccelPeerRelVy] = relvy / vmax;
    peer[kAccelPeerDist] = dist / map_size;
    peer[kAccelPeerClosingSpeed] = dist > 0.0f ? fmaxf(0.0f, -(relx * relvx + rely * relvy) / (dist * vmax)) : 0.0f;
    peer[kAccelPeerSafeDistanceMargin] = (dist - fp(a, kFpDSafe, 0.0f)) / map_size;
    peer[kAccelPeerUnsafeFlag] = dist < fp(a, kFpDSafe, 0.0f) ? 1.0f : 0.0f;
    peer[kAccelPeerAlertFlag] = dist < d_alert ? 1.0f : 0.0f;
    peer[kAccelPeerLastSharedSatFrac] = selected_shared_fraction_last(a, e, u, v) * safe_div(static_cast<float>(ip(a, kParamSatNumSelect, 1)), static_cast<float>(select_ref));
    const float* peer_cell = accel_cell_summary_cache != nullptr
        ? accel_cell_summary_cache + v * kAccelCellDim
        : a.f[live_f + 1] + static_cast<int64_t>(row_local(a, e, v)) * kAccelCellDim;
    for (int d = 0; d < kAccelCellDim; ++d) {
      peer[kAccelPeerCellOffset + d] = peer_cell[d];
    }
    a.b[live_b + 1][row * peer_width + peer_slot] = true;
  }
  finish_profile_mark(
      accel_obs_profile_out,
      accel_obs_profile_stride,
      e,
      kFinishProfAccelObsPeerTokens,
      accel_obs_profile_clock);

  for (int idx = threadIdx.x; idx < ucount * sat_width; idx += blockDim.x) {
    const int u = idx / max(sat_width, 1);
    const int j = idx - u * max(sat_width, 1);
    const int row = row_local(a, e, u);
    float* st = a.f[live_f + 4] + (static_cast<int64_t>(row) * sat_width + j) * kAccelSatTokenDim;
    for (int d = 0; d < kAccelSatTokenDim; ++d) st[d] = 0.0f;
    const bool listed = j < visible_all && a.b[stage_b(stage_slot, 1)][(e * ucount + u) * visible_all + j];
    const int sid = listed ? static_cast<int>(a.l[stage_l(stage_slot, kSlVisibleIds)][(e * ucount + u) * visible_all + j]) : -1;
    const bool token_valid = sid >= 0 && sid < sat;
    a.b[live_b + 2][row * sat_width + j] = token_valid;
    if (!token_valid) continue;
    float rel[3];
    float relv[3];
    sat_rel_for_stage_us(a, stage_slot, e, u, sid, rel, relv);
    const float range_sq = rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2];
    if (range_sq <= 0.0f) {
      asm("trap;");
    }
    const float range = sqrtf(range_sq);
    const float elevation = a.f[stage_f(stage_slot, kSfElevationMatrix)][(e * ucount + u) * sat + sid];
    const float nu_eff = doppler_eff_hz(a, e, u, sid, rel, relv);
    const float doppler_ratio = ip(a, kParamDopplerObserved) || ip(a, kParamDopplerEnabled) || ip(a, kParamDopplerAttenEnabled)
        ? nu_eff / fp(a, kFpNuMax, 1.0f)
        : 0.0f;
    float gain = positive_coeff(fp(a, kFpBwBackhaulGainConst, 1.0f)) / (range * range);
    gain *= atmospheric_loss_factor_device(a, elevation);
    float snr = fp(a, kFpBwUavTxPower, 1.0f) * gain / backhaul_noise_ref;
    if (ip(a, kParamDopplerAttenEnabled)) {
      const float spacing = fp(a, kFpSubcarrierSpacing, 0.0f);
      if (spacing > 0.0f) {
        const float s = sinc_pi_device(nu_eff / spacing);
        snr *= s * s;
      }
    }
    const bool link_valid = listed && elevation >= fp(a, kFpThetaMinRad, -3.1415926f) &&
        (!ip(a, kParamDopplerEnabled) || fabsf(nu_eff) <= fp(a, kFpNuMax, 1.0f));
    const float q = a.f[stage_f(stage_slot, kSfSatQueue)][e * sat + sid];
    const float sat_cost = sat_cost_current(a, e, sid);
    st[kAccelSatX] = a.f[stage_f(stage_slot, kSfSatPos)][(e * sat + sid) * 3 + 0] / orbit;
    st[kAccelSatY] = a.f[stage_f(stage_slot, kSfSatPos)][(e * sat + sid) * 3 + 1] / orbit;
    st[kAccelSatZ] = a.f[stage_f(stage_slot, kSfSatPos)][(e * sat + sid) * 3 + 2] / orbit;
    st[kAccelSatVx] = a.f[stage_f(stage_slot, kSfSatVel)][(e * sat + sid) * 3 + 0] / sat_speed;
    st[kAccelSatVy] = a.f[stage_f(stage_slot, kSfSatVel)][(e * sat + sid) * 3 + 1] / sat_speed;
    st[kAccelSatVz] = a.f[stage_f(stage_slot, kSfSatVel)][(e * sat + sid) * 3 + 2] / sat_speed;
    st[kAccelSatQueueSteps] = log1p_nonnegative(q / sat_flow_ref);
    st[kAccelSatQueueFill] = q / qmax_sat;
    st[kAccelSatLastIncomingSteps] = log1p_nonnegative(accel_sat_last_incoming(a, e, sid) / sat_flow_ref);
    st[kAccelSatLastProcessedSteps] = has_f(a, kFStateLastSatProcessed) ? log1p_nonnegative(a.f[kFStateLastSatProcessed][e * sat + sid] / sat_flow_ref) : 0.0f;
    st[kAccelSatLastDropSteps] = has_f(a, kFStateSatDrop) ? log1p_nonnegative(a.f[kFStateSatDrop][e * sat + sid] / sat_flow_ref) : 0.0f;
    st[kAccelSatServiceEmaSteps] = log1p_nonnegative(a.f[kFStateSatEma][e * sat + sid] / sat_flow_ref);
    st[kAccelSatCostLogRatio] = logf(log_argument(sat_cost / sat_cost_ref));
    st[kAccelSatLastWorkloadLog1p] = log1pf(fmaxf(sat_cost * q, 0.0f));
    st[kAccelSatLastSelectedLoadFrac] = has_f(a, kFStateLastSatConnectionCounts) ? a.f[kFStateLastSatConnectionCounts][e * sat + sid] / static_cast<float>(ucount) : 0.0f;
    st[kAccelSatProcCapacitySteps] = log1p_nonnegative(sat_compute_rate_for(a, e, sid) * tau / sat_flow_ref);
    st[kAccelSatRelX] = rel[0] / orbit;
    st[kAccelSatRelY] = rel[1] / orbit;
    st[kAccelSatRelZ] = rel[2] / orbit;
    st[kAccelSatRelVx] = relv[0] / sat_speed;
    st[kAccelSatRelVy] = relv[1] / sat_speed;
    st[kAccelSatRelVz] = relv[2] / sat_speed;
    st[kAccelSatRange] = range / orbit;
    st[kAccelSatRadialVelocity] = (rel[0] * relv[0] + rel[1] * relv[1] + rel[2] * relv[2]) / (range * sat_speed);
    st[kAccelSatElevation] = elevation / (0.5f * 3.14159265358979323846f);
    st[kAccelSatDopplerRatio] = doppler_ratio;
    st[kAccelSatDopplerAbsRatio] = fabsf(doppler_ratio);
    st[kAccelSatBackhaulSeRef] = spectral_efficiency_device(snr);
    st[kAccelSatVisibleFlag] = listed ? 1.0f : 0.0f;
    st[kAccelSatValidFlag] = link_valid ? 1.0f : 0.0f;
    st[kAccelSatLastSelectedFlag] = has_f(a, kFStateLastSelectedMaskByUavSat) ? a.f[kFStateLastSelectedMaskByUavSat][(e * ucount + u) * sat + sid] : 0.0f;
    st[kAccelSatLastOutflowSteps] = has_f(a, kFStateLastUavToSatOutflowMatrix) ? log1p_nonnegative(a.f[kFStateLastUavToSatOutflowMatrix][(e * ucount + u) * sat + sid] / uav_flow_ref) : 0.0f;
  }
  finish_profile_mark(
      accel_obs_profile_out,
      accel_obs_profile_stride,
      e,
      kFinishProfAccelObsSatTokens,
      accel_obs_profile_clock);
}

__device__ void write_sat_live_obs_parallel(const PackedAbi& a, int stage_slot, int e) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int visible = sat_visible_width(a);
  const int subset_count = static_cast<int>(ip(a, kParamSubsetCount));
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  constexpr int kSatEgoDim = 13;
  constexpr int kSatDemandDim = 8;
  constexpr int kSatRoleDim = 1;
  constexpr int kSatTokenDim = 26;
  const float tau = fp(a, kFpBwTau0, fp(a, kFpTau0, 1.0f));
  const float qmax_gu = positive_config_scale(fp(a, kFpQueueMaxGu, 1.0f));
  const float qmax_uav = positive_config_scale(fp(a, kFpQueueMaxUav, 1.0f));
  const float qmax_sat = positive_config_scale(fp(a, kFpQueueMaxSat, 1.0f));
  const float arrival_ref = accel_arrival_ref(a, e);
  const float gu_flow_ref = positive_config_scale(arrival_ref / fmaxf(static_cast<float>(gu), 1.0f));
  const float uav_flow_ref = positive_config_scale(arrival_ref / fmaxf(static_cast<float>(ucount), 1.0f));
  const float sat_flow_ref = sat_flow_ref_current(a, arrival_ref);
  const float eps = positive_coeff(fp(a, kFpWorkloadEps, kRelativeLogEps));
  const float gu_flow_cost_ref = safe_div(1.0f, fmaxf(gu_flow_ref, eps));
  const float uav_flow_cost_ref = safe_div(1.0f, fmaxf(uav_flow_ref, eps));
  const float sat_cost_ref = safe_div(1.0f, fmaxf(sat_flow_ref, eps));
  const float uav_total_cost_ref = uav_flow_cost_ref + sat_cost_ref;
  const float gu_total_cost_ref = gu_flow_cost_ref + uav_total_cost_ref;
  const float access_noise_ref = accel_access_noise_ref(a);
  const float backhaul_noise_ref = accel_backhaul_noise_ref(a);
  const float orbit = fmaxf(fp(a, kFpLocalEarthRadius, 0.0f) + fp(a, kFpSatHeight, 0.0f), 1.0f);
  const float sat_speed = sqrtf(fmaxf(3.986004418e14f / orbit, 1.0f));
  const float doppler_ref = fmaxf(fp(a, kFpSatCarrierFreq, fp(a, kFpCarrierFreq, 2.0e9f)) * sat_speed / fmaxf(fp(a, kFpSpeedOfLight, 299792458.0f), 1.0f), 1.0f);
  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    const int row = row_local(a, e, u);
    float* ego = a.f[kFLiveSatObs + 0] + static_cast<int64_t>(row) * kSatEgoDim;
    float* demand = a.f[kFLiveSatObs + 1] + static_cast<int64_t>(row) * kSatDemandDim;
    float* role = a.f[kFLiveSatObs + 2] + static_cast<int64_t>(row) * kSatRoleDim;
    float* sat_tokens = a.f[kFLiveSatObs + 3] + static_cast<int64_t>(row) * visible * kSatTokenDim;
    for (int d = 0; d < kSatEgoDim; ++d) ego[d] = 0.0f;
    for (int d = 0; d < kSatDemandDim; ++d) demand[d] = 0.0f;
    role[0] = static_cast<float>(u) / fmaxf(static_cast<float>(ucount - 1), 1.0f);
    const float q_uav = a.f[stage_f(stage_slot, kSfUavQueue)][e * ucount + u];
    const float uav_local_cost = safe_div(1.0f, fmaxf(a.f[kFStateUavEma][e * ucount + u], eps));
    const float last_uav_cost = uav_cost_last_route(a, e, u);
    ego[0] = log1p_nonnegative(q_uav / uav_flow_ref);
    ego[1] = q_uav / qmax_uav;
    ego[2] = has_f(a, kFStateLastGuToUavInflowByUav) ? log1p_nonnegative(a.f[kFStateLastGuToUavInflowByUav][e * ucount + u] / uav_flow_ref) : 0.0f;
    ego[3] = log1p_nonnegative(accel_uav_last_outflow(a, e, u) / uav_flow_ref);
    ego[4] = has_f(a, kFStateUavDrop) ? log1p_nonnegative(a.f[kFStateUavDrop][e * ucount + u] / uav_flow_ref) : 0.0f;
    ego[5] = log1p_nonnegative(a.f[kFStateUavEma][e * ucount + u] / uav_flow_ref);
    ego[6] = logf(log_argument(uav_local_cost / uav_flow_cost_ref));
    ego[7] = logf(log_argument(last_uav_cost / uav_total_cost_ref));
    ego[8] = log1pf(fmaxf(last_uav_cost * q_uav, 0.0f));
    ego[9] = log1pf(fmaxf((has_f(a, kFStateLastAccessInterferenceByUav) ? a.f[kFStateLastAccessInterferenceByUav][e * ucount + u] : 0.0f) / access_noise_ref, 0.0f));
    float last_selected_count = 0.0f;
    float last_backhaul_out = 0.0f;
    for (int s = 0; s < sat; ++s) {
      last_selected_count += has_f(a, kFStateLastSelectedMaskByUavSat) ? a.f[kFStateLastSelectedMaskByUavSat][(e * ucount + u) * sat + s] : 0.0f;
      last_backhaul_out += has_f(a, kFStateLastUavToSatOutflowMatrix) ? a.f[kFStateLastUavToSatOutflowMatrix][(e * ucount + u) * sat + s] : 0.0f;
    }
    ego[10] = last_selected_count / fmaxf(static_cast<float>(select_k), 1.0f);
    ego[11] = log1p_nonnegative(last_backhaul_out / uav_flow_ref);
    ego[12] = remaining_horizon_frac_device(a, e);

    for (int g = 0; g < gu; ++g) {
      const int owner = static_cast<int>(a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g]);
      if (owner != u) continue;
      const float q = a.f[stage_f(stage_slot, kSfGuQueue)][e * gu + g];
      const float expected = has_f(a, kFStateLastArrivalRateVec) ? a.f[kFStateLastArrivalRateVec][e * gu + g] * tau : 0.0f;
      const float last_arrival = has_f(a, kFStateLastGuArrival) ? a.f[kFStateLastGuArrival][e * gu + g] : 0.0f;
      const float last_outflow = has_f(a, kFStateLastGuOutflow) ? a.f[kFStateLastGuOutflow][e * gu + g] : 0.0f;
      const float drop = has_f(a, kFStateGuDrop) ? a.f[kFStateGuDrop][e * gu + g] : 0.0f;
      const float last_cost = gu_cost_last_route(a, e, g);
      const float gain = a.f[stage_f(stage_slot, kSfAccessGainMatrix)][(e * gu + g) * ucount + u];
      demand[0] += 1.0f / fmaxf(static_cast<float>(gu), 1.0f);
      demand[1] += (q / gu_flow_ref) / fmaxf(static_cast<float>(gu), 1.0f);
      demand[2] += (expected / gu_flow_ref) / fmaxf(static_cast<float>(gu), 1.0f);
      demand[3] += (last_arrival / gu_flow_ref) / fmaxf(static_cast<float>(gu), 1.0f);
      demand[4] += (last_outflow / gu_flow_ref) / fmaxf(static_cast<float>(gu), 1.0f);
      demand[5] += (drop / gu_flow_ref) / fmaxf(static_cast<float>(gu), 1.0f);
      demand[6] += fmaxf(last_cost * q, 0.0f) / fmaxf(static_cast<float>(gu), 1.0f);
      demand[7] += (fp(a, kFpAccessBAcc, 1.0f) * accel_access_se_from_gain(a, gain, access_noise_ref) * tau / gu_flow_ref) / fmaxf(static_cast<float>(gu), 1.0f);
      (void)gu_total_cost_ref;
      (void)qmax_gu;
    }
    demand[1] = log1p_nonnegative(demand[1]);
    demand[2] = log1p_nonnegative(demand[2]);
    demand[3] = log1p_nonnegative(demand[3]);
    demand[4] = log1p_nonnegative(demand[4]);
    demand[5] = log1p_nonnegative(demand[5]);
    demand[6] = log1p_nonnegative(demand[6]);
    demand[7] = log1p_nonnegative(demand[7]);

    for (int j = 0; j < visible; ++j) {
      const int sid = static_cast<int>(a.l[stage_l(stage_slot, kSlVisibleIds)][(e * ucount + u) * visible + j]);
      float* st = sat_tokens + static_cast<int64_t>(j) * kSatTokenDim;
      for (int d = 0; d < kSatTokenDim; ++d) st[d] = 0.0f;
      const bool listed = sid >= 0 && sid < sat;
      if (has_l(a, kLLiveSatCandidateIds)) {
        a.l[kLLiveSatCandidateIds][static_cast<int64_t>(row) * visible + j] = listed ? static_cast<int64_t>(sid) : -1;
      }
      a.b[kBLiveSatObs + 0][row * visible + j] = listed;
      if (!listed) {
        a.b[kBLiveSatObs + 1][row * visible + j] = false;
        continue;
      }
      float rel[3];
      float relv[3];
      sat_rel_for_stage_us(a, stage_slot, e, u, sid, rel, relv);
      const float range2 = fmaxf(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2], 1.0f);
      const float range = sqrtf(range2);
      const float nu_eff = doppler_eff_hz(a, e, u, sid, rel, relv);
      const float elevation = has_f(a, stage_f(stage_slot, kSfElevationMatrix))
          ? a.f[stage_f(stage_slot, kSfElevationMatrix)][(e * ucount + u) * sat + sid]
          : fp(a, kFpThetaMinRad, 0.0f);
      const bool valid = elevation >= fp(a, kFpThetaMinRad, -3.1415926f) &&
          (!ip(a, kParamDopplerEnabled) || fabsf(nu_eff) <= fmaxf(fp(a, kFpNuMax, 1.0f), 1.0f));
      a.b[kBLiveSatObs + 1][row * visible + j] = valid;
      const float sat_cost = sat_cost_current(a, e, sid);
      const float sat_q = a.f[stage_f(stage_slot, kSfSatQueue)][e * sat + sid];
      const float gain = positive_coeff(fp(a, kFpBwBackhaulGainConst, 1.0f)) / range2 *
          atmospheric_loss_factor_device(a, elevation);
      float snr = fp(a, kFpSatUavTxPower, fp(a, kFpBwUavTxPower, fp(a, kFpLocalUavTxPower, 1.0f))) * gain / backhaul_noise_ref;
      if (ip(a, kParamDopplerAttenEnabled)) {
        const float spacing = fp(a, kFpSubcarrierSpacing, 0.0f);
        if (spacing > 0.0f) {
          const float s = sinc_pi_device(nu_eff / spacing);
          snr *= s * s;
        }
      }
      st[0] = log1p_nonnegative(sat_q / sat_flow_ref);
      st[1] = sat_q / qmax_sat;
      st[2] = log1p_nonnegative(accel_sat_last_incoming(a, e, sid) / sat_flow_ref);
      st[3] = has_f(a, kFStateLastSatProcessed) ? log1p_nonnegative(a.f[kFStateLastSatProcessed][e * sat + sid] / sat_flow_ref) : 0.0f;
      st[4] = has_f(a, kFStateSatDrop) ? log1p_nonnegative(a.f[kFStateSatDrop][e * sat + sid] / sat_flow_ref) : 0.0f;
      st[5] = log1p_nonnegative(a.f[kFStateSatEma][e * sat + sid] / sat_flow_ref);
      st[6] = logf(log_argument(sat_cost / sat_cost_ref));
      st[7] = log1pf(fmaxf(sat_cost * sat_q, 0.0f));
      st[8] = has_f(a, kFStateLastSatConnectionCounts) ? a.f[kFStateLastSatConnectionCounts][e * sat + sid] / fmaxf(static_cast<float>(ucount), 1.0f) : 0.0f;
      st[9] = log1p_nonnegative(sat_compute_rate_for(a, e, sid) * tau / sat_flow_ref);
      st[10] = rel[0] / orbit;
      st[11] = rel[1] / orbit;
      st[12] = rel[2] / orbit;
      st[13] = relv[0] / sat_speed;
      st[14] = relv[1] / sat_speed;
      st[15] = relv[2] / sat_speed;
      st[16] = range / orbit;
      st[17] = (rel[0] * relv[0] + rel[1] * relv[1] + rel[2] * relv[2]) / (range * sat_speed);
      st[18] = elevation / (0.5f * 3.14159265358979323846f);
      st[19] = nu_eff / doppler_ref;
      st[20] = fabsf(nu_eff) / fmaxf(fp(a, kFpNuMax, 1.0f), 1.0f);
      st[21] = spectral_efficiency_device(snr);
      st[22] = listed ? 1.0f : 0.0f;
      st[23] = valid ? 1.0f : 0.0f;
      st[24] = has_f(a, kFStateLastSelectedMaskByUavSat) ? a.f[kFStateLastSelectedMaskByUavSat][(e * ucount + u) * sat + sid] : 0.0f;
      st[25] = has_f(a, kFStateLastUavToSatOutflowMatrix) ? log1p_nonnegative(a.f[kFStateLastUavToSatOutflowMatrix][(e * ucount + u) * sat + sid] / uav_flow_ref) : 0.0f;
    }
  }
  __syncthreads();
}

__device__ void write_bw_live_obs_parallel(const PackedAbi& a, int stage_slot, int e) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  const float qmax_gu = positive_config_scale(fp(a, kFpQueueMaxGu, 1.0f));
  const float qmax_uav = positive_config_scale(fp(a, kFpQueueMaxUav, 1.0f));
  const float qmax_sat = positive_config_scale(fp(a, kFpQueueMaxSat, 1.0f));
  const float tau = fp(a, kFpBwTau0, fp(a, kFpTau0, 1.0f));
  const float arrival_ref = accel_arrival_ref(a, e);
  const float gu_flow_ref = positive_config_scale(arrival_ref / fmaxf(static_cast<float>(gu), 1.0f));
  const float uav_flow_ref = positive_config_scale(arrival_ref / fmaxf(static_cast<float>(ucount), 1.0f));
  const float sat_flow_ref = sat_flow_ref_current(a, arrival_ref);
  const float eps = positive_coeff(fp(a, kFpWorkloadEps, kRelativeLogEps));
  const float gu_local_cost_ref = safe_div(1.0f, fmaxf(gu_flow_ref, eps));
  const float uav_flow_cost_ref = safe_div(1.0f, fmaxf(uav_flow_ref, eps));
  const float sat_cost_ref = safe_div(1.0f, fmaxf(sat_flow_ref, eps));
  const float uav_total_cost_ref = uav_flow_cost_ref + sat_cost_ref;
  const float gu_total_cost_ref = gu_local_cost_ref + uav_total_cost_ref;
  const float access_noise_ref = accel_access_noise_ref(a);
  const float gu_tx_power = fp(a, kFpAccessGuTxPower, fp(a, kFpLocalUavTxPower, 1.0f));

  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    const int row = row_local(a, e, u);
    float* ego = a.f[kFLiveBwObs + 0] + static_cast<int64_t>(row) * kBwEgoDim;
    for (int d = 0; d < kBwEgoDim; ++d) ego[d] = 0.0f;
    const float q = a.f[stage_f(stage_slot, kSfUavQueue)][e * ucount + u];
    const float local_cost = safe_div(1.0f, fmaxf(a.f[kFStateUavEma][e * ucount + u], eps));
    const float last_cost = uav_cost_last_route(a, e, u);
    ego[kBwEgoUavQueueSteps] = log1p_nonnegative(q / uav_flow_ref);
    ego[kBwEgoUavQueueFill] = q / qmax_uav;
    ego[kBwEgoUavLastInflowSteps] =
        has_f(a, kFStateLastGuToUavInflowByUav) ? log1p_nonnegative(a.f[kFStateLastGuToUavInflowByUav][e * ucount + u] / uav_flow_ref) : 0.0f;
    ego[kBwEgoUavLastOutflowSteps] = log1p_nonnegative(accel_uav_last_outflow(a, e, u) / uav_flow_ref);
    ego[kBwEgoUavLastDropSteps] = has_f(a, kFStateUavDrop) ? log1p_nonnegative(a.f[kFStateUavDrop][e * ucount + u] / uav_flow_ref) : 0.0f;
    ego[kBwEgoUavServiceEmaSteps] = log1p_nonnegative(a.f[kFStateUavEma][e * ucount + u] / uav_flow_ref);
    ego[kBwEgoUavLocalCostLogRatio] = logf(log_argument(local_cost / uav_flow_cost_ref));
    ego[kBwEgoUavLastTotalCostLogRatio] = logf(log_argument(last_cost / uav_total_cost_ref));
    ego[kBwEgoUavLastWorkloadLog1p] = log1pf(fmaxf(last_cost * q, 0.0f));
    ego[kBwEgoUavLastAccessInterferenceLog1p] =
        log1pf(fmaxf((has_f(a, kFStateLastAccessInterferenceByUav) ? a.f[kFStateLastAccessInterferenceByUav][e * ucount + u] : 0.0f) / access_noise_ref, 0.0f));
    ego[kBwEgoRemainingHorizonFrac] = remaining_horizon_frac_device(a, e);
  }

  for (int idx = threadIdx.x; idx < ucount * gu; idx += blockDim.x) {
    const int u = idx / gu;
    const int g = idx - u * gu;
    const int row = row_local(a, e, u);
    float* tok = a.f[kFLiveBwObs + 2] + (static_cast<int64_t>(row) * gu + g) * kBwGuTokenDim;
    for (int d = 0; d < kBwGuTokenDim; ++d) tok[d] = 0.0f;
    const int owner = static_cast<int>(a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g]);
    const bool is_valid = owner == u;
    a.b[kBLiveBwObs + 1][row * gu + g] = true;
    a.b[kBLiveBwObs + 2][row * gu + g] = is_valid;
    if (!is_valid) continue;
    const float q = a.f[stage_f(stage_slot, kSfGuQueue)][e * gu + g];
    const float expected = has_f(a, kFStateLastArrivalRateVec) ? a.f[kFStateLastArrivalRateVec][e * gu + g] * tau : 0.0f;
    const float last_arrival = has_f(a, kFStateLastGuArrival) ? a.f[kFStateLastGuArrival][e * gu + g] : 0.0f;
    const float last_outflow = has_f(a, kFStateLastGuOutflow) ? a.f[kFStateLastGuOutflow][e * gu + g] : 0.0f;
    const float drop = has_f(a, kFStateGuDrop) ? a.f[kFStateGuDrop][e * gu + g] : 0.0f;
    const float local_cost = safe_div(1.0f, fmaxf(a.f[kFStateGuEma][e * gu + g], eps));
    const float last_cost = gu_cost_last_route(a, e, g);
    const float gain_ego = a.f[stage_f(stage_slot, kSfAccessGainMatrix)][(e * gu + g) * ucount + u];
    const float full_rate = positive_config_scale(fp(a, kFpAccessBAcc, 1.0f)) *
        accel_access_se_from_gain(a, gain_ego, access_noise_ref);
    float cross_sum = 0.0f;
    float cross_max = 0.0f;
    int cross_count = 0;
    for (int v = 0; v < ucount; ++v) {
      if (v == u) continue;
      const float gain_v = a.f[stage_f(stage_slot, kSfAccessGainMatrix)][(e * gu + g) * ucount + v];
      const float scaled = log1pf(fmaxf(gu_tx_power * gain_v / access_noise_ref, 0.0f));
      cross_sum += scaled;
      cross_max = fmaxf(cross_max, scaled);
      ++cross_count;
    }
    tok[kBwGuQueueSteps] = log1p_nonnegative(q / gu_flow_ref);
    tok[kBwGuQueueFill] = q / qmax_gu;
    tok[kBwGuExpectedArrivalSteps] = log1p_nonnegative(expected / gu_flow_ref);
    tok[kBwGuLastArrivalSteps] = log1p_nonnegative(last_arrival / gu_flow_ref);
    tok[kBwGuLastOutflowSteps] = log1p_nonnegative(last_outflow / gu_flow_ref);
    tok[kBwGuLastDropSteps] = log1p_nonnegative(drop / gu_flow_ref);
    tok[kBwGuServiceEmaSteps] = log1p_nonnegative(a.f[kFStateGuEma][e * gu + g] / gu_flow_ref);
    tok[kBwGuLocalCostLogRatio] = logf(log_argument(local_cost / gu_local_cost_ref));
    tok[kBwGuLastTotalCostLogRatio] = logf(log_argument(last_cost / gu_total_cost_ref));
    tok[kBwGuLastWorkloadLog1p] = log1pf(fmaxf(last_cost * q, 0.0f));
    tok[kBwGuAccessRateFullBwRefSteps] = log1p_nonnegative(full_rate * tau / gu_flow_ref);
    tok[kBwGuCrossInterferenceMeanLog1p] = cross_count > 0 ? cross_sum / static_cast<float>(cross_count) : 0.0f;
    tok[kBwGuCrossInterferenceMaxLog1p] = cross_max;
  }

  for (int idx = threadIdx.x; idx < ucount * select_k; idx += blockDim.x) {
    const int u = idx / select_k;
    const int k = idx - u * select_k;
    const int row = row_local(a, e, u);
    float* st = a.f[kFLiveBwObs + 1] + (static_cast<int64_t>(row) * select_k + k) * kBwSatTokenDim;
    for (int d = 0; d < kBwSatTokenDim; ++d) st[d] = 0.0f;
    const int sid = static_cast<int>(a.l[stage_l(stage_slot, kSlSatSelectionMatrix)][(e * ucount + u) * select_k + k]);
    const bool valid = sid >= 0 && sid < sat;
    a.b[kBLiveBwObs + 0][row * select_k + k] = valid;
    if (!valid) continue;
    const float q = a.f[stage_f(stage_slot, kSfSatQueue)][e * sat + sid];
    const float cost = sat_cost_current(a, e, sid);
    st[kBwSatPrefixBackhaulCapacitySteps] = log1p_nonnegative(backhaul_rate_for_us(a, stage_slot, e, u, sid) * tau / uav_flow_ref);
    st[kBwSatQueueSteps] = log1p_nonnegative(q / sat_flow_ref);
    st[kBwSatQueueFill] = q / qmax_sat;
    st[kBwSatLastIncomingSteps] = log1p_nonnegative(accel_sat_last_incoming(a, e, sid) / sat_flow_ref);
    st[kBwSatLastProcessedSteps] = has_f(a, kFStateLastSatProcessed) ? log1p_nonnegative(a.f[kFStateLastSatProcessed][e * sat + sid] / sat_flow_ref) : 0.0f;
    st[kBwSatLastDropSteps] = has_f(a, kFStateSatDrop) ? log1p_nonnegative(a.f[kFStateSatDrop][e * sat + sid] / sat_flow_ref) : 0.0f;
    st[kBwSatServiceEmaSteps] = log1p_nonnegative(a.f[kFStateSatEma][e * sat + sid] / sat_flow_ref);
    st[kBwSatCostLogRatio] = logf(log_argument(cost / sat_cost_ref));
    st[kBwSatLastWorkloadLog1p] = log1pf(fmaxf(cost * q, 0.0f));
  }
}

__device__ void copy_live_to_history_local_parallel(const PackedAbi& a, int slot, int active_idx, int e) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int visible = sat_visible_width(a);
  const int accel_visible = accel_sat_width(a);
  const int users_obs = static_cast<int>(ip(a, kParamUsersObsMax));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int user_dim = static_cast<int>(ip(a, kParamUserNodeDim));
  const int ug_edge_dim = static_cast<int>(ip(a, kParamUavGuEdgeDim, 9));
  const int subset_count = static_cast<int>(ip(a, kParamSubsetCount));
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  constexpr int kSatEgoDim = 13;
  constexpr int kSatDemandDim = 8;
  constexpr int kSatRoleDim = 1;
  constexpr int kSatTokenDim = 26;
  const int live_accel_f = active_idx == 0 ? kFLiveAccelObs0 : kFLiveAccelObs1;
  const int live_accel_b = active_idx == 0 ? kBLiveAccelObs0 : kBLiveAccelObs1;
  const int peer_width = max(ucount - 1, 0);

  for (int idx = threadIdx.x; idx < ucount * kAccelEgoDim; idx += blockDim.x) {
    const int u = idx / kAccelEgoDim;
    const int d = idx - u * kAccelEgoDim;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.f[kFHistAccelLocal + 0][dst_row * kAccelEgoDim + d] = a.f[live_accel_f + 0][src_row * kAccelEgoDim + d];
  }
  for (int idx = threadIdx.x; idx < ucount * kAccelCellDim; idx += blockDim.x) {
    const int u = idx / kAccelCellDim;
    const int d = idx - u * kAccelCellDim;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.f[kFHistAccelLocal + 1][dst_row * kAccelCellDim + d] = a.f[live_accel_f + 1][src_row * kAccelCellDim + d];
  }
  for (int idx = threadIdx.x; idx < ucount * gu * kAccelGuTokenDim; idx += blockDim.x) {
    const int d = idx % kAccelGuTokenDim;
    const int tmp = idx / kAccelGuTokenDim;
    const int g = tmp % gu;
    const int u = tmp / gu;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.f[kFHistAccelLocal + 2][(dst_row * gu + g) * kAccelGuTokenDim + d] =
        a.f[live_accel_f + 2][(src_row * gu + g) * kAccelGuTokenDim + d];
  }
  for (int idx = threadIdx.x; idx < ucount * gu; idx += blockDim.x) {
    const int u = idx / gu;
    const int g = idx - u * gu;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.b[kBHistAccelLocal + 0][dst_row * gu + g] = a.b[live_accel_b + 0][src_row * gu + g];
  }

  for (int idx = threadIdx.x; idx < ucount * peer_width * kAccelPeerTokenDim; idx += blockDim.x) {
    const int d = idx % kAccelPeerTokenDim;
    const int tmp = idx / kAccelPeerTokenDim;
    const int p = tmp % peer_width;
    const int u = tmp / peer_width;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.f[kFHistAccelLocal + 3][(dst_row * peer_width + p) * kAccelPeerTokenDim + d] =
        a.f[live_accel_f + 3][(src_row * peer_width + p) * kAccelPeerTokenDim + d];
  }
  for (int idx = threadIdx.x; idx < ucount * peer_width; idx += blockDim.x) {
    const int u = peer_width > 0 ? idx / peer_width : 0;
    const int p = peer_width > 0 ? idx - u * peer_width : 0;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    if (peer_width > 0) a.b[kBHistAccelLocal + 1][dst_row * peer_width + p] = a.b[live_accel_b + 1][src_row * peer_width + p];
  }

  for (int idx = threadIdx.x; idx < ucount * accel_visible * kAccelSatTokenDim; idx += blockDim.x) {
    const int d = idx % kAccelSatTokenDim;
    const int tmp = idx / kAccelSatTokenDim;
    const int j = tmp % accel_visible;
    const int u = tmp / accel_visible;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.f[kFHistAccelLocal + 4][(dst_row * accel_visible + j) * kAccelSatTokenDim + d] =
        a.f[live_accel_f + 4][(src_row * accel_visible + j) * kAccelSatTokenDim + d];
  }
  for (int idx = threadIdx.x; idx < ucount * accel_visible; idx += blockDim.x) {
    const int u = idx / accel_visible;
    const int j = idx - u * accel_visible;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.b[kBHistAccelLocal + 2][dst_row * accel_visible + j] = a.b[live_accel_b + 2][src_row * accel_visible + j];
  }

  for (int idx = threadIdx.x; idx < ucount * kSatEgoDim; idx += blockDim.x) {
    const int u = idx / kSatEgoDim;
    const int d = idx - u * kSatEgoDim;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.f[kFHistSatLocal + 0][dst_row * kSatEgoDim + d] =
        a.f[kFLiveSatObs + 0][src_row * kSatEgoDim + d];
  }
  for (int idx = threadIdx.x; idx < ucount * kSatDemandDim; idx += blockDim.x) {
    const int u = idx / kSatDemandDim;
    const int d = idx - u * kSatDemandDim;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.f[kFHistSatLocal + 1][dst_row * kSatDemandDim + d] =
        a.f[kFLiveSatObs + 1][src_row * kSatDemandDim + d];
  }
  for (int idx = threadIdx.x; idx < ucount * kSatRoleDim; idx += blockDim.x) {
    const int u = idx / kSatRoleDim;
    const int d = idx - u * kSatRoleDim;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.f[kFHistSatLocal + 2][dst_row * kSatRoleDim + d] =
        a.f[kFLiveSatObs + 2][src_row * kSatRoleDim + d];
  }
  for (int idx = threadIdx.x; idx < ucount * visible * kSatTokenDim; idx += blockDim.x) {
    const int d = idx % kSatTokenDim;
    const int tmp = idx / kSatTokenDim;
    const int j = tmp % visible;
    const int u = tmp / visible;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.f[kFHistSatLocal + 3][(dst_row * visible + j) * kSatTokenDim + d] =
        a.f[kFLiveSatObs + 3][(src_row * visible + j) * kSatTokenDim + d];
  }
  for (int idx = threadIdx.x; idx < ucount * visible; idx += blockDim.x) {
    const int u = idx / visible;
    const int j = idx - u * visible;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.b[kBHistSatLocal + 0][dst_row * visible + j] = a.b[kBLiveSatObs + 0][src_row * visible + j];
  }
  for (int idx = threadIdx.x; idx < ucount * visible; idx += blockDim.x) {
    const int u = idx / visible;
    const int j = idx - u * visible;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.b[kBHistSatLocal + 1][dst_row * visible + j] = a.b[kBLiveSatObs + 1][src_row * visible + j];
  }
  if (has_l(a, kLLiveSatCandidateIds) && has_l(a, kLHistSatCandidateIds)) {
    for (int idx = threadIdx.x; idx < ucount * visible; idx += blockDim.x) {
      const int u = idx / visible;
      const int j = idx - u * visible;
      const int src_row = row_local(a, e, u);
      const int dst_row = hist_local_row(a, slot, e, u);
      a.l[kLHistSatCandidateIds][dst_row * visible + j] =
          a.l[kLLiveSatCandidateIds][src_row * visible + j];
    }
  }
  if (has_l(a, kLHistSatSubsetMembers) && has_l(a, kLMainSatSubsetMembersBase)) {
    for (int idx = threadIdx.x; idx < ucount * subset_count * select_k; idx += blockDim.x) {
      const int k = idx % select_k;
      const int tmp = idx / select_k;
      const int subset = tmp % subset_count;
      const int u = tmp / subset_count;
      const int dst_row = hist_local_row(a, slot, e, u);
      a.l[kLHistSatSubsetMembers][(dst_row * subset_count + subset) * select_k + k] =
          a.l[kLMainSatSubsetMembersBase][subset * select_k + k];
    }
  }
  if (has_b(a, kBHistSatSubsetMask) && has_l(a, kLMainSatSubsetMembersBase)) {
    for (int idx = threadIdx.x; idx < ucount * subset_count; idx += blockDim.x) {
      const int u = idx / subset_count;
      const int subset = idx - u * subset_count;
      const int src_row = row_local(a, e, u);
      const int dst_row = hist_local_row(a, slot, e, u);
      int valid_count = 0;
      for (int j = 0; j < visible; ++j) {
        if (a.b[kBLiveSatObs + 0][src_row * visible + j] && a.b[kBLiveSatObs + 1][src_row * visible + j]) {
          ++valid_count;
        }
      }
      int size = 0;
      bool legal = true;
      for (int k = 0; k < select_k; ++k) {
        const int member = static_cast<int>(a.l[kLMainSatSubsetMembersBase][subset * select_k + k]);
        if (member < 0) continue;
        ++size;
        if (member >= visible || !a.b[kBLiveSatObs + 0][src_row * visible + member] || !a.b[kBLiveSatObs + 1][src_row * visible + member]) {
          legal = false;
        }
      }
      if (valid_count <= 0) {
        legal = legal && size == 0 && subset == 0;
      } else {
        legal = legal && size > 0 && size <= select_k && size <= valid_count;
      }
      a.b[kBHistSatSubsetMask][dst_row * subset_count + subset] = legal;
    }
  }

  for (int idx = threadIdx.x; idx < ucount * kBwEgoDim; idx += blockDim.x) {
    const int u = idx / kBwEgoDim;
    const int d = idx - u * kBwEgoDim;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.f[kFHistBwLocal + 0][dst_row * kBwEgoDim + d] =
        a.f[kFLiveBwObs + 0][src_row * kBwEgoDim + d];
  }
  for (int idx = threadIdx.x; idx < ucount * select_k * kBwSatTokenDim; idx += blockDim.x) {
    const int d = idx % kBwSatTokenDim;
    const int tmp = idx / kBwSatTokenDim;
    const int k = tmp % select_k;
    const int u = tmp / select_k;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.f[kFHistBwLocal + 1][(dst_row * select_k + k) * kBwSatTokenDim + d] =
        a.f[kFLiveBwObs + 1][(src_row * select_k + k) * kBwSatTokenDim + d];
  }
  for (int idx = threadIdx.x; idx < ucount * select_k; idx += blockDim.x) {
    const int u = idx / select_k;
    const int k = idx - u * select_k;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.b[kBHistBwLocal + 0][dst_row * select_k + k] = a.b[kBLiveBwObs + 0][src_row * select_k + k];
  }
  for (int idx = threadIdx.x; idx < ucount * gu * kBwGuTokenDim; idx += blockDim.x) {
    const int d = idx % kBwGuTokenDim;
    const int tmp = idx / kBwGuTokenDim;
    const int g = tmp % gu;
    const int u = tmp / gu;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.f[kFHistBwLocal + 2][(dst_row * gu + g) * kBwGuTokenDim + d] =
        a.f[kFLiveBwObs + 2][(src_row * gu + g) * kBwGuTokenDim + d];
  }
  for (int idx = threadIdx.x; idx < ucount * gu; idx += blockDim.x) {
    const int u = idx / gu;
    const int g = idx - u * gu;
    const int src_row = row_local(a, e, u);
    const int dst_row = hist_local_row(a, slot, e, u);
    a.b[kBHistBwLocal + 1][dst_row * gu + g] = a.b[kBLiveBwObs + 1][src_row * gu + g];
    a.b[kBHistBwLocal + 2][dst_row * gu + g] = a.b[kBLiveBwObs + 2][src_row * gu + g];
  }

  (void)users_obs;
  (void)user_dim;
  (void)ug_edge_dim;
}

__device__ float sat_cost_current(const PackedAbi& a, int e, int s);
__device__ float uav_cost_current(const PackedAbi& a, int e, int u);
__device__ float gu_cost_current(const PackedAbi& a, int stage_slot, int e, int g);
__device__ bool sat_selected_presence_for_u(const PackedAbi& a, int e, int u, int sid);
__device__ bool sat_selected_presence_for_u_stage(const PackedAbi& a, int stage_slot, int e, int u, int sid);

__device__ float sat_flow_ref_current(const PackedAbi& a, float arrival_ref) {
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const float sat_active_ref = positive_config_scale(fp(a, kFpBwWorkloadSatActiveRef, fmaxf(static_cast<float>(sat), 1.0f)));
  return positive_config_scale(arrival_ref / sat_active_ref);
}

__device__ float uav_cost_last_route(const PackedAbi& a, int e, int u) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const float eps = positive_coeff(fp(a, kFpWorkloadEps, kRelativeLogEps));
  if (u < 0 || u >= ucount) return 0.0f;
  float selected_cost_sum = 0.0f;
  float selected_count = 0.0f;
  for (int s = 0; s < sat; ++s) {
    float selected = 0.0f;
    if (has_f(a, kFStateLastSelectedMaskByUavSat)) {
      selected = a.f[kFStateLastSelectedMaskByUavSat][(e * ucount + u) * sat + s];
    } else if (has_l(a, kLStateLastSatSelectionMatrix)) {
      const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
      for (int k = 0; k < select_k; ++k) {
        selected = selected || (a.l[kLStateLastSatSelectionMatrix][(e * ucount + u) * select_k + k] == s);
      }
    }
    if (selected > 0.5f) {
      selected_cost_sum += sat_cost_current(a, e, s);
      selected_count += 1.0f;
    }
  }
  float downstream = 0.0f;
  if (selected_count > 0.0f) {
    downstream = selected_cost_sum / selected_count;
  } else {
    for (int s = 0; s < sat; ++s) downstream += sat_cost_current(a, e, s);
    downstream = safe_div(downstream, fmaxf(static_cast<float>(sat), 1.0f));
  }
  return safe_div(1.0f, fmaxf(a.f[kFStateUavEma][e * ucount + u], eps)) + downstream;
}

__device__ float gu_cost_last_route(const PackedAbi& a, int e, int g) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const float eps = positive_coeff(fp(a, kFpWorkloadEps, kRelativeLogEps));
  if (g < 0 || g >= gu) return 0.0f;
  const int assoc = has_i(a, kIStateLastAssociation) ? a.i[kIStateLastAssociation][e * gu + g] : -1;
  float downstream = 0.0f;
  if (assoc >= 0 && assoc < ucount) {
    downstream = uav_cost_last_route(a, e, assoc);
  } else {
    for (int u = 0; u < ucount; ++u) downstream += uav_cost_last_route(a, e, u);
    downstream = safe_div(downstream, fmaxf(static_cast<float>(ucount), 1.0f));
  }
  return safe_div(1.0f, fmaxf(a.f[kFStateGuEma][e * gu + g], eps)) + downstream;
}

__device__ float* last_route_cache_uav_ptr(float* cache) {
  return cache;
}

__device__ float* last_route_cache_gu_ptr(const PackedAbi& a, float* cache) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  return cache + ucount;
}

__device__ void last_route_cost_cache_fill_parallel(const PackedAbi& a, int e, float* cache) {
  if (cache == nullptr) return;
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  const float eps = positive_coeff(fp(a, kFpWorkloadEps, kRelativeLogEps));
  float* uav_cost = last_route_cache_uav_ptr(cache);
  float* gu_cost = last_route_cache_gu_ptr(a, cache);
  __shared__ float sh_reduce_last_route[128];
  __shared__ float sh_mean_uav_last_route;

  float uav_sum_local = 0.0f;
  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    float selected_cost_sum = 0.0f;
    float selected_count = 0.0f;
    for (int s = 0; s < sat; ++s) {
      float selected = 0.0f;
      if (has_f(a, kFStateLastSelectedMaskByUavSat)) {
        selected = a.f[kFStateLastSelectedMaskByUavSat][(e * ucount + u) * sat + s];
      } else if (has_l(a, kLStateLastSatSelectionMatrix)) {
        for (int k = 0; k < select_k; ++k) {
          selected = selected || (a.l[kLStateLastSatSelectionMatrix][(e * ucount + u) * select_k + k] == s);
        }
      }
      if (selected > 0.5f) {
        selected_cost_sum += sat_cost_current(a, e, s);
        selected_count += 1.0f;
      }
    }
    float downstream = 0.0f;
    if (selected_count > 0.0f) {
      downstream = selected_cost_sum / selected_count;
    } else {
      for (int s = 0; s < sat; ++s) downstream += sat_cost_current(a, e, s);
      downstream = safe_div(downstream, fmaxf(static_cast<float>(sat), 1.0f));
    }
    const float cost = safe_div(1.0f, fmaxf(a.f[kFStateUavEma][e * ucount + u], eps)) + downstream;
    uav_cost[u] = cost;
    uav_sum_local += cost;
  }
  const float uav_sum = block_reduce_sum_128(uav_sum_local, sh_reduce_last_route);
  if (threadIdx.x == 0) {
    sh_mean_uav_last_route = safe_div(uav_sum, fmaxf(static_cast<float>(ucount), 1.0f));
  }
  __syncthreads();

  for (int g = threadIdx.x; g < gu; g += blockDim.x) {
    const int assoc = has_i(a, kIStateLastAssociation) ? a.i[kIStateLastAssociation][e * gu + g] : -1;
    const float downstream = (assoc >= 0 && assoc < ucount) ? uav_cost[assoc] : sh_mean_uav_last_route;
    gu_cost[g] = safe_div(1.0f, fmaxf(a.f[kFStateGuEma][e * gu + g], eps)) + downstream;
  }
  __syncthreads();
}

__device__ float uav_cost_last_route_cached(const PackedAbi& a, int e, int u, float* cache) {
  if (cache == nullptr) return uav_cost_last_route(a, e, u);
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  if (u < 0 || u >= ucount) return 0.0f;
  return last_route_cache_uav_ptr(cache)[u];
}

__device__ float gu_cost_last_route_cached(const PackedAbi& a, int e, int g, float* cache) {
  if (cache == nullptr) return gu_cost_last_route(a, e, g);
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  if (g < 0 || g >= gu) return 0.0f;
  return last_route_cache_gu_ptr(a, cache)[g];
}

__device__ float selected_shared_fraction_stage(const PackedAbi& a, int stage_slot, int e, int u, int v) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  if (u == v || select_k <= 0) return 0.0f;
  float shared = 0.0f;
  for (int ku = 0; ku < select_k; ++ku) {
    const int64_t su = a.l[stage_l(stage_slot, kSlSatSelectionMatrix)][(e * ucount + u) * select_k + ku];
    if (su < 0) continue;
    for (int kv = 0; kv < select_k; ++kv) {
      const int64_t sv = a.l[stage_l(stage_slot, kSlSatSelectionMatrix)][(e * ucount + v) * select_k + kv];
      if (su == sv) {
        shared += 1.0f;
        break;
      }
    }
  }
  return safe_div(shared, fmaxf(static_cast<float>(select_k), 1.0f));
}

__device__ float selected_shared_fraction_last(const PackedAbi& a, int e, int u, int v) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  if (u == v || select_k <= 0) return 0.0f;
  float shared = 0.0f;
  if (has_f(a, kFStateLastSelectedMaskByUavSat)) {
    for (int s = 0; s < sat; ++s) {
      const float au = a.f[kFStateLastSelectedMaskByUavSat][(e * ucount + u) * sat + s];
      const float av = a.f[kFStateLastSelectedMaskByUavSat][(e * ucount + v) * sat + s];
      shared += (au > 0.5f && av > 0.5f) ? 1.0f : 0.0f;
    }
  } else if (has_l(a, kLStateLastSatSelectionMatrix)) {
    for (int ku = 0; ku < select_k; ++ku) {
      const int64_t su = a.l[kLStateLastSatSelectionMatrix][(e * ucount + u) * select_k + ku];
      if (su < 0) continue;
      for (int kv = 0; kv < select_k; ++kv) {
        const int64_t sv = a.l[kLStateLastSatSelectionMatrix][(e * ucount + v) * select_k + kv];
        if (su == sv) {
          shared += 1.0f;
          break;
        }
      }
    }
  }
  return safe_div(shared, fmaxf(static_cast<float>(select_k), 1.0f));
}

__device__ void write_world_from_stage_parallel(const PackedAbi& a, int stage_slot, int world_f, int world_b, int row, int e) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int active = active_width(a);
  const int uav_dim = kCriticUavDim;
  const int user_dim = kCriticGuDim;
  const int sat_dim = kCriticSatDim;
  const int ug_dim = kCriticUgDim;
  const int us_dim = kCriticUsDim;
  const int uu_dim = kCriticUuDim;
  __shared__ float sh_world_reduce[128];
  const int world_global = world_global_f_index(world_f);
  const int world_sat_ids = world_sat_ids_l_index(world_f);
  const int stage_id = static_cast<int>(a.l[stage_l(stage_slot, kSlStageId)][e]);
  const bool prefix_bw_known = stage_id >= 1;
  const bool prefix_sat_known = stage_id >= 2;
  const bool prefix_workload_known = stage_id >= 2;
  const float map_size = positive_config_scale(fp(a, kFpMapSize, 1.0f));
  const float vmax = positive_config_scale(fp(a, kFpVMax, 1.0f));
  const float emax = positive_config_scale(fp(a, kFpUavEnergyInit, 1.0f));
  const float qmax_gu = positive_config_scale(fp(a, kFpQueueMaxGu, 1.0f));
  const float qmax_uav = positive_config_scale(fp(a, kFpQueueMaxUav, 1.0f));
  const float qmax_sat = positive_config_scale(fp(a, kFpQueueMaxSat, 1.0f));
  const float tau = fp(a, kFpBwTau0, fp(a, kFpTau0, 1.0f));
  const float arrival_ref = require_positive_reward_ref(has_f(a, kFStateArrivalRef) ? a.f[kFStateArrivalRef][e] : fmaxf(static_cast<float>(gu), 1.0f) * tau);
  const float gu_flow_ref = positive_config_scale(arrival_ref / fmaxf(static_cast<float>(gu), 1.0f));
  const float uav_flow_ref = positive_config_scale(arrival_ref / fmaxf(static_cast<float>(ucount), 1.0f));
  const float sat_flow_ref = sat_flow_ref_current(a, arrival_ref);
  const float eps = positive_coeff(fp(a, kFpWorkloadEps, kRelativeLogEps));
  const float gu_local_cost_ref = safe_div(1.0f, fmaxf(gu_flow_ref, eps));
  const float sat_cost_ref = safe_div(1.0f, fmaxf(sat_flow_ref, eps));
  const float uav_total_cost_ref = safe_div(1.0f, fmaxf(uav_flow_ref, eps)) + sat_cost_ref;
  const float gu_total_cost_ref = gu_local_cost_ref + uav_total_cost_ref;
  const float orbit = fmaxf(fp(a, kFpLocalEarthRadius, 0.0f) + fp(a, kFpSatHeight, 0.0f), 1.0f);
  const float sat_speed = sqrtf(3.986004418e14f / orbit);
  const float b_acc = positive_config_scale(fp(a, kFpAccessBAcc, 1.0f));
  const float access_noise_full_band = positive_config_scale(fp(a, kFpAccessNoiseDensity, fp(a, kFpLocalNoiseDensity, kDefaultNoiseDensity)) * fp(a, kFpAccessNoiseFigureLinear, 1.0f) * b_acc);
  const float b_backhaul_ref = positive_config_scale(fp(a, kFpBwEffectiveBSatTotal, fp(a, kFpBSatTotal, 1.0f)));
  const float backhaul_noise_full_band = positive_config_scale(fp(a, kFpBwNoiseDensity, kDefaultNoiseDensity) * fp(a, kFpBwNoiseFigureLinear, 1.0f) * b_backhaul_ref);
  if (threadIdx.x == 0 && has_f(a, world_global)) {
    for (int d = 0; d < kCriticGlobalDim; ++d) a.f[world_global][row * kCriticGlobalDim + d] = 0.0f;
    a.f[world_global][row * kCriticGlobalDim + kGlobalRemainingHorizonFrac] = remaining_horizon_frac_device(a, e);
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < ucount; idx += blockDim.x) {
    const int u = idx;
    for (int d = 0; d < uav_dim; ++d) a.f[world_f + 0][(row * ucount + u) * uav_dim + d] = 0.0f;
    const int base = (row * ucount + u) * uav_dim;
    const float q = a.f[stage_f(stage_slot, kSfUavQueue)][e * ucount + u];
    const float last_cost = uav_cost_last_route(a, e, u);
    const float prefix_cost = prefix_workload_known ? uav_cost_current(a, e, u) : 0.0f;
    a.f[world_f + 0][base + kUavX] = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 0] / map_size;
    a.f[world_f + 0][base + kUavY] = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 1] / map_size;
    a.f[world_f + 0][base + kUavVx] = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 0] / vmax;
    a.f[world_f + 0][base + kUavVy] = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 1] / vmax;
    a.f[world_f + 0][base + kUavEnergy] = a.f[stage_f(stage_slot, kSfUavEnergy)][e * ucount + u] / emax;
    a.f[world_f + 0][base + kUavQueueSteps] = log1p_nonnegative(q / uav_flow_ref);
    a.f[world_f + 0][base + kUavQueueFill] = q / qmax_uav;
    a.f[world_f + 0][base + kUavLastInflowSteps] = has_f(a, kFStateLastGuToUavInflowByUav) ? log1p_nonnegative(a.f[kFStateLastGuToUavInflowByUav][e * ucount + u] / uav_flow_ref) : 0.0f;
    float last_uav_out = 0.0f;
    if (has_f(a, kFStateLastUavToSatOutflowMatrix)) {
      for (int s = 0; s < sat; ++s) last_uav_out += a.f[kFStateLastUavToSatOutflowMatrix][(e * ucount + u) * sat + s];
    }
    a.f[world_f + 0][base + kUavLastOutflowSteps] = log1p_nonnegative(last_uav_out / uav_flow_ref);
    a.f[world_f + 0][base + kUavLastDropSteps] = has_f(a, kFStateUavDrop) ? log1p_nonnegative(a.f[kFStateUavDrop][e * ucount + u] / uav_flow_ref) : 0.0f;
    a.f[world_f + 0][base + kUavServiceEmaSteps] = log1p_nonnegative(a.f[kFStateUavEma][e * ucount + u] / uav_flow_ref);
    a.f[world_f + 0][base + kUavLocalCostLogRatio] = logf(log_argument(safe_div(1.0f, fmaxf(a.f[kFStateUavEma][e * ucount + u], eps)) / safe_div(1.0f, fmaxf(uav_flow_ref, eps))));
    a.f[world_f + 0][base + kUavLastTotalCostLogRatio] = logf(log_argument(last_cost / uav_total_cost_ref));
    a.f[world_f + 0][base + kUavPrefixTotalCostLogRatio] = prefix_workload_known ? logf(log_argument(prefix_cost / uav_total_cost_ref)) : 0.0f;
    a.f[world_f + 0][base + kUavPrefixCostKnown] = prefix_workload_known ? 1.0f : 0.0f;
    a.f[world_f + 0][base + kUavLastWorkloadLog1p] = log1pf(fmaxf(last_cost * q, 0.0f));
    a.f[world_f + 0][base + kUavPrefixWorkloadLog1p] = prefix_workload_known ? log1pf(fmaxf(prefix_cost * q, 0.0f)) : 0.0f;
    float bw_valid_count = 0.0f;
    if (prefix_bw_known) {
      for (int g = 0; g < gu; ++g) bw_valid_count += a.f[stage_f(stage_slot, kSfBwValidFlag)][(e * ucount + u) * gu + g] > 0.5f ? 1.0f : 0.0f;
    }
    a.f[world_f + 0][base + kUavPrefixBwValidCountFrac] = prefix_bw_known ? safe_div(bw_valid_count, fmaxf(static_cast<float>(gu), 1.0f)) : 0.0f;
    const float interference = has_f(a, kFStateLastAccessInterferenceByUav) ? a.f[kFStateLastAccessInterferenceByUav][e * ucount + u] : 0.0f;
    a.f[world_f + 0][base + kUavLastAccessInterferenceLog1p] = log1pf(fmaxf(interference / access_noise_full_band, 0.0f));
    if (has_f(a, world_global)) {
      atomicAdd(&a.f[world_global][row * kCriticGlobalDim + kGlobalTotalUavQueueSteps], q / uav_flow_ref);
      atomicAdd(&a.f[world_global][row * kCriticGlobalDim + kGlobalTotalUavDropSteps], has_f(a, kFStateUavDrop) ? a.f[kFStateUavDrop][e * ucount + u] / uav_flow_ref : 0.0f);
      atomicAdd(&a.f[world_global][row * kCriticGlobalDim + kGlobalTotalLastUavOutflowSteps], last_uav_out / uav_flow_ref);
      atomicAdd(&a.f[world_global][row * kCriticGlobalDim + kGlobalTotalLastWeightedWorkloadSteps], last_cost * q);
      if (prefix_workload_known) atomicAdd(&a.f[world_global][row * kCriticGlobalDim + kGlobalTotalPrefixWeightedWorkloadSteps], prefix_cost * q);
    }
  }
  for (int g = threadIdx.x; g < gu; g += blockDim.x) {
    for (int d = 0; d < user_dim; ++d) a.f[world_f + 1][(row * gu + g) * user_dim + d] = 0.0f;
    const int base = (row * gu + g) * user_dim;
    const float q = a.f[stage_f(stage_slot, kSfGuQueue)][e * gu + g];
    const float expected = has_f(a, kFStateLastArrivalRateVec) ? a.f[kFStateLastArrivalRateVec][e * gu + g] * tau : 0.0f;
    const float last_cost = gu_cost_last_route(a, e, g);
    const float prefix_cost = prefix_workload_known ? gu_cost_current(a, stage_slot, e, g) : 0.0f;
    a.f[world_f + 1][base + kGuX] = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + 0] / map_size;
    a.f[world_f + 1][base + kGuY] = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + 1] / map_size;
    a.f[world_f + 1][base + kGuQueueSteps] = log1p_nonnegative(q / gu_flow_ref);
    a.f[world_f + 1][base + kGuQueueFill] = q / qmax_gu;
    a.f[world_f + 1][base + kGuExpectedArrivalSteps] = log1p_nonnegative(expected / gu_flow_ref);
    a.f[world_f + 1][base + kGuLastArrivalSteps] = has_f(a, kFStateLastGuArrival) ? log1p_nonnegative(a.f[kFStateLastGuArrival][e * gu + g] / gu_flow_ref) : 0.0f;
    a.f[world_f + 1][base + kGuLastOutflowSteps] = has_f(a, kFStateLastGuOutflow) ? log1p_nonnegative(a.f[kFStateLastGuOutflow][e * gu + g] / gu_flow_ref) : 0.0f;
    a.f[world_f + 1][base + kGuLastDropSteps] = has_f(a, kFStateGuDrop) ? log1p_nonnegative(a.f[kFStateGuDrop][e * gu + g] / gu_flow_ref) : 0.0f;
    a.f[world_f + 1][base + kGuServiceEmaSteps] = log1p_nonnegative(a.f[kFStateGuEma][e * gu + g] / gu_flow_ref);
    a.f[world_f + 1][base + kGuLocalCostLogRatio] = logf(log_argument(safe_div(1.0f, fmaxf(a.f[kFStateGuEma][e * gu + g], eps)) / gu_local_cost_ref));
    a.f[world_f + 1][base + kGuLastTotalCostLogRatio] = logf(log_argument(last_cost / gu_total_cost_ref));
    a.f[world_f + 1][base + kGuPrefixTotalCostLogRatio] = prefix_workload_known ? logf(log_argument(prefix_cost / gu_total_cost_ref)) : 0.0f;
    a.f[world_f + 1][base + kGuPrefixCostKnown] = prefix_workload_known ? 1.0f : 0.0f;
    a.f[world_f + 1][base + kGuLastWorkloadLog1p] = log1pf(fmaxf(last_cost * q, 0.0f));
    a.f[world_f + 1][base + kGuPrefixWorkloadLog1p] = prefix_workload_known ? log1pf(fmaxf(prefix_cost * q, 0.0f)) : 0.0f;
    a.b[world_b + 0][row * gu + g] = true;
    if (has_f(a, world_global)) {
      atomicAdd(&a.f[world_global][row * kCriticGlobalDim + kGlobalTotalGuQueueSteps], q / gu_flow_ref);
      atomicAdd(&a.f[world_global][row * kCriticGlobalDim + kGlobalTotalGuDropSteps], has_f(a, kFStateGuDrop) ? a.f[kFStateGuDrop][e * gu + g] / gu_flow_ref : 0.0f);
      atomicAdd(&a.f[world_global][row * kCriticGlobalDim + kGlobalTotalExpectedArrivalSteps], expected / gu_flow_ref);
      atomicAdd(&a.f[world_global][row * kCriticGlobalDim + kGlobalTotalLastGuOutflowSteps], has_f(a, kFStateLastGuOutflow) ? a.f[kFStateLastGuOutflow][e * gu + g] / gu_flow_ref : 0.0f);
      atomicAdd(&a.f[world_global][row * kCriticGlobalDim + kGlobalTotalLastWeightedWorkloadSteps], last_cost * q);
      if (prefix_workload_known) atomicAdd(&a.f[world_global][row * kCriticGlobalDim + kGlobalTotalPrefixWeightedWorkloadSteps], prefix_cost * q);
    }
  }
  for (int idx = threadIdx.x; idx < ucount * gu; idx += blockDim.x) {
    const int u = idx / gu;
    const int g = idx - u * gu;
    const int out = (row * ucount + u) * gu + g;
    for (int d = 0; d < ug_dim; ++d) a.f[world_f + 3][out * ug_dim + d] = 0.0f;
    const float dx = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + 0] - a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 0];
    const float dy = a.f[stage_f(stage_slot, kSfGuPos)][(e * gu + g) * 2 + 1] - a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 1];
    const float horiz = sqrtf(dx * dx + dy * dy);
    const float elev = atan2f(fp(a, kFpCandidateUavHeight, fp(a, kFpSatGeomUavHeight, 0.0f)), geometry_denominator(horiz));
    const float gain = a.f[stage_f(stage_slot, kSfAccessGainMatrix)][(e * gu + g) * ucount + u];
    const float access_snr_ref = fp(a, kFpAccessGuTxPower, fp(a, kFpLocalUavTxPower, 1.0f)) * gain / access_noise_full_band;
    a.f[world_f + 3][out * ug_dim + kUgRelX] = dx / map_size;
    a.f[world_f + 3][out * ug_dim + kUgRelY] = dy / map_size;
    a.f[world_f + 3][out * ug_dim + kUgHorizontalDist] = horiz / map_size;
    a.f[world_f + 3][out * ug_dim + kUgElevationNorm] = elev / (0.5f * 3.14159265358979323846f);
    a.f[world_f + 3][out * ug_dim + kUgAccessSeRef] = access_spectral_efficiency_device(a, access_snr_ref);
    const float last_bw_fraction = has_f(a, kFStateLastBwFractionByUavGu) ? a.f[kFStateLastBwFractionByUavGu][(e * ucount + u) * gu + g] : 0.0f;
    a.f[world_f + 3][out * ug_dim + kUgLastBwFraction] = last_bw_fraction;
    a.f[world_f + 3][out * ug_dim + kUgLastServedFlag] = last_bw_fraction;
    a.f[world_f + 3][out * ug_dim + kUgPrefixBwValidFlag] = prefix_bw_known ? a.f[stage_f(stage_slot, kSfBwValidFlag)][(e * ucount + u) * gu + g] : 0.0f;
    a.f[world_f + 3][out * ug_dim + kUgPrefixBwValidKnown] = prefix_bw_known ? 1.0f : 0.0f;
    a.b[world_b + 2][out] = true;
  }
  for (int aidx = threadIdx.x; aidx < active; aidx += blockDim.x) {
    const int sid = static_cast<int>(a.l[stage_l(stage_slot, kSlActiveSatIds)][e * active + aidx]);
    const bool valid_sat = sid >= 0 && sid < sat;
    for (int d = 0; d < sat_dim; ++d) a.f[world_f + 2][(row * active + aidx) * sat_dim + d] = 0.0f;
    if (has_l(a, world_sat_ids)) a.l[world_sat_ids][row * active + aidx] = valid_sat ? static_cast<int64_t>(sid) : -1;
    if (valid_sat) {
      const int base = (row * active + aidx) * sat_dim;
      for (int d = 0; d < 3 && d < sat_dim; ++d) a.f[world_f + 2][(row * active + aidx) * sat_dim + d] = a.f[stage_f(stage_slot, kSfSatPosActive)][(e * active + aidx) * 3 + d] / orbit;
      for (int d = 0; d < 3 && d + 3 < sat_dim; ++d) a.f[world_f + 2][(row * active + aidx) * sat_dim + 3 + d] = a.f[stage_f(stage_slot, kSfSatVelActive)][(e * active + aidx) * 3 + d] / sat_speed;
      const float q = a.f[stage_f(stage_slot, kSfSatQueueActive)][e * active + aidx];
      const float sat_cost = sat_cost_current(a, e, sid);
      float incoming = 0.0f;
      if (has_f(a, kFStateLastUavToSatOutflowMatrix)) {
        for (int u = 0; u < ucount; ++u) incoming += a.f[kFStateLastUavToSatOutflowMatrix][(e * ucount + u) * sat + sid];
      }
      a.f[world_f + 2][base + kSatQueueSteps] = log1p_nonnegative(q / sat_flow_ref);
      a.f[world_f + 2][base + kSatQueueFill] = q / qmax_sat;
      a.f[world_f + 2][base + kSatLastIncomingSteps] = log1p_nonnegative(incoming / sat_flow_ref);
      a.f[world_f + 2][base + kSatLastProcessedSteps] = has_f(a, kFStateLastSatProcessed) ? log1p_nonnegative(a.f[kFStateLastSatProcessed][e * sat + sid] / sat_flow_ref) : 0.0f;
      a.f[world_f + 2][base + kSatLastDropSteps] = has_f(a, kFStateSatDrop) ? log1p_nonnegative(a.f[kFStateSatDrop][e * sat + sid] / sat_flow_ref) : 0.0f;
      a.f[world_f + 2][base + kSatServiceEmaSteps] = log1p_nonnegative(a.f[kFStateSatEma][e * sat + sid] / sat_flow_ref);
      a.f[world_f + 2][base + kSatCostLogRatio] = logf(log_argument(sat_cost / sat_cost_ref));
      a.f[world_f + 2][base + kSatLastWorkloadLog1p] = log1pf(fmaxf(sat_cost * q, 0.0f));
      a.f[world_f + 2][base + kSatPrefixSelectedLoadFrac] = prefix_sat_known ? sat_selected_load_frac_for_stage(a, stage_slot, e, sid) : 0.0f;
      a.f[world_f + 2][base + kSatPrefixLoadKnown] = prefix_sat_known ? 1.0f : 0.0f;
      a.f[world_f + 2][base + kSatLastSelectedLoadFrac] = has_f(a, kFStateLastSatConnectionCounts) ? a.f[kFStateLastSatConnectionCounts][e * sat + sid] / fmaxf(static_cast<float>(ucount), 1.0f) : 0.0f;
      a.f[world_f + 2][base + kSatProcCapacitySteps] = log1p_nonnegative(sat_compute_rate_for(a, e, sid) * tau / sat_flow_ref);
    }
    a.b[world_b + 1][row * active + aidx] = valid_sat;
  }
  for (int idx = threadIdx.x; idx < ucount * active; idx += blockDim.x) {
    const int u = idx / active;
    const int aidx = idx - u * active;
    const int sid = static_cast<int>(a.l[stage_l(stage_slot, kSlActiveSatIds)][e * active + aidx]);
    const bool valid_sat = sid >= 0 && sid < sat;
    const int out = (row * ucount + u) * active + aidx;
    for (int d = 0; d < us_dim; ++d) a.f[world_f + 4][out * us_dim + d] = 0.0f;
    if (valid_sat) {
      float rel[3];
      float relv[3];
      for (int d = 0; d < 3; ++d) {
        rel[d] = a.f[stage_f(stage_slot, kSfUsRelPosActive)][((e * ucount + u) * active + aidx) * 3 + d];
        relv[d] = a.f[stage_f(stage_slot, kSfUsRelVelActive)][((e * ucount + u) * active + aidx) * 3 + d];
        a.f[world_f + 4][out * us_dim + kUsRelX + d] = rel[d] / orbit;
        a.f[world_f + 4][out * us_dim + kUsRelVx + d] = relv[d] / sat_speed;
      }
      const float range = sqrtf(geometry_denominator(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]));
      const float radial = (rel[0] * relv[0] + rel[1] * relv[1] + rel[2] * relv[2]) / (range * sat_speed);
      const float nu_eff = a.f[stage_f(stage_slot, kSfUsNuEffActive)][(e * ucount + u) * active + aidx];
      const float doppler_ref = positive_config_scale(fp(a, kFpSatCarrierFreq, fp(a, kFpCarrierFreq, 1.0f)) * sat_speed / fmaxf(fp(a, kFpSpeedOfLight, 299792458.0f), 1.0f));
      const float gain = a.f[stage_f(stage_slot, kSfUsGainActive)][(e * ucount + u) * active + aidx];
      float snr = fp(a, kFpBwUavTxPower, 1.0f) * gain / backhaul_noise_full_band;
      if (ip(a, kParamDopplerAttenEnabled)) {
        const float spacing = fp(a, kFpSubcarrierSpacing, 0.0f);
        if (spacing > 0.0f) {
          const float s = sinc_pi_device(nu_eff / spacing);
          snr *= s * s;
        }
      }
      a.f[world_f + 4][out * us_dim + kUsRadialVelocityNorm] = radial;
      a.f[world_f + 4][out * us_dim + kUsRangeNorm] = range / orbit;
      a.f[world_f + 4][out * us_dim + kUsElevationNorm] = a.f[stage_f(stage_slot, kSfElevationMatrix)][(e * ucount + u) * sat + sid] / (0.5f * 3.14159265358979323846f);
      a.f[world_f + 4][out * us_dim + kUsDopplerNorm] = ip(a, kParamDopplerObserved) ? nu_eff / doppler_ref : 0.0f;
      a.f[world_f + 4][out * us_dim + kUsDopplerMargin] = ip(a, kParamDopplerObserved) ? nu_eff / fmaxf(fp(a, kFpNuMax, 1.0f), 1.0f) : 0.0f;
      a.f[world_f + 4][out * us_dim + kUsBackhaulSeRef] = spectral_efficiency_device(snr);
      a.f[world_f + 4][out * us_dim + kUsVisibleFlag] = a.f[stage_f(stage_slot, kSfVisibleFlagActive)][(e * ucount + u) * active + aidx];
      a.f[world_f + 4][out * us_dim + kUsValidFlag] = a.f[stage_f(stage_slot, kSfUsValidFlagActive)][(e * ucount + u) * active + aidx];
      a.f[world_f + 4][out * us_dim + kUsLastSelectedFlag] = has_f(a, kFStateLastSelectedMaskByUavSat) ? a.f[kFStateLastSelectedMaskByUavSat][(e * ucount + u) * sat + sid] : 0.0f;
      const float prefix_selected = prefix_sat_known ? sat_selected_presence_for_u_stage(a, stage_slot, e, u, sid) : 0.0f;
      a.f[world_f + 4][out * us_dim + kUsPrefixSelectedFlag] = prefix_selected;
      a.f[world_f + 4][out * us_dim + kUsPrefixSelectedKnown] = prefix_sat_known ? 1.0f : 0.0f;
      a.f[world_f + 4][out * us_dim + kUsPrefixBackhaulCapacitySteps] = prefix_sat_known && prefix_selected > 0.5f ? log1p_nonnegative(backhaul_rate_for_selected_us(a, stage_slot, e, u, sid) * tau / uav_flow_ref) : 0.0f;
    }
    a.b[world_b + 3][out] = valid_sat;
  }
  for (int idx = threadIdx.x; idx < ucount * ucount; idx += blockDim.x) {
    const int u = idx / ucount;
    const int v = idx - u * ucount;
    const int out = (row * ucount + u) * ucount + v;
    const bool offdiag = u != v;
    for (int d = 0; d < uu_dim; ++d) a.f[world_f + 5][out * uu_dim + d] = 0.0f;
    const float ux = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 0];
    const float uy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + u) * 2 + 1];
    const float vx = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + v) * 2 + 0];
    const float vy = a.f[stage_f(stage_slot, kSfUavPos)][(e * ucount + v) * 2 + 1];
    const float uvx = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 0];
    const float uvy = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + u) * 2 + 1];
    const float vvx = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + v) * 2 + 0];
    const float vvy = a.f[stage_f(stage_slot, kSfUavVel)][(e * ucount + v) * 2 + 1];
    const float dx = vx - ux;
    const float dy = vy - uy;
    const float dvx = vvx - uvx;
    const float dvy = vvy - uvy;
    const float dist = sqrtf(dx * dx + dy * dy);
    a.f[world_f + 5][out * uu_dim + kUuRelX] = dx / map_size;
    a.f[world_f + 5][out * uu_dim + kUuRelY] = dy / map_size;
    a.f[world_f + 5][out * uu_dim + kUuRelVx] = dvx / vmax;
    a.f[world_f + 5][out * uu_dim + kUuRelVy] = dvy / vmax;
    a.f[world_f + 5][out * uu_dim + kUuDistNorm] = dist / map_size;
    a.f[world_f + 5][out * uu_dim + kUuClosingSpeedNorm] = -safe_div(dx * dvx + dy * dvy, dist * vmax);
    const float d_alert = ip(a, kParamUseAvoidance) ? fp(a, kFpAvoidanceAlertFactor, 1.0f) * fp(a, kFpDSafe, 1.0f) : fp(a, kFpDSafe, 1.0f);
    a.f[world_f + 5][out * uu_dim + kUuAlertFlag] = offdiag && dist < d_alert ? 1.0f : 0.0f;
    a.f[world_f + 5][out * uu_dim + kUuUnsafeFlag] = offdiag && dist < fp(a, kFpDSafe, 1.0f) ? 1.0f : 0.0f;
    a.f[world_f + 5][out * uu_dim + kUuLastSharedSatFrac] = selected_shared_fraction_last(a, e, u, v);
    a.f[world_f + 5][out * uu_dim + kUuPrefixSharedSatFrac] = prefix_sat_known ? selected_shared_fraction_stage(a, stage_slot, e, u, v) : 0.0f;
    a.f[world_f + 5][out * uu_dim + kUuPrefixSharedSatKnown] = prefix_sat_known ? 1.0f : 0.0f;
    a.b[world_b + 4][out] = offdiag;
  }
  __syncthreads();
  if (has_f(a, world_global)) {
    float local_total_sat_queue = 0.0f;
    float local_total_sat_drop = 0.0f;
    float local_total_sat_processed = 0.0f;
    float local_total_sat_workload = 0.0f;
    float local_total_sat_prefix_workload = 0.0f;
    float local_nontoken_count = 0.0f;
    float local_nontoken_queue = 0.0f;
    float local_nontoken_drop = 0.0f;
    float local_nontoken_processed = 0.0f;
    float local_nontoken_workload = 0.0f;
    float local_nontoken_drop_workload = 0.0f;
    for (int s = threadIdx.x; s < sat; s += blockDim.x) {
      bool token = false;
      for (int aidx = 0; aidx < active; ++aidx) {
        const int sid = static_cast<int>(a.l[stage_l(stage_slot, kSlActiveSatIds)][e * active + aidx]);
        if (sid == s) {
          token = true;
          break;
        }
      }
      const float q = has_f(a, kFStateSatQueue) ? a.f[kFStateSatQueue][e * sat + s] : 0.0f;
      const float drop = has_f(a, kFStateSatDrop) ? a.f[kFStateSatDrop][e * sat + s] : 0.0f;
      const float processed = has_f(a, kFStateLastSatProcessed) ? a.f[kFStateLastSatProcessed][e * sat + s] : 0.0f;
      const float cost = sat_cost_current(a, e, s);
      local_total_sat_queue += q / sat_flow_ref;
      local_total_sat_drop += drop / sat_flow_ref;
      local_total_sat_processed += processed / sat_flow_ref;
      local_total_sat_workload += cost * q;
      local_total_sat_prefix_workload += prefix_workload_known ? cost * q : 0.0f;
      if (!token) {
        local_nontoken_count += 1.0f / fmaxf(static_cast<float>(sat), 1.0f);
        local_nontoken_queue += q / sat_flow_ref;
        local_nontoken_drop += drop / sat_flow_ref;
        local_nontoken_processed += processed / sat_flow_ref;
        local_nontoken_workload += cost * q;
        local_nontoken_drop_workload += cost * drop;
      }
    }
    const int global_base = row * kCriticGlobalDim;
    float reduced = block_reduce_sum_128(local_total_sat_queue, sh_world_reduce);
    if (threadIdx.x == 0) a.f[world_global][global_base + kGlobalTotalSatQueueSteps] += reduced;
    __syncthreads();
    reduced = block_reduce_sum_128(local_total_sat_drop, sh_world_reduce);
    if (threadIdx.x == 0) a.f[world_global][global_base + kGlobalTotalSatDropSteps] += reduced;
    __syncthreads();
    reduced = block_reduce_sum_128(local_total_sat_processed, sh_world_reduce);
    if (threadIdx.x == 0) a.f[world_global][global_base + kGlobalTotalLastSatProcessedSteps] += reduced;
    __syncthreads();
    reduced = block_reduce_sum_128(local_total_sat_workload, sh_world_reduce);
    if (threadIdx.x == 0) a.f[world_global][global_base + kGlobalTotalLastWeightedWorkloadSteps] += reduced;
    __syncthreads();
    reduced = block_reduce_sum_128(local_total_sat_prefix_workload, sh_world_reduce);
    if (threadIdx.x == 0) a.f[world_global][global_base + kGlobalTotalPrefixWeightedWorkloadSteps] += reduced;
    __syncthreads();
    reduced = block_reduce_sum_128(local_nontoken_count, sh_world_reduce);
    if (threadIdx.x == 0) a.f[world_global][global_base + kGlobalNonTokenSatCountFrac] += reduced;
    __syncthreads();
    reduced = block_reduce_sum_128(local_nontoken_queue, sh_world_reduce);
    if (threadIdx.x == 0) a.f[world_global][global_base + kGlobalNonTokenSatQueueSteps] += reduced;
    __syncthreads();
    reduced = block_reduce_sum_128(local_nontoken_drop, sh_world_reduce);
    if (threadIdx.x == 0) a.f[world_global][global_base + kGlobalNonTokenSatDropSteps] += reduced;
    __syncthreads();
    reduced = block_reduce_sum_128(local_nontoken_processed, sh_world_reduce);
    if (threadIdx.x == 0) a.f[world_global][global_base + kGlobalNonTokenSatLastProcessedSteps] += reduced;
    __syncthreads();
    reduced = block_reduce_sum_128(local_nontoken_workload, sh_world_reduce);
    if (threadIdx.x == 0) a.f[world_global][global_base + kGlobalNonTokenSatWorkloadSteps] += reduced;
    __syncthreads();
    reduced = block_reduce_sum_128(local_nontoken_drop_workload, sh_world_reduce);
    if (threadIdx.x == 0) a.f[world_global][global_base + kGlobalNonTokenSatDropWorkloadSteps] += reduced;
    __syncthreads();
  }
  __syncthreads();
  if (threadIdx.x == 0 && has_f(a, world_global)) {
    if (prefix_workload_known) a.f[world_global][row * kCriticGlobalDim + kGlobalPrefixWorkloadKnown] = 1.0f;
    if (prefix_sat_known) a.f[world_global][row * kCriticGlobalDim + kGlobalSelectedSatLoadKnown] = 1.0f;
    float interference_sum = 0.0f;
    float interference_max = 0.0f;
    for (int u = 0; u < ucount; ++u) {
      const float interference = has_f(a, kFStateLastAccessInterferenceByUav) ? a.f[kFStateLastAccessInterferenceByUav][e * ucount + u] : 0.0f;
      const float value = log1pf(fmaxf(interference / access_noise_full_band, 0.0f));
      interference_sum += value;
      interference_max = fmaxf(interference_max, value);
    }
    a.f[world_global][row * kCriticGlobalDim + kGlobalLastInterferenceMean] = safe_div(interference_sum, fmaxf(static_cast<float>(ucount), 1.0f));
    a.f[world_global][row * kCriticGlobalDim + kGlobalLastInterferenceMax] = interference_max;
    if (prefix_sat_known) {
      float load_sum = 0.0f;
      float load_max = 0.0f;
      float load_count = 0.0f;
      for (int aidx = 0; aidx < active; ++aidx) {
        const int sid = static_cast<int>(a.l[stage_l(stage_slot, kSlActiveSatIds)][e * active + aidx]);
        if (sid < 0 || sid >= sat) continue;
        const float frac = sat_selected_load_frac_for_stage(a, stage_slot, e, sid);
        if (frac > 0.0f) {
          load_sum += frac;
          load_max = fmaxf(load_max, frac);
          load_count += 1.0f;
        }
      }
      a.f[world_global][row * kCriticGlobalDim + kGlobalSelectedSatLoadMean] = safe_div(load_sum, load_count);
      a.f[world_global][row * kCriticGlobalDim + kGlobalSelectedSatLoadMax] = load_max;
    }
    float last_sum = 0.0f;
    float last_max = 0.0f;
    float last_count = 0.0f;
    for (int s = 0; s < sat; ++s) {
      const float frac = has_f(a, kFStateLastSatConnectionCounts)
          ? a.f[kFStateLastSatConnectionCounts][e * sat + s] / fmaxf(static_cast<float>(ucount), 1.0f)
          : 0.0f;
      if (frac > 0.0f) {
        last_sum += frac;
        last_max = fmaxf(last_max, frac);
        last_count += 1.0f;
      }
    }
    a.f[world_global][row * kCriticGlobalDim + kGlobalLastSelectedSatLoadMean] = safe_div(last_sum, last_count);
    a.f[world_global][row * kCriticGlobalDim + kGlobalLastSelectedSatLoadMax] = last_max;
    const int global_base = row * kCriticGlobalDim;
    a.f[world_global][global_base + kGlobalTotalGuQueueSteps] = log1p_nonnegative(a.f[world_global][global_base + kGlobalTotalGuQueueSteps]);
    a.f[world_global][global_base + kGlobalTotalUavQueueSteps] = log1p_nonnegative(a.f[world_global][global_base + kGlobalTotalUavQueueSteps]);
    a.f[world_global][global_base + kGlobalTotalSatQueueSteps] = log1p_nonnegative(a.f[world_global][global_base + kGlobalTotalSatQueueSteps]);
    a.f[world_global][global_base + kGlobalTotalGuDropSteps] = log1p_nonnegative(a.f[world_global][global_base + kGlobalTotalGuDropSteps]);
    a.f[world_global][global_base + kGlobalTotalUavDropSteps] = log1p_nonnegative(a.f[world_global][global_base + kGlobalTotalUavDropSteps]);
    a.f[world_global][global_base + kGlobalTotalSatDropSteps] = log1p_nonnegative(a.f[world_global][global_base + kGlobalTotalSatDropSteps]);
    a.f[world_global][global_base + kGlobalTotalExpectedArrivalSteps] = log1p_nonnegative(a.f[world_global][global_base + kGlobalTotalExpectedArrivalSteps]);
    a.f[world_global][global_base + kGlobalTotalLastGuOutflowSteps] = log1p_nonnegative(a.f[world_global][global_base + kGlobalTotalLastGuOutflowSteps]);
    a.f[world_global][global_base + kGlobalTotalLastUavOutflowSteps] = log1p_nonnegative(a.f[world_global][global_base + kGlobalTotalLastUavOutflowSteps]);
    a.f[world_global][global_base + kGlobalTotalLastSatProcessedSteps] = log1p_nonnegative(a.f[world_global][global_base + kGlobalTotalLastSatProcessedSteps]);
    a.f[world_global][global_base + kGlobalTotalLastWeightedWorkloadSteps] = log1p_nonnegative(a.f[world_global][global_base + kGlobalTotalLastWeightedWorkloadSteps]);
    a.f[world_global][global_base + kGlobalTotalPrefixWeightedWorkloadSteps] = log1p_nonnegative(a.f[world_global][global_base + kGlobalTotalPrefixWeightedWorkloadSteps]);
    a.f[world_global][global_base + kGlobalNonTokenSatQueueSteps] = log1p_nonnegative(a.f[world_global][global_base + kGlobalNonTokenSatQueueSteps]);
    a.f[world_global][global_base + kGlobalNonTokenSatDropSteps] = log1p_nonnegative(a.f[world_global][global_base + kGlobalNonTokenSatDropSteps]);
    a.f[world_global][global_base + kGlobalNonTokenSatLastProcessedSteps] = log1p_nonnegative(a.f[world_global][global_base + kGlobalNonTokenSatLastProcessedSteps]);
    a.f[world_global][global_base + kGlobalNonTokenSatWorkloadSteps] = log1p_nonnegative(a.f[world_global][global_base + kGlobalNonTokenSatWorkloadSteps]);
    a.f[world_global][global_base + kGlobalNonTokenSatDropWorkloadSteps] = log1p_nonnegative(a.f[world_global][global_base + kGlobalNonTokenSatDropWorkloadSteps]);
  }
}

__device__ int nearest_uav_for_state(const PackedAbi& a, int e, int g) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  float best_dist2 = 3.402823466e38f;
  int best_u = 0;
  const float gx = a.f[kFStateGuPos][(e * static_cast<int>(ip(a, kParamNumGu)) + g) * 2 + 0];
  const float gy = a.f[kFStateGuPos][(e * static_cast<int>(ip(a, kParamNumGu)) + g) * 2 + 1];
  for (int u = 0; u < ucount; ++u) {
    const float ux = a.f[kFStateUavPos][(e * ucount + u) * 2 + 0];
    const float uy = a.f[kFStateUavPos][(e * ucount + u) * 2 + 1];
    const float dx = gx - ux;
    const float dy = gy - uy;
    const float dist2 = dx * dx + dy * dy;
    if (dist2 < best_dist2) {
      best_dist2 = dist2;
      best_u = u;
    }
  }
  return best_u;
}

__device__ bool sat_selected_presence_for_u(const PackedAbi& a, int e, int u, int sid) {
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  if (sid < 0 || sid >= sat) return false;
  for (int k = 0; k < select_k; ++k) {
    const int64_t selected = a.l[stage_l(3, kSlSatSelectionMatrix)][(e * static_cast<int>(ip(a, kParamNumUav)) + u) * select_k + k];
    if (selected == sid) return true;
  }
  return false;
}

__device__ bool sat_selected_presence_for_u_stage(const PackedAbi& a, int stage_slot, int e, int u, int sid) {
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  if (sid < 0 || sid >= sat) return false;
  for (int k = 0; k < select_k; ++k) {
    const int64_t selected = a.l[stage_l(stage_slot, kSlSatSelectionMatrix)][(e * static_cast<int>(ip(a, kParamNumUav)) + u) * select_k + k];
    if (selected == sid) return true;
  }
  return false;
}

__device__ float total_backhaul_rate_for_u_cached(const PackedAbi& a, int stage_slot, int e, int u) {
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  float total_rate = 0.0f;
  for (int s = 0; s < sat; ++s) {
    total_rate += has_f(a, kFMainBwLinkRateMatrix)
        ? a.f[kFMainBwLinkRateMatrix][(e * ucount + u) * sat + s]
        : backhaul_rate_for_us(a, stage_slot, e, u, s);
  }
  return total_rate;
}

__device__ float gu_inflow_for_u(const PackedAbi& a, int stage_slot, int e, int u) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  float inflow = 0.0f;
  for (int g = 0; g < gu; ++g) {
    const int64_t assoc = a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g];
    if (assoc == u) inflow += a.f[kFStateLastGuOutflow][e * gu + g];
  }
  return inflow;
}

__device__ float uav_outflow_from_prev_queue(const PackedAbi& a, int stage_slot, int e, int u, float tau) {
  const float inflow = gu_inflow_for_u(a, stage_slot, e, u);
  const float total_rate = total_backhaul_rate_for_u_cached(a, stage_slot, e, u);
  const float q_before = a.f[kFStatePrevUavQueueVec][e * static_cast<int>(ip(a, kParamNumUav)) + u] + inflow;
  const float service_bits = quantize_device(total_rate * tau, fp(a, kFpFlowBitsQuantum, 32.0f));
  return fminf(q_before, service_bits);
}

__device__ float sat_incoming_from_uav_outflow(const PackedAbi& a, int stage_slot, int e, int s, float tau) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  float incoming = 0.0f;
  for (int u = 0; u < ucount; ++u) {
    float total_rate = 0.0f;
    for (int sj = 0; sj < sat; ++sj) {
      total_rate += has_f(a, kFMainBwLinkRateMatrix)
          ? a.f[kFMainBwLinkRateMatrix][(e * ucount + u) * sat + sj]
          : backhaul_rate_for_us(a, stage_slot, e, u, sj);
    }
    const float rate_us = has_f(a, kFMainBwLinkRateMatrix)
        ? a.f[kFMainBwLinkRateMatrix][(e * ucount + u) * sat + s]
        : backhaul_rate_for_us(a, stage_slot, e, u, s);
    if (total_rate > 0.0f && rate_us > 0.0f) {
      incoming += rate_us / total_rate * uav_outflow_from_prev_queue(a, stage_slot, e, u, tau);
    }
  }
  return quantize_device(incoming, fp(a, kFpFlowBitsQuantum, 32.0f));
}

__device__ float reward_metric_quantize(const PackedAbi& a, float value) {
  return quantize_device(value, fp(a, kFpSummaryMetricQuantum, 0.0f));
}

__device__ float sat_cost_current(const PackedAbi& a, int e, int s) {
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const float eps = positive_coeff(fp(a, kFpWorkloadEps, kRelativeLogEps));
  return safe_div(1.0f, fmaxf(a.f[kFStateSatEma][e * sat + s], eps));
}

__device__ float mean_sat_cost_current(const PackedAbi& a, int e) {
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  float total = 0.0f;
  for (int s = 0; s < sat; ++s) total += sat_cost_current(a, e, s);
  return safe_div(total, fmaxf(static_cast<float>(sat), 1.0f));
}

__device__ float uav_cost_current(const PackedAbi& a, int e, int u) {
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const float eps = positive_coeff(fp(a, kFpWorkloadEps, kRelativeLogEps));
  float sat_cost_sum = 0.0f;
  float sat_selected = 0.0f;
  for (int s = 0; s < sat; ++s) {
    if (sat_selected_presence_for_u(a, e, u, s)) {
      sat_cost_sum += sat_cost_current(a, e, s);
      sat_selected += 1.0f;
    }
  }
  const float downstream = sat_selected > 0.0f ? sat_cost_sum / sat_selected : mean_sat_cost_current(a, e);
  return safe_div(1.0f, fmaxf(a.f[kFStateUavEma][e * ucount + u], eps)) + downstream;
}

__device__ float mean_uav_cost_current(const PackedAbi& a, int e) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  float total = 0.0f;
  for (int u = 0; u < ucount; ++u) total += uav_cost_current(a, e, u);
  return safe_div(total, fmaxf(static_cast<float>(ucount), 1.0f));
}

__device__ float gu_cost_current(const PackedAbi& a, int stage_slot, int e, int g) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const float eps = positive_coeff(fp(a, kFpWorkloadEps, kRelativeLogEps));
  const int64_t assoc = a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g];
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const float downstream = (assoc >= 0 && assoc < ucount) ? uav_cost_current(a, e, static_cast<int>(assoc)) : mean_uav_cost_current(a, e);
  return safe_div(1.0f, fmaxf(a.f[kFStateGuEma][e * gu + g], eps)) + downstream;
}

__device__ bool gu_proxy_feature_enabled(const PackedAbi& a, int code) {
  switch (code) {
    case kGuProxyArrivalRate: return ip(a, kParamObsUserIncludeArrivalRate) != 0;
    case kGuProxyRecentArrival: return ip(a, kParamObsUserIncludeRecentArrival) != 0;
    case kGuProxyRecentService: return ip(a, kParamObsUserIncludeRecentService) != 0;
    case kGuProxyQueueHeadroom: return ip(a, kParamObsUserIncludeQueueHeadroom) != 0;
    case kGuProxyLocalGuServiceCost: return ip(a, kParamObsUserIncludeLocalGuServiceCost) != 0;
    case kGuProxyAssocUavCost: return ip(a, kParamObsUserIncludeAssocUavCost) != 0;
    case kGuProxyAssocSatCostMean: return ip(a, kParamObsUserIncludeAssocSatCostMean) != 0;
    case kGuProxyWeightedQueueCost: return ip(a, kParamObsUserIncludeWeightedQueueCost) != 0;
    case kGuProxyWeightedQueueCostRelative: return ip(a, kParamObsUserIncludeWeightedQueueCostRelative) != 0;
    case kGuProxyUrgencyRisk: return ip(a, kParamObsUserIncludeUrgencyRisk) != 0;
    case kGuProxyDownstreamPressure: return ip(a, kParamObsUserIncludeDownstreamPressure) != 0;
    case kGuProxyServiceGap: return ip(a, kParamObsUserIncludeServiceGap) != 0;
    case kGuProxyServiceGapRisk: return ip(a, kParamObsUserIncludeServiceGapRisk) != 0;
    case kGuProxyDeadlineSlack: return ip(a, kParamObsUserIncludeDeadlineSlack) != 0;
    case kGuProxyDeadlineRisk: return ip(a, kParamObsUserIncludeDeadlineRisk) != 0;
    default: return false;
  }
}

__device__ int gu_proxy_feature_code_for_column(const PackedAbi& a, int p) {
  int col = 0;
  for (int code = kGuProxyArrivalRate; code <= kGuProxyDeadlineRisk; ++code) {
    if (!gu_proxy_feature_enabled(a, code)) continue;
    if (col == p) return code;
    ++col;
  }
  return -1;
}

__device__ float current_expected_arrival_ref_per_gu_bits(const PackedAbi& a, int e) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int obs_slot = has_i(a, kIRandomStepTensor) ? a.i[kIRandomStepTensor][0] : 0;
  float total_rate = 0.0f;
  for (int h = 0; h < gu; ++h) total_rate += state_arrival_rate(a, e, h, obs_slot);
  const float mean_rate = safe_div(total_rate, fmaxf(static_cast<float>(gu), 1.0f));
  return require_positive_reward_ref(mean_rate * fp(a, kFpTau0, 1.0f));
}

__device__ float local_gu_service_cost_raw_current(const PackedAbi& a, int e, int g) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const float eps = positive_coeff(fp(a, kFpWorkloadEps, kRelativeLogEps));
  return safe_div(1.0f, fmaxf(a.f[kFStateGuEma][e * gu + g], eps));
}

__device__ float mean_local_gu_service_cost_current(const PackedAbi& a, int e) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  float total = 0.0f;
  for (int h = 0; h < gu; ++h) total += local_gu_service_cost_raw_current(a, e, h);
  return positive_coeff(safe_div(total, fmaxf(static_cast<float>(gu), 1.0f)));
}

__device__ float assoc_uav_cost_raw_current(const PackedAbi& a, int stage_slot, int e, int g) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int64_t assoc = a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g];
  return (assoc >= 0 && assoc < ucount) ? uav_cost_current(a, e, static_cast<int>(assoc)) : mean_uav_cost_current(a, e);
}

__device__ float mean_assoc_uav_cost_current(const PackedAbi& a, int stage_slot, int e) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  float total = 0.0f;
  for (int h = 0; h < gu; ++h) total += assoc_uav_cost_raw_current(a, stage_slot, e, h);
  return positive_coeff(safe_div(total, fmaxf(static_cast<float>(gu), 1.0f)));
}

__device__ float uav_downstream_cost_current(const PackedAbi& a, int e, int u) {
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  float sat_cost_sum = 0.0f;
  float sat_selected = 0.0f;
  for (int s = 0; s < sat; ++s) {
    if (sat_selected_presence_for_u(a, e, u, s)) {
      sat_cost_sum += sat_cost_current(a, e, s);
      sat_selected += 1.0f;
    }
  }
  return sat_selected > 0.0f ? sat_cost_sum / sat_selected : mean_sat_cost_current(a, e);
}

__device__ float assoc_sat_cost_raw_current(const PackedAbi& a, int stage_slot, int e, int g) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int64_t assoc = a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g];
  return (assoc >= 0 && assoc < ucount) ? uav_downstream_cost_current(a, e, static_cast<int>(assoc)) : mean_sat_cost_current(a, e);
}

__device__ float mean_assoc_sat_cost_current(const PackedAbi& a, int stage_slot, int e) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  float total = 0.0f;
  for (int h = 0; h < gu; ++h) total += assoc_sat_cost_raw_current(a, stage_slot, e, h);
  return positive_coeff(safe_div(total, fmaxf(static_cast<float>(gu), 1.0f)));
}

__device__ float weighted_queue_cost_raw_current(const PackedAbi& a, int stage_slot, int e, int g) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const float q = a.f[stage_f(stage_slot, kSfGuQueue)][e * gu + g];
  return gu_cost_current(a, stage_slot, e, g) * q;
}

__device__ float mean_weighted_queue_cost_current(const PackedAbi& a, int stage_slot, int e) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  float total = 0.0f;
  for (int h = 0; h < gu; ++h) total += weighted_queue_cost_raw_current(a, stage_slot, e, h);
  return positive_config_scale(safe_div(total, fmaxf(static_cast<float>(gu), 1.0f)));
}

__device__ float weighted_queue_feature_ref_current(const PackedAbi& a, int e) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const float eps = positive_coeff(fp(a, kFpWorkloadEps, kRelativeLogEps));
  const float arrival_ref = require_positive_reward_ref(
      has_f(a, kFStateArrivalRef) ? a.f[kFStateArrivalRef][e] : current_expected_arrival_ref_per_gu_bits(a, e) * fmaxf(static_cast<float>(gu), 1.0f));
  const float sat_active_ref = positive_config_scale(fp(a, kFpBwWorkloadSatActiveRef, fmaxf(static_cast<float>(sat), 1.0f)));
  const float gu_default = arrival_ref / fmaxf(static_cast<float>(gu), 1.0f);
  const float uav_default = arrival_ref / fmaxf(static_cast<float>(ucount), 1.0f);
  const float sat_default = arrival_ref / sat_active_ref;
  const float gu_local_cost_ref = safe_div(1.0f, fmaxf(gu_default, eps));
  const float uav_cost_ref = safe_div(1.0f, fmaxf(uav_default, eps)) + safe_div(1.0f, fmaxf(sat_default, eps));
  return require_positive_reward_ref(arrival_ref * (gu_local_cost_ref + uav_cost_ref));
}

__device__ __forceinline__ float* refresh_cache_sat_cost_ptr(const PackedAbi& a, float* cache) {
  (void)a;
  return cache;
}

__device__ __forceinline__ float* refresh_cache_uav_downstream_ptr(const PackedAbi& a, float* cache) {
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  return cache + sat;
}

__device__ __forceinline__ float* refresh_cache_uav_cost_ptr(const PackedAbi& a, float* cache) {
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  return cache + sat + ucount;
}

__device__ __forceinline__ float* refresh_cache_local_gu_cost_ptr(const PackedAbi& a, float* cache) {
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  return cache + sat + 2 * ucount;
}

__device__ __forceinline__ float* refresh_cache_assoc_uav_cost_ptr(const PackedAbi& a, float* cache) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  return refresh_cache_local_gu_cost_ptr(a, cache) + gu;
}

__device__ __forceinline__ float* refresh_cache_assoc_sat_cost_ptr(const PackedAbi& a, float* cache) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  return refresh_cache_assoc_uav_cost_ptr(a, cache) + gu;
}

__device__ __forceinline__ float* refresh_cache_gu_cost_ptr(const PackedAbi& a, float* cache) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  return refresh_cache_assoc_sat_cost_ptr(a, cache) + gu;
}

__device__ __forceinline__ float* refresh_cache_weighted_queue_cost_ptr(const PackedAbi& a, float* cache) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  return refresh_cache_gu_cost_ptr(a, cache) + gu;
}

__device__ __forceinline__ float* refresh_cache_scalar_ptr(const PackedAbi& a, float* cache) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  return refresh_cache_weighted_queue_cost_ptr(a, cache) + gu;
}

__device__ float refresh_cached_sat_cost(const PackedAbi& a, float* cache, int s) {
  return refresh_cache_sat_cost_ptr(a, cache)[s];
}

__device__ float refresh_cached_uav_cost(const PackedAbi& a, float* cache, int u) {
  return refresh_cache_uav_cost_ptr(a, cache)[u];
}

__device__ float refresh_cached_gu_cost(const PackedAbi& a, float* cache, int g) {
  return refresh_cache_gu_cost_ptr(a, cache)[g];
}

__device__ void refresh_cost_cache_fill_base_parallel(const PackedAbi& a, int e, float* cache) {
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const float eps = positive_coeff(fp(a, kFpWorkloadEps, kRelativeLogEps));
  float* sat_cost = refresh_cache_sat_cost_ptr(a, cache);
  float* uav_downstream = refresh_cache_uav_downstream_ptr(a, cache);
  float* uav_cost = refresh_cache_uav_cost_ptr(a, cache);
  float* scalars = refresh_cache_scalar_ptr(a, cache);
  __shared__ float sh_reduce_cache[128];
  __shared__ float sh_mean_sat_cost_cache;

  float sat_sum_local = 0.0f;
  for (int s = threadIdx.x; s < sat; s += blockDim.x) {
    const float cost = sat_cost_current(a, e, s);
    sat_cost[s] = cost;
    sat_sum_local += cost;
  }
  const float sat_sum = block_reduce_sum_128(sat_sum_local, sh_reduce_cache);
  if (threadIdx.x == 0) {
    const float mean_sat = safe_div(sat_sum, fmaxf(static_cast<float>(sat), 1.0f));
    sh_mean_sat_cost_cache = mean_sat;
    scalars[kRefreshCostScalarMeanSat] = mean_sat;
    scalars[kRefreshCostScalarBaseArrival] = current_expected_arrival_ref_per_gu_bits(a, e);
    scalars[kRefreshCostScalarWeightedQueueRef] = weighted_queue_feature_ref_current(a, e);
    scalars[kRefreshCostScalarReserved] = 0.0f;
  }
  __syncthreads();

  float uav_sum_local = 0.0f;
  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    float sat_cost_sum = 0.0f;
    float sat_selected = 0.0f;
    for (int s = 0; s < sat; ++s) {
      if (sat_selected_presence_for_u(a, e, u, s)) {
        sat_cost_sum += sat_cost[s];
        sat_selected += 1.0f;
      }
    }
    const float downstream = sat_selected > 0.0f ? sat_cost_sum / sat_selected : sh_mean_sat_cost_cache;
    const float cost = safe_div(1.0f, fmaxf(a.f[kFStateUavEma][e * ucount + u], eps)) + downstream;
    uav_downstream[u] = downstream;
    uav_cost[u] = cost;
    uav_sum_local += cost;
  }
  const float uav_sum = block_reduce_sum_128(uav_sum_local, sh_reduce_cache);
  if (threadIdx.x == 0) {
    scalars[kRefreshCostScalarMeanUav] = safe_div(uav_sum, fmaxf(static_cast<float>(ucount), 1.0f));
  }
  __syncthreads();
}

__device__ void refresh_cost_cache_fill_gu_parallel(const PackedAbi& a, int stage_slot, int e, float* cache) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const float eps = positive_coeff(fp(a, kFpWorkloadEps, kRelativeLogEps));
  float* uav_downstream = refresh_cache_uav_downstream_ptr(a, cache);
  float* uav_cost = refresh_cache_uav_cost_ptr(a, cache);
  float* local_gu_cost = refresh_cache_local_gu_cost_ptr(a, cache);
  float* assoc_uav_cost = refresh_cache_assoc_uav_cost_ptr(a, cache);
  float* assoc_sat_cost = refresh_cache_assoc_sat_cost_ptr(a, cache);
  float* gu_cost = refresh_cache_gu_cost_ptr(a, cache);
  float* weighted_queue_cost = refresh_cache_weighted_queue_cost_ptr(a, cache);
  float* scalars = refresh_cache_scalar_ptr(a, cache);
  const float mean_sat = scalars[kRefreshCostScalarMeanSat];
  const float mean_uav = scalars[kRefreshCostScalarMeanUav];
  __shared__ float sh_reduce_cache[128];

  float local_sum = 0.0f;
  float assoc_uav_sum = 0.0f;
  float assoc_sat_sum = 0.0f;
  float weighted_sum = 0.0f;
  for (int g = threadIdx.x; g < gu; g += blockDim.x) {
    const int64_t assoc = a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g];
    const float local = safe_div(1.0f, fmaxf(a.f[kFStateGuEma][e * gu + g], eps));
    const float assoc_uav = (assoc >= 0 && assoc < ucount) ? uav_cost[static_cast<int>(assoc)] : mean_uav;
    const float assoc_sat = (assoc >= 0 && assoc < ucount) ? uav_downstream[static_cast<int>(assoc)] : mean_sat;
    const float total_cost = local + assoc_uav;
    const float q = a.f[stage_f(stage_slot, kSfGuQueue)][e * gu + g];
    const float weighted = total_cost * q;
    local_gu_cost[g] = local;
    assoc_uav_cost[g] = assoc_uav;
    assoc_sat_cost[g] = assoc_sat;
    gu_cost[g] = total_cost;
    weighted_queue_cost[g] = weighted;
    local_sum += local;
    assoc_uav_sum += assoc_uav;
    assoc_sat_sum += assoc_sat;
    weighted_sum += weighted;
  }
  const float local_total = block_reduce_sum_128(local_sum, sh_reduce_cache);
  if (threadIdx.x == 0) {
    scalars[kRefreshCostScalarMeanLocalGu] =
        positive_coeff(safe_div(local_total, fmaxf(static_cast<float>(gu), 1.0f)));
  }
  __syncthreads();
  const float assoc_uav_total = block_reduce_sum_128(assoc_uav_sum, sh_reduce_cache);
  if (threadIdx.x == 0) {
    scalars[kRefreshCostScalarMeanAssocUav] =
        positive_coeff(safe_div(assoc_uav_total, fmaxf(static_cast<float>(gu), 1.0f)));
  }
  __syncthreads();
  const float assoc_sat_total = block_reduce_sum_128(assoc_sat_sum, sh_reduce_cache);
  if (threadIdx.x == 0) {
    scalars[kRefreshCostScalarMeanAssocSat] =
        positive_coeff(safe_div(assoc_sat_total, fmaxf(static_cast<float>(gu), 1.0f)));
  }
  __syncthreads();
  const float weighted_total = block_reduce_sum_128(weighted_sum, sh_reduce_cache);
  if (threadIdx.x == 0) {
    scalars[kRefreshCostScalarMeanWeightedQueue] =
        positive_config_scale(safe_div(weighted_total, fmaxf(static_cast<float>(gu), 1.0f)));
  }
  __syncthreads();
}

__device__ float gu_proxy_feature_value(const PackedAbi& a, int stage_slot, int e, int g, int p) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int code = gu_proxy_feature_code_for_column(a, p);
  const int obs_slot = has_i(a, kIRandomStepTensor) ? a.i[kIRandomStepTensor][0] : 0;
  switch (code) {
    case kGuProxyArrivalRate: {
      const float base_arrival = current_expected_arrival_ref_per_gu_bits(a, e);
      return state_arrival_rate(a, e, g, obs_slot) / base_arrival;
    }
    case kGuProxyRecentArrival: {
      const float base_arrival = current_expected_arrival_ref_per_gu_bits(a, e);
      return has_f(a, kFStateLastGuArrival) ? a.f[kFStateLastGuArrival][e * gu + g] / base_arrival : 0.0f;
    }
    case kGuProxyRecentService: {
      const float base_arrival = current_expected_arrival_ref_per_gu_bits(a, e);
      return has_f(a, kFStateLastGuOutflow) ? a.f[kFStateLastGuOutflow][e * gu + g] / base_arrival : 0.0f;
    }
    case kGuProxyQueueHeadroom:
      return 1.0f - safe_div(a.f[stage_f(stage_slot, kSfGuQueue)][e * gu + g], positive_config_scale(fp(a, kFpQueueMaxGu, 1.0f)));
    case kGuProxyLocalGuServiceCost:
      return logf(positive_coeff(local_gu_service_cost_raw_current(a, e, g)) / mean_local_gu_service_cost_current(a, e));
    case kGuProxyAssocUavCost:
      return logf(positive_coeff(assoc_uav_cost_raw_current(a, stage_slot, e, g)) / mean_assoc_uav_cost_current(a, stage_slot, e));
    case kGuProxyAssocSatCostMean:
      return logf(positive_coeff(assoc_sat_cost_raw_current(a, stage_slot, e, g)) / mean_assoc_sat_cost_current(a, stage_slot, e));
    case kGuProxyWeightedQueueCost:
      return log1pf(fmaxf(weighted_queue_cost_raw_current(a, stage_slot, e, g), 0.0f) / weighted_queue_feature_ref_current(a, e));
    case kGuProxyWeightedQueueCostRelative:
      return logf(log_argument(weighted_queue_cost_raw_current(a, stage_slot, e, g) / mean_weighted_queue_cost_current(a, stage_slot, e)));
    case kGuProxyUrgencyRisk:
      return has_f(a, kFStateUrgencyRisk) ? a.f[kFStateUrgencyRisk][e * gu + g] : 0.0f;
    case kGuProxyDownstreamPressure:
      return has_f(a, kFStateDownstreamPressure) ? a.f[kFStateDownstreamPressure][e * gu + g] : 0.0f;
    case kGuProxyServiceGap:
      return has_f(a, kFStateServiceGap) ? safe_div(a.f[kFStateServiceGap][e * gu + g], fp(a, kFpServiceGapCapSteps, 1.0f)) : 0.0f;
    case kGuProxyServiceGapRisk:
      return has_f(a, kFStateServiceGapRisk) ? a.f[kFStateServiceGapRisk][e * gu + g] : 0.0f;
    case kGuProxyDeadlineSlack: {
      const float deadline_steps = has_f(a, kFStateGuDeadlineSteps) ? fmaxf(a.f[kFStateGuDeadlineSteps][e * gu + g], 1.0e-6f) : 1.0f;
      const float slack = has_f(a, kFStateDeadlineSlack) ? a.f[kFStateDeadlineSlack][e * gu + g] : 0.0f;
      return clampf_device(slack / deadline_steps, -1.0f, 1.0f);
    }
    case kGuProxyDeadlineRisk:
      return has_f(a, kFStateDeadlineRisk) ? clampf_device(a.f[kFStateDeadlineRisk][e * gu + g], 0.0f, 2.0f) : 0.0f;
    default:
      return 0.0f;
  }
}

__device__ float gu_proxy_feature_value_cached(const PackedAbi& a, int stage_slot, int e, int g, int p, float* cache) {
  if (cache == nullptr) return gu_proxy_feature_value(a, stage_slot, e, g, p);
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int code = gu_proxy_feature_code_for_column(a, p);
  const int obs_slot = has_i(a, kIRandomStepTensor) ? a.i[kIRandomStepTensor][0] : 0;
  float* local_gu_cost = refresh_cache_local_gu_cost_ptr(a, cache);
  float* assoc_uav_cost = refresh_cache_assoc_uav_cost_ptr(a, cache);
  float* assoc_sat_cost = refresh_cache_assoc_sat_cost_ptr(a, cache);
  float* weighted_queue_cost = refresh_cache_weighted_queue_cost_ptr(a, cache);
  float* scalars = refresh_cache_scalar_ptr(a, cache);
  const float base_arrival = scalars[kRefreshCostScalarBaseArrival];
  switch (code) {
    case kGuProxyArrivalRate:
      return state_arrival_rate(a, e, g, obs_slot) / base_arrival;
    case kGuProxyRecentArrival:
      return has_f(a, kFStateLastGuArrival) ? a.f[kFStateLastGuArrival][e * gu + g] / base_arrival : 0.0f;
    case kGuProxyRecentService:
      return has_f(a, kFStateLastGuOutflow) ? a.f[kFStateLastGuOutflow][e * gu + g] / base_arrival : 0.0f;
    case kGuProxyQueueHeadroom:
      return 1.0f - safe_div(a.f[stage_f(stage_slot, kSfGuQueue)][e * gu + g], positive_config_scale(fp(a, kFpQueueMaxGu, 1.0f)));
    case kGuProxyLocalGuServiceCost:
      return logf(positive_coeff(local_gu_cost[g]) / scalars[kRefreshCostScalarMeanLocalGu]);
    case kGuProxyAssocUavCost:
      return logf(positive_coeff(assoc_uav_cost[g]) / scalars[kRefreshCostScalarMeanAssocUav]);
    case kGuProxyAssocSatCostMean:
      return logf(positive_coeff(assoc_sat_cost[g]) / scalars[kRefreshCostScalarMeanAssocSat]);
    case kGuProxyWeightedQueueCost:
      return log1pf(fmaxf(weighted_queue_cost[g], 0.0f) / scalars[kRefreshCostScalarWeightedQueueRef]);
    case kGuProxyWeightedQueueCostRelative:
      return logf(log_argument(weighted_queue_cost[g] / scalars[kRefreshCostScalarMeanWeightedQueue]));
    case kGuProxyUrgencyRisk:
      return has_f(a, kFStateUrgencyRisk) ? a.f[kFStateUrgencyRisk][e * gu + g] : 0.0f;
    case kGuProxyDownstreamPressure:
      return has_f(a, kFStateDownstreamPressure) ? a.f[kFStateDownstreamPressure][e * gu + g] : 0.0f;
    case kGuProxyServiceGap:
      return has_f(a, kFStateServiceGap) ? safe_div(a.f[kFStateServiceGap][e * gu + g], fp(a, kFpServiceGapCapSteps, 1.0f)) : 0.0f;
    case kGuProxyServiceGapRisk:
      return has_f(a, kFStateServiceGapRisk) ? a.f[kFStateServiceGapRisk][e * gu + g] : 0.0f;
    case kGuProxyDeadlineSlack: {
      const float deadline_steps = has_f(a, kFStateGuDeadlineSteps) ? fmaxf(a.f[kFStateGuDeadlineSteps][e * gu + g], 1.0e-6f) : 1.0f;
      const float slack = has_f(a, kFStateDeadlineSlack) ? a.f[kFStateDeadlineSlack][e * gu + g] : 0.0f;
      return clampf_device(slack / deadline_steps, -1.0f, 1.0f);
    }
    case kGuProxyDeadlineRisk:
      return has_f(a, kFStateDeadlineRisk) ? clampf_device(a.f[kFStateDeadlineRisk][e * gu + g], 0.0f, 2.0f) : 0.0f;
    default:
      return 0.0f;
  }
}

__device__ float sat_overlap_eval_current(const PackedAbi& a, int e) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  if (ucount <= 1 || sat <= 0) return 0.0f;
  const float denom = fmaxf(static_cast<float>(ucount - 1), 1.0f);
  float overlap_u_sum = 0.0f;
  for (int u = 0; u < ucount; ++u) {
    float selected_count = 0.0f;
    float overlap_sum = 0.0f;
    for (int s = 0; s < sat; ++s) {
      if (!sat_selected_presence_for_u(a, e, u, s)) continue;
      selected_count += 1.0f;
      overlap_sum += fmaxf(static_cast<float>(sat_selected_count(a, e, s)) - 1.0f, 0.0f);
    }
    if (selected_count > 0.0f) overlap_u_sum += overlap_sum / (selected_count * denom);
  }
  return overlap_u_sum / fmaxf(static_cast<float>(ucount), 1.0f);
}

__device__ bool flow_proxy_slot_valid(const PackedAbi& a, int stage_slot, int e, int u, int c) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  if (c < 0 || c >= gu) return false;
  return bw_gu_assoc_match(a, stage_slot, e, u, c);
}

__device__ float flow_proxy_base_action_value(const PackedAbi& a, int e, int u, int c, int bw_mode) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int idx = (e * static_cast<int>(ip(a, kParamNumUav)) + u) * gu + c;
  if (bw_mode == kSourceZero) return 0.0f;
  if (!ip(a, kParamEnableBwAction)) return 1.0f;
  if (ip(a, kParamFlowBaseActionMode) == 1) {
    return a.f[kFLiveBwRefAction][idx];
  }
  if (ip(a, kParamFlowBaseActionMode) == 2) {
    return a.f[kFLiveBwFlowProxyOverrideAction][idx];
  }
  return a.f[kFLiveBwAction][idx];
}

__device__ void flow_proxy_target_stats(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int target_u,
    int target_c,
    int bw_mode,
    float* used_delta,
    float* scale,
    float* cf_norm,
    bool* active) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const float eps = positive_coeff(fp(a, kFpFlowProxyEps, kRelativeLogEps));
  const float delta = fmaxf(fp(a, kFpFlowProxyAuxDelta, 0.05f), 0.0f);
  float donor_mass = 0.0f;
  int valid_count = 0;
  for (int c = 0; c < gu; ++c) {
    if (!flow_proxy_slot_valid(a, stage_slot, e, target_u, c)) continue;
    ++valid_count;
    if (c != target_c) donor_mass += flow_proxy_base_action_value(a, e, target_u, c, bw_mode);
  }
  const bool target_valid = flow_proxy_slot_valid(a, stage_slot, e, target_u, target_c);
  float delta_used = fminf(delta, donor_mass * 0.5f);
  const bool is_active = target_valid && valid_count > 1 && donor_mass > eps && delta_used > eps;
  if (!is_active) {
    *used_delta = 0.0f;
    *scale = 1.0f;
    *cf_norm = 1.0f;
    *active = false;
    return;
  }
  const float scale_v = (donor_mass - delta_used) / fmaxf(donor_mass, eps);
  float norm = 0.0f;
  for (int c = 0; c < gu; ++c) {
    if (!flow_proxy_slot_valid(a, stage_slot, e, target_u, c)) continue;
    float v = flow_proxy_base_action_value(a, e, target_u, c, bw_mode);
    if (c == target_c) v += delta_used;
    else v *= scale_v;
    norm += v;
  }
  *used_delta = delta_used;
  *scale = scale_v;
  *cf_norm = fmaxf(norm, eps);
  *active = true;
}

__device__ float flow_proxy_action_value(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    int c,
    int bw_mode,
    bool counterfactual,
    int target_u,
    int target_c,
    float used_delta,
    float scale,
    float cf_norm) {
  if (!flow_proxy_slot_valid(a, stage_slot, e, u, c)) return 0.0f;
  const float base = flow_proxy_base_action_value(a, e, u, c, bw_mode);
  if (!counterfactual || u != target_u) return base;
  float v = 0.0f;
  if (c == target_c) v = base + used_delta;
  else v = base * scale;
  const float eps = positive_coeff(fp(a, kFpFlowProxyEps, kRelativeLogEps));
  return fabsf(cf_norm) > eps ? v / cf_norm : 0.0f;
}

__device__ float flow_proxy_beta_for_slot(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    int c,
    int bw_mode,
    bool counterfactual,
    int target_u,
    int target_c,
    float used_delta,
    float scale,
    float cf_norm) {
  if (!bw_gu_assoc_match(a, stage_slot, e, u, c)) return 0.0f;
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  int assoc_count = 0;
  float denom = 0.0f;
  for (int j = 0; j < gu; ++j) {
    if (!bw_gu_assoc_match(a, stage_slot, e, u, j)) continue;
    ++assoc_count;
    denom += flow_proxy_action_value(
        a, stage_slot, e, u, j, bw_mode, counterfactual, target_u, target_c, used_delta, scale, cf_norm);
  }
  if (assoc_count <= 0) return 0.0f;
  if (ip(a, kParamEnableBwAction) && denom > kNormDenomEps) {
    return flow_proxy_action_value(
        a, stage_slot, e, u, c, bw_mode, counterfactual, target_u, target_c, used_delta, scale, cf_norm) /
        denom;
  }
  return 1.0f / static_cast<float>(assoc_count);
}

__device__ float flow_proxy_band_fraction_for_gu(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int g,
    int bw_mode,
    bool counterfactual,
    int target_u,
    int target_c,
    float used_delta,
    float scale,
    float cf_norm) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  if (g < 0 || g >= gu) return 0.0f;
  const int64_t assoc64 = a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g];
  if (assoc64 < 0 || assoc64 >= ucount) return 0.0f;
  const int u = static_cast<int>(assoc64);
  return flow_proxy_beta_for_slot(
      a, stage_slot, e, u, g, bw_mode, counterfactual, target_u, target_c, used_delta, scale, cf_norm);
}

__device__ float flow_proxy_access_interference_for_u(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    int bw_mode,
    bool counterfactual,
    int target_u,
    int target_c,
    float used_delta,
    float scale,
    float cf_norm) {
  if (!ip(a, kParamInterferenceEnabled)) return 0.0f;
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const float p = fp(a, kFpAccessGuTxPower, fp(a, kFpLocalUavTxPower, 1.0f));
  float total_received = 0.0f;
  float same_cell = 0.0f;
  for (int g = 0; g < gu; ++g) {
    const float band = flow_proxy_band_fraction_for_gu(
        a, stage_slot, e, g, bw_mode, counterfactual, target_u, target_c, used_delta, scale, cf_norm);
    const int64_t assoc64 = a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g];
    if (assoc64 < 0 || assoc64 >= ucount || band <= 0.0f) continue;
    const float gain = a.f[stage_f(stage_slot, kSfAccessGainMatrix)][(e * gu + g) * ucount + u];
    const float received = p * gain * band;
    total_received += received;
    if (assoc64 == u) same_cell += received;
  }
  return fmaxf(quantize_device(total_received - same_cell, fp(a, kFpAccessInterferenceQuantum, 0.0f)), 0.0f);
}

__device__ float flow_proxy_access_rate_for_slot(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    int c,
    int bw_mode,
    bool counterfactual,
    int target_u,
    int target_c,
    float used_delta,
    float scale,
    float cf_norm) {
  const int gid = c;
  if (!bw_gu_assoc_match(a, stage_slot, e, u, gid)) return 0.0f;
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const float beta = flow_proxy_beta_for_slot(
      a, stage_slot, e, u, c, bw_mode, counterfactual, target_u, target_c, used_delta, scale, cf_norm);
  if (beta <= 0.0f) return 0.0f;
  const float gain = a.f[stage_f(stage_slot, kSfAccessGainMatrix)][(e * gu + gid) * ucount + u];
  const float interference = flow_proxy_access_interference_for_u(
      a, stage_slot, e, u, bw_mode, counterfactual, target_u, target_c, used_delta, scale, cf_norm);
  const float eff_bw = beta * fp(a, kFpAccessBAcc, 1.0f);
  if (eff_bw <= 0.0f) return 0.0f;
  const float eff_interference = ip(a, kParamInterferenceEnabled) ? beta * interference : 0.0f;
  const float denom = fp(a, kFpAccessNoiseDensity, fp(a, kFpLocalNoiseDensity, kDefaultNoiseDensity)) *
      fp(a, kFpAccessNoiseFigureLinear, 1.0f) * eff_bw + eff_interference;
  if (denom <= 0.0f) return 0.0f;
  const float snr = fp(a, kFpAccessGuTxPower, fp(a, kFpLocalUavTxPower, 1.0f)) * gain / denom;
  const float se = quantize_device(access_spectral_efficiency_device(a, snr), fp(a, kFpAccessEtaQuantum, kDefaultRateQuantum));
  return quantize_device(eff_bw * se, fp(a, kFpAccessRateQuantum, 32.0f));
}

__device__ float flow_proxy_gu_outflow_for_g(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int g,
    int bw_mode,
    bool counterfactual,
    int target_u,
    int target_c,
    float used_delta,
    float scale,
    float cf_norm) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const float tau = fp(a, kFpBwTau0, fp(a, kFpTau0, 1.0f));
  float rate_sum = 0.0f;
  for (int u = 0; u < ucount; ++u) {
    rate_sum += flow_proxy_access_rate_for_slot(
        a, stage_slot, e, u, g, bw_mode, counterfactual, target_u, target_c, used_delta, scale, cf_norm);
  }
  const float arrival = a.f[kFStateLastGuArrival][e * static_cast<int>(ip(a, kParamNumGu)) + g];
  const float demand = a.f[stage_f(stage_slot, kSfGuQueue)][e * static_cast<int>(ip(a, kParamNumGu)) + g] + arrival;
  const float service_bits = quantize_device(fmaxf(rate_sum, 0.0f) * tau, fp(a, kFpFlowBitsQuantum, 32.0f));
  return fminf(demand, service_bits);
}

__device__ float flow_proxy_uav_inflow_for_u(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    int bw_mode,
    bool counterfactual,
    int target_u,
    int target_c,
    float used_delta,
    float scale,
    float cf_norm) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  float inflow = 0.0f;
  for (int g = 0; g < gu; ++g) {
    const int64_t assoc = a.l[stage_l(stage_slot, kSlAssoc)][e * gu + g];
    if (assoc == u) {
      inflow += flow_proxy_gu_outflow_for_g(
          a, stage_slot, e, g, bw_mode, counterfactual, target_u, target_c, used_delta, scale, cf_norm);
    }
  }
  return inflow;
}

__device__ float flow_proxy_total_backhaul_rate_for_u(const PackedAbi& a, int stage_slot, int e, int u) {
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  float total = 0.0f;
  for (int s = 0; s < sat; ++s) total += backhaul_rate_for_us(a, stage_slot, e, u, s);
  return total;
}

__device__ float flow_proxy_uav_outflow_for_u(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    int bw_mode,
    bool counterfactual,
    int target_u,
    int target_c,
    float used_delta,
    float scale,
    float cf_norm) {
  const float tau = fp(a, kFpBwTau0, fp(a, kFpTau0, 1.0f));
  const float q_before = a.f[stage_f(stage_slot, kSfUavQueue)][e * static_cast<int>(ip(a, kParamNumUav)) + u] +
      flow_proxy_uav_inflow_for_u(a, stage_slot, e, u, bw_mode, counterfactual, target_u, target_c, used_delta, scale, cf_norm);
  const float service_bits = quantize_device(flow_proxy_total_backhaul_rate_for_u(a, stage_slot, e, u) * tau, fp(a, kFpFlowBitsQuantum, 32.0f));
  return fminf(q_before, service_bits);
}

__device__ float flow_proxy_sat_incoming_for_s(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int s,
    int bw_mode,
    bool counterfactual,
    int target_u,
    int target_c,
    float used_delta,
    float scale,
    float cf_norm) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  float incoming = 0.0f;
  for (int u = 0; u < ucount; ++u) {
    const float total_rate = flow_proxy_total_backhaul_rate_for_u(a, stage_slot, e, u);
    const float rate_us = backhaul_rate_for_us(a, stage_slot, e, u, s);
    if (total_rate > 0.0f && rate_us > 0.0f) {
      incoming += rate_us / total_rate *
          flow_proxy_uav_outflow_for_u(a, stage_slot, e, u, bw_mode, counterfactual, target_u, target_c, used_delta, scale, cf_norm);
    }
  }
  return incoming;
}

__device__ float flow_proxy_reward(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int bw_mode,
    bool counterfactual,
    int target_u,
    int target_c,
    float used_delta,
    float scale,
    float cf_norm) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const float tau = fp(a, kFpBwTau0, fp(a, kFpTau0, 1.0f));
  const float qmax_gu = positive_config_scale(fp(a, kFpQueueMaxGu, 1.0f));
  const float qmax_uav = positive_config_scale(fp(a, kFpQueueMaxUav, 1.0f));
  const float qmax_sat = positive_config_scale(fp(a, kFpQueueMaxSat, 1.0f));
  float arrival_sum = 0.0f;
  float gu_out_sum = 0.0f;
  float gu_drop_sum = 0.0f;
  float q_gu_sum = 0.0f;
  float workload_before = 0.0f;
  float workload_after = 0.0f;
  float workload_drop = 0.0f;
  for (int g = 0; g < gu; ++g) {
    const float arrival = a.f[kFStateLastGuArrival][e * gu + g];
    const float before = a.f[stage_f(stage_slot, kSfGuQueue)][e * gu + g] + arrival;
    const float served = flow_proxy_gu_outflow_for_g(
        a, stage_slot, e, g, bw_mode, counterfactual, target_u, target_c, used_delta, scale, cf_norm);
    float after_raw = fmaxf(before - served, 0.0f);
    const float dropped = fmaxf(after_raw - qmax_gu, 0.0f);
    const float after = fminf(after_raw, qmax_gu);
    const float cost = gu_cost_current(a, stage_slot, e, g);
    arrival_sum += arrival;
    gu_out_sum += served;
    gu_drop_sum += dropped;
    q_gu_sum += after;
    workload_before += cost * before;
    workload_after += cost * after;
    workload_drop += cost * dropped;
  }

  float uav_out_sum = 0.0f;
  float uav_drop_sum = 0.0f;
  float q_uav_sum = 0.0f;
  for (int u = 0; u < ucount; ++u) {
    const float inflow = flow_proxy_uav_inflow_for_u(
        a, stage_slot, e, u, bw_mode, counterfactual, target_u, target_c, used_delta, scale, cf_norm);
    const float before = a.f[stage_f(stage_slot, kSfUavQueue)][e * ucount + u] + inflow;
    const float service_bits = quantize_device(flow_proxy_total_backhaul_rate_for_u(a, stage_slot, e, u) * tau, fp(a, kFpFlowBitsQuantum, 32.0f));
    const float outflow = fminf(before, service_bits);
    float after_raw = fmaxf(before - outflow, 0.0f);
    const float dropped = fmaxf(after_raw - qmax_uav, 0.0f);
    const float after = fminf(after_raw, qmax_uav);
    const float cost = uav_cost_current(a, e, u);
    uav_out_sum += outflow;
    uav_drop_sum += dropped;
    q_uav_sum += after;
    workload_before += cost * a.f[stage_f(stage_slot, kSfUavQueue)][e * ucount + u];
    workload_after += cost * after;
    workload_drop += cost * dropped;
  }

  float sat_in_sum = 0.0f;
  float sat_drop_sum = 0.0f;
  float q_sat_sum = 0.0f;
  for (int s = 0; s < sat; ++s) {
    const float incoming = flow_proxy_sat_incoming_for_s(
        a, stage_slot, e, s, bw_mode, counterfactual, target_u, target_c, used_delta, scale, cf_norm);
    const float before = a.f[stage_f(stage_slot, kSfSatQueue)][e * sat + s] + incoming;
    const float service_bits = quantize_device(sat_compute_rate_for(a, e, s) * tau, fp(a, kFpFlowBitsQuantum, 32.0f));
    const float processed = fminf(before, service_bits);
    float after_raw = fmaxf(before - processed, 0.0f);
    const float dropped = fmaxf(after_raw - qmax_sat, 0.0f);
    const float after = fminf(after_raw, qmax_sat);
    const float cost = sat_cost_current(a, e, s);
    sat_in_sum += incoming;
    sat_drop_sum += dropped;
    q_sat_sum += after;
    workload_before += cost * a.f[stage_f(stage_slot, kSfSatQueue)][e * sat + s];
    workload_after += cost * after;
    workload_drop += cost * dropped;
  }

  const float arrival_ref = require_positive_reward_ref(has_f(a, kFStateArrivalRef) ? a.f[kFStateArrivalRef][e] : arrival_sum);
  const int mode = ip(a, kParamFlowProxyRewardMode);
  if (mode == 1) {
    const float x_acc = gu_out_sum / arrival_ref;
    const float x_rel = sat_in_sum / arrival_ref;
    const float d_pre = (gu_drop_sum + uav_drop_sum) / arrival_ref;
    const float b_pre_steps = (q_gu_sum + q_uav_sum) / arrival_ref;
    return fp(a, kFpRewardWAccess, 1.0f) * x_acc +
        fp(a, kFpRewardWRelay, 0.0f) * x_rel -
        fp(a, kFpRewardWPreDrop, 0.0f) * d_pre -
        fp(a, kFpRewardWPreBacklog, 0.0f) * log1pf(fmaxf(b_pre_steps, 0.0f));
  }
  if (mode == 2) return -(workload_after + workload_drop);
  if (mode == 3) return -(workload_after - workload_before) - workload_drop;
  if (mode == 4) return (-(workload_after - workload_before) - workload_drop) / fmaxf(workload_before, 1.0f);
  (void)uav_out_sum;
  (void)sat_drop_sum;
  (void)q_sat_sum;
  return 0.0f;
}

__device__ void write_flow_proxy_scores_parallel(const PackedAbi& a, int stage_slot, int e, int row, int bw_mode) {
  if (!has_f(a, kFHistBwFlowProxyScores) && !has_f(a, kFHistBwFlowProxyMasks) && !has_f(a, kFHistBwFlowProxyDeltas)) return;
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const bool enabled = ip(a, kParamFlowProxyEnabled) && ip(a, kParamFlowProxyRewardMode) > 0;
  for (int idx_local = threadIdx.x; idx_local < ucount * gu; idx_local += blockDim.x) {
    const int u = idx_local / gu;
    const int c = idx_local - u * gu;
    const int idx = (row * ucount + u) * gu + c;
    float used_delta = 0.0f;
    float scale = 1.0f;
    float cf_norm = 1.0f;
    bool active = false;
    if (enabled) {
      flow_proxy_target_stats(a, stage_slot, e, u, c, bw_mode, &used_delta, &scale, &cf_norm, &active);
    }
    float score = 0.0f;
    if (active) {
      const float base_reward = flow_proxy_reward(a, stage_slot, e, bw_mode, false, -1, -1, 0.0f, 1.0f, 1.0f);
      const float cf_reward = flow_proxy_reward(a, stage_slot, e, bw_mode, true, u, c, used_delta, scale, cf_norm);
      const float eps = positive_coeff(fp(a, kFpFlowProxyEps, kRelativeLogEps));
      score = fabsf(used_delta) > eps ? (cf_reward - base_reward) / used_delta : 0.0f;
    }
    if (has_f(a, kFHistBwFlowProxyScores)) a.f[kFHistBwFlowProxyScores][idx] = active ? score : 0.0f;
    if (has_f(a, kFHistBwFlowProxyMasks)) a.f[kFHistBwFlowProxyMasks][idx] = active ? 1.0f : 0.0f;
    if (has_f(a, kFHistBwFlowProxyDeltas)) a.f[kFHistBwFlowProxyDeltas][idx] = active ? used_delta : 0.0f;
  }
}

__device__ void commit_actions_parallel(const PackedAbi& a, int slot, int active_idx, int e, int accel_mode, int sat_mode, int bw_mode) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int row = hist_env_row(a, slot, e);
  __shared__ float logprob_reduce[128];
  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    for (int d = 0; d < 2; ++d) {
      const float value = accel_mode == kSourceZero ? 0.0f : a.f[kFLiveAccelAction][(e * ucount + u) * 2 + d];
      a.f[kFHistAccelActions][(row * ucount + u) * 2 + d] = value;
      if (has_f(a, kFHistAccelLatentActions)) {
        const float z_value = (
            accel_mode != kSourceZero && has_f(a, kFLiveAccelLatentAction))
            ? a.f[kFLiveAccelLatentAction][(e * ucount + u) * 2 + d]
            : 0.0f;
        a.f[kFHistAccelLatentActions][(row * ucount + u) * 2 + d] = z_value;
      }
    }
    const int64_t sat_subset_index = sat_mode == kSourceZero ? 0 : a.l[kLLiveSatSubsetIndex][e * ucount + u];
    a.l[kLHistSatActions][row * ucount + u] = sat_subset_index;
    const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
    if (has_l(a, kLHistSatActionIndices) && has_l(a, kLLiveSatActionIndices)) {
      for (int k = 0; k < select_k; ++k) {
        a.l[kLHistSatActionIndices][(row * ucount + u) * select_k + k] =
            a.l[kLLiveSatActionIndices][(e * ucount + u) * select_k + k];
      }
    }
    if (has_f(a, kFHistSatOldLogprobsPerAgent)) {
      a.f[kFHistSatOldLogprobsPerAgent][row * ucount + u] =
          (sat_mode != kSourceZero && has_f(a, kFLiveSatOldLogprobPerAgent))
              ? a.f[kFLiveSatOldLogprobPerAgent][e * ucount + u]
              : 0.0f;
    }
    for (int g = 0; g < gu; ++g) {
      const float value = bw_mode == kSourceZero ? 0.0f : a.f[kFLiveBwAction][(e * ucount + u) * gu + g];
      a.f[kFHistBwActions][(row * ucount + u) * gu + g] = value;
      a.f[kFHistBwRefActions][(row * ucount + u) * gu + g] =
          bw_mode == kSourceZero ? 0.0f : a.f[kFLiveBwRefAction][(e * ucount + u) * gu + g];
    }
    a.f[kFHistBwOldLogprobsPerAgent][row * ucount + u] =
        bw_mode == kSourceZero ? 0.0f : a.f[kFLiveBwOldLogprobPerAgent][e * ucount + u];
    if (has_f(a, kFHistBwEntropyPerAgent) && has_f(a, kFLiveBwEntropyPerAgent)) {
      a.f[kFHistBwEntropyPerAgent][row * ucount + u] =
          bw_mode == kSourceZero ? 0.0f : a.f[kFLiveBwEntropyPerAgent][e * ucount + u];
    }
    if (has_f(a, kFHistBwLogprobRawPerAgent) && has_f(a, kFLiveBwLogprobRawPerAgent)) {
      a.f[kFHistBwLogprobRawPerAgent][row * ucount + u] =
          bw_mode == kSourceZero ? 0.0f : a.f[kFLiveBwLogprobRawPerAgent][e * ucount + u];
    }
    if (has_f(a, kFHistBwEntropyRawPerAgent) && has_f(a, kFLiveBwEntropyRawPerAgent)) {
      a.f[kFHistBwEntropyRawPerAgent][row * ucount + u] =
          bw_mode == kSourceZero ? 0.0f : a.f[kFLiveBwEntropyRawPerAgent][e * ucount + u];
    }
    if (has_f(a, kFHistBwTau) && has_f(a, kFLiveBwTau)) {
      a.f[kFHistBwTau][row * ucount + u] =
          bw_mode == kSourceZero ? 0.0f : a.f[kFLiveBwTau][e * ucount + u];
    }
    if (has_f(a, kFHistBwKappa) && has_f(a, kFLiveBwKappa)) {
      a.f[kFHistBwKappa][row * ucount + u] =
          bw_mode == kSourceZero ? 0.0f : a.f[kFLiveBwKappa][e * ucount + u];
    }
    if (has_l(a, kLHistBwValidCount) && has_l(a, kLLiveBwValidCount)) {
      a.l[kLHistBwValidCount][row * ucount + u] =
          bw_mode == kSourceZero ? 0 : a.l[kLLiveBwValidCount][e * ucount + u];
    }
    if (has_l(a, kLHistBwLatentCount) && has_l(a, kLLiveBwLatentCount)) {
      a.l[kLHistBwLatentCount][row * ucount + u] =
          bw_mode == kSourceZero ? 0 : a.l[kLLiveBwLatentCount][e * ucount + u];
    }
  }
  __syncthreads();
  float local_accel_logprob = 0.0f;
  float local_sat_logprob = 0.0f;
  float local_bw_logprob = 0.0f;
  if (accel_mode != kSourceZero && has_f(a, kFLiveAccelOldLogprob)) {
    for (int u = threadIdx.x; u < ucount; u += blockDim.x) local_accel_logprob += a.f[kFLiveAccelOldLogprob][e * ucount + u];
  }
  if (sat_mode != kSourceZero && has_f(a, kFLiveSatOldLogprobPerAgent)) {
    for (int u = threadIdx.x; u < ucount; u += blockDim.x) local_sat_logprob += a.f[kFLiveSatOldLogprobPerAgent][e * ucount + u];
  }
  if (bw_mode != kSourceZero && has_f(a, kFLiveBwOldLogprobPerAgent)) {
    for (int u = threadIdx.x; u < ucount; u += blockDim.x) local_bw_logprob += a.f[kFLiveBwOldLogprobPerAgent][e * ucount + u];
  }
  const float accel_logprob = block_reduce_sum_128(local_accel_logprob, logprob_reduce);
  __syncthreads();
  const float sat_logprob = block_reduce_sum_128(local_sat_logprob, logprob_reduce);
  __syncthreads();
  const float bw_agent_logprob = block_reduce_sum_128(local_bw_logprob, logprob_reduce);
  __syncthreads();
  if (threadIdx.x == 0) {
    float bw_logprob = bw_agent_logprob;
    if (!(bw_mode != kSourceZero && has_f(a, kFLiveBwOldLogprobPerAgent)) &&
        bw_mode != kSourceZero && has_f(a, kFLiveBwOldLogprob)) {
      bw_logprob = a.f[kFLiveBwOldLogprob][e];
    }
    a.f[kFHistAccelOldLogprobs][row] = accel_logprob;
    a.f[kFHistSatOldLogprobs][row] = sat_logprob;
    a.f[kFHistBwOldLogprobs][row] = bw_logprob;
    a.f[kFHistAccelValues][row] = 0.0f;
    a.f[kFHistSatValues][row] = 0.0f;
    a.f[kFHistBwValues][row] = 0.0f;
  }
  (void)active_idx;
}

__device__ void commit_reward_parts_parallel(
    const PackedAbi& a,
    int row,
    float service_ratio,
    float drop_ratio,
    float arrival_ref,
    float b_pre_steps,
    float x_acc,
    float x_rel,
    float g_pre,
    float d_pre,
    float processed_ratio_eval,
    float drop_ratio_eval,
    float pre_backlog_steps_eval,
    float sat_overlap_eval,
    float d_sys_report,
    float drop_sum,
    float gu_queue_sum,
    float uav_queue_sum,
    float sat_queue_sum,
    float queue_total_sum,
    float drop_sum_active,
    float expire_sum,
    float gu_drop_sum,
    float uav_drop_sum,
    float sat_drop_sum,
    float arrival_sum,
    float outflow_sum,
    float backhaul_sum,
    float sat_processed_sum,
    float collision_event,
    float overflow_risk_mean,
    float downstream_pressure_mean,
    float service_gap_mean,
    float service_gap_risk_mean,
    float weighted_delta,
    float weighted_level,
    float gu_queue_level,
    float system_queue_level,
    float gu_service_queue,
    float intervention_norm,
    float intervention_rate,
    float intervention_norm_top1,
    float danger_active_rate,
    float close_risk,
    float term_close_risk,
    float reward_raw) {
  for (int idx = threadIdx.x; idx < 44; idx += blockDim.x) {
    if (has_f(a, kFHistRewardParts + idx)) a.f[kFHistRewardParts + idx][row] = 0.0f;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (has_f(a, kFHistRewardParts + 0)) a.f[kFHistRewardParts + 0][row] = service_ratio;
    if (has_f(a, kFHistRewardParts + 1)) a.f[kFHistRewardParts + 1][row] = drop_ratio;
    if (has_f(a, kFHistRewardParts + 2)) a.f[kFHistRewardParts + 2][row] = arrival_ref;
    if (has_f(a, kFHistRewardParts + 3)) a.f[kFHistRewardParts + 3][row] = b_pre_steps;
    if (has_f(a, kFHistRewardParts + 4)) a.f[kFHistRewardParts + 4][row] = x_acc;
    if (has_f(a, kFHistRewardParts + 5)) a.f[kFHistRewardParts + 5][row] = x_rel;
    if (has_f(a, kFHistRewardParts + 6)) a.f[kFHistRewardParts + 6][row] = g_pre;
    if (has_f(a, kFHistRewardParts + 7)) a.f[kFHistRewardParts + 7][row] = d_pre;
    if (has_f(a, kFHistRewardParts + 8)) a.f[kFHistRewardParts + 8][row] = processed_ratio_eval;
    if (has_f(a, kFHistRewardParts + 9)) a.f[kFHistRewardParts + 9][row] = drop_ratio_eval;
    if (has_f(a, kFHistRewardParts + 10)) a.f[kFHistRewardParts + 10][row] = pre_backlog_steps_eval;
    if (has_f(a, kFHistRewardParts + 11)) a.f[kFHistRewardParts + 11][row] = sat_overlap_eval;
    if (has_f(a, kFHistRewardParts + 12)) a.f[kFHistRewardParts + 12][row] = d_sys_report;
    if (has_f(a, kFHistRewardParts + 13)) a.f[kFHistRewardParts + 13][row] = drop_sum;
    if (has_f(a, kFHistRewardParts + 14)) a.f[kFHistRewardParts + 14][row] = gu_queue_sum;
    if (has_f(a, kFHistRewardParts + 15)) a.f[kFHistRewardParts + 15][row] = uav_queue_sum;
    if (has_f(a, kFHistRewardParts + 16)) a.f[kFHistRewardParts + 16][row] = sat_queue_sum;
    if (has_f(a, kFHistRewardParts + 17)) a.f[kFHistRewardParts + 17][row] = queue_total_sum;
    if (has_f(a, kFHistRewardParts + 18)) a.f[kFHistRewardParts + 18][row] = drop_sum_active;
    if (has_f(a, kFHistRewardParts + 19)) a.f[kFHistRewardParts + 19][row] = expire_sum;
    if (has_f(a, kFHistRewardParts + 20)) a.f[kFHistRewardParts + 20][row] = gu_drop_sum;
    if (has_f(a, kFHistRewardParts + 21)) a.f[kFHistRewardParts + 21][row] = uav_drop_sum;
    if (has_f(a, kFHistRewardParts + 22)) a.f[kFHistRewardParts + 22][row] = sat_drop_sum;
    if (has_f(a, kFHistRewardParts + 23)) a.f[kFHistRewardParts + 23][row] = arrival_sum;
    if (has_f(a, kFHistRewardParts + 24)) a.f[kFHistRewardParts + 24][row] = outflow_sum;
    if (has_f(a, kFHistRewardParts + 25)) a.f[kFHistRewardParts + 25][row] = backhaul_sum;
    if (has_f(a, kFHistRewardParts + 26)) a.f[kFHistRewardParts + 26][row] = sat_processed_sum;
    if (has_f(a, kFHistRewardParts + 27)) a.f[kFHistRewardParts + 27][row] = collision_event;
    if (has_f(a, kFHistRewardParts + 28)) a.f[kFHistRewardParts + 28][row] = overflow_risk_mean;
    if (has_f(a, kFHistRewardParts + 29)) a.f[kFHistRewardParts + 29][row] = downstream_pressure_mean;
    if (has_f(a, kFHistRewardParts + 30)) a.f[kFHistRewardParts + 30][row] = service_gap_mean;
    if (has_f(a, kFHistRewardParts + 31)) a.f[kFHistRewardParts + 31][row] = service_gap_risk_mean;
    if (has_f(a, kFHistRewardParts + 32)) a.f[kFHistRewardParts + 32][row] = weighted_delta;
    if (has_f(a, kFHistRewardParts + 33)) a.f[kFHistRewardParts + 33][row] = weighted_level;
    if (has_f(a, kFHistRewardParts + 34)) a.f[kFHistRewardParts + 34][row] = gu_queue_level;
    if (has_f(a, kFHistRewardParts + 35)) a.f[kFHistRewardParts + 35][row] = system_queue_level;
    if (has_f(a, kFHistRewardParts + 36)) a.f[kFHistRewardParts + 36][row] = gu_service_queue;
    if (has_f(a, kFHistRewardParts + 37)) a.f[kFHistRewardParts + 37][row] = intervention_norm;
    if (has_f(a, kFHistRewardParts + 38)) a.f[kFHistRewardParts + 38][row] = intervention_rate;
    if (has_f(a, kFHistRewardParts + 39)) a.f[kFHistRewardParts + 39][row] = intervention_norm_top1;
    if (has_f(a, kFHistRewardParts + 40)) a.f[kFHistRewardParts + 40][row] = danger_active_rate;
    if (has_f(a, kFHistRewardParts + 41)) a.f[kFHistRewardParts + 41][row] = close_risk;
    if (has_f(a, kFHistRewardParts + 42)) a.f[kFHistRewardParts + 42][row] = term_close_risk;
    if (has_f(a, kFHistRewardParts + 43)) a.f[kFHistRewardParts + 43][row] = reward_raw;
  }
}

__device__ void apply_reset_or_state_commit_parallel(const PackedAbi& a, int slot, int e, bool done) {
  const int ecount = static_cast<int>(ip(a, kParamNumEnvs));
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  const int bw_stage = 3;
  if (done) {
    const int reset_ordinal = has_i(a, kIRandomResetCount) ? a.i[kIRandomResetCount][e] : slot;
    int reset_rows = 1;
    if (has_f(a, kFRandomResetArrivalRefTape)) {
      reset_rows = max(static_cast<int>(a.f_numel[kFRandomResetArrivalRefTape] / max(static_cast<int64_t>(ecount), static_cast<int64_t>(1))), 1);
    } else if (has_i(a, kIRandomResetEpisodeIdxTape)) {
      reset_rows = max(static_cast<int>(a.i_numel[kIRandomResetEpisodeIdxTape] / max(static_cast<int64_t>(ecount), static_cast<int64_t>(1))), 1);
    } else if (has_f(a, kFRandomResetGuPosTape)) {
      const int64_t denom = max(static_cast<int64_t>(ecount) * max(gu, 1) * 2, static_cast<int64_t>(1));
      reset_rows = max(static_cast<int>(a.f_numel[kFRandomResetGuPosTape] / denom), 1);
    }
    const int reset_idx = min(max(reset_ordinal, 0), reset_rows - 1);
    const int64_t reset_row = static_cast<int64_t>(reset_idx) * ecount + e;
    float reset_arrival_ref = has_f(a, kFStateArrivalRef) ? a.f[kFStateArrivalRef][e] : 0.0f;
    if (has_f(a, kFRandomResetArrivalRefTape)) {
      const int64_t idx = reset_row;
      if (idx >= 0 && idx < a.f_numel[kFRandomResetArrivalRefTape]) {
        reset_arrival_ref = a.f[kFRandomResetArrivalRefTape][idx];
      }
    }
    float reset_effective_rate = has_f(a, kFStateEffectiveArrivalRate) ? a.f[kFStateEffectiveArrivalRate][e] : 0.0f;
    if (has_f(a, kFRandomResetEffectiveArrivalRateTape)) {
      const int64_t idx = reset_row;
      if (idx >= 0 && idx < a.f_numel[kFRandomResetEffectiveArrivalRateTape]) {
        reset_effective_rate = a.f[kFRandomResetEffectiveArrivalRateTape][idx];
      }
    }
    reset_arrival_ref = require_positive_reward_ref(reset_arrival_ref);
    reset_effective_rate = fmaxf(reset_effective_rate, 0.0f);
    const float reset_gu_ema = reset_arrival_ref / fmaxf(static_cast<float>(gu), 1.0f);
    const float reset_uav_ema = reset_arrival_ref / fmaxf(static_cast<float>(ucount), 1.0f);
    float sat_ref_count = fp(a, kFpBwWorkloadSatActiveRef, 0.0f);
    if (sat_ref_count <= 0.0f) {
      sat_ref_count = (select_k > 0 && ucount > 0)
          ? fmaxf(fminf(static_cast<float>(sat), static_cast<float>(select_k * ucount)), 1.0f)
          : fmaxf(static_cast<float>(sat), 1.0f);
    }
    const float reset_sat_ema = reset_arrival_ref / fmaxf(sat_ref_count, 1.0f);
    if (threadIdx.x == 0) {
      if (has_f(a, kFStateArrivalRef)) a.f[kFStateArrivalRef][e] = reset_arrival_ref;
      if (has_f(a, kFStateEffectiveArrivalRate)) a.f[kFStateEffectiveArrivalRate][e] = reset_effective_rate;
      if (has_f(a, kFStatePrevQueueSumGu)) a.f[kFStatePrevQueueSumGu][e] = 0.0f;
      if (has_f(a, kFStatePrevQueueSumUav)) a.f[kFStatePrevQueueSumUav][e] = 0.0f;
      if (has_f(a, kFStatePrevQueueSumSat)) a.f[kFStatePrevQueueSumSat][e] = 0.0f;
      if (has_f(a, kFStatePrevQNormActive)) a.f[kFStatePrevQNormActive][e] = 0.0f;
      if (has_i(a, kIStateTrafficResetStep)) a.i[kIStateTrafficResetStep][e] = slot;
      if (has_i(a, kIStateTrafficResetOrdinal)) a.i[kIStateTrafficResetOrdinal][e] = reset_ordinal;
      if (has_i(a, kIStateEpisodeIdx)) {
        int episode_idx = a.i[kIStateEpisodeIdx][e] + 1;
        if (has_i(a, kIRandomResetEpisodeIdxTape) && reset_row >= 0 && reset_row < a.i_numel[kIRandomResetEpisodeIdxTape]) {
          episode_idx = a.i[kIRandomResetEpisodeIdxTape][reset_row];
        }
        a.i[kIStateEpisodeIdx][e] = episode_idx;
      }
      if (has_i(a, kIStateHotspotActiveIdx)) {
        int active_idx = -1;
        if (has_i(a, kIRandomResetHotspotActiveIdxTape) && reset_row >= 0 && reset_row < a.i_numel[kIRandomResetHotspotActiveIdxTape]) {
          active_idx = a.i[kIRandomResetHotspotActiveIdxTape][reset_row];
        }
        a.i[kIStateHotspotActiveIdx][e] = active_idx;
      }
      if (has_i(a, kIStateHotspotSubsetCount)) {
        int subset_count = 0;
        if (has_i(a, kIRandomResetHotspotSubsetCountTape) && reset_row >= 0 && reset_row < a.i_numel[kIRandomResetHotspotSubsetCountTape]) {
          subset_count = a.i[kIRandomResetHotspotSubsetCountTape][reset_row];
        }
        a.i[kIStateHotspotSubsetCount][e] = subset_count;
      }
    }
    __syncthreads();
    for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
      for (int d = 0; d < 2; ++d) {
        float pos = a.f[stage_f(bw_stage, kSfUavPos)][(e * ucount + u) * 2 + d];
        float vel = 0.0f;
        const int64_t idx = (reset_row * ucount + u) * 2 + d;
        if (has_f(a, kFRandomResetUavPosTape) && idx < a.f_numel[kFRandomResetUavPosTape]) pos = a.f[kFRandomResetUavPosTape][idx];
        if (has_f(a, kFRandomResetUavVelTape) && idx < a.f_numel[kFRandomResetUavVelTape]) vel = a.f[kFRandomResetUavVelTape][idx];
        a.f[kFStateUavPos][(e * ucount + u) * 2 + d] = pos;
        a.f[kFStateUavVel][(e * ucount + u) * 2 + d] = vel;
      }
      a.f[kFStateUavEnergy][e * ucount + u] = fp(a, kFpUavEnergyInit, 1.0f);
      float q = 0.0f;
      const int64_t qidx = reset_row * ucount + u;
      if (has_f(a, kFRandomResetUavQueueTape) && qidx < a.f_numel[kFRandomResetUavQueueTape]) q = a.f[kFRandomResetUavQueueTape][qidx];
      a.f[kFStateUavQueue][e * ucount + u] = q;
      a.f[kFStatePrevUavQueueVec][e * ucount + u] = q;
      if (has_f(a, kFStateUavEma)) a.f[kFStateUavEma][e * ucount + u] = reset_uav_ema;
      if (has_f(a, kFStateLastExecAccel)) {
        a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 0] = 0.0f;
        a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 1] = 0.0f;
      }
      if (has_f(a, kFStateLastPolicyAccel)) {
        a.f[kFStateLastPolicyAccel][(e * ucount + u) * 2 + 0] = 0.0f;
        a.f[kFStateLastPolicyAccel][(e * ucount + u) * 2 + 1] = 0.0f;
      }
      if (has_f(a, kFStatePrevQueueSumUav)) atomicAdd(&a.f[kFStatePrevQueueSumUav][e], q);
    }
    for (int g = threadIdx.x; g < gu; g += blockDim.x) {
      for (int d = 0; d < 2; ++d) {
        float pos = a.f[stage_f(bw_stage, kSfGuPos)][(e * gu + g) * 2 + d];
        const int64_t idx = (reset_row * gu + g) * 2 + d;
        if (has_f(a, kFRandomResetGuPosTape) && idx < a.f_numel[kFRandomResetGuPosTape]) pos = a.f[kFRandomResetGuPosTape][idx];
        a.f[kFStateGuPos][(e * gu + g) * 2 + d] = pos;
      }
      float q = 0.0f;
      const int64_t qidx = reset_row * gu + g;
      if (has_f(a, kFRandomResetGuQueueTape) && qidx < a.f_numel[kFRandomResetGuQueueTape]) q = a.f[kFRandomResetGuQueueTape][qidx];
      a.f[kFStateGuQueue][e * gu + g] = q;
      a.f[kFStatePrevGuQueueVec][e * gu + g] = q;
      a.f[kFStateLastGuArrival][e * gu + g] = 0.0f;
      a.f[kFStateLastGuOutflow][e * gu + g] = 0.0f;
      if (has_f(a, kFStateLastArrivalRateVec)) {
        float rate = reset_effective_rate;
        const int64_t ridx = reset_row * gu + g;
        if (has_f(a, kFRandomResetArrivalRateVecTape) && ridx >= 0 && ridx < a.f_numel[kFRandomResetArrivalRateVecTape]) {
          rate = a.f[kFRandomResetArrivalRateVecTape][ridx];
        }
        a.f[kFStateLastArrivalRateVec][e * gu + g] = fmaxf(rate, 0.0f);
      }
      if (has_f(a, kFStateArrivalBaseScale)) {
        float scale = 1.0f;
        const int64_t ridx = reset_row * gu + g;
        if (has_f(a, kFRandomResetArrivalBaseScaleTape) && ridx >= 0 && ridx < a.f_numel[kFRandomResetArrivalBaseScaleTape]) {
          scale = a.f[kFRandomResetArrivalBaseScaleTape][ridx];
        }
        a.f[kFStateArrivalBaseScale][e * gu + g] = scale;
      }
      if (has_f(a, kFStateGuDeadlineSteps)) {
        float deadline = a.f[kFStateGuDeadlineSteps][e * gu + g];
        const int64_t ridx = reset_row * gu + g;
        if (has_f(a, kFRandomResetDeadlineStepsTape) && ridx >= 0 && ridx < a.f_numel[kFRandomResetDeadlineStepsTape]) {
          deadline = a.f[kFRandomResetDeadlineStepsTape][ridx];
        }
        deadline = fmaxf(deadline, 0.0f);
        a.f[kFStateGuDeadlineSteps][e * gu + g] = deadline;
        if (has_f(a, kFStateDeadlineSlack)) a.f[kFStateDeadlineSlack][e * gu + g] = deadline;
      }
      if (has_f(a, kFStateGuEma)) a.f[kFStateGuEma][e * gu + g] = reset_gu_ema;
      if (has_f(a, kFStateUrgencyRisk)) a.f[kFStateUrgencyRisk][e * gu + g] = 0.0f;
      if (has_f(a, kFStateDownstreamPressure)) a.f[kFStateDownstreamPressure][e * gu + g] = 0.0f;
      if (has_f(a, kFStateServiceGapRisk)) a.f[kFStateServiceGapRisk][e * gu + g] = 0.0f;
      a.f[kFStateServiceGap][e * gu + g] = 0.0f;
      a.f[kFStateDeadlineAge][e * gu + g] = 0.0f;
      if (has_f(a, kFStateDeadlineRisk)) a.f[kFStateDeadlineRisk][e * gu + g] = 0.0f;
      if (has_i(a, kIStateLastAssociation)) a.i[kIStateLastAssociation][e * gu + g] = -1;
      if (has_i(a, kIStatePrevAssociation)) a.i[kIStatePrevAssociation][e * gu + g] = -1;
      if (has_f(a, kFStatePrevQueueSumGu)) atomicAdd(&a.f[kFStatePrevQueueSumGu][e], q);
    }
    if (has_f(a, kFStateGuClusterCenters)) {
      const int centers = static_cast<int>(a.f_numel[kFStateGuClusterCenters] / max(static_cast<int64_t>(ecount * 2), static_cast<int64_t>(1)));
      for (int idx = threadIdx.x; idx < centers * 2; idx += blockDim.x) {
        const int c = idx / 2;
        const int d = idx - c * 2;
        float value = a.f[kFStateGuClusterCenters][(e * centers + c) * 2 + d];
        const int64_t tape_idx = (reset_row * centers + c) * 2 + d;
        if (has_f(a, kFRandomResetGuClusterCentersTape) && tape_idx < a.f_numel[kFRandomResetGuClusterCentersTape]) {
          value = a.f[kFRandomResetGuClusterCentersTape][tape_idx];
        }
        a.f[kFStateGuClusterCenters][(e * centers + c) * 2 + d] = value;
      }
      for (int c = threadIdx.x; c < centers; c += blockDim.x) {
        float value = has_f(a, kFStateGuClusterCounts) ? a.f[kFStateGuClusterCounts][e * centers + c] : 0.0f;
        const int64_t tape_idx = reset_row * centers + c;
        if (has_f(a, kFRandomResetGuClusterCountsTape) && tape_idx < a.f_numel[kFRandomResetGuClusterCountsTape]) {
          value = a.f[kFRandomResetGuClusterCountsTape][tape_idx];
        }
        if (has_f(a, kFStateGuClusterCounts)) a.f[kFStateGuClusterCounts][e * centers + c] = value;
      }
    }
    for (int s = threadIdx.x; s < sat; s += blockDim.x) {
      float q = 0.0f;
      const int64_t qidx = reset_row * sat + s;
      if (has_f(a, kFRandomResetSatQueueTape) && qidx < a.f_numel[kFRandomResetSatQueueTape]) q = a.f[kFRandomResetSatQueueTape][qidx];
      a.f[kFStateSatQueue][e * sat + s] = q;
      a.f[kFStatePrevSatQueueVec][e * sat + s] = q;
      if (has_f(a, kFStateSatEma)) a.f[kFStateSatEma][e * sat + s] = reset_sat_ema;
      if (has_f(a, kFStateLastSatConnectionCounts)) a.f[kFStateLastSatConnectionCounts][e * sat + s] = 0.0f;
      if (has_f(a, kFStatePrevQueueSumSat)) atomicAdd(&a.f[kFStatePrevQueueSumSat][e], q);
    }
    if (has_f(a, kFStateDopplerResidual)) {
      for (int idx = threadIdx.x; idx < ucount * sat; idx += blockDim.x) {
        float value = 0.0f;
        const int64_t tape_idx = (reset_row * ucount * sat) + idx;
        if (has_f(a, kFRandomResetDopplerResidualTape) && tape_idx >= 0 && tape_idx < a.f_numel[kFRandomResetDopplerResidualTape]) {
          value = a.f[kFRandomResetDopplerResidualTape][tape_idx];
        }
        a.f[kFStateDopplerResidual][e * ucount * sat + idx] = value;
      }
    }
    if (has_f(a, kFStateHotspotMemberMask)) {
      const int64_t denom = max(static_cast<int64_t>(ecount) * max(static_cast<int64_t>(gu), static_cast<int64_t>(1)), static_cast<int64_t>(1));
      const int max_subsets = static_cast<int>(a.f_numel[kFStateHotspotMemberMask] / denom);
      for (int idx = threadIdx.x; idx < max_subsets * gu; idx += blockDim.x) {
        float value = 0.0f;
        const int64_t tape_idx = (reset_row * max_subsets * gu) + idx;
        if (has_f(a, kFRandomResetHotspotMemberMaskTape) && tape_idx >= 0 && tape_idx < a.f_numel[kFRandomResetHotspotMemberMaskTape]) {
          value = a.f[kFRandomResetHotspotMemberMaskTape][tape_idx];
        }
        a.f[kFStateHotspotMemberMask][e * max_subsets * gu + idx] = value;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0 && has_f(a, kFStatePrevQNormActive)) {
      const float q_active =
          (has_f(a, kFStatePrevQueueSumGu) ? a.f[kFStatePrevQueueSumGu][e] : 0.0f) +
          (has_f(a, kFStatePrevQueueSumUav) ? a.f[kFStatePrevQueueSumUav][e] : 0.0f);
      float arrival_floor = fp(a, kFpQueueNormArrivalFloor, 0.0f);
      if (arrival_floor <= 0.0f) {
        arrival_floor = reset_effective_rate * static_cast<float>(gu) * fp(a, kFpTau0, 1.0f);
      }
      const float scale = positive_config_scale(fp(a, kFpQueueNormK, 1.0f)) *
          require_positive_reward_ref(fmaxf(reset_arrival_ref, arrival_floor));
      a.f[kFStatePrevQNormActive][e] = clampf_device(q_active / scale, 0.0f, 1.0f);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < ucount * select_k; idx += blockDim.x) {
      if (has_l(a, kLStateLastSatSelectionMatrix)) a.l[kLStateLastSatSelectionMatrix][e * ucount * select_k + idx] = -1;
    }
    for (int g = threadIdx.x; g < gu; g += blockDim.x) {
      if (has_f(a, kFStateGuDrop)) a.f[kFStateGuDrop][e * gu + g] = 0.0f;
    }
    for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
      if (has_f(a, kFStateUavDrop)) a.f[kFStateUavDrop][e * ucount + u] = 0.0f;
      if (has_f(a, kFStateLastAccessInterferenceByUav)) a.f[kFStateLastAccessInterferenceByUav][e * ucount + u] = 0.0f;
      if (has_f(a, kFStateLastGuToUavInflowByUav)) a.f[kFStateLastGuToUavInflowByUav][e * ucount + u] = 0.0f;
    }
    for (int s = threadIdx.x; s < sat; s += blockDim.x) {
      if (has_f(a, kFStateSatDrop)) a.f[kFStateSatDrop][e * sat + s] = 0.0f;
      if (has_f(a, kFStateLastSatProcessed)) a.f[kFStateLastSatProcessed][e * sat + s] = 0.0f;
    }
    for (int idx = threadIdx.x; idx < ucount * gu; idx += blockDim.x) {
      if (has_f(a, kFStateLastBwFractionByUavGu)) a.f[kFStateLastBwFractionByUavGu][e * ucount * gu + idx] = 0.0f;
    }
    for (int idx = threadIdx.x; idx < ucount * sat; idx += blockDim.x) {
      if (has_f(a, kFStateLastUavToSatOutflowMatrix)) a.f[kFStateLastUavToSatOutflowMatrix][e * ucount * sat + idx] = 0.0f;
      if (has_f(a, kFStateLastSelectedMaskByUavSat)) a.f[kFStateLastSelectedMaskByUavSat][e * ucount * sat + idx] = 0.0f;
    }
    if (threadIdx.x == 0) {
      if (has_i(a, kIStateT)) a.i[kIStateT][e] = 0;
      if (has_i(a, kIRandomResetCount)) a.i[kIRandomResetCount][e] += 1;
      if (has_i(a, kIRandomStepTensor) && a.i_numel[kIRandomStepTensor] > 1) a.i[kIRandomStepTensor][1] = 1;
    }
  } else {
    for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
      for (int d = 0; d < 2; ++d) {
        a.f[kFStateUavPos][(e * ucount + u) * 2 + d] = a.f[stage_f(bw_stage, kSfUavPos)][(e * ucount + u) * 2 + d];
        a.f[kFStateUavVel][(e * ucount + u) * 2 + d] = a.f[stage_f(bw_stage, kSfUavVel)][(e * ucount + u) * 2 + d];
      }
      a.f[kFStateUavEnergy][e * ucount + u] = a.f[stage_f(bw_stage, kSfUavEnergy)][e * ucount + u];
      a.f[kFStateUavQueue][e * ucount + u] = a.f[stage_f(bw_stage, kSfUavQueue)][e * ucount + u];
      a.f[kFStatePrevUavQueueVec][e * ucount + u] = a.f[stage_f(bw_stage, kSfUavQueue)][e * ucount + u];
    }
    for (int g = threadIdx.x; g < gu; g += blockDim.x) {
      a.f[kFStateGuQueue][e * gu + g] = a.f[stage_f(bw_stage, kSfGuQueue)][e * gu + g];
      a.f[kFStatePrevGuQueueVec][e * gu + g] = a.f[stage_f(bw_stage, kSfGuQueue)][e * gu + g];
      a.i[kIStateLastAssociation][e * gu + g] = static_cast<int>(a.l[stage_l(bw_stage, kSlAssoc)][e * gu + g]);
      a.i[kIStatePrevAssociation][e * gu + g] = static_cast<int>(a.l[stage_l(bw_stage, kSlPrevAssociation)][e * gu + g]);
    }
    for (int s = threadIdx.x; s < sat; s += blockDim.x) {
      a.f[kFStateSatQueue][e * sat + s] = a.f[stage_f(bw_stage, kSfSatQueue)][e * sat + s];
      a.f[kFStatePrevSatQueueVec][e * sat + s] = a.f[stage_f(bw_stage, kSfSatQueue)][e * sat + s];
    }
    for (int idx = threadIdx.x; idx < ucount * select_k; idx += blockDim.x) {
      if (has_l(a, kLStateLastSatSelectionMatrix)) {
        a.l[kLStateLastSatSelectionMatrix][e * ucount * select_k + idx] =
            a.l[stage_l(bw_stage, kSlSatSelectionMatrix)][e * ucount * select_k + idx];
      }
    }
    if (threadIdx.x == 0) {
      if (has_i(a, kIStateHotspotActiveIdx)) {
        int active_after = a.i[kIStateHotspotActiveIdx][e];
        if (hotspot_active_after_for_step(a, e, slot, &active_after)) {
          a.i[kIStateHotspotActiveIdx][e] = active_after;
        }
      }
      if (has_i(a, kIStateT)) a.i[kIStateT][e] += 1;
    }
  }
  __syncthreads();
  const int current_t = has_i(a, kIStateT) ? a.i[kIStateT][e] : 0;
  sync_state_orbit_from_t_parallel(a, e, current_t);
}

__device__ float baseline_energy_term_component(const PackedAbi& a, int live_f, int row, int component) {
  if (!ip(a, kParamEnergyEnabled)) return 0.0f;
  const int uav_dim = kAccelEgoDim;
  const float weight = fp(a, kFpBaselineEnergyWeight, 1.0f);
  if (weight <= 0.0f) return 0.0f;
  const float energy_low = fp(a, kFpBaselineEnergyLow, 0.3f);
  const float energy_norm = a.f[live_f + 0][row * uav_dim + kAccelEgoEnergy];
  if (!(energy_norm < energy_low)) return 0.0f;
  const float vx = a.f[live_f + 0][row * uav_dim + kAccelEgoVx];
  const float vy = a.f[live_f + 0][row * uav_dim + kAccelEgoVy];
  const float speed = sqrtf(vx * vx + vy * vy);
  const float target_speed = fminf(fp(a, kFpUavOptSpeed, 0.0f) / positive_config_scale(fp(a, kFpVMax, 1.0f)), 1.0f);
  const float delta = target_speed - speed;
  if (!(speed > kDynamicsDenomEps && delta < 0.0f)) return 0.0f;
  const float scale = (energy_low - energy_norm) / dynamics_denominator(energy_low);
  const float vel_component = component == 0 ? vx : vy;
  return weight * scale * (vel_component / dynamics_denominator(speed)) * delta;
}

constexpr int kSourceBlockMaxThreads = 256;

__device__ float block_sum_float(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  return scratch[0];
}

__device__ int block_sum_int(int value, int* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  return scratch[0];
}

__device__ float block_min_float(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] = fminf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    __syncthreads();
  }
  return scratch[0];
}

__device__ float block_max_float(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    __syncthreads();
  }
  return scratch[0];
}

__device__ void block_argmax_min_index(
    float value,
    int index,
    float* value_scratch,
    int* index_scratch,
    float* out_value,
    int* out_index) {
  value_scratch[threadIdx.x] = value;
  index_scratch[threadIdx.x] = index;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other_value = value_scratch[threadIdx.x + stride];
      const int other_index = index_scratch[threadIdx.x + stride];
      const bool replace =
          other_index >= 0 &&
          (other_value > value_scratch[threadIdx.x] ||
           (other_value == value_scratch[threadIdx.x] &&
            (index_scratch[threadIdx.x] < 0 || other_index < index_scratch[threadIdx.x])));
      if (replace) {
        value_scratch[threadIdx.x] = other_value;
        index_scratch[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  *out_value = value_scratch[0];
  *out_index = index_scratch[0];
}

__device__ void block_argmin_min_index(
    float value,
    int index,
    float* value_scratch,
    int* index_scratch,
    float* out_value,
    int* out_index) {
  value_scratch[threadIdx.x] = value;
  index_scratch[threadIdx.x] = index;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other_value = value_scratch[threadIdx.x + stride];
      const int other_index = index_scratch[threadIdx.x + stride];
      const bool replace =
          other_index >= 0 &&
          (other_value < value_scratch[threadIdx.x] ||
           (other_value == value_scratch[threadIdx.x] &&
            (index_scratch[threadIdx.x] < 0 || other_index < index_scratch[threadIdx.x])));
      if (replace) {
        value_scratch[threadIdx.x] = other_value;
        index_scratch[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  *out_value = value_scratch[0];
  *out_index = index_scratch[0];
}

__device__ __forceinline__ unsigned int source_hash_mix(unsigned int x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

__device__ __forceinline__ int state_int_for_env(const PackedAbi& a, int idx, int e, int default_value = 0) {
  if (!has_i(a, idx) || a.i_numel[idx] <= 0) return default_value;
  const int64_t n = a.i_numel[idx];
  const int64_t pos = (e >= 0 && static_cast<int64_t>(e) < n) ? static_cast<int64_t>(e) : 0LL;
  return a.i[idx][pos];
}

__device__ __forceinline__ float source_hash01(const PackedAbi& a, int e, int u, int item, int salt) {
  const unsigned int t = static_cast<unsigned int>(state_int_for_env(a, kIStateT, e, 0));
  const unsigned int ep = static_cast<unsigned int>(state_int_for_env(a, kIStateEpisodeIdx, e, e));
  const unsigned int global = static_cast<unsigned int>(state_int_for_env(a, kIStateGlobalStep, e, 0));
  unsigned int x = 2166136261U;
  x ^= static_cast<unsigned int>(e + 1) * 16777619U;
  x ^= static_cast<unsigned int>(u + 17) * 2246822519U;
  x ^= static_cast<unsigned int>(item + 131) * 3266489917U;
  x ^= static_cast<unsigned int>(salt + 8191) * 668265263U;
  x ^= t * 374761393U;
  x ^= ep * 1274126177U;
  x ^= global * 2147483647U;
  const unsigned int mixed = source_hash_mix(x);
  return static_cast<float>(mixed & 0x00ffffffU) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float source_clip_positive(float value, float eps = 1.0e-6f) {
  return fmaxf(value, eps);
}

__device__ __forceinline__ int64_t lyapunov_state_offset(const PackedAbi& a, int e, int u, int g) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  return (static_cast<int64_t>(e) * ucount + u) * gu + g;
}

__device__ __forceinline__ float lyapunov_steps_from_log1p(float value) {
  return expm1f(clampf_device(value, 0.0f, 20.0f));
}

__device__ __forceinline__ float lyapunov_queue_from_accel_token(const float* tok) {
  const float queue_steps = lyapunov_steps_from_log1p(tok[kAccelGuQueueSteps]);
  const float expected_steps = lyapunov_steps_from_log1p(tok[kAccelGuExpectedArrivalSteps]);
  return fmaxf(queue_steps + expected_steps, 0.0f);
}

__device__ __forceinline__ float lyapunov_queue_from_bw_token(const float* tok) {
  return lyapunov_steps_from_log1p(tok[kBwGuQueueSteps]);
}

__device__ __forceinline__ float lyapunov_eta_from_accel_token(const float* tok) {
  return fmaxf(tok[kAccelGuAccessSeRef], 0.0f);
}

__device__ __forceinline__ float lyapunov_eta_for_accel_slot(
    const PackedAbi& a,
    int e,
    int u,
    int g,
    const float* tok,
    int accel_stage) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  float eta = tok[kAccelGuAccessSeRef];
  if (has_f(a, stage_f(accel_stage, kSfEtaRefFeature))) {
    eta = a.f[stage_f(accel_stage, kSfEtaRefFeature)][(e * ucount + u) * gu + g];
  }
  return fmaxf(eta, 0.0f);
}

__device__ __forceinline__ float lyapunov_eta_for_bw_slot(
    const PackedAbi& a,
    int e,
    int u,
    int g,
    const float* tok,
    int bw_stage) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  float eta = tok[kBwGuAccessRateFullBwRefSteps];
  if (has_f(a, stage_f(bw_stage, kSfEtaRefFeature))) {
    eta = a.f[stage_f(bw_stage, kSfEtaRefFeature)][(e * ucount + u) * gu + g];
  }
  return fmaxf(eta, 0.0f);
}

__device__ __forceinline__ void lyapunov_sat_profile_from_values(
    float se,
    float q,
    float load,
    float doppler_margin,
    float* se_abs,
    float* queue,
    float* relay_support) {
  const float se_abs_v = tanhf(fmaxf(se, 0.0f) / 6.0f);
  const float doppler_v = clampf_device(doppler_margin, 0.0f, 1.0f);
  const float congestion = 1.0f / (1.0f + fmaxf(load, 0.0f));
  *se_abs = se_abs_v;
  *queue = fmaxf(q, 0.0f);
  *relay_support = clampf_device(se_abs_v * (0.5f + 0.5f * doppler_v) * congestion, 0.0f, 1.0f);
}

__device__ void queue_aware_sat_values(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    int row,
    int slot,
    int width,
    float* se,
    float* q,
    float* load,
    float* bw,
    float* stay);

__device__ float lyapunov_relay_gate(const PackedAbi& a, int stage_slot, int e, int u, int row, int width) {
  float best = -3.402823466e38f;
  for (int s = 0; s < width; ++s) {
    if (!a.b[kBLiveSatObs + 0][row * width + s] || !a.b[kBLiveSatObs + 1][row * width + s]) continue;
    float se = 0.0f, q = 0.0f, load = 0.0f, bw = 0.0f, stay = 0.0f;
    queue_aware_sat_values(a, stage_slot, e, u, row, s, width, &se, &q, &load, &bw, &stay);
    const int idx = (row * width + s) * 26;
    const float observed_load = fmaxf(a.f[kFLiveSatObs + 3][idx + 8], 0.0f);
    const float doppler_margin = 1.0f - clampf_device(a.f[kFLiveSatObs + 3][idx + 20], 0.0f, 1.0f);
    float se_abs = 0.0f, queue = 0.0f, relay = 0.0f;
    lyapunov_sat_profile_from_values(se, q, observed_load, doppler_margin, &se_abs, &queue, &relay);
    best = fmaxf(best, relay);
  }
  return best > -1.0e30f ? 0.5f + 0.5f * best : 0.5f;
}

__device__ __forceinline__ void project_unit_action(float* ax, float* ay) {
  const float norm = sqrtf((*ax) * (*ax) + (*ay) * (*ay));
  if (norm > 1.0f) {
    *ax /= norm;
    *ay /= norm;
  }
}

__device__ __forceinline__ void topology_dpp_accel_candidate_action(
    int candidate,
    int candidate_count,
    float step,
    float gain,
    float* raw_x,
    float* raw_y,
    float* action_x,
    float* action_y) {
  float cx = 0.0f;
  float cy = 0.0f;
  if (candidate > 0 && candidate_count > 1 && step > 0.0f) {
    const float theta = 6.28318530717958647692f
        * static_cast<float>(candidate - 1)
        / fmaxf(static_cast<float>(candidate_count - 1), 1.0f);
    cx = cosf(theta) * step;
    cy = sinf(theta) * step;
  }
  float ax = cx * gain;
  float ay = cy * gain;
  project_unit_action(&ax, &ay);
  *raw_x = cx;
  *raw_y = cy;
  *action_x = ax;
  *action_y = ay;
}

__device__ __forceinline__ float observable_cluster_token_weight(
    const PackedAbi& a,
    int accel_stage,
    int e,
    int u,
    int g,
    const float* tok,
    float assoc_bonus) {
  const float pressure = fmaxf(
      tok[kAccelGuQueueSteps] + tok[kAccelGuExpectedArrivalSteps] - tok[kAccelGuLastOutflowSteps],
      0.0f) + fmaxf(tok[kAccelGuLastDropSteps], 0.0f);
  const float se = fmaxf(lyapunov_eta_for_accel_slot(a, e, u, g, tok, accel_stage), 0.0f);
  float weight = fmaxf(pressure, 1.0e-3f) * (0.5f + se);
  if (assoc_bonus > 0.0f) {
    weight *= 1.0f + assoc_bonus * clampf_device(tok[kAccelGuPreOwnerIsEgo], 0.0f, 1.0f);
  }
  return fmaxf(weight, 0.0f);
}

__global__ void observable_cluster_accel_live_kernel(int64_t active_idx) {
  const PackedAbi& a = cLiveAbi;
  const int e = blockIdx.x;
  const int u = blockIdx.y;
  if (e >= ip(a, kParamNumEnvs)) return;
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  if (u >= ucount) return;
  const int live_f = active_idx == 0 ? kFLiveAccelObs0 : kFLiveAccelObs1;
  const int live_b = active_idx == 0 ? kBLiveAccelObs0 : kBLiveAccelObs1;
  const int accel_stage = active_idx == 0 ? 0 : 1;
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int row = row_local(a, e, u);
  const int center_limit = min(min(max(ucount, 1), max(gu, 0)), kSourceBlockMaxThreads);
  const float assoc_bonus = fmaxf(fp(a, kFpBaselineAssocBonus, 0.3f), 0.0f);
  const float gain = fmaxf(fp(a, kFpBaselineAccelGain, 2.0f), 0.0f);
  __shared__ float scratch_a[kSourceBlockMaxThreads];
  __shared__ float scratch_b[kSourceBlockMaxThreads];
  __shared__ float scratch_c[kSourceBlockMaxThreads];
  __shared__ int scratch_i[kSourceBlockMaxThreads];
  __shared__ int selected[kSourceBlockMaxThreads];

  if (threadIdx.x == 0) {
    for (int r = 0; r < center_limit; ++r) selected[r] = -1;
  }
  __syncthreads();

  for (int rank = 0; rank < center_limit; ++rank) {
    float local_best_score = -3.402823466e38f;
    int local_best_g = -1;
    for (int g = threadIdx.x; g < gu; g += blockDim.x) {
      const int mask_idx = row * gu + g;
      if (!a.b[live_b + 0][mask_idx]) continue;
      bool used = false;
      for (int r = 0; r < rank; ++r) used = used || selected[r] == g;
      if (used) continue;
      const float* tok = a.f[live_f + 2] + static_cast<int64_t>(mask_idx) * kAccelGuTokenDim;
      const float weight = observable_cluster_token_weight(a, accel_stage, e, u, g, tok, assoc_bonus);
      float diversity = 1.0f;
      if (rank > 0) {
        float best_d2 = 3.402823466e38f;
        for (int r = 0; r < rank; ++r) {
          const int sg = selected[r];
          if (sg < 0) continue;
          const float* stok = a.f[live_f + 2] + static_cast<int64_t>(row * gu + sg) * kAccelGuTokenDim;
          const float dx = tok[kAccelGuRelX] - stok[kAccelGuRelX];
          const float dy = tok[kAccelGuRelY] - stok[kAccelGuRelY];
          best_d2 = fminf(best_d2, dx * dx + dy * dy);
        }
        diversity = 0.25f + sqrtf(fmaxf(best_d2, 0.0f));
      }
      const float score = weight * diversity;
      if (score > local_best_score || (score == local_best_score && (local_best_g < 0 || g < local_best_g))) {
        local_best_score = score;
        local_best_g = g;
      }
    }
    float best_score = -3.402823466e38f;
    int best_g = -1;
    block_argmax_min_index(local_best_score, local_best_g, scratch_a, scratch_i, &best_score, &best_g);
    if (threadIdx.x == 0) selected[rank] = best_g;
    __syncthreads();
    if (selected[rank] < 0) break;
  }

  int selected_count = 0;
  for (int r = 0; r < center_limit; ++r) {
    if (selected[r] >= 0) ++selected_count;
  }
  int target_rank = -1;
  if (selected_count > 0) {
    float best_d2 = 3.402823466e38f;
    for (int r = 0; r < selected_count; ++r) {
      const int sg = selected[r];
      const float* stok = a.f[live_f + 2] + static_cast<int64_t>(row * gu + sg) * kAccelGuTokenDim;
      const float d2 = stok[kAccelGuRelX] * stok[kAccelGuRelX] + stok[kAccelGuRelY] * stok[kAccelGuRelY];
      if (d2 < best_d2 || (d2 == best_d2 && r < target_rank)) {
        best_d2 = d2;
        target_rank = r;
      }
    }
  }

  float local_cluster_w = 0.0f;
  float local_cluster_x = 0.0f;
  float local_cluster_y = 0.0f;
  if (target_rank >= 0) {
    for (int g = threadIdx.x; g < gu; g += blockDim.x) {
      const int mask_idx = row * gu + g;
      if (!a.b[live_b + 0][mask_idx]) continue;
      const float* tok = a.f[live_f + 2] + static_cast<int64_t>(mask_idx) * kAccelGuTokenDim;
      int nearest_rank = 0;
      float nearest_d2 = 3.402823466e38f;
      for (int r = 0; r < selected_count; ++r) {
        const int sg = selected[r];
        const float* stok = a.f[live_f + 2] + static_cast<int64_t>(row * gu + sg) * kAccelGuTokenDim;
        const float dx = tok[kAccelGuRelX] - stok[kAccelGuRelX];
        const float dy = tok[kAccelGuRelY] - stok[kAccelGuRelY];
        const float d2 = dx * dx + dy * dy;
        if (d2 < nearest_d2 || (d2 == nearest_d2 && r < nearest_rank)) {
          nearest_d2 = d2;
          nearest_rank = r;
        }
      }
      if (nearest_rank != target_rank) continue;
      const float weight = observable_cluster_token_weight(a, accel_stage, e, u, g, tok, assoc_bonus);
      local_cluster_w += weight;
      local_cluster_x += tok[kAccelGuRelX] * weight;
      local_cluster_y += tok[kAccelGuRelY] * weight;
    }
  }

  const float cluster_w = block_sum_float(local_cluster_w, scratch_a);
  const float cluster_x = block_sum_float(local_cluster_x, scratch_b);
  const float cluster_y = block_sum_float(local_cluster_y, scratch_c);
  float ax = 0.0f;
  float ay = 0.0f;
  if (cluster_w > kNormDenomEps) {
    ax = cluster_x / cluster_w * gain;
    ay = cluster_y / cluster_w * gain;
  } else if (target_rank >= 0) {
    const int target_g = selected[target_rank];
    const float* target_tok = a.f[live_f + 2] + static_cast<int64_t>(row * gu + target_g) * kAccelGuTokenDim;
    ax = target_tok[kAccelGuRelX] * gain;
    ay = target_tok[kAccelGuRelY] * gain;
  }

  const int nbr_width = max(ucount - 1, 0);
  const float repulse_gain = fp(a, kFpBaselineRepulseGain, 0.0f);
  const float repulse_radius = fp(a, kFpDSafe, 0.0f) * fp(a, kFpBaselineRepulseRadiusFactor, 1.5f);
  float local_rx = 0.0f;
  float local_ry = 0.0f;
  if (repulse_gain > 0.0f && repulse_radius > 0.0f && nbr_width > 0) {
    const float map_size = positive_config_scale(fp(a, kFpMapSize, 1.0f));
    for (int n = threadIdx.x; n < nbr_width; n += blockDim.x) {
      const int nidx = row * nbr_width + n;
      if (!a.b[live_b + 1][nidx]) continue;
      const float* peer = a.f[live_f + 3] + static_cast<int64_t>(nidx) * kAccelPeerTokenDim;
      const float relx = peer[kAccelPeerRelX];
      const float rely = peer[kAccelPeerRelY];
      const float dist_norm = sqrtf(relx * relx + rely * rely);
      const float dist = dist_norm * map_size;
      if (!(dist > kDynamicsDenomEps && dist < repulse_radius)) continue;
      const float strength = (1.0f / dynamics_denominator(dist) - 1.0f / repulse_radius);
      local_rx += (relx / geometry_denominator(dist_norm)) * strength;
      local_ry += (rely / geometry_denominator(dist_norm)) * strength;
    }
  }
  const float rx = block_sum_float(local_rx, scratch_a);
  const float ry = block_sum_float(local_ry, scratch_b);
  if (repulse_gain > 0.0f && repulse_radius > 0.0f && nbr_width > 0) {
    ax += repulse_gain * rx;
    ay += repulse_gain * ry;
  }

  if (threadIdx.x == 0) {
    ax += baseline_energy_term_component(a, live_f, row, 0);
    ay += baseline_energy_term_component(a, live_f, row, 1);
    project_unit_action(&ax, &ay);
    a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 0] = ax;
    a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 1] = ay;
    if (has_f(a, kFLiveAccelLatentAction)) {
      a.f[kFLiveAccelLatentAction][(e * ucount + u) * 2 + 0] = 0.0f;
      a.f[kFLiveAccelLatentAction][(e * ucount + u) * 2 + 1] = 0.0f;
    }
    if (has_f(a, kFLiveAccelOldLogprob)) a.f[kFLiveAccelOldLogprob][e * ucount + u] = 0.0f;
    if (e == 0 && u == 0 && has_i(a, kIMarker)) a.i[kIMarker][0] = 22;
  }
}

__global__ void baseline_accel_live_kernel(int64_t active_idx, int64_t source_mode64) {
  const PackedAbi& a = cLiveAbi;
  const int e = blockIdx.x;
  const int u = blockIdx.y;
  if (e >= ip(a, kParamNumEnvs)) return;
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  if (u >= ucount) return;
  const int mode = static_cast<int>(source_mode64);
  const int live_f = active_idx == 0 ? kFLiveAccelObs0 : kFLiveAccelObs1;
  const int live_b = active_idx == 0 ? kBLiveAccelObs0 : kBLiveAccelObs1;
  const int accel_stage = active_idx == 0 ? 0 : 1;
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int row = row_local(a, e, u);
  __shared__ float scratch_a[kSourceBlockMaxThreads];
  __shared__ float scratch_b[kSourceBlockMaxThreads];
  __shared__ float scratch_c[kSourceBlockMaxThreads];
  __shared__ int scratch_i[kSourceBlockMaxThreads];
  float local_sum_w = 0.0f;
  float local_vx = 0.0f;
  float local_vy = 0.0f;

  if (mode == kSourceTopologyDppAccel) {
    const int candidate_count = min(max(static_cast<int>(ip(a, kParamTopologyDppAccelNumCandidates, 9)), 1), kSourceBlockMaxThreads);
    const float step = clampf_device(fp(a, kFpTopologyDppAccelStepScale, 0.6f), 0.0f, 1.0f);
    const float gain = fmaxf(fp(a, kFpBaselineAccelGain, 2.0f), 0.0f);
    const float access_w = fmaxf(fp(a, kFpTopologyDppAccessWeight, 1.0f), 0.0f);
    const float backhaul_w = fmaxf(fp(a, kFpTopologyDppBackhaulWeight, 1.0f), 0.0f);
    const float mobility_w = fmaxf(fp(a, kFpTopologyDppMobilityWeight, 0.75f), 0.0f);
    const float accel_cost = fmaxf(fp(a, kFpTopologyDppAccelCost, 0.08f), 0.0f);
    const float smooth_w = fmaxf(fp(a, kFpTopologyDppSmoothness, 0.05f), 0.0f);
    const float safety_w = fmaxf(fp(a, kFpTopologyDppAccelSafetyWeight, 4.0f), 0.0f);
    const float role_w = fmaxf(fp(a, kFpTopologyDppAccelRoleWeight, 0.35f), 0.0f);
    const float dist_penalty = fmaxf(fp(a, kFpTopologyDppDistPenalty, 0.1f), 0.0f);
    const float assoc_bonus = fmaxf(fp(a, kFpBaselineAssocBonus, 0.3f), 0.0f);
    const float map_size = positive_config_scale(fp(a, kFpMapSize, 1.0f));
    const float tau = fmaxf(fp(a, kFpTau0, 1.0f), 0.0f);
    const float amax = positive_config_scale(fp(a, kFpAccelAMax, 1.0f));
    const float vmax = positive_config_scale(fp(a, kFpVMax, 1.0f));
    const float* ego = a.f[live_f + 0] + static_cast<int64_t>(row) * kAccelEgoDim;
    float role_x = ego[kAccelEgoX] - 0.5f;
    float role_y = ego[kAccelEgoY] - 0.5f;
    float role_norm = sqrtf(role_x * role_x + role_y * role_y);
    if (role_norm <= 1.0e-3f && ucount > 0) {
      const float theta = 6.28318530717958647692f * static_cast<float>(u) / fmaxf(static_cast<float>(ucount), 1.0f);
      role_x = cosf(theta);
      role_y = sinf(theta);
      role_norm = 1.0f;
    }
    if (role_norm > kNormDenomEps) {
      role_x /= role_norm;
      role_y /= role_norm;
    }
    const int cand = threadIdx.x;
    float local_score = -3.402823466e38f;
    int local_candidate = -1;
    if (cand < candidate_count) {
      float raw_x = 0.0f, raw_y = 0.0f, ax = 0.0f, ay = 0.0f;
      topology_dpp_accel_candidate_action(cand, candidate_count, step, gain, &raw_x, &raw_y, &ax, &ay);
      const float velx = ego[kAccelEgoVx] * vmax;
      const float vely = ego[kAccelEgoVy] * vmax;
      const float delta_x = (velx * tau + 0.5f * ax * amax * tau * tau) / map_size;
      const float delta_y = (vely * tau + 0.5f * ay * amax * tau * tau) / map_size;
      float access_term = 0.0f;
      float pressure_sum = 0.0f;
      float target_x = 0.0f;
      float target_y = 0.0f;
      for (int g = 0; g < gu; ++g) {
        const int mask_idx = row * gu + g;
        if (!a.b[live_b + 0][mask_idx]) continue;
        const float* tok = a.f[live_f + 2] + static_cast<int64_t>(mask_idx) * kAccelGuTokenDim;
        const float relx = tok[kAccelGuRelX] - delta_x;
        const float rely = tok[kAccelGuRelY] - delta_y;
        const float dist = sqrtf(relx * relx + rely * rely);
        const float eta_obs = lyapunov_eta_for_accel_slot(a, e, u, g, tok, accel_stage);
        const float range_norm = clampf_device(dist / 0.5f, 0.0f, 1.0f);
        const float eta_new = clampf_device(0.8f * (1.0f - 0.9f * range_norm) + 0.1f, 0.1f, 1.0f);
        const float eta_blend = fmaxf(0.4f * eta_obs + 0.6f * eta_new, 0.0f);
        const float demand = lyapunov_queue_from_accel_token(tok);
        const float rate_proxy = clampf_device(0.5f + eta_blend, 0.0f, 2.0f);
        const float prev_assoc = clampf_device(tok[kAccelGuPreOwnerIsEgo], 0.0f, 1.0f);
        const float service_pressure = demand * rate_proxy * (1.0f + assoc_bonus * prev_assoc);
        const float slot_score = service_pressure - dist_penalty * dist;
        access_term += fmaxf(slot_score, 0.0f) * rate_proxy;
        pressure_sum += service_pressure;
        target_x += relx * service_pressure;
        target_y += rely * service_pressure;
      }
      float mobility_term = 0.0f;
      if (pressure_sum > kNormDenomEps) {
        float desired_x = target_x / pressure_sum * gain;
        float desired_y = target_y / pressure_sum * gain;
        project_unit_action(&desired_x, &desired_y);
        mobility_term = ax * desired_x + ay * desired_y;
      }
      float safety_penalty = 0.0f;
      const int nbr_width = max(ucount - 1, 0);
      const float d_safe_norm = fp(a, kFpDSafe, 0.0f) / map_size;
      const float d_alert_norm = fmaxf(fp(a, kFpAvoidanceAlertFactor, 1.5f) * d_safe_norm, d_safe_norm);
      if (d_alert_norm > kDynamicsDenomEps && nbr_width > 0) {
        for (int n = 0; n < nbr_width; ++n) {
          const int nidx = row * nbr_width + n;
          if (!a.b[live_b + 1][nidx]) continue;
          const float* peer = a.f[live_f + 3] + static_cast<int64_t>(nidx) * kAccelPeerTokenDim;
          const float pred_x = peer[kAccelPeerRelX] + delta_x;
          const float pred_y = peer[kAccelPeerRelY] + delta_y;
          const float pred_dist = sqrtf(pred_x * pred_x + pred_y * pred_y);
          const float alert_risk = fmaxf(d_alert_norm - pred_dist, 0.0f) / d_alert_norm;
          const float unsafe_risk = d_safe_norm > kDynamicsDenomEps
              ? fmaxf(d_safe_norm - pred_dist, 0.0f) / d_safe_norm
              : 0.0f;
          const float closing = fmaxf(peer[kAccelPeerClosingSpeed], 0.0f);
          safety_penalty += alert_risk * alert_risk * (1.0f + closing) + 4.0f * unsafe_risk * unsafe_risk;
        }
      }
      float backhaul_term = 0.0f;
      const int sat_width = max(static_cast<int>(ip(a, kParamAccelSatWidth, ip(a, kParamSatsObsMax, 0))), 0);
      const float own_q = lyapunov_steps_from_log1p(ego[kAccelEgoUavQueueSteps]);
      for (int s = 0; s < sat_width; ++s) {
        const float* st = a.f[live_f + 4] + (static_cast<int64_t>(row) * sat_width + s) * kAccelSatTokenDim;
        if (st[kAccelSatVisibleFlag] <= 0.5f || st[kAccelSatValidFlag] <= 0.5f) continue;
        const float sat_q = lyapunov_steps_from_log1p(st[kAccelSatQueueSteps]);
        const float gap = fmaxf(own_q - sat_q, 0.0f);
        const float se = fmaxf(st[kAccelSatBackhaulSeRef], 0.0f);
        const float load = fmaxf(st[kAccelSatLastSelectedLoadFrac], 0.0f);
        const float doppler_margin = 1.0f - clampf_device(st[kAccelSatDopplerAbsRatio], 0.0f, 1.0f);
        float se_abs = 0.0f, queue_unused = 0.0f, relay = 0.0f;
        lyapunov_sat_profile_from_values(se, sat_q, load, doppler_margin, &se_abs, &queue_unused, &relay);
        const float sx = st[kAccelSatRelX];
        const float sy = st[kAccelSatRelY];
        const float sat_xy_norm = sqrtf(sx * sx + sy * sy);
        const float align = sat_xy_norm > kNormDenomEps
            ? clampf_device(1.0f + 0.25f * (ax * sx + ay * sy) / sat_xy_norm, 0.75f, 1.25f)
            : 1.0f;
        backhaul_term += gap * relay * align;
      }
      const float last_x = ego[kAccelEgoLastExecAccelX];
      const float last_y = ego[kAccelEgoLastExecAccelY];
      const float reg =
          accel_cost * (raw_x * raw_x + raw_y * raw_y)
          + smooth_w * ((ax - last_x) * (ax - last_x) + (ay - last_y) * (ay - last_y));
      const float role_term = pressure_sum * (ax * role_x + ay * role_y);
      local_score =
          access_w * access_term
          + backhaul_w * backhaul_term
          + mobility_w * mobility_term
          + role_w * role_term
          - reg
          - safety_w * safety_penalty * (1.0f + pressure_sum);
      local_candidate = cand;
    }
    float best_score = -3.402823466e38f;
    int best_candidate = -1;
    block_argmax_min_index(local_score, local_candidate, scratch_a, scratch_i, &best_score, &best_candidate);
    if (threadIdx.x == 0) {
      float raw_x = 0.0f, raw_y = 0.0f, ax = 0.0f, ay = 0.0f;
      topology_dpp_accel_candidate_action(max(best_candidate, 0), candidate_count, step, gain, &raw_x, &raw_y, &ax, &ay);
      const int nbr_width = max(ucount - 1, 0);
      const float repulse_gain = fp(a, kFpBaselineRepulseGain, 0.0f);
      const float repulse_radius = fp(a, kFpDSafe, 0.0f) * fp(a, kFpBaselineRepulseRadiusFactor, 1.5f);
      float rx = 0.0f;
      float ry = 0.0f;
      if (repulse_gain > 0.0f && repulse_radius > 0.0f && nbr_width > 0) {
        for (int n = 0; n < nbr_width; ++n) {
          const int nidx = row * nbr_width + n;
          if (!a.b[live_b + 1][nidx]) continue;
          const float* peer = a.f[live_f + 3] + static_cast<int64_t>(nidx) * kAccelPeerTokenDim;
          const float relx = peer[kAccelPeerRelX];
          const float rely = peer[kAccelPeerRelY];
          const float relvx = peer[kAccelPeerRelVx];
          const float relvy = peer[kAccelPeerRelVy];
          const float dist_norm = sqrtf(relx * relx + rely * rely);
          const float dist = dist_norm * map_size;
          if (!(dist > kDynamicsDenomEps && dist < repulse_radius)) continue;
          const float dirx = relx / geometry_denominator(dist_norm);
          const float diry = rely / geometry_denominator(dist_norm);
          const float approach_speed = relvx * dirx + relvy * diry;
          const float strength = (1.0f / dynamics_denominator(dist) - 1.0f / repulse_radius)
              + (approach_speed < 0.0f ? -approach_speed : 0.0f);
          rx += dirx * strength;
          ry += diry * strength;
        }
        ax += repulse_gain * rx;
        ay += repulse_gain * ry;
      }
      ax += baseline_energy_term_component(a, live_f, row, 0);
      ay += baseline_energy_term_component(a, live_f, row, 1);
      project_unit_action(&ax, &ay);
      a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 0] = ax;
      a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 1] = ay;
    }
  } else if (mode == kSourceLyapunov) {
    const float gain = fp(a, kFpBaselineAccelGain, 2.0f);
    const float urgency_alpha = fmaxf(fp(a, kFpBaselineLyapunovUrgencyAlpha, 1.0f), 0.0f);
    const int t_now = state_int_for_env(a, kIStateT, e, 0);
    const int nbr_width = max(ucount - 1, 0);
    for (int g = threadIdx.x; g < gu; g += blockDim.x) {
      const int mask_idx = row * gu + g;
      const int64_t st = lyapunov_state_offset(a, e, u, g);
      if (t_now == 0) {
        if (has_f(a, kFLyapunovPressureEma)) a.f[kFLyapunovPressureEma][st] = 0.0f;
        if (has_f(a, kFLyapunovVirtualQueue)) a.f[kFLyapunovVirtualQueue][st] = 0.0f;
        if (has_f(a, kFLyapunovServiceEst)) a.f[kFLyapunovServiceEst][st] = 0.0f;
      }
      if (!a.b[live_b + 0][mask_idx]) {
        if (has_f(a, kFLyapunovInstantPressure)) a.f[kFLyapunovInstantPressure][st] = 0.0f;
        continue;
      }
      const float* tok = a.f[live_f + 2] + static_cast<int64_t>(mask_idx) * kAccelGuTokenDim;
      const float q = lyapunov_queue_from_accel_token(tok);
      const float se = lyapunov_eta_for_accel_slot(a, e, u, g, tok, accel_stage);
      float instant = q * (0.5f + se);
      instant = fmaxf(instant, 0.0f);
      if (has_f(a, kFLyapunovInstantPressure)) a.f[kFLyapunovInstantPressure][st] = instant;
      float w = instant;
      if (urgency_alpha > 0.0f && nbr_width > 0) {
        const float dist_gu = sqrtf(tok[kAccelGuRelX] * tok[kAccelGuRelX] + tok[kAccelGuRelY] * tok[kAccelGuRelY]);
        float min_nbr_dist = 3.402823466e38f;
        for (int n = 0; n < nbr_width; ++n) {
          const int nidx = row * nbr_width + n;
          if (!a.b[live_b + 1][nidx]) continue;
          const float* peer = a.f[live_f + 3] + static_cast<int64_t>(nidx) * kAccelPeerTokenDim;
          const float dx = tok[kAccelGuRelX] + peer[kAccelPeerRelX];
          const float dy = tok[kAccelGuRelY] + peer[kAccelPeerRelY];
          min_nbr_dist = fminf(min_nbr_dist, sqrtf(dx * dx + dy * dy));
        }
        if (min_nbr_dist < 1.0e30f) {
          const float responsibility = clampf_device(expf(urgency_alpha * (min_nbr_dist - dist_gu)), 0.0f, 1.0f);
          w *= responsibility;
        }
      }
      if (w <= 0.0f) continue;
      const float move_w = w;
      local_vx += tok[kAccelGuRelX] * move_w;
      local_vy += tok[kAccelGuRelY] * move_w;
      local_sum_w += move_w;
    }
    const float sum_w = block_sum_float(local_sum_w, scratch_a);
    const float vx = block_sum_float(local_vx, scratch_b);
    const float vy = block_sum_float(local_vy, scratch_c);
    float ax = 0.0f;
    float ay = 0.0f;
    if (sum_w > kNormDenomEps) {
      const float map_size = positive_config_scale(fp(a, kFpMapSize, 1.0f));
      const float ex = (vx / sum_w) * map_size;
      const float ey = (vy / sum_w) * map_size;
      const float dist = sqrtf(ex * ex + ey * ey);
      const float* ego = a.f[live_f + 0] + static_cast<int64_t>(row) * kAccelEgoDim;
      const float velx = ego[kAccelEgoVx] * fp(a, kFpVMax, 1.0f);
      const float vely = ego[kAccelEgoVy] * fp(a, kFpVMax, 1.0f);
      const float speed = sqrtf(velx * velx + vely * vely);
      const float stop_radius = fmaxf(fp(a, kFpBaselineClusterStopRadius, 20.0f), 0.0f);
      const float speed_tol = fmaxf(fp(a, kFpBaselineClusterSpeedTol, 2.0f), 0.0f);
      const float slow_radius = fmaxf(fp(a, kFpBaselineClusterSlowRadius, 120.0f), stop_radius + kDynamicsDenomEps);
      const float cruise = fminf(fmaxf(fp(a, kFpBaselineClusterCruiseSpeed, fp(a, kFpUavOptSpeed, 0.0f)), 0.0f), fp(a, kFpVMax, 1.0f));
      const float vel_gain = fmaxf(fp(a, kFpBaselineClusterVelGain, 1.0f), 0.0f) * gain;
      float desired_x = 0.0f;
      float desired_y = 0.0f;
      if (dist > stop_radius) {
        const float desired_speed = cruise * fminf(dist / dynamics_denominator(slow_radius), 1.0f);
        desired_x = ex / dynamics_denominator(dist) * desired_speed;
        desired_y = ey / dynamics_denominator(dist) * desired_speed;
      }
      ax = vel_gain * (desired_x - velx) / dynamics_denominator(fp(a, kFpTau0, 1.0f)) / positive_config_scale(fp(a, kFpAccelAMax, 1.0f));
      ay = vel_gain * (desired_y - vely) / dynamics_denominator(fp(a, kFpTau0, 1.0f)) / positive_config_scale(fp(a, kFpAccelAMax, 1.0f));
      if (dist <= stop_radius && speed <= speed_tol) { ax = 0.0f; ay = 0.0f; }
    }
    const float repulse_gain = fp(a, kFpBaselineRepulseGain, 0.0f);
    const float repulse_radius = fp(a, kFpDSafe, 0.0f) * fp(a, kFpBaselineRepulseRadiusFactor, 1.5f);
    float local_rx = 0.0f;
    float local_ry = 0.0f;
    if (repulse_gain > 0.0f && repulse_radius > 0.0f && nbr_width > 0) {
      const float map_size = positive_config_scale(fp(a, kFpMapSize, 1.0f));
      for (int n = threadIdx.x; n < nbr_width; n += blockDim.x) {
        const int nidx = row * nbr_width + n;
        if (!a.b[live_b + 1][nidx]) continue;
        const float* peer = a.f[live_f + 3] + static_cast<int64_t>(nidx) * kAccelPeerTokenDim;
        const float relx = peer[kAccelPeerRelX];
        const float rely = peer[kAccelPeerRelY];
        const float relvx = peer[kAccelPeerRelVx];
        const float relvy = peer[kAccelPeerRelVy];
        const float dist_norm = sqrtf(relx * relx + rely * rely);
        const float dist = dist_norm * map_size;
        if (!(dist > kDynamicsDenomEps && dist < repulse_radius)) continue;
        const float dirx = relx / geometry_denominator(dist_norm);
        const float diry = rely / geometry_denominator(dist_norm);
        const float approach_speed = relvx * dirx + relvy * diry;
        const float strength = (1.0f / dynamics_denominator(dist) - 1.0f / repulse_radius) +
            (approach_speed < 0.0f ? -approach_speed : 0.0f);
        local_rx += dirx * strength;
        local_ry += diry * strength;
      }
    }
    const float rx = block_sum_float(local_rx, scratch_a);
    const float ry = block_sum_float(local_ry, scratch_b);
    if (repulse_gain > 0.0f && repulse_radius > 0.0f && nbr_width > 0) {
      ax += repulse_gain * rx;
      ay += repulse_gain * ry;
    }
    if (threadIdx.x == 0) {
      ax += baseline_energy_term_component(a, live_f, row, 0);
      ay += baseline_energy_term_component(a, live_f, row, 1);
      const float norm = sqrtf(ax * ax + ay * ay);
      const float scale = 1.0f / fmaxf(norm, 1.0f);
      a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 0] = ax * scale;
      a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 1] = ay * scale;
    }
  } else if (mode == kSourceRandom) {
    if (threadIdx.x == 0) {
      float ax = 2.0f * source_hash01(a, e, u, 0, 301) - 1.0f;
      float ay = 2.0f * source_hash01(a, e, u, 1, 307) - 1.0f;
      const float norm = sqrtf(ax * ax + ay * ay);
      if (norm > 1.0f) {
        ax /= norm;
        ay /= norm;
      }
      a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 0] = ax;
      a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 1] = ay;
    }
  } else if (threadIdx.x == 0) {
    a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 0] = 0.0f;
    a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 1] = 0.0f;
  }

  if (threadIdx.x == 0) {
    if (has_f(a, kFLiveAccelLatentAction)) {
      a.f[kFLiveAccelLatentAction][(e * ucount + u) * 2 + 0] = 0.0f;
      a.f[kFLiveAccelLatentAction][(e * ucount + u) * 2 + 1] = 0.0f;
    }
    if (has_f(a, kFLiveAccelOldLogprob)) a.f[kFLiveAccelOldLogprob][e * ucount + u] = 0.0f;
    if (e == 0 && u == 0 && has_i(a, kIMarker)) a.i[kIMarker][0] = 21;
  }
}

__global__ void queue_aware_accel_live_kernel(int64_t active_idx) {
  const PackedAbi& a = cLiveAbi;
  const int e = blockIdx.x;
  const int u = blockIdx.y;
  if (e >= ip(a, kParamNumEnvs)) return;
  const int live_f = active_idx == 0 ? kFLiveAccelObs0 : kFLiveAccelObs1;
  const int live_b = active_idx == 0 ? kBLiveAccelObs0 : kBLiveAccelObs1;
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  if (u >= ucount) return;
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int row = row_local(a, e, u);
  const float gain = fp(a, kFpBaselineAccelGain, 2.0f);
  const float assoc_bonus = fp(a, kFpBaselineAssocBonus, 0.3f);
  __shared__ float scratch_a[kSourceBlockMaxThreads];
  __shared__ float scratch_b[kSourceBlockMaxThreads];
  __shared__ float scratch_c[kSourceBlockMaxThreads];
  float local_sum_w = 0.0f;
  float local_vx = 0.0f;
  float local_vy = 0.0f;
  for (int g = threadIdx.x; g < gu; g += blockDim.x) {
    const int mask_idx = row * gu + g;
    const bool visible = a.b[live_b + 0][mask_idx];
    if (!visible) continue;
    const float* tok = a.f[live_f + 2] + static_cast<int64_t>(mask_idx) * kAccelGuTokenDim;
    const float q = fmaxf(tok[kAccelGuQueueSteps] + tok[kAccelGuExpectedArrivalSteps] - tok[kAccelGuLastOutflowSteps], 0.0f) + tok[kAccelGuLastDropSteps];
    const float se = fmaxf(tok[kAccelGuAccessSeRef], 0.0f);
    const float owner = tok[kAccelGuPreOwnerIsEgo];
    float w = q * (0.5f + se);
    if (assoc_bonus > 0.0f) w *= (1.0f + assoc_bonus * owner);
    if (w <= 0.0f) continue;
    local_vx += tok[kAccelGuRelX] * w;
    local_vy += tok[kAccelGuRelY] * w;
    local_sum_w += w;
  }
  const float sum_w = block_sum_float(local_sum_w, scratch_a);
  const float vx = block_sum_float(local_vx, scratch_b);
  const float vy = block_sum_float(local_vy, scratch_c);
  float ax = 0.0f;
  float ay = 0.0f;
  if (sum_w > kNormDenomEps) {
    ax = vx / sum_w * gain;
    ay = vy / sum_w * gain;
  }
  const int nbr_width = max(ucount - 1, 0);
  const float repulse_gain = fp(a, kFpBaselineRepulseGain, 0.0f);
  const float repulse_radius = fp(a, kFpDSafe, 0.0f) * fp(a, kFpBaselineRepulseRadiusFactor, 1.5f);
  float local_rx = 0.0f;
  float local_ry = 0.0f;
  if (repulse_gain > 0.0f && repulse_radius > 0.0f && nbr_width > 0) {
    const float map_size = positive_config_scale(fp(a, kFpMapSize, 1.0f));
    for (int n = threadIdx.x; n < nbr_width; n += blockDim.x) {
      const int nidx = row * nbr_width + n;
      if (!a.b[live_b + 1][nidx]) continue;
      const float* peer = a.f[live_f + 3] + static_cast<int64_t>(nidx) * kAccelPeerTokenDim;
      const float relx = peer[kAccelPeerRelX];
      const float rely = peer[kAccelPeerRelY];
      const float dist_norm = sqrtf(relx * relx + rely * rely);
      const float dist = dist_norm * map_size;
      if (!(dist > kDynamicsDenomEps && dist < repulse_radius)) continue;
      const float strength = (1.0f / dynamics_denominator(dist) - 1.0f / repulse_radius);
      local_rx += (relx / geometry_denominator(dist_norm)) * strength;
      local_ry += (rely / geometry_denominator(dist_norm)) * strength;
    }
  }
  const float rx = block_sum_float(local_rx, scratch_a);
  const float ry = block_sum_float(local_ry, scratch_b);
  if (repulse_gain > 0.0f && repulse_radius > 0.0f && nbr_width > 0) {
    ax += repulse_gain * rx;
    ay += repulse_gain * ry;
  }
  if (threadIdx.x == 0) {
    ax += baseline_energy_term_component(a, live_f, row, 0);
    ay += baseline_energy_term_component(a, live_f, row, 1);
    a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 0] = clampf_device(ax, -1.0f, 1.0f);
    a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 1] = clampf_device(ay, -1.0f, 1.0f);
    if (has_f(a, kFLiveAccelLatentAction)) {
      a.f[kFLiveAccelLatentAction][(e * ucount + u) * 2 + 0] = 0.0f;
      a.f[kFLiveAccelLatentAction][(e * ucount + u) * 2 + 1] = 0.0f;
    }
    if (has_f(a, kFLiveAccelOldLogprob)) a.f[kFLiveAccelOldLogprob][e * ucount + u] = 0.0f;
    if (e == 0 && has_i(a, kIMarker)) a.i[kIMarker][0] = 11;
  }
}

__global__ void queue_aware_bw_live_kernel() {
  const PackedAbi& a = cLiveAbi;
  const int e = blockIdx.x;
  const int u = blockIdx.y;
  if (e >= ip(a, kParamNumEnvs)) return;
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  if (u >= ucount) return;
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int row = e * ucount + u;
  constexpr int bw_stage = 3;
  const bool has_stage_gain = has_f(a, stage_f(bw_stage, kSfAccessGainMatrix));
  const bool has_prev_assoc = has_l(a, stage_l(bw_stage, kSlPrevAssociation));
  const float access_noise_ref = accel_access_noise_ref(a);
  const float assoc_bonus = fp(a, kFpBaselineAssocBonus, 0.3f);
  __shared__ float float_scratch[kSourceBlockMaxThreads];
  __shared__ int int_scratch[kSourceBlockMaxThreads];
  float local_denom = 0.0f;
  int local_valid = 0;
  for (int g = threadIdx.x; g < gu; g += blockDim.x) {
    const int idx = row * gu + g;
    const bool valid = a.b[kBLiveBwObs + 1][idx] && a.b[kBLiveBwObs + 2][idx];
    if (!valid) continue;
    ++local_valid;
    const float* tok = a.f[kFLiveBwObs + 2] + idx * kBwGuTokenDim;
    const float q = fmaxf(tok[kBwGuQueueFill], 0.0f);
    float eta = fmaxf(tok[kBwGuAccessRateFullBwRefSteps], 0.0f);
    if (has_stage_gain) {
      const float gain = a.f[stage_f(bw_stage, kSfAccessGainMatrix)][(e * gu + g) * ucount + u];
      eta = fmaxf(accel_access_se_from_gain(a, gain, access_noise_ref), 0.0f);
    }
    float w = q * (0.5f + eta);
    if (has_prev_assoc && a.l[stage_l(bw_stage, kSlPrevAssociation)][e * gu + g] == u) {
      w *= (1.0f + assoc_bonus);
    }
    local_denom += fmaxf(w, 0.0f);
  }
  const float denom = block_sum_float(local_denom, float_scratch);
  const int valid_count = block_sum_int(local_valid, int_scratch);
  for (int g = threadIdx.x; g < gu; g += blockDim.x) {
    const int idx = row * gu + g;
    const bool valid = a.b[kBLiveBwObs + 1][idx] && a.b[kBLiveBwObs + 2][idx];
    float value = 0.0f;
    if (valid && ip(a, kParamEnableBwAction)) {
      const float* tok = a.f[kFLiveBwObs + 2] + idx * kBwGuTokenDim;
      const float q = fmaxf(tok[kBwGuQueueFill], 0.0f);
      float eta = fmaxf(tok[kBwGuAccessRateFullBwRefSteps], 0.0f);
      if (has_stage_gain) {
        const float gain = a.f[stage_f(bw_stage, kSfAccessGainMatrix)][(e * gu + g) * ucount + u];
        eta = fmaxf(accel_access_se_from_gain(a, gain, access_noise_ref), 0.0f);
      }
      float w = q * (0.5f + eta);
      if (has_prev_assoc && a.l[stage_l(bw_stage, kSlPrevAssociation)][e * gu + g] == u) {
        w *= (1.0f + assoc_bonus);
      }
      value = denom > kNormDenomEps ? fmaxf(w, 0.0f) / denom : 1.0f / fmaxf(static_cast<float>(valid_count), 1.0f);
    }
    a.f[kFLiveBwAction][idx] = value;
    a.f[kFLiveBwRefAction][idx] = value;
    a.f[kFLiveBwFlowProxyOverrideAction][idx] = value;
  }
  if (threadIdx.x == 0) {
    if (has_f(a, kFLiveBwOldLogprob) && u == 0) a.f[kFLiveBwOldLogprob][e] = 0.0f;
    if (has_f(a, kFLiveBwOldLogprobPerAgent)) a.f[kFLiveBwOldLogprobPerAgent][row] = 0.0f;
    if (has_f(a, kFLiveBwEntropyPerAgent)) a.f[kFLiveBwEntropyPerAgent][row] = 0.0f;
    if (has_f(a, kFLiveBwLogprobRawPerAgent)) a.f[kFLiveBwLogprobRawPerAgent][row] = 0.0f;
    if (has_f(a, kFLiveBwEntropyRawPerAgent)) a.f[kFLiveBwEntropyRawPerAgent][row] = 0.0f;
    if (has_f(a, kFLiveBwTau)) a.f[kFLiveBwTau][row] = 0.0f;
    if (has_f(a, kFLiveBwKappa)) a.f[kFLiveBwKappa][row] = 0.0f;
    if (has_l(a, kLLiveBwValidCount)) a.l[kLLiveBwValidCount][row] = static_cast<int64_t>(valid_count);
    if (has_l(a, kLLiveBwLatentCount)) a.l[kLLiveBwLatentCount][row] = static_cast<int64_t>(max(valid_count - 1, 0));
    if (e == 0 && u == 0 && has_i(a, kIMarker)) a.i[kIMarker][0] = 12;
  }
}

__device__ bool dpp_resource_bw_slot_valid(const PackedAbi& a, int row, int g) {
  const int idx = row * static_cast<int>(ip(a, kParamNumGu)) + g;
  return a.b[kBLiveBwObs + 1][idx] && a.b[kBLiveBwObs + 2][idx];
}

__device__ float dpp_resource_bw_slot_score(
    const PackedAbi& a,
    int e,
    int u,
    int g,
    int row,
    int bw_stage,
    bool has_stage_gain,
    bool has_prev_assoc,
    float access_noise_ref) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int idx = row * gu + g;
  const float* tok = a.f[kFLiveBwObs + 2] + static_cast<int64_t>(idx) * kBwGuTokenDim;
  const float q = fmaxf(tok[kBwGuQueueSteps], 0.0f);
  const float expected = fmaxf(tok[kBwGuExpectedArrivalSteps], 0.0f);
  float eta = fmaxf(tok[kBwGuAccessRateFullBwRefSteps], 0.0f);
  if (has_stage_gain) {
    const float gain = a.f[stage_f(bw_stage, kSfAccessGainMatrix)][(e * gu + g) * ucount + u];
    eta = fmaxf(accel_access_se_from_gain(a, gain, access_noise_ref), 0.0f);
  }
  const float assoc_bonus = fp(a, kFpBaselineAssocBonus, 0.3f);
  const float prev_assoc =
      has_prev_assoc && a.l[stage_l(bw_stage, kSlPrevAssociation)][e * gu + g] == u ? 1.0f : 0.0f;
  const float demand_pressure = q + expected;
  const float rate_proxy = clampf_device(0.5f + eta, 0.0f, 2.0f);
  float score = demand_pressure * rate_proxy * (1.0f + assoc_bonus * prev_assoc);
  if (has_f(a, stage_f(bw_stage, kSfGuPos)) && has_f(a, stage_f(bw_stage, kSfUavPos))) {
    const float gx = a.f[stage_f(bw_stage, kSfGuPos)][(e * gu + g) * 2 + 0];
    const float gy = a.f[stage_f(bw_stage, kSfGuPos)][(e * gu + g) * 2 + 1];
    const float ux = a.f[stage_f(bw_stage, kSfUavPos)][(e * ucount + u) * 2 + 0];
    const float uy = a.f[stage_f(bw_stage, kSfUavPos)][(e * ucount + u) * 2 + 1];
    const float dist = sqrtf((gx - ux) * (gx - ux) + (gy - uy) * (gy - uy));
    score -= fmaxf(fp(a, kFpTopologyDppDistPenalty, 0.1f), 0.0f)
        * dist
        / positive_config_scale(fp(a, kFpMapSize, 1.0f));
  }
  return score;
}

__device__ bool dpp_resource_bw_slot_selected(
    const PackedAbi& a,
    int e,
    int u,
    int g,
    int row,
    int bw_stage,
    bool has_stage_gain,
    bool has_prev_assoc,
    float access_noise_ref,
    float my_score) {
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int topk = max(1, min(static_cast<int>(ip(a, kParamTopologyDppGuMaxSelect, 6)), gu));
  int better = 0;
  for (int other = 0; other < gu; ++other) {
    if (other == g || !dpp_resource_bw_slot_valid(a, row, other)) continue;
    const float other_score =
        dpp_resource_bw_slot_score(a, e, u, other, row, bw_stage, has_stage_gain, has_prev_assoc, access_noise_ref);
    if (other_score > my_score || (other_score == my_score && other < g)) {
      ++better;
    }
  }
  return better < topk;
}

__global__ void baseline_bw_live_kernel(int64_t source_mode64) {
  const PackedAbi& a = cLiveAbi;
  const int e = blockIdx.x;
  const int u = blockIdx.y;
  if (e >= ip(a, kParamNumEnvs)) return;
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  if (u >= ucount) return;
  const int mode = static_cast<int>(source_mode64);
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int row = e * ucount + u;
  constexpr int bw_stage = 3;
  constexpr int sat_stage = 2;
  const bool has_stage_gain = has_f(a, stage_f(bw_stage, kSfAccessGainMatrix));
  const bool has_prev_assoc = has_l(a, stage_l(bw_stage, kSlPrevAssociation));
  const float access_noise_ref = accel_access_noise_ref(a);
  const float relay_gate = mode == kSourceLyapunov
      ? lyapunov_relay_gate(a, sat_stage, e, u, row, sat_visible_width(a))
      : 1.0f;
  __shared__ float float_scratch[kSourceBlockMaxThreads];
  __shared__ int int_scratch[kSourceBlockMaxThreads];

  float local_sum = 0.0f;
  float local_max = -3.402823466e38f;
  int local_valid = 0;
  int local_selected = 0;
  for (int g = threadIdx.x; g < gu; g += blockDim.x) {
    const int idx = row * gu + g;
    const bool valid = a.b[kBLiveBwObs + 1][idx] && a.b[kBLiveBwObs + 2][idx];
    if (!valid) continue;
    ++local_valid;
    const float* tok = a.f[kFLiveBwObs + 2] + static_cast<int64_t>(idx) * kBwGuTokenDim;
    const float q = lyapunov_queue_from_bw_token(tok);
    const float q_fill = fmaxf(tok[kBwGuQueueFill], 0.0f);
    float eta = fmaxf(tok[kBwGuAccessRateFullBwRefSteps], 0.0f);
    if (has_stage_gain) {
      const float gain = a.f[stage_f(bw_stage, kSfAccessGainMatrix)][(e * gu + g) * ucount + u];
      eta = fmaxf(accel_access_se_from_gain(a, gain, access_noise_ref), 0.0f);
    }
    float score = 1.0f;
    if (mode == kSourceRandom) {
      score = source_clip_positive(source_hash01(a, e, u, g, 401));
    } else if (mode == kSourceLinkPriority) {
      score = source_clip_positive(eta);
    } else if (mode == kSourceDemandPriority) {
      score = source_clip_positive(q_fill);
    } else if (mode == kSourceLyapunov) {
      eta = lyapunov_eta_for_bw_slot(a, e, u, g, tok, bw_stage);
      const float urgency = fmaxf(q, 0.0f);
      const float service_gain = relay_gate * (0.5f + eta);
      score = source_clip_positive(fmaxf(fp(a, kFpBaselineLyapunovV, 2.0f), 0.0f) * urgency * service_gain);
    } else if (mode == kSourceDppResourceBw) {
      const float dpp_score =
          dpp_resource_bw_slot_score(a, e, u, g, row, bw_stage, has_stage_gain, has_prev_assoc, access_noise_ref);
      const bool selected = dpp_resource_bw_slot_selected(
          a, e, u, g, row, bw_stage, has_stage_gain, has_prev_assoc, access_noise_ref, dpp_score);
      if (selected) {
        const float temp = fmaxf(fp(a, kFpTopologyDppBwTemp, 0.55f), 1.0e-3f);
        score = fmaxf(fp(a, kFpBaselineLyapunovV, 2.0f), 0.0f) * dpp_score / temp;
        local_max = fmaxf(local_max, score);
        ++local_selected;
      } else {
        score = 0.0f;
      }
    }
    if (mode != kSourceDppResourceBw) {
      local_sum += fmaxf(score, 0.0f);
    }
  }
  const int valid_count = block_sum_int(local_valid, int_scratch);
  const int selected_count = block_sum_int(local_selected, int_scratch);
  float denom = 0.0f;
  float max_score = -3.402823466e38f;
  if (mode == kSourceDppResourceBw) {
    max_score = block_max_float(local_max, float_scratch);
    float local_exp_sum = 0.0f;
    if (max_score > -3.0e38f) {
      for (int g = threadIdx.x; g < gu; g += blockDim.x) {
        if (!dpp_resource_bw_slot_valid(a, row, g)) continue;
        const float dpp_score =
            dpp_resource_bw_slot_score(a, e, u, g, row, bw_stage, has_stage_gain, has_prev_assoc, access_noise_ref);
        if (!dpp_resource_bw_slot_selected(
                a, e, u, g, row, bw_stage, has_stage_gain, has_prev_assoc, access_noise_ref, dpp_score)) {
          continue;
        }
        const float temp = fmaxf(fp(a, kFpTopologyDppBwTemp, 0.55f), 1.0e-3f);
        const float logit = fmaxf(fp(a, kFpBaselineLyapunovV, 2.0f), 0.0f) * dpp_score / temp;
        local_exp_sum += expf(logit - max_score);
      }
    }
    denom = block_sum_float(local_exp_sum, float_scratch);
  } else if (mode == kSourceLyapunov) {
    denom = block_sum_float(local_sum, float_scratch);
  } else {
    denom = block_sum_float(local_sum, float_scratch);
  }

  for (int g = threadIdx.x; g < gu; g += blockDim.x) {
    const int idx = row * gu + g;
    const bool valid = a.b[kBLiveBwObs + 1][idx] && a.b[kBLiveBwObs + 2][idx];
    const float* tok = a.f[kFLiveBwObs + 2] + static_cast<int64_t>(idx) * kBwGuTokenDim;
    const float q = lyapunov_queue_from_bw_token(tok);
    const float q_fill = fmaxf(tok[kBwGuQueueFill], 0.0f);
    float eta = fmaxf(tok[kBwGuAccessRateFullBwRefSteps], 0.0f);
    if (has_stage_gain) {
      const float gain = a.f[stage_f(bw_stage, kSfAccessGainMatrix)][(e * gu + g) * ucount + u];
      eta = fmaxf(accel_access_se_from_gain(a, gain, access_noise_ref), 0.0f);
    }
    float value = 0.0f;
    if (valid && ip(a, kParamEnableBwAction)) {
      if (mode == kSourceDppResourceBw) {
        const float dpp_score =
            dpp_resource_bw_slot_score(a, e, u, g, row, bw_stage, has_stage_gain, has_prev_assoc, access_noise_ref);
        const bool selected = dpp_resource_bw_slot_selected(
            a, e, u, g, row, bw_stage, has_stage_gain, has_prev_assoc, access_noise_ref, dpp_score);
        if (selected && selected_count > 0) {
          const float temp = fmaxf(fp(a, kFpTopologyDppBwTemp, 0.55f), 1.0e-3f);
          const float logit = fmaxf(fp(a, kFpBaselineLyapunovV, 2.0f), 0.0f) * dpp_score / temp;
          value = denom > kNormDenomEps ? expf(logit - max_score) / denom : 1.0f / static_cast<float>(selected_count);
          float floor = clampf_device(fp(a, kFpTopologyDppBwFloor, 0.01f), 0.0f, 0.2f);
          floor = fminf(floor, 0.99f / fmaxf(static_cast<float>(selected_count), 1.0f));
          if (floor > 0.0f) {
            value = (1.0f - floor * static_cast<float>(selected_count)) * value + floor;
          }
        }
      } else if (mode == kSourceLyapunov) {
        eta = lyapunov_eta_for_bw_slot(a, e, u, g, tok, bw_stage);
        const float urgency = fmaxf(q, 0.0f);
        const float service_gain = relay_gate * (0.5f + eta);
        const float weight = source_clip_positive(fmaxf(fp(a, kFpBaselineLyapunovV, 2.0f), 0.0f) * urgency * service_gain);
        value = denom > kNormDenomEps ? weight / denom : 1.0f / fmaxf(static_cast<float>(valid_count), 1.0f);
      } else {
        float weight = 1.0f;
        if (mode == kSourceRandom) weight = source_clip_positive(source_hash01(a, e, u, g, 401));
        else if (mode == kSourceLinkPriority) weight = source_clip_positive(eta);
        else if (mode == kSourceDemandPriority) weight = source_clip_positive(q_fill);
        value = denom > kNormDenomEps ? fmaxf(weight, 0.0f) / denom : 1.0f / fmaxf(static_cast<float>(valid_count), 1.0f);
      }
    }
    a.f[kFLiveBwAction][idx] = value;
    a.f[kFLiveBwRefAction][idx] = value;
    a.f[kFLiveBwFlowProxyOverrideAction][idx] = value;
    if (mode == kSourceLyapunov && has_f(a, kFLyapunovServiceEst)) {
      const int64_t st = lyapunov_state_offset(a, e, u, g);
      const float eta_service = valid ? lyapunov_eta_for_bw_slot(a, e, u, g, tok, bw_stage) : 0.0f;
      a.f[kFLyapunovServiceEst][st] =
          fmaxf(fp(a, kFpBaselineLyapunovBwServiceScale, 1.0f), 0.0f) * value * relay_gate * (0.5f + eta_service);
    }
  }
  if (threadIdx.x == 0) {
    if (has_f(a, kFLiveBwOldLogprob) && u == 0) a.f[kFLiveBwOldLogprob][e] = 0.0f;
    if (has_f(a, kFLiveBwOldLogprobPerAgent)) a.f[kFLiveBwOldLogprobPerAgent][row] = 0.0f;
    if (has_f(a, kFLiveBwEntropyPerAgent)) a.f[kFLiveBwEntropyPerAgent][row] = 0.0f;
    if (has_f(a, kFLiveBwLogprobRawPerAgent)) a.f[kFLiveBwLogprobRawPerAgent][row] = 0.0f;
    if (has_f(a, kFLiveBwEntropyRawPerAgent)) a.f[kFLiveBwEntropyRawPerAgent][row] = 0.0f;
    if (has_f(a, kFLiveBwTau)) a.f[kFLiveBwTau][row] = 0.0f;
    if (has_f(a, kFLiveBwKappa)) a.f[kFLiveBwKappa][row] = 0.0f;
    if (has_l(a, kLLiveBwValidCount)) a.l[kLLiveBwValidCount][row] = static_cast<int64_t>(valid_count);
    if (has_l(a, kLLiveBwLatentCount)) a.l[kLLiveBwLatentCount][row] = static_cast<int64_t>(max(valid_count - 1, 0));
    if (e == 0 && u == 0 && has_i(a, kIMarker)) a.i[kIMarker][0] = 22;
  }
}

__device__ int live_sat_candidate_sid(const PackedAbi& a, int stage_slot, int e, int u, int row, int slot, int width) {
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  if (has_l(a, kLLiveSatCandidateIds)) {
    const int64_t sid = a.l[kLLiveSatCandidateIds][static_cast<int64_t>(row) * width + slot];
    return static_cast<int>(sid);
  }
  return static_cast<int>(a.l[stage_l(stage_slot, kSlVisibleIds)][(e * ucount + u) * width + slot]);
}

__device__ float sat_feature(const PackedAbi& a, int row, int slot, int width, int feature) {
  constexpr int kSatTokenDim = 26;
  const int idx = (row * width + slot) * kSatTokenDim;
  if (feature == 7) return a.f[kFLiveSatObs + 3][idx + 21];  // backhaul_se_ref
  if (feature == 8) return a.f[kFLiveSatObs + 3][idx + 0];   // sat_queue_steps
  if (feature == 9) return a.f[kFLiveSatObs + 3][idx + 8];   // last selected load
  if (feature == 10) return a.f[kFLiveSatObs + 3][idx + 9];  // processing capacity
  if (feature == 11) return a.f[kFLiveSatObs + 3][idx + 24]; // last selected by ego
  return 0.0f;
}

__device__ void queue_aware_sat_values(
    const PackedAbi& a,
    int stage_slot,
    int e,
    int u,
    int row,
    int slot,
    int width,
    float* se,
    float* q,
    float* load,
    float* bw,
    float* stay) {
  constexpr int kSatTokenDim = 26;
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int idx = (row * width + slot) * kSatTokenDim;
  const int sid = live_sat_candidate_sid(a, stage_slot, e, u, row, slot, width);
  const bool listed = sid >= 0 && sid < sat;
  const float stay_value = listed && has_f(a, kFStateLastSelectedMaskByUavSat)
      ? a.f[kFStateLastSelectedMaskByUavSat][(e * static_cast<int>(ip(a, kParamNumUav)) + u) * sat + sid]
      : (listed ? a.f[kFLiveSatObs + 3][idx + 24] : 0.0f);
  const float last_count = listed && has_f(a, stage_f(stage_slot, kSfSatLoads))
      ? fmaxf(a.f[stage_f(stage_slot, kSfSatLoads)][e * sat + sid], 0.0f)
      : 0.0f;
  const float projected_count = fmaxf(last_count + (stay_value > 0.5f ? 0.0f : 1.0f), 1.0f);
  const float elevation = listed && has_f(a, stage_f(stage_slot, kSfElevationMatrix))
      ? a.f[stage_f(stage_slot, kSfElevationMatrix)][(e * static_cast<int>(ip(a, kParamNumUav)) + u) * sat + sid]
      : (listed ? sat_elevation_for_stage_us(a, stage_slot, e, u, sid) : 0.0f);
  *se = listed ? fmaxf(sat_score_se_for_stage_us(a, stage_slot, e, u, sid, elevation), 0.0f) : 0.0f;
  *q = listed ? fmaxf(a.f[kFLiveSatObs + 3][idx + 1], 0.0f) : 0.0f;  // Python legacy queue-aware uses queue fill.
  *load = log1pf(projected_count);
  *bw = 1.0f / projected_count;
  *stay = stay_value;
}

__device__ float normalize_masked_value(float value, float min_v, float max_v) {
  const float span = max_v - min_v;
  return (span > kNormDenomEps) ? (value - min_v) / span : 0.0f;
}

__device__ float queue_aware_sat_score_from_values(
    const PackedAbi& a,
    float se,
    float q,
    float load,
    float bw,
    float stay,
    float se_min,
    float se_max,
    float q_min,
    float q_max,
    float load_min,
    float load_max,
    float bw_min,
    float bw_max) {
  const float logit_scale = fmaxf(fp(a, kFpSatLogitScale, 1.0e9f), 0.0f);
  const float score =
      fp(a, kFpBaselineSatSeWeight, 1.0f) * normalize_masked_value(se, se_min, se_max)
      - fp(a, kFpBaselineSatQueuePenalty, 0.5f) * normalize_masked_value(q, q_min, q_max)
      - fp(a, kFpBaselineSatLoadPenalty, 1.0f) * normalize_masked_value(load, load_min, load_max)
      + fp(a, kFpBaselineSatBwReward, 0.75f) * normalize_masked_value(bw, bw_min, bw_max)
      + fp(a, kFpBaselineSatStayBonus, 0.25f) * stay;
  return clampf_device(score, -logit_scale, logit_scale);
}

__device__ float queue_aware_sat_score(const PackedAbi& a, int stage_slot, int e, int u, int row, int slot, int width, float se_min, float se_max, float q_min, float q_max, float load_min, float load_max, float bw_min, float bw_max) {
  float se = 0.0f, q = 0.0f, load = 0.0f, bw = 0.0f, stay = 0.0f;
  queue_aware_sat_values(a, stage_slot, e, u, row, slot, width, &se, &q, &load, &bw, &stay);
  return queue_aware_sat_score_from_values(
      a,
      se,
      q,
      load,
      bw,
      stay,
      se_min,
      se_max,
      q_min,
      q_max,
      load_min,
      load_max,
      bw_min,
      bw_max);
}

__device__ float baseline_sat_slot_score(
    const PackedAbi& a,
    int source_mode,
    int stage_slot,
    int e,
    int u,
    int row,
    int slot,
    int width,
    float se_min,
    float se_max,
    float q_min,
    float q_max,
    float load_min,
    float load_max,
    float bw_min,
    float bw_max,
    float uav_pressure_steps) {
  float se = 0.0f, q = 0.0f, load = 0.0f, bw = 0.0f, stay = 0.0f;
  queue_aware_sat_values(a, stage_slot, e, u, row, slot, width, &se, &q, &load, &bw, &stay);
  if (source_mode == kSourceLinkPriority) {
    return se * bw;
  }
  if (source_mode == kSourceDemandPriority) {
    return -q + 1.0e-4f * se * bw;
  }
  if (source_mode == kSourceLyapunov) {
    const float drift_w = fmaxf(fp(a, kFpBaselineLyapunovSatDriftWeight, 0.6f), 0.0f);
    const int idx = (row * width + slot) * 26;
    const float sat_queue_steps = lyapunov_steps_from_log1p(a.f[kFLiveSatObs + 3][idx + 0]);
    const float observed_load = fmaxf(a.f[kFLiveSatObs + 3][idx + 8], 0.0f);
    const float doppler_margin = 1.0f - clampf_device(a.f[kFLiveSatObs + 3][idx + 20], 0.0f, 1.0f);
    float se_abs = 0.0f, queue_unused = 0.0f, relay_support = 0.0f;
    lyapunov_sat_profile_from_values(se, sat_queue_steps, observed_load, doppler_margin, &se_abs, &queue_unused, &relay_support);
    return drift_w * (uav_pressure_steps - sat_queue_steps) * relay_support;
  }
  if (source_mode == kSourceTopologyDppSat) {
    const int idx = (row * width + slot) * 26;
    const float sat_queue_steps = lyapunov_steps_from_log1p(a.f[kFLiveSatObs + 3][idx + 0]);
    const float gap = fmaxf(uav_pressure_steps - sat_queue_steps, 0.0f);
    const float backhaul_proxy =
        0.5f * normalize_masked_value(se, se_min, se_max)
        + 0.5f * normalize_masked_value(bw, bw_min, bw_max);
    const float gap_w = fmaxf(fp(a, kFpTopologyDppSatQueueGapWeight, 1.0f), 0.0f);
    return queue_aware_sat_score_from_values(
        a,
        se,
        q,
        load,
        bw,
        stay,
        se_min,
        se_max,
        q_min,
        q_max,
        load_min,
        load_max,
        bw_min,
        bw_max)
        + gap_w * gap * backhaul_proxy;
  }
  return queue_aware_sat_score(
      a,
      stage_slot,
      e,
      u,
      row,
      slot,
      width,
      se_min,
      se_max,
      q_min,
      q_max,
      load_min,
      load_max,
      bw_min,
      bw_max);
}

__global__ void baseline_sat_live_kernel(int64_t source_mode64) {
  const PackedAbi& a = cLiveAbi;
  const int e = blockIdx.x;
  const int u = blockIdx.y;
  if (e >= ip(a, kParamNumEnvs)) return;
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  if (u >= ucount) return;
  constexpr int sat_stage = 2;
  const int mode = static_cast<int>(source_mode64);
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int width = sat_visible_width(a);
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  const int subset_count = static_cast<int>(ip(a, kParamSubsetCount));
  const int row = e * ucount + u;
  if (select_k <= 0 || subset_count <= 0 || !has_l(a, kLMainSatSubsetMembersBase)) {
    if (threadIdx.x == 0) {
      if (has_l(a, kLLiveSatSubsetIndex)) a.l[kLLiveSatSubsetIndex][row] = 0;
      if (has_f(a, kFLiveSatOldLogprobPerAgent)) a.f[kFLiveSatOldLogprobPerAgent][row] = 0.0f;
      if (has_f(a, kFLiveSatEntropyPerAgent)) a.f[kFLiveSatEntropyPerAgent][row] = 0.0f;
    }
    return;
  }
  __shared__ float float_scratch[kSourceBlockMaxThreads];
  __shared__ int int_scratch[kSourceBlockMaxThreads];
  int local_valid_count = 0;
  float local_se_min = 3.402823466e38f, local_se_max = -3.402823466e38f;
  float local_q_min = 3.402823466e38f, local_q_max = -3.402823466e38f;
  float local_load_min = 3.402823466e38f, local_load_max = -3.402823466e38f;
  float local_bw_min = 3.402823466e38f, local_bw_max = -3.402823466e38f;
  for (int s = threadIdx.x; s < width; s += blockDim.x) {
    if (!a.b[kBLiveSatObs + 0][row * width + s] || !a.b[kBLiveSatObs + 1][row * width + s]) continue;
    ++local_valid_count;
    float se = 0.0f, q = 0.0f, load = 0.0f, bw = 0.0f, stay = 0.0f;
    queue_aware_sat_values(a, sat_stage, e, u, row, s, width, &se, &q, &load, &bw, &stay);
    local_se_min = fminf(local_se_min, se); local_se_max = fmaxf(local_se_max, se);
    local_q_min = fminf(local_q_min, q); local_q_max = fmaxf(local_q_max, q);
    local_load_min = fminf(local_load_min, load); local_load_max = fmaxf(local_load_max, load);
    local_bw_min = fminf(local_bw_min, bw); local_bw_max = fmaxf(local_bw_max, bw);
  }
  const int valid_count = block_sum_int(local_valid_count, int_scratch);
  const float se_min = block_min_float(local_se_min, float_scratch);
  const float se_max = block_max_float(local_se_max, float_scratch);
  const float q_min = block_min_float(local_q_min, float_scratch);
  const float q_max = block_max_float(local_q_max, float_scratch);
  const float load_min = block_min_float(local_load_min, float_scratch);
  const float load_max = block_max_float(local_load_max, float_scratch);
  const float bw_min = block_min_float(local_bw_min, float_scratch);
  const float bw_max = block_max_float(local_bw_max, float_scratch);
  constexpr int kLyapunovSatEgoDim = 13;
  const float* sat_ego = a.f[kFLiveSatObs + 0] + static_cast<int64_t>(row) * kLyapunovSatEgoDim;
  const bool needs_uav_pressure = mode == kSourceLyapunov || mode == kSourceTopologyDppSat;
  const float uav_pressure_steps = needs_uav_pressure ? lyapunov_steps_from_log1p(sat_ego[0]) : 0.0f;
  const int target_size = min(max(select_k, 0), valid_count);
  float local_best_sum = -3.402823466e38f;
  int local_best_subset = -1;
  for (int subset = threadIdx.x; subset < subset_count; subset += blockDim.x) {
    int size = 0;
    float score_sum = 0.0f;
    float backhaul_term = 0.0f;
    float contention_penalty = 0.0f;
    bool valid_subset = true;
    for (int k = 0; k < select_k; ++k) {
      const int slot = static_cast<int>(a.l[kLMainSatSubsetMembersBase][subset * select_k + k]);
      if (slot < 0) continue;
      ++size;
      if (slot >= width || !a.b[kBLiveSatObs + 0][row * width + slot] || !a.b[kBLiveSatObs + 1][row * width + slot]) {
        valid_subset = false;
        break;
      }
      score_sum += baseline_sat_slot_score(
          a,
          mode,
          sat_stage,
          e,
          u,
          row,
          slot,
          width,
          se_min,
          se_max,
          q_min,
          q_max,
          load_min,
          load_max,
          bw_min,
          bw_max,
          uav_pressure_steps);
      if (mode == kSourceTopologyDppSat) {
        float se = 0.0f, q = 0.0f, load = 0.0f, bw = 0.0f, stay = 0.0f;
        queue_aware_sat_values(a, sat_stage, e, u, row, slot, width, &se, &q, &load, &bw, &stay);
        const int idx = (row * width + slot) * 26;
        const float sat_queue_steps = lyapunov_steps_from_log1p(a.f[kFLiveSatObs + 3][idx + 0]);
        const float gap = fmaxf(uav_pressure_steps - sat_queue_steps, 0.0f);
        const float proxy =
            0.5f * normalize_masked_value(se, se_min, se_max)
            + 0.5f * normalize_masked_value(bw, bw_min, bw_max);
        backhaul_term += gap * proxy;
        contention_penalty += fmaxf(1.0f - bw, 0.0f);
      }
    }
    if (!valid_subset || size != target_size) continue;
    if (mode == kSourceUniform || mode == kSourceRandom) {
      score_sum = source_hash01(a, e, u, subset, mode == kSourceUniform ? 501 : 503);
    } else if (mode == kSourceTopologyDppSat) {
      const float subset_penalty = fmaxf(fp(a, kFpTopologyDppSatSubsetPenalty, 0.02f), 0.0f);
      const float contention_w = fmaxf(fp(a, kFpTopologyDppSatContentionWeight, 0.15f), 0.0f);
      score_sum += backhaul_term;
      score_sum -= subset_penalty * static_cast<float>(size * size);
      score_sum -= contention_w * contention_penalty;
    }
    if (score_sum > local_best_sum || (score_sum == local_best_sum && (local_best_subset < 0 || subset < local_best_subset))) {
      local_best_sum = score_sum;
      local_best_subset = subset;
    }
  }
  float reduced_best_sum = -3.402823466e38f;
  int reduced_best_subset = -1;
  block_argmax_min_index(local_best_sum, local_best_subset, float_scratch, int_scratch, &reduced_best_sum, &reduced_best_subset);
  if (threadIdx.x == 0) {
    a.l[kLLiveSatSubsetIndex][row] = static_cast<int64_t>(reduced_best_subset >= 0 ? reduced_best_subset : 0);
    if (has_f(a, kFLiveSatOldLogprobPerAgent)) a.f[kFLiveSatOldLogprobPerAgent][row] = 0.0f;
    if (has_f(a, kFLiveSatEntropyPerAgent)) a.f[kFLiveSatEntropyPerAgent][row] = 0.0f;
    if (e == 0 && u == 0 && has_i(a, kIMarker)) a.i[kIMarker][0] = 23;
  }
}

__global__ void queue_aware_sat_live_kernel() {
  const PackedAbi& a = cLiveAbi;
  const int e = blockIdx.x;
  const int u = blockIdx.y;
  if (e >= ip(a, kParamNumEnvs)) return;
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  if (u >= ucount) return;
  constexpr int sat_stage = 2;
  const int width = sat_visible_width(a);
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  const int subset_count = static_cast<int>(ip(a, kParamSubsetCount));
  const int row = e * ucount + u;
  if (select_k <= 0 || subset_count <= 0 || !has_l(a, kLMainSatSubsetMembersBase)) {
    if (threadIdx.x == 0) {
      if (has_l(a, kLLiveSatSubsetIndex)) a.l[kLLiveSatSubsetIndex][row] = 0;
      if (has_f(a, kFLiveSatOldLogprobPerAgent)) a.f[kFLiveSatOldLogprobPerAgent][row] = 0.0f;
      if (has_f(a, kFLiveSatEntropyPerAgent)) a.f[kFLiveSatEntropyPerAgent][row] = 0.0f;
    }
    return;
  }
  __shared__ float float_scratch[kSourceBlockMaxThreads];
  __shared__ int int_scratch[kSourceBlockMaxThreads];
  __shared__ int best_subset_shared;
  int local_valid_count = 0;
  float local_se_min = 3.402823466e38f, local_se_max = -3.402823466e38f;
  float local_q_min = 3.402823466e38f, local_q_max = -3.402823466e38f;
  float local_load_min = 3.402823466e38f, local_load_max = -3.402823466e38f;
  float local_bw_min = 3.402823466e38f, local_bw_max = -3.402823466e38f;
  for (int s = threadIdx.x; s < width; s += blockDim.x) {
    if (!a.b[kBLiveSatObs + 0][row * width + s] || !a.b[kBLiveSatObs + 1][row * width + s]) continue;
    ++local_valid_count;
    float se = 0.0f, q = 0.0f, load = 0.0f, bw = 0.0f, stay = 0.0f;
    queue_aware_sat_values(a, sat_stage, e, u, row, s, width, &se, &q, &load, &bw, &stay);
    local_se_min = fminf(local_se_min, se); local_se_max = fmaxf(local_se_max, se);
    local_q_min = fminf(local_q_min, q); local_q_max = fmaxf(local_q_max, q);
    local_load_min = fminf(local_load_min, load); local_load_max = fmaxf(local_load_max, load);
    local_bw_min = fminf(local_bw_min, bw); local_bw_max = fmaxf(local_bw_max, bw);
  }
  const int valid_count = block_sum_int(local_valid_count, int_scratch);
  const float se_min = block_min_float(local_se_min, float_scratch);
  const float se_max = block_max_float(local_se_max, float_scratch);
  const float q_min = block_min_float(local_q_min, float_scratch);
  const float q_max = block_max_float(local_q_max, float_scratch);
  const float load_min = block_min_float(local_load_min, float_scratch);
  const float load_max = block_max_float(local_load_max, float_scratch);
  const float bw_min = block_min_float(local_bw_min, float_scratch);
  const float bw_max = block_max_float(local_bw_max, float_scratch);
  const int target_size = min(max(select_k, 0), valid_count);
  float local_best_sum = -3.402823466e38f;
  int local_best_subset = -1;
  for (int subset = threadIdx.x; subset < subset_count; subset += blockDim.x) {
    int size = 0;
    float score_sum = 0.0f;
    bool valid_subset = true;
    for (int k = 0; k < select_k; ++k) {
      const int slot = static_cast<int>(a.l[kLMainSatSubsetMembersBase][subset * select_k + k]);
      if (slot < 0) continue;
      ++size;
      if (slot >= width || !a.b[kBLiveSatObs + 0][row * width + slot] || !a.b[kBLiveSatObs + 1][row * width + slot]) {
        valid_subset = false;
        break;
      }
      score_sum += queue_aware_sat_score(a, sat_stage, e, u, row, slot, width, se_min, se_max, q_min, q_max, load_min, load_max, bw_min, bw_max);
    }
    if (!valid_subset || size != target_size) continue;
    if (score_sum > local_best_sum || (score_sum == local_best_sum && (local_best_subset < 0 || subset < local_best_subset))) {
      local_best_sum = score_sum;
      local_best_subset = subset;
    }
  }
  float reduced_best_sum = -3.402823466e38f;
  int reduced_best_subset = -1;
  block_argmax_min_index(local_best_sum, local_best_subset, float_scratch, int_scratch, &reduced_best_sum, &reduced_best_subset);
  if (threadIdx.x == 0) best_subset_shared = reduced_best_subset >= 0 ? reduced_best_subset : 0;
  __syncthreads();

  if (select_k == 1 && target_size == 1) {
    float local_best_score = -3.402823466e38f;
    int local_best_slot = -1;
    int local_current_count = 0;
    int local_current_slot = -1;
    for (int s = threadIdx.x; s < width; s += blockDim.x) {
      if (!a.b[kBLiveSatObs + 0][row * width + s] || !a.b[kBLiveSatObs + 1][row * width + s]) continue;
      const float score = queue_aware_sat_score(a, sat_stage, e, u, row, s, width, se_min, se_max, q_min, q_max, load_min, load_max, bw_min, bw_max);
      if (score > local_best_score || (score == local_best_score && (local_best_slot < 0 || s < local_best_slot))) {
        local_best_score = score;
        local_best_slot = s;
      }
      float se_tmp = 0.0f, q_tmp = 0.0f, load_tmp = 0.0f, bw_tmp = 0.0f, stay_tmp = 0.0f;
      queue_aware_sat_values(a, sat_stage, e, u, row, s, width, &se_tmp, &q_tmp, &load_tmp, &bw_tmp, &stay_tmp);
      if (stay_tmp > 0.5f) {
        ++local_current_count;
        local_current_slot = s;
      }
    }
    float best_score = -3.402823466e38f;
    int best_slot = -1;
    block_argmax_min_index(local_best_score, local_best_slot, float_scratch, int_scratch, &best_score, &best_slot);
    const int current_count = block_sum_int(local_current_count, int_scratch);
    float current_slot_value = static_cast<float>(local_current_slot);
    int current_slot_index = local_current_slot;
    float reduced_current_slot_value = -1.0f;
    int current_slot = -1;
    block_argmax_min_index(current_slot_value, current_slot_index, float_scratch, int_scratch, &reduced_current_slot_value, &current_slot);
    const float current_score = current_slot >= 0
        ? queue_aware_sat_score(a, sat_stage, e, u, row, current_slot, width, se_min, se_max, q_min, q_max, load_min, load_max, bw_min, bw_max)
        : -3.402823466e38f;
    if (current_count == 1 && best_slot != current_slot &&
        best_score <= current_score + fp(a, kFpBaselineSatSwitchMargin, 0.15f)) {
      float local_subset_value = 3.402823466e38f;
      int local_subset_index = -1;
      for (int subset = threadIdx.x; subset < subset_count; subset += blockDim.x) {
        const int slot = static_cast<int>(a.l[kLMainSatSubsetMembersBase][subset * select_k]);
        if (slot == current_slot) {
          local_subset_value = static_cast<float>(subset);
          local_subset_index = subset;
          break;
        }
      }
      float selected_subset_value = 3.402823466e38f;
      int selected_subset_index = -1;
      block_argmin_min_index(local_subset_value, local_subset_index, float_scratch, int_scratch, &selected_subset_value, &selected_subset_index);
      if (threadIdx.x == 0 && selected_subset_index >= 0) best_subset_shared = selected_subset_index;
      __syncthreads();
    }
  }
  if (threadIdx.x == 0) {
    a.l[kLLiveSatSubsetIndex][row] = static_cast<int64_t>(best_subset_shared);
    if (has_f(a, kFLiveSatOldLogprobPerAgent)) a.f[kFLiveSatOldLogprobPerAgent][row] = 0.0f;
    if (has_f(a, kFLiveSatEntropyPerAgent)) a.f[kFLiveSatEntropyPerAgent][row] = 0.0f;
    if (e == 0 && u == 0 && has_i(a, kIMarker)) a.i[kIMarker][0] = 13;
  }
}

__global__ void cluster_center_accel_live_kernel(int64_t active_idx) {
  const PackedAbi& a = cLiveAbi;
  const int e = blockIdx.x;
  if (e >= ip(a, kParamNumEnvs)) return;
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int live_f = active_idx == 0 ? kFLiveAccelObs0 : kFLiveAccelObs1;
  constexpr int kMaxNativeClusterUav = 2048;
  __shared__ int selected[kMaxNativeClusterUav];
  __shared__ int target[kMaxNativeClusterUav];
  __shared__ unsigned char assigned[kMaxNativeClusterUav];
  __shared__ int selected_count_shared;
  __shared__ int stop_selection_shared;
  __shared__ float reduce_value[kSourceBlockMaxThreads];
  __shared__ int reduce_index[kSourceBlockMaxThreads];
  const int center_count = (has_f(a, kFStateGuClusterCenters) && ip(a, kParamNumEnvs) > 0)
      ? static_cast<int>(a.f_numel[kFStateGuClusterCenters] / (ip(a, kParamNumEnvs) * 2))
      : 0;
  const int usable_uav = min(ucount, kMaxNativeClusterUav);
  for (int u = threadIdx.x; u < usable_uav; u += blockDim.x) {
    assigned[u] = 0;
    target[u] = -1;
    selected[u] = -1;
  }
  if (threadIdx.x == 0) selected_count_shared = 0;
  if (threadIdx.x == 0) stop_selection_shared = 0;
  __syncthreads();

  for (int rank = 0; rank < usable_uav && rank < center_count; ++rank) {
    const int selected_count = selected_count_shared;
    float local_best_count = 0.0f;
    int local_best_cluster = -1;
    for (int c = threadIdx.x; c < center_count; c += blockDim.x) {
      bool used = false;
      for (int r = 0; r < selected_count; ++r) used = used || selected[r] == c;
      if (used) continue;
      const bool count_available = has_f(a, kFStateGuClusterCounts)
          && a.f_numel[kFStateGuClusterCounts] >= static_cast<int64_t>((e + 1) * center_count);
      const float count = count_available ? a.f[kFStateGuClusterCounts][e * center_count + c] : 0.0f;
      if (count > local_best_count || (count == local_best_count && local_best_cluster >= 0 && c < local_best_cluster)) {
        local_best_count = count;
        local_best_cluster = c;
      }
    }
    float best_count = 0.0f;
    int best_cluster = -1;
    block_argmax_min_index(local_best_count, local_best_cluster, reduce_value, reduce_index, &best_count, &best_cluster);
    if (threadIdx.x == 0) {
      if (best_cluster < 0) {
        stop_selection_shared = 1;
      } else {
        selected[selected_count] = best_cluster;
        selected_count_shared = selected_count + 1;
      }
    }
    __syncthreads();
    if (stop_selection_shared) break;
  }

  for (int r = 0; r < selected_count_shared; ++r) {
    const int c = selected[r];
    const float cx = a.f[kFStateGuClusterCenters][(e * center_count + c) * 2 + 0];
    const float cy = a.f[kFStateGuClusterCenters][(e * center_count + c) * 2 + 1];
    float local_best_d2 = 3.402823466e38f;
    int local_best_u = -1;
    for (int cand_u = threadIdx.x; cand_u < usable_uav; cand_u += blockDim.x) {
      if (assigned[cand_u]) continue;
      const int row = row_local(a, e, cand_u);
      const int uav_dim = kAccelEgoDim;
      const float ux = a.f[live_f + 0][row * uav_dim + 0] * fp(a, kFpMapSize, 1.0f);
      const float uy = a.f[live_f + 0][row * uav_dim + 1] * fp(a, kFpMapSize, 1.0f);
      const float dx = ux - cx;
      const float dy = uy - cy;
      const float d2 = dx * dx + dy * dy;
      if (d2 < local_best_d2 || (d2 == local_best_d2 && (local_best_u < 0 || cand_u < local_best_u))) {
        local_best_d2 = d2;
        local_best_u = cand_u;
      }
    }
    float best_d2 = 3.402823466e38f;
    int best_u = -1;
    block_argmin_min_index(local_best_d2, local_best_u, reduce_value, reduce_index, &best_d2, &best_u);
    if (threadIdx.x == 0 && best_u >= 0) {
      assigned[best_u] = 1;
      target[best_u] = c;
    }
    __syncthreads();
  }

  const int selected_count = selected_count_shared;
  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    const int row = row_local(a, e, u);
    const int uav_dim = kAccelEgoDim;
    int target_u = (u < usable_uav) ? target[u] : -1;
    if (target_u < 0 && selected_count > 0) {
      const float ux = a.f[live_f + 0][row * uav_dim + 0] * fp(a, kFpMapSize, 1.0f);
      const float uy = a.f[live_f + 0][row * uav_dim + 1] * fp(a, kFpMapSize, 1.0f);
      float best_d2 = 3.402823466e38f;
      int best_cluster = selected[0];
      for (int r = 0; r < selected_count; ++r) {
        const int c = selected[r];
        const float cx = a.f[kFStateGuClusterCenters][(e * center_count + c) * 2 + 0];
        const float cy = a.f[kFStateGuClusterCenters][(e * center_count + c) * 2 + 1];
        const float dx = ux - cx;
        const float dy = uy - cy;
        const float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) { best_d2 = d2; best_cluster = c; }
      }
      target_u = best_cluster;
    }
    float ax = 0.0f;
    float ay = 0.0f;
    if (target_u >= 0 && center_count > 0) {
      const float cx = a.f[kFStateGuClusterCenters][(e * center_count + target_u) * 2 + 0];
      const float cy = a.f[kFStateGuClusterCenters][(e * center_count + target_u) * 2 + 1];
      const float px = a.f[live_f + 0][row * uav_dim + 0] * fp(a, kFpMapSize, 1.0f);
      const float py = a.f[live_f + 0][row * uav_dim + 1] * fp(a, kFpMapSize, 1.0f);
      const float velx = (uav_dim > 2 ? a.f[live_f + 0][row * uav_dim + 2] : 0.0f) * fp(a, kFpVMax, 1.0f);
      const float vely = (uav_dim > 3 ? a.f[live_f + 0][row * uav_dim + 3] : 0.0f) * fp(a, kFpVMax, 1.0f);
      const float ex = cx - px;
      const float ey = cy - py;
      const float dist = sqrtf(ex * ex + ey * ey);
      const float speed = sqrtf(velx * velx + vely * vely);
      const float stop_radius = fmaxf(fp(a, kFpBaselineClusterStopRadius, 20.0f), 0.0f);
      const float speed_tol = fmaxf(fp(a, kFpBaselineClusterSpeedTol, 2.0f), 0.0f);
      const float slow_radius = fmaxf(fp(a, kFpBaselineClusterSlowRadius, 120.0f), stop_radius + kDynamicsDenomEps);
      const float cruise = fminf(fmaxf(fp(a, kFpBaselineClusterCruiseSpeed, fp(a, kFpUavOptSpeed, 0.0f)), 0.0f), fp(a, kFpVMax, 1.0f));
      const float vel_gain = fmaxf(fp(a, kFpBaselineClusterVelGain, 1.0f), 0.0f);
      float desired_x = 0.0f;
      float desired_y = 0.0f;
      if (dist > stop_radius) {
        const float desired_speed = cruise * fminf(dist / dynamics_denominator(slow_radius), 1.0f);
        desired_x = ex / dynamics_denominator(dist) * desired_speed;
        desired_y = ey / dynamics_denominator(dist) * desired_speed;
      }
      ax = vel_gain * (desired_x - velx) / dynamics_denominator(fp(a, kFpTau0, 1.0f)) / positive_config_scale(fp(a, kFpAccelAMax, 1.0f));
      ay = vel_gain * (desired_y - vely) / dynamics_denominator(fp(a, kFpTau0, 1.0f)) / positive_config_scale(fp(a, kFpAccelAMax, 1.0f));
      if (dist <= stop_radius && speed <= speed_tol) { ax = 0.0f; ay = 0.0f; }
    }
    ax += baseline_energy_term_component(a, live_f, row, 0);
    ay += baseline_energy_term_component(a, live_f, row, 1);
    a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 0] = clampf_device(ax, -1.0f, 1.0f);
    a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 1] = clampf_device(ay, -1.0f, 1.0f);
    if (has_f(a, kFLiveAccelLatentAction)) {
      a.f[kFLiveAccelLatentAction][(e * ucount + u) * 2 + 0] = 0.0f;
      a.f[kFLiveAccelLatentAction][(e * ucount + u) * 2 + 1] = 0.0f;
    }
    if (has_f(a, kFLiveAccelOldLogprob)) a.f[kFLiveAccelOldLogprob][e * ucount + u] = 0.0f;
  }
  if (threadIdx.x == 0 && has_i(a, kIMarker)) a.i[kIMarker][0] = 14;
}

__device__ void copy_runtime_state_snapshot_to_history_parallel(
    const PackedAbi& a,
    int float_base,
    int long_base,
    int int_base,
    int bool_base,
    int slot,
    int e) {
  if (!history_snapshots_enabled(a)) return;
  const int ecount = static_cast<int>(ip(a, kParamNumEnvs));
  const int hist_rows = max(static_cast<int>(ip(a, kParamHistoryCapacity, 0)), 0) * max(ecount, 0);
  if (ecount <= 0 || hist_rows <= 0) return;
  const int dst_row = hist_env_row(a, slot, e);
  if (dst_row < 0 || dst_row >= hist_rows) return;

#define COPY_RUNTIME_STATE_F(offset, src_tensor) \
  copy_float_row_parallel(a, float_base + (offset), (src_tensor), dst_row, e, hist_rows, ecount)
  COPY_RUNTIME_STATE_F(0, kFStateUavPos);
  COPY_RUNTIME_STATE_F(1, kFStateUavVel);
  COPY_RUNTIME_STATE_F(2, kFStateUavEnergy);
  COPY_RUNTIME_STATE_F(3, kFStateUavQueue);
  COPY_RUNTIME_STATE_F(4, kFStateGuPos);
  COPY_RUNTIME_STATE_F(5, kFStateGuClusterCenters);
  COPY_RUNTIME_STATE_F(6, kFStateGuClusterCounts);
  COPY_RUNTIME_STATE_F(7, kFStateGuQueue);
  COPY_RUNTIME_STATE_F(8, kFStateSatQueue);
  COPY_RUNTIME_STATE_F(9, kFStateSatPos);
  COPY_RUNTIME_STATE_F(10, kFStateSatVel);
  COPY_RUNTIME_STATE_F(11, kFStateLastSatConnectionCounts);
  COPY_RUNTIME_STATE_F(12, kFStateArrivalRef);
  COPY_RUNTIME_STATE_F(13, kFStateEffectiveArrivalRate);
  COPY_RUNTIME_STATE_F(14, kFStateArrivalBaseScale);
  COPY_RUNTIME_STATE_F(15, kFStateGuEma);
  COPY_RUNTIME_STATE_F(16, kFStateUavEma);
  COPY_RUNTIME_STATE_F(17, kFStateSatEma);
  COPY_RUNTIME_STATE_F(18, kFStateLastArrivalRateVec);
  COPY_RUNTIME_STATE_F(19, kFStateGuDeadlineSteps);
  COPY_RUNTIME_STATE_F(20, kFStateLastGuArrival);
  COPY_RUNTIME_STATE_F(21, kFStateLastGuOutflow);
  COPY_RUNTIME_STATE_F(22, kFStateGuDrop);
  COPY_RUNTIME_STATE_F(23, kFStateUavDrop);
  COPY_RUNTIME_STATE_F(24, kFStateSatDrop);
  COPY_RUNTIME_STATE_F(25, kFStateLastAccessInterferenceByUav);
  COPY_RUNTIME_STATE_F(26, kFStateLastBwFractionByUavGu);
  COPY_RUNTIME_STATE_F(27, kFStateLastGuToUavInflowByUav);
  COPY_RUNTIME_STATE_F(28, kFStateLastUavToSatOutflowMatrix);
  COPY_RUNTIME_STATE_F(29, kFStateLastSelectedMaskByUavSat);
  COPY_RUNTIME_STATE_F(30, kFStateLastSatProcessed);
  COPY_RUNTIME_STATE_F(31, kFStateUrgencyRisk);
  COPY_RUNTIME_STATE_F(32, kFStateDownstreamPressure);
  COPY_RUNTIME_STATE_F(33, kFStateServiceGapRisk);
  COPY_RUNTIME_STATE_F(34, kFStateDeadlineSlack);
  COPY_RUNTIME_STATE_F(35, kFStateDeadlineRisk);
  COPY_RUNTIME_STATE_F(36, kFStateServiceGap);
  COPY_RUNTIME_STATE_F(37, kFStateDeadlineAge);
  COPY_RUNTIME_STATE_F(38, kFStateLastExecAccel);
  COPY_RUNTIME_STATE_F(39, kFStateLastPolicyAccel);
  COPY_RUNTIME_STATE_F(40, kFStateAvoidanceEtaEff);
  COPY_RUNTIME_STATE_F(41, kFStateLastAvoidanceEtaExec);
  COPY_RUNTIME_STATE_F(42, kFStateDopplerResidual);
  COPY_RUNTIME_STATE_F(43, kFStatePrevQueueSumGu);
  COPY_RUNTIME_STATE_F(44, kFStatePrevQueueSumUav);
  COPY_RUNTIME_STATE_F(45, kFStatePrevQueueSumSat);
  COPY_RUNTIME_STATE_F(46, kFStatePrevQNormActive);
  COPY_RUNTIME_STATE_F(47, kFStatePrevGuQueueVec);
  COPY_RUNTIME_STATE_F(48, kFStatePrevUavQueueVec);
  COPY_RUNTIME_STATE_F(49, kFStatePrevSatQueueVec);
  COPY_RUNTIME_STATE_F(50, kFStateHotspotMemberMask);
#undef COPY_RUNTIME_STATE_F

  copy_long_row_parallel(a, long_base + 0, kLStateLastSatSelectionMatrix, dst_row, e, hist_rows, ecount);
  copy_int_row_parallel(a, int_base + 0, kIStatePrevAssociation, dst_row, e, hist_rows, ecount);
  copy_int_row_parallel(a, int_base + 1, kIStateLastAssociation, dst_row, e, hist_rows, ecount);
  copy_int_row_parallel(a, int_base + 2, kIStateHotspotActiveIdx, dst_row, e, hist_rows, ecount);
  copy_int_row_parallel(a, int_base + 3, kIStateHotspotSubsetCount, dst_row, e, hist_rows, ecount);
  copy_int_row_parallel(a, int_base + 4, kIStateTrafficResetStep, dst_row, e, hist_rows, ecount);
  copy_int_row_parallel(a, int_base + 5, kIStateTrafficResetOrdinal, dst_row, e, hist_rows, ecount);
  copy_int_row_parallel(a, int_base + 6, kIStateEpisodeIdx, dst_row, e, hist_rows, ecount);
  copy_int_row_parallel(a, int_base + 7, kIStateT, dst_row, e, hist_rows, ecount);
  copy_int_row_parallel(a, int_base + 8, kIStateGlobalStep, dst_row, e, hist_rows, ecount);
  copy_bool_row_parallel(a, bool_base + 0, kBStateHotspotMemberMask, dst_row, e, hist_rows, ecount);
}

__device__ void copy_stage_snapshot_to_history_parallel(
    const PackedAbi& a,
    int float_base,
    int long_base,
    int bool_base,
    int slot,
    int e,
    int stage) {
  if (!history_snapshots_enabled(a)) return;
  const int ecount = static_cast<int>(ip(a, kParamNumEnvs));
  const int hist_rows = max(static_cast<int>(ip(a, kParamHistoryCapacity, 0)), 0) * max(ecount, 0);
  if (ecount <= 0 || hist_rows <= 0) return;
  const int dst_row = hist_env_row(a, slot, e);
  if (dst_row < 0 || dst_row >= hist_rows) return;
  for (int field = 0; field <= kSfUsSatQueueAll; ++field) {
    copy_float_row_parallel(a, float_base + field, stage_f(stage, field), dst_row, e, hist_rows, ecount);
  }
  for (int field = 0; field <= kSlActiveSatIds; ++field) {
    copy_long_row_parallel(a, long_base + field, stage_l(stage, field), dst_row, e, hist_rows, ecount);
  }
  copy_bool_row_parallel(a, bool_base + 0, stage_b(stage, 0), dst_row, e, hist_rows, ecount);
  copy_bool_row_parallel(a, bool_base + 1, stage_b(stage, 1), dst_row, e, hist_rows, ecount);
}

__device__ void copy_bw_runtime_snapshot_to_history_parallel(const PackedAbi& a, int slot, int e, int bw_stage) {
  if (!history_snapshots_enabled(a)) return;
  const int ecount = static_cast<int>(ip(a, kParamNumEnvs));
  const int hist_rows = max(static_cast<int>(ip(a, kParamHistoryCapacity, 0)), 0) * max(ecount, 0);
  if (ecount <= 0 || hist_rows <= 0) return;
  const int dst_row = hist_env_row(a, slot, e);
  if (dst_row < 0 || dst_row >= hist_rows) return;

  copy_runtime_state_snapshot_to_history_parallel(
      a,
      kFHistBwRuntimeStateBase,
      kLHistBwRuntimeStateBase,
      kIHistBwRuntimeStateBase,
      kBHistBwRuntimeStateBase,
      slot,
      e);

#define COPY_BW_CACHE_F(offset, src_tensor) \
  copy_float_row_parallel(a, kFHistBwRuntimeCacheBase + (offset), (src_tensor), dst_row, e, hist_rows, ecount)
  COPY_BW_CACHE_F(0, kFBwInputBase + 0);
  COPY_BW_CACHE_F(1, kFBwInputBase + 1);
  COPY_BW_CACHE_F(2, kFBwInputBase + 2);
  COPY_BW_CACHE_F(3, kFBwInputBase + 3);
  COPY_BW_CACHE_F(4, kFBwInputBase + 4);
  COPY_BW_CACHE_F(5, kFBwInputBase + 5);
  COPY_BW_CACHE_F(6, kFBwInputBase + 6);
  COPY_BW_CACHE_F(7, kFBwInputBase + 7);
  COPY_BW_CACHE_F(8, kFBwInputBase + 8);
  COPY_BW_CACHE_F(9, kFBwInputBase + 9);
#undef COPY_BW_CACHE_F

  copy_long_row_parallel(a, kLHistBwRuntimeCacheBase + 0, kLBwInputBase + 2, dst_row, e, hist_rows, ecount);
  copy_long_row_parallel(a, kLHistBwRuntimeCacheBase + 1, kLBwInputBase + 0, dst_row, e, hist_rows, ecount);
  copy_long_row_parallel(a, kLHistBwRuntimeCacheBase + 2, kLBwInputBase + 1, dst_row, e, hist_rows, ecount);
  copy_long_row_parallel(a, kLHistBwRuntimeCacheBase + 3, kLBwInputBase + 3, dst_row, e, hist_rows, ecount);
  copy_long_row_parallel(a, kLHistBwRuntimeCacheBase + 4, kLBwInputBase + 4, dst_row, e, hist_rows, ecount);
  copy_bool_row_parallel(a, kBHistBwRuntimeCacheBase + 0, kBBwInputCandidateMask, dst_row, e, hist_rows, ecount);

  copy_stage_snapshot_to_history_parallel(
      a,
      kFHistBwRuntimeStageBase,
      kLHistBwRuntimeStageBase,
      kBHistBwRuntimeStageBase,
      slot,
      e,
      bw_stage);
}

__device__ int macro_start_slot_for_env(const PackedAbi& a, int slot, int e, int interval) {
  if (interval <= 1 || slot <= 0) return slot;
  int episode_start = 0;
  for (int back = 1; slot - back >= 0; ++back) {
    const int prev_row = hist_env_row(a, slot - back, e);
    const bool prev_done =
        (has_b(a, kBHistTerminated) && a.b[kBHistTerminated][prev_row]) ||
        (has_b(a, kBHistTruncated) && a.b[kBHistTruncated][prev_row]);
    if (prev_done) {
      episode_start = slot - back + 1;
      break;
    }
  }
  const int age = max(slot - episode_start, 0);
  return slot - (age % interval);
}

__device__ int bw_macro_start_slot_for_env(const PackedAbi& a, int slot, int e) {
  const int interval = max(static_cast<int>(ip(a, kParamAccessBwDecisionInterval, 1)), 1);
  return macro_start_slot_for_env(a, slot, e, interval);
}

__device__ int sat_macro_start_slot_for_env(const PackedAbi& a, int slot, int e) {
  const int interval = max(static_cast<int>(ip(a, kParamSatDecisionInterval, 1)), 1);
  return macro_start_slot_for_env(a, slot, e, interval);
}

__global__ void apply_bw_macro_live_kernel(int64_t slot64, int64_t bw_source_mode) {
  const PackedAbi& a = cLiveAbi;
  const int e = static_cast<int>(blockIdx.x);
  const int ecount = static_cast<int>(ip(a, kParamNumEnvs));
  if (e >= ecount) return;
  const int interval = max(static_cast<int>(ip(a, kParamAccessBwDecisionInterval, 1)), 1);
  const int slot = static_cast<int>(slot64);
  const int hist_cap = max(static_cast<int>(ip(a, kParamHistoryCapacity, 0)), 0);
  const int hist_rows = hist_cap * max(ecount, 0);
  if (interval <= 1 || slot <= 0 || hist_rows <= 0) return;
  const int start_slot = bw_macro_start_slot_for_env(a, slot, e);
  if (start_slot == slot) return;
  const int src_row = hist_env_row(a, start_slot, e);
  if (src_row < 0 || src_row >= hist_rows) return;

  const int bw_stage = 3;
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  if (has_l(a, kLHistBwRuntimeStageBase + kSlAssoc)) {
    for (int g = threadIdx.x; g < gu; g += blockDim.x) {
      a.l[stage_l(bw_stage, kSlAssoc)][e * gu + g] =
          a.l[kLHistBwRuntimeStageBase + kSlAssoc][src_row * gu + g];
    }
  }
  copy_long_row_parallel(a, stage_l(bw_stage, kSlCandidateIndices), kLHistBwRuntimeStageBase + kSlCandidateIndices, e, src_row, ecount, hist_rows);
  copy_bool_row_parallel(a, stage_b(bw_stage, 0), kBHistBwRuntimeStageBase + 0, e, src_row, ecount, hist_rows);
  copy_float_row_parallel(a, stage_f(bw_stage, kSfBwValidMask), kFHistBwRuntimeStageBase + kSfBwValidMask, e, src_row, ecount, hist_rows);
  copy_float_row_parallel(a, stage_f(bw_stage, kSfCandidateFlag), kFHistBwRuntimeStageBase + kSfCandidateFlag, e, src_row, ecount, hist_rows);
  copy_float_row_parallel(a, stage_f(bw_stage, kSfBwValidFlag), kFHistBwRuntimeStageBase + kSfBwValidFlag, e, src_row, ecount, hist_rows);
  for (int idx = threadIdx.x; idx < ucount * gu; idx += blockDim.x) {
    const int out = (e * ucount) * gu + idx;
    const int src = (src_row * ucount) * gu + idx;
    if (has_f(a, kFHistBwActions)) {
      const float value = bw_source_mode == kSourceZero ? 0.0f : a.f[kFHistBwActions][src];
      a.f[kFLiveBwAction][out] = value;
      if (has_f(a, kFLiveBwFlowProxyOverrideAction)) a.f[kFLiveBwFlowProxyOverrideAction][out] = value;
    }
    if (has_f(a, kFHistBwRefActions)) {
      a.f[kFLiveBwRefAction][out] = bw_source_mode == kSourceZero ? 0.0f : a.f[kFHistBwRefActions][src];
    }
  }
  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    const int live_u = e * ucount + u;
    const int hist_u = src_row * ucount + u;
    if (has_f(a, kFHistBwOldLogprobsPerAgent) && has_f(a, kFLiveBwOldLogprobPerAgent)) {
      a.f[kFLiveBwOldLogprobPerAgent][live_u] =
          bw_source_mode == kSourceZero ? 0.0f : a.f[kFHistBwOldLogprobsPerAgent][hist_u];
    }
    if (has_f(a, kFHistBwLogprobRawPerAgent) && has_f(a, kFLiveBwLogprobRawPerAgent)) {
      a.f[kFLiveBwLogprobRawPerAgent][live_u] =
          bw_source_mode == kSourceZero ? 0.0f : a.f[kFHistBwLogprobRawPerAgent][hist_u];
    }
    if (has_f(a, kFHistBwEntropyPerAgent) && has_f(a, kFLiveBwEntropyPerAgent)) {
      a.f[kFLiveBwEntropyPerAgent][live_u] =
          bw_source_mode == kSourceZero ? 0.0f : a.f[kFHistBwEntropyPerAgent][hist_u];
    }
    if (has_f(a, kFHistBwEntropyRawPerAgent) && has_f(a, kFLiveBwEntropyRawPerAgent)) {
      a.f[kFLiveBwEntropyRawPerAgent][live_u] =
          bw_source_mode == kSourceZero ? 0.0f : a.f[kFHistBwEntropyRawPerAgent][hist_u];
    }
    if (has_f(a, kFHistBwTau) && has_f(a, kFLiveBwTau)) {
      a.f[kFLiveBwTau][live_u] = bw_source_mode == kSourceZero ? 0.0f : a.f[kFHistBwTau][hist_u];
    }
    if (has_f(a, kFHistBwKappa) && has_f(a, kFLiveBwKappa)) {
      a.f[kFLiveBwKappa][live_u] = bw_source_mode == kSourceZero ? 0.0f : a.f[kFHistBwKappa][hist_u];
    }
    if (has_l(a, kLHistBwValidCount) && has_l(a, kLLiveBwValidCount)) {
      a.l[kLLiveBwValidCount][live_u] = bw_source_mode == kSourceZero ? 0 : a.l[kLHistBwValidCount][hist_u];
    }
    if (has_l(a, kLHistBwLatentCount) && has_l(a, kLLiveBwLatentCount)) {
      a.l[kLLiveBwLatentCount][live_u] = bw_source_mode == kSourceZero ? 0 : a.l[kLHistBwLatentCount][hist_u];
    }
  }
  if (threadIdx.x == 0 && has_f(a, kFHistBwOldLogprobs) && has_f(a, kFLiveBwOldLogprob)) {
    a.f[kFLiveBwOldLogprob][e] = bw_source_mode == kSourceZero ? 0.0f : a.f[kFHistBwOldLogprobs][src_row];
  }
}

__device__ void restore_runtime_state_from_history_parallel(
    const PackedAbi& src,
    const PackedAbi& dst,
    int float_base,
    int long_base,
    int int_base,
    int bool_base,
    int src_row,
    int dst_e,
    int src_rows,
    int dst_rows) {
#define RESTORE_STATE_F(offset, dst_tensor) \
  copy_float_row_between_parallel(dst, (dst_tensor), src, float_base + (offset), dst_e, src_row, dst_rows, src_rows)
  RESTORE_STATE_F(0, kFStateUavPos);
  RESTORE_STATE_F(1, kFStateUavVel);
  RESTORE_STATE_F(2, kFStateUavEnergy);
  RESTORE_STATE_F(3, kFStateUavQueue);
  RESTORE_STATE_F(4, kFStateGuPos);
  RESTORE_STATE_F(5, kFStateGuClusterCenters);
  RESTORE_STATE_F(6, kFStateGuClusterCounts);
  RESTORE_STATE_F(7, kFStateGuQueue);
  RESTORE_STATE_F(8, kFStateSatQueue);
  RESTORE_STATE_F(9, kFStateSatPos);
  RESTORE_STATE_F(10, kFStateSatVel);
  RESTORE_STATE_F(11, kFStateLastSatConnectionCounts);
  RESTORE_STATE_F(12, kFStateArrivalRef);
  RESTORE_STATE_F(13, kFStateEffectiveArrivalRate);
  RESTORE_STATE_F(14, kFStateArrivalBaseScale);
  RESTORE_STATE_F(15, kFStateGuEma);
  RESTORE_STATE_F(16, kFStateUavEma);
  RESTORE_STATE_F(17, kFStateSatEma);
  RESTORE_STATE_F(18, kFStateLastArrivalRateVec);
  RESTORE_STATE_F(19, kFStateGuDeadlineSteps);
  RESTORE_STATE_F(20, kFStateLastGuArrival);
  RESTORE_STATE_F(21, kFStateLastGuOutflow);
  RESTORE_STATE_F(22, kFStateGuDrop);
  RESTORE_STATE_F(23, kFStateUavDrop);
  RESTORE_STATE_F(24, kFStateSatDrop);
  RESTORE_STATE_F(25, kFStateLastAccessInterferenceByUav);
  RESTORE_STATE_F(26, kFStateLastBwFractionByUavGu);
  RESTORE_STATE_F(27, kFStateLastGuToUavInflowByUav);
  RESTORE_STATE_F(28, kFStateLastUavToSatOutflowMatrix);
  RESTORE_STATE_F(29, kFStateLastSelectedMaskByUavSat);
  RESTORE_STATE_F(30, kFStateLastSatProcessed);
  RESTORE_STATE_F(31, kFStateUrgencyRisk);
  RESTORE_STATE_F(32, kFStateDownstreamPressure);
  RESTORE_STATE_F(33, kFStateServiceGapRisk);
  RESTORE_STATE_F(34, kFStateDeadlineSlack);
  RESTORE_STATE_F(35, kFStateDeadlineRisk);
  RESTORE_STATE_F(36, kFStateServiceGap);
  RESTORE_STATE_F(37, kFStateDeadlineAge);
  RESTORE_STATE_F(38, kFStateLastExecAccel);
  RESTORE_STATE_F(39, kFStateLastPolicyAccel);
  RESTORE_STATE_F(40, kFStateAvoidanceEtaEff);
  RESTORE_STATE_F(41, kFStateLastAvoidanceEtaExec);
  RESTORE_STATE_F(42, kFStateDopplerResidual);
  RESTORE_STATE_F(43, kFStatePrevQueueSumGu);
  RESTORE_STATE_F(44, kFStatePrevQueueSumUav);
  RESTORE_STATE_F(45, kFStatePrevQueueSumSat);
  RESTORE_STATE_F(46, kFStatePrevQNormActive);
  RESTORE_STATE_F(47, kFStatePrevGuQueueVec);
  RESTORE_STATE_F(48, kFStatePrevUavQueueVec);
  RESTORE_STATE_F(49, kFStatePrevSatQueueVec);
  RESTORE_STATE_F(50, kFStateHotspotMemberMask);
#undef RESTORE_STATE_F
  copy_long_row_between_parallel(dst, kLStateLastSatSelectionMatrix, src, long_base + 0, dst_e, src_row, dst_rows, src_rows);
  copy_int_row_between_parallel(dst, kIStatePrevAssociation, src, int_base + 0, dst_e, src_row, dst_rows, src_rows);
  copy_int_row_between_parallel(dst, kIStateLastAssociation, src, int_base + 1, dst_e, src_row, dst_rows, src_rows);
  copy_int_row_between_parallel(dst, kIStateHotspotActiveIdx, src, int_base + 2, dst_e, src_row, dst_rows, src_rows);
  copy_int_row_between_parallel(dst, kIStateHotspotSubsetCount, src, int_base + 3, dst_e, src_row, dst_rows, src_rows);
  copy_int_row_between_parallel(dst, kIStateTrafficResetStep, src, int_base + 4, dst_e, src_row, dst_rows, src_rows);
  copy_int_row_between_parallel(dst, kIStateTrafficResetOrdinal, src, int_base + 5, dst_e, src_row, dst_rows, src_rows);
  copy_int_row_between_parallel(dst, kIStateEpisodeIdx, src, int_base + 6, dst_e, src_row, dst_rows, src_rows);
  copy_int_row_between_parallel(dst, kIStateT, src, int_base + 7, dst_e, src_row, dst_rows, src_rows);
  copy_int_row_between_parallel(dst, kIStateGlobalStep, src, int_base + 8, dst_e, src_row, dst_rows, src_rows);
  copy_bool_row_between_parallel(dst, kBStateHotspotMemberMask, src, bool_base + 0, dst_e, src_row, dst_rows, src_rows);
}

__device__ void restore_stage_from_history_parallel(
    const PackedAbi& src,
    const PackedAbi& dst,
    int float_base,
    int long_base,
    int bool_base,
    int src_row,
    int dst_e,
    int src_rows,
    int dst_rows,
    int dst_stage) {
  for (int field = 0; field <= kSfUsSatQueueAll; ++field) {
    copy_float_row_between_parallel(dst, stage_f(dst_stage, field), src, float_base + field, dst_e, src_row, dst_rows, src_rows);
  }
  for (int field = 0; field <= kSlActiveSatIds; ++field) {
    copy_long_row_between_parallel(dst, stage_l(dst_stage, field), src, long_base + field, dst_e, src_row, dst_rows, src_rows);
  }
  copy_bool_row_between_parallel(dst, stage_b(dst_stage, 0), src, bool_base + 0, dst_e, src_row, dst_rows, src_rows);
  copy_bool_row_between_parallel(dst, stage_b(dst_stage, 1), src, bool_base + 1, dst_e, src_row, dst_rows, src_rows);
}

__device__ void restore_bw_cache_from_history_parallel(
    const PackedAbi& src,
    const PackedAbi& dst,
    int src_row,
    int dst_e,
    int src_rows,
    int dst_rows) {
#define RESTORE_BW_CACHE_F(offset, dst_tensor) \
  copy_float_row_between_parallel(dst, (dst_tensor), src, kFHistBwRuntimeCacheBase + (offset), dst_e, src_row, dst_rows, src_rows)
  RESTORE_BW_CACHE_F(0, kFBwInputBase + 0);
  RESTORE_BW_CACHE_F(1, kFBwInputBase + 1);
  RESTORE_BW_CACHE_F(2, kFBwInputBase + 2);
  RESTORE_BW_CACHE_F(3, kFBwInputBase + 3);
  RESTORE_BW_CACHE_F(4, kFBwInputBase + 4);
  RESTORE_BW_CACHE_F(5, kFBwInputBase + 5);
  RESTORE_BW_CACHE_F(6, kFBwInputBase + 6);
  RESTORE_BW_CACHE_F(7, kFBwInputBase + 7);
  RESTORE_BW_CACHE_F(8, kFBwInputBase + 8);
  RESTORE_BW_CACHE_F(9, kFBwInputBase + 9);
#undef RESTORE_BW_CACHE_F
  copy_long_row_between_parallel(dst, kLBwInputBase + 2, src, kLHistBwRuntimeCacheBase + 0, dst_e, src_row, dst_rows, src_rows);
  copy_long_row_between_parallel(dst, kLBwInputBase + 0, src, kLHistBwRuntimeCacheBase + 1, dst_e, src_row, dst_rows, src_rows);
  copy_long_row_between_parallel(dst, kLBwInputBase + 1, src, kLHistBwRuntimeCacheBase + 2, dst_e, src_row, dst_rows, src_rows);
  copy_long_row_between_parallel(dst, kLBwInputBase + 3, src, kLHistBwRuntimeCacheBase + 3, dst_e, src_row, dst_rows, src_rows);
  copy_long_row_between_parallel(dst, kLBwInputBase + 4, src, kLHistBwRuntimeCacheBase + 4, dst_e, src_row, dst_rows, src_rows);
  copy_bool_row_between_parallel(dst, kBBwInputCandidateMask, src, kBHistBwRuntimeCacheBase + 0, dst_e, src_row, dst_rows, src_rows);
}

__device__ void restore_local_obs_from_history_parallel(
    const PackedAbi& src,
    const PackedAbi& dst,
    int stage_id,
    int src_row,
    int dst_e,
    int src_rows,
    int dst_rows) {
  const int ucount = static_cast<int>(ip(dst, kParamNumUav));
  const int src_local_rows = src_rows * max(ucount, 1);
  const int dst_local_rows = dst_rows * max(ucount, 1);
  for (int u = 0; u < ucount; ++u) {
    const int src_local = src_row * ucount + u;
    const int dst_local = dst_e * ucount + u;
    if (stage_id == 0) {
      for (int field = 0; field < 5; ++field) {
        copy_float_row_between_parallel(dst, kFLiveAccelObs0 + field, src, kFHistAccelLocal + field, dst_local, src_local, dst_local_rows, src_local_rows);
      }
      for (int field = 0; field < 3; ++field) {
        copy_bool_row_between_parallel(dst, kBLiveAccelObs0 + field, src, kBHistAccelLocal + field, dst_local, src_local, dst_local_rows, src_local_rows);
      }
    } else if (stage_id == 1) {
      for (int field = 0; field < 4; ++field) {
        copy_float_row_between_parallel(dst, kFLiveSatObs + field, src, kFHistSatLocal + field, dst_local, src_local, dst_local_rows, src_local_rows);
      }
      for (int field = 0; field < 2; ++field) {
        copy_bool_row_between_parallel(dst, kBLiveSatObs + field, src, kBHistSatLocal + field, dst_local, src_local, dst_local_rows, src_local_rows);
      }
      copy_long_row_between_parallel(dst, kLLiveSatCandidateIds, src, kLHistSatCandidateIds, dst_local, src_local, dst_local_rows, src_local_rows);
    } else {
      for (int field = 0; field < 3; ++field) {
        copy_float_row_between_parallel(dst, kFLiveBwObs + field, src, kFHistBwLocal + field, dst_local, src_local, dst_local_rows, src_local_rows);
      }
      for (int field = 0; field < 3; ++field) {
        copy_bool_row_between_parallel(dst, kBLiveBwObs + field, src, kBHistBwLocal + field, dst_local, src_local, dst_local_rows, src_local_rows);
      }
    }
  }
}

__global__ void prepare_branch_replay_from_history_kernel(
    const int64_t* history_rows,
    int64_t branch_count64,
    int64_t stage_id64) {
  const PackedAbi& src = cBranchSourceAbi;
  const PackedAbi& dst = cBranchTargetAbi;
  const int e = blockIdx.x;
  const int branch_count = static_cast<int>(branch_count64);
  if (e >= branch_count) return;
  const int stage_id = static_cast<int>(stage_id64);
  const int src_ecount = static_cast<int>(ip(src, kParamNumEnvs));
  const int src_capacity = static_cast<int>(ip(src, kParamHistoryCapacity, 0));
  const int src_rows = max(src_capacity, 0) * max(src_ecount, 0);
  const int dst_rows = max(static_cast<int>(ip(dst, kParamNumEnvs)), 0);
  if (src_rows <= 0 || dst_rows <= 0 || e >= dst_rows) return;
  const int src_row = static_cast<int>(history_rows[e]);
  if (src_row < 0 || src_row >= src_rows) return;

  if (stage_id == 0) {
    restore_runtime_state_from_history_parallel(src, dst, kFHistAccelRuntimeStateBase, kLHistAccelRuntimeStateBase, kIHistAccelRuntimeStateBase, kBHistAccelRuntimeStateBase, src_row, e, src_rows, dst_rows);
    __syncthreads();
    restore_stage_from_history_parallel(src, dst, kFHistAccelRuntimeStageBase, kLHistAccelRuntimeStageBase, kBHistAccelRuntimeStageBase, src_row, e, src_rows, dst_rows, 0);
  } else if (stage_id == 1) {
    restore_runtime_state_from_history_parallel(src, dst, kFHistSatRuntimeStateBase, kLHistSatRuntimeStateBase, kIHistSatRuntimeStateBase, kBHistSatRuntimeStateBase, src_row, e, src_rows, dst_rows);
    __syncthreads();
    restore_stage_from_history_parallel(src, dst, kFHistSatRuntimeStageBase, kLHistSatRuntimeStageBase, kBHistSatRuntimeStageBase, src_row, e, src_rows, dst_rows, 2);
  } else {
    restore_runtime_state_from_history_parallel(src, dst, kFHistBwRuntimeStateBase, kLHistBwRuntimeStateBase, kIHistBwRuntimeStateBase, kBHistBwRuntimeStateBase, src_row, e, src_rows, dst_rows);
    __syncthreads();
    restore_bw_cache_from_history_parallel(src, dst, src_row, e, src_rows, dst_rows);
    __syncthreads();
    restore_stage_from_history_parallel(src, dst, kFHistBwRuntimeStageBase, kLHistBwRuntimeStageBase, kBHistBwRuntimeStageBase, src_row, e, src_rows, dst_rows, 3);
  }
  __syncthreads();
  restore_local_obs_from_history_parallel(src, dst, stage_id, src_row, e, src_rows, dst_rows);
  if (threadIdx.x == 0) {
    if (has_i(dst, kIRandomStepTensor)) dst.i[kIRandomStepTensor][0] = 0;
    if (has_i(dst, kIRandomResetCount)) dst.i[kIRandomResetCount][e] = 0;
    if (has_i(dst, kIMarker)) dst.i[kIMarker][0] = 30 + stage_id;
  }
}

__global__ void prepare_initial_accel_live_kernel(int64_t slot64, int64_t active_idx) {
  const PackedAbi& a = cLiveAbi;
  const int e = blockIdx.x;
  if (e >= ip(a, kParamNumEnvs)) return;
  const int slot = static_cast<int>(slot64);
  const int current_t = has_i(a, kIStateT) ? a.i[kIStateT][e] : 0;
  sync_state_orbit_from_t_parallel(a, e, current_t);
  __syncthreads();
  copy_random_step_tapes_parallel(a, 0, e);
  __syncthreads();
  prepare_stage_from_state_parallel(a, static_cast<int>(active_idx), e, 0);
  __syncthreads();
  write_world_from_stage_parallel(a, static_cast<int>(active_idx), kFHistAccelWorld, kBHistAccelWorld, hist_env_row(a, slot, e), e);
  __syncthreads();
  write_accel_obs_parallel(a, static_cast<int>(active_idx), static_cast<int>(active_idx), e);
  __syncthreads();
  copy_runtime_state_snapshot_to_history_parallel(
      a,
      kFHistAccelRuntimeStateBase,
      kLHistAccelRuntimeStateBase,
      kIHistAccelRuntimeStateBase,
      kBHistAccelRuntimeStateBase,
      slot,
      e);
  copy_stage_snapshot_to_history_parallel(
      a,
      kFHistAccelRuntimeStageBase,
      kLHistAccelRuntimeStageBase,
      kBHistAccelRuntimeStageBase,
      slot,
      e,
      static_cast<int>(active_idx));
  if (threadIdx.x == 0 && has_i(a, kIMarker)) a.i[kIMarker][0] = 1;
}

__global__ void accel_to_sat_live_kernel(int64_t slot, int64_t active_idx, int64_t accel_source_mode) {
  const PackedAbi& a = cLiveAbi;
  const int e = blockIdx.x;
  if (e >= ip(a, kParamNumEnvs)) return;
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int cur = static_cast<int>(active_idx);
  const int sat_stage = 2;
  copy_stage_env_parallel(a, cur, sat_stage, e, 1);
  __syncthreads();
  const float tau = fp(a, kFpAccelTau0, fp(a, kFpTau0, 1.0f));
  const float amax = positive_config_scale(fp(a, kFpAccelAMax, 1.0f));
  const float vmax = positive_config_scale(fp(a, kFpAccelVMax, fp(a, kFpVMax, 1.0f)));
  const float map_size = positive_config_scale(fp(a, kFpAccelMapSize, fp(a, kFpMapSize, 1.0f)));
  const bool native_safety = ip(a, kParamSafetyShieldNative, 0) != 0;
  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    float policy_ax = 0.0f;
    float policy_ay = 0.0f;
    if (accel_source_mode != kSourceZero) {
      policy_ax = clampf_device(a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 0], -1.0f, 1.0f) * amax;
      policy_ay = clampf_device(a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 1], -1.0f, 1.0f) * amax;
      project_l2_device(&policy_ax, &policy_ay, amax);
    }
    float ax = policy_ax;
    float ay = policy_ay;
    if (native_safety) {
      if (u == 0 && has_f(a, kFStateLastAvoidanceEtaExec)) {
        a.f[kFStateLastAvoidanceEtaExec][e] = 0.0f;
      }
    } else {
      accel_apply_avoidance(a, cur, e, u, policy_ax, policy_ay, &ax, &ay);
    }
    a.f[kFStateLastPolicyAccel][(e * ucount + u) * 2 + 0] = policy_ax;
    a.f[kFStateLastPolicyAccel][(e * ucount + u) * 2 + 1] = policy_ay;
    a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 0] = ax;
    a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 1] = ay;
  }
  __syncthreads();
  if (native_safety && threadIdx.x == 0) {
    native_safety_project_env(a, cur, e);
  }
  __syncthreads();
  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    const float ax = a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 0];
    const float ay = a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 1];
    float vx = a.f[stage_f(cur, kSfUavVel)][(e * ucount + u) * 2 + 0] + ax * tau;
    float vy = a.f[stage_f(cur, kSfUavVel)][(e * ucount + u) * 2 + 1] + ay * tau;
    const float speed = sqrtf(vx * vx + vy * vy);
    if (speed > vmax) {
      const float scale = vmax / dynamics_denominator(speed);
      vx *= scale;
      vy *= scale;
    }
    float px = a.f[stage_f(cur, kSfUavPos)][(e * ucount + u) * 2 + 0] + vx * tau;
    float py = a.f[stage_f(cur, kSfUavPos)][(e * ucount + u) * 2 + 1] + vy * tau;
    if (ip(a, kParamBoundaryMode) == 1) {
      if (px < 0.0f) {
        px = -px;
        vx = -vx;
      } else if (px > map_size) {
        px = 2.0f * map_size - px;
        vx = -vx;
      }
      if (py < 0.0f) {
        py = -py;
        vy = -vy;
      } else if (py > map_size) {
        py = 2.0f * map_size - py;
        vy = -vy;
      }
    }
    px = clampf_device(px, 0.0f, map_size);
    py = clampf_device(py, 0.0f, map_size);
    a.f[stage_f(sat_stage, kSfUavPos)][(e * ucount + u) * 2 + 0] = px;
    a.f[stage_f(sat_stage, kSfUavPos)][(e * ucount + u) * 2 + 1] = py;
    a.f[stage_f(sat_stage, kSfUavVel)][(e * ucount + u) * 2 + 0] = vx;
    a.f[stage_f(sat_stage, kSfUavVel)][(e * ucount + u) * 2 + 1] = vy;
    if (ip(a, kParamEnergyEnabled)) {
      const float cost = (sqrtf(ax * ax + ay * ay) * 0.01f + sqrtf(vx * vx + vy * vy) * 0.001f) * tau;
      a.f[stage_f(sat_stage, kSfUavEnergy)][e * ucount + u] =
          fmaxf(a.f[stage_f(sat_stage, kSfUavEnergy)][e * ucount + u] - cost, 0.0f);
    }
  }
  __syncthreads();
  refresh_stage_derived_parallel(a, sat_stage, e);
  __syncthreads();
  write_world_from_stage_parallel(a, sat_stage, kFHistSatWorld, kBHistSatWorld, hist_env_row(a, static_cast<int>(slot), e), e);
  __syncthreads();
  write_sat_live_obs_parallel(a, sat_stage, e);
  __syncthreads();
  copy_runtime_state_snapshot_to_history_parallel(
      a,
      kFHistSatRuntimeStateBase,
      kLHistSatRuntimeStateBase,
      kIHistSatRuntimeStateBase,
      kBHistSatRuntimeStateBase,
      static_cast<int>(slot),
      e);
  copy_stage_snapshot_to_history_parallel(
      a,
      kFHistSatRuntimeStageBase,
      kLHistSatRuntimeStageBase,
      kBHistSatRuntimeStageBase,
      static_cast<int>(slot),
      e,
      sat_stage);
  if (threadIdx.x == 0 && has_i(a, kIMarker)) a.i[kIMarker][0] = 2;
  (void)slot;
}

__global__ void sat_to_bw_live_kernel(int64_t slot, int64_t sat_source_mode) {
  const PackedAbi& a = cLiveAbi;
  const int e = blockIdx.x;
  if (e >= ip(a, kParamNumEnvs)) return;
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  const int subset_count = static_cast<int>(ip(a, kParamSubsetCount));
  const int visible = sat_visible_width(a);
  const int active = active_width(a);
  const int bw_stage = 3;
  const int slot_i = static_cast<int>(slot);
  const int hist_cap = max(static_cast<int>(ip(a, kParamHistoryCapacity, 0)), 0);
  const int hist_rows = hist_cap * max(static_cast<int>(ip(a, kParamNumEnvs)), 0);
  const int sat_interval = max(static_cast<int>(ip(a, kParamSatDecisionInterval, 1)), 1);
  const int sat_start_slot = sat_macro_start_slot_for_env(a, slot_i, e);
  const bool sat_macro_continue =
      sat_interval > 1 && slot_i > 0 && sat_start_slot != slot_i && hist_rows > 0;
  const int sat_src_row = sat_macro_continue ? hist_env_row(a, sat_start_slot, e) : -1;
  copy_stage_env_parallel(a, 2, bw_stage, e, 2);
  __syncthreads();
  if (sat_macro_continue && sat_src_row >= 0 && sat_src_row < hist_rows && has_l(a, kLHistSatActionIndices)) {
    for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
      const int live_u = e * ucount + u;
      const int hist_u = sat_src_row * ucount + u;
      if (has_l(a, kLHistSatActions)) {
        a.l[kLLiveSatSubsetIndex][live_u] = a.l[kLHistSatActions][hist_u];
      } else if (sat_source_mode == kSourceZero) {
        a.l[kLLiveSatSubsetIndex][live_u] = 0;
      }
      if (has_f(a, kFLiveSatOldLogprobPerAgent)) {
        a.f[kFLiveSatOldLogprobPerAgent][live_u] =
            (sat_source_mode != kSourceZero && has_f(a, kFHistSatOldLogprobsPerAgent))
                ? a.f[kFHistSatOldLogprobsPerAgent][hist_u]
                : 0.0f;
      }
      if (has_f(a, kFLiveSatEntropyPerAgent)) a.f[kFLiveSatEntropyPerAgent][live_u] = 0.0f;
    }
    for (int idx = threadIdx.x; idx < ucount * select_k; idx += blockDim.x) {
      const int u = idx / select_k;
      const int k = idx - u * select_k;
      const int64_t sid = a.l[kLHistSatActionIndices][(sat_src_row * ucount + u) * select_k + k];
      a.l[stage_l(bw_stage, kSlSatSelectionMatrix)][(e * ucount + u) * select_k + k] = sid;
      a.l[kLBwInputBase + 3][(e * ucount + u) * select_k + k] = sid;
      if (has_l(a, kLLiveSatActionIndices)) {
        a.l[kLLiveSatActionIndices][(e * ucount + u) * select_k + k] = sid;
      }
    }
  } else {
    for (int idx = threadIdx.x; idx < ucount * select_k; idx += blockDim.x) {
      const int u = idx / select_k;
      const int k = idx - u * select_k;
      if (sat_source_mode == kSourceZero && k == 0) {
        a.l[kLLiveSatSubsetIndex][e * ucount + u] = 0;
        if (has_f(a, kFLiveSatOldLogprobPerAgent)) a.f[kFLiveSatOldLogprobPerAgent][e * ucount + u] = 0.0f;
        if (has_f(a, kFLiveSatEntropyPerAgent)) a.f[kFLiveSatEntropyPerAgent][e * ucount + u] = 0.0f;
      }
      int64_t sid = -1;
      if (sat_source_mode == kSourceZero) {
        if (k == 0) {
          float best_dist2 = 3.402823466e38f;
          for (int j = 0; j < visible; ++j) {
            const int visible_idx = (e * ucount + u) * visible + j;
            if (!a.b[stage_b(2, 1)][visible_idx]) continue;
            const int64_t candidate = a.l[stage_l(2, kSlVisibleIds)][visible_idx];
            if (candidate < 0 || candidate >= ip(a, kParamNumSat)) continue;
            float rel[3];
            float relv[3];
            sat_rel_for_stage_us(a, 2, e, u, static_cast<int>(candidate), rel, relv);
            const float dist2 = rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2];
            if (dist2 < best_dist2 || (dist2 == best_dist2 && (sid < 0 || candidate < sid))) {
              best_dist2 = dist2;
              sid = candidate;
            }
          }
        }
      } else {
        const int64_t action = a.l[kLLiveSatSubsetIndex][e * ucount + u];
        int64_t member = -1;
        if (action >= 0 && action < subset_count) member = a.l[kLMainSatSubsetMembersBase][action * select_k + k];
        if (member >= 0 && member < visible) {
          const int visible_idx = (e * ucount + u) * visible + static_cast<int>(member);
          const bool valid_member = a.b[stage_b(2, 1)][visible_idx];
          sid = valid_member ? a.l[stage_l(2, kSlVisibleIds)][visible_idx] : -1;
        }
      }
      a.l[stage_l(bw_stage, kSlSatSelectionMatrix)][(e * ucount + u) * select_k + k] = sid;
      a.l[kLBwInputBase + 3][(e * ucount + u) * select_k + k] = sid;
      if (has_l(a, kLLiveSatActionIndices)) {
        a.l[kLLiveSatActionIndices][(e * ucount + u) * select_k + k] = sid;
      }
    }
  }
  __syncthreads();
  refresh_stage_derived_parallel(a, bw_stage, e);
  __syncthreads();
  write_world_from_stage_parallel(a, bw_stage, kFHistBwWorld, kBHistBwWorld, hist_env_row(a, static_cast<int>(slot), e), e);
  __syncthreads();
  write_bw_live_obs_parallel(a, bw_stage, e);
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int users_obs = static_cast<int>(ip(a, kParamUsersObsMax));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  for (int g = threadIdx.x; g < gu; g += blockDim.x) {
    a.l[kLBwInputBase + 0][e * gu + g] = a.l[stage_l(bw_stage, kSlAssoc)][e * gu + g];
    a.l[kLBwInputBase + 1][e * gu + g] = a.l[stage_l(bw_stage, kSlPrevAssociation)][e * gu + g];
  }
  for (int idx_local = threadIdx.x; idx_local < ucount * users_obs; idx_local += blockDim.x) {
    const int idx = e * ucount * users_obs + idx_local;
    a.l[kLBwInputBase + 2][idx] = a.l[stage_l(bw_stage, kSlCandidateIndices)][idx];
    a.b[kBBwInputCandidateMask][idx] = a.b[stage_b(bw_stage, 0)][idx];
  }
  for (int idx_local = threadIdx.x; idx_local < ucount * gu; idx_local += blockDim.x) {
    a.f[kFBwInputBase + 0][e * ucount * gu + idx_local] =
        a.f[stage_f(bw_stage, kSfBwValidFlag)][e * ucount * gu + idx_local];
  }
  for (int idx_local = threadIdx.x; idx_local < gu * ucount; idx_local += blockDim.x) {
    a.f[kFBwInputBase + 1][e * gu * ucount + idx_local] = a.f[stage_f(bw_stage, kSfAccessGainMatrix)][e * gu * ucount + idx_local];
  }
  for (int aidx = threadIdx.x; aidx < active; aidx += blockDim.x) {
    a.l[kLBwInputBase + 4][e * active + aidx] = a.l[stage_l(bw_stage, kSlActiveSatIds)][e * active + aidx];
  }
  for (int idx_local = threadIdx.x; idx_local < ucount * active; idx_local += blockDim.x) {
    const int idx = e * ucount * active + idx_local;
    a.f[kFBwInputBase + 2][idx] = a.f[stage_f(bw_stage, kSfUsGainActive)][idx];
    a.f[kFBwInputBase + 3][idx] = a.f[stage_f(bw_stage, kSfUsNuEffActive)][idx];
    a.f[kFBwInputBase + 4][idx] = a.f[stage_f(bw_stage, kSfUsValidFlagActive)][idx];
  }
  for (int idx_local = threadIdx.x; idx_local < sat * 3; idx_local += blockDim.x) {
    a.f[kFBwInputBase + 5][e * sat * 3 + idx_local] = a.f[stage_f(bw_stage, kSfSatPos)][e * sat * 3 + idx_local];
  }
  for (int idx_local = threadIdx.x; idx_local < ucount * 3; idx_local += blockDim.x) {
    a.f[kFBwInputBase + 6][e * ucount * 3 + idx_local] = a.f[stage_f(bw_stage, kSfUavEcefAll)][e * ucount * 3 + idx_local];
  }
  for (int idx_local = threadIdx.x; idx_local < ucount * 2; idx_local += blockDim.x) {
    a.f[kFBwInputBase + 7][e * ucount * 2 + idx_local] = a.f[stage_f(bw_stage, kSfUavPos)][e * ucount * 2 + idx_local];
    a.f[kFBwInputBase + 8][e * ucount * 2 + idx_local] = a.f[stage_f(bw_stage, kSfUavVel)][e * ucount * 2 + idx_local];
  }
  for (int idx_local = threadIdx.x; idx_local < gu * 2; idx_local += blockDim.x) {
    a.f[kFBwInputBase + 9][e * gu * 2 + idx_local] = a.f[stage_f(bw_stage, kSfGuPos)][e * gu * 2 + idx_local];
  }
  __syncthreads();
  copy_bw_runtime_snapshot_to_history_parallel(a, static_cast<int>(slot), e, bw_stage);
  if (threadIdx.x == 0 && has_i(a, kIMarker)) a.i[kIMarker][0] = 3;
  (void)slot;
}

__global__ void finish_commit_prepare_live_kernel(
    int64_t slot64,
    int64_t active_idx,
    bool rollout_tail,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode,
    float* finish_profile_out,
    int finish_profile_stride) {
  const PackedAbi& a = cLiveAbi;
  const int e = blockIdx.x;
  if (e >= ip(a, kParamNumEnvs)) return;
  const int slot = static_cast<int>(slot64);
  const int ucount = static_cast<int>(ip(a, kParamNumUav));
  const int gu = static_cast<int>(ip(a, kParamNumGu));
  const int users_obs = static_cast<int>(ip(a, kParamUsersObsMax));
  const int sat = static_cast<int>(ip(a, kParamNumSat));
  const int active = active_width(a);
  const int select_k = static_cast<int>(ip(a, kParamSatNumSelect));
  const int bw_stage = 3;
  const float qmax_gu = positive_config_scale(fp(a, kFpQueueMaxGu, 1.0f));
  const float qmax_uav = positive_config_scale(fp(a, kFpQueueMaxUav, 1.0f));
  const float qmax_sat = positive_config_scale(fp(a, kFpQueueMaxSat, 1.0f));
  const float tau = fp(a, kFpBwTau0, fp(a, kFpTau0, 1.0f));
  const int row = hist_env_row(a, slot, e);
  __shared__ float sh_service_sum;
  __shared__ float sh_arrival_sum;
  __shared__ float sh_drop_sum;
  __shared__ float sh_backlog;
  __shared__ float sh_sat_processed;
  __shared__ float sh_close_risk;
  __shared__ float sh_term_close_risk;
  __shared__ float sh_accel_norm;
  __shared__ float sh_danger_active_rate;
  __shared__ float sh_intervention_norm;
  __shared__ float sh_intervention_rate;
  __shared__ float sh_intervention_norm_top1;
  __shared__ float sh_service_ratio;
  __shared__ float sh_drop_ratio;
  __shared__ float sh_queue_penalty;
  __shared__ float sh_reward;
  __shared__ bool sh_done;
  __shared__ bool sh_terminated;
  __shared__ bool sh_truncated;
  __shared__ float sh_reduce[128];
  __shared__ float sh_backhaul_sum;
  __shared__ float sh_gu_drop_sum;
  __shared__ float sh_uav_drop_sum;
  __shared__ float sh_sat_drop_sum;
  __shared__ float sh_expire_sum;
  __shared__ float sh_q_gu;
  __shared__ float sh_q_uav;
  __shared__ float sh_q_sat;
  __shared__ float sh_arrival_ref;
  __shared__ float sh_b_pre_steps;
  __shared__ float sh_x_acc;
  __shared__ float sh_x_rel;
  __shared__ float sh_g_pre;
  __shared__ float sh_d_pre;
  __shared__ float sh_processed_ratio_eval;
  __shared__ float sh_drop_ratio_eval;
  __shared__ float sh_pre_backlog_steps_eval;
  __shared__ float sh_d_sys_report;
  __shared__ float sh_sat_overlap_eval;
  __shared__ float sh_overflow_risk_mean;
  __shared__ float sh_downstream_pressure_mean;
  __shared__ float sh_service_gap_mean;
  __shared__ float sh_service_gap_risk_mean;
  __shared__ float sh_weighted_delta;
  __shared__ float sh_weighted_level;
  __shared__ float sh_gu_queue_level;
  __shared__ float sh_system_queue_level;
  __shared__ float sh_gu_service_queue;
  __shared__ float sh_reward_raw;
  __shared__ unsigned long long sh_finish_profile_clock;
  __shared__ unsigned long long sh_refresh_profile_clock;
  __shared__ unsigned long long sh_accel_obs_profile_clock;
  extern __shared__ float sh_finish_dynamic[];
  float* sh_uav_inflow_cache = sh_finish_dynamic;
  float* sh_uav_total_rate_cache = sh_uav_inflow_cache + max(ucount, 0);
  float* sh_uav_outflow_cache = sh_uav_total_rate_cache + max(ucount, 0);
  float* sh_sat_incoming_cache = sh_uav_outflow_cache + max(ucount, 0);
  float* sh_sat_processed_cache = sh_sat_incoming_cache + max(sat, 0);
  float* sh_access_interference_cache = sh_sat_processed_cache + max(sat, 0);
  float* sh_accel_cell_summary_cache = sh_access_interference_cache + max(ucount, 0);
  float* sh_refresh_cost_cache = sh_accel_cell_summary_cache + max(ucount, 0) * kAccelCellDim;
  float* sh_last_route_cost_cache =
      sh_refresh_cost_cache + refresh_cost_cache_size_for_dims(max(sat, 0), max(ucount, 0), max(gu, 0));
  if (finish_profile_out != nullptr && threadIdx.x == 0) {
    sh_finish_profile_clock = clock64();
  }
  if (finish_profile_out != nullptr) {
    __syncthreads();
  }

  float local_arrival_sum = 0.0f;
  for (int g = threadIdx.x; g < gu; g += blockDim.x) {
    const float arrival = state_arrival_bits(a, e, g, slot);
    local_arrival_sum += arrival;
    a.f[kFStateLastGuArrival][e * gu + g] = arrival;
    a.f[kFStateLastGuOutflow][e * gu + g] = 0.0f;
  }
  const float total_arrival_sum = block_reduce_sum_128(local_arrival_sum, sh_reduce);
  if (threadIdx.x == 0) sh_arrival_sum = total_arrival_sum;
  __syncthreads();

  write_flow_proxy_scores_parallel(a, bw_stage, e, row, static_cast<int>(bw_source_mode));
  __syncthreads();
  finish_profile_mark(
      finish_profile_out,
      finish_profile_stride,
      e,
      kFinishProfArrivalAndFlowProxy,
      &sh_finish_profile_clock);

  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    sh_access_interference_cache[u] =
        access_interference_for_u(a, bw_stage, e, u, static_cast<int>(bw_source_mode));
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < ucount * gu; idx += blockDim.x) {
    const int u = idx / gu;
    const int gid = idx - u * gu;
    if (!bw_gu_assoc_match(a, bw_stage, e, u, gid)) continue;
    const float rate = access_rate_for_gu_with_interference(
        a, bw_stage, e, u, gid, static_cast<int>(bw_source_mode), sh_access_interference_cache[u]);
    if (rate > 0.0f) atomicAdd(&a.f[kFStateLastGuOutflow][e * gu + gid], rate);
  }
  __syncthreads();
  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    if (has_f(a, kFStateLastAccessInterferenceByUav)) {
      a.f[kFStateLastAccessInterferenceByUav][e * ucount + u] = sh_access_interference_cache[u];
    }
  }
  for (int idx = threadIdx.x; idx < ucount * gu; idx += blockDim.x) {
    const int u = idx / gu;
    const int g = idx - u * gu;
    if (has_f(a, kFStateLastBwFractionByUavGu)) {
      const int64_t assoc64 = a.l[stage_l(bw_stage, kSlAssoc)][e * gu + g];
      a.f[kFStateLastBwFractionByUavGu][(e * ucount + u) * gu + g] =
          assoc64 == u ? bw_band_fraction_for_gu(a, bw_stage, e, g, static_cast<int>(bw_source_mode)) : 0.0f;
    }
  }
  finish_profile_mark(
      finish_profile_out,
      finish_profile_stride,
      e,
      kFinishProfAccess,
      &sh_finish_profile_clock);
  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    sh_uav_inflow_cache[u] = 0.0f;
    sh_uav_total_rate_cache[u] = 0.0f;
    sh_uav_outflow_cache[u] = 0.0f;
  }
  for (int s = threadIdx.x; s < sat; s += blockDim.x) {
    sh_sat_incoming_cache[s] = 0.0f;
    sh_sat_processed_cache[s] = 0.0f;
    if (has_f(a, kFMainBwLinkSatLoads)) a.f[kFMainBwLinkSatLoads][e * sat + s] = 0.0f;
    if (has_f(a, kFStateLastSatConnectionCounts)) a.f[kFStateLastSatConnectionCounts][e * sat + s] = 0.0f;
    if (has_i(a, kIStateLastSatConnectionCounts)) a.i[kIStateLastSatConnectionCounts][e * sat + s] = 0;
  }
  for (int idx = threadIdx.x; idx < ucount * sat; idx += blockDim.x) {
    if (has_f(a, kFStateLastSelectedMaskByUavSat)) a.f[kFStateLastSelectedMaskByUavSat][e * ucount * sat + idx] = 0.0f;
    if (has_f(a, kFMainBwLinkRateMatrix)) a.f[kFMainBwLinkRateMatrix][e * ucount * sat + idx] = 0.0f;
    if (has_f(a, kFStateLastUavToSatOutflowMatrix)) a.f[kFStateLastUavToSatOutflowMatrix][e * ucount * sat + idx] = 0.0f;
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < ucount * select_k; idx += blockDim.x) {
    const int u = idx / max(select_k, 1);
    const int k = idx - u * max(select_k, 1);
    const int64_t sid64 = a.l[stage_l(bw_stage, kSlSatSelectionMatrix)][(e * ucount + u) * select_k + k];
    if (sid64 < 0 || sid64 >= sat) continue;
    const int sid = static_cast<int>(sid64);
    bool duplicate_for_u = false;
    for (int kk = 0; kk < k; ++kk) {
      const int64_t prev = a.l[stage_l(bw_stage, kSlSatSelectionMatrix)][(e * ucount + u) * select_k + kk];
      duplicate_for_u = duplicate_for_u || (prev == sid64);
    }
    if (!duplicate_for_u) {
      if (has_f(a, kFStateLastSelectedMaskByUavSat)) a.f[kFStateLastSelectedMaskByUavSat][(e * ucount + u) * sat + sid] = 1.0f;
      if (has_f(a, kFMainBwLinkSatLoads)) atomicAdd(&a.f[kFMainBwLinkSatLoads][e * sat + sid], 1.0f);
      if (has_f(a, kFStateLastSatConnectionCounts)) atomicAdd(&a.f[kFStateLastSatConnectionCounts][e * sat + sid], 1.0f);
      if (has_i(a, kIStateLastSatConnectionCounts)) atomicAdd(&a.i[kIStateLastSatConnectionCounts][e * sat + sid], 1);
      const float rate = backhaul_rate_for_selected_us(a, bw_stage, e, u, sid);
      if (has_f(a, kFMainBwLinkRateMatrix)) a.f[kFMainBwLinkRateMatrix][(e * ucount + u) * sat + sid] = rate;
      atomicAdd(&sh_uav_total_rate_cache[u], rate);
    }
  }
  __syncthreads();
  finish_profile_mark(
      finish_profile_out,
      finish_profile_stride,
      e,
      kFinishProfBackhaulSetup,
      &sh_finish_profile_clock);

  float local_service_sum = 0.0f;
  float local_drop_sum = 0.0f;
  float local_backlog = 0.0f;
  float local_expire_sum = 0.0f;
  for (int g = threadIdx.x; g < gu; g += blockDim.x) {
    const float demand = a.f[stage_f(bw_stage, kSfGuQueue)][e * gu + g] + a.f[kFStateLastGuArrival][e * gu + g];
    const float service_bits = quantize_device(fmaxf(a.f[kFStateLastGuOutflow][e * gu + g], 0.0f) * tau, fp(a, kFpFlowBitsQuantum, 32.0f));
    const float served = fminf(service_bits, demand);
    a.f[kFStateLastGuOutflow][e * gu + g] = served;
    float next_q = fmaxf(demand - served, 0.0f);
    float dropped = fmaxf(next_q - qmax_gu, 0.0f);
    next_q = fminf(next_q, qmax_gu);
    const bool served_any = served > 0.0f;
    float expired = 0.0f;
    a.f[kFStateServiceGap][e * gu + g] = clampf_device(
        a.f[kFStateServiceGap][e * gu + g] + fp(a, kFpServiceGapIncrement, 1.0f) -
            fp(a, kFpServiceGapReliefCoef, 0.5f) * served_any,
        0.0f,
        fp(a, kFpBwServiceGapCap, 8.0f));
    a.f[kFStateServiceGapRisk][e * gu + g] = safe_div(a.f[kFStateServiceGap][e * gu + g], fp(a, kFpBwServiceGapCap, 8.0f));
    if (ip(a, kParamDeadlineEnabled)) {
      float age = a.f[kFStateDeadlineAge][e * gu + g] + fp(a, kFpDeadlineAgeIncrement, 1.0f);
      if (served_any) age = fmaxf(age - fp(a, kFpDeadlineServiceReliefCoef, 0.75f), 0.0f);
      const float cap = fmaxf(fp(a, kFpDeadlineAgeCap, 1.0f), 1.0f);
      age = clampf_device(age, 0.0f, cap);
      const float deadline_steps = has_f(a, kFStateGuDeadlineSteps) ? fmaxf(a.f[kFStateGuDeadlineSteps][e * gu + g], 1.0f) : cap;
      const float risk = clampf_device(age / deadline_steps, 0.0f, 1.0f);
      expired = risk >= 1.0f ? fminf(next_q, fp(a, kFpDeadlineExpireRate, 0.0f) * next_q) : 0.0f;
      next_q = fmaxf(next_q - expired, 0.0f);
      dropped += expired;
      a.f[kFStateDeadlineAge][e * gu + g] = age;
      a.f[kFStateDeadlineRisk][e * gu + g] = risk;
      a.f[kFStateDeadlineSlack][e * gu + g] = clampf_device((deadline_steps - age) / deadline_steps, 0.0f, 1.0f);
    } else {
      a.f[kFStateDeadlineRisk][e * gu + g] = 0.0f;
      a.f[kFStateDeadlineSlack][e * gu + g] = 1.0f;
    }
    a.f[stage_f(bw_stage, kSfGuQueue)][e * gu + g] = next_q;
    if (has_f(a, kFStateGuDrop)) a.f[kFStateGuDrop][e * gu + g] = dropped;
    const float ema_keep = clampf_device(fp(a, kFpBwWorkloadEmaDecay, 0.95f), 0.0f, 1.0f);
    a.f[kFStateGuEma][e * gu + g] =
        ema_keep * a.f[kFStateGuEma][e * gu + g] + (1.0f - ema_keep) * service_bits;
    a.f[kFStateUrgencyRisk][e * gu + g] = ip(a, kParamDeadlineEnabled)
        ? a.f[kFStateDeadlineRisk][e * gu + g]
        : safe_div(next_q, qmax_gu);
    const int64_t assoc = a.l[stage_l(bw_stage, kSlAssoc)][e * gu + g];
    a.f[kFStateDownstreamPressure][e * gu + g] =
        (assoc >= 0 && assoc < ucount) ? safe_div(a.f[stage_f(bw_stage, kSfUavQueue)][e * ucount + static_cast<int>(assoc)], qmax_uav) : 0.0f;
    local_service_sum += served;
    local_drop_sum += dropped;
    local_backlog += next_q;
    local_expire_sum += expired;
  }
  const float total_service_sum = block_reduce_sum_128(local_service_sum, sh_reduce);
  if (threadIdx.x == 0) sh_service_sum = total_service_sum;
  __syncthreads();
  const float total_gu_drop_sum = block_reduce_sum_128(local_drop_sum, sh_reduce);
  if (threadIdx.x == 0) {
    sh_drop_sum = total_gu_drop_sum;
    sh_gu_drop_sum = total_gu_drop_sum;
  }
  __syncthreads();
  const float total_gu_backlog = block_reduce_sum_128(local_backlog, sh_reduce);
  if (threadIdx.x == 0) {
    sh_backlog = total_gu_backlog;
    sh_q_gu = total_gu_backlog;
  }
  __syncthreads();
  const float total_expire_sum = block_reduce_sum_128(local_expire_sum, sh_reduce);
  if (threadIdx.x == 0) sh_expire_sum = total_expire_sum;
  __syncthreads();
  finish_profile_mark(
      finish_profile_out,
      finish_profile_stride,
      e,
      kFinishProfGuQueue,
      &sh_finish_profile_clock);

  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    float inflow = 0.0f;
    for (int g = 0; g < gu; ++g) {
      const int64_t assoc = a.l[stage_l(bw_stage, kSlAssoc)][e * gu + g];
      if (assoc == u) inflow += a.f[kFStateLastGuOutflow][e * gu + g];
    }
    sh_uav_inflow_cache[u] = inflow;
    sh_uav_outflow_cache[u] = 0.0f;
  }
  __syncthreads();

  float local_backhaul_sum = 0.0f;
  float local_uav_drop_sum = 0.0f;
  float local_uav_backlog = 0.0f;
  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    const float inflow = sh_uav_inflow_cache[u];
    const float total_rate = sh_uav_total_rate_cache[u];
    const float q_before = a.f[stage_f(bw_stage, kSfUavQueue)][e * ucount + u] + inflow;
    const float service_bits = quantize_device(total_rate * tau, fp(a, kFpFlowBitsQuantum, 32.0f));
    const float uav_outflow = fminf(q_before, service_bits);
    float q = fmaxf(q_before - uav_outflow, 0.0f);
    const float dropped = fmaxf(q - qmax_uav, 0.0f);
    q = quantize_device(fminf(q, qmax_uav), fp(a, kFpQueueStateQuantum, 128.0f));
    a.f[stage_f(bw_stage, kSfUavQueue)][e * ucount + u] = q;
    if (has_f(a, kFStateLastGuToUavInflowByUav)) a.f[kFStateLastGuToUavInflowByUav][e * ucount + u] = inflow;
    if (has_f(a, kFStateUavDrop)) a.f[kFStateUavDrop][e * ucount + u] = dropped;
    const float ema_keep = clampf_device(fp(a, kFpBwWorkloadEmaDecay, 0.95f), 0.0f, 1.0f);
    a.f[kFStateUavEma][e * ucount + u] =
        ema_keep * a.f[kFStateUavEma][e * ucount + u] + (1.0f - ema_keep) * service_bits;
    sh_uav_outflow_cache[u] = uav_outflow;
    local_backhaul_sum += uav_outflow;
    local_uav_drop_sum += dropped;
    local_uav_backlog += q;
  }
  __syncthreads();
  const float total_uav_outflow_sum = block_reduce_sum_128(local_backhaul_sum, sh_reduce);
  (void)total_uav_outflow_sum;
  __syncthreads();
  const float total_uav_drop_sum = block_reduce_sum_128(local_uav_drop_sum, sh_reduce);
  if (threadIdx.x == 0) {
    sh_drop_sum += total_uav_drop_sum;
    sh_uav_drop_sum = total_uav_drop_sum;
  }
  __syncthreads();
  const float total_uav_backlog = block_reduce_sum_128(local_uav_backlog, sh_reduce);
  if (threadIdx.x == 0) {
    sh_backlog += total_uav_backlog;
    sh_q_uav = total_uav_backlog;
  }
  __syncthreads();
  finish_profile_mark(
      finish_profile_out,
      finish_profile_stride,
      e,
      kFinishProfUavQueue,
      &sh_finish_profile_clock);
  for (int idx = threadIdx.x; idx < ucount * select_k; idx += blockDim.x) {
    const int u = idx / max(select_k, 1);
    const int k = idx - u * max(select_k, 1);
    const int64_t sid64 = a.l[stage_l(bw_stage, kSlSatSelectionMatrix)][(e * ucount + u) * select_k + k];
    if (sid64 < 0 || sid64 >= sat) continue;
    const int s = static_cast<int>(sid64);
    bool duplicate_for_u = false;
    for (int kk = 0; kk < k; ++kk) {
      const int64_t prev = a.l[stage_l(bw_stage, kSlSatSelectionMatrix)][(e * ucount + u) * select_k + kk];
      duplicate_for_u = duplicate_for_u || (prev == sid64);
    }
    if (duplicate_for_u) continue;
    const float total_rate = sh_uav_total_rate_cache[u];
    const float rate_us = has_f(a, kFMainBwLinkRateMatrix)
        ? a.f[kFMainBwLinkRateMatrix][(e * ucount + u) * sat + s]
        : backhaul_rate_for_selected_us(a, bw_stage, e, u, s);
    const float outflow = (total_rate > 0.0f && rate_us > 0.0f)
        ? rate_us / total_rate * sh_uav_outflow_cache[u]
        : 0.0f;
    if (has_f(a, kFStateLastUavToSatOutflowMatrix)) {
      a.f[kFStateLastUavToSatOutflowMatrix][(e * ucount + u) * sat + s] = outflow;
    }
    atomicAdd(&sh_sat_incoming_cache[s], outflow);
  }
  __syncthreads();

  float local_sat_incoming_sum = 0.0f;
  float local_sat_processed = 0.0f;
  float local_sat_drop_sum = 0.0f;
  float local_sat_backlog = 0.0f;
  for (int s = threadIdx.x; s < sat; s += blockDim.x) {
    const float incoming_raw = sh_sat_incoming_cache[s];
    const float incoming = quantize_device(incoming_raw, fp(a, kFpFlowBitsQuantum, 32.0f));
    float q = a.f[stage_f(bw_stage, kSfSatQueue)][e * sat + s] + incoming;
    const float sat_service = quantize_device(sat_compute_rate_for(a, e, s) * tau, fp(a, kFpFlowBitsQuantum, 32.0f));
    const float processed = fminf(q, sat_service);
    q = fmaxf(q - processed, 0.0f);
    const float dropped = fmaxf(q - qmax_sat, 0.0f);
    q = quantize_device(fminf(q, qmax_sat), fp(a, kFpQueueStateQuantum, 128.0f));
    a.f[stage_f(bw_stage, kSfSatQueue)][e * sat + s] = q;
    if (has_f(a, kFStateSatDrop)) a.f[kFStateSatDrop][e * sat + s] = dropped;
    if (has_f(a, kFStateLastSatProcessed)) a.f[kFStateLastSatProcessed][e * sat + s] = processed;
    const float ema_keep = clampf_device(fp(a, kFpBwWorkloadEmaDecay, 0.95f), 0.0f, 1.0f);
    a.f[kFStateSatEma][e * sat + s] =
        ema_keep * a.f[kFStateSatEma][e * sat + s] + (1.0f - ema_keep) * sat_service;
    sh_sat_incoming_cache[s] = incoming;
    sh_sat_processed_cache[s] = processed;
    local_sat_incoming_sum += incoming;
    local_sat_processed += processed;
    local_sat_drop_sum += dropped;
    local_sat_backlog += q;
  }
  const float total_sat_incoming_sum = block_reduce_sum_128(local_sat_incoming_sum, sh_reduce);
  if (threadIdx.x == 0) sh_backhaul_sum = total_sat_incoming_sum;
  __syncthreads();
  const float total_sat_processed = block_reduce_sum_128(local_sat_processed, sh_reduce);
  if (threadIdx.x == 0) sh_sat_processed = total_sat_processed;
  __syncthreads();
  const float total_sat_drop_sum = block_reduce_sum_128(local_sat_drop_sum, sh_reduce);
  if (threadIdx.x == 0) {
    sh_drop_sum += total_sat_drop_sum;
    sh_sat_drop_sum = total_sat_drop_sum;
  }
  __syncthreads();
  const float total_sat_backlog = block_reduce_sum_128(local_sat_backlog, sh_reduce);
  if (threadIdx.x == 0) {
    sh_backlog += total_sat_backlog;
    sh_q_sat = total_sat_backlog;
  }
  __syncthreads();
  finish_profile_mark(
      finish_profile_out,
      finish_profile_stride,
      e,
      kFinishProfSatQueue,
      &sh_finish_profile_clock);

  if (finish_profile_out != nullptr && threadIdx.x == 0) {
    sh_refresh_profile_clock = clock64();
  }
  if (finish_profile_out != nullptr) {
    __syncthreads();
  }
  refresh_stage_queue_derived_parallel(
      a,
      bw_stage,
      e,
      finish_profile_out,
      finish_profile_stride,
      &sh_refresh_profile_clock,
      sh_refresh_cost_cache,
      false);
  __syncthreads();
  finish_profile_mark(
      finish_profile_out,
      finish_profile_stride,
      e,
      kFinishProfRefreshQueueDerived,
      &sh_finish_profile_clock);
  float local_accel_norm = 0.0f;
  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    const float ax = accel_source_mode == kSourceZero ? 0.0f : a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 0];
    const float ay = accel_source_mode == kSourceZero ? 0.0f : a.f[kFLiveAccelAction][(e * ucount + u) * 2 + 1];
    local_accel_norm += sqrtf(ax * ax + ay * ay) * fp(a, kFpBwInvAMax, 1.0f);
  }
  float accel_norm_sum = block_reduce_sum_128(local_accel_norm, sh_reduce);
  __syncthreads();

  float local_close_risk_sum = 0.0f;
  float local_pair_count = 0.0f;
  float local_term_close_risk = 0.0f;
  for (int pair = threadIdx.x; pair < ucount * ucount; pair += blockDim.x) {
    const int u = pair / ucount;
    const int v = pair - u * ucount;
    if (v <= u) continue;
    bool valid_pair = false;
    bool collision = false;
    const float pair_risk = close_risk_pair_value(a, bw_stage, e, u, v, &valid_pair, &collision);
    local_pair_count += valid_pair ? 1.0f : 0.0f;
    local_close_risk_sum += pair_risk;
    local_term_close_risk = fmaxf(local_term_close_risk, collision ? 1.0f : 0.0f);
  }
  const float close_risk_sum = block_reduce_sum_128(local_close_risk_sum, sh_reduce);
  __syncthreads();
  const float close_pair_count = block_reduce_sum_128(local_pair_count, sh_reduce);
  __syncthreads();
  const float term_close_risk = block_reduce_max_128(local_term_close_risk, sh_reduce);
  __syncthreads();
  const float close_risk = ip(a, kParamCloseRiskEnabled)
      ? safe_div(close_risk_sum, fmaxf(close_pair_count, 1.0f))
      : 0.0f;

  float local_energy_done = 0.0f;
  if (ip(a, kParamEnergyEnabled)) {
    for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
      local_energy_done = fmaxf(local_energy_done, a.f[stage_f(bw_stage, kSfUavEnergy)][e * ucount + u] <= 0.0f ? 1.0f : 0.0f);
    }
  }
  const float energy_done = block_reduce_max_128(local_energy_done, sh_reduce);
  __syncthreads();

  float local_overflow_risk_sum = 0.0f;
  float local_downstream_pressure_sum = 0.0f;
  float local_service_gap_sum = 0.0f;
  float local_service_gap_risk_sum = 0.0f;
  for (int g = threadIdx.x; g < gu; g += blockDim.x) {
    const float qn = safe_div(a.f[stage_f(bw_stage, kSfGuQueue)][e * gu + g], qmax_gu);
    local_overflow_risk_sum += has_f(a, kFStateUrgencyRisk) ? a.f[kFStateUrgencyRisk][e * gu + g] : qn;
    local_downstream_pressure_sum += has_f(a, kFStateDownstreamPressure) ? a.f[kFStateDownstreamPressure][e * gu + g] : 0.0f;
    local_service_gap_sum += has_f(a, kFStateServiceGap) ? a.f[kFStateServiceGap][e * gu + g] : 0.0f;
    local_service_gap_risk_sum += (has_f(a, kFStateServiceGapRisk) ? a.f[kFStateServiceGapRisk][e * gu + g] : 0.0f) * qn;
  }
  const float overflow_risk_sum = block_reduce_sum_128(local_overflow_risk_sum, sh_reduce);
  __syncthreads();
  const float downstream_pressure_sum = block_reduce_sum_128(local_downstream_pressure_sum, sh_reduce);
  __syncthreads();
  const float service_gap_sum = block_reduce_sum_128(local_service_gap_sum, sh_reduce);
  __syncthreads();
  const float service_gap_risk_sum = block_reduce_sum_128(local_service_gap_risk_sum, sh_reduce);
  __syncthreads();
  const float inv_gu = safe_div(1.0f, fmaxf(static_cast<float>(gu), 1.0f));
  const float overflow_risk_mean = overflow_risk_sum * inv_gu;
  const float downstream_pressure_mean = downstream_pressure_sum * inv_gu;
  const float service_gap_mean = service_gap_sum * inv_gu;
  const float service_gap_risk_mean = service_gap_risk_sum * inv_gu;

  float local_workload_before = 0.0f;
  float local_workload_after = 0.0f;
  float local_workload_drop = 0.0f;
  const int workload_count = gu + ucount + sat;
  for (int idx = threadIdx.x; idx < workload_count; idx += blockDim.x) {
    if (idx < gu) {
      const int g = idx;
      const float cost = refresh_cached_gu_cost(a, sh_refresh_cost_cache, g);
      const float before = a.f[kFStatePrevGuQueueVec][e * gu + g] + a.f[kFStateLastGuArrival][e * gu + g];
      const float after = a.f[stage_f(bw_stage, kSfGuQueue)][e * gu + g];
      const float served = a.f[kFStateLastGuOutflow][e * gu + g];
      const float cap_drop = fmaxf(before - served - qmax_gu, 0.0f);
      local_workload_before += cost * before;
      local_workload_after += cost * after;
      local_workload_drop += cost * cap_drop;
    } else if (idx < gu + ucount) {
      const int u = idx - gu;
      const float cost = refresh_cached_uav_cost(a, sh_refresh_cost_cache, u);
      const float before = a.f[kFStatePrevUavQueueVec][e * ucount + u];
      const float after = a.f[stage_f(bw_stage, kSfUavQueue)][e * ucount + u];
      const float inflow = sh_uav_inflow_cache[u];
      const float outflow = sh_uav_outflow_cache[u];
      const float cap_drop = fmaxf(before + inflow - outflow - qmax_uav, 0.0f);
      local_workload_before += cost * before;
      local_workload_after += cost * after;
      local_workload_drop += cost * cap_drop;
    } else {
      const int s = idx - gu - ucount;
      const float cost = refresh_cached_sat_cost(a, sh_refresh_cost_cache, s);
      const float incoming = sh_sat_incoming_cache[s];
      const float before = a.f[kFStatePrevSatQueueVec][e * sat + s];
      const float q_before = before + incoming;
      const float processed = sh_sat_processed_cache[s];
      const float after = a.f[stage_f(bw_stage, kSfSatQueue)][e * sat + s];
      const float cap_drop = fmaxf(q_before - processed - qmax_sat, 0.0f);
      local_workload_before += cost * before;
      local_workload_after += cost * after;
      local_workload_drop += cost * cap_drop;
    }
  }
  const float workload_before = block_reduce_sum_128(local_workload_before, sh_reduce);
  __syncthreads();
  const float workload_after = block_reduce_sum_128(local_workload_after, sh_reduce);
  __syncthreads();
  const float workload_drop = block_reduce_sum_128(local_workload_drop, sh_reduce);
  __syncthreads();

  float local_sat_overlap = 0.0f;
  if (ucount > 1 && sat > 0 && select_k > 0) {
    const float denom = fmaxf(static_cast<float>(ucount - 1), 1.0f);
    for (int idx = threadIdx.x; idx < ucount * select_k; idx += blockDim.x) {
      const int u = idx / select_k;
      const int k = idx - u * select_k;
      const int64_t sid64 = a.l[stage_l(bw_stage, kSlSatSelectionMatrix)][(e * ucount + u) * select_k + k];
      if (sid64 < 0 || sid64 >= sat) continue;
      int selected_count_u = 0;
      for (int kk = 0; kk < select_k; ++kk) {
        const int64_t cur = a.l[stage_l(bw_stage, kSlSatSelectionMatrix)][(e * ucount + u) * select_k + kk];
        if (cur >= 0 && cur < sat) ++selected_count_u;
      }
      int shared_count = 0;
      for (int uu = 0; uu < ucount; ++uu) {
        for (int kk = 0; kk < select_k; ++kk) {
          const int64_t cur = a.l[stage_l(bw_stage, kSlSatSelectionMatrix)][(e * ucount + uu) * select_k + kk];
          if (cur == sid64) ++shared_count;
        }
      }
      if (selected_count_u > 0) {
        local_sat_overlap += fmaxf(static_cast<float>(shared_count) - 1.0f, 0.0f) /
            (static_cast<float>(selected_count_u) * denom);
      }
    }
  }
  const float sat_overlap_sum = block_reduce_sum_128(local_sat_overlap, sh_reduce);
  if (threadIdx.x == 0) sh_sat_overlap_eval = reward_metric_quantize(a, sat_overlap_sum / fmaxf(static_cast<float>(ucount), 1.0f));
  __syncthreads();

  if (threadIdx.x == 0) {
    const float accel_norm = safe_div(accel_norm_sum, fmaxf(static_cast<float>(ucount), 1.0f));
    const float accel_norm2 = accel_norm * accel_norm;
    const float arrival_ref = require_positive_reward_ref(has_f(a, kFStateArrivalRef) ? a.f[kFStateArrivalRef][e] : sh_arrival_sum);
    const float service_ratio = clampf_device(ratio_or_zero(sh_service_sum, sh_arrival_sum), 0.0f, 1.0f);
    const float drop_ratio = clampf_device(ratio_or_zero(sh_drop_sum, sh_arrival_sum), 0.0f, 1.0f);
    const float prev_gu_sum = has_f(a, kFStatePrevQueueSumGu) ? a.f[kFStatePrevQueueSumGu][e] : 0.0f;
    const float prev_uav_sum = has_f(a, kFStatePrevQueueSumUav) ? a.f[kFStatePrevQueueSumUav][e] : 0.0f;
    const float prev_sat_sum = has_f(a, kFStatePrevQueueSumSat) ? a.f[kFStatePrevQueueSumSat][e] : 0.0f;
    const float q_total = sh_q_gu + sh_q_uav + sh_q_sat;
    const float q_active = sh_q_gu + sh_q_uav;
    const float b_pre_steps = prev_gu_sum + prev_uav_sum;
    const float x_acc_raw = sh_service_sum / arrival_ref;
    const float x_rel_raw = sh_backhaul_sum / arrival_ref;
    const float g_pre_raw = (q_active - b_pre_steps) / arrival_ref;
    const float d_pre_raw = (sh_gu_drop_sum + sh_uav_drop_sum) / arrival_ref;
    const float processed_ratio_raw = sh_sat_processed / arrival_ref;
    const float drop_ratio_eval_raw = sh_drop_sum / arrival_ref;
    const float pre_backlog_steps_raw = q_active / arrival_ref;
    const float d_sys_report_raw = ratio_or_zero(q_total, sh_sat_processed);

    const float weighted_delta = -(workload_after - workload_before) - workload_drop;
    const float weighted_level = -workload_after - workload_drop;
    const float relative_weighted_delta = weighted_delta / fmaxf(workload_before, 1.0f);
    const float positive_weighted_level = 1.0f / (1.0f + log1pf(fmaxf(workload_after + workload_drop, 0.0f)));
    const float gu_queue_level = -(sh_q_gu + sh_gu_drop_sum) / arrival_ref;
    const float system_queue_level = -(q_total + sh_drop_sum) / arrival_ref;
    const float gu_service_queue = (sh_service_sum - sh_q_gu - sh_gu_drop_sum) / arrival_ref;

    const float qmax_total = positive_config_scale(qmax_gu * fmaxf(static_cast<float>(gu), 1.0f) +
                                       qmax_uav * fmaxf(static_cast<float>(ucount), 1.0f) +
                                       qmax_sat * fmaxf(static_cast<float>(sat), 1.0f));
    const float queue_penalty = clampf_device(q_total / qmax_total, 0.0f, 1.0f);
    float reward = 0.0f;
    if (ip(a, kParamRewardMode) == 1) {
      reward = fp(a, kFpRewardWAccess, 1.0f) * x_acc_raw +
          fp(a, kFpRewardWRelay, 0.0f) * x_rel_raw -
          fp(a, kFpRewardWPreDrop, 0.0f) * d_pre_raw -
          fp(a, kFpRewardWPreBacklog, 0.0f) * log1pf(fmaxf(pre_backlog_steps_raw, 0.0f)) -
          fp(a, kFpRewardWPreOverflowRisk, 0.0f) * overflow_risk_mean -
          fp(a, kFpRewardWPreServiceGap, 0.0f) * service_gap_risk_mean;
    } else if (ip(a, kParamRewardMode) == 10) {
      reward = 0.5f * x_rel_raw + 0.5f * processed_ratio_raw - drop_ratio_eval_raw -
          0.05f * sh_sat_overlap_eval;
    } else if (ip(a, kParamRewardMode) == 11) {
      reward = x_rel_raw - d_pre_raw - 0.05f * sh_sat_overlap_eval;
    } else if (ip(a, kParamRewardMode) == 2) {
      reward = fp(a, kFpThroughputOnlyAccessCoef, 1.0f) * x_acc_raw +
          fp(a, kFpThroughputOnlyBackhaulCoef, 1.0f) * x_rel_raw -
          fmaxf(fp(a, kFpThroughputOnlyGuQueueCoef, 0.0f), 0.0f) * (sh_q_gu / arrival_ref);
    } else if (ip(a, kParamRewardMode) == 3) {
      reward = weighted_delta;
    } else if (ip(a, kParamRewardMode) == 4) {
      reward = weighted_level;
    } else if (ip(a, kParamRewardMode) == 5) {
      reward = positive_weighted_level;
    } else if (ip(a, kParamRewardMode) == 6) {
      reward = gu_queue_level;
    } else if (ip(a, kParamRewardMode) == 7) {
      reward = system_queue_level;
    } else if (ip(a, kParamRewardMode) == 8) {
      reward = gu_service_queue;
    } else if (ip(a, kParamRewardMode) == 9) {
      reward = relative_weighted_delta;
    } else {
      const float queue_delta = clampf_device((prev_gu_sum + prev_uav_sum + prev_sat_sum - q_total) / qmax_total, -1.0f, 1.0f);
      float raw_reward = fp(a, kFpEtaService, 1.0f) * x_acc_raw +
          fp(a, kFpEtaThroughputAccess, 0.0f) * x_acc_raw +
          fp(a, kFpEtaThroughputBackhaul, 0.0f) * x_rel_raw -
          fp(a, kFpEtaDropGu, fp(a, kFpEtaDrop, 0.0f)) * (sh_gu_drop_sum / arrival_ref) -
          fp(a, kFpEtaDropUav, fp(a, kFpEtaDrop, 0.0f)) * (sh_uav_drop_sum / arrival_ref) -
          fp(a, kFpEtaDropSat, fp(a, kFpEtaDrop, 0.0f)) * (sh_sat_drop_sum / arrival_ref) -
          fp(a, kFpEtaDropStep, 0.0f) * (sh_drop_sum > kRuntimeRatioZeroTol ? 1.0f : 0.0f) -
          fp(a, kFpOmegaQ, 0.0f) * queue_penalty +
          fp(a, kFpEtaQDelta, 0.0f) * queue_delta -
          fp(a, kFpEtaAccel, 0.0f) * accel_norm2 -
          fmaxf(fp(a, kFpEtaCloseRisk, 0.0f), 0.0f) * close_risk;
      if (ip(a, kParamUseRewardTanh)) raw_reward = tanhf(raw_reward);
      reward = raw_reward - fp(a, kFpEtaCrash, 0.0f) * term_close_risk -
          (energy_done > 0.5f ? fp(a, kFpEtaBatt, 0.0f) : 0.0f);
    }
    reward = reward_metric_quantize(a, reward);
    const int t_current = has_i(a, kIStateT) ? a.i[kIStateT][e] : slot;
    const bool truncated = static_cast<float>(t_current) >= fmaxf(fp(a, kFpRewardTSteps, 1.0f) - 1.0f, 0.0f);
    const bool terminated = energy_done > 0.5f || term_close_risk > 0.5f;
    sh_terminated = terminated;
    sh_truncated = truncated;
    sh_done = terminated || truncated;
    sh_close_risk = close_risk;
    sh_term_close_risk = term_close_risk;
    sh_accel_norm = accel_norm;
    sh_service_ratio = service_ratio;
    sh_drop_ratio = drop_ratio;
    sh_queue_penalty = queue_penalty;
    sh_arrival_ref = arrival_ref;
    sh_b_pre_steps = b_pre_steps;
    sh_x_acc = reward_metric_quantize(a, x_acc_raw);
    sh_x_rel = reward_metric_quantize(a, x_rel_raw);
    sh_g_pre = reward_metric_quantize(a, g_pre_raw);
    sh_d_pre = reward_metric_quantize(a, d_pre_raw);
    sh_processed_ratio_eval = reward_metric_quantize(a, processed_ratio_raw);
    sh_drop_ratio_eval = reward_metric_quantize(a, drop_ratio_eval_raw);
    sh_pre_backlog_steps_eval = reward_metric_quantize(a, pre_backlog_steps_raw);
    sh_d_sys_report = reward_metric_quantize(a, d_sys_report_raw);
    sh_overflow_risk_mean = overflow_risk_mean;
    sh_downstream_pressure_mean = downstream_pressure_mean;
    sh_service_gap_mean = service_gap_mean;
    sh_service_gap_risk_mean = service_gap_risk_mean;
    sh_weighted_delta = weighted_delta;
    sh_weighted_level = weighted_level;
    sh_gu_queue_level = gu_queue_level;
    sh_system_queue_level = system_queue_level;
    sh_gu_service_queue = gu_service_queue;
    sh_reward_raw = reward;
    sh_reward = reward;
    if (has_f(a, kFStatePrevQueueSumGu)) a.f[kFStatePrevQueueSumGu][e] = sh_q_gu;
    if (has_f(a, kFStatePrevQueueSumUav)) a.f[kFStatePrevQueueSumUav][e] = sh_q_uav;
    if (has_f(a, kFStatePrevQueueSumSat)) a.f[kFStatePrevQueueSumSat][e] = sh_q_sat;
    if (has_f(a, kFStatePrevQNormActive)) {
      const float arrival_floor = fp(a, kFpQueueNormArrivalFloor, 0.0f) > 0.0f
          ? fp(a, kFpQueueNormArrivalFloor, 0.0f)
          : fp(a, kFpBwTau0, fp(a, kFpTau0, 1.0f)) * fmaxf(static_cast<float>(gu), 1.0f) *
              (has_f(a, kFStateEffectiveArrivalRate) ? a.f[kFStateEffectiveArrivalRate][e] : 1.0f);
      const float queue_scale = positive_config_scale(fp(a, kFpQueueNormK, 1.0f)) *
          require_positive_reward_ref(fmaxf(sh_arrival_sum, arrival_floor));
      a.f[kFStatePrevQNormActive][e] = clampf_device(q_active / queue_scale, 0.0f, 1.0f);
    }
  }
  __syncthreads();
  finish_profile_mark(
      finish_profile_out,
      finish_profile_stride,
      e,
      kFinishProfRewardStats,
      &sh_finish_profile_clock);

  copy_live_to_history_local_parallel(a, slot, static_cast<int>(active_idx), e);
  commit_actions_parallel(a, slot, static_cast<int>(active_idx), e, static_cast<int>(accel_source_mode), static_cast<int>(sat_source_mode), static_cast<int>(bw_source_mode));
  if (threadIdx.x == 0) {
    a.f[kFHistBwRewards][row] = sh_reward;
    a.b[kBHistTerminated][row] = sh_terminated;
    a.b[kBHistTruncated][row] = sh_truncated;
    if (has_f(a, kFHistBwAccessRewards)) a.f[kFHistBwAccessRewards][row] = sh_x_acc;
    if (has_f(a, kFHistBwWeightedWorkloadDeltaRewards)) a.f[kFHistBwWeightedWorkloadDeltaRewards][row] = sh_weighted_delta;
    if (has_f(a, kFHistBwWeightedWorkloadLevelRewards)) a.f[kFHistBwWeightedWorkloadLevelRewards][row] = sh_weighted_level;
    if (has_f(a, kFHistBwGuQueueLevelRewards)) a.f[kFHistBwGuQueueLevelRewards][row] = sh_gu_queue_level;
    if (has_f(a, kFHistBwSystemQueueLevelRewards)) a.f[kFHistBwSystemQueueLevelRewards][row] = sh_system_queue_level;
    if (has_f(a, kFHistBwGuServiceQueueRewards)) a.f[kFHistBwGuServiceQueueRewards][row] = sh_gu_service_queue;
  }
  float local_danger_active = 0.0f;
  float local_intervention_norm_sum = 0.0f;
  float local_intervention_rate_sum = 0.0f;
  float local_intervention_norm_top1 = 0.0f;
  for (int u = threadIdx.x; u < ucount; u += blockDim.x) {
    const float exec_x = has_f(a, kFStateLastExecAccel) ? a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 0] : 0.0f;
    const float exec_y = has_f(a, kFStateLastExecAccel) ? a.f[kFStateLastExecAccel][(e * ucount + u) * 2 + 1] : 0.0f;
    const float policy_x = has_f(a, kFStateLastPolicyAccel) ? a.f[kFStateLastPolicyAccel][(e * ucount + u) * 2 + 0] : exec_x;
    const float policy_y = has_f(a, kFStateLastPolicyAccel) ? a.f[kFStateLastPolicyAccel][(e * ucount + u) * 2 + 1] : exec_y;
    const float dx = exec_x - policy_x;
    const float dy = exec_y - policy_y;
    const float intervention_norm = sqrtf(dx * dx + dy * dy);
    const float inv_a_for_norm = fp(a, kFpBwInvAMax, 0.0f);
    const float intervention_norm_uav = inv_a_for_norm > 0.0f
        ? intervention_norm * inv_a_for_norm
        : divide_or_zero(intervention_norm, positive_config_scale(fp(a, kFpAccelAMax, 1.0f)));
    local_intervention_norm_sum += intervention_norm_uav;
    local_intervention_rate_sum += intervention_norm > kDynamicsDenomEps ? 1.0f : 0.0f;
    local_intervention_norm_top1 = fmaxf(local_intervention_norm_top1, intervention_norm_uav);
    float close_risk_uav = 0.0f;
    if (ip(a, kParamDangerTriggerMode) == 0 || ip(a, kParamCloseRiskEnabled)) {
      for (int v = 0; v < ucount; ++v) {
        if (v == u) continue;
        const int lo = min(u, v);
        const int hi = max(u, v);
        close_risk_uav = fmaxf(close_risk_uav, close_risk_pair_value(a, bw_stage, e, lo, hi, nullptr, nullptr));
      }
    }
    bool danger_active = false;
    if (ip(a, kParamDangerImitationEnabled)) {
      const int trigger_mode = static_cast<int>(ip(a, kParamDangerTriggerMode));
      if (trigger_mode == 1) {
        danger_active = intervention_norm > kDynamicsDenomEps;
      } else if (trigger_mode == 2) {
        danger_active = intervention_norm_uav > fp(a, kFpDangerInterventionThresh, 0.0f);
      } else {
        danger_active = close_risk_uav > fp(a, kFpDangerCloseRiskThresh, 0.0f) || intervention_norm > kDynamicsDenomEps;
      }
    }
    local_danger_active += danger_active ? 1.0f : 0.0f;
    const int base = (row * ucount + u) * 2;
    if (has_f(a, kFHistDangerTargets)) {
      const float inv_a = fp(a, kFpBwInvAMax, 1.0f);
      const float amax_inv = inv_a > 0.0f ? inv_a : divide_or_zero(1.0f, positive_config_scale(fp(a, kFpAccelAMax, 1.0f)));
      a.f[kFHistDangerTargets][base + 0] = clampf_device(exec_x * amax_inv, -1.0f, 1.0f);
      a.f[kFHistDangerTargets][base + 1] = clampf_device(exec_y * amax_inv, -1.0f, 1.0f);
    }
    if (has_f(a, kFHistDangerMasks)) {
      a.f[kFHistDangerMasks][base + 0] = danger_active ? 1.0f : 0.0f;
      a.f[kFHistDangerMasks][base + 1] = danger_active ? 1.0f : 0.0f;
    }
  }
  const float danger_active_sum = block_reduce_sum_128(local_danger_active, sh_reduce);
  __syncthreads();
  const float intervention_norm_sum = block_reduce_sum_128(local_intervention_norm_sum, sh_reduce);
  __syncthreads();
  const float intervention_rate_sum = block_reduce_sum_128(local_intervention_rate_sum, sh_reduce);
  __syncthreads();
  const float intervention_top1 = block_reduce_max_128(local_intervention_norm_top1, sh_reduce);
  __syncthreads();
  if (threadIdx.x == 0) {
    sh_danger_active_rate = safe_div(danger_active_sum, fmaxf(static_cast<float>(ucount), 1.0f));
    sh_intervention_norm = safe_div(intervention_norm_sum, fmaxf(static_cast<float>(ucount), 1.0f));
    sh_intervention_rate = safe_div(intervention_rate_sum, fmaxf(static_cast<float>(ucount), 1.0f));
    sh_intervention_norm_top1 = intervention_top1;
  }
  __syncthreads();
  commit_reward_parts_parallel(
      a,
      row,
      sh_service_ratio,
      sh_drop_ratio,
      sh_arrival_ref,
      sh_b_pre_steps,
      sh_x_acc,
      sh_x_rel,
      sh_g_pre,
      sh_d_pre,
      sh_processed_ratio_eval,
      sh_drop_ratio_eval,
      sh_pre_backlog_steps_eval,
      sh_sat_overlap_eval,
      sh_d_sys_report,
      sh_drop_sum,
      sh_q_gu,
      sh_q_uav,
      sh_q_sat,
      sh_q_gu + sh_q_uav + sh_q_sat,
      sh_gu_drop_sum + sh_uav_drop_sum,
      sh_expire_sum,
      sh_gu_drop_sum,
      sh_uav_drop_sum,
      sh_sat_drop_sum,
      sh_arrival_sum,
      sh_service_sum,
      sh_backhaul_sum,
      sh_sat_processed,
      sh_term_close_risk,
      sh_overflow_risk_mean,
      sh_downstream_pressure_mean,
      sh_service_gap_mean,
      sh_service_gap_risk_mean,
      sh_weighted_delta,
      sh_weighted_level,
      sh_gu_queue_level,
      sh_system_queue_level,
      sh_gu_service_queue,
      sh_intervention_norm,
      sh_intervention_rate,
      sh_intervention_norm_top1,
      sh_danger_active_rate,
      sh_close_risk,
      sh_term_close_risk,
      sh_reward_raw);
  if (threadIdx.x == 0) a.b[kBHistTerminalNextWorldMask][row] = sh_done;
  __syncthreads();
  finish_profile_mark(
      finish_profile_out,
      finish_profile_stride,
      e,
      kFinishProfHistoryAndDanger,
      &sh_finish_profile_clock);
  if (sh_done) {
    write_world_from_stage_parallel(a, bw_stage, kFHistTerminalWorld, kBHistTerminalWorld, row, e);
  }
  __syncthreads();
  finish_profile_mark(
      finish_profile_out,
      finish_profile_stride,
      e,
      kFinishProfTerminalWorld,
      &sh_finish_profile_clock);
  apply_reset_or_state_commit_parallel(a, slot, e, sh_done);
  __syncthreads();
  finish_profile_mark(
      finish_profile_out,
      finish_profile_stride,
      e,
      kFinishProfCommitOrReset,
      &sh_finish_profile_clock);
  const int next_stage = 1 - static_cast<int>(active_idx);
  if (rollout_tail) {
    if (finish_profile_out != nullptr && threadIdx.x == 0) {
      sh_refresh_profile_clock = clock64();
    }
    if (finish_profile_out != nullptr) {
      __syncthreads();
    }
    prepare_stage_from_state_parallel(
        a,
        next_stage,
        e,
        0,
        finish_profile_out,
        finish_profile_stride,
        &sh_refresh_profile_clock,
        sh_refresh_cost_cache);
    __syncthreads();
    finish_profile_mark(
        finish_profile_out,
        finish_profile_stride,
        e,
        kFinishProfNextPrepareStage,
        &sh_finish_profile_clock);
    write_world_from_stage_parallel(a, next_stage, kFHistAccelWorld, kBHistAccelWorld, hist_env_row(a, slot + 1, e), e);
    __syncthreads();
    finish_profile_mark(
        finish_profile_out,
        finish_profile_stride,
        e,
        kFinishProfNextWorld,
        &sh_finish_profile_clock);
    copy_runtime_state_snapshot_to_history_parallel(
        a,
        kFHistAccelRuntimeStateBase,
        kLHistAccelRuntimeStateBase,
        kIHistAccelRuntimeStateBase,
        kBHistAccelRuntimeStateBase,
        static_cast<int>(slot + 1),
        e);
    copy_stage_snapshot_to_history_parallel(
        a,
        kFHistAccelRuntimeStageBase,
        kLHistAccelRuntimeStageBase,
        kBHistAccelRuntimeStageBase,
        static_cast<int>(slot + 1),
        e,
        next_stage);
    finish_profile_mark(
        finish_profile_out,
        finish_profile_stride,
        e,
        kFinishProfNextSnapshots,
        &sh_finish_profile_clock);
  } else {
    copy_random_step_tapes_parallel(a, slot + 1, e);
    __syncthreads();
    finish_profile_mark(
        finish_profile_out,
        finish_profile_stride,
        e,
        kFinishProfNextRandomTape,
        &sh_finish_profile_clock);
    if (finish_profile_out != nullptr && threadIdx.x == 0) {
      sh_refresh_profile_clock = clock64();
    }
    if (finish_profile_out != nullptr) {
      __syncthreads();
    }
    prepare_stage_from_state_parallel(
        a,
        next_stage,
        e,
        0,
        finish_profile_out,
        finish_profile_stride,
        &sh_refresh_profile_clock,
        sh_refresh_cost_cache);
    __syncthreads();
    finish_profile_mark(
        finish_profile_out,
        finish_profile_stride,
        e,
        kFinishProfNextPrepareStage,
        &sh_finish_profile_clock);
    write_world_from_stage_parallel(a, next_stage, kFHistAccelWorld, kBHistAccelWorld, hist_env_row(a, slot + 1, e), e);
    __syncthreads();
    finish_profile_mark(
        finish_profile_out,
        finish_profile_stride,
        e,
        kFinishProfNextWorld,
        &sh_finish_profile_clock);
    copy_runtime_state_snapshot_to_history_parallel(
        a,
        kFHistAccelRuntimeStateBase,
        kLHistAccelRuntimeStateBase,
        kIHistAccelRuntimeStateBase,
        kBHistAccelRuntimeStateBase,
        static_cast<int>(slot + 1),
        e);
    copy_stage_snapshot_to_history_parallel(
        a,
        kFHistAccelRuntimeStageBase,
        kLHistAccelRuntimeStageBase,
        kBHistAccelRuntimeStageBase,
        static_cast<int>(slot + 1),
        e,
        next_stage);
    __syncthreads();
    finish_profile_mark(
        finish_profile_out,
        finish_profile_stride,
        e,
        kFinishProfNextSnapshots,
        &sh_finish_profile_clock);
    if (finish_profile_out != nullptr && threadIdx.x == 0) {
      sh_accel_obs_profile_clock = clock64();
    }
    if (finish_profile_out != nullptr) {
      __syncthreads();
    }
    write_accel_obs_parallel(
        a,
        next_stage,
        next_stage,
        e,
        sh_accel_cell_summary_cache,
        sh_last_route_cost_cache,
        finish_profile_out,
        finish_profile_stride,
        &sh_accel_obs_profile_clock);
    finish_profile_mark(
        finish_profile_out,
        finish_profile_stride,
        e,
        kFinishProfNextAccelObs,
        &sh_finish_profile_clock);
  }
  if (threadIdx.x == 0) {
    if (has_i(a, kIStateGlobalStep)) a.i[kIStateGlobalStep][e] += 1;
    if (has_i(a, kIRandomStepTensor) && e == 0) a.i[kIRandomStepTensor][0] = slot + 1;
    if (has_i(a, kIMarker)) a.i[kIMarker][0] = 4;
  }
}

void launch_phase(
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
    int phase,
    float* finish_profile_out = nullptr,
    int finish_profile_stride = 0) {
  check_launch_contract(
      float_tensors,
      long_tensors,
      bool_tensors,
      int_tensors,
      int_params,
      active_idx,
      accel_source_mode,
      sat_source_mode,
      bw_source_mode);
  const c10::cuda::CUDAGuard device_guard(int_tensors.front().device());
  PackedAbi abi = pack_abi(float_tensors, long_tensors, bool_tensors, int_tensors, int_params, float_params);
  const int num_envs = static_cast<int>(int_params[kParamNumEnvs]);
  if (num_envs <= 0) {
    return;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  copy_live_abi_to_symbol(abi, stream);
  dim3 grid(static_cast<unsigned int>(num_envs));
  dim3 block(128);
  const int num_uav = static_cast<int>(int_params[kParamNumUav]);
  dim3 source_row_grid(static_cast<unsigned int>(num_envs), static_cast<unsigned int>(num_uav > 0 ? num_uav : 1));
  dim3 source_block(kSourceBlockMaxThreads);
  if (phase == 1) {
    prepare_initial_accel_live_kernel<<<grid, block, 0, stream>>>(slot, active_idx);
  } else if (phase == 2) {
    accel_to_sat_live_kernel<<<grid, block, 0, stream>>>(slot, active_idx, accel_source_mode);
  } else if (phase == 15) {
    if (accel_source_mode == kSourceObservableClusterQueueAware) {
      observable_cluster_accel_live_kernel<<<source_row_grid, source_block, 0, stream>>>(active_idx);
    } else {
      baseline_accel_live_kernel<<<source_row_grid, source_block, 0, stream>>>(active_idx, accel_source_mode);
    }
  } else if (phase == 11) {
    queue_aware_accel_live_kernel<<<source_row_grid, source_block, 0, stream>>>(active_idx);
  } else if (phase == 12) {
    cluster_center_accel_live_kernel<<<grid, source_block, 0, stream>>>(active_idx);
  } else if (phase == 16) {
    baseline_sat_live_kernel<<<source_row_grid, source_block, 0, stream>>>(sat_source_mode);
  } else if (phase == 13) {
    queue_aware_sat_live_kernel<<<source_row_grid, source_block, 0, stream>>>();
  } else if (phase == 17) {
    baseline_bw_live_kernel<<<source_row_grid, source_block, 0, stream>>>(bw_source_mode);
  } else if (phase == 14) {
    queue_aware_bw_live_kernel<<<source_row_grid, source_block, 0, stream>>>();
  } else if (phase == 3) {
    sat_to_bw_live_kernel<<<grid, block, 0, stream>>>(slot, sat_source_mode);
  } else if (phase == 5) {
    apply_bw_macro_live_kernel<<<grid, block, 0, stream>>>(slot, bw_source_mode);
  } else if (phase == 4) {
    const int num_sat = static_cast<int>(int_params[kParamNumSat]);
    const int num_gu = static_cast<int>(int_params[kParamNumGu]);
    const size_t finish_shared_bytes =
        static_cast<size_t>(
            4 * (num_uav > 0 ? num_uav : 0) +
            2 * (num_sat > 0 ? num_sat : 0) +
            (num_uav > 0 ? num_uav : 0) * kAccelCellDim +
            refresh_cost_cache_size_for_dims(
                (num_sat > 0 ? num_sat : 0),
                (num_uav > 0 ? num_uav : 0),
                (num_gu > 0 ? num_gu : 0)) +
            (num_uav > 0 ? num_uav : 0) +
            (num_gu > 0 ? num_gu : 0)) *
        sizeof(float);
    finish_commit_prepare_live_kernel<<<grid, block, finish_shared_bytes, stream>>>(
        slot,
        active_idx,
        rollout_tail,
        accel_source_mode,
        sat_source_mode,
        bw_source_mode,
        finish_profile_out,
        finish_profile_stride);
  } else {
    throw std::runtime_error("unknown native CUDA rollout phase.");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

namespace {

__global__ void stage_mc_gae_kernel(
    const float* __restrict__ values,
    const float* __restrict__ mc_returns,
    const bool* __restrict__ terminated,
    const bool* __restrict__ truncated,
    float* __restrict__ returns_out,
    float* __restrict__ advantages_out,
    int num_steps,
    int num_envs,
    float gamma,
    float gae_lambda) {
  const int e = static_cast<int>(blockIdx.x);
  if (e >= num_envs || threadIdx.x != 0) {
    return;
  }
  float next_adv = 0.0f;
  float next_value = 0.0f;
  float next_mc = 0.0f;
  bool have_next = false;
  for (int t = num_steps - 1; t >= 0; --t) {
    const int pos = t * num_envs + e;
    const bool ended_here = terminated[pos] || truncated[pos];
    float collapsed_reward;
    float bootstrap_value;
    float bootstrap_adv;
    if (have_next && !ended_here) {
      collapsed_reward = mc_returns[pos] - gamma * next_mc;
      bootstrap_value = next_value;
      bootstrap_adv = next_adv;
    } else {
      collapsed_reward = mc_returns[pos];
      bootstrap_value = 0.0f;
      bootstrap_adv = 0.0f;
    }
    const float value = values[pos];
    const float delta = collapsed_reward + gamma * bootstrap_value - value;
    const float adv = delta + gamma * gae_lambda * bootstrap_adv;
    advantages_out[pos] = adv;
    returns_out[pos] = adv + value;
    next_adv = adv;
    next_value = value;
    next_mc = mc_returns[pos];
    have_next = !ended_here;
  }
}

void check_stage_mc_gae_tensor(
    const at::Tensor& tensor,
    const char* name,
    c10::Device device,
    at::ScalarType dtype,
    int64_t expected_numel) {
  if (!tensor.defined()) {
    throw std::runtime_error(std::string("stage_mc_gae missing tensor ") + name);
  }
  if (!tensor.is_cuda()) {
    throw std::runtime_error(std::string("stage_mc_gae tensor is not CUDA: ") + name);
  }
  if (tensor.device() != device) {
    throw std::runtime_error(std::string("stage_mc_gae tensor device mismatch: ") + name);
  }
  if (tensor.scalar_type() != dtype) {
    throw std::runtime_error(std::string("stage_mc_gae tensor dtype mismatch: ") + name);
  }
  if (!tensor.is_contiguous()) {
    throw std::runtime_error(std::string("stage_mc_gae tensor must be contiguous: ") + name);
  }
  if (tensor.numel() != expected_numel) {
    throw std::runtime_error(std::string("stage_mc_gae tensor size mismatch: ") + name);
  }
}

}  // namespace

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
    double gae_lambda) {
  if (num_steps <= 0 || num_envs <= 0) {
    return;
  }
  if (!values.defined() || !values.is_cuda()) {
    throw std::runtime_error("stage_mc_gae requires CUDA values tensor.");
  }
  const int64_t expected = num_steps * num_envs;
  const c10::Device device = values.device();
  check_stage_mc_gae_tensor(values, "values", device, at::kFloat, expected);
  check_stage_mc_gae_tensor(mc_returns, "mc_returns", device, at::kFloat, expected);
  check_stage_mc_gae_tensor(terminated, "terminated", device, at::kBool, expected);
  check_stage_mc_gae_tensor(truncated, "truncated", device, at::kBool, expected);
  check_stage_mc_gae_tensor(returns_out, "returns_out", device, at::kFloat, expected);
  check_stage_mc_gae_tensor(advantages_out, "advantages_out", device, at::kFloat, expected);
  const c10::cuda::CUDAGuard guard(device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
  dim3 grid(static_cast<unsigned int>(num_envs));
  dim3 block(1);
  stage_mc_gae_kernel<<<grid, block, 0, stream>>>(
      values.data_ptr<float>(),
      mc_returns.data_ptr<float>(),
      terminated.data_ptr<bool>(),
      truncated.data_ptr<bool>(),
      returns_out.data_ptr<float>(),
      advantages_out.data_ptr<float>(),
      static_cast<int>(num_steps),
      static_cast<int>(num_envs),
      static_cast<float>(gamma),
      static_cast<float>(gae_lambda));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
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
    int64_t bw_source_mode) {
  launch_phase(
      float_tensors,
      long_tensors,
      bool_tensors,
      int_tensors,
      int_params,
      float_params,
      slot,
      active_idx,
      rollout_tail,
      accel_source_mode,
      sat_source_mode,
      bw_source_mode,
      1);
}

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
    int64_t bw_source_mode) {
  launch_phase(
      float_tensors,
      long_tensors,
      bool_tensors,
      int_tensors,
      int_params,
      float_params,
      slot,
      active_idx,
      rollout_tail,
      accel_source_mode,
      sat_source_mode,
      bw_source_mode,
      2);
}

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
    int64_t bw_source_mode) {
  launch_phase(
      float_tensors,
      long_tensors,
      bool_tensors,
      int_tensors,
      int_params,
      float_params,
      0,
      active_idx,
      false,
      accel_source_mode,
      sat_source_mode,
      bw_source_mode,
      11);
}

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
    int64_t bw_source_mode) {
  launch_phase(
      float_tensors,
      long_tensors,
      bool_tensors,
      int_tensors,
      int_params,
      float_params,
      0,
      active_idx,
      false,
      accel_source_mode,
      sat_source_mode,
      bw_source_mode,
      12);
}

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
    int64_t bw_source_mode) {
  launch_phase(
      float_tensors,
      long_tensors,
      bool_tensors,
      int_tensors,
      int_params,
      float_params,
      0,
      active_idx,
      false,
      accel_source_mode,
      sat_source_mode,
      bw_source_mode,
      15);
}

void queue_aware_sat_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode) {
  launch_phase(
      float_tensors,
      long_tensors,
      bool_tensors,
      int_tensors,
      int_params,
      float_params,
      0,
      0,
      false,
      accel_source_mode,
      sat_source_mode,
      bw_source_mode,
      13);
}

void baseline_sat_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode) {
  launch_phase(
      float_tensors,
      long_tensors,
      bool_tensors,
      int_tensors,
      int_params,
      float_params,
      0,
      0,
      false,
      accel_source_mode,
      sat_source_mode,
      bw_source_mode,
      16);
}

void queue_aware_bw_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode) {
  launch_phase(
      float_tensors,
      long_tensors,
      bool_tensors,
      int_tensors,
      int_params,
      float_params,
      0,
      0,
      false,
      accel_source_mode,
      sat_source_mode,
      bw_source_mode,
      14);
}

void baseline_bw_live_launcher(
    const TensorVec& float_tensors,
    const TensorVec& long_tensors,
    const TensorVec& bool_tensors,
    const TensorVec& int_tensors,
    const std::vector<int64_t>& int_params,
    const std::vector<double>& float_params,
    int64_t accel_source_mode,
    int64_t sat_source_mode,
    int64_t bw_source_mode) {
  launch_phase(
      float_tensors,
      long_tensors,
      bool_tensors,
      int_tensors,
      int_params,
      float_params,
      0,
      0,
      false,
      accel_source_mode,
      sat_source_mode,
      bw_source_mode,
      17);
}

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
    int64_t bw_source_mode) {
  launch_phase(
      float_tensors,
      long_tensors,
      bool_tensors,
      int_tensors,
      int_params,
      float_params,
      slot,
      active_idx,
      rollout_tail,
      accel_source_mode,
      sat_source_mode,
      bw_source_mode,
      3);
}

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
    int64_t bw_source_mode) {
  launch_phase(
      float_tensors,
      long_tensors,
      bool_tensors,
      int_tensors,
      int_params,
      float_params,
      slot,
      active_idx,
      rollout_tail,
      accel_source_mode,
      sat_source_mode,
      bw_source_mode,
      4);
}

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
    int64_t bw_source_mode) {
  launch_phase(
      float_tensors,
      long_tensors,
      bool_tensors,
      int_tensors,
      int_params,
      float_params,
      slot,
      active_idx,
      rollout_tail,
      accel_source_mode,
      sat_source_mode,
      bw_source_mode,
      5);
}

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
    at::Tensor finish_profile_out) {
  if (!finish_profile_out.defined() || !finish_profile_out.is_cuda()) {
    throw std::runtime_error("finish internal profile tensor must be a CUDA tensor.");
  }
  if (finish_profile_out.scalar_type() != at::kFloat) {
    throw std::runtime_error("finish internal profile tensor must be float32.");
  }
  if (!finish_profile_out.is_contiguous()) {
    throw std::runtime_error("finish internal profile tensor must be contiguous.");
  }
  if (finish_profile_out.dim() != 2) {
    throw std::runtime_error("finish internal profile tensor must have shape [num_envs, segments].");
  }
  if (static_cast<int64_t>(int_params.size()) <= kParamNumEnvs) {
    throw std::runtime_error("finish internal profile requires native int params.");
  }
  const int64_t num_envs = int_params[kParamNumEnvs];
  if (finish_profile_out.size(0) < num_envs || finish_profile_out.size(1) < kFinishProfileSegments) {
    throw std::runtime_error("finish internal profile tensor is too small.");
  }
  if (!int_tensors.empty() && finish_profile_out.device() != int_tensors.front().device()) {
    throw std::runtime_error("finish internal profile tensor device mismatch.");
  }
  launch_phase(
      float_tensors,
      long_tensors,
      bool_tensors,
      int_tensors,
      int_params,
      float_params,
      slot,
      active_idx,
      rollout_tail,
      accel_source_mode,
      sat_source_mode,
      bw_source_mode,
      4,
      finish_profile_out.data_ptr<float>(),
      static_cast<int>(finish_profile_out.size(1)));
}

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
    int64_t stage_id) {
  check_cuda_group(source_float_tensors, at::ScalarType::Float, "source_float_tensors");
  check_cuda_group(source_long_tensors, at::ScalarType::Long, "source_long_tensors");
  check_cuda_group(source_bool_tensors, at::ScalarType::Bool, "source_bool_tensors");
  check_cuda_group(source_int_tensors, at::ScalarType::Int, "source_int_tensors");
  check_cuda_group(target_float_tensors, at::ScalarType::Float, "target_float_tensors");
  check_cuda_group(target_long_tensors, at::ScalarType::Long, "target_long_tensors");
  check_cuda_group(target_bool_tensors, at::ScalarType::Bool, "target_bool_tensors");
  check_cuda_group(target_int_tensors, at::ScalarType::Int, "target_int_tensors");
  if (!history_rows.defined() || !history_rows.is_cuda() || !history_rows.is_contiguous() ||
      history_rows.scalar_type() != at::ScalarType::Long) {
    throw std::runtime_error("prepare_branch_replay_from_history requires a contiguous CUDA int64 history_rows tensor.");
  }
  if (stage_id < 0 || stage_id > 2) {
    throw std::runtime_error("prepare_branch_replay_from_history stage_id must be 0, 1, or 2.");
  }
  const int64_t branch_count = history_rows.numel();
  if (branch_count <= 0) {
    return;
  }
  const c10::cuda::CUDAGuard device_guard(history_rows.device());
  PackedAbi src = pack_abi(
      source_float_tensors,
      source_long_tensors,
      source_bool_tensors,
      source_int_tensors,
      source_int_params,
      source_float_params);
  PackedAbi dst = pack_abi(
      target_float_tensors,
      target_long_tensors,
      target_bool_tensors,
      target_int_tensors,
      target_int_params,
      target_float_params);
  C10_CUDA_CHECK(cudaMemcpyToSymbol(cBranchSourceAbi, &src, sizeof(PackedAbi), 0, cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpyToSymbol(cBranchTargetAbi, &dst, sizeof(PackedAbi), 0, cudaMemcpyHostToDevice));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned int>(branch_count));
  dim3 block(128);
  prepare_branch_replay_from_history_kernel<<<grid, block, 0, stream>>>(
      history_rows.data_ptr<int64_t>(),
      branch_count,
      stage_id);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
