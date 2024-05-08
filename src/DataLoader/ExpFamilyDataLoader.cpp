//
// Created by tony on 10/05/23.
//

#include <DataLoader/ExpFamilyDataLoader.h>

torch::Tensor CANDY::ExpFamilyDataLoader::generateExp() {
  return torch::exponential(torch::empty({vecVolume, vecDim}));
}

bool CANDY::ExpFamilyDataLoader::hijackConfig(INTELLI::ConfigMapPtr cfg) {
  distributionOverwrite = cfg->tryString("distributionOverwrite", "exp", true);
  return true;
}
torch::Tensor CANDY::ExpFamilyDataLoader::generateBeta() {
  auto tensor1 = torch::randn({vecVolume, vecDim}).abs_();
  auto tensor2 = torch::randn({vecVolume, vecDim}).abs_();
  tensor1 = tensor1.pow(1. / parameterBetaA);
  tensor2 = tensor2.pow(1. / parameterBetaB);
  return tensor1 / (tensor1 + tensor2);
}

torch::Tensor CANDY::ExpFamilyDataLoader::generateGaussian() {
  return torch::randn({vecVolume, vecDim});
}
torch::Tensor CANDY::ExpFamilyDataLoader::generatePoisson() {
  return torch::poisson(torch::empty({vecVolume, vecDim}));
}
torch::Tensor CANDY::ExpFamilyDataLoader::generateData() {
  torch::Tensor ru;
  if (distributionOverwrite == "poisson") {
    ru = generatePoisson();
  } else if (distributionOverwrite == "gaussian") {
    ru = generateGaussian();
  } else if (distributionOverwrite == "beta") {
    ru = generateBeta();
  } else {
    ru = generateExp();
  }
  if (normalizeTensor) {
    ru = INTELLI::IntelliTensorOP::l2Normalize(ru);
  }
  return ru;
}
bool CANDY::ExpFamilyDataLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
  vecDim = cfg->tryI64("vecDim", 768, true);
  vecVolume = cfg->tryI64("vecVolume", 1000, true);
  querySize = cfg->tryI64("querySize", 500, true);
  seed = cfg->tryI64("seed", 7758258, true);
  driftPosition = cfg->tryI64("driftPosition", 0, true);
  driftOffset = cfg->tryDouble("driftOffset", 0.5, true);
  parameterBetaA = cfg->tryDouble("parameterBetaA", 2.0, true);
  parameterBetaB = cfg->tryDouble("parameterBetaB", 2.0, true);
  queryNoiseFraction = cfg->tryDouble("queryNoiseFraction", 0, true);
  manualChangeDistribution = cfg->tryI64("manualChangeDistribution", 0, true);
  normalizeTensor = cfg->tryI64("normalizeTensor", 0, true);
  if (manualChangeDistribution) {
    distributionOverwrite = cfg->tryString("distributionOverwrite", "exp", true);
  }
  if (queryNoiseFraction < 0) {
    queryNoiseFraction = 0;
  }
  if (queryNoiseFraction > 1) {
    queryNoiseFraction = 1;
  }
  if (querySize > vecVolume) {
    INTELLI_ERROR("invalid size of query");
    return false;
  }
  INTELLI_INFO("Chose " + distributionOverwrite + " in the exponential family");
  INTELLI_INFO(
      "Generating [" + to_string(vecVolume) + "x" + to_string(vecDim) + "]" + ", query size " + to_string(querySize));
  torch::manual_seed(seed);
  A = generateData();
  if (driftPosition > 0 && driftPosition < vecVolume) {
    INTELLI_INFO(
        "I will introduce concept drift from" + std::to_string(driftPosition) + ",drift offset="
            + std::to_string(driftOffset));
    A.slice(0, driftPosition, vecVolume) = A.slice(0, driftPosition, vecVolume) + driftOffset;
  }
  // Generate ExpFamily indices
  auto indices = torch::randperm(A.size(0), torch::kLong).slice(/*dim=*/0, /*start=*/0, /*end=*/querySize);
  // Use the ExpFamily indices to select rows from tensor A
  B = A.index_select(/*dim=*/0, indices);
  B = (1 - queryNoiseFraction) * B + queryNoiseFraction * torch::rand({querySize, vecDim});
  return true;
}

torch::Tensor CANDY::ExpFamilyDataLoader::getData() {
  return A;
}

torch::Tensor CANDY::ExpFamilyDataLoader::getQuery() {
  return B;
}
