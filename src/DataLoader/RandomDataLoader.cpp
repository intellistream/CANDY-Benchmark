//
// Created by tony on 10/05/23.
//

#include <DataLoader/RandomDataLoader.h>

//do nothing in Random class

bool CANDY::RandomDataLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
  vecDim = cfg->tryI64("vecDim", 768, true);
  vecVolume = cfg->tryI64("vecVolume", 1000, true);
  querySize = cfg->tryI64("querySize", 500, true);
  seed = cfg->tryI64("seed", 7758258, true);
  driftPosition = cfg->tryI64("driftPosition", 0, true);
  driftOffset = cfg->tryDouble("driftOffset", 0.5, true);
  queryNoiseFraction = cfg->tryDouble("queryNoiseFraction", 0, true);
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
  INTELLI_INFO(
      "Generating [" + to_string(vecVolume) + "x" + to_string(vecDim) + "]" + ", query size " + to_string(querySize));
  torch::manual_seed(seed);
  A = torch::rand({vecVolume, vecDim});
  if (driftPosition > 0 && driftPosition < vecVolume) {
    INTELLI_INFO(
        "I will introduce concept drift from" + std::to_string(driftPosition) + ",drift offset="
            + std::to_string(driftOffset));
    A.slice(0, driftPosition, vecVolume) = A.slice(0, driftPosition, vecVolume) * (1.0 - driftOffset);
  }
  // Generate random indices
  auto indices = torch::randperm(A.size(0), torch::kLong).slice(/*dim=*/0, /*start=*/0, /*end=*/querySize);
  // Use the random indices to select rows from tensor A
  B = A.index_select(/*dim=*/0, indices);
  B = (1 - queryNoiseFraction) * B + queryNoiseFraction * torch::rand({querySize, vecDim});
  return true;
}

torch::Tensor CANDY::RandomDataLoader::getData() {
  return A;
}

torch::Tensor CANDY::RandomDataLoader::getQuery() {
  return B;
}
