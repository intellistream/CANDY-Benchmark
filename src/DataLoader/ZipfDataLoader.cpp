//
// Created by tony on 10/05/23.
//

#include <DataLoader/ZipfDataLoader.h>

//do nothing in Zipf class
torch::Tensor CANDY::ZipfDataLoader::generateZipfDistribution(int64_t n, int64_t m, double alpha) {
  /*torch::Tensor indices = torch::arange(1, n * m + 1, torch::kFloat32);
  torch::Tensor probabilities = 1.0 / torch::pow(indices, alpha);
  torch::Tensor normalizedProbabilities = probabilities / torch::sum(probabilities);

  // Generate Zipf-distributed samples
  torch::Tensor zipfSamples = torch::multinomial(normalizedProbabilities, n * m, true);
  torch::Tensor zipfMatrix = zipfSamples.view({n, m}).clone();

  // Normalize the values to the range [0, 1]
  auto ru = zipfMatrix / (zipfMatrix.max());*/
  torch::Tensor values = torch::rand({n, m}, torch::kFloat32);
  if (alpha == 0) {
    return values;
  }
  values = 1.0 / torch::pow(values, 1.0 / alpha);
  auto ru = values / (values.max());
  // Reshape the 1D tensor to an nxm tensor
  return ru;
}
bool CANDY::ZipfDataLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
  vecDim = cfg->tryI64("vecDim", 768, true);
  vecVolume = cfg->tryI64("vecVolume", 1000, true);
  querySize = cfg->tryI64("querySize", 500, true);
  seed = cfg->tryI64("seed", 7758258, true);
  driftPosition = cfg->tryI64("driftPosition", 0, true);
  driftOffset = cfg->tryDouble("driftOffset", 0.5, true);
  queryNoiseFraction = cfg->tryDouble("queryNoiseFraction", 0, true);
  zipfAlpha = cfg->tryDouble("zipfAlpha", 0.0, false);
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
  A = generateZipfDistribution((int64_t) vecVolume, (int64_t) vecDim, zipfAlpha);
  if (driftPosition > 0 && driftPosition < vecVolume) {
    INTELLI_INFO(
        "I will introduce concept drift from" + std::to_string(driftPosition) + ",drift offset="
            + std::to_string(driftOffset));
    A.slice(0, driftPosition, vecVolume) =
        generateZipfDistribution((int64_t) (vecVolume - driftPosition), (int64_t) vecDim, zipfAlpha + driftOffset);
  }
  // Generate Zipf indices
  auto indices = torch::randperm(A.size(0), torch::kLong).slice(/*dim=*/0, /*start=*/0, /*end=*/querySize);
  // Use the Zipf indices to select rows from tensor A
  B = A.index_select(/*dim=*/0, indices);
  B = (1 - queryNoiseFraction) * B + queryNoiseFraction * torch::rand({querySize, vecDim});
  return true;
}

torch::Tensor CANDY::ZipfDataLoader::getData() {
  return A;
}

torch::Tensor CANDY::ZipfDataLoader::getQuery() {
  return B;
}
